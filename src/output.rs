//! Host-supplied audio output contracts.
//!
//! [`AudioContext::builder`](crate::context::AudioContext::builder) accepts these contracts for an
//! exact hosted context while render-thread storage and lifecycle proof types remain private. The
//! [`SystemAudioOutput`] supplies the additive system-device adapter; legacy constructors continue
//! using their established backend path.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::error::Error;
use std::fmt;
use std::future::{ready, Future};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};

use crate::context::{AudioContextLatencyCategory, AudioContextRenderSizeCategory};
use crate::render::RenderThread;
use crate::{is_valid_sample_rate, MAX_CHANNELS};

mod system;

pub use system::SystemAudioOutput;

/// Hard upper bound for frames supplied to one injected output callback invocation.
pub const MAX_AUDIO_OUTPUT_CALLBACK_FRAMES: usize = 8192;

/// Opaque identity of the independent audio context requesting an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AudioOutputContextId(u64);

impl AudioOutputContextId {
    /// Constructs a nonzero context identity.
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Returns the numeric identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Neutral output facts supplied to an [`AudioOutputFactory`].
///
/// A request is constructed only by this crate. Its fields describe a request; they are not proof
/// of permission or resource admission. The hosted builder validates these facts before invoking
/// a factory.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AudioOutputRequest {
    context_id: AudioOutputContextId,
    sink_id: String,
    requested_sample_rate: Option<f32>,
    number_of_channels: usize,
    latency_hint: AudioContextLatencyCategory,
    render_size_hint: AudioContextRenderSizeCategory,
    diagnostic_label: Option<String>,
}

impl AudioOutputRequest {
    pub(crate) fn validate_parts(
        requested_sample_rate: Option<f32>,
        number_of_channels: usize,
        latency_hint: AudioContextLatencyCategory,
    ) -> Result<(), AudioOutputError> {
        if let Some(sample_rate) = requested_sample_rate {
            if !is_valid_sample_rate(sample_rate) {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::InvalidArgument,
                    format!("invalid requested sample rate: {sample_rate}"),
                ));
            }
        }
        if !(1..=MAX_CHANNELS).contains(&number_of_channels) {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::InvalidArgument,
                format!("invalid requested output channel count: {number_of_channels}"),
            ));
        }

        if let AudioContextLatencyCategory::Custom(latency) = latency_hint {
            if !latency.is_finite() || latency <= 0. {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::InvalidArgument,
                    format!("invalid custom output latency: {latency}"),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn new(
        context_id: AudioOutputContextId,
        sink_id: impl Into<String>,
        requested_sample_rate: Option<f32>,
        number_of_channels: usize,
        latency_hint: AudioContextLatencyCategory,
        render_size_hint: AudioContextRenderSizeCategory,
        diagnostic_label: Option<String>,
    ) -> Result<Self, AudioOutputError> {
        Self::validate_parts(requested_sample_rate, number_of_channels, latency_hint)?;

        Ok(Self {
            context_id,
            sink_id: sink_id.into(),
            requested_sample_rate,
            number_of_channels,
            latency_hint,
            render_size_hint,
            diagnostic_label,
        })
    }

    pub(crate) fn validate_config(
        &self,
        config: &AudioOutputConfig,
    ) -> Result<(), AudioOutputError> {
        if let Some(requested) = self.requested_sample_rate {
            if config.format().sample_rate() != requested {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::NotSupported,
                    format!(
                        "prepared logical sample rate {} does not match requested rate {requested}",
                        config.format().sample_rate()
                    ),
                ));
            }
        }
        if config.format().number_of_channels() != self.number_of_channels {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::NotSupported,
                format!(
                    "prepared logical channel count {} does not match requested count {}",
                    config.format().number_of_channels(),
                    self.number_of_channels
                ),
            ));
        }
        if !self.sink_id.is_empty() && config.accepted_sink_id() != self.sink_id {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::NotSupported,
                format!(
                    "prepared sink {:?} does not match requested sink {:?}",
                    config.accepted_sink_id(),
                    self.sink_id
                ),
            ));
        }

        Ok(())
    }

    /// Identity of the requesting context.
    #[must_use]
    pub const fn context_id(&self) -> AudioOutputContextId {
        self.context_id
    }

    /// Requested logical sink identifier. An empty string denotes the default output.
    #[must_use]
    pub fn sink_id(&self) -> &str {
        &self.sink_id
    }

    /// Explicit requested logical sample rate, if any.
    ///
    /// [`AudioOutputFactory::prepare`] must fail when it cannot return a configuration whose
    /// logical sample rate exactly equals `Some(rate)`. A private physical rate may differ only
    /// when the endpoint performs bounded resampling.
    #[must_use]
    pub const fn requested_sample_rate(&self) -> Option<f32> {
        self.requested_sample_rate
    }

    /// Requested logical output channel count.
    #[must_use]
    pub const fn number_of_channels(&self) -> usize {
        self.number_of_channels
    }

    /// Requested latency tradeoff.
    #[must_use]
    pub const fn latency_hint(&self) -> AudioContextLatencyCategory {
        self.latency_hint
    }

    /// Requested render-quantum-size category.
    #[must_use]
    pub const fn render_size_hint(&self) -> AudioContextRenderSizeCategory {
        self.render_size_hint
    }

    /// Optional neutral diagnostic label.
    #[must_use]
    pub fn diagnostic_label(&self) -> Option<&str> {
        self.diagnostic_label.as_deref()
    }
}

/// Logical format in which one context renders into its output endpoint.
///
/// `sample_rate` is the Web Audio context's logical rate, not necessarily a physical device or
/// aggregate-mixer rate. Samples are interleaved `f32` values.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct AudioRenderFormat {
    sample_rate: f32,
    number_of_channels: usize,
    max_frames_per_callback: usize,
}

impl AudioRenderFormat {
    /// Validates and constructs a logical render format.
    pub fn new(
        sample_rate: f32,
        number_of_channels: usize,
        max_frames_per_callback: usize,
    ) -> Result<Self, AudioOutputError> {
        if !is_valid_sample_rate(sample_rate) {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::InvalidArgument,
                format!("invalid logical sample rate: {sample_rate}"),
            ));
        }
        if !(1..=MAX_CHANNELS).contains(&number_of_channels) {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::InvalidArgument,
                format!("invalid output channel count: {number_of_channels}"),
            ));
        }
        if !(1..=MAX_AUDIO_OUTPUT_CALLBACK_FRAMES).contains(&max_frames_per_callback) {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::InvalidArgument,
                format!("invalid maximum callback frame count: {max_frames_per_callback}"),
            ));
        }

        Ok(Self {
            sample_rate,
            number_of_channels,
            max_frames_per_callback,
        })
    }

    /// Logical Web Audio sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(self) -> f32 {
        self.sample_rate
    }

    /// Number of interleaved output channels.
    #[must_use]
    pub const fn number_of_channels(self) -> usize {
        self.number_of_channels
    }

    /// Maximum number of frames accepted by one callback invocation.
    #[must_use]
    pub const fn max_frames_per_callback(self) -> usize {
        self.max_frames_per_callback
    }
}

/// Negotiated facts returned by a prepared output.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AudioOutputConfig {
    format: AudioRenderFormat,
    accepted_sink_id: String,
    output_latency: f64,
}

impl AudioOutputConfig {
    /// Validates and constructs a negotiated output configuration.
    pub fn new(
        format: AudioRenderFormat,
        accepted_sink_id: impl Into<String>,
        output_latency: f64,
    ) -> Result<Self, AudioOutputError> {
        if !output_latency.is_finite() || output_latency < 0. {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::InvalidArgument,
                format!("invalid output latency: {output_latency}"),
            ));
        }

        Ok(Self {
            format,
            accepted_sink_id: accepted_sink_id.into(),
            output_latency,
        })
    }

    /// Negotiated logical render format.
    #[must_use]
    pub const fn format(&self) -> AudioRenderFormat {
        self.format
    }

    /// Accepted logical sink identifier.
    ///
    /// This must exactly equal a nonempty requested sink identifier. A default request (`""`) may
    /// resolve to a concrete sink identifier.
    #[must_use]
    pub fn accepted_sink_id(&self) -> &str {
        &self.accepted_sink_id
    }

    /// Initial output latency estimate in seconds.
    #[must_use]
    pub const fn output_latency(&self) -> f64 {
        self.output_latency
    }
}

/// Stable category of an output factory, endpoint, or lifecycle failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioOutputErrorKind {
    /// The request or format is invalid.
    InvalidArgument,
    /// The factory or selected endpoint does not support the request.
    NotSupported,
    /// The requested physical or logical device is unavailable.
    DeviceUnavailable,
    /// The selected backend returned an implementation-specific failure.
    BackendSpecific,
    /// The render callback died or was retired unexpectedly.
    CallbackDied,
    /// Endpoint-local shutdown failed.
    Shutdown,
}

/// Error returned by the additive output contracts.
///
/// This string-bearing type is confined to construction, control, and lifecycle paths. Render
/// callbacks report only fixed-size [`AudioOutputEvent`] values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioOutputError {
    kind: AudioOutputErrorKind,
    message: String,
}

impl AudioOutputError {
    /// Constructs an output error with a stable category and diagnostic message.
    #[must_use]
    pub fn new(kind: AudioOutputErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Stable error category.
    #[must_use]
    pub const fn kind(&self) -> AudioOutputErrorKind {
        self.kind
    }

    /// Human-readable diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for AudioOutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for AudioOutputError {}

/// Fixed-size reason recorded when a logical endpoint dies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum AudioOutputDeathReason {
    /// The endpoint dropped a still-open callback without controller-initiated shutdown.
    CallbackRetiredUnexpectedly = 1,
    /// The callback panicked and was converted to silence.
    CallbackPanicked = 2,
    /// The physical or logical device became unavailable.
    DeviceUnavailable = 3,
    /// The backend reported an implementation-specific terminal failure.
    BackendFailure = 4,
    /// The factory or its shared endpoint was shut down.
    FactoryShutdown = 5,
    /// The endpoint supplied a buffer that violated the negotiated render format.
    CallbackProtocolViolation = 6,
}

impl AudioOutputDeathReason {
    fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            1 => Self::CallbackRetiredUnexpectedly,
            2 => Self::CallbackPanicked,
            3 => Self::DeviceUnavailable,
            4 => Self::BackendFailure,
            5 => Self::FactoryShutdown,
            6 => Self::CallbackProtocolViolation,
            _ => return None,
        })
    }
}

/// Fixed-size, copy-only diagnostic emitted by a running output endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioOutputEvent {
    /// The endpoint could not supply this many logical frames by its deadline.
    Underrun {
        /// Number of affected logical frames.
        frames: u64,
    },
    /// The endpoint discarded this many logical frames because its bounded transport was full.
    Overrun {
        /// Number of affected logical frames.
        frames: u64,
    },
    /// Best-effort wake for the authoritative endpoint-death latch.
    EndpointDied(AudioOutputDeathReason),
}

#[derive(Default)]
struct AudioOutputEventState {
    death_reason: AtomicU8,
}

/// Cloneable, nonblocking diagnostic sink handed to a prepared endpoint at startup.
///
/// Endpoint death is recorded first-writer-wins in an authoritative atomic latch. The event is
/// only a best-effort wake, so a full or disconnected channel cannot lose terminal knowledge or
/// deallocate event-owned heap storage. The crate retains the watcher side until callbacks retire.
#[derive(Clone)]
pub struct AudioOutputEventSink {
    sender: Sender<AudioOutputEvent>,
    state: Arc<AudioOutputEventState>,
}

impl fmt::Debug for AudioOutputEventSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioOutputEventSink")
            .field("death_reason", &self.death_reason())
            .finish()
    }
}

impl AudioOutputEventSink {
    pub(crate) fn matches_watcher(&self, watcher: &AudioOutputEventWatcher) -> bool {
        Arc::ptr_eq(&self.state, &watcher.state)
    }

    /// Attempts to deliver a diagnostic without waiting.
    ///
    /// [`AudioOutputEvent::EndpointDied`] is routed through the authoritative death latch.
    #[must_use]
    pub fn try_send(&self, event: AudioOutputEvent) -> bool {
        match event {
            AudioOutputEvent::EndpointDied(reason) => self.report_endpoint_death(reason),
            event => self.sender.try_send(event).is_ok(),
        }
    }

    /// Records endpoint death exactly once and emits a best-effort bounded wake.
    ///
    /// Returns `true` only for the first reporter.
    #[must_use]
    pub fn report_endpoint_death(&self, reason: AudioOutputDeathReason) -> bool {
        if self
            .state
            .death_reason
            .compare_exchange(0, reason as u8, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }

        let _ = self.sender.try_send(AudioOutputEvent::EndpointDied(reason));
        true
    }

    /// Returns the authoritative first endpoint-death reason, if any.
    #[must_use]
    pub fn death_reason(&self) -> Option<AudioOutputDeathReason> {
        AudioOutputDeathReason::from_u8(self.state.death_reason.load(Ordering::Acquire))
    }

    #[allow(dead_code)] // hosted lifecycle and lower-level test seam
    pub(crate) fn bounded(capacity: usize) -> (Self, AudioOutputEventWatcher) {
        let (sender, receiver) = crossbeam_channel::bounded(capacity);
        let state = Arc::new(AudioOutputEventState::default());
        (
            Self {
                sender,
                state: Arc::clone(&state),
            },
            AudioOutputEventWatcher { receiver, state },
        )
    }
}

/// Crate-owned watcher for best-effort wakes and authoritative endpoint death.
#[allow(dead_code)]
pub(crate) struct AudioOutputEventWatcher {
    receiver: Receiver<AudioOutputEvent>,
    state: Arc<AudioOutputEventState>,
}

impl AudioOutputEventWatcher {
    #[allow(dead_code)]
    pub(crate) fn try_recv(&self) -> Option<AudioOutputEvent> {
        self.receiver.try_recv().ok()
    }

    /// Authoritative first endpoint-death reason, independent of lossy diagnostic wakes.
    pub(crate) fn death_reason(&self) -> Option<AudioOutputDeathReason> {
        AudioOutputDeathReason::from_u8(self.state.death_reason.load(Ordering::Acquire))
    }

    /// Diagnostic receiver used only by the private lifecycle worker to wake on endpoint death.
    ///
    /// Consumers must always re-read [`Self::death_reason`]; channel records are best effort and
    /// are not themselves terminal authority.
    pub(crate) fn receiver(&self) -> &Receiver<AudioOutputEvent> {
        &self.receiver
    }
}

/// Result of one render callback invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioRenderStatus {
    /// The endpoint may request another callback.
    Continue,
    /// Rendering has stopped and the endpoint must not invoke this callback again.
    Stop,
}

const RENDER_GATE_OPEN: u8 = 0;
const RENDER_GATE_ACTIVE: u8 = 1;
const RENDER_GATE_CLOSED: u8 = 2;
const RENDER_GATE_CLOSED_ACTIVE: u8 = RENDER_GATE_CLOSED | RENDER_GATE_ACTIVE;

#[allow(dead_code)] // used by hosted and lower-level lifecycle paths
trait AudioRenderDriver: Send + 'static {
    fn render_interleaved_f32(&mut self, output: &mut [f32]);

    fn reclaim_off_thread(self: Box<Self>) -> Result<(), AudioOutputError>;
}

#[allow(dead_code)] // constructed by the private B3b injected lifecycle path
struct RenderThreadDriver {
    renderer: Option<RenderThread>,
    garbage_collector_join: JoinHandle<()>,
}

impl AudioRenderDriver for RenderThreadDriver {
    fn render_interleaved_f32(&mut self, output: &mut [f32]) {
        self.renderer
            .as_mut()
            .expect("render thread driver retains its renderer until reclamation")
            .render(output);
    }

    fn reclaim_off_thread(self: Box<Self>) -> Result<(), AudioOutputError> {
        let Self {
            mut renderer,
            garbage_collector_join,
        } = *self;
        let mut failed = false;
        if let Some(mut renderer) = renderer.take() {
            failed |=
                panic::catch_unwind(AssertUnwindSafe(|| renderer.prepare_for_reclaim())).is_err();
            failed |= panic::catch_unwind(AssertUnwindSafe(|| drop(renderer))).is_err();
        }
        failed |= garbage_collector_join.join().is_err();

        if failed {
            Err(AudioOutputError::new(
                AudioOutputErrorKind::Shutdown,
                "audio renderer or garbage collector panicked during reclamation",
            ))
        } else {
            Ok(())
        }
    }
}

/// The render driver and a panic payload are manually retired by [`AudioRenderOwner`].
///
/// # Safety invariant
///
/// The callback is the only code allowed to enter `renderer`. The owner may take either
/// `UnsafeCell` only after endpoint shutdown, `Arc::try_unwrap` proves that the callback object was
/// destroyed, and the combined gate is closed with no in-flight callback.
struct RenderSlot {
    gate: AtomicU8,
    renderer: UnsafeCell<ManuallyDrop<Option<Box<dyn AudioRenderDriver>>>>,
    panic_payload: UnsafeCell<ManuallyDrop<Option<Box<dyn Any + Send>>>>,
    events: AudioOutputEventSink,
}

// SAFETY: `renderer` has exactly one callback-side accessor, protected by the combined gate.
// Owner-side access is allowed only after endpoint shutdown, Arc uniqueness, and CLOSED/zero.
// `panic_payload` is written by that same callback before its in-flight Release and is read only
// by the uniquely owning reclaimer after an Acquire load observes CLOSED/zero.
unsafe impl Sync for RenderSlot {}

impl RenderSlot {
    fn close(&self) -> u8 {
        self.gate.fetch_or(RENDER_GATE_CLOSED, Ordering::AcqRel)
    }

    fn try_enter(&self) -> Option<RenderInFlight<'_>> {
        self.gate
            .compare_exchange(
                RENDER_GATE_OPEN,
                RENDER_GATE_ACTIVE,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .ok()
            .map(|_| RenderInFlight { gate: &self.gate })
    }

    fn store_panic_payload(&self, payload: Box<dyn Any + Send>) {
        // SAFETY: only the unique, non-Sync callback can write this slot. It closes the gate on
        // the first panic, so the payload is written at most once. The owner reads it only after
        // callback retirement and Arc uniqueness.
        unsafe {
            *self.panic_payload.get() = ManuallyDrop::new(Some(payload));
        }
    }
}

struct RenderInFlight<'a> {
    gate: &'a AtomicU8,
}

impl Drop for RenderInFlight<'_> {
    fn drop(&mut self) {
        let previous = self.gate.fetch_and(!RENDER_GATE_ACTIVE, Ordering::Release);
        debug_assert!(matches!(
            previous,
            RENDER_GATE_ACTIVE | RENDER_GATE_CLOSED_ACTIVE
        ));
    }
}

/// Crate-owned lifetime authority for an [`AudioRenderCallback`].
///
/// Dropping this value without confirmed reclamation intentionally leaks its slot. That
/// fail-closed behavior prevents a retained or concurrently executing callback from observing
/// freed render state.
pub(crate) struct AudioRenderOwner {
    slot: Option<Arc<RenderSlot>>,
}

impl fmt::Debug for AudioRenderOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slot = self.slot.as_ref();
        f.debug_struct("AudioRenderOwner")
            .field("gate", &slot.map(|slot| slot.gate.load(Ordering::Acquire)))
            .finish_non_exhaustive()
    }
}

#[allow(dead_code)] // used by hosted and lower-level lifecycle paths
impl AudioRenderOwner {
    pub(crate) fn begin_shutdown(&self) {
        if let Some(slot) = &self.slot {
            slot.close();
        }
    }

    /// Returns whether the closed callback gate has no invocation in flight.
    ///
    /// This does not prove that the endpoint destroyed its callback or authorize renderer
    /// reclamation. It is the narrower authority needed by fail-closed lifecycle paths before
    /// they stop the event consumer: a retained callback can subsequently produce only silence
    /// and its eventual Drop observes an already-closed gate without reporting another event.
    pub(crate) fn callback_producer_quiescent(&self) -> bool {
        self.slot
            .as_ref()
            .is_none_or(|slot| slot.gate.load(Ordering::Acquire) == RENDER_GATE_CLOSED)
    }

    /// Reclaims render resources after endpoint-local shutdown has confirmed.
    ///
    /// `EndpointShutdownConfirmed` promises that the endpoint destroyed the callback object. The
    /// subsequent `Arc::try_unwrap` is the unforgeable local proof of that promise. On any
    /// ambiguous callback state, ownership is returned for explicit quarantine. Dropping that
    /// returned owner is also safe because its Drop implementation leaks the render slot.
    pub(crate) fn try_reclaim_after_shutdown(
        mut self,
        _confirmed: EndpointShutdownConfirmed,
    ) -> Result<Result<(), AudioOutputError>, Self> {
        let slot = self.slot.take().expect("render owner is single-use");
        let slot = match Arc::try_unwrap(slot) {
            Ok(slot) => slot,
            Err(slot) => {
                self.slot = Some(slot);
                return Err(self);
            }
        };
        if slot.gate.load(Ordering::Acquire) != RENDER_GATE_CLOSED {
            self.slot = Some(Arc::new(slot));
            return Err(self);
        }

        // SAFETY: endpoint confirmation, Arc uniqueness, and CLOSED/zero jointly prove that no
        // callback exists, can enter, or retains access to these manually managed values.
        let renderer = unsafe { ManuallyDrop::take(&mut *slot.renderer.get()) };
        // SAFETY: the panic payload uses the same callback-to-owner handoff as `renderer`.
        let panic_payload = unsafe { ManuallyDrop::take(&mut *slot.panic_payload.get()) };
        drop(slot);

        let renderer_result = panic::catch_unwind(AssertUnwindSafe(|| {
            renderer.map_or(Ok(()), |renderer| renderer.reclaim_off_thread())
        }))
        .unwrap_or_else(|_| {
            Err(AudioOutputError::new(
                AudioOutputErrorKind::Shutdown,
                "audio renderer panicked during off-thread reclamation",
            ))
        });
        let panic_drop_result = panic::catch_unwind(AssertUnwindSafe(|| drop(panic_payload)))
            .map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "audio callback panic payload panicked while being reclaimed",
                )
            });

        Ok(renderer_result.and(panic_drop_result))
    }
}

impl Drop for AudioRenderOwner {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            // A missing endpoint acknowledgement, retained callback, lifecycle cancellation, or
            // controller panic is ambiguous. Leaking is the only generally safe fallback.
            std::mem::forget(slot);
        }
    }
}

/// Proof that endpoint-local shutdown completed successfully.
///
/// Only the private B3b lifecycle controller constructs this marker from an `Ok` endpoint receipt.
/// Such a receipt must guarantee no future or in-flight calls and destruction of the callback
/// object that was passed to the endpoint.
pub(crate) struct EndpointShutdownConfirmed(());

impl EndpointShutdownConfirmed {
    #[allow(dead_code)] // constructed by hosted and lower-level lifecycle paths
    pub(crate) const fn new() -> Self {
        Self(())
    }
}

fn audio_render_pair(
    format: AudioRenderFormat,
    renderer: Box<dyn AudioRenderDriver>,
    events: AudioOutputEventSink,
) -> (AudioRenderOwner, AudioRenderCallback) {
    let slot = Arc::new(RenderSlot {
        gate: AtomicU8::new(RENDER_GATE_OPEN),
        renderer: UnsafeCell::new(ManuallyDrop::new(Some(renderer))),
        panic_payload: UnsafeCell::new(ManuallyDrop::new(None)),
        events,
    });
    let callback = AudioRenderCallback {
        format,
        slot: Arc::clone(&slot),
        not_sync: PhantomData,
    };
    (AudioRenderOwner { slot: Some(slot) }, callback)
}

#[cfg(test)]
pub(crate) fn audio_render_test_pair<R, C>(
    format: AudioRenderFormat,
    events: AudioOutputEventSink,
    render: R,
    reclaim: C,
) -> (AudioRenderOwner, AudioRenderCallback)
where
    R: FnMut(&mut [f32]) + Send + 'static,
    C: FnOnce() -> Result<(), AudioOutputError> + Send + 'static,
{
    struct TestDriver<R, C> {
        render: R,
        reclaim: Option<C>,
    }

    impl<R, C> AudioRenderDriver for TestDriver<R, C>
    where
        R: FnMut(&mut [f32]) + Send + 'static,
        C: FnOnce() -> Result<(), AudioOutputError> + Send + 'static,
    {
        fn render_interleaved_f32(&mut self, output: &mut [f32]) {
            (self.render)(output);
        }

        fn reclaim_off_thread(mut self: Box<Self>) -> Result<(), AudioOutputError> {
            self.reclaim.take().expect("test reclaimer is single-use")()
        }
    }

    audio_render_pair(
        format,
        Box::new(TestDriver {
            render,
            reclaim: Some(reclaim),
        }),
        events,
    )
}

pub(crate) struct AudioRenderThreadPairFailure {
    pub(crate) error: AudioOutputError,
    pub(crate) renderer: RenderThread,
    pub(crate) events: AudioOutputEventSink,
}

/// Installs the mandatory joinable GC before publishing either callback-side ownership object.
/// Spawn failure returns the unchanged exact renderer and event sink to its branded caller.
#[allow(clippy::result_large_err)] // exact renderer return must not depend on a second allocation
pub(crate) fn try_audio_render_thread_pair(
    format: AudioRenderFormat,
    mut renderer: RenderThread,
    events: AudioOutputEventSink,
) -> Result<(AudioRenderOwner, AudioRenderCallback), AudioRenderThreadPairFailure> {
    let garbage_collector_join = match renderer.try_spawn_joinable_garbage_collector_thread() {
        Ok(Some(join)) => join,
        Ok(None) => {
            return Err(AudioRenderThreadPairFailure {
                error: AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "injected renderer already owns a garbage collector",
                ),
                renderer,
                events,
            });
        }
        Err(error) => {
            return Err(AudioRenderThreadPairFailure {
                error: AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    format!("failed to spawn injected garbage collector: {error}"),
                ),
                renderer,
                events,
            });
        }
    };
    Ok(audio_render_pair(
        format,
        Box::new(RenderThreadDriver {
            renderer: Some(renderer),
            garbage_collector_join,
        }),
        events,
    ))
}

#[cfg(test)]
pub(crate) fn audio_render_thread_pair(
    format: AudioRenderFormat,
    renderer: RenderThread,
    events: AudioOutputEventSink,
) -> (AudioRenderOwner, AudioRenderCallback) {
    try_audio_render_thread_pair(format, renderer, events)
        .map_err(|failure| failure.error)
        .expect("injected render pair test bootstrap should succeed")
}

/// Opaque, single-owner callback for one independent context.
///
/// It is `Send`, but deliberately neither `Sync` nor `Clone`:
///
/// ```compile_fail
/// use web_audio_api::output::AudioRenderCallback;
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<AudioRenderCallback>();
/// ```
///
/// ```compile_fail
/// use web_audio_api::output::AudioRenderCallback;
/// fn assert_clone<T: Clone>() {}
/// assert_clone::<AudioRenderCallback>();
/// ```
///
/// Dropping the callback only closes its atomic gate and drops its memory lease. The renderer and
/// any captured panic payload are reclaimed exclusively by the crate-owned lifecycle authority
/// after endpoint shutdown is confirmed and `Arc::try_unwrap` proves callback destruction.
pub struct AudioRenderCallback {
    format: AudioRenderFormat,
    slot: Arc<RenderSlot>,
    not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for AudioRenderCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioRenderCallback")
            .field("format", &self.format)
            .field(
                "closed",
                &(self.slot.gate.load(Ordering::Acquire) & RENDER_GATE_CLOSED != 0),
            )
            .finish_non_exhaustive()
    }
}

impl AudioRenderCallback {
    /// Logical format produced by this callback.
    #[must_use]
    pub const fn format(&self) -> AudioRenderFormat {
        self.format
    }

    /// Renders interleaved `f32` samples in the negotiated logical format.
    ///
    /// An empty, non-frame-aligned, or oversized buffer is a terminal protocol violation. Any
    /// terminal call fills the entire supplied buffer with silence. Panics from the renderer are
    /// contained, converted to silence, and retained for off-real-time reclamation.
    #[must_use]
    pub fn render_interleaved_f32(&mut self, output: &mut [f32]) -> AudioRenderStatus {
        let channels = self.format.number_of_channels();
        let valid = !output.is_empty()
            && output.len() % channels == 0
            && (output.len() / channels) <= self.format.max_frames_per_callback();
        if !valid {
            self.slot.close();
            output.fill(0.);
            let _ = self
                .slot
                .events
                .report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
            return AudioRenderStatus::Stop;
        }

        let Some(_in_flight) = self.slot.try_enter() else {
            output.fill(0.);
            return AudioRenderStatus::Stop;
        };

        // SAFETY: `AudioRenderCallback` is non-Clone and non-Sync, and the combined gate grants
        // this invocation exclusive callback-side access. Owner access is forbidden until the
        // callback retires and the gate reaches CLOSED/zero.
        let renderer = unsafe { &mut *self.slot.renderer.get() };
        let Some(renderer) = renderer.as_mut() else {
            self.slot.close();
            output.fill(0.);
            let _ = self
                .slot
                .events
                .report_endpoint_death(AudioOutputDeathReason::BackendFailure);
            return AudioRenderStatus::Stop;
        };
        match panic::catch_unwind(AssertUnwindSafe(|| {
            renderer.render_interleaved_f32(output);
        })) {
            Ok(()) => AudioRenderStatus::Continue,
            Err(payload) => {
                self.slot.store_panic_payload(payload);
                self.slot.close();
                output.fill(0.);
                let _ = self
                    .slot
                    .events
                    .report_endpoint_death(AudioOutputDeathReason::CallbackPanicked);
                AudioRenderStatus::Stop
            }
        }
    }
}

impl Drop for AudioRenderCallback {
    fn drop(&mut self) {
        let was_open = self.slot.close() & RENDER_GATE_CLOSED == 0;
        if was_open {
            // This is an unexpected endpoint-side callback retirement. The owner owns (or has
            // deliberately leaked) the other Arc, so normal field destruction below cannot
            // final-drop callback resources.
            let _ = self
                .slot
                .events
                .report_endpoint_death(AudioOutputDeathReason::CallbackRetiredUnexpectedly);
        }
    }
}

/// Factory for one independently owned logical output per audio context.
///
/// This trait is object-safe and is consumed by
/// [`AudioContext::builder`](crate::context::AudioContext::builder).
pub trait AudioOutputFactory: Send + Sync + 'static {
    /// Validates the request, negotiates its logical configuration, and prepares endpoint-owned
    /// resources without starting a render callback.
    ///
    /// When [`AudioOutputRequest::requested_sample_rate`] is `Some(rate)`, the returned prepared
    /// output's configuration must use exactly `rate`; its channel count must always equal the
    /// request's channel count. A nonempty requested sink identifier must also be accepted exactly;
    /// only the empty default identifier may resolve to a concrete sink. Otherwise this method must
    /// fail.
    fn prepare(
        &self,
        request: &AudioOutputRequest,
    ) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError>;
}

/// Prepared but not yet rendering output endpoint.
pub trait PreparedAudioOutput: Send + 'static {
    /// Negotiated output configuration.
    fn config(&self) -> &AudioOutputConfig;

    /// Installs the context render callback and starts endpoint ownership.
    ///
    /// On success the returned endpoint is logically running. A context whose graph begins
    /// suspended must reconcile that native endpoint state through its lifecycle owner.
    fn start(
        self: Box<Self>,
        callback: AudioRenderCallback,
        events: AudioOutputEventSink,
    ) -> Result<Box<dyn RunningAudioOutput>, AudioOutputStartFailure>;

    /// Synchronously commits abort, then returns endpoint-local retirement work.
    ///
    /// Once this method returns, this endpoint cannot start a callback. The returned future only
    /// awaits cleanup already committed by this call.
    fn abort(self: Box<Self>) -> AudioOutputEndpointShutdown;
}

/// Single-use prepared endpoint whose negotiated configuration was observed, cloned, and
/// validated exactly once before any exact graph or callback was constructed.
///
/// This remains crate-private so later lifecycle code cannot accidentally re-read a hostile
/// [`PreparedAudioOutput::config`] implementation after the graph has been configured.
pub(crate) struct ValidatedPreparedAudioOutput {
    prepared: Box<dyn PreparedAudioOutput>,
    config: AudioOutputConfig,
}

pub(crate) struct ValidatePreparedAudioOutputFailure {
    pub(crate) error: AudioOutputError,
    pub(crate) prepared: Box<dyn PreparedAudioOutput>,
    pub(crate) panicked: bool,
}

pub(crate) struct ValidatePreparedAudioOutputConfigFailure {
    pub(crate) error: AudioOutputError,
    pub(crate) panicked: bool,
}

impl ValidatedPreparedAudioOutput {
    fn inspect_once(
        request: &AudioOutputRequest,
        prepared: &dyn PreparedAudioOutput,
    ) -> Result<AudioOutputConfig, ValidatePreparedAudioOutputConfigFailure> {
        let config = match panic::catch_unwind(AssertUnwindSafe(|| prepared.config().clone())) {
            Ok(config) => config,
            Err(payload) => {
                std::mem::forget(payload);
                return Err(ValidatePreparedAudioOutputConfigFailure {
                    error: AudioOutputError::new(
                        AudioOutputErrorKind::BackendSpecific,
                        "prepared output panicked while reporting its negotiated configuration",
                    ),
                    panicked: true,
                });
            }
        };
        request.validate_config(&config).map_err(|error| {
            ValidatePreparedAudioOutputConfigFailure {
                error,
                panicked: false,
            }
        })?;
        Ok(config)
    }

    pub(crate) fn try_new(
        request: &AudioOutputRequest,
        prepared: Box<dyn PreparedAudioOutput>,
    ) -> Result<Self, ValidatePreparedAudioOutputFailure> {
        let config = match Self::inspect_once(request, &*prepared) {
            Ok(config) => config,
            Err(failure) => {
                return Err(ValidatePreparedAudioOutputFailure {
                    error: failure.error,
                    panicked: failure.panicked,
                    prepared,
                });
            }
        };
        Ok(Self { prepared, config })
    }

    pub(crate) const fn config(&self) -> &AudioOutputConfig {
        &self.config
    }

    pub(crate) fn into_parts(self) -> (Box<dyn PreparedAudioOutput>, AudioOutputConfig) {
        (self.prepared, self.config)
    }

    #[cfg(test)]
    pub(crate) fn from_prevalidated_for_test(
        prepared: Box<dyn PreparedAudioOutput>,
        config: AudioOutputConfig,
    ) -> Self {
        Self { prepared, config }
    }
}

/// Running endpoint owner, separate from the render callback it installed.
///
/// These methods run on a control or lifecycle thread, never on the render callback. Shutdown
/// consumes the owner so it can be committed exactly once.
pub trait RunningAudioOutput: Send + 'static {
    /// Starts or resumes this logical endpoint.
    fn resume(&mut self) -> Result<(), AudioOutputError>;

    /// Suspends this logical endpoint.
    fn suspend(&mut self) -> Result<(), AudioOutputError>;

    /// Synchronously commits endpoint-local shutdown.
    ///
    /// Before returning, this method must prevent the endpoint from accepting or initiating new
    /// render work. The returned future only awaits already-committed endpoint retirement.
    /// Completion is not proof that the graph, reclaim queues, or context threads have retired.
    /// This method must report failures through its returned future rather than panic.
    fn shutdown(self: Box<Self>) -> AudioOutputEndpointShutdown;
}

/// Failure to transition a prepared endpoint into its running state.
#[derive(Debug)]
pub struct AudioOutputStartFailure {
    error: AudioOutputError,
    shutdown: AudioOutputEndpointShutdown,
}

impl AudioOutputStartFailure {
    /// Constructs a startup failure and its endpoint-local partial-start retirement future.
    ///
    /// The endpoint must already have synchronously committed to accepting no new work.
    #[must_use]
    pub fn new(error: AudioOutputError, shutdown: AudioOutputEndpointShutdown) -> Self {
        Self { error, shutdown }
    }

    /// Startup error.
    #[must_use]
    pub const fn error(&self) -> &AudioOutputError {
        &self.error
    }

    /// Mutable access to the single-owner endpoint-local cleanup future.
    pub const fn shutdown(&mut self) -> &mut AudioOutputEndpointShutdown {
        &mut self.shutdown
    }

    /// Splits the failure into its error and endpoint-local cleanup future.
    pub fn into_parts(self) -> (AudioOutputError, AudioOutputEndpointShutdown) {
        (self.error, self.shutdown)
    }
}

impl fmt::Display for AudioOutputStartFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "audio output failed to start: {}", self.error)
    }
}

impl Error for AudioOutputStartFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}

/// Single-owner future awaiting already-committed endpoint-local shutdown work.
///
/// `Ok(())` certifies that the endpoint destroyed the [`AudioRenderCallback`] passed to `start`,
/// no callback invocation remains in flight, and no future callback invocation can begin. (For a
/// prepared endpoint that never received a callback, this condition is vacuous.) The future is
/// intentionally not cloneable and does not certify context shutdown. The crate-owned hosted
/// lifecycle controller retains and polls it, then creates an authoritative context receipt
/// only after render-state reclamation and context-thread joins are also confirmed.
///
/// Each poll must be nonblocking. `Ok(())` additionally confirms retirement of endpoint-owned
/// logical threads and handles; resources shared by a factory-wide physical mixer are excluded.
/// On `Err`, the future must continue owning any admission or host leases whose destruction would
/// make an unconfirmed shutdown unsafe, because the lifecycle controller quarantines that future.
#[must_use = "endpoint-local shutdown does not run to completion unless this future is polled"]
pub struct AudioOutputEndpointShutdown {
    future: Pin<Box<dyn Future<Output = Result<(), AudioOutputError>> + Send + 'static>>,
}

impl fmt::Debug for AudioOutputEndpointShutdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioOutputEndpointShutdown")
            .finish_non_exhaustive()
    }
}

impl AudioOutputEndpointShutdown {
    /// Constructs endpoint-local shutdown that is already finished.
    pub fn ready(result: Result<(), AudioOutputError>) -> Self {
        Self::from_future(ready(result))
    }

    /// Wraps endpoint-local asynchronous retirement work.
    ///
    /// Endpoint implementations must synchronously commit to accepting no new work before
    /// constructing or returning this future. The future may resolve to `Ok(())` only after any
    /// installed [`AudioRenderCallback`] object has been destroyed and no invocation remains in
    /// flight.
    pub fn from_future<F>(future: F) -> Self
    where
        F: Future<Output = Result<(), AudioOutputError>> + Send + 'static,
    {
        Self {
            future: Box::pin(future),
        }
    }
}

impl Future for AudioOutputEndpointShutdown {
    type Output = Result<(), AudioOutputError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.future.as_mut().poll(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::panic_any;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{self, SyncSender};
    use std::sync::Arc;
    use std::task::Poll;
    use std::thread::{self, ThreadId};

    use futures_util::task::noop_waker;

    use super::*;

    fn format() -> AudioRenderFormat {
        AudioRenderFormat::new(48_000., 2, 128).unwrap()
    }

    fn config() -> AudioOutputConfig {
        AudioOutputConfig::new(format(), "test", 0.).unwrap()
    }

    fn request(sample_rate: Option<f32>) -> Result<AudioOutputRequest, AudioOutputError> {
        request_with_channels(sample_rate, 2)
    }

    fn request_with_channels(
        sample_rate: Option<f32>,
        number_of_channels: usize,
    ) -> Result<AudioOutputRequest, AudioOutputError> {
        AudioOutputRequest::new(
            AudioOutputContextId::new(1).unwrap(),
            "test",
            sample_rate,
            number_of_channels,
            AudioContextLatencyCategory::Interactive,
            AudioContextRenderSizeCategory::Default,
            Some("output test".to_string()),
        )
    }

    type InstrumentedRender = Box<dyn FnMut(&mut [f32]) + Send>;

    struct InstrumentedDriver {
        render: InstrumentedRender,
        dropped: Option<SyncSender<ThreadId>>,
    }

    impl AudioRenderDriver for InstrumentedDriver {
        fn render_interleaved_f32(&mut self, output: &mut [f32]) {
            (self.render)(output);
        }

        fn reclaim_off_thread(self: Box<Self>) -> Result<(), AudioOutputError> {
            drop(self);
            Ok(())
        }
    }

    impl Drop for InstrumentedDriver {
        fn drop(&mut self) {
            if let Some(sender) = self.dropped.take() {
                let _ = sender.send(thread::current().id());
            }
        }
    }

    struct DropThreadProbe(SyncSender<ThreadId>);

    impl Drop for DropThreadProbe {
        fn drop(&mut self) {
            let _ = self.0.send(thread::current().id());
        }
    }

    fn instrumented_pair<F>(
        render: F,
        dropped: Option<SyncSender<ThreadId>>,
    ) -> (
        AudioRenderOwner,
        AudioRenderCallback,
        AudioOutputEventWatcher,
    )
    where
        F: FnMut(&mut [f32]) + Send + 'static,
    {
        let (events, watcher) = AudioOutputEventSink::bounded(1);
        let (owner, callback) = audio_render_pair(
            format(),
            Box::new(InstrumentedDriver {
                render: Box::new(render),
                dropped,
            }),
            events,
        );
        (owner, callback, watcher)
    }

    fn confirmed_reclaim(owner: AudioRenderOwner) -> Result<(), AudioOutputError> {
        owner.begin_shutdown();
        match owner.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new()) {
            Ok(result) => result,
            Err(_) => panic!("confirmed endpoint did not retire its callback"),
        }
    }

    #[test]
    fn request_and_configuration_validation() {
        assert!(AudioOutputContextId::new(0).is_none());
        assert_eq!(AudioOutputContextId::new(7).unwrap().get(), 7);

        let valid_request = request(Some(48_000.)).unwrap();
        assert_eq!(valid_request.context_id().get(), 1);
        assert_eq!(valid_request.sink_id(), "test");
        assert_eq!(valid_request.requested_sample_rate(), Some(48_000.));
        assert_eq!(valid_request.number_of_channels(), 2);
        assert_eq!(valid_request.diagnostic_label(), Some("output test"));
        assert_eq!(
            request(Some(2_999.)).unwrap_err().kind(),
            AudioOutputErrorKind::InvalidArgument
        );
        assert_eq!(
            request_with_channels(None, 0).unwrap_err().kind(),
            AudioOutputErrorKind::InvalidArgument
        );

        let render_format = format();
        assert_eq!(render_format.sample_rate(), 48_000.);
        assert_eq!(render_format.number_of_channels(), 2);
        assert_eq!(render_format.max_frames_per_callback(), 128);
        assert!(AudioRenderFormat::new(48_000., 2, 0).is_err());
        assert!(AudioRenderFormat::new(48_000., 2, MAX_AUDIO_OUTPUT_CALLBACK_FRAMES + 1).is_err());
        assert!(AudioOutputConfig::new(render_format, "test", -1.).is_err());

        valid_request.validate_config(&config()).unwrap();
        let mismatched =
            AudioOutputConfig::new(AudioRenderFormat::new(44_100., 2, 128).unwrap(), "test", 0.)
                .unwrap();
        assert_eq!(
            valid_request
                .validate_config(&mismatched)
                .unwrap_err()
                .kind(),
            AudioOutputErrorKind::NotSupported
        );
        let mismatched_channels =
            AudioOutputConfig::new(AudioRenderFormat::new(48_000., 1, 128).unwrap(), "test", 0.)
                .unwrap();
        assert_eq!(
            valid_request
                .validate_config(&mismatched_channels)
                .unwrap_err()
                .kind(),
            AudioOutputErrorKind::NotSupported
        );
        let mismatched_sink = AudioOutputConfig::new(format(), "other", 0.).unwrap();
        assert_eq!(
            valid_request
                .validate_config(&mismatched_sink)
                .unwrap_err()
                .kind(),
            AudioOutputErrorKind::NotSupported
        );

        let default_request = AudioOutputRequest::new(
            AudioOutputContextId::new(1).unwrap(),
            "",
            Some(48_000.),
            2,
            AudioContextLatencyCategory::Interactive,
            AudioContextRenderSizeCategory::Default,
            None,
        )
        .unwrap();
        default_request
            .validate_config(&AudioOutputConfig::new(format(), "resolved-default", 0.).unwrap())
            .unwrap();
    }

    #[test]
    fn endpoint_death_is_authoritative_when_wake_is_full() {
        let (sink, watcher) = AudioOutputEventSink::bounded(1);
        assert!(sink.try_send(AudioOutputEvent::Underrun { frames: 2 }));
        assert!(sink.try_send(AudioOutputEvent::EndpointDied(
            AudioOutputDeathReason::BackendFailure
        )));
        assert!(!sink.report_endpoint_death(AudioOutputDeathReason::FactoryShutdown));
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::BackendFailure)
        );
        assert_eq!(
            watcher.try_recv(),
            Some(AudioOutputEvent::Underrun { frames: 2 })
        );
        assert_eq!(watcher.try_recv(), None);
    }

    #[test]
    fn callback_is_send_and_exposes_its_format() {
        fn assert_send<T: Send>() {}
        assert_send::<AudioRenderCallback>();

        let (owner, callback, _) = instrumented_pair(|_| {}, None);
        assert_eq!(callback.format(), format());
        owner.begin_shutdown();
        drop(callback);
        confirmed_reclaim(owner).unwrap();
    }

    #[test]
    fn successful_callback_is_allocation_free() {
        let (owner, mut callback, _) = instrumented_pair(|output| output.fill(0.25), None);
        let mut output = [0.; 16];
        let mut status = AudioRenderStatus::Stop;
        alloc_counter::deny_alloc(|| {
            status = callback.render_interleaved_f32(&mut output);
        });
        assert_eq!(status, AudioRenderStatus::Continue);
        assert_eq!(output, [0.25; 16]);

        owner.begin_shutdown();
        drop(callback);
        confirmed_reclaim(owner).unwrap();
    }

    #[test]
    fn protocol_violations_silence_the_entire_buffer_and_close() {
        for length in [0, 3, 258] {
            let calls = Arc::new(AtomicUsize::new(0));
            let calls_for_render = Arc::clone(&calls);
            let (owner, mut callback, watcher) = instrumented_pair(
                move |_| {
                    calls_for_render.fetch_add(1, Ordering::Relaxed);
                },
                None,
            );
            let mut output = vec![1.; length];
            assert_eq!(
                callback.render_interleaved_f32(&mut output),
                AudioRenderStatus::Stop
            );
            assert!(output.iter().all(|&sample| sample == 0.));
            assert_eq!(calls.load(Ordering::Relaxed), 0);
            assert_eq!(
                watcher.death_reason(),
                Some(AudioOutputDeathReason::CallbackProtocolViolation)
            );

            let mut valid_output = [1.; 4];
            assert_eq!(
                callback.render_interleaved_f32(&mut valid_output),
                AudioRenderStatus::Stop
            );
            assert_eq!(valid_output, [0.; 4]);
            drop(callback);
            confirmed_reclaim(owner).unwrap();
        }
    }

    #[test]
    fn close_during_entry_waits_for_the_active_callback() {
        let (entered_send, entered_recv) = mpsc::sync_channel(0);
        let (release_send, release_recv) = mpsc::sync_channel(0);
        let (owner, mut callback, _) = instrumented_pair(
            move |output| {
                entered_send.send(()).unwrap();
                release_recv.recv().unwrap();
                output.fill(0.5);
            },
            None,
        );

        let callback_thread = thread::spawn(move || {
            let mut output = [0.; 4];
            let status = callback.render_interleaved_f32(&mut output);
            drop(callback);
            (status, output)
        });
        entered_recv.recv().unwrap();
        owner.begin_shutdown();
        assert_eq!(
            owner.slot.as_ref().unwrap().gate.load(Ordering::Acquire),
            RENDER_GATE_CLOSED_ACTIVE
        );
        release_send.send(()).unwrap();
        let (status, output) = callback_thread.join().unwrap();
        assert_eq!(status, AudioRenderStatus::Continue);
        assert_eq!(output, [0.5; 4]);
        assert_eq!(
            owner.slot.as_ref().unwrap().gate.load(Ordering::Acquire),
            RENDER_GATE_CLOSED
        );
        confirmed_reclaim(owner).unwrap();
    }

    #[test]
    fn arc_uniqueness_is_the_callback_retirement_proof() {
        let (owner, callback, _) = instrumented_pair(|_| {}, None);
        owner.begin_shutdown();
        let owner = owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .expect_err("a live callback lease must prevent reclamation");
        drop(callback);
        confirmed_reclaim(owner).unwrap();
    }

    #[test]
    fn dropping_an_open_callback_latches_unexpected_retirement() {
        let (owner, callback, watcher) = instrumented_pair(|_| {}, None);
        drop(callback);
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackRetiredUnexpectedly)
        );
        confirmed_reclaim(owner).unwrap();
    }

    #[test]
    fn panic_payload_and_renderer_are_reclaimed_off_callback_thread() {
        let (payload_drop_send, payload_drop_recv) = mpsc::sync_channel(1);
        let (renderer_drop_send, renderer_drop_recv) = mpsc::sync_channel(1);
        let mut payload = Some(DropThreadProbe(payload_drop_send));
        let (owner, mut callback, watcher) = instrumented_pair(
            move |_| panic_any(payload.take().unwrap()),
            Some(renderer_drop_send),
        );

        let callback_thread = thread::spawn(move || {
            let callback_thread = thread::current().id();
            let mut output = [1.; 4];
            let status = callback.render_interleaved_f32(&mut output);
            drop(callback);
            (callback_thread, status, output)
        });
        let (callback_thread, status, output) = callback_thread.join().unwrap();
        assert_eq!(status, AudioRenderStatus::Stop);
        assert_eq!(output, [0.; 4]);
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackPanicked)
        );

        let reclaimer = thread::spawn(move || {
            let reclaimer_thread = thread::current().id();
            confirmed_reclaim(owner).unwrap();
            reclaimer_thread
        });
        let reclaimer_thread = reclaimer.join().unwrap();
        assert_ne!(reclaimer_thread, callback_thread);
        assert_eq!(renderer_drop_recv.recv().unwrap(), reclaimer_thread);
        assert_eq!(payload_drop_recv.recv().unwrap(), reclaimer_thread);
    }

    #[test]
    fn lying_or_missing_confirmation_quarantines_render_state() {
        let (lying_drop_send, lying_drop_recv) = mpsc::sync_channel(1);
        let (owner, callback, _) = instrumented_pair(|_| {}, Some(lying_drop_send));
        owner.begin_shutdown();
        let quarantined = owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .expect_err("confirmation cannot override a live callback Arc");
        drop(quarantined);
        drop(callback);
        assert_eq!(lying_drop_recv.try_recv(), Err(mpsc::TryRecvError::Empty));

        let (missing_drop_send, missing_drop_recv) = mpsc::sync_channel(1);
        let (owner, callback, _) = instrumented_pair(|_| {}, Some(missing_drop_send));
        owner.begin_shutdown();
        drop(callback);
        drop(owner);
        assert_eq!(missing_drop_recv.try_recv(), Err(mpsc::TryRecvError::Empty));
    }

    #[test]
    fn callback_death_remains_authoritative_when_event_queue_is_saturated() {
        let (events, watcher) = AudioOutputEventSink::bounded(1);
        assert!(events.try_send(AudioOutputEvent::Underrun { frames: 7 }));
        let (owner, mut callback) = audio_render_pair(
            format(),
            Box::new(InstrumentedDriver {
                render: Box::new(|_| panic!("render failure")),
                dropped: None,
            }),
            events,
        );
        let mut output = [1.; 4];
        assert_eq!(
            callback.render_interleaved_f32(&mut output),
            AudioRenderStatus::Stop
        );
        assert_eq!(output, [0.; 4]);
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackPanicked)
        );
        assert_eq!(
            watcher.try_recv(),
            Some(AudioOutputEvent::Underrun { frames: 7 })
        );
        assert_eq!(watcher.try_recv(), None);
        drop(callback);
        confirmed_reclaim(owner).unwrap();
    }

    #[test]
    fn endpoint_shutdown_is_single_owner_and_polls_wrapped_work() {
        fn assert_send<T: Send>() {}
        assert_send::<AudioOutputEndpointShutdown>();

        let error = AudioOutputError::new(AudioOutputErrorKind::Shutdown, "failed");
        let mut ready_shutdown = Box::pin(AudioOutputEndpointShutdown::ready(Err(error.clone())));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert_eq!(
            ready_shutdown.as_mut().poll(&mut context),
            Poll::Ready(Err(error))
        );

        let mut wrapped = Box::pin(AudioOutputEndpointShutdown::from_future(async { Ok(()) }));
        assert_eq!(wrapped.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    }

    #[test]
    fn output_traits_are_object_safe() {
        fn accept_factory(_: &dyn AudioOutputFactory) {}
        fn accept_prepared(_: &dyn PreparedAudioOutput) {}
        fn accept_running(_: &dyn RunningAudioOutput) {}

        struct Factory;
        struct Prepared(AudioOutputConfig);
        struct Running;

        impl AudioOutputFactory for Factory {
            fn prepare(
                &self,
                _request: &AudioOutputRequest,
            ) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
                Ok(Box::new(Prepared(config())))
            }
        }

        impl PreparedAudioOutput for Prepared {
            fn config(&self) -> &AudioOutputConfig {
                &self.0
            }

            fn start(
                self: Box<Self>,
                _callback: AudioRenderCallback,
                _events: AudioOutputEventSink,
            ) -> Result<Box<dyn RunningAudioOutput>, AudioOutputStartFailure> {
                Ok(Box::new(Running))
            }

            fn abort(self: Box<Self>) -> AudioOutputEndpointShutdown {
                AudioOutputEndpointShutdown::ready(Ok(()))
            }
        }

        impl RunningAudioOutput for Running {
            fn resume(&mut self) -> Result<(), AudioOutputError> {
                Ok(())
            }

            fn suspend(&mut self) -> Result<(), AudioOutputError> {
                Ok(())
            }

            fn shutdown(self: Box<Self>) -> AudioOutputEndpointShutdown {
                AudioOutputEndpointShutdown::ready(Ok(()))
            }
        }

        let factory = Factory;
        accept_factory(&factory);
        let prepared = factory.prepare(&request(None).unwrap()).unwrap();
        accept_prepared(prepared.as_ref());
        assert_eq!(prepared.config().format(), format());
        let mut running: Box<dyn RunningAudioOutput> = Box::new(Running);
        accept_running(running.as_ref());
        running.resume().unwrap();
    }
}
