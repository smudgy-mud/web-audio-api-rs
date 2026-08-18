//! Host-supplied audio output contracts.
//!
//! This module is an additive API foundation. [`AudioContext`](crate::context::AudioContext)
//! does not invoke these traits yet, and this module does not expose render-thread storage or
//! implement endpoint ownership. Context-level shutdown acknowledgment, callback retirement,
//! reclamation, and thread joins belong to later lifecycle work.

use std::cell::Cell;
use std::error::Error;
use std::fmt;
use std::future::{ready, Future};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use crossbeam_channel::{Receiver, Sender};

use crate::context::{AudioContextLatencyCategory, AudioContextRenderSizeCategory};
use crate::{is_valid_sample_rate, MAX_CHANNELS};

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
/// of permission or resource admission. The embedding layer must perform those checks before
/// future lifecycle integration invokes a factory.
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
    #[allow(dead_code)] // constructed by the pending context lifecycle integration
    pub(crate) fn new(
        context_id: AudioOutputContextId,
        sink_id: impl Into<String>,
        requested_sample_rate: Option<f32>,
        number_of_channels: usize,
        latency_hint: AudioContextLatencyCategory,
        render_size_hint: AudioContextRenderSizeCategory,
        diagnostic_label: Option<String>,
    ) -> Result<Self, AudioOutputError> {
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

    #[allow(dead_code)] // enforced by the pending context lifecycle integration
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
    /// The render callback returned [`AudioRenderStatus::Stop`].
    CallbackStopped = 1,
    /// The callback panicked and was converted to silence.
    CallbackPanicked = 2,
    /// The physical or logical device became unavailable.
    DeviceUnavailable = 3,
    /// The backend reported an implementation-specific terminal failure.
    BackendFailure = 4,
    /// The factory or its shared endpoint was shut down.
    FactoryShutdown = 5,
}

impl AudioOutputDeathReason {
    fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            1 => Self::CallbackStopped,
            2 => Self::CallbackPanicked,
            3 => Self::DeviceUnavailable,
            4 => Self::BackendFailure,
            5 => Self::FactoryShutdown,
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

    #[allow(dead_code)] // constructed by the pending context lifecycle integration
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

    #[allow(dead_code)]
    pub(crate) fn death_reason(&self) -> Option<AudioOutputDeathReason> {
        AudioOutputDeathReason::from_u8(self.state.death_reason.load(Ordering::Acquire))
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
/// This foundation deliberately exposes no callback invocation. A later lifecycle slice will
/// replace this format-only placeholder with bounded render storage and a `RenderSlot`; panic
/// payloads and retired renderer ownership must then move to off-real-time reclamation.
pub struct AudioRenderCallback {
    format: AudioRenderFormat,
    not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for AudioRenderCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioRenderCallback")
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl AudioRenderCallback {
    /// Logical format produced by this callback.
    #[must_use]
    pub const fn format(&self) -> AudioRenderFormat {
        self.format
    }

    #[allow(dead_code)] // constructed by the pending RenderSlot integration
    pub(crate) const fn new(format: AudioRenderFormat) -> Self {
        Self {
            format,
            not_sync: PhantomData,
        }
    }
}

/// Factory for one independently owned logical output per audio context.
///
/// This trait is object-safe. Context construction does not call it yet.
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
/// This future is intentionally not cloneable and does not certify context shutdown. A later
/// crate-owned lifecycle controller will retain and poll it, then create an authoritative context
/// receipt only after callback retirement, reclamation gates, and thread joins are confirmed.
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
    /// constructing or returning this future.
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
    use std::task::Poll;

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
    fn callback_placeholder_is_send_and_exposes_only_its_format() {
        fn assert_send<T: Send>() {}
        assert_send::<AudioRenderCallback>();
        assert_eq!(AudioRenderCallback::new(format()).format(), format());
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
