//! Public builder for one exact audio graph rendered by a caller-supplied output factory.

use std::error::Error;
use std::fmt;
use std::future::Future;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread;

use super::injected_control::{injected_control_channel, InjectedControlError};
use super::injected_ids::injected_node_id_pair;
use super::injected_magic_construction::MAGIC_COMMAND_COUNT;
use super::injected_node_construction::InjectedNodeConstructor;
use super::injected_node_lifetime::{
    injected_node_lifetime_registry, DEFAULT_NODE_LIFETIME_CAPACITY,
};
use super::output_lifecycle::{
    InjectedOutputLifecycleController, InjectedOutputStart, InjectedOutputStateControl,
    InjectedOutputWorkerBootstrap, InjectedStateChangeFailure, InjectedStateChangeOutcome,
    InjectedStateChangeReceipt, OutputShutdownIssue, OutputShutdownIssueKind, OutputShutdownMode,
    OutputShutdownOutcome, OutputShutdownReceipt,
};
use super::{
    AudioContext, AudioContextOptions, AudioControlBatchReservation, ConcreteBaseAudioContext,
    InjectedContextAdmissionGate,
};
use crate::events::injected_event_dispatch_setup;
use crate::output::{
    AudioOutputConfig, AudioOutputContextId, AudioOutputDeathReason, AudioOutputError,
    AudioOutputEventSink, AudioOutputFactory, AudioOutputRequest, PreparedAudioOutput,
    ValidatedPreparedAudioOutput,
};
use crate::stats::AudioStats;
use crate::{AudioPlaybackStats, AudioRenderCapacity};

const HOSTED_CONTROL_CAPACITY: usize = 256;
const HOSTED_OUTPUT_EVENT_CAPACITY: usize = 8;
static NEXT_HOSTED_OUTPUT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
#[cfg(test)]
thread_local! {
    static HOSTED_CONTEXT_ID_SOURCE_FOR_TEST: std::cell::RefCell<Option<Arc<AtomicU64>>> = const {
        std::cell::RefCell::new(None)
    };
    static FORCE_NEXT_OUTPUT_EVENT_MISMATCH_FOR_TEST: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

/// Stable category for hosted-context construction failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioContextBuildErrorKind {
    /// Builder options or a returned output configuration were invalid.
    InvalidConfiguration,
    /// The supplied output factory rejected the request.
    OutputRejected,
    /// The supplied output implementation panicked.
    OutputPanicked,
    /// An exact graph, event, or lifecycle invariant failed before publication.
    BootstrapFailed,
    /// The prepared output did not become a running endpoint.
    StartFailed,
    /// A lifecycle worker could not be established.
    LifecycleUnavailable,
}

/// Construction error with a cancellation-independent cleanup observation when crate-owned
/// output resources had already been acquired.
#[derive(Clone, Debug)]
pub struct AudioContextBuildError {
    kind: AudioContextBuildErrorKind,
    message: Arc<str>,
    output_error: Option<AudioOutputError>,
    cleanup: Option<AudioContextShutdownReceipt>,
}

impl AudioContextBuildError {
    /// Stable error category.
    #[must_use]
    pub const fn kind(&self) -> AudioContextBuildErrorKind {
        self.kind
    }

    /// Output error returned by the factory or endpoint, when available.
    #[must_use]
    pub const fn output_error(&self) -> Option<&AudioOutputError> {
        self.output_error.as_ref()
    }

    /// Observer for cleanup already owned by the crate.
    ///
    /// `None` means no `PreparedAudioOutput` was acquired; it makes no claim about private work a
    /// factory performed before returning an error or panicking. An immediately ready
    /// [`AudioContextShutdownOutcome::Unconfirmed`] is a fail-closed proof classification: it
    /// does not imply that a transferred `PreparedAudioOutput::abort` future has finished or that
    /// quarantined resources are eligible for release.
    #[must_use]
    pub fn cleanup_receipt(&self) -> Option<AudioContextShutdownReceipt> {
        self.cleanup.clone()
    }
}

impl fmt::Display for AudioContextBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for AudioContextBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.output_error
            .as_ref()
            .map(|error| error as &(dyn Error + 'static))
    }
}

/// Immediate hosted lifecycle request failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioContextLifecycleError {
    /// Another caller currently owns the short request serialization boundary.
    Contended,
    /// The fixed request or graph-control capacity is exhausted.
    Capacity,
    /// Shutdown has sealed this context.
    Closed,
    /// The exact control transport is no longer usable.
    Failed,
    /// This receipt-oriented API is unavailable on a legacy-system context.
    LegacyContext,
}

/// Result of one accepted hosted suspend or resume request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioContextStateChangeOutcome {
    /// Graph state, state event enqueue, and native endpoint transition all completed.
    Applied,
    /// Graph and endpoint were already settled in the requested state.
    Unchanged,
    /// Authoritative shutdown won after this request was admitted.
    SupersededByShutdown,
    /// The request observed an already sealed context.
    Closed,
    /// Graph-control transport failed.
    TransportFailed,
    /// State changed but its state event could not be enqueued.
    EventDeliveryFailed,
    /// The native endpoint's state became uncertain; fail-closed shutdown owns it now.
    EndpointUncertain,
    /// The lifecycle worker panicked or terminated without its normal result.
    WorkerFailed,
}

/// Cloneable observation of a cancellation-independent hosted state transition.
///
/// There is no lifecycle deadline. A live endpoint or callback stall leaves this pending until
/// native progress resumes or authoritative shutdown supersedes the request.
#[derive(Clone)]
#[must_use = "dropping the receipt abandons observation but does not cancel the accepted request"]
pub struct AudioContextStateChangeReceipt {
    inner: InjectedStateChangeReceipt,
}

impl fmt::Debug for AudioContextStateChangeReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioContextStateChangeReceipt")
            .finish_non_exhaustive()
    }
}

impl Future for AudioContextStateChangeReceipt {
    type Output = AudioContextStateChangeOutcome;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.inner).poll(cx).map(map_state_outcome)
    }
}

impl AudioContextStateChangeReceipt {
    /// Blocks without a deadline until the already-accepted transition is observed.
    #[must_use]
    pub fn wait(self) -> AudioContextStateChangeOutcome {
        block_on_receipt(self)
    }
}

/// Requested/effective context shutdown mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioContextShutdownMode {
    /// Drain eligible events before final `Closed` delivery.
    Graceful,
    /// Discard unrelated event backlog while still publishing terminal state when proven.
    Silent,
}

/// Stable category for a hosted lifecycle degradation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioContextShutdownIssueKind {
    OutputMethod,
    OutputFuture,
    OutputState,
    Callback,
    GraphReclamation,
    EventDelivery,
    EventThread,
    Worker,
    Bootstrap,
    Protocol,
}

/// Clone-cheap hosted lifecycle diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioContextShutdownIssue {
    kind: AudioContextShutdownIssueKind,
    message: Arc<str>,
}

impl AudioContextShutdownIssue {
    #[must_use]
    pub const fn kind(&self) -> AudioContextShutdownIssueKind {
        self.kind
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Confirmed hosted shutdown details.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioContextShutdownReport {
    mode: AudioContextShutdownMode,
    endpoint_death: Option<AudioOutputDeathReason>,
    reclaim_issue: Option<AudioContextShutdownIssue>,
    event_issue: Option<AudioContextShutdownIssue>,
}

impl AudioContextShutdownReport {
    #[must_use]
    pub const fn mode(&self) -> AudioContextShutdownMode {
        self.mode
    }

    #[must_use]
    pub const fn endpoint_death(&self) -> Option<AudioOutputDeathReason> {
        self.endpoint_death
    }

    #[must_use]
    pub const fn reclaim_issue(&self) -> Option<&AudioContextShutdownIssue> {
        self.reclaim_issue.as_ref()
    }

    #[must_use]
    pub const fn event_issue(&self) -> Option<&AudioContextShutdownIssue> {
        self.event_issue.as_ref()
    }
}

/// Terminal proof classification for hosted context or failed-build cleanup.
///
/// `Unconfirmed` may be published immediately after ownership is safely transferred or
/// quarantined. It does not certify completion of endpoint abort work or resource release.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioContextShutdownOutcome {
    Confirmed(AudioContextShutdownReport),
    Unconfirmed {
        failure: AudioContextShutdownIssue,
        event_issue: Option<AudioContextShutdownIssue>,
    },
    ControllerTerminated,
}

#[derive(Clone)]
enum ShutdownReceiptInner {
    Lifecycle(OutputShutdownReceipt),
    Ready(AudioContextShutdownOutcome),
}

/// Cloneable observation of cancellation-independent hosted shutdown.
///
/// [`AudioContext::request_close`] completes its short internal latch before returning this
/// receipt, but does not wait for endpoint, graph, or event-thread retirement. Receipt polling
/// only observes the independently owned lifecycle work.
///
/// There is no lifecycle deadline. Polling this receipt from the same hosted context's event
/// thread panics because confirmation requires that event thread to retire and be joined.
#[derive(Clone)]
#[must_use = "dropping the receipt abandons observation but does not cancel shutdown"]
pub struct AudioContextShutdownReceipt {
    inner: ShutdownReceiptInner,
    event_thread_id: Option<thread::ThreadId>,
}

impl fmt::Debug for AudioContextShutdownReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioContextShutdownReceipt")
            .finish_non_exhaustive()
    }
}

impl Future for AudioContextShutdownReceipt {
    type Output = AudioContextShutdownOutcome;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self
            .event_thread_id
            .is_some_and(|thread_id| thread_id == thread::current().id())
        {
            panic!("InvalidStateError: cannot await context shutdown from its event thread");
        }
        match &mut self.inner {
            ShutdownReceiptInner::Lifecycle(receipt) => {
                Pin::new(receipt).poll(cx).map(map_shutdown_outcome)
            }
            ShutdownReceiptInner::Ready(outcome) => Poll::Ready(outcome.clone()),
        }
    }
}

impl From<OutputShutdownReceipt> for AudioContextShutdownReceipt {
    fn from(inner: OutputShutdownReceipt) -> Self {
        Self {
            inner: ShutdownReceiptInner::Lifecycle(inner),
            event_thread_id: None,
        }
    }
}

impl AudioContextShutdownReceipt {
    fn ready(outcome: AudioContextShutdownOutcome) -> Self {
        Self {
            inner: ShutdownReceiptInner::Ready(outcome),
            event_thread_id: None,
        }
    }

    /// Blocks the current thread until shutdown reaches a terminal observation.
    ///
    /// There is no deadline: a conforming live endpoint may keep this pending until its native
    /// retirement completes. Calling this from the same hosted context's event thread panics,
    /// because confirmed retirement must join that thread.
    ///
    /// # Panics
    ///
    /// Panics when called from the same hosted context's event thread.
    #[must_use]
    pub fn wait(self) -> AudioContextShutdownOutcome {
        if self
            .event_thread_id
            .is_some_and(|thread_id| thread_id == thread::current().id())
        {
            panic!(
                "InvalidStateError: cannot synchronously wait for a context from its event thread"
            );
        }
        block_on_receipt(self)
    }

    fn lifecycle(inner: OutputShutdownReceipt, event_thread_id: thread::ThreadId) -> Self {
        Self {
            inner: ShutdownReceiptInner::Lifecycle(inner),
            event_thread_id: Some(event_thread_id),
        }
    }
}

struct ReceiptThreadWake(thread::Thread);

impl Wake for ReceiptThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on_receipt<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ReceiptThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

/// Builder for an [`AudioContext`] driven by a caller-supplied logical output factory.
///
/// This exact hosted path supports the destination/listener magic graph, Gain, ConstantSource,
/// fixed-wave Oscillator, and AudioBufferSource nodes, scalar AudioParam mutation, explicit
/// connections, typed suspend/resume/close receipts, and caller-supplied or system output
/// factories. Custom PeriodicWave data and additional node families remain outside this slice.
/// Disconnected scheduled-source rooting and lossy-ended-event reconciliation remain the embedder's
/// responsibility. A dropped or saturated exact ended event is generation-safe but its callback
/// may remain retained until confirmed whole-context event retirement.
pub struct AudioContextBuilder {
    output: Arc<dyn AudioOutputFactory>,
    options: AudioContextOptions,
    number_of_channels: usize,
    initially_suspended: bool,
    diagnostic_label: Option<String>,
    initial_control_reservation: Option<AudioControlBatchReservation>,
}

impl fmt::Debug for AudioContextBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioContextBuilder")
            .field("options", &self.options)
            .field("number_of_channels", &self.number_of_channels)
            .field("initially_suspended", &self.initially_suspended)
            .field("diagnostic_label", &self.diagnostic_label)
            .finish_non_exhaustive()
    }
}

impl AudioContextBuilder {
    pub(super) fn new(output: Arc<dyn AudioOutputFactory>) -> Self {
        Self {
            output,
            options: AudioContextOptions::default(),
            number_of_channels: 2,
            initially_suspended: false,
            diagnostic_label: None,
            initial_control_reservation: None,
        }
    }

    /// Number of exact graph-control commands used to install the permanent destination and
    /// listener namespace before a hosted context is published.
    pub const INITIAL_GRAPH_CONTROL_COMMAND_COUNT: usize = MAGIC_COMMAND_COUNT;

    /// Replaces the Web Audio output request options.
    #[must_use]
    pub fn options(mut self, options: AudioContextOptions) -> Self {
        self.options = options;
        self
    }

    /// Requests an exact logical output channel count.
    #[must_use]
    pub fn number_of_channels(mut self, number_of_channels: usize) -> Self {
        self.number_of_channels = number_of_channels;
        self
    }

    /// Selects whether graph time and mutations begin suspended.
    #[must_use]
    pub fn initially_suspended(mut self, initially_suspended: bool) -> Self {
        self.initially_suspended = initially_suspended;
        self
    }

    /// Adds a neutral label visible only to the output factory.
    #[must_use]
    pub fn diagnostic_label(mut self, label: impl Into<String>) -> Self {
        self.diagnostic_label = Some(label.into());
        self
    }

    /// Attaches host accounting for the permanent destination/listener graph-control batch.
    ///
    /// The caller must reserve [`Self::INITIAL_GRAPH_CONTROL_COMMAND_COUNT`] commands before
    /// invoking the output factory. The reservation is released with that exact batch on
    /// rejection or after accepted off-render-thread reclamation; a fail-closed quarantine
    /// deliberately retains it.
    #[must_use]
    pub fn initial_control_reservation(
        mut self,
        reservation: AudioControlBatchReservation,
    ) -> Self {
        self.initial_control_reservation = Some(reservation);
        self
    }

    /// Constructs the exact graph, starts one logical output, and transfers it to a native
    /// lifecycle owner before returning the public context.
    pub fn build(self) -> Result<AudioContext, AudioContextBuildError> {
        build_hosted_context(self)
    }
}

pub(super) struct HostedAudioContextMode {
    config: AudioOutputConfig,
    state_control: InjectedOutputStateControl,
    lifecycle: Mutex<HostedLifecycleState>,
    event_thread_id: thread::ThreadId,
}

struct HostedLifecycleState {
    controller: Option<InjectedOutputLifecycleController>,
    receipt: OutputShutdownReceipt,
}

/// Fail-safe ownership for a prepared endpoint from the instant a hostile factory returns it.
/// Every unwind before callback transfer therefore schedules abort instead of invoking ordinary
/// `PreparedAudioOutput` destruction on the constructing thread.
struct PreparedOutputStage {
    prepared: Option<PreparedStageOwner>,
    worker: Option<InjectedOutputWorkerBootstrap>,
}

enum PreparedStageOwner {
    Raw(Box<dyn PreparedAudioOutput>),
    Validated(ValidatedPreparedAudioOutput),
}

impl PreparedOutputStage {
    fn new(prepared: Box<dyn PreparedAudioOutput>, worker: InjectedOutputWorkerBootstrap) -> Self {
        Self {
            prepared: Some(PreparedStageOwner::Raw(prepared)),
            worker: Some(worker),
        }
    }

    fn validate(
        &mut self,
        request: &AudioOutputRequest,
    ) -> Result<(), crate::output::ValidatePreparedAudioOutputConfigFailure> {
        let PreparedStageOwner::Raw(prepared) = self.prepared.take().unwrap() else {
            unreachable!("prepared output stage validates exactly once")
        };
        match ValidatedPreparedAudioOutput::try_new(request, prepared) {
            Ok(validated) => {
                self.prepared = Some(PreparedStageOwner::Validated(validated));
                Ok(())
            }
            Err(failure) => {
                self.prepared = Some(PreparedStageOwner::Raw(failure.prepared));
                Err(crate::output::ValidatePreparedAudioOutputConfigFailure {
                    error: failure.error,
                    panicked: failure.panicked,
                })
            }
        }
    }

    fn config(&self) -> &AudioOutputConfig {
        match self.prepared.as_ref().unwrap() {
            PreparedStageOwner::Validated(prepared) => prepared.config(),
            PreparedStageOwner::Raw(_) => unreachable!("prepared output has not been validated"),
        }
    }

    fn into_start_parts(mut self) -> (InjectedOutputWorkerBootstrap, ValidatedPreparedAudioOutput) {
        let PreparedStageOwner::Validated(prepared) = self.prepared.take().unwrap() else {
            unreachable!("only a validated prepared output can start")
        };
        let worker = self.worker.take().unwrap();
        (worker, prepared)
    }

    fn abort(mut self) -> AudioContextShutdownReceipt {
        let prepared = match self.prepared.take().unwrap() {
            PreparedStageOwner::Raw(prepared) => prepared,
            PreparedStageOwner::Validated(prepared) => prepared.into_parts().0,
        };
        let worker = self.worker.take().unwrap();
        worker.abort_prepared(prepared).into()
    }
}

impl Drop for PreparedOutputStage {
    fn drop(&mut self) {
        let (Some(prepared), Some(worker)) = (self.prepared.take(), self.worker.take()) else {
            return;
        };
        let prepared = match prepared {
            PreparedStageOwner::Raw(prepared) => prepared,
            PreparedStageOwner::Validated(prepared) => prepared.into_parts().0,
        };
        // This is an unwind-only safety net. A hostile abort future runs only on a lifecycle
        // worker, never on this thread; the receipt may be abandoned without cancelling it.
        let abort = panic::catch_unwind(AssertUnwindSafe(|| worker.abort_prepared(prepared)));
        if let Err(payload) = abort {
            std::mem::forget(payload);
        }
    }
}

impl HostedAudioContextMode {
    pub(super) fn config(&self) -> &AudioOutputConfig {
        &self.config
    }

    pub(super) fn request_suspend(
        &self,
    ) -> Result<AudioContextStateChangeReceipt, AudioContextLifecycleError> {
        self.state_control
            .suspend()
            .map(|inner| AudioContextStateChangeReceipt { inner })
            .map_err(map_control_error)
    }

    pub(super) fn request_resume(
        &self,
    ) -> Result<AudioContextStateChangeReceipt, AudioContextLifecycleError> {
        self.state_control
            .resume()
            .map(|inner| AudioContextStateChangeReceipt { inner })
            .map_err(map_control_error)
    }

    pub(super) fn request_close(
        &self,
    ) -> Result<AudioContextShutdownReceipt, AudioContextLifecycleError> {
        let mut lifecycle = match self.lifecycle.lock() {
            Ok(lifecycle) => lifecycle,
            Err(poisoned) => {
                let mut lifecycle = poisoned.into_inner();
                if let Some(controller) = lifecycle.controller.take() {
                    lifecycle.receipt = controller.shutdown_silently();
                }
                return Ok(AudioContextShutdownReceipt::lifecycle(
                    lifecycle.receipt.clone(),
                    self.event_thread_id,
                ));
            }
        };
        if let Some(controller) = lifecycle.controller.take() {
            lifecycle.receipt = controller.shutdown_gracefully();
        }
        Ok(AudioContextShutdownReceipt::lifecycle(
            lifecycle.receipt.clone(),
            self.event_thread_id,
        ))
    }

    pub(super) fn shutdown_receipt(&self) -> AudioContextShutdownReceipt {
        let lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        AudioContextShutdownReceipt::lifecycle(lifecycle.receipt.clone(), self.event_thread_id)
    }

    pub(super) fn is_event_thread(&self) -> bool {
        self.event_thread_id == thread::current().id()
    }

    #[cfg(test)]
    pub(super) fn set_state_request_hook_for_test(
        &self,
        observer: Arc<dyn Fn() + Send + Sync + 'static>,
    ) {
        self.state_control.set_after_serialize_for_test(observer);
    }

    pub(super) fn request_silent_on_drop(&mut self) {
        let lifecycle = self
            .lifecycle
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(controller) = lifecycle.controller.take() {
            lifecycle.receipt = controller.shutdown_silently();
        }
    }
}

fn build_hosted_context(
    builder: AudioContextBuilder,
) -> Result<AudioContext, AudioContextBuildError> {
    AudioOutputRequest::validate_parts(
        builder.options.sample_rate,
        builder.number_of_channels,
        builder.options.latency_hint,
    )
    .map_err(|error| {
        build_output_error(
            AudioContextBuildErrorKind::InvalidConfiguration,
            error,
            None,
        )
    })?;

    let worker =
        InjectedOutputWorkerBootstrap::try_new().map_err(|error| AudioContextBuildError {
            kind: AudioContextBuildErrorKind::LifecycleUnavailable,
            message: Arc::from(error.to_string()),
            output_error: None,
            cleanup: None,
        })?;

    let request = AudioOutputRequest::new(
        next_context_id()?,
        builder.options.sink_id.clone(),
        builder.options.sample_rate,
        builder.number_of_channels,
        builder.options.latency_hint,
        builder.options.render_size_hint,
        builder.diagnostic_label,
    )
    .map_err(|error| {
        build_output_error(
            AudioContextBuildErrorKind::InvalidConfiguration,
            error,
            None,
        )
    })?;

    let prepared = match panic::catch_unwind(AssertUnwindSafe(|| builder.output.prepare(&request)))
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(error)) => {
            drop(worker);
            return Err(build_output_error(
                AudioContextBuildErrorKind::OutputRejected,
                error,
                None,
            ));
        }
        Err(payload) => {
            std::mem::forget(payload);
            drop(worker);
            return Err(AudioContextBuildError {
                kind: AudioContextBuildErrorKind::OutputPanicked,
                message: Arc::from("audio output factory panicked during prepare"),
                output_error: None,
                cleanup: None,
            });
        }
    };

    let mut stage = PreparedOutputStage::new(prepared, worker);
    if let Err(failure) = stage.validate(&request) {
        let kind = if failure.panicked {
            AudioContextBuildErrorKind::OutputPanicked
        } else {
            AudioContextBuildErrorKind::InvalidConfiguration
        };
        let cleanup = stage.abort();
        return Err(build_output_error(kind, failure.error, Some(cleanup)));
    }
    let config = stage.config().clone();
    let format = config.format();
    let stats = AudioStats::new();
    stats.record_latency_seconds(config.output_latency());
    let frames_played = Arc::new(AtomicU64::new(0));
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) = match injected_control_channel(
        gate,
        HOSTED_CONTROL_CAPACITY,
        builder.initially_suspended,
    ) {
        Ok(parts) => parts,
        Err(_error) => {
            return Err(prepared_bootstrap_error(stage, "control bootstrap failed"));
        }
    };
    let (allocator, node_ids, graph) = injected_node_id_pair(0);
    let (registrar, lifetime_bootstrap) = match injected_node_lifetime_registry(
        DEFAULT_NODE_LIFETIME_CAPACITY,
        &producer,
        node_ids,
        graph,
    ) {
        Ok(parts) => parts,
        Err(failure) => {
            drop(failure);
            return Err(prepared_bootstrap_error(
                stage,
                "node-lifetime bootstrap failed",
            ));
        }
    };
    let constructor = match InjectedNodeConstructor::new(producer, allocator, registrar) {
        Ok(constructor) => constructor,
        Err(failure) => {
            drop(failure);
            return Err(prepared_bootstrap_error(
                stage,
                "node-constructor bootstrap failed",
            ));
        }
    };
    let event_setup = match injected_event_dispatch_setup() {
        Ok(setup) => setup,
        Err(error) => {
            drop(error);
            return Err(prepared_bootstrap_error(
                stage,
                "event-thread bootstrap failed",
            ));
        }
    };
    let renderer = match render_init.build_output_render_thread(
        lifetime_bootstrap,
        format.sample_rate(),
        format.number_of_channels(),
        Arc::clone(&frames_played),
        stats.clone(),
        event_setup,
    ) {
        Ok(renderer) => renderer,
        Err(failure) => {
            drop(failure);
            return Err(prepared_bootstrap_error(
                stage,
                "render bootstrap identity mismatch",
            ));
        }
    };
    let (renderer, binding) = match renderer.bind_output_lifecycle_exact(lifecycle) {
        Ok(parts) => parts,
        Err(failure) => {
            drop(failure);
            return Err(prepared_bootstrap_error(
                stage,
                "output lifecycle identity mismatch",
            ));
        }
    };
    let bootstrap = match ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        renderer,
        constructor,
        binding,
    ) {
        Ok(bootstrap) => bootstrap,
        Err(failure) => {
            drop(failure);
            return Err(prepared_bootstrap_error(
                stage,
                "exact base identity mismatch",
            ));
        }
    };
    let renderer =
        match bootstrap.try_build_with_control_reservation(builder.initial_control_reservation) {
            Ok(renderer) => renderer,
            Err(failure) => {
                drop(failure);
                return Err(prepared_bootstrap_error(
                    stage,
                    "magic graph bootstrap failed",
                ));
            }
        };

    let base = renderer.base().clone();
    let event_thread_id = base
        .injected_event_thread_id()
        .expect("magic-initialized hosted base retains exact event identity");
    let render_capacity = AudioRenderCapacity::new(base.clone(), stats.clone());
    let playback_stats = AudioPlaybackStats::new(base.clone(), stats);
    let (events, output_events) = AudioOutputEventSink::bounded(HOSTED_OUTPUT_EVENT_CAPACITY);
    #[cfg(test)]
    let output_events = if FORCE_NEXT_OUTPUT_EVENT_MISMATCH_FOR_TEST.replace(false) {
        let (_foreign_sink, foreign_events) =
            AudioOutputEventSink::bounded(HOSTED_OUTPUT_EVENT_CAPACITY);
        foreign_events
    } else {
        output_events
    };
    let (worker, prepared) = stage.into_start_parts();
    match worker.start(prepared, renderer, events, output_events) {
        Ok(InjectedOutputStart::Running(controller)) => {
            let state_control = controller.state_control();
            let receipt = controller.receipt();
            Ok(AudioContext::from_hosted_parts(
                base,
                render_capacity,
                playback_stats,
                HostedAudioContextMode {
                    config,
                    state_control,
                    event_thread_id,
                    lifecycle: Mutex::new(HostedLifecycleState {
                        controller: Some(controller),
                        receipt,
                    }),
                },
            ))
        }
        Ok(InjectedOutputStart::Cleanup(cleanup)) => {
            let output_error = cleanup.startup_error().cloned();
            let cleanup_receipt = cleanup.receipt().into();
            drop(cleanup);
            Err(AudioContextBuildError {
                kind: AudioContextBuildErrorKind::StartFailed,
                message: Arc::from(
                    output_error
                        .as_ref()
                        .map_or("audio output start panicked", AudioOutputError::message),
                ),
                output_error,
                cleanup: Some(cleanup_receipt),
            })
        }
        Err(failure) => {
            let issue = map_shutdown_issue(failure.issue());
            let output_error = failure.startup_error().cloned();
            let message = output_error.as_ref().map_or_else(
                || Arc::clone(&issue.message),
                |error| {
                    Arc::from(format!(
                        "{}; output start also failed: {}",
                        issue.message,
                        error.message()
                    ))
                },
            );
            let cleanup = failure.recover_with_fallback().into();
            Err(AudioContextBuildError {
                kind: AudioContextBuildErrorKind::LifecycleUnavailable,
                message,
                output_error,
                cleanup: Some(cleanup),
            })
        }
    }
}

fn next_context_id() -> Result<AudioOutputContextId, AudioContextBuildError> {
    #[cfg(test)]
    let test_source = HOSTED_CONTEXT_ID_SOURCE_FOR_TEST.with(|source| source.borrow().clone());
    #[cfg(test)]
    let source = test_source
        .as_deref()
        .unwrap_or(&NEXT_HOSTED_OUTPUT_CONTEXT_ID);
    #[cfg(not(test))]
    let source = &NEXT_HOSTED_OUTPUT_CONTEXT_ID;
    let id = source
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(1)
        })
        .map_err(|_| bootstrap_error("hosted output context identity exhausted"))?;
    AudioOutputContextId::new(id)
        .ok_or_else(|| bootstrap_error("hosted output context identity exhausted"))
}

#[cfg(test)]
pub(super) fn set_context_id_source_for_test(source: Option<Arc<AtomicU64>>) {
    HOSTED_CONTEXT_ID_SOURCE_FOR_TEST.with(|slot| *slot.borrow_mut() = source);
}

#[cfg(test)]
pub(super) fn fail_next_worker_spawn_for_test() {
    InjectedOutputWorkerBootstrap::fail_next_spawn_for_test();
}

#[cfg(test)]
pub(super) fn fail_next_worker_transfer_for_test() {
    InjectedOutputWorkerBootstrap::fail_next_transfer_for_test();
}

#[cfg(test)]
pub(super) fn fail_next_fallback_worker_spawn_for_test() {
    InjectedOutputWorkerBootstrap::fail_next_fallback_spawn_for_test();
}

#[cfg(test)]
pub(super) fn force_next_prepared_start_failure_for_test() {
    FORCE_NEXT_OUTPUT_EVENT_MISMATCH_FOR_TEST.set(true);
}

fn build_output_error(
    kind: AudioContextBuildErrorKind,
    error: AudioOutputError,
    cleanup: Option<AudioContextShutdownReceipt>,
) -> AudioContextBuildError {
    AudioContextBuildError {
        kind,
        message: Arc::from(error.message()),
        output_error: Some(error),
        cleanup,
    }
}

fn prepared_bootstrap_error(
    stage: PreparedOutputStage,
    message: &'static str,
) -> AudioContextBuildError {
    // Prepared abort is transferred to a native lifecycle worker, but these early exact graph
    // failures have no callback capable of rendering Close. Their proof domain is deliberately
    // quarantined, so the public cleanup observation must never claim confirmation merely because
    // endpoint abort later succeeds.
    drop(stage.abort());
    let cleanup = AudioContextShutdownReceipt::ready(AudioContextShutdownOutcome::Unconfirmed {
        failure: AudioContextShutdownIssue {
            kind: AudioContextShutdownIssueKind::Bootstrap,
            message: Arc::from(message),
        },
        event_issue: None,
    });
    AudioContextBuildError {
        kind: AudioContextBuildErrorKind::BootstrapFailed,
        message: Arc::from(message),
        output_error: None,
        cleanup: Some(cleanup),
    }
}

fn bootstrap_error(message: impl Into<Arc<str>>) -> AudioContextBuildError {
    AudioContextBuildError {
        kind: AudioContextBuildErrorKind::BootstrapFailed,
        message: message.into(),
        output_error: None,
        cleanup: None,
    }
}

fn map_control_error(error: InjectedControlError) -> AudioContextLifecycleError {
    match error {
        InjectedControlError::Contended => AudioContextLifecycleError::Contended,
        InjectedControlError::LogicalCommandCredits
        | InjectedControlError::BatchStorageCredits
        | InjectedControlError::OrdinaryPhysicalCredits
        | InjectedControlError::StagingFull => AudioContextLifecycleError::Capacity,
        InjectedControlError::Sealed => AudioContextLifecycleError::Closed,
        _ => AudioContextLifecycleError::Failed,
    }
}

fn map_state_outcome(outcome: InjectedStateChangeOutcome) -> AudioContextStateChangeOutcome {
    match outcome {
        InjectedStateChangeOutcome::Applied => AudioContextStateChangeOutcome::Applied,
        InjectedStateChangeOutcome::Unchanged => AudioContextStateChangeOutcome::Unchanged,
        InjectedStateChangeOutcome::SupersededByShutdown => {
            AudioContextStateChangeOutcome::SupersededByShutdown
        }
        InjectedStateChangeOutcome::Closed => AudioContextStateChangeOutcome::Closed,
        InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::Transport) => {
            AudioContextStateChangeOutcome::TransportFailed
        }
        InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::EventDelivery) => {
            AudioContextStateChangeOutcome::EventDeliveryFailed
        }
        InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::EndpointUncertain) => {
            AudioContextStateChangeOutcome::EndpointUncertain
        }
        InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::WorkerPanicked)
        | InjectedStateChangeOutcome::ControllerTerminated => {
            AudioContextStateChangeOutcome::WorkerFailed
        }
    }
}

fn map_shutdown_outcome(outcome: OutputShutdownOutcome) -> AudioContextShutdownOutcome {
    match outcome {
        OutputShutdownOutcome::Confirmed(report) => {
            AudioContextShutdownOutcome::Confirmed(AudioContextShutdownReport {
                mode: match report.mode() {
                    OutputShutdownMode::Graceful => AudioContextShutdownMode::Graceful,
                    OutputShutdownMode::Silent => AudioContextShutdownMode::Silent,
                },
                endpoint_death: report.endpoint_death(),
                reclaim_issue: report.reclaim_issue().map(map_shutdown_issue),
                event_issue: report.event_issue().map(map_shutdown_issue),
            })
        }
        OutputShutdownOutcome::Unconfirmed {
            failure,
            event_issue,
        } => AudioContextShutdownOutcome::Unconfirmed {
            failure: map_shutdown_issue(&failure),
            event_issue: event_issue.as_ref().map(map_shutdown_issue),
        },
        OutputShutdownOutcome::ControllerTerminated => {
            AudioContextShutdownOutcome::ControllerTerminated
        }
    }
}

fn map_shutdown_issue(issue: &OutputShutdownIssue) -> AudioContextShutdownIssue {
    let kind = match issue.kind() {
        OutputShutdownIssueKind::EndpointMethodPanicked => {
            AudioContextShutdownIssueKind::OutputMethod
        }
        OutputShutdownIssueKind::EndpointFuturePanicked
        | OutputShutdownIssueKind::EndpointRejectedShutdown => {
            AudioContextShutdownIssueKind::OutputFuture
        }
        OutputShutdownIssueKind::EndpointStateTransitionFailed => {
            AudioContextShutdownIssueKind::OutputState
        }
        OutputShutdownIssueKind::CallbackRetained => AudioContextShutdownIssueKind::Callback,
        OutputShutdownIssueKind::RenderReclaimDegraded
        | OutputShutdownIssueKind::InjectedNodeLifetime
        | OutputShutdownIssueKind::InjectedWholeGraphQuarantined => {
            AudioContextShutdownIssueKind::GraphReclamation
        }
        OutputShutdownIssueKind::EventDeliveryDegraded => {
            AudioContextShutdownIssueKind::EventDelivery
        }
        OutputShutdownIssueKind::EventThreadUnretired => AudioContextShutdownIssueKind::EventThread,
        OutputShutdownIssueKind::WorkerPanicked | OutputShutdownIssueKind::WorkerSpawnFailed => {
            AudioContextShutdownIssueKind::Worker
        }
        OutputShutdownIssueKind::BootstrapFailed => AudioContextShutdownIssueKind::Bootstrap,
        OutputShutdownIssueKind::InjectedStartPanicked
        | OutputShutdownIssueKind::InjectedControlClose => AudioContextShutdownIssueKind::Protocol,
    };
    AudioContextShutdownIssue {
        kind,
        message: Arc::from(issue.message()),
    }
}

pub(super) fn hosted_state_outcome_is_success(outcome: AudioContextStateChangeOutcome) -> bool {
    matches!(
        outcome,
        AudioContextStateChangeOutcome::Applied
            | AudioContextStateChangeOutcome::Unchanged
            | AudioContextStateChangeOutcome::SupersededByShutdown
            | AudioContextStateChangeOutcome::Closed
    )
}

pub(super) fn hosted_shutdown_outcome_is_success(outcome: &AudioContextShutdownOutcome) -> bool {
    matches!(outcome, AudioContextShutdownOutcome::Confirmed(_))
}
