//! Private ownership controller for an injected audio output.
//!
//! No `AudioContext` constructor uses this foundation yet. It deliberately leaves legacy backend
//! ownership untouched until the surrounding context can prove producer quiescence.

#![allow(dead_code)] // production wiring begins with the pending injected AudioContext constructor

mod injected;
#[allow(unused_imports)] // public context wiring follows this private lifecycle slice
pub(crate) use injected::{
    start_injected_output, InjectedOutputLifecycleController, InjectedOutputStart,
    InjectedOutputStartFailure, InjectedOutputStartFailureParts,
};

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use futures_channel::oneshot;
use futures_util::future::{FutureExt, Shared};

use crate::events::{EventLoopExit, EventLoopJoinError, JoinableEventLoop};
use crate::output::{
    AudioOutputDeathReason, AudioOutputEndpointShutdown, AudioOutputError, AudioOutputEventWatcher,
    AudioOutputStartFailure, AudioRenderOwner, EndpointShutdownConfirmed, RunningAudioOutput,
};

/// Logical proof that every external/control event producer for one context is quiescent.
///
/// Graceful event retirement is illegal without this proof. The controller-owned renderer may
/// still emit graph/drop records while it is reclaimed; that is why graceful event-loop stop is
/// requested only after renderer reclamation. The eventual context integration will construct
/// this marker after retiring every producer not owned by the controller.
#[derive(Debug)]
pub(crate) struct EventProducersQuiesced(());

impl EventProducersQuiesced {
    #[cfg(test)]
    const fn for_test() -> Self {
        Self(())
    }

    fn after_injected_graph_retired(
        _retired: &super::injected_node_lifetime::RetiredInjectedGraph,
    ) -> Self {
        Self(())
    }
}

/// Successful shutdown mode certified by [`OutputShutdownReceipt`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputShutdownMode {
    Graceful,
    Silent,
}

/// Stable category for a lifecycle degradation or unconfirmed shutdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputShutdownIssueKind {
    EndpointMethodPanicked,
    EndpointFuturePanicked,
    EndpointStateTransitionFailed,
    EndpointRejectedShutdown,
    CallbackRetained,
    RenderReclaimDegraded,
    EventDeliveryDegraded,
    EventThreadUnretired,
    WorkerPanicked,
    WorkerSpawnFailed,
    BootstrapFailed,
    InjectedStartPanicked,
    InjectedControlClose,
    InjectedNodeLifetime,
    InjectedWholeGraphQuarantined,
}

/// Clone-cheap lifecycle diagnostic retained by all receipt observers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutputShutdownIssue {
    kind: OutputShutdownIssueKind,
    message: Arc<str>,
}

impl OutputShutdownIssue {
    fn new(kind: OutputShutdownIssueKind, message: impl Into<Arc<str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) const fn kind(&self) -> OutputShutdownIssueKind {
        self.kind
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

/// Confirmed shutdown report. Optional issues describe cleanup degradation after callback
/// retirement was already authoritatively proven.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutputShutdownReport {
    mode: OutputShutdownMode,
    endpoint_death: Option<AudioOutputDeathReason>,
    reclaim_issue: Option<OutputShutdownIssue>,
    event_issue: Option<OutputShutdownIssue>,
}

impl OutputShutdownReport {
    pub(crate) const fn mode(&self) -> OutputShutdownMode {
        self.mode
    }

    pub(crate) const fn endpoint_death(&self) -> Option<AudioOutputDeathReason> {
        self.endpoint_death
    }

    pub(crate) const fn reclaim_issue(&self) -> Option<&OutputShutdownIssue> {
        self.reclaim_issue.as_ref()
    }

    pub(crate) const fn event_issue(&self) -> Option<&OutputShutdownIssue> {
        self.event_issue.as_ref()
    }
}

/// Receipt result for the private injected-output lifecycle foundation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OutputShutdownOutcome {
    /// Callback destruction and Arc retirement were proven. Degraded post-proof cleanup is
    /// described in the report.
    Confirmed(OutputShutdownReport),
    /// Callback destruction could not be proven. Ambiguous resources are quarantined for the
    /// process lifetime; this state never becomes successful through a timeout.
    Unconfirmed {
        failure: OutputShutdownIssue,
        event_issue: Option<OutputShutdownIssue>,
    },
    /// The worker-owned completion sender disappeared without publishing an outcome.
    ControllerTerminated,
}

type CompletionReceiver = oneshot::Receiver<OutputShutdownOutcome>;

/// Cloneable, cancellation-proof observer for injected-output retirement.
///
/// No current `AudioContext` method returns this type. The eventual contract completes only after
/// endpoint callback acknowledgement, Arc proof and renderer/GC reclamation, followed by event
/// thread stop and join. It excludes surviving node/base/receiver clones, queued payloads left by
/// silent stop, and factory-global mixer resources. Registered-waker storage is O(the maximum
/// number of concurrently live polled clones); a future Deno op must separately bound admitted
/// JavaScript waiters.
#[derive(Clone)]
pub(crate) struct OutputShutdownReceipt {
    shared: Shared<CompletionReceiver>,
}

impl fmt::Debug for OutputShutdownReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputShutdownReceipt")
            .finish_non_exhaustive()
    }
}

impl Future for OutputShutdownReceipt {
    type Output = OutputShutdownOutcome;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.shared).poll(cx) {
            Poll::Ready(Ok(outcome)) => Poll::Ready(outcome),
            Poll::Ready(Err(_)) => Poll::Ready(OutputShutdownOutcome::ControllerTerminated),
            Poll::Pending => Poll::Pending,
        }
    }
}

struct OutputShutdownCompleter(Option<oneshot::Sender<OutputShutdownOutcome>>);

impl OutputShutdownCompleter {
    fn complete(mut self, outcome: OutputShutdownOutcome) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(outcome);
        }
    }
}

fn shutdown_receipt_pair() -> (OutputShutdownCompleter, OutputShutdownReceipt) {
    let (sender, receiver) = oneshot::channel();
    (
        OutputShutdownCompleter(Some(sender)),
        OutputShutdownReceipt {
            shared: receiver.shared(),
        },
    )
}

/// Panic payload destruction is untrusted: `panic_any` accepts values whose destructor can panic.
/// Quarantining the payload avoids turning a contained infrastructure panic into worker death.
fn quarantine_panic_payload(payload: Box<dyn Any + Send + 'static>) {
    std::mem::forget(payload);
}

struct OutputLifecycleResources {
    endpoint: Box<dyn RunningAudioOutput>,
    render_owner: AudioRenderOwner,
    output_events: AudioOutputEventWatcher,
    event_loop: JoinableEventLoop,
}

enum LifecycleCommand {
    Graceful(EventProducersQuiesced),
    Silent,
}

trait LifecycleWorkerSpawner {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<JoinHandle<()>>;
}

struct ThreadSpawner;

impl LifecycleWorkerSpawner for ThreadSpawner {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<JoinHandle<()>> {
        thread::Builder::new()
            .name("web-audio-output-lifecycle".to_owned())
            .spawn(job)
    }
}

/// Start failure retaining every live resource that never crossed to the worker.
pub(crate) struct OutputLifecycleStartFailure {
    issue: OutputShutdownIssue,
    resources: Option<OutputLifecycleResources>,
    abandoned_worker: Option<JoinHandle<()>>,
}

impl fmt::Debug for OutputLifecycleStartFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputLifecycleStartFailure")
            .field("issue", &self.issue)
            .field("has_resources", &self.resources.is_some())
            .field("has_abandoned_worker", &self.abandoned_worker.is_some())
            .finish()
    }
}

impl OutputLifecycleStartFailure {
    /// Recovers resources that never crossed the bootstrap boundary.
    ///
    /// The optional handle belongs to an empty worker whose bootstrap receiver disappeared. A
    /// caller may join it off the handler thread or retain it for later joining; dropping it only
    /// detaches and does not acknowledge anything about the returned live resources.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        mut self,
    ) -> (
        OutputShutdownIssue,
        Box<dyn RunningAudioOutput>,
        AudioRenderOwner,
        AudioOutputEventWatcher,
        JoinableEventLoop,
        Option<JoinHandle<()>>,
    ) {
        let resources = self
            .resources
            .take()
            .expect("start failure retains untransferred lifecycle resources");
        (
            self.issue.clone(),
            resources.endpoint,
            resources.render_owner,
            resources.output_events,
            resources.event_loop,
            self.abandoned_worker.take(),
        )
    }
}

impl Drop for OutputLifecycleStartFailure {
    fn drop(&mut self) {
        if let Some(mut resources) = self.resources.take() {
            // The endpoint never reached a lifecycle worker, so its destructor is not trusted to
            // retire an installed callback. Close callback admission before quarantining it, then
            // let the existing owner/event guards take their fail-closed silent-detach paths.
            resources.render_owner.begin_shutdown();
            std::mem::forget(resources.endpoint);
            drop(resources.render_owner);
            resources.event_loop.request_silent_stop();
            drop(resources.output_events);
            drop(resources.event_loop);
        }
        self.abandoned_worker.take(); // detach; this empty worker owns no live resources
    }
}

/// Private controller that admits exactly one nonblocking shutdown request.
///
/// The worker is spawned empty before live resources are transferred. Dropping the controller
/// requests silent shutdown and detaches; it never claims acknowledgement.
pub(crate) struct OutputLifecycleController {
    command_send: Sender<LifecycleCommand>,
    receipt: OutputShutdownReceipt,
    worker: Option<JoinHandle<()>>,
    requested: bool,
}

impl fmt::Debug for OutputLifecycleController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputLifecycleController")
            .field("requested", &self.requested)
            .finish_non_exhaustive()
    }
}

impl OutputLifecycleController {
    pub(crate) fn start(
        endpoint: Box<dyn RunningAudioOutput>,
        render_owner: AudioRenderOwner,
        output_events: AudioOutputEventWatcher,
        event_loop: JoinableEventLoop,
    ) -> Result<Self, Box<OutputLifecycleStartFailure>> {
        Self::start_with_spawner(
            OutputLifecycleResources {
                endpoint,
                render_owner,
                output_events,
                event_loop,
            },
            &ThreadSpawner,
        )
    }

    fn start_with_spawner(
        resources: OutputLifecycleResources,
        spawner: &dyn LifecycleWorkerSpawner,
    ) -> Result<Self, Box<OutputLifecycleStartFailure>> {
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
        let (completer, receipt) = shutdown_receipt_pair();
        let job = Box::new(move || lifecycle_worker(bootstrap_recv, command_recv, completer));
        let worker = match spawner.spawn(job) {
            Ok(worker) => worker,
            Err(error) => {
                return Err(Box::new(OutputLifecycleStartFailure {
                    issue: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::WorkerSpawnFailed,
                        Arc::<str>::from(error.to_string()),
                    ),
                    resources: Some(resources),
                    abandoned_worker: None,
                }));
            }
        };

        if let Err(error) = bootstrap_send.send(resources) {
            return Err(Box::new(OutputLifecycleStartFailure {
                issue: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::BootstrapFailed,
                    "lifecycle worker rejected ownership bootstrap",
                ),
                resources: Some(error.0),
                abandoned_worker: Some(worker),
            }));
        }

        Ok(Self {
            command_send,
            receipt,
            worker: Some(worker),
            requested: false,
        })
    }

    pub(crate) fn receipt(&self) -> OutputShutdownReceipt {
        self.receipt.clone()
    }

    pub(crate) fn shutdown_gracefully(
        mut self,
        proof: EventProducersQuiesced,
    ) -> OutputShutdownReceipt {
        self.requested = true;
        let _ = self
            .command_send
            .try_send(LifecycleCommand::Graceful(proof));
        self.receipt.clone()
    }

    pub(crate) fn shutdown_silently(mut self) -> OutputShutdownReceipt {
        self.requested = true;
        let _ = self.command_send.try_send(LifecycleCommand::Silent);
        self.receipt.clone()
    }
}

impl Drop for OutputLifecycleController {
    fn drop(&mut self) {
        if !self.requested {
            let _ = self.command_send.try_send(LifecycleCommand::Silent);
        }
        // Dropping a JoinHandle detaches. Only the shared receipt can acknowledge cleanup.
        self.worker.take();
    }
}

struct OutputStartCleanupResources {
    endpoint_shutdown: AudioOutputEndpointShutdown,
    render_owner: AudioRenderOwner,
    output_events: AudioOutputEventWatcher,
    event_loop: JoinableEventLoop,
}

/// Handle for cleanup of an endpoint whose start partially committed before failing.
///
/// Cleanup begins immediately and always uses silent event retirement. Dropping this handle or
/// its receipt does not cancel cleanup; the worker remains the sole resource owner.
pub(crate) struct OutputStartCleanupHandle {
    startup_error: AudioOutputError,
    receipt: OutputShutdownReceipt,
    worker: Option<JoinHandle<()>>,
}

impl OutputStartCleanupHandle {
    pub(crate) fn start(
        start_failure: AudioOutputStartFailure,
        render_owner: AudioRenderOwner,
        output_events: AudioOutputEventWatcher,
        event_loop: JoinableEventLoop,
    ) -> Result<Self, Box<OutputStartCleanupBootstrapFailure>> {
        let (startup_error, endpoint_shutdown) = start_failure.into_parts();
        Self::start_with_spawner(
            startup_error,
            OutputStartCleanupResources {
                endpoint_shutdown,
                render_owner,
                output_events,
                event_loop,
            },
            &ThreadSpawner,
        )
    }

    fn start_with_spawner(
        startup_error: AudioOutputError,
        resources: OutputStartCleanupResources,
        spawner: &dyn LifecycleWorkerSpawner,
    ) -> Result<Self, Box<OutputStartCleanupBootstrapFailure>> {
        let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
        let (completer, receipt) = shutdown_receipt_pair();
        let job = Box::new(move || start_cleanup_worker(bootstrap_recv, completer));
        let worker = match spawner.spawn(job) {
            Ok(worker) => worker,
            Err(error) => {
                return Err(Box::new(OutputStartCleanupBootstrapFailure {
                    issue: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::WorkerSpawnFailed,
                        Arc::<str>::from(error.to_string()),
                    ),
                    startup_error: Some(startup_error),
                    resources: Some(resources),
                    abandoned_worker: None,
                }));
            }
        };

        if let Err(error) = bootstrap_send.send(resources) {
            return Err(Box::new(OutputStartCleanupBootstrapFailure {
                issue: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::BootstrapFailed,
                    "start-cleanup worker rejected ownership bootstrap",
                ),
                startup_error: Some(startup_error),
                resources: Some(error.0),
                abandoned_worker: Some(worker),
            }));
        }

        Ok(Self {
            startup_error,
            receipt,
            worker: Some(worker),
        })
    }

    pub(crate) const fn startup_error(&self) -> &AudioOutputError {
        &self.startup_error
    }

    pub(crate) fn receipt(&self) -> OutputShutdownReceipt {
        self.receipt.clone()
    }
}

impl Drop for OutputStartCleanupHandle {
    fn drop(&mut self) {
        // Dropping a JoinHandle detaches. Completion remains worker-owned and observable through
        // any surviving receipt clone.
        self.worker.take();
    }
}

/// Bootstrap failure retaining the complete, still-untransferred partial-start state.
pub(crate) struct OutputStartCleanupBootstrapFailure {
    issue: OutputShutdownIssue,
    startup_error: Option<AudioOutputError>,
    resources: Option<OutputStartCleanupResources>,
    abandoned_worker: Option<JoinHandle<()>>,
}

impl OutputStartCleanupBootstrapFailure {
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        mut self,
    ) -> (
        OutputShutdownIssue,
        AudioOutputStartFailure,
        AudioRenderOwner,
        AudioOutputEventWatcher,
        JoinableEventLoop,
        Option<JoinHandle<()>>,
    ) {
        let resources = self
            .resources
            .take()
            .expect("bootstrap failure retains untransferred partial-start resources");
        (
            self.issue.clone(),
            AudioOutputStartFailure::new(
                self.startup_error
                    .take()
                    .expect("bootstrap failure retains original startup error"),
                resources.endpoint_shutdown,
            ),
            resources.render_owner,
            resources.output_events,
            resources.event_loop,
            self.abandoned_worker.take(),
        )
    }
}

impl Drop for OutputStartCleanupBootstrapFailure {
    fn drop(&mut self) {
        if let Some(mut resources) = self.resources.take() {
            resources.render_owner.begin_shutdown();
            std::mem::forget(resources.endpoint_shutdown);
            drop(resources.render_owner); // fail-closed leak
            resources.event_loop.request_silent_stop();
            drop(resources.event_loop); // detach; no acknowledgement
            drop(resources.output_events);
        }
        self.abandoned_worker.take();
    }
}

fn start_cleanup_worker(
    bootstrap_recv: Receiver<OutputStartCleanupResources>,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap_recv.recv() {
        Ok(resources) => {
            panic::catch_unwind(AssertUnwindSafe(|| run_start_cleanup_worker(resources)))
                .unwrap_or_else(|payload| {
                    quarantine_panic_payload(payload);
                    OutputShutdownOutcome::Unconfirmed {
                        failure: OutputShutdownIssue::new(
                            OutputShutdownIssueKind::WorkerPanicked,
                            "partial-start cleanup worker panicked unexpectedly",
                        ),
                        event_issue: Some(OutputShutdownIssue::new(
                            OutputShutdownIssueKind::EventThreadUnretired,
                            "worker panic left event-thread retirement unconfirmed",
                        )),
                    }
                })
        }
        Err(_) => OutputShutdownOutcome::ControllerTerminated,
    };
    completer.complete(outcome);
}

fn run_start_cleanup_worker(mut resources: OutputStartCleanupResources) -> OutputShutdownOutcome {
    // A partial start is never publicly observable as a live context. Silence event dispatch
    // before waiting for endpoint acknowledgement, even if that future never resolves.
    resources.event_loop.request_silent_stop();
    resources.render_owner.begin_shutdown();
    finish_endpoint_shutdown(
        resources.endpoint_shutdown,
        resources.render_owner,
        resources.output_events,
        resources.event_loop,
        LifecycleRequest {
            mode: OutputShutdownMode::Silent,
            endpoint_death: None,
        },
    )
}

fn lifecycle_worker(
    bootstrap_recv: Receiver<OutputLifecycleResources>,
    command_recv: Receiver<LifecycleCommand>,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap_recv.recv() {
        Ok(resources) => panic::catch_unwind(AssertUnwindSafe(|| {
            run_lifecycle_worker(resources, command_recv)
        }))
        .unwrap_or_else(|payload| {
            quarantine_panic_payload(payload);
            OutputShutdownOutcome::Unconfirmed {
                failure: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::WorkerPanicked,
                    "output lifecycle worker panicked unexpectedly",
                ),
                event_issue: Some(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventThreadUnretired,
                    "worker panic left event-thread retirement unconfirmed",
                )),
            }
        }),
        Err(_) => OutputShutdownOutcome::ControllerTerminated,
    };
    completer.complete(outcome);
}

struct LifecycleRequest {
    mode: OutputShutdownMode,
    endpoint_death: Option<AudioOutputDeathReason>,
}

fn wait_for_shutdown_request(
    commands: &Receiver<LifecycleCommand>,
    output_events: &AudioOutputEventWatcher,
) -> LifecycleRequest {
    let mut diagnostics_connected = true;
    loop {
        if let Some(reason) = output_events.death_reason() {
            return LifecycleRequest {
                mode: OutputShutdownMode::Silent,
                endpoint_death: Some(reason),
            };
        }

        if !diagnostics_connected {
            return command_to_request(commands.recv().ok(), output_events);
        }

        crossbeam_channel::select_biased! {
            recv(commands) -> command => {
                return command_to_request(command.ok(), output_events);
            }
            recv(output_events.receiver()) -> diagnostic => {
                if diagnostic.is_err() {
                    diagnostics_connected = false;
                }
            }
        }
    }
}

fn command_to_request(
    command: Option<LifecycleCommand>,
    output_events: &AudioOutputEventWatcher,
) -> LifecycleRequest {
    let endpoint_death = output_events.death_reason();
    let requested = match command {
        Some(LifecycleCommand::Graceful(_proof)) if endpoint_death.is_none() => {
            OutputShutdownMode::Graceful
        }
        Some(LifecycleCommand::Graceful(_) | LifecycleCommand::Silent) | None => {
            OutputShutdownMode::Silent
        }
    };
    LifecycleRequest {
        mode: requested,
        endpoint_death,
    }
}

fn run_lifecycle_worker(
    resources: OutputLifecycleResources,
    commands: Receiver<LifecycleCommand>,
) -> OutputShutdownOutcome {
    let OutputLifecycleResources {
        endpoint,
        render_owner,
        output_events,
        mut event_loop,
    } = resources;
    let request = wait_for_shutdown_request(&commands, &output_events);
    render_owner.begin_shutdown();

    if request.mode == OutputShutdownMode::Silent {
        event_loop.request_silent_stop();
    }

    let endpoint_future = match panic::catch_unwind(AssertUnwindSafe(|| endpoint.shutdown())) {
        Ok(future) => future,
        Err(payload) => {
            quarantine_panic_payload(payload);
            drop(render_owner); // fail-closed leak
            let event_issue = retire_events(event_loop, OutputShutdownMode::Silent).issue;
            return OutputShutdownOutcome::Unconfirmed {
                failure: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EndpointMethodPanicked,
                    "endpoint shutdown method panicked instead of reporting an error",
                ),
                event_issue,
            };
        }
    };

    finish_endpoint_shutdown(
        endpoint_future,
        render_owner,
        output_events,
        event_loop,
        request,
    )
}

fn finish_endpoint_shutdown(
    endpoint_future: AudioOutputEndpointShutdown,
    render_owner: AudioRenderOwner,
    output_events: AudioOutputEventWatcher,
    mut event_loop: JoinableEventLoop,
    mut request: LifecycleRequest,
) -> OutputShutdownOutcome {
    if let EndpointPollOutcome::Quarantined { future, failure } = poll_endpoint_shutdown(
        endpoint_future,
        &output_events,
        &mut request,
        &mut event_loop,
        true,
    ) {
        std::mem::forget(future);
        event_loop.request_silent_stop();
        drop(render_owner); // fail-closed leak
        let event_issue = retire_events(event_loop, OutputShutdownMode::Silent).issue;
        return OutputShutdownOutcome::Unconfirmed {
            failure,
            event_issue,
        };
    }

    promote_death_to_silent(&output_events, &mut request, &mut event_loop);

    let reclaim = panic::catch_unwind(AssertUnwindSafe(|| {
        render_owner.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
    }));
    let reclaim_issue = match reclaim {
        Err(payload) => {
            quarantine_panic_payload(payload);
            event_loop.request_silent_stop();
            let event_issue = retire_events(event_loop, OutputShutdownMode::Silent).issue;
            return OutputShutdownOutcome::Unconfirmed {
                failure: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::CallbackRetained,
                    "render ownership panicked before callback retirement could be proven",
                ),
                event_issue,
            };
        }
        Ok(Err(owner)) => {
            drop(owner); // existing fail-closed quarantine
            event_loop.request_silent_stop();
            let event_issue = retire_events(event_loop, OutputShutdownMode::Silent).issue;
            return OutputShutdownOutcome::Unconfirmed {
                failure: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::CallbackRetained,
                    "endpoint reported success while retaining its render callback",
                ),
                event_issue,
            };
        }
        Ok(Ok(Ok(()))) => None,
        Ok(Ok(Err(error))) => {
            request.mode = OutputShutdownMode::Silent;
            event_loop.request_silent_stop();
            Some(issue_from_audio_error(
                OutputShutdownIssueKind::RenderReclaimDegraded,
                error,
            ))
        }
    };

    promote_death_to_silent(&output_events, &mut request, &mut event_loop);

    let event_retirement = retire_events(event_loop, request.mode);
    if !event_retirement.retired {
        return OutputShutdownOutcome::Unconfirmed {
            failure: OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventThreadUnretired,
                "lifecycle worker refused to join its own event thread",
            ),
            event_issue: event_retirement.issue,
        };
    }

    OutputShutdownOutcome::Confirmed(OutputShutdownReport {
        mode: request.mode,
        endpoint_death: request.endpoint_death,
        reclaim_issue,
        event_issue: event_retirement.issue,
    })
}

fn promote_death_to_silent(
    output_events: &AudioOutputEventWatcher,
    request: &mut LifecycleRequest,
    event_loop: &mut JoinableEventLoop,
) {
    request.endpoint_death = request
        .endpoint_death
        .or_else(|| output_events.death_reason());
    if request.endpoint_death.is_some() {
        request.mode = OutputShutdownMode::Silent;
        event_loop.request_silent_stop();
    }
}

fn issue_from_audio_error(
    kind: OutputShutdownIssueKind,
    error: AudioOutputError,
) -> OutputShutdownIssue {
    OutputShutdownIssue::new(kind, Arc::<str>::from(error.message()))
}

struct ChannelWake(Sender<()>);

impl Wake for ChannelWake {
    fn wake(self: Arc<Self>) {
        let _ = self.0.try_send(());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}

enum EndpointPollOutcome {
    Confirmed,
    Quarantined {
        future: AudioOutputEndpointShutdown,
        failure: OutputShutdownIssue,
    },
}

struct EndpointShutdownQuarantine(Option<AudioOutputEndpointShutdown>);

impl Drop for EndpointShutdownQuarantine {
    fn drop(&mut self) {
        if let Some(future) = self.0.take() {
            std::mem::forget(future);
        }
    }
}

fn poll_endpoint_shutdown(
    future: AudioOutputEndpointShutdown,
    output_events: &AudioOutputEventWatcher,
    request: &mut LifecycleRequest,
    event_loop: &mut JoinableEventLoop,
    stop_events_on_death: bool,
) -> EndpointPollOutcome {
    poll_endpoint_shutdown_inner(
        future,
        output_events,
        request,
        Some(event_loop),
        stop_events_on_death,
    )
}

fn poll_injected_endpoint_shutdown(
    future: AudioOutputEndpointShutdown,
    output_events: &AudioOutputEventWatcher,
    request: &mut LifecycleRequest,
) -> EndpointPollOutcome {
    poll_endpoint_shutdown_inner(future, output_events, request, None, false)
}

fn poll_endpoint_shutdown_inner(
    future: AudioOutputEndpointShutdown,
    output_events: &AudioOutputEventWatcher,
    request: &mut LifecycleRequest,
    mut event_loop: Option<&mut JoinableEventLoop>,
    stop_events_on_death: bool,
) -> EndpointPollOutcome {
    let mut future = EndpointShutdownQuarantine(Some(future));
    let (wake_send, wake_recv) = crossbeam_channel::bounded(1);
    let waker = Waker::from(Arc::new(ChannelWake(wake_send)));
    let mut context = Context::from_waker(&waker);
    let mut diagnostics_connected = true;
    loop {
        // The atomic latch is authoritative. Rechecking before every poll prevents a self-waking
        // future from starving a queued or dropped best-effort diagnostic.
        promote_death_to_silent_for_poll(
            output_events,
            request,
            event_loop.as_deref_mut(),
            stop_events_on_death,
        );
        let polled = panic::catch_unwind(AssertUnwindSafe(|| {
            Pin::new(future.0.as_mut().unwrap()).poll(&mut context)
        }));
        match polled {
            Ok(Poll::Ready(Ok(()))) => {
                drop(future.0.take());
                return EndpointPollOutcome::Confirmed;
            }
            Ok(Poll::Ready(Err(error))) => {
                let issue = issue_from_audio_error(
                    OutputShutdownIssueKind::EndpointRejectedShutdown,
                    error,
                );
                return EndpointPollOutcome::Quarantined {
                    future: future.0.take().unwrap(),
                    failure: issue,
                };
            }
            Ok(Poll::Pending) => {
                promote_death_to_silent_for_poll(
                    output_events,
                    request,
                    event_loop.as_deref_mut(),
                    stop_events_on_death,
                );
                if diagnostics_connected {
                    crossbeam_channel::select_biased! {
                        recv(wake_recv) -> _ => {}
                        recv(output_events.receiver()) -> diagnostic => {
                            diagnostics_connected = diagnostic.is_ok();
                            promote_death_to_silent_for_poll(
                                output_events,
                                request,
                                event_loop.as_deref_mut(),
                                stop_events_on_death,
                            );
                        }
                    }
                } else {
                    // Once every diagnostic sender is gone, the death latch cannot change. The
                    // endpoint future's capacity-one wake channel remains the only useful wait.
                    let _ = wake_recv.recv();
                }
            }
            Err(payload) => {
                // The future may own callback/host leases and its destructor is ambiguous after a
                // poll panic. Quarantine it before touching the equally untrusted panic payload.
                quarantine_panic_payload(payload);
                return EndpointPollOutcome::Quarantined {
                    future: future.0.take().unwrap(),
                    failure: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::EndpointFuturePanicked,
                        "endpoint shutdown future panicked while being polled",
                    ),
                };
            }
        }
    }
}

fn promote_death_to_silent_for_poll(
    output_events: &AudioOutputEventWatcher,
    request: &mut LifecycleRequest,
    event_loop: Option<&mut JoinableEventLoop>,
    stop_events_on_death: bool,
) {
    if stop_events_on_death {
        promote_death_to_silent(
            output_events,
            request,
            event_loop.expect("legacy death-stop polling retains its event-loop authority"),
        );
    } else {
        request.endpoint_death = request
            .endpoint_death
            .or_else(|| output_events.death_reason());
        if request.endpoint_death.is_some() {
            request.mode = OutputShutdownMode::Silent;
        }
    }
}

struct EventRetirement {
    retired: bool,
    issue: Option<OutputShutdownIssue>,
}

fn retire_events(mut event_loop: JoinableEventLoop, mode: OutputShutdownMode) -> EventRetirement {
    match mode {
        OutputShutdownMode::Graceful => event_loop.request_graceful_stop(),
        OutputShutdownMode::Silent => event_loop.request_silent_stop(),
    }

    let expected = match mode {
        OutputShutdownMode::Graceful => EventLoopExit::Graceful,
        OutputShutdownMode::Silent => EventLoopExit::Silent,
    };
    match panic::catch_unwind(AssertUnwindSafe(|| event_loop.join())) {
        Ok(result) => event_retirement_from_join(result, expected),
        Err(payload) => {
            quarantine_panic_payload(payload);
            EventRetirement {
                retired: false,
                issue: Some(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventThreadUnretired,
                    "event-loop join panicked outside its typed retirement result",
                )),
            }
        }
    }
}

fn event_retirement_from_join(
    result: Result<EventLoopExit, EventLoopJoinError>,
    expected: EventLoopExit,
) -> EventRetirement {
    match result {
        Ok(exit) => EventRetirement {
            retired: true,
            issue: (exit != expected).then(|| {
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventDeliveryDegraded,
                    "event loop exited in an unexpected stop mode",
                )
            }),
        },
        Err(EventLoopJoinError::CurrentThread) => EventRetirement {
            retired: false,
            issue: None,
        },
        Err(EventLoopJoinError::StopChannelDisconnected) => EventRetirement {
            retired: true,
            issue: Some(OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventDeliveryDegraded,
                "event stop authority disconnected before selecting a mode",
            )),
        },
        Err(EventLoopJoinError::Panicked(payload)) => {
            quarantine_panic_payload(payload);
            EventRetirement {
                retired: true,
                issue: Some(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventDeliveryDegraded,
                    "event handler panicked before event-thread retirement",
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use futures::executor;
    use futures_util::task::noop_waker;

    use super::*;
    use crate::context::AudioNodeId;
    use crate::events::{EventDispatch, EventHandler, EventLoop, EventType};
    use crate::output::{
        audio_render_test_pair, AudioOutputErrorKind, AudioOutputEventSink, AudioRenderCallback,
        AudioRenderFormat, AudioRenderStatus,
    };

    const TIMEOUT: Duration = Duration::from_secs(2);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Step {
        EndpointShutdown,
        CallbackDestroyed,
        Reclaim,
        Sentinel,
        Closed,
    }

    enum EndpointBehavior {
        ReadyOk,
        ReadyErr(Arc<AtomicBool>),
        PollPanic(Arc<AtomicBool>),
        HostilePollPanic(Arc<AtomicBool>),
        Pending(oneshot::Receiver<()>),
        Retain(Arc<Mutex<Option<AudioRenderCallback>>>),
        RetainedBeforeShutdown(Arc<Mutex<Option<AudioRenderCallback>>>),
        MethodPanic,
    }

    struct TestEndpoint {
        callback: Option<AudioRenderCallback>,
        behavior: EndpointBehavior,
        log: Arc<Mutex<Vec<Step>>>,
        lifecycle_thread: Arc<Mutex<Option<thread::ThreadId>>>,
    }

    impl RunningAudioOutput for TestEndpoint {
        fn resume(&mut self) -> Result<(), AudioOutputError> {
            Ok(())
        }

        fn suspend(&mut self) -> Result<(), AudioOutputError> {
            Ok(())
        }

        fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
            *self.lifecycle_thread.lock().unwrap() = Some(thread::current().id());
            self.log.lock().unwrap().push(Step::EndpointShutdown);
            match self.behavior {
                EndpointBehavior::ReadyOk => {
                    drop(self.callback.take());
                    self.log.lock().unwrap().push(Step::CallbackDestroyed);
                    AudioOutputEndpointShutdown::ready(Ok(()))
                }
                EndpointBehavior::ReadyErr(ref drop_probe) => {
                    AudioOutputEndpointShutdown::from_future(UnconfirmedFuture {
                        callback: self.callback.take(),
                        drop_probe: Arc::clone(drop_probe),
                        outcome: UnconfirmedPoll::Error,
                    })
                }
                EndpointBehavior::PollPanic(ref drop_probe) => {
                    AudioOutputEndpointShutdown::from_future(UnconfirmedFuture {
                        callback: self.callback.take(),
                        drop_probe: Arc::clone(drop_probe),
                        outcome: UnconfirmedPoll::Panic,
                    })
                }
                EndpointBehavior::HostilePollPanic(ref drop_probe) => {
                    AudioOutputEndpointShutdown::from_future(UnconfirmedFuture {
                        callback: self.callback.take(),
                        drop_probe: Arc::clone(drop_probe),
                        outcome: UnconfirmedPoll::HostilePanic,
                    })
                }
                EndpointBehavior::Pending(ref mut release) => {
                    let release = std::mem::replace(release, oneshot::channel().1);
                    let callback = self.callback.take();
                    let log = Arc::clone(&self.log);
                    AudioOutputEndpointShutdown::from_future(async move {
                        release.await.map_err(|_| {
                            AudioOutputError::new(
                                AudioOutputErrorKind::Shutdown,
                                "pending endpoint release sender disappeared",
                            )
                        })?;
                        drop(callback);
                        log.lock().unwrap().push(Step::CallbackDestroyed);
                        Ok(())
                    })
                }
                EndpointBehavior::Retain(ref retained) => {
                    *retained.lock().unwrap() = self.callback.take();
                    AudioOutputEndpointShutdown::ready(Ok(()))
                }
                EndpointBehavior::RetainedBeforeShutdown(_) => {
                    AudioOutputEndpointShutdown::ready(Ok(()))
                }
                EndpointBehavior::MethodPanic => {
                    panic!("injected endpoint shutdown method panic");
                }
            }
        }
    }

    struct UnconfirmedFuture {
        callback: Option<AudioRenderCallback>,
        drop_probe: Arc<AtomicBool>,
        outcome: UnconfirmedPoll,
    }

    #[derive(Clone, Copy)]
    enum UnconfirmedPoll {
        Error,
        Panic,
        HostilePanic,
    }

    struct HostilePanicPayload;

    impl Drop for HostilePanicPayload {
        fn drop(&mut self) {
            panic!("hostile panic payload destructor");
        }
    }

    impl Future for UnconfirmedFuture {
        type Output = Result<(), AudioOutputError>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            let _ = &self.callback;
            match self.outcome {
                UnconfirmedPoll::Panic => panic!("injected endpoint future poll panic"),
                UnconfirmedPoll::HostilePanic => panic::panic_any(HostilePanicPayload),
                UnconfirmedPoll::Error => {}
            }
            Poll::Ready(Err(AudioOutputError::new(
                AudioOutputErrorKind::Shutdown,
                "injected endpoint refused shutdown",
            )))
        }
    }

    impl Drop for UnconfirmedFuture {
        fn drop(&mut self) {
            self.drop_probe.store(true, Ordering::Release);
        }
    }

    struct Fixture {
        resources: Option<OutputLifecycleResources>,
        output_sink: AudioOutputEventSink,
        event_send: Sender<EventDispatch>,
        log: Arc<Mutex<Vec<Step>>>,
        reclaim_ran: Arc<AtomicBool>,
        lifecycle_thread: Arc<Mutex<Option<thread::ThreadId>>>,
        reclaim_thread: Arc<Mutex<Option<thread::ThreadId>>>,
        event_thread: Arc<Mutex<Option<thread::ThreadId>>>,
        event_exit: Receiver<()>,
    }

    struct EventExitProbe(Option<Sender<()>>);

    impl Drop for EventExitProbe {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    fn fixture(
        behavior: EndpointBehavior,
        reclaim_error: bool,
        panic_closed_handler: bool,
        enqueue_sentinel_during_reclaim: bool,
    ) -> Fixture {
        let log = Arc::new(Mutex::new(Vec::new()));
        let lifecycle_thread = Arc::new(Mutex::new(None));
        let reclaim_thread = Arc::new(Mutex::new(None));
        let event_thread = Arc::new(Mutex::new(None));
        let reclaim_ran = Arc::new(AtomicBool::new(false));
        let (output_sink, output_events) = AudioOutputEventSink::bounded(4);
        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();

        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let (event_exit_send, event_exit) = crossbeam_channel::bounded(1);
        let event_loop = EventLoop::new(event_recv);
        let event_exit_probe = EventExitProbe(Some(event_exit_send));
        event_loop.set_handler(
            EventType::Ended(AudioNodeId(777)),
            EventHandler::Once(Box::new(move |_| {
                let _ = &event_exit_probe;
            })),
        );
        let sentinel_log = Arc::clone(&log);
        let sentinel_thread = Arc::clone(&event_thread);
        event_loop.set_handler(
            EventType::Ended(AudioNodeId(42)),
            EventHandler::Once(Box::new(move |_| {
                *sentinel_thread.lock().unwrap() = Some(thread::current().id());
                sentinel_log.lock().unwrap().push(Step::Sentinel);
            })),
        );
        let close_log = Arc::clone(&log);
        let close_thread = Arc::clone(&event_thread);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                *close_thread.lock().unwrap() = Some(thread::current().id());
                if panic_closed_handler {
                    panic!("injected final close handler panic");
                }
                close_log.lock().unwrap().push(Step::Closed);
            })),
        );
        let joinable_events = event_loop.run_joinable().unwrap();

        let reclaim_log = Arc::clone(&log);
        let reclaim_probe = Arc::clone(&reclaim_ran);
        let reclaim_thread_probe = Arc::clone(&reclaim_thread);
        let sentinel_send = event_send.clone();
        let (render_owner, callback) = audio_render_test_pair(
            format,
            output_sink.clone(),
            |_| {},
            move || {
                *reclaim_thread_probe.lock().unwrap() = Some(thread::current().id());
                reclaim_probe.store(true, Ordering::Release);
                reclaim_log.lock().unwrap().push(Step::Reclaim);
                if enqueue_sentinel_during_reclaim {
                    sentinel_send
                        .send(EventDispatch::ended(AudioNodeId(42)))
                        .unwrap();
                }
                if reclaim_error {
                    Err(AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected reclaim degradation",
                    ))
                } else {
                    Ok(())
                }
            },
        );
        let callback = match &behavior {
            EndpointBehavior::RetainedBeforeShutdown(retained) => {
                *retained.lock().unwrap() = Some(callback);
                None
            }
            _ => Some(callback),
        };
        let endpoint = Box::new(TestEndpoint {
            callback,
            behavior,
            log: Arc::clone(&log),
            lifecycle_thread: Arc::clone(&lifecycle_thread),
        });

        Fixture {
            resources: Some(OutputLifecycleResources {
                endpoint,
                render_owner,
                output_events,
                event_loop: joinable_events,
            }),
            output_sink,
            event_send,
            log,
            reclaim_ran,
            lifecycle_thread,
            reclaim_thread,
            event_thread,
            event_exit,
        }
    }

    fn start(fixture: &mut Fixture) -> OutputLifecycleController {
        let resources = fixture.resources.take().unwrap();
        OutputLifecycleController::start_with_spawner(resources, &ThreadSpawner).unwrap()
    }

    fn confirmed(outcome: OutputShutdownOutcome) -> OutputShutdownReport {
        match outcome {
            OutputShutdownOutcome::Confirmed(report) => report,
            other => panic!("expected confirmed shutdown, got {other:?}"),
        }
    }

    fn unconfirmed(outcome: OutputShutdownOutcome) -> OutputShutdownIssue {
        match outcome {
            OutputShutdownOutcome::Unconfirmed { failure, .. } => failure,
            other => panic!("expected unconfirmed shutdown, got {other:?}"),
        }
    }

    #[test]
    fn graceful_shutdown_orders_endpoint_reclaim_events_and_shared_receipts() {
        let caller = thread::current().id();
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, true);
        let controller = start(&mut fixture);
        let receipt_clone = controller.receipt();
        let receipt = controller.shutdown_gracefully(EventProducersQuiesced::for_test());
        let waiter = thread::spawn(move || executor::block_on(receipt_clone));
        let report = confirmed(executor::block_on(receipt));
        assert_eq!(
            waiter.join().unwrap(),
            OutputShutdownOutcome::Confirmed(report.clone())
        );

        assert_eq!(report.mode(), OutputShutdownMode::Graceful);
        assert_eq!(report.reclaim_issue(), None);
        assert_eq!(report.event_issue(), None);
        assert_eq!(
            *fixture.log.lock().unwrap(),
            [
                Step::EndpointShutdown,
                Step::CallbackDestroyed,
                Step::Reclaim,
                Step::Sentinel,
                Step::Closed,
            ]
        );
        let lifecycle = fixture.lifecycle_thread.lock().unwrap().unwrap();
        let reclaim = fixture.reclaim_thread.lock().unwrap().unwrap();
        let event = fixture.event_thread.lock().unwrap().unwrap();
        assert_eq!(lifecycle, reclaim);
        assert_ne!(lifecycle, caller);
        assert_ne!(event, caller);
        assert_ne!(event, lifecycle);
    }

    #[test]
    fn silent_shutdown_dispatches_no_final_closed() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let receipt = start(&mut fixture).shutdown_silently();
        let report = confirmed(executor::block_on(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn dropping_controller_and_all_receipts_does_not_cancel_cleanup() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let (done_send, done_recv) = crossbeam_channel::bounded(1);
        let controller = OutputLifecycleController::start_with_spawner(
            fixture.resources.take().unwrap(),
            &CompletionProbeSpawner(done_send),
        )
        .unwrap();
        drop(controller.receipt());
        drop(controller);

        done_recv.recv_timeout(TIMEOUT).unwrap();
        assert!(fixture.reclaim_ran.load(Ordering::Acquire));
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn dropped_completion_sender_maps_to_controller_terminated() {
        let (completer, receipt) = shutdown_receipt_pair();
        drop(completer);
        assert_eq!(
            executor::block_on(receipt),
            OutputShutdownOutcome::ControllerTerminated
        );
    }

    #[test]
    fn receipt_traits_and_late_clone_replay_completed_outcome() {
        fn assert_traits<T: Send + Sync + Unpin>() {}
        assert_traits::<OutputShutdownReceipt>();

        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let receipt = start(&mut fixture).shutdown_silently();
        let outcome = executor::block_on(receipt.clone());
        assert_eq!(executor::block_on(receipt.clone()), outcome);
    }

    #[test]
    fn endpoint_error_quarantines_future_and_owner_then_silently_joins_events() {
        let future_dropped = Arc::new(AtomicBool::new(false));
        let mut fixture = fixture(
            EndpointBehavior::ReadyErr(Arc::clone(&future_dropped)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(start(&mut fixture).shutdown_silently()));
        assert_eq!(
            issue.kind(),
            OutputShutdownIssueKind::EndpointRejectedShutdown
        );
        assert!(!future_dropped.load(Ordering::Acquire));
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn endpoint_poll_panic_quarantines_future_without_running_its_drop() {
        let future_dropped = Arc::new(AtomicBool::new(false));
        let mut fixture = fixture(
            EndpointBehavior::PollPanic(Arc::clone(&future_dropped)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(start(&mut fixture).shutdown_silently()));
        assert_eq!(
            issue.kind(),
            OutputShutdownIssueKind::EndpointFuturePanicked
        );
        assert!(!future_dropped.load(Ordering::Acquire));
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn hostile_poll_panic_payload_cannot_terminate_controller() {
        let future_dropped = Arc::new(AtomicBool::new(false));
        let mut fixture = fixture(
            EndpointBehavior::HostilePollPanic(Arc::clone(&future_dropped)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(start(&mut fixture).shutdown_silently()));
        assert_eq!(
            issue.kind(),
            OutputShutdownIssueKind::EndpointFuturePanicked
        );
        assert!(!future_dropped.load(Ordering::Acquire));
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn endpoint_method_panic_is_unconfirmed_and_events_still_join_silently() {
        let mut fixture = fixture(EndpointBehavior::MethodPanic, false, false, false);
        let issue = unconfirmed(executor::block_on(start(&mut fixture).shutdown_silently()));
        assert_eq!(
            issue.kind(),
            OutputShutdownIssueKind::EndpointMethodPanicked
        );
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn pending_endpoint_keeps_receipt_and_resources_live_until_woken() {
        let (release_send, release_recv) = oneshot::channel();
        let mut fixture = fixture(EndpointBehavior::Pending(release_recv), false, false, false);
        let mut receipt = Box::pin(start(&mut fixture).shutdown_silently());
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(receipt.as_mut().poll(&mut context).is_pending());
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
        fixture.event_exit.recv_timeout(TIMEOUT).unwrap();
        assert!(fixture
            .event_send
            .send(EventDispatch::ended(AudioNodeId(42)))
            .is_err());
        assert!(!fixture.log.lock().unwrap().contains(&Step::Sentinel));
        release_send.send(()).unwrap();
        let report = confirmed(executor::block_on(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert!(fixture.reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn late_endpoint_death_silences_events_while_graceful_future_is_pending() {
        let (release_send, release_recv) = oneshot::channel();
        let mut fixture = fixture(EndpointBehavior::Pending(release_recv), false, false, false);
        let receipt = start(&mut fixture).shutdown_gracefully(EventProducersQuiesced::for_test());

        while !fixture
            .log
            .lock()
            .unwrap()
            .contains(&Step::EndpointShutdown)
        {
            thread::yield_now();
        }
        assert!(fixture
            .output_sink
            .report_endpoint_death(AudioOutputDeathReason::DeviceUnavailable));
        fixture.event_exit.recv_timeout(TIMEOUT).unwrap();
        release_send.send(()).unwrap();

        let report = confirmed(executor::block_on(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert_eq!(
            report.endpoint_death(),
            Some(AudioOutputDeathReason::DeviceUnavailable)
        );
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn false_endpoint_ok_with_retained_callback_is_unconfirmed() {
        let retained = Arc::new(Mutex::new(None));
        let mut fixture = fixture(
            EndpointBehavior::Retain(Arc::clone(&retained)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(start(&mut fixture).shutdown_silently()));
        assert_eq!(issue.kind(), OutputShutdownIssueKind::CallbackRetained);
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
        assert!(retained.lock().unwrap().take().is_some());
    }

    #[test]
    fn reclaim_error_is_confirmed_degraded_and_forces_silent_events() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, true, false, false);
        let report = confirmed(executor::block_on(
            start(&mut fixture).shutdown_gracefully(EventProducersQuiesced::for_test()),
        ));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert_eq!(
            report.reclaim_issue().unwrap().kind(),
            OutputShutdownIssueKind::RenderReclaimDegraded
        );
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn event_handler_panic_is_confirmed_degraded_after_thread_join() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, true, false);
        let report = confirmed(executor::block_on(
            start(&mut fixture).shutdown_gracefully(EventProducersQuiesced::for_test()),
        ));
        assert_eq!(report.mode(), OutputShutdownMode::Graceful);
        assert_eq!(
            report.event_issue().unwrap().kind(),
            OutputShutdownIssueKind::EventDeliveryDegraded
        );
    }

    #[test]
    fn endpoint_death_latch_dominates_graceful_and_triggers_silent_cleanup() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let controller = start(&mut fixture);
        let receipt = controller.receipt();
        assert!(fixture
            .output_sink
            .report_endpoint_death(AudioOutputDeathReason::DeviceUnavailable));
        let report = confirmed(executor::block_on(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert_eq!(
            report.endpoint_death(),
            Some(AudioOutputDeathReason::DeviceUnavailable)
        );
        drop(controller);
    }

    fn partial_start_resources(
        fixture: &mut Fixture,
    ) -> (AudioOutputError, OutputStartCleanupResources) {
        let OutputLifecycleResources {
            endpoint,
            render_owner,
            output_events,
            event_loop,
        } = fixture.resources.take().unwrap();
        let shutdown = endpoint.shutdown();
        (
            AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "injected partial-start failure",
            ),
            OutputStartCleanupResources {
                endpoint_shutdown: shutdown,
                render_owner,
                output_events,
                event_loop,
            },
        )
    }

    fn start_partial_cleanup(fixture: &mut Fixture) -> OutputStartCleanupHandle {
        let (startup_error, resources) = partial_start_resources(fixture);
        match OutputStartCleanupHandle::start_with_spawner(startup_error, resources, &ThreadSpawner)
        {
            Ok(handle) => handle,
            Err(_) => panic!("partial-start cleanup worker should start"),
        }
    }

    #[test]
    fn partial_start_success_reclaims_and_reports_original_error() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let handle = start_partial_cleanup(&mut fixture);
        assert_eq!(
            handle.startup_error().message(),
            "injected partial-start failure"
        );
        let report = confirmed(executor::block_on(handle.receipt()));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert!(fixture.reclaim_ran.load(Ordering::Acquire));
        assert!(!fixture.log.lock().unwrap().contains(&Step::Closed));
    }

    #[test]
    fn partial_start_endpoint_error_is_quarantined() {
        let future_dropped = Arc::new(AtomicBool::new(false));
        let mut fixture = fixture(
            EndpointBehavior::ReadyErr(Arc::clone(&future_dropped)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(
            start_partial_cleanup(&mut fixture).receipt(),
        ));
        assert_eq!(
            issue.kind(),
            OutputShutdownIssueKind::EndpointRejectedShutdown
        );
        assert!(!future_dropped.load(Ordering::Acquire));
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn partial_start_endpoint_poll_panic_is_quarantined() {
        let future_dropped = Arc::new(AtomicBool::new(false));
        let mut fixture = fixture(
            EndpointBehavior::PollPanic(Arc::clone(&future_dropped)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(
            start_partial_cleanup(&mut fixture).receipt(),
        ));
        assert_eq!(
            issue.kind(),
            OutputShutdownIssueKind::EndpointFuturePanicked
        );
        assert!(!future_dropped.load(Ordering::Acquire));
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn partial_start_pending_future_silences_events_before_acknowledgement() {
        let (release_send, release_recv) = oneshot::channel();
        let mut fixture = fixture(EndpointBehavior::Pending(release_recv), false, false, false);
        let handle = start_partial_cleanup(&mut fixture);
        let mut receipt = Box::pin(handle.receipt());
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(receipt.as_mut().poll(&mut context).is_pending());
        fixture.event_exit.recv_timeout(TIMEOUT).unwrap();
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));

        release_send.send(()).unwrap();
        let report = confirmed(executor::block_on(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
    }

    #[test]
    fn dropping_partial_start_handle_and_receipts_does_not_cancel_cleanup() {
        let (release_send, release_recv) = oneshot::channel();
        let mut fixture = fixture(EndpointBehavior::Pending(release_recv), false, false, false);
        let (startup_error, resources) = partial_start_resources(&mut fixture);
        let (done_send, done_recv) = crossbeam_channel::bounded(1);
        let handle = match OutputStartCleanupHandle::start_with_spawner(
            startup_error,
            resources,
            &CompletionProbeSpawner(done_send),
        ) {
            Ok(handle) => handle,
            Err(_) => panic!("partial-start cleanup worker should start"),
        };
        drop(handle.receipt());
        drop(handle);
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));

        release_send.send(()).unwrap();
        done_recv.recv_timeout(TIMEOUT).unwrap();
        assert!(fixture.reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn partial_start_false_ok_with_retained_callback_is_unconfirmed() {
        let retained = Arc::new(Mutex::new(None));
        let mut fixture = fixture(
            EndpointBehavior::Retain(Arc::clone(&retained)),
            false,
            false,
            false,
        );
        let issue = unconfirmed(executor::block_on(
            start_partial_cleanup(&mut fixture).receipt(),
        ));
        assert_eq!(issue.kind(), OutputShutdownIssueKind::CallbackRetained);
        assert!(!fixture.reclaim_ran.load(Ordering::Acquire));
        assert!(retained.lock().unwrap().take().is_some());
    }

    struct FailSpawner;

    struct CompletionProbeSpawner(Sender<()>);

    impl LifecycleWorkerSpawner for CompletionProbeSpawner {
        fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<JoinHandle<()>> {
            let done = self.0.clone();
            thread::Builder::new().spawn(move || {
                job();
                let _ = done.send(());
            })
        }
    }

    impl LifecycleWorkerSpawner for FailSpawner {
        fn spawn(&self, _job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<JoinHandle<()>> {
            Err(io::Error::other("injected worker spawn failure"))
        }
    }

    struct DropJobSpawner;

    impl LifecycleWorkerSpawner for DropJobSpawner {
        fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<JoinHandle<()>> {
            drop(job);
            thread::Builder::new().spawn(|| {})
        }
    }

    fn quarantine_start_failure(failure: OutputLifecycleStartFailure) {
        let (_issue, endpoint, render_owner, output_events, mut event_loop, worker) =
            failure.into_parts();
        if let Some(worker) = worker {
            worker.join().unwrap();
        }
        std::mem::forget(endpoint);
        drop(render_owner);
        event_loop.request_silent_stop();
        let _ = event_loop.join();
        drop(output_events);
    }

    fn assert_quarantined_callback_is_closed(retained: &Arc<Mutex<Option<AudioRenderCallback>>>) {
        let mut callback = retained
            .lock()
            .unwrap()
            .take()
            .expect("test retains callback outside the failed owner");
        let mut output = vec![1.; 256];
        assert_eq!(
            callback.render_interleaved_f32(&mut output),
            AudioRenderStatus::Stop
        );
        assert!(output.iter().all(|sample| *sample == 0.));
    }

    #[test]
    fn worker_spawn_failure_returns_live_resources_without_transfer() {
        let fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let failure =
            OutputLifecycleController::start_with_spawner(fixture.resources.unwrap(), &FailSpawner)
                .unwrap_err();
        assert_eq!(
            failure.issue.kind(),
            OutputShutdownIssueKind::WorkerSpawnFailed
        );
        assert!(failure.resources.is_some());
        quarantine_start_failure(*failure);
    }

    #[test]
    fn dropped_running_bootstrap_failure_closes_retained_callback_before_quarantine() {
        let retained = Arc::new(Mutex::new(None));
        let fixture = fixture(
            EndpointBehavior::RetainedBeforeShutdown(Arc::clone(&retained)),
            false,
            false,
            false,
        );
        let reclaim_ran = Arc::clone(&fixture.reclaim_ran);
        let failure =
            OutputLifecycleController::start_with_spawner(fixture.resources.unwrap(), &FailSpawner)
                .unwrap_err();
        drop(failure);

        assert_quarantined_callback_is_closed(&retained);
        assert!(!reclaim_ran.load(Ordering::Acquire));
    }

    #[test]
    fn bootstrap_failure_returns_payload_intact() {
        let fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let failure = OutputLifecycleController::start_with_spawner(
            fixture.resources.unwrap(),
            &DropJobSpawner,
        )
        .unwrap_err();
        assert_eq!(
            failure.issue.kind(),
            OutputShutdownIssueKind::BootstrapFailed
        );
        assert!(failure.resources.is_some());
        quarantine_start_failure(*failure);
    }

    #[test]
    fn partial_start_bootstrap_failure_returns_every_resource_intact() {
        let mut fixture = fixture(EndpointBehavior::ReadyOk, false, false, false);
        let (startup_error, resources) = partial_start_resources(&mut fixture);
        let failure = match OutputStartCleanupHandle::start_with_spawner(
            startup_error,
            resources,
            &DropJobSpawner,
        ) {
            Ok(_) => panic!("injected bootstrap failure should fail"),
            Err(failure) => failure,
        };
        let (issue, start_failure, render_owner, output_events, mut event_loop, worker) =
            failure.into_parts();
        assert_eq!(issue.kind(), OutputShutdownIssueKind::BootstrapFailed);
        assert_eq!(
            start_failure.error().message(),
            "injected partial-start failure"
        );
        worker.unwrap().join().unwrap();

        event_loop.request_silent_stop();
        render_owner.begin_shutdown();
        let (_error, future) = start_failure.into_parts();
        executor::block_on(future).unwrap();
        match render_owner.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new()) {
            Ok(result) => result.unwrap(),
            Err(_) => panic!("returned callback should be uniquely reclaimable"),
        }
        assert_eq!(event_loop.join().unwrap(), EventLoopExit::Silent);
        drop(output_events);
    }

    #[test]
    fn dropped_partial_start_bootstrap_failure_closes_retained_callback_before_quarantine() {
        let retained = Arc::new(Mutex::new(None));
        let mut fixture = fixture(
            EndpointBehavior::Retain(Arc::clone(&retained)),
            false,
            false,
            false,
        );
        let reclaim_ran = Arc::clone(&fixture.reclaim_ran);
        let (startup_error, resources) = partial_start_resources(&mut fixture);
        let failure = match OutputStartCleanupHandle::start_with_spawner(
            startup_error,
            resources,
            &DropJobSpawner,
        ) {
            Ok(_) => panic!("injected bootstrap failure should fail"),
            Err(failure) => failure,
        };
        drop(failure);

        assert_quarantined_callback_is_closed(&retained);
        assert!(!reclaim_ran.load(Ordering::Acquire));
    }
}
