//! Private lifecycle worker for one exactly bound injected output.

use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, TryLockError};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::JoinHandle;

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_PRESPAWN_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_NEXT_PRESPAWN_TRANSFER_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

use super::*;
use crate::context::injected_admission::GraphControlAdmission;
use crate::context::injected_control::{
    BeginControlStateTransition, ControlStateBoundary, ControlStateObservation,
    InjectedControlError, SubmittedControlStateTransition,
};
use crate::context::injected_node_lifetime::{
    InjectedCloseObservation, InjectedNodeRetireOutcome, InjectedOutputRenderOwner,
    InjectedRenderReclaimOutcome, MagicInitializedInjectedOutputRenderer,
    MagicInitializedOutputPairFailure, NodeLifetimeDriveOutcome, ReadyForInjectedPhysicalReclaim,
    ReclaimedInjectedGraph, SealedInjectedOutput, NODE_LIFETIME_RETRY_INTERVAL,
};
use crate::context::ConcreteBaseAudioContext;
use crate::events::{
    InjectedConfirmedEventRetirement, InjectedLifecycleEventLoop, InjectedTerminalStateOutcome,
};
use crate::message::GraphLifecycleTransition;
use crate::output::{
    AudioOutputEventSink, AudioOutputStartFailure, PreparedAudioOutput,
    ValidatedPreparedAudioOutput,
};

const OPEN_DRIVE_BUDGET: usize = 32;

struct PreSpawnedLifecycleWorker {
    job_send: Option<Sender<Box<dyn FnOnce() + Send + 'static>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl PreSpawnedLifecycleWorker {
    fn try_new() -> io::Result<Self> {
        let (job_send, job_recv) =
            crossbeam_channel::bounded::<Box<dyn FnOnce() + Send + 'static>>(1);
        let worker = thread::Builder::new()
            .name("web-audio-output-lifecycle".to_owned())
            .spawn(move || {
                if let Ok(job) = job_recv.recv() {
                    job();
                }
            })?;
        Ok(Self {
            job_send: Some(job_send),
            worker: Mutex::new(Some(worker)),
        })
    }
}

impl LifecycleWorkerSpawner for PreSpawnedLifecycleWorker {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<JoinHandle<()>> {
        #[cfg(test)]
        if FAIL_NEXT_PRESPAWN_TRANSFER_FOR_TEST.replace(false) {
            drop(job);
            return Err(io::Error::other(
                "forced pre-spawned lifecycle ownership-transfer failure",
            ));
        }
        let Some(job_send) = self.job_send.as_ref() else {
            return Err(io::Error::other(
                "pre-spawned lifecycle worker already consumed",
            ));
        };
        job_send.send(job).map_err(|error| {
            // The job owns only empty bootstrap receivers; exact resources remain with the
            // caller until its subsequent bootstrap send succeeds.
            drop(error.0);
            io::Error::other("pre-spawned lifecycle worker retired before job transfer")
        })?;
        self.worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .ok_or_else(|| io::Error::other("pre-spawned lifecycle worker handle was consumed"))
    }
}

impl Drop for PreSpawnedLifecycleWorker {
    fn drop(&mut self) {
        self.job_send.take();
        if let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = worker.join();
        }
    }
}

/// Empty lifecycle-thread owner created before a host factory is invoked or a callback can start.
pub(crate) struct InjectedOutputWorkerBootstrap {
    spawner: PreSpawnedLifecycleWorker,
}

impl InjectedOutputWorkerBootstrap {
    pub(crate) fn try_new() -> io::Result<Self> {
        #[cfg(test)]
        if FAIL_NEXT_PRESPAWN_FOR_TEST.replace(false) {
            return Err(io::Error::other(
                "forced pre-spawn lifecycle worker failure",
            ));
        }
        Ok(Self {
            spawner: PreSpawnedLifecycleWorker::try_new()?,
        })
    }

    #[cfg(test)]
    pub(crate) fn fail_next_spawn_for_test() {
        FAIL_NEXT_PRESPAWN_FOR_TEST.set(true);
    }

    #[cfg(test)]
    pub(crate) fn fail_next_transfer_for_test() {
        FAIL_NEXT_PRESPAWN_TRANSFER_FOR_TEST.set(true);
    }

    #[cfg(test)]
    pub(crate) fn fail_next_fallback_spawn_for_test() {
        super::fail_next_thread_spawn_for_test();
    }

    pub(crate) fn start(
        self,
        prepared: ValidatedPreparedAudioOutput,
        renderer: MagicInitializedInjectedOutputRenderer,
        events: AudioOutputEventSink,
        output_events: AudioOutputEventWatcher,
    ) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
        start_validated_injected_output_with_spawner(
            prepared,
            renderer,
            events,
            output_events,
            &self.spawner,
        )
    }

    pub(crate) fn abort_prepared(
        self,
        prepared: Box<dyn PreparedAudioOutput>,
    ) -> OutputShutdownReceipt {
        match start_prepared_abort_worker(prepared, &self.spawner) {
            Ok(receipt) => receipt,
            Err(prepared) => match start_prepared_abort_worker(prepared, &ThreadSpawner) {
                Ok(receipt) => receipt,
                Err(prepared) => {
                    // No executor exists which can safely poll a potentially hostile shutdown
                    // future. Retain the Prepared owner permanently rather than running ordinary
                    // Drop, and publish an honest terminal observation immediately.
                    std::mem::forget(prepared);
                    let (completer, receipt) = shutdown_receipt_pair();
                    completer.complete(OutputShutdownOutcome::Unconfirmed {
                        failure: OutputShutdownIssue::new(
                            OutputShutdownIssueKind::WorkerSpawnFailed,
                            "no lifecycle worker was available to abort prepared output",
                        ),
                        event_issue: None,
                    });
                    receipt
                }
            },
        }
    }
}

fn start_prepared_abort_worker(
    prepared: Box<dyn PreparedAudioOutput>,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<OutputShutdownReceipt, Box<dyn PreparedAudioOutput>> {
    let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
    let (completer, receipt) = shutdown_receipt_pair();
    let job = Box::new(move || prepared_abort_worker(bootstrap_recv, completer));
    let _worker = match spawner.spawn(job) {
        Ok(worker) => worker,
        Err(_) => return Err(prepared),
    };
    match bootstrap_send.send(prepared) {
        Ok(()) => Ok(receipt),
        Err(error) => Err(error.0),
    }
}

fn prepared_abort_worker(
    bootstrap: Receiver<Box<dyn PreparedAudioOutput>>,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap.recv() {
        Ok(prepared) => match panic::catch_unwind(AssertUnwindSafe(|| prepared.abort())) {
            Ok(shutdown) => match drive_prepared_abort(shutdown) {
                PreparedAbortDriveOutcome::Confirmed => {
                    OutputShutdownOutcome::Confirmed(OutputShutdownReport {
                        mode: OutputShutdownMode::Silent,
                        endpoint_death: None,
                        reclaim_issue: None,
                        event_issue: None,
                    })
                }
                PreparedAbortDriveOutcome::Rejected(error) => OutputShutdownOutcome::Unconfirmed {
                    failure: issue_from_audio_error(
                        OutputShutdownIssueKind::EndpointRejectedShutdown,
                        error,
                    ),
                    event_issue: None,
                },
                PreparedAbortDriveOutcome::Panicked => OutputShutdownOutcome::Unconfirmed {
                    failure: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::EndpointFuturePanicked,
                        "prepared output abort future panicked",
                    ),
                    event_issue: None,
                },
            },
            Err(payload) => {
                quarantine_panic_payload(payload);
                OutputShutdownOutcome::Unconfirmed {
                    failure: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::EndpointMethodPanicked,
                        "prepared output panicked while committing abort",
                    ),
                    event_issue: None,
                }
            }
        },
        Err(_) => OutputShutdownOutcome::ControllerTerminated,
    };
    completer.complete(outcome);
}

struct PreparedAbortWake {
    ready: Arc<(Mutex<bool>, Condvar)>,
}

impl Wake for PreparedAbortWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let (ready, notify) = &*self.ready;
        *ready
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        notify.notify_one();
    }
}

enum PreparedAbortDriveOutcome {
    Confirmed,
    Rejected(crate::output::AudioOutputError),
    Panicked,
}

fn drive_prepared_abort(
    mut shutdown: crate::output::AudioOutputEndpointShutdown,
) -> PreparedAbortDriveOutcome {
    let ready = Arc::new((Mutex::new(false), Condvar::new()));
    let waker = Waker::from(Arc::new(PreparedAbortWake {
        ready: Arc::clone(&ready),
    }));
    let mut context = Context::from_waker(&waker);
    loop {
        let poll = panic::catch_unwind(AssertUnwindSafe(|| {
            Pin::new(&mut shutdown).poll(&mut context)
        }));
        match poll {
            Ok(Poll::Ready(Ok(()))) => return PreparedAbortDriveOutcome::Confirmed,
            Ok(Poll::Ready(Err(error))) => {
                std::mem::forget(shutdown);
                return PreparedAbortDriveOutcome::Rejected(error);
            }
            Err(payload) => {
                std::mem::forget(shutdown);
                quarantine_panic_payload(payload);
                return PreparedAbortDriveOutcome::Panicked;
            }
            Ok(Poll::Pending) => {}
        }
        let (flag, notify) = &*ready;
        let mut flag = flag
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*flag {
            flag = notify
                .wait(flag)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *flag = false;
    }
}

/// Event-loop authority defaults to quarantine on every unexpected unwind. Only paths which have
/// independently proved producer quiescence may extract it and request/join a stop.
struct FailClosedEventLoop(Option<InjectedLifecycleEventLoop>);

impl FailClosedEventLoop {
    fn new(event_loop: InjectedLifecycleEventLoop) -> Self {
        Self(Some(event_loop))
    }

    fn retire_confirmed(
        mut self,
        retired: &crate::context::RetiredInjectedGraph,
        graceful: bool,
    ) -> Result<InjectedConfirmedEventRetirement, Self> {
        match self.0.take().unwrap().retire_confirmed(retired, graceful) {
            Ok(retirement) => Ok(retirement),
            Err(event_loop) => {
                self.0 = Some(event_loop);
                Err(self)
            }
        }
    }
}

impl Drop for FailClosedEventLoop {
    fn drop(&mut self) {
        if let Some(event_loop) = self.0.take() {
            std::mem::forget(event_loop);
        }
    }
}

const STATE_REQUEST_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedStateChangeOutcome {
    Applied,
    Unchanged,
    SupersededByShutdown,
    Closed,
    Failed(InjectedStateChangeFailure),
    ControllerTerminated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedStateChangeFailure {
    Transport,
    EventDelivery,
    EndpointUncertain,
    WorkerPanicked,
}

type StateCompletionReceiver = futures_channel::oneshot::Receiver<InjectedStateChangeOutcome>;

#[derive(Clone)]
pub(crate) struct InjectedStateChangeReceipt {
    shared: futures_util::future::Shared<StateCompletionReceiver>,
}

impl std::future::Future for InjectedStateChangeReceipt {
    type Output = InjectedStateChangeOutcome;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.shared).poll(cx) {
            Poll::Ready(Ok(outcome)) => Poll::Ready(outcome),
            Poll::Ready(Err(_)) => Poll::Ready(InjectedStateChangeOutcome::ControllerTerminated),
            Poll::Pending => Poll::Pending,
        }
    }
}

struct StateChangeCompleter(Option<futures_channel::oneshot::Sender<InjectedStateChangeOutcome>>);

impl StateChangeCompleter {
    fn complete(mut self, outcome: InjectedStateChangeOutcome) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(outcome);
        }
    }
}

impl Drop for StateChangeCompleter {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(InjectedStateChangeOutcome::Failed(
                InjectedStateChangeFailure::WorkerPanicked,
            ));
        }
    }
}

fn state_change_receipt_pair() -> (StateChangeCompleter, InjectedStateChangeReceipt) {
    use futures_util::FutureExt as _;

    let (sender, receiver) = futures_channel::oneshot::channel();
    (
        StateChangeCompleter(Some(sender)),
        InjectedStateChangeReceipt {
            shared: receiver.shared(),
        },
    )
}

struct InjectedStateChangeCommand {
    target: crate::context::AudioContextState,
    admission: GraphControlAdmission,
    completer: StateChangeCompleter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShutdownLatchState {
    Open = 0,
    Graceful = 1,
    Silent = 2,
}

impl ShutdownLatchState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Open,
            1 => Self::Graceful,
            2 => Self::Silent,
            _ => Self::Silent,
        }
    }
}

struct InjectedLifecycleRequestGate {
    phase: AtomicU8,
    serialize: Mutex<()>,
    wake: crossbeam_channel::Sender<()>,
    #[cfg(test)]
    after_serialize: Mutex<Option<Arc<dyn Fn() + Send + Sync + 'static>>>,
}

impl InjectedLifecycleRequestGate {
    fn load(&self) -> ShutdownLatchState {
        ShutdownLatchState::from_u8(self.phase.load(Ordering::Acquire))
    }

    fn latch(&self, mode: OutputShutdownMode) {
        let next = match mode {
            OutputShutdownMode::Graceful => ShutdownLatchState::Graceful,
            OutputShutdownMode::Silent => ShutdownLatchState::Silent,
        };
        let _serialization = self
            .serialize
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = self.load();
        if current == ShutdownLatchState::Open
            || (current == ShutdownLatchState::Graceful && next == ShutdownLatchState::Silent)
        {
            self.phase.store(next as u8, Ordering::Release);
        }
        let _ = self.wake.try_send(());
    }
}

/// Cancellation-independent control for the private exact output lifecycle.
///
/// The fixed-capacity request queue provides admission only: a successful call linearizes before
/// a concurrently latched shutdown and transfers the native transition to the lifecycle worker.
/// Dropping every clone of the returned receipt merely abandons observation; it never cancels an
/// accepted transition. Requests linearized after shutdown complete as
/// [`InjectedStateChangeOutcome::Closed`], and queue saturation is reported synchronously.
///
/// Receipts have no wall-clock deadline. A live stalled renderer or endpoint therefore remains
/// pending until it advances or an authoritative shutdown/death condition supersedes it.
/// [`Applied`](InjectedStateChangeOutcome::Applied) means the renderer changed the exact shared
/// state, successfully enqueued its state event, and the endpoint method completed; it does not
/// mean the event handler has run. Any endpoint error or panic makes endpoint ownership uncertain
/// and starts fail-closed silent teardown rather than permitting a retry on that endpoint.
#[derive(Clone)]
pub(crate) struct InjectedOutputStateControl {
    command_send: Sender<InjectedStateChangeCommand>,
    request_gate: Arc<InjectedLifecycleRequestGate>,
    admission_gate: crate::context::InjectedContextAdmissionGate,
}

impl InjectedOutputStateControl {
    #[cfg(test)]
    pub(crate) fn set_after_serialize_for_test(
        &self,
        observer: Arc<dyn Fn() + Send + Sync + 'static>,
    ) {
        *self
            .request_gate
            .after_serialize
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(observer);
    }

    pub(crate) fn suspend(&self) -> Result<InjectedStateChangeReceipt, InjectedControlError> {
        self.request(crate::context::AudioContextState::Suspended)
    }

    pub(crate) fn resume(&self) -> Result<InjectedStateChangeReceipt, InjectedControlError> {
        self.request(crate::context::AudioContextState::Running)
    }

    fn request(
        &self,
        target: crate::context::AudioContextState,
    ) -> Result<InjectedStateChangeReceipt, InjectedControlError> {
        let serialization = match self.request_gate.serialize.try_lock() {
            Ok(serialization) => serialization,
            Err(TryLockError::WouldBlock) => return Err(InjectedControlError::Contended),
            Err(TryLockError::Poisoned(_)) => return Err(InjectedControlError::Poisoned),
        };
        #[cfg(test)]
        let observer = self
            .request_gate
            .after_serialize
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        #[cfg(test)]
        if let Some(observer) = observer {
            observer();
        }
        let (completer, receipt) = state_change_receipt_pair();
        if self.request_gate.load() != ShutdownLatchState::Open {
            drop(serialization);
            completer.complete(InjectedStateChangeOutcome::Closed);
            return Ok(receipt);
        }
        let admission = self
            .admission_gate
            .try_graph_control()
            .map_err(InjectedControlError::from)?;
        match self.command_send.try_send(InjectedStateChangeCommand {
            target,
            admission,
            completer,
        }) {
            Ok(()) => {
                drop(serialization);
                let _ = self.request_gate.wake.try_send(());
                Ok(receipt)
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                Err(InjectedControlError::LogicalCommandCredits)
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(InjectedControlError::Disconnected)
            }
        }
    }
}

pub(crate) enum InjectedOutputStart {
    Running(InjectedOutputLifecycleController),
    Cleanup(InjectedOutputStartCleanupHandle),
}

#[allow(clippy::large_enum_variant)] // bootstrap failure returns exact owners without extra boxing
enum InjectedStartFailureResources {
    Prepared {
        prepared: Box<dyn PreparedAudioOutput>,
        renderer: MagicInitializedInjectedOutputRenderer,
        events: AudioOutputEventSink,
        output_events: AudioOutputEventWatcher,
    },
    Running {
        endpoint: Box<dyn RunningAudioOutput>,
        owner: InjectedOutputRenderOwner,
        base: ConcreteBaseAudioContext,
        output_events: AudioOutputEventWatcher,
        event_loop: InjectedLifecycleEventLoop,
    },
    Partial {
        start_failure: AudioOutputStartFailure,
        owner: InjectedOutputRenderOwner,
        output_events: AudioOutputEventWatcher,
        event_loop: InjectedLifecycleEventLoop,
    },
    Uncertain {
        owner: InjectedOutputRenderOwner,
        output_events: AudioOutputEventWatcher,
        event_loop: InjectedLifecycleEventLoop,
    },
}

pub(crate) struct InjectedOutputStartFailure {
    issue: OutputShutdownIssue,
    resources: Option<InjectedStartFailureResources>,
    abandoned_worker: Option<JoinHandle<()>>,
}

#[allow(clippy::large_enum_variant)] // consuming recovery keeps exact owners directly reusable
pub(crate) enum InjectedOutputStartFailureParts {
    Prepared {
        prepared: Box<dyn PreparedAudioOutput>,
        renderer: MagicInitializedInjectedOutputRenderer,
        events: AudioOutputEventSink,
        output_events: AudioOutputEventWatcher,
    },
    Running {
        endpoint: Box<dyn RunningAudioOutput>,
        owner: InjectedOutputRenderOwner,
        base: ConcreteBaseAudioContext,
        output_events: AudioOutputEventWatcher,
        event_loop: InjectedLifecycleEventLoop,
    },
    Partial {
        start_failure: AudioOutputStartFailure,
        owner: InjectedOutputRenderOwner,
        output_events: AudioOutputEventWatcher,
        event_loop: InjectedLifecycleEventLoop,
    },
    Uncertain {
        owner: InjectedOutputRenderOwner,
        output_events: AudioOutputEventWatcher,
        event_loop: InjectedLifecycleEventLoop,
    },
}

impl InjectedOutputStartFailure {
    pub(crate) fn issue(&self) -> &OutputShutdownIssue {
        &self.issue
    }

    pub(crate) fn startup_error(&self) -> Option<&AudioOutputError> {
        match self.resources.as_ref()? {
            InjectedStartFailureResources::Partial { start_failure, .. } => {
                Some(start_failure.error())
            }
            InjectedStartFailureResources::Prepared { .. }
            | InjectedStartFailureResources::Running { .. }
            | InjectedStartFailureResources::Uncertain { .. } => None,
        }
    }

    pub(crate) fn into_parts(
        mut self,
    ) -> (
        OutputShutdownIssue,
        InjectedOutputStartFailureParts,
        Option<JoinHandle<()>>,
    ) {
        let resources = match self.resources.take().unwrap() {
            InjectedStartFailureResources::Prepared {
                prepared,
                renderer,
                events,
                output_events,
            } => InjectedOutputStartFailureParts::Prepared {
                prepared,
                renderer,
                events,
                output_events,
            },
            InjectedStartFailureResources::Running {
                endpoint,
                owner,
                base,
                output_events,
                event_loop,
            } => InjectedOutputStartFailureParts::Running {
                endpoint,
                owner,
                base,
                output_events,
                event_loop,
            },
            InjectedStartFailureResources::Partial {
                start_failure,
                owner,
                output_events,
                event_loop,
            } => InjectedOutputStartFailureParts::Partial {
                start_failure,
                owner,
                output_events,
                event_loop,
            },
            InjectedStartFailureResources::Uncertain {
                owner,
                output_events,
                event_loop,
            } => InjectedOutputStartFailureParts::Uncertain {
                owner,
                output_events,
                event_loop,
            },
        };
        (self.issue.clone(), resources, self.abandoned_worker.take())
    }

    /// Recovers a failed pre-spawn/bootstrap transfer and hands the exact owner set to a fresh
    /// lifecycle worker. If no worker can be created, the returned observation is explicitly
    /// unconfirmed and the exact resources retain their existing fail-closed Drop behavior.
    pub(crate) fn recover_with_fallback(self: Box<Self>) -> OutputShutdownReceipt {
        let (issue, parts, abandoned_worker) = (*self).into_parts();
        // A rejected bootstrap worker never received exact resources. Detaching its handle avoids
        // making public construction wait for scheduler progress on an irrelevant empty thread.
        drop(abandoned_worker);
        match parts {
            InjectedOutputStartFailureParts::Prepared {
                prepared,
                renderer,
                events,
                output_events,
            } => {
                let abort_started = match start_prepared_abort_worker(prepared, &ThreadSpawner) {
                    Ok(receipt) => {
                        drop(receipt);
                        true
                    }
                    Err(prepared) => {
                        std::mem::forget(prepared);
                        false
                    }
                };
                // No callback was published, but there is no renderer drive capable of applying
                // exact Close. Retain that open proof domain and report it honestly as unconfirmed.
                std::mem::forget(renderer);
                drop(events);
                drop(output_events);
                ready_unconfirmed_receipt(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::BootstrapFailed,
                    if abort_started {
                        "prepared output abort was transferred, but the unstarted exact graph was quarantined"
                    } else {
                        "prepared output and unstarted exact graph were quarantined after lifecycle transfer failure"
                    },
                ))
            }
            InjectedOutputStartFailureParts::Running {
                endpoint,
                owner,
                base,
                output_events,
                event_loop,
            } => match start_running(
                endpoint,
                owner,
                base,
                output_events,
                event_loop,
                &ThreadSpawner,
            ) {
                Ok(controller) => controller.shutdown_silently(),
                Err(failure) => {
                    drop(failure);
                    ready_unconfirmed_receipt(issue)
                }
            },
            InjectedOutputStartFailureParts::Partial {
                start_failure,
                owner,
                output_events,
                event_loop,
            } => {
                let (startup_error, endpoint_shutdown) = start_failure.into_parts();
                match start_partial(
                    startup_error,
                    endpoint_shutdown,
                    owner,
                    output_events,
                    event_loop,
                    &ThreadSpawner,
                ) {
                    Ok(cleanup) => cleanup.receipt(),
                    Err(failure) => {
                        drop(failure);
                        ready_unconfirmed_receipt(issue)
                    }
                }
            }
            InjectedOutputStartFailureParts::Uncertain {
                owner,
                output_events,
                event_loop,
            } => match start_uncertain(owner, output_events, event_loop, &ThreadSpawner) {
                Ok(cleanup) => cleanup.receipt(),
                Err(failure) => {
                    drop(failure);
                    ready_unconfirmed_receipt(issue)
                }
            },
        }
    }
}

fn ready_unconfirmed_receipt(failure: OutputShutdownIssue) -> OutputShutdownReceipt {
    let (completer, receipt) = shutdown_receipt_pair();
    completer.complete(OutputShutdownOutcome::Unconfirmed {
        failure,
        event_issue: None,
    });
    receipt
}

fn quarantine_open_owner(owner: InjectedOutputRenderOwner) {
    let render = owner.quarantine_into_render_owner();
    render.begin_shutdown();
    drop(render);
}

impl Drop for InjectedOutputStartFailure {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            match resources {
                InjectedStartFailureResources::Prepared {
                    prepared,
                    renderer,
                    events,
                    output_events,
                } => {
                    std::mem::forget(prepared);
                    std::mem::forget(renderer);
                    drop(events);
                    drop(output_events);
                    // The renderer structurally retains the exact event-loop authority. Forgetting
                    // the composite prevents a stop request while its open graph may still emit.
                }
                InjectedStartFailureResources::Running {
                    endpoint,
                    owner,
                    base,
                    output_events,
                    event_loop,
                } => {
                    drop(base);
                    std::mem::forget(endpoint);
                    quarantine_open_owner(owner);
                    drop(output_events);
                    std::mem::forget(event_loop);
                }
                InjectedStartFailureResources::Partial {
                    start_failure,
                    owner,
                    output_events,
                    event_loop,
                } => {
                    std::mem::forget(start_failure);
                    quarantine_open_owner(owner);
                    drop(output_events);
                    std::mem::forget(event_loop);
                }
                InjectedStartFailureResources::Uncertain {
                    owner,
                    output_events,
                    event_loop,
                } => {
                    quarantine_open_owner(owner);
                    drop(output_events);
                    std::mem::forget(event_loop);
                }
            }
        }
        self.abandoned_worker.take();
    }
}

pub(crate) struct InjectedOutputLifecycleController {
    base: ConcreteBaseAudioContext,
    state_control: InjectedOutputStateControl,
    request_gate: Arc<InjectedLifecycleRequestGate>,
    receipt: OutputShutdownReceipt,
    worker: Option<JoinHandle<()>>,
    requested: bool,
}

impl InjectedOutputLifecycleController {
    pub(crate) const fn base(&self) -> &ConcreteBaseAudioContext {
        &self.base
    }

    pub(crate) fn receipt(&self) -> OutputShutdownReceipt {
        self.receipt.clone()
    }

    pub(crate) fn state_control(&self) -> InjectedOutputStateControl {
        self.state_control.clone()
    }

    pub(crate) fn shutdown_gracefully(mut self) -> OutputShutdownReceipt {
        self.requested = true;
        self.request_gate.latch(OutputShutdownMode::Graceful);
        self.receipt.clone()
    }

    pub(crate) fn shutdown_silently(mut self) -> OutputShutdownReceipt {
        self.requested = true;
        self.request_gate.latch(OutputShutdownMode::Silent);
        self.receipt.clone()
    }
}

impl Drop for InjectedOutputLifecycleController {
    fn drop(&mut self) {
        if !self.requested {
            self.request_gate.latch(OutputShutdownMode::Silent);
        }
        self.worker.take();
    }
}

pub(crate) struct InjectedOutputStartCleanupHandle {
    startup_error: Option<AudioOutputError>,
    receipt: OutputShutdownReceipt,
    worker: Option<JoinHandle<()>>,
}

impl InjectedOutputStartCleanupHandle {
    pub(crate) fn startup_error(&self) -> Option<&AudioOutputError> {
        self.startup_error.as_ref()
    }

    pub(crate) fn receipt(&self) -> OutputShutdownReceipt {
        self.receipt.clone()
    }
}

impl Drop for InjectedOutputStartCleanupHandle {
    fn drop(&mut self) {
        self.worker.take();
    }
}

struct InjectedRunningResources {
    endpoint: Option<Box<dyn RunningAudioOutput>>,
    owner: Option<InjectedOutputRenderOwner>,
    output_events: Option<AudioOutputEventWatcher>,
    event_loop: Option<InjectedLifecycleEventLoop>,
}

impl Drop for InjectedRunningResources {
    fn drop(&mut self) {
        if let Some(endpoint) = self.endpoint.take() {
            std::mem::forget(endpoint);
        }
        if let Some(owner) = self.owner.take() {
            quarantine_open_owner(owner);
        }
        self.output_events.take();
        if let Some(event_loop) = self.event_loop.take() {
            std::mem::forget(event_loop);
        }
    }
}

struct InjectedPartialResources {
    endpoint_shutdown: Option<AudioOutputEndpointShutdown>,
    owner: Option<InjectedOutputRenderOwner>,
    output_events: Option<AudioOutputEventWatcher>,
    event_loop: Option<InjectedLifecycleEventLoop>,
}

impl Drop for InjectedPartialResources {
    fn drop(&mut self) {
        if let Some(shutdown) = self.endpoint_shutdown.take() {
            std::mem::forget(shutdown);
        }
        if let Some(owner) = self.owner.take() {
            quarantine_open_owner(owner);
        }
        self.output_events.take();
        if let Some(event_loop) = self.event_loop.take() {
            std::mem::forget(event_loop);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_validated_injected_output(
    prepared: ValidatedPreparedAudioOutput,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
    start_validated_injected_output_with_spawner(
        prepared,
        renderer,
        events,
        output_events,
        &ThreadSpawner,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn start_injected_output(
    prepared: Box<dyn PreparedAudioOutput>,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
    start_injected_output_with_spawner(prepared, renderer, events, output_events, &ThreadSpawner)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn start_injected_output_with_spawner(
    prepared: Box<dyn PreparedAudioOutput>,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
    let config = match panic::catch_unwind(AssertUnwindSafe(|| prepared.config().clone())) {
        Ok(config) => config,
        Err(payload) => {
            quarantine_panic_payload(payload);
            return Err(prepared_failure(
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::BootstrapFailed,
                    "prepared output panicked while reporting its negotiated format",
                ),
                prepared,
                renderer,
                events,
                output_events,
            ));
        }
    };
    start_validated_injected_output_with_spawner(
        ValidatedPreparedAudioOutput::from_prevalidated_for_test(prepared, config),
        renderer,
        events,
        output_events,
        spawner,
    )
}

#[allow(clippy::too_many_arguments)]
fn start_validated_injected_output_with_spawner(
    prepared: ValidatedPreparedAudioOutput,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
    let format = prepared.config().format();
    let (prepared, _config) = prepared.into_parts();
    if !events.matches_watcher(&output_events) {
        return Err(prepared_failure(
            OutputShutdownIssue::new(
                OutputShutdownIssueKind::BootstrapFailed,
                "injected output event sink and watcher do not match",
            ),
            prepared,
            renderer,
            events,
            output_events,
        ));
    }

    let (owner, callback, event_loop, base) =
        match renderer.try_into_audio_output_pair(format, events.clone()) {
            Ok(pair) => pair,
            Err(MagicInitializedOutputPairFailure {
                error,
                renderer,
                events: returned_events,
            }) => {
                drop(returned_events);
                return Err(prepared_failure(
                    issue_from_audio_error(OutputShutdownIssueKind::BootstrapFailed, error),
                    prepared,
                    renderer,
                    events,
                    output_events,
                ));
            }
        };

    match panic::catch_unwind(AssertUnwindSafe(|| prepared.start(callback, events))) {
        Ok(Ok(endpoint)) => {
            start_running(endpoint, owner, base, output_events, event_loop, spawner)
                .map(InjectedOutputStart::Running)
        }
        Ok(Err(start_failure)) => {
            drop(base);
            let (startup_error, endpoint_shutdown) = start_failure.into_parts();
            start_partial(
                startup_error,
                endpoint_shutdown,
                owner,
                output_events,
                event_loop,
                spawner,
            )
            .map(InjectedOutputStart::Cleanup)
        }
        Err(payload) => {
            quarantine_panic_payload(payload);
            drop(base);
            start_uncertain(owner, output_events, event_loop, spawner)
                .map(InjectedOutputStart::Cleanup)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepared_failure(
    issue: OutputShutdownIssue,
    prepared: Box<dyn PreparedAudioOutput>,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
) -> Box<InjectedOutputStartFailure> {
    Box::new(InjectedOutputStartFailure {
        issue,
        resources: Some(InjectedStartFailureResources::Prepared {
            prepared,
            renderer,
            events,
            output_events,
        }),
        abandoned_worker: None,
    })
}

fn start_running(
    endpoint: Box<dyn RunningAudioOutput>,
    owner: InjectedOutputRenderOwner,
    base: ConcreteBaseAudioContext,
    output_events: AudioOutputEventWatcher,
    event_loop: InjectedLifecycleEventLoop,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<InjectedOutputLifecycleController, Box<InjectedOutputStartFailure>> {
    let initially_suspended = base.state() == crate::context::AudioContextState::Suspended;
    let admission_gate = owner.state_request_gate();
    let resources = InjectedRunningResources {
        endpoint: Some(endpoint),
        owner: Some(owner),
        output_events: Some(output_events),
        event_loop: Some(event_loop),
    };
    let (command_send, command_recv) = crossbeam_channel::bounded(STATE_REQUEST_CAPACITY);
    let (request_wake, request_wake_receiver) = crossbeam_channel::bounded(1);
    let request_gate = Arc::new(InjectedLifecycleRequestGate {
        phase: AtomicU8::new(ShutdownLatchState::Open as u8),
        serialize: Mutex::new(()),
        wake: request_wake,
        #[cfg(test)]
        after_serialize: Mutex::new(None),
    });
    let worker_request_gate = Arc::clone(&request_gate);
    let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
    let (completer, receipt) = shutdown_receipt_pair();
    let job = Box::new(move || {
        injected_running_worker(
            bootstrap_recv,
            command_recv,
            request_wake_receiver,
            worker_request_gate,
            initially_suspended,
            completer,
        )
    });
    let worker = match spawner.spawn(job) {
        Ok(worker) => worker,
        Err(error) => return Err(running_transfer_failure(error, resources, base, None)),
    };
    if let Err(error) = bootstrap_send.send(resources) {
        return Err(running_transfer_failure(
            io::Error::other("injected lifecycle worker rejected ownership bootstrap"),
            error.0,
            base,
            Some(worker),
        ));
    }
    Ok(InjectedOutputLifecycleController {
        base,
        state_control: InjectedOutputStateControl {
            command_send,
            request_gate: Arc::clone(&request_gate),
            admission_gate,
        },
        request_gate,
        receipt,
        worker: Some(worker),
        requested: false,
    })
}

fn running_transfer_failure(
    error: io::Error,
    mut resources: InjectedRunningResources,
    base: ConcreteBaseAudioContext,
    worker: Option<JoinHandle<()>>,
) -> Box<InjectedOutputStartFailure> {
    Box::new(InjectedOutputStartFailure {
        issue: OutputShutdownIssue::new(
            if worker.is_some() {
                OutputShutdownIssueKind::BootstrapFailed
            } else {
                OutputShutdownIssueKind::WorkerSpawnFailed
            },
            Arc::<str>::from(error.to_string()),
        ),
        resources: Some(InjectedStartFailureResources::Running {
            endpoint: resources.endpoint.take().unwrap(),
            owner: resources.owner.take().unwrap(),
            base,
            output_events: resources.output_events.take().unwrap(),
            event_loop: resources.event_loop.take().unwrap(),
        }),
        abandoned_worker: worker,
    })
}

fn start_partial(
    startup_error: AudioOutputError,
    endpoint_shutdown: AudioOutputEndpointShutdown,
    owner: InjectedOutputRenderOwner,
    output_events: AudioOutputEventWatcher,
    event_loop: InjectedLifecycleEventLoop,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<InjectedOutputStartCleanupHandle, Box<InjectedOutputStartFailure>> {
    let resources = InjectedPartialResources {
        endpoint_shutdown: Some(endpoint_shutdown),
        owner: Some(owner),
        output_events: Some(output_events),
        event_loop: Some(event_loop),
    };
    let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
    let (completer, receipt) = shutdown_receipt_pair();
    let job = Box::new(move || injected_partial_worker(bootstrap_recv, completer));
    let worker = match spawner.spawn(job) {
        Ok(worker) => worker,
        Err(error) => {
            return Err(partial_transfer_failure(
                error,
                startup_error,
                resources,
                None,
            ));
        }
    };
    if let Err(error) = bootstrap_send.send(resources) {
        return Err(partial_transfer_failure(
            io::Error::other("injected partial-start worker rejected ownership bootstrap"),
            startup_error,
            error.0,
            Some(worker),
        ));
    }
    Ok(InjectedOutputStartCleanupHandle {
        startup_error: Some(startup_error),
        receipt,
        worker: Some(worker),
    })
}

fn partial_transfer_failure(
    error: io::Error,
    startup_error: AudioOutputError,
    mut resources: InjectedPartialResources,
    worker: Option<JoinHandle<()>>,
) -> Box<InjectedOutputStartFailure> {
    Box::new(InjectedOutputStartFailure {
        issue: OutputShutdownIssue::new(
            if worker.is_some() {
                OutputShutdownIssueKind::BootstrapFailed
            } else {
                OutputShutdownIssueKind::WorkerSpawnFailed
            },
            Arc::<str>::from(error.to_string()),
        ),
        resources: Some(InjectedStartFailureResources::Partial {
            start_failure: AudioOutputStartFailure::new(
                startup_error,
                resources.endpoint_shutdown.take().unwrap(),
            ),
            owner: resources.owner.take().unwrap(),
            output_events: resources.output_events.take().unwrap(),
            event_loop: resources.event_loop.take().unwrap(),
        }),
        abandoned_worker: worker,
    })
}

fn start_uncertain(
    owner: InjectedOutputRenderOwner,
    output_events: AudioOutputEventWatcher,
    event_loop: InjectedLifecycleEventLoop,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<InjectedOutputStartCleanupHandle, Box<InjectedOutputStartFailure>> {
    let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
    let (completer, receipt) = shutdown_receipt_pair();
    let job = Box::new(move || uncertain_worker(bootstrap_recv, completer));
    let worker = match spawner.spawn(job) {
        Ok(worker) => worker,
        Err(error) => {
            return Err(Box::new(InjectedOutputStartFailure {
                issue: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::WorkerSpawnFailed,
                    Arc::<str>::from(error.to_string()),
                ),
                resources: Some(InjectedStartFailureResources::Uncertain {
                    owner,
                    output_events,
                    event_loop,
                }),
                abandoned_worker: None,
            }));
        }
    };
    if let Err(error) = bootstrap_send.send((owner, output_events, event_loop)) {
        let (owner, output_events, event_loop) = error.0;
        return Err(Box::new(InjectedOutputStartFailure {
            issue: OutputShutdownIssue::new(
                OutputShutdownIssueKind::BootstrapFailed,
                "injected start-panic worker rejected ownership bootstrap",
            ),
            resources: Some(InjectedStartFailureResources::Uncertain {
                owner,
                output_events,
                event_loop,
            }),
            abandoned_worker: Some(worker),
        }));
    }
    Ok(InjectedOutputStartCleanupHandle {
        startup_error: None,
        receipt,
        worker: Some(worker),
    })
}

fn uncertain_worker(
    bootstrap: Receiver<(
        InjectedOutputRenderOwner,
        AudioOutputEventWatcher,
        InjectedLifecycleEventLoop,
    )>,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap.recv() {
        Ok(resources) => panic::catch_unwind(AssertUnwindSafe(|| run_uncertain(resources)))
            .unwrap_or_else(|payload| {
                quarantine_panic_payload(payload);
                OutputShutdownOutcome::Unconfirmed {
                    failure: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::WorkerPanicked,
                        "injected uncertain-start cleanup worker panicked unexpectedly",
                    ),
                    event_issue: Some(OutputShutdownIssue::new(
                        OutputShutdownIssueKind::EventThreadUnretired,
                        "worker panic left injected event-thread retirement unconfirmed",
                    )),
                }
            }),
        Err(_) => OutputShutdownOutcome::ControllerTerminated,
    };
    completer.complete(outcome);
}

fn run_uncertain(
    (owner, output_events, event_loop): (
        InjectedOutputRenderOwner,
        AudioOutputEventWatcher,
        InjectedLifecycleEventLoop,
    ),
) -> OutputShutdownOutcome {
    let event_loop = FailClosedEventLoop::new(event_loop);
    let mut request = InjectedLifecycleRequest {
        mode: OutputShutdownMode::Silent,
        endpoint_death: output_events.death_reason(),
        issue: None,
    };
    match prepare_injected_close(owner, &mut request, &output_events, None) {
        PreparedInjectedClose::Ready(ready) => {
            wait_for_ready_render_quiescence(&ready);
            drop(ready);
            drop(output_events);
            quarantine_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::InjectedStartPanicked,
                    "prepared output panicked while accepting the injected callback",
                ),
            )
        }
        PreparedInjectedClose::NoProof { render, .. } => {
            wait_for_render_quiescence(&render);
            drop(render);
            drop(output_events);
            quarantine_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::InjectedStartPanicked,
                    "prepared output panicked before producer quiescence was established",
                ),
            )
        }
        PreparedInjectedClose::EndpointUncertain { render, issue } => {
            wait_for_render_quiescence(&render);
            drop(render);
            drop(output_events);
            finish_unconfirmed_events(event_loop, issue)
        }
    }
}

fn injected_running_worker(
    bootstrap: Receiver<InjectedRunningResources>,
    commands: Receiver<InjectedStateChangeCommand>,
    request_wake: Receiver<()>,
    request_gate: Arc<InjectedLifecycleRequestGate>,
    initially_suspended: bool,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap.recv() {
        Ok(mut resources) => panic::catch_unwind(AssertUnwindSafe(|| {
            run_injected_running(
                &mut resources,
                &commands,
                &request_wake,
                &request_gate,
                initially_suspended,
            )
        }))
        .unwrap_or_else(|payload| {
            quarantine_panic_payload(payload);
            request_gate.latch(OutputShutdownMode::Silent);
            drain_state_commands(
                &commands,
                InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::WorkerPanicked),
            );
            OutputShutdownOutcome::Unconfirmed {
                failure: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::WorkerPanicked,
                    "injected output lifecycle worker panicked unexpectedly",
                ),
                event_issue: Some(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventThreadUnretired,
                    "worker panic left injected event-thread retirement unconfirmed",
                )),
            }
        }),
        Err(_) => OutputShutdownOutcome::ControllerTerminated,
    };
    completer.complete(outcome);
}

fn injected_partial_worker(
    bootstrap: Receiver<InjectedPartialResources>,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap.recv() {
        Ok(mut resources) => {
            panic::catch_unwind(AssertUnwindSafe(|| run_injected_partial(&mut resources)))
                .unwrap_or_else(|payload| {
                    quarantine_panic_payload(payload);
                    OutputShutdownOutcome::Unconfirmed {
                        failure: OutputShutdownIssue::new(
                            OutputShutdownIssueKind::WorkerPanicked,
                            "injected partial-start cleanup worker panicked unexpectedly",
                        ),
                        event_issue: Some(OutputShutdownIssue::new(
                            OutputShutdownIssueKind::EventThreadUnretired,
                            "worker panic left injected event-thread retirement unconfirmed",
                        )),
                    }
                })
        }
        Err(_) => OutputShutdownOutcome::ControllerTerminated,
    };
    completer.complete(outcome);
}

struct InjectedLifecycleRequest {
    mode: OutputShutdownMode,
    endpoint_death: Option<AudioOutputDeathReason>,
    issue: Option<OutputShutdownIssue>,
}

fn promote_late_endpoint_death(
    output_events: &AudioOutputEventWatcher,
    request: &mut InjectedLifecycleRequest,
) {
    if let Some(reason) = output_events.death_reason() {
        request.mode = OutputShutdownMode::Silent;
        request.endpoint_death = Some(reason);
    }
}

fn run_injected_running(
    resources: &mut InjectedRunningResources,
    commands: &Receiver<InjectedStateChangeCommand>,
    request_wake: &Receiver<()>,
    request_gate: &InjectedLifecycleRequestGate,
    initially_suspended: bool,
) -> OutputShutdownOutcome {
    let mut owner = resources.owner.take().unwrap();
    let output_events = resources.output_events.as_ref().unwrap();
    let endpoint = resources.endpoint.as_deref_mut().unwrap();
    let live = drive_running_until_shutdown(
        &mut owner,
        endpoint,
        commands,
        request_wake,
        request_gate,
        output_events,
        initially_suspended,
    );
    let mut request = match live {
        Ok(request) => request,
        Err(issue) => {
            request_gate.latch(OutputShutdownMode::Silent);
            drain_state_commands(commands, InjectedStateChangeOutcome::SupersededByShutdown);
            let mut request = InjectedLifecycleRequest {
                mode: OutputShutdownMode::Silent,
                endpoint_death: output_events.death_reason(),
                issue: Some(issue.clone()),
            };
            let prepared = prepare_injected_close(owner, &mut request, output_events, None);
            let event_loop = FailClosedEventLoop::new(resources.event_loop.take().unwrap());
            let output_events = resources.output_events.take().unwrap();
            let endpoint = resources.endpoint.take().unwrap();
            std::mem::forget(endpoint);
            return match prepared {
                PreparedInjectedClose::Ready(ready) => {
                    wait_for_ready_render_quiescence(&ready);
                    drop(ready);
                    drop(output_events);
                    finish_unconfirmed_events(event_loop, issue)
                }
                PreparedInjectedClose::NoProof { render, .. }
                | PreparedInjectedClose::EndpointUncertain { render, .. } => {
                    wait_for_render_quiescence(&render);
                    drop(render);
                    drop(output_events);
                    finish_unconfirmed_events(event_loop, issue)
                }
            };
        }
    };
    request_gate.latch(request.mode);
    drain_state_commands(commands, InjectedStateChangeOutcome::SupersededByShutdown);
    let prepared = prepare_injected_close(
        owner,
        &mut request,
        output_events,
        resources.endpoint.as_deref_mut(),
    );
    let event_loop = FailClosedEventLoop::new(resources.event_loop.take().unwrap());
    let output_events = resources.output_events.take().unwrap();
    let endpoint = resources.endpoint.take().unwrap();
    match prepared {
        PreparedInjectedClose::Ready(ready) => {
            finish_running_ready(endpoint, ready, output_events, event_loop, request)
        }
        PreparedInjectedClose::NoProof { render, issue } => finish_running_without_graph_proof(
            endpoint,
            render,
            output_events,
            event_loop,
            request,
            issue,
        ),
        PreparedInjectedClose::EndpointUncertain { render, issue } => {
            std::mem::forget(endpoint);
            wait_for_render_quiescence(&render);
            drop(render);
            drop(output_events);
            finish_unconfirmed_events(event_loop, issue)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerGraphState {
    Running,
    Suspended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerEndpointState {
    Running,
    Suspended,
}

enum StateCommandDrive {
    Continue,
    Shutdown(InjectedLifecycleRequest),
    EndpointUncertain(OutputShutdownIssue),
}

enum StateAckFailure {
    Superseded(InjectedLifecycleRequest),
    Terminal {
        request: InjectedLifecycleRequest,
        failure: InjectedStateChangeFailure,
    },
}

fn drain_state_commands(
    commands: &Receiver<InjectedStateChangeCommand>,
    outcome: InjectedStateChangeOutcome,
) {
    while let Ok(command) = commands.try_recv() {
        let InjectedStateChangeCommand {
            admission,
            completer,
            ..
        } = command;
        drop(admission);
        completer.complete(outcome);
    }
}

fn request_from_latch(
    request_gate: &InjectedLifecycleRequestGate,
    output_events: &AudioOutputEventWatcher,
    issue: Option<OutputShutdownIssue>,
) -> Option<InjectedLifecycleRequest> {
    if let Some(reason) = output_events.death_reason() {
        return Some(InjectedLifecycleRequest {
            mode: OutputShutdownMode::Silent,
            endpoint_death: Some(reason),
            issue,
        });
    }
    let mode = match request_gate.load() {
        ShutdownLatchState::Open => return None,
        ShutdownLatchState::Graceful => OutputShutdownMode::Graceful,
        ShutdownLatchState::Silent => OutputShutdownMode::Silent,
    };
    Some(InjectedLifecycleRequest {
        mode,
        endpoint_death: None,
        issue,
    })
}

fn drive_open_node_lifetimes(
    owner: &mut InjectedOutputRenderOwner,
    issue: &mut Option<OutputShutdownIssue>,
) {
    if issue.is_some() {
        return;
    }
    for _ in 0..OPEN_DRIVE_BUDGET {
        match owner.try_drive_node_lifetimes() {
            NodeLifetimeDriveOutcome::Idle | NodeLifetimeDriveOutcome::Retry { .. } => break,
            NodeLifetimeDriveOutcome::Quarantined { .. } => {
                *issue = Some(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::InjectedNodeLifetime,
                    "injected node-lifetime driver entered quarantine",
                ));
                break;
            }
            NodeLifetimeDriveOutcome::Submitted { .. }
            | NodeLifetimeDriveOutcome::Reconciled { .. }
            | NodeLifetimeDriveOutcome::CloseSuperseded { .. } => {}
        }
    }
}

fn drive_running_until_shutdown(
    owner: &mut InjectedOutputRenderOwner,
    endpoint: &mut dyn RunningAudioOutput,
    commands: &Receiver<InjectedStateChangeCommand>,
    request_wake: &Receiver<()>,
    request_gate: &InjectedLifecycleRequestGate,
    output_events: &AudioOutputEventWatcher,
    initially_suspended: bool,
) -> Result<InjectedLifecycleRequest, OutputShutdownIssue> {
    let mut graph = if initially_suspended {
        WorkerGraphState::Suspended
    } else {
        WorkerGraphState::Running
    };
    // Prepared::start publishes a logically Running endpoint. An initially-suspended graph is
    // reconciled natively before servicing user requests and without emitting a duplicate event.
    let mut endpoint_state = WorkerEndpointState::Running;
    if initially_suspended {
        if let Some(request) = request_from_latch(request_gate, output_events, None) {
            return Ok(request);
        }
        match panic::catch_unwind(AssertUnwindSafe(|| endpoint.suspend())) {
            Ok(Ok(())) => endpoint_state = WorkerEndpointState::Suspended,
            Ok(Err(error)) => {
                return Err(issue_from_audio_error(
                    OutputShutdownIssueKind::EndpointStateTransitionFailed,
                    error,
                ));
            }
            Err(payload) => {
                quarantine_panic_payload(payload);
                return Err(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EndpointMethodPanicked,
                    "injected endpoint panicked during initial native suspension",
                ));
            }
        }
        // Close/death may have committed while the synchronous native method was in flight. It
        // dominates before any caller state request is observed.
        if let Some(request) = request_from_latch(request_gate, output_events, None) {
            return Ok(request);
        }
    }

    let mut issue = None;
    loop {
        if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
            return Ok(request);
        }
        drive_open_node_lifetimes(owner, &mut issue);
        if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
            return Ok(request);
        }
        match commands.try_recv() {
            Ok(command) => match drive_state_command(
                owner,
                endpoint,
                command,
                request_gate,
                output_events,
                &mut graph,
                &mut endpoint_state,
                issue.clone(),
            ) {
                StateCommandDrive::Continue => continue,
                StateCommandDrive::Shutdown(request) => return Ok(request),
                StateCommandDrive::EndpointUncertain(issue) => return Err(issue),
            },
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return Ok(InjectedLifecycleRequest {
                    mode: OutputShutdownMode::Silent,
                    endpoint_death: None,
                    issue,
                });
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }
        crossbeam_channel::select! {
            recv(request_wake) -> _ => {},
            recv(commands) -> command => {
                if let Ok(command) = command {
                    match drive_state_command(
                        owner,
                        endpoint,
                        command,
                        request_gate,
                        output_events,
                        &mut graph,
                        &mut endpoint_state,
                        issue.clone(),
                    ) {
                        StateCommandDrive::Continue => {},
                        StateCommandDrive::Shutdown(request) => return Ok(request),
                        StateCommandDrive::EndpointUncertain(issue) => return Err(issue),
                    }
                }
            },
            default(NODE_LIFETIME_RETRY_INTERVAL) => {},
        }
    }
}

fn terminal_state_failure(
    snapshot: crate::message::GraphLifecycleSnapshot,
) -> (InjectedStateChangeFailure, OutputShutdownIssue) {
    let event_failed = matches!(
        snapshot,
        crate::message::GraphLifecycleSnapshot::Applied {
            outcome: crate::message::GraphLifecycleOutcome::EventDeliveryFailed,
            ..
        }
    );
    if event_failed {
        (
            InjectedStateChangeFailure::EventDelivery,
            OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventDeliveryDegraded,
                "renderer changed injected state but could not enqueue its exact state event",
            ),
        )
    } else {
        (
            InjectedStateChangeFailure::Transport,
            OutputShutdownIssue::new(
                OutputShutdownIssueKind::InjectedControlClose,
                "renderer rejected the exact injected state barrier",
            ),
        )
    }
}

fn wait_for_boundary(
    owner: &mut InjectedOutputRenderOwner,
    boundary: ControlStateBoundary,
    request_gate: &InjectedLifecycleRequestGate,
    output_events: &AudioOutputEventWatcher,
    issue: Option<OutputShutdownIssue>,
) -> Result<Option<InjectedLifecycleRequest>, InjectedControlError> {
    loop {
        if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
            owner.cancel_state_transition(boundary)?;
            return Ok(Some(request));
        }
        match owner.state_boundary_ready(boundary) {
            Ok(true) => return Ok(None),
            Ok(false) | Err(InjectedControlError::Contended) => {
                std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
            }
            Err(error) => return Err(error),
        }
    }
}

fn wait_for_state_ack(
    owner: &InjectedOutputRenderOwner,
    submitted: SubmittedControlStateTransition,
    request_gate: &InjectedLifecycleRequestGate,
    output_events: &AudioOutputEventWatcher,
    issue: Option<OutputShutdownIssue>,
) -> Result<(), StateAckFailure> {
    loop {
        if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
            return Err(StateAckFailure::Superseded(request));
        }
        match owner.observe_state_transition(submitted) {
            ControlStateObservation::Applied => return Ok(()),
            ControlStateObservation::Terminal(snapshot) => {
                let (failure, issue) = terminal_state_failure(snapshot);
                request_gate.latch(OutputShutdownMode::Silent);
                return Err(StateAckFailure::Terminal {
                    request: InjectedLifecycleRequest {
                        mode: OutputShutdownMode::Silent,
                        endpoint_death: output_events.death_reason(),
                        issue: Some(issue),
                    },
                    failure,
                });
            }
            ControlStateObservation::Pending => {
                match owner.state_transition_wake_receiver().try_recv() {
                    Ok(()) => {}
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
                    }
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        request_gate.latch(OutputShutdownMode::Silent);
                        return Err(StateAckFailure::Terminal {
                            request: InjectedLifecycleRequest {
                                mode: OutputShutdownMode::Silent,
                                endpoint_death: output_events.death_reason(),
                                issue: Some(OutputShutdownIssue::new(
                                    OutputShutdownIssueKind::InjectedControlClose,
                                    "renderer state-barrier publisher retired before acknowledgement",
                                )),
                            },
                            failure: InjectedStateChangeFailure::Transport,
                        });
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_state_command(
    owner: &mut InjectedOutputRenderOwner,
    endpoint: &mut dyn RunningAudioOutput,
    command: InjectedStateChangeCommand,
    request_gate: &InjectedLifecycleRequestGate,
    output_events: &AudioOutputEventWatcher,
    graph: &mut WorkerGraphState,
    endpoint_state: &mut WorkerEndpointState,
    issue: Option<OutputShutdownIssue>,
) -> StateCommandDrive {
    let InjectedStateChangeCommand {
        target,
        admission,
        completer,
    } = command;
    if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
        drop(admission);
        completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
        return StateCommandDrive::Shutdown(request);
    }

    let transition = match target {
        crate::context::AudioContextState::Suspended => GraphLifecycleTransition::Suspend,
        crate::context::AudioContextState::Running => GraphLifecycleTransition::Resume,
        crate::context::AudioContextState::Closed => {
            drop(admission);
            completer.complete(InjectedStateChangeOutcome::Closed);
            return StateCommandDrive::Continue;
        }
    };
    let graph_matches = matches!(
        (target, *graph),
        (
            crate::context::AudioContextState::Suspended,
            WorkerGraphState::Suspended
        ) | (
            crate::context::AudioContextState::Running,
            WorkerGraphState::Running
        )
    );
    let endpoint_matches = matches!(
        (target, *endpoint_state),
        (
            crate::context::AudioContextState::Suspended,
            WorkerEndpointState::Suspended
        ) | (
            crate::context::AudioContextState::Running,
            WorkerEndpointState::Running
        )
    );
    if graph_matches && endpoint_matches {
        drop(admission);
        completer.complete(InjectedStateChangeOutcome::Unchanged);
        return StateCommandDrive::Continue;
    }

    // Resume the native endpoint before exposing staged graph work. An initially-suspended graph
    // may still have a Running endpoint only during bootstrap, which skips this redundant call.
    if transition == GraphLifecycleTransition::Resume && !endpoint_matches {
        match panic::catch_unwind(AssertUnwindSafe(|| endpoint.resume())) {
            Ok(Ok(())) => *endpoint_state = WorkerEndpointState::Running,
            Ok(Err(error)) => {
                drop(admission);
                completer.complete(InjectedStateChangeOutcome::Failed(
                    InjectedStateChangeFailure::EndpointUncertain,
                ));
                return StateCommandDrive::EndpointUncertain(issue_from_audio_error(
                    OutputShutdownIssueKind::EndpointStateTransitionFailed,
                    error,
                ));
            }
            Err(payload) => {
                quarantine_panic_payload(payload);
                drop(admission);
                completer.complete(InjectedStateChangeOutcome::Failed(
                    InjectedStateChangeFailure::EndpointUncertain,
                ));
                return StateCommandDrive::EndpointUncertain(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EndpointMethodPanicked,
                    "injected endpoint resume panicked",
                ));
            }
        }
        if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
            drop(admission);
            completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
            return StateCommandDrive::Shutdown(request);
        }
    }

    let mut submitted = None;
    if !graph_matches {
        let boundary = loop {
            match owner.try_begin_state_transition(transition) {
                Ok(BeginControlStateTransition::Boundary(boundary)) => break boundary,
                Ok(BeginControlStateTransition::AlreadyPlaced) => {
                    drop(admission);
                    completer.complete(InjectedStateChangeOutcome::Failed(
                        InjectedStateChangeFailure::Transport,
                    ));
                    request_gate.latch(OutputShutdownMode::Silent);
                    return StateCommandDrive::Shutdown(InjectedLifecycleRequest {
                        mode: OutputShutdownMode::Silent,
                        endpoint_death: output_events.death_reason(),
                        issue: Some(OutputShutdownIssue::new(
                            OutputShutdownIssueKind::InjectedControlClose,
                            "worker graph state disagreed with exact transport placement",
                        )),
                    });
                }
                Err(InjectedControlError::Contended) => {
                    if let Some(request) =
                        request_from_latch(request_gate, output_events, issue.clone())
                    {
                        drop(admission);
                        completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
                        return StateCommandDrive::Shutdown(request);
                    }
                    std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
                }
                Err(_) => {
                    drop(admission);
                    completer.complete(InjectedStateChangeOutcome::Failed(
                        InjectedStateChangeFailure::Transport,
                    ));
                    request_gate.latch(OutputShutdownMode::Silent);
                    return StateCommandDrive::Shutdown(InjectedLifecycleRequest {
                        mode: OutputShutdownMode::Silent,
                        endpoint_death: output_events.death_reason(),
                        issue: Some(OutputShutdownIssue::new(
                            OutputShutdownIssueKind::InjectedControlClose,
                            "injected state transition could not establish its boundary",
                        )),
                    });
                }
            }
        };
        match wait_for_boundary(owner, boundary, request_gate, output_events, issue.clone()) {
            Ok(Some(request)) => {
                drop(admission);
                completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
                return StateCommandDrive::Shutdown(request);
            }
            Ok(None) => {}
            Err(_) => {
                drop(admission);
                completer.complete(InjectedStateChangeOutcome::Failed(
                    InjectedStateChangeFailure::Transport,
                ));
                request_gate.latch(OutputShutdownMode::Silent);
                return StateCommandDrive::Shutdown(InjectedLifecycleRequest {
                    mode: OutputShutdownMode::Silent,
                    endpoint_death: output_events.death_reason(),
                    issue: Some(OutputShutdownIssue::new(
                        OutputShutdownIssueKind::InjectedControlClose,
                        "injected state reservation drain failed",
                    )),
                });
            }
        }

        if transition == GraphLifecycleTransition::Resume {
            loop {
                if let Some(request) =
                    request_from_latch(request_gate, output_events, issue.clone())
                {
                    let _ = owner.cancel_state_transition(boundary);
                    drop(admission);
                    completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
                    return StateCommandDrive::Shutdown(request);
                }
                match owner.try_flush_state_transition(boundary) {
                    Ok(flush) if flush.remaining_staged == 0 => break,
                    Ok(_) | Err(InjectedControlError::Contended) => {
                        std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
                    }
                    Err(_) => {
                        drop(admission);
                        completer.complete(InjectedStateChangeOutcome::Failed(
                            InjectedStateChangeFailure::Transport,
                        ));
                        request_gate.latch(OutputShutdownMode::Silent);
                        return StateCommandDrive::Shutdown(InjectedLifecycleRequest {
                            mode: OutputShutdownMode::Silent,
                            endpoint_death: output_events.death_reason(),
                            issue: Some(OutputShutdownIssue::new(
                                OutputShutdownIssueKind::InjectedControlClose,
                                "injected staged Resume flush failed",
                            )),
                        });
                    }
                }
            }
        }
        let exact = match owner.try_submit_state_transition(boundary) {
            Ok(exact) => exact,
            Err(_) => {
                drop(admission);
                completer.complete(InjectedStateChangeOutcome::Failed(
                    InjectedStateChangeFailure::Transport,
                ));
                request_gate.latch(OutputShutdownMode::Silent);
                return StateCommandDrive::Shutdown(InjectedLifecycleRequest {
                    mode: OutputShutdownMode::Silent,
                    endpoint_death: output_events.death_reason(),
                    issue: Some(OutputShutdownIssue::new(
                        OutputShutdownIssueKind::InjectedControlClose,
                        "injected state barrier submission failed",
                    )),
                });
            }
        };
        submitted = Some(exact);
    }
    // Admission ends only after the exact boundary is published (or proved unnecessary). Close
    // may now seal without waiting on a live callback or endpoint method.
    drop(admission);

    if let Some(exact) = submitted {
        if let Err(failure) =
            wait_for_state_ack(owner, exact, request_gate, output_events, issue.clone())
        {
            return match failure {
                StateAckFailure::Superseded(request) => {
                    completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
                    StateCommandDrive::Shutdown(request)
                }
                StateAckFailure::Terminal { request, failure } => {
                    completer.complete(InjectedStateChangeOutcome::Failed(failure));
                    StateCommandDrive::Shutdown(request)
                }
            };
        }
        *graph = match transition {
            GraphLifecycleTransition::Suspend => WorkerGraphState::Suspended,
            GraphLifecycleTransition::Resume => WorkerGraphState::Running,
            GraphLifecycleTransition::Close => unreachable!(),
        };
    }

    // Suspend mirrors legacy order: graph/state/event acknowledgement first, native endpoint
    // suspension second. Any endpoint Err is conservatively uncertain under the public trait.
    if transition == GraphLifecycleTransition::Suspend && !endpoint_matches {
        if let Some(request) = request_from_latch(request_gate, output_events, issue.clone()) {
            completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
            return StateCommandDrive::Shutdown(request);
        }
        match panic::catch_unwind(AssertUnwindSafe(|| endpoint.suspend())) {
            Ok(Ok(())) => *endpoint_state = WorkerEndpointState::Suspended,
            Ok(Err(error)) => {
                completer.complete(InjectedStateChangeOutcome::Failed(
                    InjectedStateChangeFailure::EndpointUncertain,
                ));
                return StateCommandDrive::EndpointUncertain(issue_from_audio_error(
                    OutputShutdownIssueKind::EndpointStateTransitionFailed,
                    error,
                ));
            }
            Err(payload) => {
                quarantine_panic_payload(payload);
                completer.complete(InjectedStateChangeOutcome::Failed(
                    InjectedStateChangeFailure::EndpointUncertain,
                ));
                return StateCommandDrive::EndpointUncertain(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EndpointMethodPanicked,
                    "injected endpoint suspend panicked",
                ));
            }
        }
        if let Some(request) = request_from_latch(request_gate, output_events, issue) {
            completer.complete(InjectedStateChangeOutcome::SupersededByShutdown);
            return StateCommandDrive::Shutdown(request);
        }
    }
    completer.complete(InjectedStateChangeOutcome::Applied);
    StateCommandDrive::Continue
}

fn run_injected_partial(resources: &mut InjectedPartialResources) -> OutputShutdownOutcome {
    let event_loop = FailClosedEventLoop::new(resources.event_loop.take().unwrap());
    let output_events = resources.output_events.take().unwrap();
    let mut request = InjectedLifecycleRequest {
        mode: OutputShutdownMode::Silent,
        endpoint_death: output_events.death_reason(),
        issue: None,
    };
    let owner = resources.owner.take().unwrap();
    let prepared = prepare_injected_close(owner, &mut request, &output_events, None);
    let future = resources.endpoint_shutdown.take().unwrap();
    match prepared {
        PreparedInjectedClose::Ready(ready) => {
            finish_ready_with_future(ready, future, output_events, event_loop, request)
        }
        PreparedInjectedClose::NoProof { render, issue } => finish_without_graph_proof_with_future(
            render,
            future,
            output_events,
            event_loop,
            request,
            issue,
        ),
        PreparedInjectedClose::EndpointUncertain { render, issue } => {
            wait_for_render_quiescence(&render);
            drop(render);
            std::mem::forget(future);
            drop(output_events);
            finish_unconfirmed_events(event_loop, issue)
        }
    }
}

enum PreparedInjectedClose {
    Ready(ReadyForInjectedPhysicalReclaim),
    EndpointUncertain {
        render: AudioRenderOwner,
        issue: OutputShutdownIssue,
    },
    NoProof {
        render: AudioRenderOwner,
        issue: OutputShutdownIssue,
    },
}

fn prepare_injected_close(
    mut owner: InjectedOutputRenderOwner,
    request: &mut InjectedLifecycleRequest,
    output_events: &AudioOutputEventWatcher,
    mut endpoint: Option<&mut dyn RunningAudioOutput>,
) -> PreparedInjectedClose {
    let retirement = loop {
        match owner.try_begin_close() {
            Ok(retirement) => break retirement,
            Err(failure) if failure.error == InjectedControlError::Contended => {
                owner = failure.into_owner();
                if let Some(reason) = output_events.death_reason() {
                    request.mode = OutputShutdownMode::Silent;
                    request.endpoint_death = Some(reason);
                }
                std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
            }
            Err(failure) => {
                return PreparedInjectedClose::NoProof {
                    render: failure.quarantine_into_render_owner(),
                    issue: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::InjectedControlClose,
                        "injected admission seal failed; graph retirement was quarantined",
                    ),
                };
            }
        }
    };

    let drained = retirement.retire_and_wait();
    let pending = match drained.seal_and_finish() {
        Ok(pending) => pending,
        Err(Ok(failure)) => {
            return PreparedInjectedClose::NoProof {
                render: failure.quarantine_into_render_owner(),
                issue: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::InjectedNodeLifetime,
                    "injected node registry could not seal after admission drain",
                ),
            };
        }
        Err(Err(failure)) => {
            request.mode = OutputShutdownMode::Silent;
            request.issue = Some(OutputShutdownIssue::new(
                OutputShutdownIssueKind::InjectedControlClose,
                "injected Close submission failed after admission drain",
            ));
            return match failure.retire_payloads_for_silent() {
                Ok(ready) => PreparedInjectedClose::Ready(ready),
                Err(failure) => PreparedInjectedClose::NoProof {
                    render: failure.quarantine_into_render_owner(),
                    issue: OutputShutdownIssue::new(
                        OutputShutdownIssueKind::InjectedWholeGraphQuarantined,
                        "staged injected payload destructor panicked",
                    ),
                },
            };
        }
    };

    let sealed = match pending.retire_payloads() {
        Ok(sealed) => sealed,
        Err(failure) => {
            return PreparedInjectedClose::NoProof {
                render: failure.quarantine_into_render_owner(),
                issue: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::InjectedWholeGraphQuarantined,
                    "staged injected payload destructor panicked",
                ),
            };
        }
    };

    if request.mode == OutputShutdownMode::Silent {
        return PreparedInjectedClose::Ready(sealed.into_silent_reclaim());
    }

    // Close is a renderer barrier. A suspended endpoint cannot observe it until callbacks resume,
    // so make one caught, idempotent resume request after all admissions are sealed and before
    // waiting for the exact Applied acknowledgement.
    let resume = endpoint
        .take()
        .map(|endpoint| panic::catch_unwind(AssertUnwindSafe(|| endpoint.resume())));
    match resume {
        Some(Ok(Ok(()))) => {}
        Some(Ok(Err(error))) => {
            request.mode = OutputShutdownMode::Silent;
            return PreparedInjectedClose::EndpointUncertain {
                render: sealed.quarantine_into_render_owner(),
                issue: issue_from_audio_error(
                    OutputShutdownIssueKind::EndpointStateTransitionFailed,
                    error,
                ),
            };
        }
        Some(Err(payload)) => {
            quarantine_panic_payload(payload);
            request.mode = OutputShutdownMode::Silent;
            let issue = OutputShutdownIssue::new(
                OutputShutdownIssueKind::EndpointMethodPanicked,
                "injected endpoint resume panicked while making Close observable",
            );
            return PreparedInjectedClose::EndpointUncertain {
                render: sealed.quarantine_into_render_owner(),
                issue,
            };
        }
        None => {
            request.mode = OutputShutdownMode::Silent;
            request.issue = Some(OutputShutdownIssue::new(
                OutputShutdownIssueKind::InjectedControlClose,
                "graceful injected Close has no running endpoint to observe it",
            ));
            return PreparedInjectedClose::Ready(sealed.into_silent_reclaim());
        }
    }
    wait_for_graceful_close(sealed, request, output_events)
}

fn wait_for_graceful_close(
    mut sealed: SealedInjectedOutput,
    request: &mut InjectedLifecycleRequest,
    output_events: &AudioOutputEventWatcher,
) -> PreparedInjectedClose {
    loop {
        if let Some(reason) = output_events.death_reason() {
            request.mode = OutputShutdownMode::Silent;
            request.endpoint_death = Some(reason);
            return PreparedInjectedClose::Ready(sealed.into_silent_reclaim());
        }
        match sealed.try_observe_close() {
            InjectedCloseObservation::Applied(ready) => {
                return PreparedInjectedClose::Ready(ready);
            }
            InjectedCloseObservation::Terminal(terminal) => {
                request.mode = OutputShutdownMode::Silent;
                request.issue = Some(OutputShutdownIssue::new(
                    OutputShutdownIssueKind::InjectedControlClose,
                    "renderer rejected the exact injected Close barrier",
                ));
                return PreparedInjectedClose::Ready(terminal.into_silent_reclaim());
            }
            InjectedCloseObservation::Pending(pending) => {
                match pending.close_wake_receiver().try_recv() {
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        request.mode = OutputShutdownMode::Silent;
                        request.issue = Some(OutputShutdownIssue::new(
                            OutputShutdownIssueKind::InjectedControlClose,
                            "injected Close publisher retired before applying the barrier",
                        ));
                        return PreparedInjectedClose::Ready(pending.into_silent_reclaim());
                    }
                    Ok(()) => {}
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
                    }
                }
                sealed = pending;
            }
        }
    }
}

fn wait_for_render_quiescence(render: &AudioRenderOwner) {
    render.begin_shutdown();
    while !render.callback_producer_quiescent() {
        std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
    }
}

fn wait_for_ready_render_quiescence(ready: &ReadyForInjectedPhysicalReclaim) {
    ready.begin_render_shutdown();
    while !ready.render_callback_quiescent() {
        std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
    }
}

fn finish_running_ready(
    endpoint: Box<dyn RunningAudioOutput>,
    ready: ReadyForInjectedPhysicalReclaim,
    output_events: AudioOutputEventWatcher,
    event_loop: FailClosedEventLoop,
    request: InjectedLifecycleRequest,
) -> OutputShutdownOutcome {
    ready.begin_render_shutdown();
    let future = match panic::catch_unwind(AssertUnwindSafe(|| endpoint.shutdown())) {
        Ok(future) => future,
        Err(payload) => {
            quarantine_panic_payload(payload);
            wait_for_ready_render_quiescence(&ready);
            drop(ready);
            return finish_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EndpointMethodPanicked,
                    "injected endpoint shutdown method panicked",
                ),
            );
        }
    };
    finish_ready_with_future(ready, future, output_events, event_loop, request)
}

fn finish_running_without_graph_proof(
    endpoint: Box<dyn RunningAudioOutput>,
    render: AudioRenderOwner,
    output_events: AudioOutputEventWatcher,
    event_loop: FailClosedEventLoop,
    request: InjectedLifecycleRequest,
    issue: OutputShutdownIssue,
) -> OutputShutdownOutcome {
    render.begin_shutdown();
    let future = match panic::catch_unwind(AssertUnwindSafe(|| endpoint.shutdown())) {
        Ok(future) => future,
        Err(payload) => {
            quarantine_panic_payload(payload);
            wait_for_render_quiescence(&render);
            drop(render);
            return quarantine_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EndpointMethodPanicked,
                    "injected endpoint shutdown method panicked",
                ),
            );
        }
    };
    finish_without_graph_proof_with_future(
        render,
        future,
        output_events,
        event_loop,
        request,
        issue,
    )
}

fn finish_ready_with_future(
    ready: ReadyForInjectedPhysicalReclaim,
    future: AudioOutputEndpointShutdown,
    output_events: AudioOutputEventWatcher,
    event_loop: FailClosedEventLoop,
    mut request: InjectedLifecycleRequest,
) -> OutputShutdownOutcome {
    ready.begin_render_shutdown();
    let mut base_request = LifecycleRequest {
        mode: request.mode,
        endpoint_death: request.endpoint_death,
    };
    match poll_injected_endpoint_shutdown(future, &output_events, &mut base_request) {
        EndpointPollOutcome::Confirmed => {}
        EndpointPollOutcome::Quarantined { future, failure } => {
            std::mem::forget(future);
            wait_for_ready_render_quiescence(&ready);
            drop(ready);
            return finish_unconfirmed_events(event_loop, failure);
        }
    }
    request.mode = base_request.mode;
    request.endpoint_death = base_request.endpoint_death;
    // Endpoint completion can race a callback already in flight. The death latch is authoritative
    // and must be promoted before a physically clean reclaim can be classified as graceful.
    promote_late_endpoint_death(&output_events, &mut request);

    match ready.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new()) {
        InjectedRenderReclaimOutcome::Reclaimed(graph) => {
            finish_reclaimed_graph(graph, output_events, event_loop, request)
        }
        InjectedRenderReclaimOutcome::Degraded {
            registry, error, ..
        } => {
            drop(registry);
            drop(output_events);
            finish_unconfirmed_events(
                event_loop,
                issue_from_audio_error(OutputShutdownIssueKind::RenderReclaimDegraded, error),
            )
        }
        InjectedRenderReclaimOutcome::CallbackRetained(ready) => {
            wait_for_ready_render_quiescence(&ready);
            drop(ready);
            drop(output_events);
            finish_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::CallbackRetained,
                    "endpoint reported success while retaining injected callback",
                ),
            )
        }
    }
}

fn finish_without_graph_proof_with_future(
    render: AudioRenderOwner,
    future: AudioOutputEndpointShutdown,
    output_events: AudioOutputEventWatcher,
    event_loop: FailClosedEventLoop,
    mut request: InjectedLifecycleRequest,
    issue: OutputShutdownIssue,
) -> OutputShutdownOutcome {
    render.begin_shutdown();
    request.mode = OutputShutdownMode::Silent;
    let mut base_request = LifecycleRequest {
        mode: OutputShutdownMode::Silent,
        endpoint_death: request.endpoint_death,
    };
    match poll_injected_endpoint_shutdown(future, &output_events, &mut base_request) {
        EndpointPollOutcome::Confirmed => {}
        EndpointPollOutcome::Quarantined { future, failure } => {
            std::mem::forget(future);
            wait_for_render_quiescence(&render);
            drop(render);
            drop(output_events);
            return quarantine_unconfirmed_events(event_loop, failure);
        }
    }
    request.endpoint_death = base_request.endpoint_death;
    promote_late_endpoint_death(&output_events, &mut request);

    match render.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new()) {
        Ok(Ok(())) => {
            drop(output_events);
            quarantine_unconfirmed_events(event_loop, issue)
        }
        Ok(Err(error)) => {
            drop(output_events);
            quarantine_unconfirmed_events(
                event_loop,
                issue_from_audio_error(OutputShutdownIssueKind::RenderReclaimDegraded, error),
            )
        }
        Err(render) => {
            wait_for_render_quiescence(&render);
            drop(render);
            drop(output_events);
            quarantine_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::CallbackRetained,
                    "endpoint reported success while retaining injected callback",
                ),
            )
        }
    }
}

fn finish_reclaimed_graph(
    mut graph: ReclaimedInjectedGraph,
    output_events: AudioOutputEventWatcher,
    event_loop: FailClosedEventLoop,
    mut request: InjectedLifecycleRequest,
) -> OutputShutdownOutcome {
    promote_late_endpoint_death(&output_events, &mut request);
    let retired = loop {
        match graph.try_retire_nodes() {
            InjectedNodeRetireOutcome::Retired(retired) => break retired,
            InjectedNodeRetireOutcome::Retry(retry) => {
                graph = retry;
                std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
            }
            InjectedNodeRetireOutcome::Terminal(terminal) => {
                drop(terminal);
                drop(output_events);
                return finish_unconfirmed_events(
                    event_loop,
                    OutputShutdownIssue::new(
                        OutputShutdownIssueKind::InjectedWholeGraphQuarantined,
                        "exact whole-graph proof was rejected during node retirement",
                    ),
                );
            }
        }
    };

    let _producers = EventProducersQuiesced::after_injected_graph_retired(&retired);
    let node_report = retired.nodes();
    let control = retired.control_degradation();
    let degraded = node_report.cleanup_panicked
        || node_report.cleanup_rejected
        || node_report.reclaim_brand_mismatch
        || node_report.connection_registry.protocol_failed
        || node_report.connection_registry.serializer_poison_recovered
        || node_report.connection_registry.ownership_mismatch
        || node_report.pre_retirement_degraded
        || control.transport_poison_recovered
        || control.capacity_worker_panicked
        || control.prior_transport_failure;
    if request.mode == OutputShutdownMode::Graceful && !retired.close_applied() {
        request.mode = OutputShutdownMode::Silent;
    }
    if degraded || request.issue.is_some() {
        request.mode = OutputShutdownMode::Silent;
    }
    let mut issue = request.issue.take().or_else(|| {
        degraded.then(|| {
            OutputShutdownIssue::new(
                OutputShutdownIssueKind::InjectedNodeLifetime,
                "injected whole-graph node retirement completed with degradation",
            )
        })
    });
    // Keep the watcher alive through the final classification: a callback death published while
    // endpoint shutdown/reclaim completed must never leave a Graceful receipt behind.
    promote_late_endpoint_death(&output_events, &mut request);
    let request_graceful = request.mode == OutputShutdownMode::Graceful && issue.is_none();
    drop(output_events);
    let retirement = match panic::catch_unwind(AssertUnwindSafe(|| {
        event_loop.retire_confirmed(&retired, request_graceful)
    })) {
        Ok(Ok(retirement)) => retirement,
        Ok(Err(event_loop)) => {
            return quarantine_unconfirmed_events(
                event_loop,
                OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventThreadUnretired,
                    "event-loop authority did not match the retired injected graph",
                ),
            );
        }
        Err(payload) => {
            quarantine_panic_payload(payload);
            return OutputShutdownOutcome::Unconfirmed {
                failure: OutputShutdownIssue::new(
                    OutputShutdownIssueKind::EventThreadUnretired,
                    "injected terminal event retirement panicked",
                ),
                event_issue: None,
            };
        }
    };

    let state_degraded = matches!(
        retirement.state,
        InjectedTerminalStateOutcome::MissingRenderedClosePublished
            | InjectedTerminalStateOutcome::UnexpectedAlreadyClosed
    );
    if !retirement.graceful {
        request.mode = OutputShutdownMode::Silent;
    }
    if state_degraded {
        request.mode = OutputShutdownMode::Silent;
        issue.get_or_insert_with(|| {
            OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventDeliveryDegraded,
                "injected terminal state did not match its rendered Close proof",
            )
        });
    }
    let expected_exit = if retirement.graceful {
        EventLoopExit::Graceful
    } else {
        EventLoopExit::TerminalClosed
    };
    let cleanup_panic = retirement.cleanup_panic;
    let mut event_retirement = event_retirement_from_join(retirement.joined, expected_exit);
    if let Some(payload) = cleanup_panic {
        quarantine_panic_payload(payload);
        event_retirement.issue.get_or_insert_with(|| {
            OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventDeliveryDegraded,
                "exact event callback cleanup panicked after event-thread retirement",
            )
        });
    }
    if !event_retirement.retired {
        return OutputShutdownOutcome::Unconfirmed {
            failure: OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventThreadUnretired,
                "injected lifecycle worker could not retire its event thread",
            ),
            event_issue: event_retirement.issue,
        };
    }
    OutputShutdownOutcome::Confirmed(OutputShutdownReport {
        mode: request.mode,
        endpoint_death: request.endpoint_death,
        reclaim_issue: issue,
        event_issue: event_retirement.issue,
    })
}

fn finish_unconfirmed_events(
    event_loop: FailClosedEventLoop,
    failure: OutputShutdownIssue,
) -> OutputShutdownOutcome {
    // No unconfirmed path owns the exact graph+node retirement proof branded to this consumer.
    // Quarantine its join authority instead of exposing a stop operation which could race a
    // substituted or still-live producer.
    quarantine_unconfirmed_events(event_loop, failure)
}

fn quarantine_unconfirmed_events(
    event_loop: FailClosedEventLoop,
    failure: OutputShutdownIssue,
) -> OutputShutdownOutcome {
    drop(event_loop); // FailClosedEventLoop forgets without requesting stop.
    OutputShutdownOutcome::Unconfirmed {
        failure,
        event_issue: Some(OutputShutdownIssue::new(
            OutputShutdownIssueKind::EventThreadUnretired,
            "event-loop authority quarantined because producer quiescence was not proven",
        )),
    }
}

#[cfg(test)]
#[path = "injected_tests.rs"]
mod tests;
