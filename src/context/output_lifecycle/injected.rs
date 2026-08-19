//! Private lifecycle worker for one exactly bound injected output.

use std::panic::{self, AssertUnwindSafe};
use std::thread::JoinHandle;

use super::*;
use crate::context::injected_control::InjectedControlError;
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
use crate::output::{AudioOutputEventSink, AudioOutputStartFailure, PreparedAudioOutput};

const OPEN_DRIVE_BUDGET: usize = 32;

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

enum InjectedLifecycleCommand {
    Graceful,
    Silent,
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
    command_send: Sender<InjectedLifecycleCommand>,
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

    pub(crate) fn shutdown_gracefully(mut self) -> OutputShutdownReceipt {
        self.requested = true;
        let _ = self
            .command_send
            .try_send(InjectedLifecycleCommand::Graceful);
        self.receipt.clone()
    }

    pub(crate) fn shutdown_silently(mut self) -> OutputShutdownReceipt {
        self.requested = true;
        let _ = self.command_send.try_send(InjectedLifecycleCommand::Silent);
        self.receipt.clone()
    }
}

impl Drop for InjectedOutputLifecycleController {
    fn drop(&mut self) {
        if !self.requested {
            let _ = self.command_send.try_send(InjectedLifecycleCommand::Silent);
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
pub(crate) fn start_injected_output(
    prepared: Box<dyn PreparedAudioOutput>,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
    start_injected_output_with_spawner(prepared, renderer, events, output_events, &ThreadSpawner)
}

#[allow(clippy::too_many_arguments)]
fn start_injected_output_with_spawner(
    prepared: Box<dyn PreparedAudioOutput>,
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
    spawner: &dyn LifecycleWorkerSpawner,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
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

    let format = match panic::catch_unwind(AssertUnwindSafe(|| prepared.config().format())) {
        Ok(format) => format,
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
    let resources = InjectedRunningResources {
        endpoint: Some(endpoint),
        owner: Some(owner),
        output_events: Some(output_events),
        event_loop: Some(event_loop),
    };
    let (command_send, command_recv) = crossbeam_channel::bounded(1);
    let (bootstrap_send, bootstrap_recv) = crossbeam_channel::bounded(1);
    let (completer, receipt) = shutdown_receipt_pair();
    let job = Box::new(move || injected_running_worker(bootstrap_recv, command_recv, completer));
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
        command_send,
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
    commands: Receiver<InjectedLifecycleCommand>,
    completer: OutputShutdownCompleter,
) {
    let outcome = match bootstrap.recv() {
        Ok(mut resources) => panic::catch_unwind(AssertUnwindSafe(|| {
            run_injected_running(&mut resources, &commands)
        }))
        .unwrap_or_else(|payload| {
            quarantine_panic_payload(payload);
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
    commands: &Receiver<InjectedLifecycleCommand>,
) -> OutputShutdownOutcome {
    let mut owner = resources.owner.take().unwrap();
    let output_events = resources.output_events.as_ref().unwrap();
    let mut request = wait_for_injected_request(&mut owner, commands, output_events);
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

fn wait_for_injected_request(
    owner: &mut InjectedOutputRenderOwner,
    commands: &Receiver<InjectedLifecycleCommand>,
    output_events: &AudioOutputEventWatcher,
) -> InjectedLifecycleRequest {
    let mut issue = None;
    let mut drive = true;
    loop {
        if let Some(request) = probe_injected_request(commands, output_events, issue.clone()) {
            return request;
        }

        if drive {
            for _ in 0..OPEN_DRIVE_BUDGET {
                match owner.try_drive_node_lifetimes() {
                    NodeLifetimeDriveOutcome::Idle | NodeLifetimeDriveOutcome::Retry { .. } => {
                        break;
                    }
                    NodeLifetimeDriveOutcome::Quarantined { .. } => {
                        issue = Some(OutputShutdownIssue::new(
                            OutputShutdownIssueKind::InjectedNodeLifetime,
                            "injected node-lifetime driver entered quarantine",
                        ));
                        drive = false;
                        break;
                    }
                    NodeLifetimeDriveOutcome::Submitted { .. }
                    | NodeLifetimeDriveOutcome::Reconciled { .. }
                    | NodeLifetimeDriveOutcome::CloseSuperseded { .. } => {}
                }
            }
        }

        if let Some(request) = probe_injected_request(commands, output_events, issue.clone()) {
            return request;
        }
        if drive {
            let _ = owner.wait_for_node_lifetime_activity();
        } else {
            std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
        }
    }
}

fn probe_injected_request(
    commands: &Receiver<InjectedLifecycleCommand>,
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
    match commands.try_recv() {
        Ok(InjectedLifecycleCommand::Graceful) => Some(InjectedLifecycleRequest {
            mode: OutputShutdownMode::Graceful,
            endpoint_death: None,
            issue,
        }),
        Ok(InjectedLifecycleCommand::Silent)
        | Err(crossbeam_channel::TryRecvError::Disconnected) => Some(InjectedLifecycleRequest {
            mode: OutputShutdownMode::Silent,
            endpoint_death: None,
            issue,
        }),
        Err(crossbeam_channel::TryRecvError::Empty) => None,
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
            request.issue = Some(issue_from_audio_error(
                OutputShutdownIssueKind::EndpointRejectedShutdown,
                error,
            ));
            return PreparedInjectedClose::Ready(sealed.into_silent_reclaim());
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
    let retirement = event_retirement_from_join(retirement.joined, expected_exit);
    if !retirement.retired {
        return OutputShutdownOutcome::Unconfirmed {
            failure: OutputShutdownIssue::new(
                OutputShutdownIssueKind::EventThreadUnretired,
                "injected lifecycle worker could not retire its event thread",
            ),
            event_issue: retirement.issue,
        };
    }
    OutputShutdownOutcome::Confirmed(OutputShutdownReport {
        mode: request.mode,
        endpoint_death: request.endpoint_death,
        reclaim_issue: issue,
        event_issue: retirement.issue,
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
