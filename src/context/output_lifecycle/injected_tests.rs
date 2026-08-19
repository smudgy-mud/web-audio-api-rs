use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use futures::executor;

use super::*;
use crate::context::injected_control::{
    injected_control_channel, BoundInjectedRenderer, InjectedControlLifecycleOwner,
};
use crate::context::injected_ids::injected_node_id_pair;
use crate::context::injected_node_construction::{InjectedGainPayload, InjectedNodeConstructor};
use crate::context::injected_node_lifetime::injected_node_lifetime_registry;
use crate::context::{AudioContextState, InjectedContextAdmissionGate};
use crate::events::{
    injected_event_loop_pair, injected_event_loop_pair_with_setup, EventHandler, EventType,
    InjectedJoinableEventLoop,
};
use crate::message::ControlMessage;
use crate::node::{ChannelConfigInner, ChannelCountMode, ChannelInterpretation};
use crate::output::{
    AudioOutputConfig, AudioOutputDeathReason, AudioOutputErrorKind, AudioRenderCallback,
    AudioRenderFormat, AudioRenderStatus,
};
use crate::param::AudioParamInitialValue;
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
    InjectedEventDispatchSender,
};
use crate::stats::AudioStats;

const TIMEOUT: Duration = Duration::from_secs(5);

struct LifecycleFixture {
    renderer: BoundInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
    gate: InjectedContextAdmissionGate,
    constructor: Option<InjectedNodeConstructor>,
    allocator: crate::context::injected_ids::InjectedNodeIdAllocator,
    registrar: crate::context::injected_node_lifetime::InjectedNodeLifetimeRegistrar,
}

fn lifecycle_fixture() -> LifecycleFixture {
    lifecycle_fixture_with_suspension(false, false)
}

fn lifecycle_fixture_with_suspension(
    initially_suspended: bool,
    stage_hostile_payload: bool,
) -> LifecycleFixture {
    lifecycle_fixture_with_capacity(initially_suspended, stage_hostile_payload, 8)
}

fn lifecycle_fixture_with_event_pair(
    initially_suspended: bool,
    stage_hostile_payload: bool,
    capacity: usize,
    (event_send, event_loop): (InjectedEventDispatchSender, InjectedJoinableEventLoop),
) -> LifecycleFixture {
    lifecycle_fixture_inner(
        initially_suspended,
        stage_hostile_payload,
        capacity,
        (event_send, event_loop),
    )
}

fn lifecycle_fixture_with_capacity(
    initially_suspended: bool,
    stage_hostile_payload: bool,
    capacity: usize,
) -> LifecycleFixture {
    lifecycle_fixture_with_event_pair(
        initially_suspended,
        stage_hostile_payload,
        capacity,
        injected_event_loop_pair().unwrap(),
    )
}

fn lifecycle_fixture_inner(
    initially_suspended: bool,
    stage_hostile_payload: bool,
    capacity: usize,
    (event_send, event_loop): (InjectedEventDispatchSender, InjectedJoinableEventLoop),
) -> LifecycleFixture {
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) =
        injected_control_channel(gate.clone(), capacity, initially_suspended).unwrap();
    let (allocator, node_ids, graph) = injected_node_id_pair(1);
    let (registrar, bootstrap) =
        injected_node_lifetime_registry(capacity, &producer, node_ids, graph)
            .ok()
            .unwrap();
    let constructor =
        InjectedNodeConstructor::new(producer.clone(), allocator.clone(), registrar.clone())
            .ok()
            .unwrap();
    producer
        .try_commit_prevalidated_for_test(vec![ControlMessage::RegisterNode {
            id: crate::context::AudioNodeId(0),
            reclaim_id: llq::Node::new(crate::context::AudioNodeId(0)),
            node: Box::new(SilentProcessor),
            inputs: 1,
            outputs: 1,
            channel_config: ChannelConfigInner {
                count: 2,
                count_mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Speakers,
            },
        }])
        .unwrap();
    if stage_hostile_payload {
        producer
            .try_commit_prevalidated_for_test(vec![ControlMessage::RegisterNode {
                id: crate::context::AudioNodeId(7),
                reclaim_id: llq::Node::new(crate::context::AudioNodeId(7)),
                node: Box::new(PanicDropProcessor),
                inputs: 1,
                outputs: 1,
                channel_config: ChannelConfigInner {
                    count: 1,
                    count_mode: ChannelCountMode::Explicit,
                    interpretation: ChannelInterpretation::Discrete,
                },
            }])
            .unwrap();
    }
    let renderer = render_init
        .build_output_render_thread(
            bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU8::new(if initially_suspended {
                AudioContextState::Suspended as u8
            } else {
                AudioContextState::Running as u8
            })),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            event_send,
        )
        .ok()
        .unwrap()
        .bind_output_lifecycle_and_events(lifecycle, event_loop)
        .ok()
        .unwrap();
    drop(producer);
    let (events, output_events) = AudioOutputEventSink::bounded(8);
    LifecycleFixture {
        renderer,
        events,
        output_events,
        gate,
        constructor: Some(constructor),
        allocator,
        registrar,
    }
}

fn unbound_event_renderer() -> (
    BoundInjectedRenderer,
    InjectedControlLifecycleOwner,
    InjectedJoinableEventLoop,
) {
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) = injected_control_channel(gate, 8, false).unwrap();
    let (_allocator, node_ids, graph) = injected_node_id_pair(0);
    let (_registrar, bootstrap) = injected_node_lifetime_registry(8, &producer, node_ids, graph)
        .ok()
        .unwrap();
    let (event_send, event_loop) = injected_event_loop_pair().unwrap();
    let renderer = render_init
        .build_output_render_thread(
            bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU8::new(AudioContextState::Running as u8)),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            event_send,
        )
        .ok()
        .unwrap();
    drop(producer);
    (renderer, lifecycle, event_loop)
}

struct PumpControl {
    running: AtomicBool,
    release: AtomicBool,
    stop: AtomicBool,
    resume_count: AtomicUsize,
    render_count: AtomicUsize,
    shutdown_called: AtomicBool,
    resume_panics: AtomicBool,
    shutdown_behavior: AtomicU8,
    future_drop_ran: AtomicBool,
}

impl PumpControl {
    fn new(suspended: bool, released: bool) -> Arc<Self> {
        Arc::new(Self {
            running: AtomicBool::new(!suspended),
            release: AtomicBool::new(released),
            stop: AtomicBool::new(false),
            resume_count: AtomicUsize::new(0),
            render_count: AtomicUsize::new(0),
            shutdown_called: AtomicBool::new(false),
            resume_panics: AtomicBool::new(false),
            shutdown_behavior: AtomicU8::new(0),
            future_drop_ran: AtomicBool::new(false),
        })
    }
}

enum PreparedBehavior {
    Pump(Arc<PumpControl>),
    Partial,
    ConfigPanicOnce(Arc<AtomicBool>),
    Panic,
    Retain(Arc<Mutex<Option<AudioRenderCallback>>>),
    LateDeath(Arc<AtomicBool>),
}

struct TestPrepared {
    config: AudioOutputConfig,
    behavior: PreparedBehavior,
}

impl TestPrepared {
    fn new(format: AudioRenderFormat, behavior: PreparedBehavior) -> Self {
        Self {
            config: AudioOutputConfig::new(format, "test", 0.).unwrap(),
            behavior,
        }
    }
}

struct PumpEndpoint {
    control: Arc<PumpControl>,
    done: futures_channel::oneshot::Receiver<()>,
    join: Option<JoinHandle<()>>,
}

impl RunningAudioOutput for PumpEndpoint {
    fn resume(&mut self) -> Result<(), AudioOutputError> {
        if self.control.resume_panics.load(Ordering::Acquire) {
            panic!("test endpoint resume panic");
        }
        self.control.resume_count.fetch_add(1, Ordering::AcqRel);
        self.control.running.store(true, Ordering::Release);
        Ok(())
    }

    fn suspend(&mut self) -> Result<(), AudioOutputError> {
        self.control.running.store(false, Ordering::Release);
        Ok(())
    }

    fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        self.control.shutdown_called.store(true, Ordering::Release);
        self.control.stop.store(true, Ordering::Release);
        if self.control.shutdown_behavior.load(Ordering::Acquire) == 3 {
            panic!("forced endpoint shutdown method panic");
        }
        let done = std::mem::replace(&mut self.done, futures_channel::oneshot::channel().1);
        let join = self.join.take().unwrap();
        let behavior = self.control.shutdown_behavior.load(Ordering::Acquire);
        if behavior != 0 {
            return AudioOutputEndpointShutdown::from_future(HostileEndpointFuture {
                behavior,
                control: Arc::clone(&self.control),
                _done: done,
                _join: join,
            });
        }
        AudioOutputEndpointShutdown::from_future(async move {
            done.await.map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "test render thread exited without acknowledgement",
                )
            })?;
            join.join().map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "test render thread panicked",
                )
            })?;
            Ok(())
        })
    }
}

struct RetainingEndpoint {
    callback: Arc<Mutex<Option<AudioRenderCallback>>>,
}

struct LateDeathEndpoint {
    callback: Option<AudioRenderCallback>,
    events: AudioOutputEventSink,
    close_rendered: Arc<AtomicBool>,
}

impl RunningAudioOutput for LateDeathEndpoint {
    fn resume(&mut self) -> Result<(), AudioOutputError> {
        let mut output = [0.; 256];
        let _ = self
            .callback
            .as_mut()
            .unwrap()
            .render_interleaved_f32(&mut output);
        self.close_rendered.store(true, Ordering::Release);
        Ok(())
    }

    fn suspend(&mut self) -> Result<(), AudioOutputError> {
        Ok(())
    }

    fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        let callback = self.callback.take().unwrap();
        let events = self.events.clone();
        AudioOutputEndpointShutdown::from_future(async move {
            // Publish death in the same endpoint-completion turn, after the lifecycle worker's
            // pre-poll probe but before it can classify a clean physical reclaim.
            drop(callback);
            let _ = events.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
            Ok(())
        })
    }
}

impl RunningAudioOutput for RetainingEndpoint {
    fn resume(&mut self) -> Result<(), AudioOutputError> {
        let mut callback = self.callback.lock().unwrap();
        let mut output = [0.; 256];
        let _ = callback
            .as_mut()
            .unwrap()
            .render_interleaved_f32(&mut output);
        Ok(())
    }

    fn suspend(&mut self) -> Result<(), AudioOutputError> {
        Ok(())
    }

    fn shutdown(self: Box<Self>) -> AudioOutputEndpointShutdown {
        AudioOutputEndpointShutdown::ready(Ok(()))
    }
}

struct HostileEndpointFuture {
    behavior: u8,
    control: Arc<PumpControl>,
    _done: futures_channel::oneshot::Receiver<()>,
    _join: JoinHandle<()>,
}

impl Future for HostileEndpointFuture {
    type Output = Result<(), AudioOutputError>;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        match self.behavior {
            1 => Poll::Ready(Err(AudioOutputError::new(
                AudioOutputErrorKind::Shutdown,
                "forced endpoint future error",
            ))),
            2 => panic!("forced endpoint future poll panic"),
            _ => unreachable!("hostile future requires an explicit behavior"),
        }
    }
}

impl Drop for HostileEndpointFuture {
    fn drop(&mut self) {
        self.control.future_drop_ran.store(true, Ordering::Release);
    }
}

impl PreparedAudioOutput for TestPrepared {
    fn config(&self) -> &AudioOutputConfig {
        if let PreparedBehavior::ConfigPanicOnce(panic_once) = &self.behavior {
            if panic_once.swap(false, Ordering::AcqRel) {
                panic!("forced prepared config panic");
            }
        }
        &self.config
    }

    fn start(
        self: Box<Self>,
        mut callback: AudioRenderCallback,
        events: AudioOutputEventSink,
    ) -> Result<Box<dyn RunningAudioOutput>, AudioOutputStartFailure> {
        match self.behavior {
            PreparedBehavior::Pump(control) => {
                let thread_control = Arc::clone(&control);
                let (done_send, done) = futures_channel::oneshot::channel();
                let join = thread::spawn(move || {
                    let mut output = [0.; 256];
                    while !thread_control.stop.load(Ordering::Acquire) {
                        if thread_control.running.load(Ordering::Acquire)
                            && thread_control.release.load(Ordering::Acquire)
                        {
                            thread_control.render_count.fetch_add(1, Ordering::AcqRel);
                            if callback.render_interleaved_f32(&mut output)
                                == AudioRenderStatus::Stop
                            {
                                break;
                            }
                        }
                        thread::sleep(Duration::from_millis(1));
                    }
                    drop(callback);
                    let _ = done_send.send(());
                });
                Ok(Box::new(PumpEndpoint {
                    control,
                    done,
                    join: Some(join),
                }))
            }
            PreparedBehavior::Partial | PreparedBehavior::ConfigPanicOnce(_) => {
                Err(AudioOutputStartFailure::new(
                    AudioOutputError::new(AudioOutputErrorKind::DeviceUnavailable, "partial start"),
                    AudioOutputEndpointShutdown::from_future(async move {
                        drop(callback);
                        Ok(())
                    }),
                ))
            }
            PreparedBehavior::Panic => panic!("test prepared start panic"),
            PreparedBehavior::Retain(retained) => {
                *retained.lock().unwrap() = Some(callback);
                Ok(Box::new(RetainingEndpoint { callback: retained }))
            }
            PreparedBehavior::LateDeath(close_rendered) => Ok(Box::new(LateDeathEndpoint {
                callback: Some(callback),
                events,
                close_rendered,
            })),
        }
    }

    fn abort(self: Box<Self>) -> AudioOutputEndpointShutdown {
        AudioOutputEndpointShutdown::ready(Ok(()))
    }
}

fn format(sample_rate: f32) -> AudioRenderFormat {
    AudioRenderFormat::new(sample_rate, 2, 128).unwrap()
}

fn start(
    prepared: TestPrepared,
    fixture: LifecycleFixture,
) -> Result<InjectedOutputStart, Box<InjectedOutputStartFailure>> {
    start_injected_output(
        Box::new(prepared),
        fixture.renderer,
        fixture.events,
        fixture.output_events,
    )
}

struct SilentProcessor;

struct PanicDropProcessor;

struct DropProbeProcessor(Arc<AtomicUsize>);

impl Drop for DropProbeProcessor {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

impl AudioProcessor for DropProbeProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        false
    }
}

impl Drop for PanicDropProcessor {
    fn drop(&mut self) {
        panic!("forced staged processor destructor panic");
    }
}

impl AudioProcessor for PanicDropProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        false
    }
}

impl AudioProcessor for SilentProcessor {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        _outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        false
    }
}

fn gain_payload() -> InjectedGainPayload {
    let config = || ChannelConfigInner {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    };
    InjectedGainPayload {
        param_processor: Box::new(SilentProcessor),
        gain_processor: Box::new(SilentProcessor),
        param_channel_config: config(),
        gain_channel_config: config(),
        initial_value: AudioParamInitialValue::new(1.),
    }
}

fn gain_payload_with_drop_probe(drops: &Arc<AtomicUsize>) -> InjectedGainPayload {
    let config = || ChannelConfigInner {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    };
    InjectedGainPayload {
        param_processor: Box::new(DropProbeProcessor(Arc::clone(drops))),
        gain_processor: Box::new(DropProbeProcessor(Arc::clone(drops))),
        param_channel_config: config(),
        gain_channel_config: config(),
        initial_value: AudioParamInitialValue::new(1.),
    }
}

fn wait_receipt(receipt: OutputShutdownReceipt) -> OutputShutdownOutcome {
    let (send, recv) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = send.send(executor::block_on(receipt));
    });
    recv.recv_timeout(TIMEOUT)
        .expect("injected lifecycle receipt timed out")
}

fn confirmed(outcome: OutputShutdownOutcome) -> OutputShutdownReport {
    match outcome {
        OutputShutdownOutcome::Confirmed(report) => report,
        other => panic!("expected confirmed injected shutdown, got {other:?}"),
    }
}

#[test]
fn foreign_event_loop_bind_returns_both_exact_bundles_for_clean_reuse() {
    let (first_renderer, first_control, first_events) = unbound_event_renderer();
    let (second_renderer, second_control, second_events) = unbound_event_renderer();

    let failure = first_renderer
        .bind_output_lifecycle_and_events(first_control, second_events)
        .err()
        .expect("foreign event consumer must fail before callback publication");
    let (first_renderer, first_control, second_events) = failure.into_parts();
    let first = first_renderer
        .bind_output_lifecycle_and_events(first_control, first_events)
        .ok()
        .unwrap();
    let second = second_renderer
        .bind_output_lifecycle_and_events(second_control, second_events)
        .ok()
        .unwrap();

    for renderer in [first, second] {
        let (events, output_events) = AudioOutputEventSink::bounded(8);
        let cleanup = start_injected_output(
            Box::new(TestPrepared::new(
                format(48_000.),
                PreparedBehavior::Partial,
            )),
            renderer,
            events,
            output_events,
        )
        .ok()
        .unwrap();
        let InjectedOutputStart::Cleanup(cleanup) = cleanup else {
            panic!("partial endpoint must enter automatic cleanup");
        };
        assert_eq!(
            confirmed(wait_receipt(cleanup.receipt())).mode(),
            OutputShutdownMode::Silent
        );
    }
}

#[test]
fn suspended_graceful_close_resumes_once_and_retires_exact_graph() {
    let control = PumpControl::new(true, true);
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(Arc::clone(&control)),
        ),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(control.resume_count.load(Ordering::Acquire), 1);
    assert!(control.render_count.load(Ordering::Acquire) > 0);
    assert!(control.shutdown_called.load(Ordering::Acquire));
    assert!(report.reclaim_issue().is_none());
    assert!(report.event_issue().is_none());
}

#[test]
fn graceful_closed_handler_runs_only_after_graph_processors_are_destroyed() {
    let processor_drops = Arc::new(AtomicUsize::new(0));
    let drops_seen_by_handler = Arc::clone(&processor_drops);
    let (closed_send, closed_recv) = mpsc::sync_channel(1);
    let event_pair = injected_event_loop_pair_with_setup(move |event_loop| {
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                closed_send
                    .send(drops_seen_by_handler.load(Ordering::Acquire))
                    .unwrap();
            })),
        );
    })
    .unwrap();
    let mut fixture = lifecycle_fixture_with_event_pair(false, false, 8, event_pair);
    let constructor = fixture.constructor.take().unwrap();
    let live_gain = constructor
        .try_begin_gain()
        .unwrap()
        .commit(gain_payload_with_drop_probe(&processor_drops))
        .unwrap();
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };
    let deadline = std::time::Instant::now() + TIMEOUT;
    while constructor.applied_batch_sequence() < 2 {
        assert!(std::time::Instant::now() < deadline, "Gain batch stalled");
        thread::sleep(Duration::from_millis(1));
    }

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(closed_recv.recv_timeout(TIMEOUT).unwrap(), 2);
    assert_eq!(processor_drops.load(Ordering::Acquire), 2);
    drop(live_gain); // surviving weak registration is harmless after exact registry retirement
}

#[test]
fn explicit_silent_and_controller_drop_reclaim_cleanly_without_closed_event() {
    for explicit in [true, false] {
        let closed_count = Arc::new(AtomicUsize::new(0));
        let closed_count_for_handler = Arc::clone(&closed_count);
        let event_pair = injected_event_loop_pair_with_setup(move |event_loop| {
            event_loop.set_handler(
                EventType::StateChange,
                EventHandler::Once(Box::new(move |_| {
                    closed_count_for_handler.fetch_add(1, Ordering::AcqRel);
                })),
            );
        })
        .unwrap();
        let fixture = lifecycle_fixture_with_event_pair(false, false, 8, event_pair);
        let lifecycle = start(
            TestPrepared::new(
                format(48_000.),
                PreparedBehavior::Pump(PumpControl::new(false, true)),
            ),
            fixture,
        )
        .ok()
        .unwrap();
        let InjectedOutputStart::Running(controller) = lifecycle else {
            panic!("pumping endpoint must start running");
        };
        let receipt = if explicit {
            controller.shutdown_silently()
        } else {
            let receipt = controller.receipt();
            drop(controller);
            receipt
        };
        let report = confirmed(wait_receipt(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert_eq!(closed_count.load(Ordering::Acquire), 0);
        assert!(report.event_issue().is_none());
    }
}

#[test]
fn endpoint_death_racing_ready_forces_silent_final_classification() {
    let close_rendered = Arc::new(AtomicBool::new(false));
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::LateDeath(Arc::clone(&close_rendered)),
        ),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("late-death endpoint must start running");
    };

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert!(close_rendered.load(Ordering::Acquire));
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert_eq!(
        report.endpoint_death(),
        Some(AudioOutputDeathReason::BackendFailure)
    );
}

#[test]
fn live_stalled_close_stays_pending_until_callback_is_released() {
    let control = PumpControl::new(false, false);
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(Arc::clone(&control)),
        ),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };
    let receipt = controller.shutdown_gracefully();
    let (send, recv) = mpsc::sync_channel(1);
    let waiter = thread::spawn(move || send.send(executor::block_on(receipt)).unwrap());

    thread::sleep(Duration::from_millis(80));
    assert_eq!(recv.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert!(!control.shutdown_called.load(Ordering::Acquire));

    control.release.store(true, Ordering::Release);
    let report = confirmed(recv.recv_timeout(TIMEOUT).unwrap());
    waiter.join().unwrap();
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(control.shutdown_called.load(Ordering::Acquire));
}

#[test]
fn partial_start_physically_reclaims_without_close_ack_and_reports_silent() {
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Partial),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = lifecycle else {
        panic!("partial start must return its automatic cleanup handle");
    };
    assert_eq!(
        cleanup.startup_error().unwrap().kind(),
        AudioOutputErrorKind::DeviceUnavailable
    );
    let report = confirmed(wait_receipt(cleanup.receipt()));
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert!(report.reclaim_issue().is_none());
}

#[test]
fn renderer_format_mismatch_returns_every_resource_for_compatible_retry() {
    let failure = start(
        TestPrepared::new(format(44_100.), PreparedBehavior::Partial),
        lifecycle_fixture(),
    )
    .err()
    .unwrap();
    assert_eq!(
        failure.issue().kind(),
        OutputShutdownIssueKind::BootstrapFailed
    );
    let (_issue, parts, worker) = failure.into_parts();
    assert!(worker.is_none());
    let InjectedOutputStartFailureParts::Prepared {
        prepared,
        renderer,
        events,
        output_events,
    } = parts
    else {
        panic!("format mismatch must remain wholly prepared");
    };
    executor::block_on(prepared.abort()).unwrap();

    let lifecycle = start_injected_output(
        Box::new(TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Partial,
        )),
        renderer,
        events,
        output_events,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = lifecycle else {
        panic!("compatible retry must install the exact returned renderer");
    };
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn mismatched_output_diagnostics_return_exact_prepared_resources_for_retry() {
    let LifecycleFixture {
        renderer,
        events,
        output_events: exact_output_events,
        ..
    } = lifecycle_fixture();
    let (foreign_events, foreign_output_events) = AudioOutputEventSink::bounded(8);
    drop(foreign_events);

    let failure = start_injected_output(
        Box::new(TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Partial,
        )),
        renderer,
        events,
        foreign_output_events,
    )
    .err()
    .expect("foreign output watcher must fail before callback or worker publication");
    assert_eq!(
        failure.issue().kind(),
        OutputShutdownIssueKind::BootstrapFailed
    );
    let (_issue, parts, worker) = failure.into_parts();
    assert!(worker.is_none());
    let InjectedOutputStartFailureParts::Prepared {
        prepared,
        renderer,
        events,
        output_events: returned_foreign_watcher,
    } = parts
    else {
        panic!("diagnostic mismatch must return every exact prepared resource");
    };
    assert!(events.matches_watcher(&exact_output_events));
    assert!(!events.matches_watcher(&returned_foreign_watcher));
    drop(returned_foreign_watcher);

    let cleanup = start_injected_output(prepared, renderer, events, exact_output_events)
        .ok()
        .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = cleanup else {
        panic!("exact resources must remain operational after diagnostic mismatch");
    };
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn prepared_config_panic_returns_the_same_prepared_bundle_for_retry() {
    let panic_once = Arc::new(AtomicBool::new(true));
    let failure = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::ConfigPanicOnce(Arc::clone(&panic_once)),
        ),
        lifecycle_fixture(),
    )
    .err()
    .expect("prepared config panic must be contained before callback publication");
    assert!(!panic_once.load(Ordering::Acquire));
    assert_eq!(
        failure.issue().kind(),
        OutputShutdownIssueKind::BootstrapFailed
    );
    let (_issue, parts, worker) = failure.into_parts();
    assert!(worker.is_none());
    let InjectedOutputStartFailureParts::Prepared {
        prepared,
        renderer,
        events,
        output_events,
    } = parts
    else {
        panic!("config panic must return the exact prepared bundle");
    };

    let cleanup = start_injected_output(prepared, renderer, events, output_events)
        .ok()
        .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = cleanup else {
        panic!("one-shot panicking prepared bundle must be operational on exact retry");
    };
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn gc_spawn_failure_returns_event_bound_composite_for_exact_retry() {
    let mut fixture = lifecycle_fixture();
    fixture.renderer.fail_next_gc_spawn_for_test();
    let failure = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Partial),
        fixture,
    )
    .err()
    .expect("forced GC spawn failure must remain wholly prepared");
    assert_eq!(
        failure.issue().kind(),
        OutputShutdownIssueKind::BootstrapFailed
    );
    let (_issue, parts, worker) = failure.into_parts();
    assert!(worker.is_none());
    let InjectedOutputStartFailureParts::Prepared {
        prepared,
        renderer,
        events,
        output_events,
    } = parts
    else {
        panic!("GC spawn failure must return the event-bound prepared composite");
    };
    drop(prepared);

    let cleanup = start_injected_output(
        Box::new(TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Partial,
        )),
        renderer,
        events,
        output_events,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = cleanup else {
        panic!("exact retry must install the returned composite");
    };
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn transient_close_gate_contention_retries_the_exact_owner() {
    let fixture = lifecycle_fixture();
    let gate = fixture.gate.clone();
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(Arc::clone(&control)),
        ),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };

    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    let holder = thread::spawn(move || gate.hold_phase_lock_for_test(entered_send, release_recv));
    entered_recv.recv().unwrap();
    let receipt = controller.shutdown_gracefully();
    thread::sleep(Duration::from_millis(40));
    assert!(!control.shutdown_called.load(Ordering::Acquire));
    release_send.send(()).unwrap();
    holder.join().unwrap();

    assert_eq!(
        confirmed(wait_receipt(receipt)).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn disconnected_close_publisher_is_terminal_silent_not_a_live_stall() {
    let mut fixture = lifecycle_fixture();
    fixture
        .renderer
        .disconnect_lifecycle_on_next_render_for_test();
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert_eq!(
        report.reclaim_issue().unwrap().kind(),
        OutputShutdownIssueKind::InjectedControlClose
    );
}

#[test]
fn open_worker_automatically_services_drop_and_reuses_ids_only_after_reconcile() {
    let mut fixture = lifecycle_fixture();
    let constructor = fixture.constructor.take().unwrap();
    let allocator = fixture.allocator.clone();
    let registrar = fixture.registrar.clone();
    let constructed = constructor
        .try_begin_gain()
        .unwrap()
        .commit(gain_payload())
        .unwrap();
    let expected = [constructed.gain_id, constructed.param_id];
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };

    let deadline = std::time::Instant::now() + TIMEOUT;
    while constructor.applied_batch_sequence() < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "Gain batch was not applied"
        );
        thread::sleep(Duration::from_millis(1));
    }
    drop(constructed);
    while registrar.slot_phase_counts_for_test() != Some([8, 0, 0, 0, 0, 0]) {
        assert!(
            std::time::Instant::now() < deadline,
            "ordinary node reconciliation did not complete"
        );
        thread::sleep(Duration::from_millis(1));
    }

    let reused = allocator.try_reserve(2).unwrap();
    let mut observed = [reused.id(0), reused.id(1)];
    observed.sort_by_key(|id| id.0);
    let mut expected = expected;
    expected.sort_by_key(|id| id.0);
    assert_eq!(observed, expected);
    drop(reused);

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.endpoint_death().is_none());
}

#[test]
fn continuously_replenished_node_teardown_cannot_starve_shutdown_command() {
    let mut fixture = lifecycle_fixture_with_capacity(false, false, 128);
    let constructor = Arc::new(fixture.constructor.take().unwrap());
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(Arc::clone(&control)),
        ),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };

    let keep_producing = Arc::new(AtomicBool::new(true));
    let produced = Arc::new(AtomicUsize::new(0));
    let producer = {
        let constructor = Arc::clone(&constructor);
        let keep_producing = Arc::clone(&keep_producing);
        let produced = Arc::clone(&produced);
        thread::spawn(move || {
            while keep_producing.load(Ordering::Acquire) {
                if let Ok(transaction) = constructor.try_begin_gain() {
                    if let Ok(handle) = transaction.commit(gain_payload()) {
                        drop(handle);
                        produced.fetch_add(2, Ordering::AcqRel);
                    }
                }
                thread::yield_now();
            }
        })
    };
    let deadline = std::time::Instant::now() + TIMEOUT;
    while produced.load(Ordering::Acquire) <= OPEN_DRIVE_BUDGET {
        assert!(
            std::time::Instant::now() < deadline,
            "continuous teardown producer failed to establish work"
        );
        thread::sleep(Duration::from_millis(1));
    }

    // Work keeps being admitted and requested until the lifecycle worker observes the command and
    // seals admission. A driver that drains an unbounded stream before probing commands starves.
    let receipt = controller.shutdown_gracefully();
    let (outcome_send, outcome_recv) = mpsc::sync_channel(1);
    let waiter = thread::spawn(move || {
        let _ = outcome_send.send(executor::block_on(receipt));
    });
    let outcome = outcome_recv.recv_timeout(TIMEOUT);
    keep_producing.store(false, Ordering::Release);
    producer.join().unwrap();
    let report = confirmed(outcome.expect("continuous teardown starved lifecycle shutdown"));
    waiter.join().unwrap();
    assert!(produced.load(Ordering::Acquire) > OPEN_DRIVE_BUDGET);
    // Saturating admission may independently degrade the exact registry and therefore downgrade
    // the requested graceful mode. Completion while the producer is still running, with no
    // endpoint death, is the invariant under test: the command probe bounded the teardown pass.
    assert!(report.endpoint_death().is_none());
    assert!(control.shutdown_called.load(Ordering::Acquire));
}

#[test]
fn active_registry_upgrade_keeps_receipt_pending_until_exact_release() {
    let fixture = lifecycle_fixture();
    let active = fixture
        .registrar
        .hold_active_registry_upgrade_for_test()
        .unwrap();
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(PumpControl::new(false, true)),
        ),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };
    let receipt = controller.shutdown_gracefully();
    let (send, recv) = mpsc::sync_channel(1);
    let waiter = thread::spawn(move || send.send(executor::block_on(receipt)).unwrap());

    thread::sleep(Duration::from_millis(80));
    assert_eq!(recv.try_recv(), Err(mpsc::TryRecvError::Empty));
    drop(active);

    assert_eq!(
        confirmed(recv.recv_timeout(TIMEOUT).unwrap()).mode(),
        OutputShutdownMode::Graceful
    );
    waiter.join().unwrap();
}

struct FailWorkerSpawn;

impl LifecycleWorkerSpawner for FailWorkerSpawn {
    fn spawn(&self, _job: Box<dyn FnOnce() + Send + 'static>) -> std::io::Result<JoinHandle<()>> {
        Err(std::io::Error::other(
            "forced lifecycle worker spawn failure",
        ))
    }
}

struct RejectBootstrap;

impl LifecycleWorkerSpawner for RejectBootstrap {
    fn spawn(&self, job: Box<dyn FnOnce() + Send + 'static>) -> std::io::Result<JoinHandle<()>> {
        let (dropped_send, dropped_recv) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            drop(job);
            dropped_send.send(()).unwrap();
        });
        dropped_recv.recv().unwrap();
        Ok(worker)
    }
}

fn assert_running_transfer_failure(
    spawner: &dyn LifecycleWorkerSpawner,
    expected: OutputShutdownIssueKind,
    expects_worker: bool,
) {
    let fixture = lifecycle_fixture();
    let result = start_injected_output_with_spawner(
        Box::new(TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(PumpControl::new(false, true)),
        )),
        fixture.renderer,
        fixture.events,
        fixture.output_events,
        spawner,
    );
    let failure = match result {
        Ok(_) => panic!("forced worker transfer failure unexpectedly succeeded"),
        Err(failure) => failure,
    };
    assert_eq!(failure.issue().kind(), expected);
    let (_issue, parts, worker) = failure.into_parts();
    assert_eq!(worker.is_some(), expects_worker);
    if let Some(worker) = worker {
        worker.join().unwrap();
    }
    let InjectedOutputStartFailureParts::Running {
        endpoint,
        owner,
        output_events,
        event_loop,
    } = parts
    else {
        panic!("post-start transfer failure must return running resources");
    };
    let controller = start_running(endpoint, owner, output_events, event_loop, &ThreadSpawner)
        .ok()
        .unwrap();
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

fn assert_partial_transfer_failure(
    spawner: &dyn LifecycleWorkerSpawner,
    expected: OutputShutdownIssueKind,
    expects_worker: bool,
) {
    let fixture = lifecycle_fixture();
    let failure = start_injected_output_with_spawner(
        Box::new(TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Partial,
        )),
        fixture.renderer,
        fixture.events,
        fixture.output_events,
        spawner,
    )
    .err()
    .expect("forced partial worker transfer failure unexpectedly succeeded");
    assert_eq!(failure.issue().kind(), expected);
    let (_issue, parts, worker) = failure.into_parts();
    assert_eq!(worker.is_some(), expects_worker);
    if let Some(worker) = worker {
        worker.join().unwrap();
    }
    let InjectedOutputStartFailureParts::Partial {
        start_failure,
        owner,
        output_events,
        event_loop,
    } = parts
    else {
        panic!("partial transfer failure must return the exact partial resources");
    };
    let (startup_error, endpoint_shutdown) = start_failure.into_parts();
    let cleanup = start_partial(
        startup_error,
        endpoint_shutdown,
        owner,
        output_events,
        event_loop,
        &ThreadSpawner,
    )
    .ok()
    .unwrap();
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
}

fn assert_uncertain_transfer_failure(
    spawner: &dyn LifecycleWorkerSpawner,
    expected: OutputShutdownIssueKind,
    expects_worker: bool,
) {
    let fixture = lifecycle_fixture();
    let failure = start_injected_output_with_spawner(
        Box::new(TestPrepared::new(format(48_000.), PreparedBehavior::Panic)),
        fixture.renderer,
        fixture.events,
        fixture.output_events,
        spawner,
    )
    .err()
    .expect("forced uncertain worker transfer failure unexpectedly succeeded");
    assert_eq!(failure.issue().kind(), expected);
    let (_issue, parts, worker) = failure.into_parts();
    assert_eq!(worker.is_some(), expects_worker);
    if let Some(worker) = worker {
        worker.join().unwrap();
    }
    let InjectedOutputStartFailureParts::Uncertain {
        owner,
        output_events,
        event_loop,
    } = parts
    else {
        panic!("uncertain transfer failure must return the exact uncertain resources");
    };
    let cleanup = start_uncertain(owner, output_events, event_loop, &ThreadSpawner)
        .ok()
        .unwrap();
    let outcome = wait_receipt(cleanup.receipt());
    assert!(matches!(outcome, OutputShutdownOutcome::Unconfirmed { .. }));
}

#[test]
fn worker_spawn_and_bootstrap_failures_return_exact_running_resources_for_retry() {
    assert_running_transfer_failure(
        &FailWorkerSpawn,
        OutputShutdownIssueKind::WorkerSpawnFailed,
        false,
    );
    assert_running_transfer_failure(
        &RejectBootstrap,
        OutputShutdownIssueKind::BootstrapFailed,
        true,
    );
}

#[test]
fn partial_and_uncertain_worker_transfer_failures_return_exact_resources_for_retry() {
    assert_partial_transfer_failure(
        &FailWorkerSpawn,
        OutputShutdownIssueKind::WorkerSpawnFailed,
        false,
    );
    assert_partial_transfer_failure(
        &RejectBootstrap,
        OutputShutdownIssueKind::BootstrapFailed,
        true,
    );
    assert_uncertain_transfer_failure(
        &FailWorkerSpawn,
        OutputShutdownIssueKind::WorkerSpawnFailed,
        false,
    );
    assert_uncertain_transfer_failure(
        &RejectBootstrap,
        OutputShutdownIssueKind::BootstrapFailed,
        true,
    );
}

#[test]
fn prepared_start_panic_is_contained_without_minting_physical_graph_proof() {
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Panic),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = lifecycle else {
        panic!("start panic must enter automatic uncertain cleanup");
    };
    let outcome = wait_receipt(cleanup.receipt());
    let OutputShutdownOutcome::Unconfirmed { failure, .. } = outcome else {
        panic!("uncertain endpoint start must never claim physical graph retirement");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::InjectedStartPanicked
    );
}

#[test]
fn resume_panic_never_reenters_endpoint_shutdown_or_mints_graph_proof() {
    let control = PumpControl::new(true, true);
    control.resume_panics.store(true, Ordering::Release);
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(Arc::clone(&control)),
        ),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };
    let outcome = wait_receipt(controller.shutdown_gracefully());
    let OutputShutdownOutcome::Unconfirmed { failure, .. } = outcome else {
        panic!("resume panic must leave endpoint and graph retirement unconfirmed");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::EndpointMethodPanicked
    );
    assert!(!control.shutdown_called.load(Ordering::Acquire));
}

#[test]
fn endpoint_future_error_and_panic_are_quarantined_without_running_hostile_drop() {
    for (behavior, expected) in [
        (1, OutputShutdownIssueKind::EndpointRejectedShutdown),
        (2, OutputShutdownIssueKind::EndpointFuturePanicked),
    ] {
        let control = PumpControl::new(false, true);
        control.shutdown_behavior.store(behavior, Ordering::Release);
        let lifecycle = start(
            TestPrepared::new(
                format(48_000.),
                PreparedBehavior::Pump(Arc::clone(&control)),
            ),
            lifecycle_fixture(),
        )
        .ok()
        .unwrap();
        let InjectedOutputStart::Running(controller) = lifecycle else {
            panic!("pumping endpoint must start running");
        };
        let outcome = wait_receipt(controller.shutdown_gracefully());
        let OutputShutdownOutcome::Unconfirmed { failure, .. } = outcome else {
            panic!("hostile endpoint future must leave physical shutdown unconfirmed");
        };
        assert_eq!(failure.kind(), expected);
        assert!(!control.future_drop_ran.load(Ordering::Acquire));
    }
}

#[test]
fn endpoint_shutdown_method_panic_is_contained_without_whole_graph_proof() {
    let control = PumpControl::new(false, true);
    control.shutdown_behavior.store(3, Ordering::Release);
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(Arc::clone(&control)),
        ),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };
    let outcome = wait_receipt(controller.shutdown_gracefully());
    let OutputShutdownOutcome::Unconfirmed { failure, .. } = outcome else {
        panic!("panicking endpoint method must not mint a confirmed graph proof");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::EndpointMethodPanicked
    );
    assert!(control.shutdown_called.load(Ordering::Acquire));
}

#[test]
fn staged_payload_destructor_panic_quarantines_graph_and_event_loop() {
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Partial),
        lifecycle_fixture_with_suspension(true, true),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = lifecycle else {
        panic!("partial endpoint must enter automatic cleanup");
    };
    let outcome = wait_receipt(cleanup.receipt());
    let OutputShutdownOutcome::Unconfirmed {
        failure,
        event_issue,
    } = outcome
    else {
        panic!("hostile staged destructor must disqualify graph proof");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::InjectedWholeGraphQuarantined
    );
    assert_eq!(
        event_issue.unwrap().kind(),
        OutputShutdownIssueKind::EventThreadUnretired
    );
}

#[test]
fn renderer_reclaim_failure_never_retires_nodes_or_confirms_shutdown() {
    let mut fixture = lifecycle_fixture();
    fixture.renderer.fail_reclaim_for_test();
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Pump(PumpControl::new(false, true)),
        ),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running");
    };
    let outcome = wait_receipt(controller.shutdown_gracefully());
    let OutputShutdownOutcome::Unconfirmed { failure, .. } = outcome else {
        panic!("renderer reclaim failure must not mint WholeGraphRetired");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::RenderReclaimDegraded
    );
}

#[test]
fn endpoint_success_while_retaining_callback_never_mints_graph_proof() {
    let retained = Arc::new(Mutex::new(None));
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Retain(Arc::clone(&retained)),
        ),
        lifecycle_fixture(),
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("retaining endpoint must start running");
    };

    let outcome = wait_receipt(controller.shutdown_gracefully());
    let OutputShutdownOutcome::Unconfirmed { failure, .. } = outcome else {
        panic!("retained callback must make graph retirement unconfirmed");
    };
    assert_eq!(failure.kind(), OutputShutdownIssueKind::CallbackRetained);
    assert!(retained.lock().unwrap().is_some());
    drop(retained.lock().unwrap().take());
}
