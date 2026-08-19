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
    injected_control_channel, BoundInjectedRenderer, InjectedConcreteEventBinding,
    InjectedControlLifecycleOwner,
};
use crate::context::injected_ids::injected_node_id_pair;
use crate::context::injected_node_construction::{
    InjectedGainPayload, InjectedNodeConstructor, InjectedOscillatorConstructionError,
    InjectedOscillatorPayload,
};
use crate::context::injected_node_lifetime::injected_node_lifetime_registry;
use crate::context::injected_node_lifetime::{
    BoundInjectedOutputRenderer, MagicInitializedInjectedOutputRenderer,
};
use crate::context::{
    AdmissionError, AudioContextState, AudioNodeId, BaseAudioContext, ConcreteBaseAudioContext,
    ControlEventSendOutcome, InjectedContextAdmissionGate,
};
use crate::events::{
    injected_event_dispatch_setup, injected_event_dispatch_setup_bounded_for_test,
    injected_event_dispatch_setup_with_handlers, EventDispatch, EventHandler, EventLoopExit,
    EventPayload, EventType, InjectedEventDispatchSetup, InjectedLifecycleEventLoop,
};
use crate::message::ControlMessage;
use crate::node::{
    AudioNode, AudioScheduledSourceNode, AudioScheduledSourceNodeExt, ChannelConfigInner,
    ChannelCountMode, ChannelInterpretation, GainNode, GainOptions, OscillatorNode,
    OscillatorOptions, OscillatorType,
};
use crate::output::{
    AudioOutputConfig, AudioOutputDeathReason, AudioOutputErrorKind, AudioRenderCallback,
    AudioRenderFormat, AudioRenderStatus, EndpointShutdownConfirmed,
};
use crate::param::{
    injected_audio_param_raw_parts, AudioParamDescriptor, AudioParamInitialValue, AutomationRate,
    InjectedAudioParamProcessor,
};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::stats::AudioStats;

const TIMEOUT: Duration = Duration::from_secs(5);

struct LifecycleFixture {
    renderer: MagicInitializedInjectedOutputRenderer,
    events: AudioOutputEventSink,
    output_events: AudioOutputEventWatcher,
    gate: InjectedContextAdmissionGate,
    base: ConcreteBaseAudioContext,
    allocator: crate::context::injected_ids::InjectedNodeIdAllocator,
    registrar: crate::context::injected_node_lifetime::InjectedNodeLifetimeRegistrar,
}

impl LifecycleFixture {
    fn take_exact_base(&self) -> ConcreteBaseAudioContext {
        self.base.clone()
    }
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

fn lifecycle_fixture_with_event_setup(
    initially_suspended: bool,
    stage_hostile_payload: bool,
    capacity: usize,
    events: InjectedEventDispatchSetup,
) -> LifecycleFixture {
    lifecycle_fixture_inner(
        initially_suspended,
        stage_hostile_payload,
        capacity,
        capacity,
        events,
    )
}

fn lifecycle_fixture_with_capacity(
    initially_suspended: bool,
    stage_hostile_payload: bool,
    capacity: usize,
) -> LifecycleFixture {
    lifecycle_fixture_with_event_setup(
        initially_suspended,
        stage_hostile_payload,
        capacity,
        injected_event_dispatch_setup().unwrap(),
    )
}

fn lifecycle_fixture_inner(
    initially_suspended: bool,
    stage_hostile_payload: bool,
    control_capacity: usize,
    lifetime_capacity: usize,
    events: InjectedEventDispatchSetup,
) -> LifecycleFixture {
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) =
        injected_control_channel(gate.clone(), control_capacity, initially_suspended).unwrap();
    let (allocator, node_ids, graph) = injected_node_id_pair(0);
    let (registrar, bootstrap) =
        injected_node_lifetime_registry(lifetime_capacity, &producer, node_ids, graph)
            .ok()
            .unwrap();
    let constructor =
        InjectedNodeConstructor::new(producer.clone(), allocator.clone(), registrar.clone())
            .ok()
            .unwrap();
    let renderer = render_init
        .build_output_render_thread(
            bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            events,
        )
        .ok()
        .unwrap()
        .bind_output_lifecycle_exact(lifecycle)
        .ok()
        .unwrap();
    let (renderer, concrete_events) = renderer;
    let renderer = ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        renderer,
        constructor,
        concrete_events,
    )
    .ok()
    .unwrap()
    .try_build()
    .ok()
    .unwrap();
    let base = renderer.base().clone();
    if stage_hostile_payload {
        producer
            .try_commit_prevalidated_for_test(vec![ControlMessage::RegisterNode {
                id: crate::context::AudioNodeId(77),
                reclaim_id: llq::Node::new(crate::context::AudioNodeId(77)),
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
    drop(producer);
    let (events, output_events) = AudioOutputEventSink::bounded(8);
    LifecycleFixture {
        renderer,
        events,
        output_events,
        gate,
        base,
        allocator,
        registrar,
    }
}

fn unbound_event_renderer_with_gate(
    gate: InjectedContextAdmissionGate,
) -> (
    BoundInjectedRenderer,
    InjectedControlLifecycleOwner,
    InjectedNodeConstructor,
) {
    unbound_event_renderer_with_gate_and_setup(gate, injected_event_dispatch_setup().unwrap())
}

fn unbound_event_renderer_with_gate_and_setup(
    gate: InjectedContextAdmissionGate,
    events: InjectedEventDispatchSetup,
) -> (
    BoundInjectedRenderer,
    InjectedControlLifecycleOwner,
    InjectedNodeConstructor,
) {
    let (producer, lifecycle, render_init) = injected_control_channel(gate, 8, false).unwrap();
    let (allocator, node_ids, graph) = injected_node_id_pair(0);
    let (registrar, bootstrap) = injected_node_lifetime_registry(8, &producer, node_ids, graph)
        .ok()
        .unwrap();
    let constructor = InjectedNodeConstructor::new(producer.clone(), allocator, registrar)
        .ok()
        .unwrap();
    let renderer = render_init
        .build_output_render_thread(
            bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            events,
        )
        .ok()
        .unwrap();
    drop(producer);
    (renderer, lifecycle, constructor)
}

fn initialize_exact_output(
    renderer: BoundInjectedOutputRenderer,
    constructor: InjectedNodeConstructor,
    binding: InjectedConcreteEventBinding,
) -> MagicInitializedInjectedOutputRenderer {
    ConcreteBaseAudioContext::try_prepare_exact_injected_base(renderer, constructor, binding)
        .ok()
        .unwrap()
        .try_build()
        .ok()
        .unwrap()
}

struct PumpControl {
    running: AtomicBool,
    release: AtomicBool,
    stop: AtomicBool,
    resume_count: AtomicUsize,
    suspend_count: AtomicUsize,
    render_count: AtomicUsize,
    max_render_count: AtomicUsize,
    output_nonzero: AtomicBool,
    deny_next_render_allocation: AtomicBool,
    shutdown_called: AtomicBool,
    resume_panics: AtomicBool,
    resume_errors: AtomicBool,
    suspend_panics: AtomicBool,
    suspend_errors: AtomicBool,
    resume_blocked: AtomicBool,
    resume_entered: AtomicBool,
    suspend_blocked: AtomicBool,
    suspend_entered: AtomicBool,
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
            suspend_count: AtomicUsize::new(0),
            render_count: AtomicUsize::new(0),
            max_render_count: AtomicUsize::new(usize::MAX),
            output_nonzero: AtomicBool::new(false),
            deny_next_render_allocation: AtomicBool::new(false),
            shutdown_called: AtomicBool::new(false),
            resume_panics: AtomicBool::new(false),
            resume_errors: AtomicBool::new(false),
            suspend_panics: AtomicBool::new(false),
            suspend_errors: AtomicBool::new(false),
            resume_blocked: AtomicBool::new(false),
            resume_entered: AtomicBool::new(false),
            suspend_blocked: AtomicBool::new(false),
            suspend_entered: AtomicBool::new(false),
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
        self.control.resume_entered.store(true, Ordering::Release);
        while self.control.resume_blocked.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(1));
        }
        if self.control.resume_panics.load(Ordering::Acquire) {
            panic!("test endpoint resume panic");
        }
        if self.control.resume_errors.load(Ordering::Acquire) {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "test endpoint resume error",
            ));
        }
        self.control.resume_count.fetch_add(1, Ordering::AcqRel);
        self.control.running.store(true, Ordering::Release);
        Ok(())
    }

    fn suspend(&mut self) -> Result<(), AudioOutputError> {
        self.control.suspend_entered.store(true, Ordering::Release);
        while self.control.suspend_blocked.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(1));
        }
        if self.control.suspend_panics.load(Ordering::Acquire) {
            panic!("test endpoint suspend panic");
        }
        if self.control.suspend_errors.load(Ordering::Acquire) {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "test endpoint suspend error",
            ));
        }
        self.control.suspend_count.fetch_add(1, Ordering::AcqRel);
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
                            && thread_control.render_count.load(Ordering::Acquire)
                                < thread_control.max_render_count.load(Ordering::Acquire)
                        {
                            thread_control.render_count.fetch_add(1, Ordering::AcqRel);
                            let status = if thread_control
                                .deny_next_render_allocation
                                .swap(false, Ordering::AcqRel)
                            {
                                alloc_counter::deny_alloc(|| {
                                    callback.render_interleaved_f32(&mut output)
                                })
                            } else {
                                callback.render_interleaved_f32(&mut output)
                            };
                            thread_control.output_nonzero.store(
                                output.iter().any(|sample| *sample != 0.),
                                Ordering::Release,
                            );
                            if status == AudioRenderStatus::Stop {
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

struct BlockingPanicDropProcessor {
    entered: crossbeam_channel::Sender<()>,
    release: crossbeam_channel::Receiver<()>,
    tokens_restored: Arc<AtomicBool>,
}

struct DropProbeProcessor(Arc<AtomicUsize>);

static REJECTED_EVENT_DROPS: AtomicUsize = AtomicUsize::new(0);
static SEALED_EVENT_FACTORIES: AtomicUsize = AtomicUsize::new(0);

struct RejectedEventDrop;

impl Drop for RejectedEventDrop {
    fn drop(&mut self) {
        REJECTED_EVENT_DROPS.fetch_add(1, Ordering::AcqRel);
    }
}

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

impl Drop for BlockingPanicDropProcessor {
    fn drop(&mut self) {
        assert!(self.tokens_restored.load(Ordering::Acquire));
        self.entered.send(()).unwrap();
        self.release.recv().unwrap();
        panic!("forced blocked oscillator processor destructor panic");
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

impl AudioProcessor for BlockingPanicDropProcessor {
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
        param_processor: InjectedAudioParamProcessor::from_boxed_for_test(Box::new(
            SilentProcessor,
        )),
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
        param_processor: InjectedAudioParamProcessor::from_boxed_for_test(Box::new(
            DropProbeProcessor(Arc::clone(drops)),
        )),
        gain_processor: Box::new(DropProbeProcessor(Arc::clone(drops))),
        param_channel_config: config(),
        gain_channel_config: config(),
        initial_value: AudioParamInitialValue::new(1.),
    }
}

fn oscillator_payload_for_test(
    oscillator_processor: Box<dyn AudioProcessor>,
) -> InjectedOscillatorPayload {
    let (frequency_raw, frequency_processor) =
        injected_audio_param_raw_parts(AudioParamDescriptor {
            name: String::new(),
            min_value: -24_000.,
            max_value: 24_000.,
            default_value: 440.,
            automation_rate: AutomationRate::A,
        });
    let (detune_raw, detune_processor) = injected_audio_param_raw_parts(AudioParamDescriptor {
        name: String::new(),
        min_value: -153_600.,
        max_value: 153_600.,
        default_value: 0.,
        automation_rate: AutomationRate::A,
    });
    InjectedOscillatorPayload {
        frequency_processor,
        detune_processor,
        oscillator_processor,
        param_channel_config: ChannelConfigInner {
            count: 1,
            count_mode: ChannelCountMode::Explicit,
            interpretation: ChannelInterpretation::Discrete,
        },
        oscillator_channel_config: ChannelConfigInner {
            count: 2,
            count_mode: ChannelCountMode::Max,
            interpretation: ChannelInterpretation::Speakers,
        },
        frequency_initial_value: frequency_raw.set_initial_value_for_injected(440.),
        detune_initial_value: detune_raw.set_initial_value_for_injected(0.),
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

fn wait_state(receipt: InjectedStateChangeReceipt) -> InjectedStateChangeOutcome {
    let (send, recv) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = send.send(executor::block_on(receipt));
    });
    recv.recv_timeout(TIMEOUT)
        .expect("injected state-change receipt timed out")
}

fn wait_until(mut predicate: impl FnMut() -> bool, message: &str) {
    let deadline = std::time::Instant::now() + TIMEOUT;
    while !predicate() {
        assert!(std::time::Instant::now() < deadline, "{message}");
        thread::sleep(Duration::from_millis(1));
    }
}

fn confirmed(outcome: OutputShutdownOutcome) -> OutputShutdownReport {
    match outcome {
        OutputShutdownOutcome::Confirmed(report) => report,
        other => panic!("expected confirmed injected shutdown, got {other:?}"),
    }
}

fn retire_fixture_without_event_loop(
    fixture: LifecycleFixture,
) -> (
    InjectedLifecycleEventLoop,
    crate::context::RetiredInjectedGraph,
) {
    let (owner, callback, event_loop, base) = fixture
        .renderer
        .try_into_audio_output_pair(format(48_000.), fixture.events)
        .ok()
        .unwrap();
    drop(base);
    drop(callback);
    drop(fixture.output_events);
    let pending = owner
        .try_begin_close()
        .ok()
        .unwrap()
        .retire_and_wait()
        .seal_and_finish()
        .ok()
        .unwrap();
    let ready = pending
        .retire_payloads()
        .ok()
        .unwrap()
        .into_silent_reclaim();
    ready.begin_render_shutdown();
    let graph = match ready.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new()) {
        InjectedRenderReclaimOutcome::Reclaimed(graph) => graph,
        _ => panic!("dropped callback must permit exact physical reclaim"),
    };
    let retired = match graph.try_retire_nodes() {
        InjectedNodeRetireOutcome::Retired(retired) => retired,
        _ => panic!("inactive exact registry must retire without retry"),
    };
    (event_loop, retired)
}

#[test]
fn foreign_control_event_branch_returns_both_exact_bundles_for_clean_reuse() {
    let (mut first_renderer, first_control, first_constructor) =
        unbound_event_renderer_with_gate(InjectedContextAdmissionGate::new());
    let (mut second_renderer, second_control, second_constructor) =
        unbound_event_renderer_with_gate(InjectedContextAdmissionGate::new());
    first_renderer.swap_control_event_branches_for_test(&mut second_renderer);

    let failure = first_renderer
        .bind_output_lifecycle_exact(first_control)
        .err()
        .expect("foreign control-event producer must fail before callback publication");
    let (mut first_renderer, first_control) = failure.into_parts();
    first_renderer.swap_control_event_branches_for_test(&mut second_renderer);
    let (first_renderer, first_binding) = first_renderer
        .bind_output_lifecycle_exact(first_control)
        .ok()
        .unwrap();
    let (second_renderer, second_binding) = second_renderer
        .bind_output_lifecycle_exact(second_control)
        .ok()
        .unwrap();
    let first = initialize_exact_output(first_renderer, first_constructor, first_binding);
    let second = initialize_exact_output(second_renderer, second_constructor, second_binding);

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
fn foreign_retired_graph_cannot_stop_event_consumer_and_both_proofs_remain_usable() {
    let (first_events, first_retired) = retire_fixture_without_event_loop(lifecycle_fixture());
    let (second_events, second_retired) = retire_fixture_without_event_loop(lifecycle_fixture());

    let first_events = first_events
        .retire_confirmed(&second_retired, false)
        .err()
        .expect("foreign retired-graph brand must return the exact event owner");
    let first = first_events
        .retire_confirmed(&first_retired, false)
        .ok()
        .unwrap();
    let second = second_events
        .retire_confirmed(&second_retired, false)
        .ok()
        .unwrap();
    assert!(!first.graceful);
    assert!(!second.graceful);
    assert_eq!(first.joined.unwrap(), EventLoopExit::TerminalClosed);
    assert_eq!(second.joined.unwrap(), EventLoopExit::TerminalClosed);
}

#[test]
fn same_gate_event_branch_swap_rejects_by_brand_then_both_branches_deliver() {
    let gate = InjectedContextAdmissionGate::new();
    let first_count = Arc::new(AtomicUsize::new(0));
    let first_for_handler = Arc::clone(&first_count);
    let first_setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                first_for_handler.fetch_add(1, Ordering::AcqRel);
            })),
        );
    })
    .unwrap();
    let second_count = Arc::new(AtomicUsize::new(0));
    let second_for_handler = Arc::clone(&second_count);
    let second_setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                second_for_handler.fetch_add(1, Ordering::AcqRel);
            })),
        );
    })
    .unwrap();
    let (mut first_renderer, first_control, first_constructor) =
        unbound_event_renderer_with_gate_and_setup(gate.clone(), first_setup);
    let (mut second_renderer, second_control, second_constructor) =
        unbound_event_renderer_with_gate_and_setup(gate, second_setup);

    first_renderer.swap_control_event_branches_for_test(&mut second_renderer);
    let failure = first_renderer
        .bind_output_lifecycle_exact(first_control)
        .err()
        .expect("shared admission gate must not hide a foreign event identity");
    let (mut first_renderer, first_control) = failure.into_parts();
    first_renderer.swap_control_event_branches_for_test(&mut second_renderer);
    let (first_renderer, first_events) = first_renderer
        .bind_output_lifecycle_exact(first_control)
        .ok()
        .unwrap();
    let (second_renderer, second_events) = second_renderer
        .bind_output_lifecycle_exact(second_control)
        .ok()
        .unwrap();
    let first_renderer = initialize_exact_output(first_renderer, first_constructor, first_events);
    let second_renderer =
        initialize_exact_output(second_renderer, second_constructor, second_events);
    let first_base = first_renderer.base().clone();
    let second_base = second_renderer.base().clone();
    assert_eq!(
        first_base.send_event_with(EventDispatch::sink_change),
        Ok(())
    );
    assert_eq!(
        second_base.send_event_with(EventDispatch::sink_change),
        Ok(())
    );
    let deadline = std::time::Instant::now() + TIMEOUT;
    while first_count.load(Ordering::Acquire) != 1 || second_count.load(Ordering::Acquire) != 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "own event delivery stalled"
        );
        thread::sleep(Duration::from_millis(1));
    }

    let (events, output_events) = AudioOutputEventSink::bounded(8);
    let cleanup = start_injected_output(
        Box::new(TestPrepared::new(
            format(48_000.),
            PreparedBehavior::Partial,
        )),
        first_renderer,
        events,
        output_events,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = cleanup else {
        panic!("first shared-gate bundle must remain operational")
    };
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
    assert_eq!(first_base.state(), AudioContextState::Closed);
    for base in [&first_base, &second_base] {
        assert_eq!(
            base.send_event_with(EventDispatch::sink_change),
            Err(ControlEventSendOutcome::AdmissionRejected(
                AdmissionError::Sealed
            ))
        );
    }
    drop(second_base);
    // The deliberately invalid two-transports/one-gate topology cannot independently close its
    // second control owner after the shared irreversible seal. Quarantine it prepublication.
    second_renderer.quarantine_prepublication_for_test();
}

#[test]
fn output_build_mismatch_returns_single_use_event_setup_for_exact_retry() {
    let first_gate = InjectedContextAdmissionGate::new();
    let (first_producer, first_lifecycle, first_init) =
        injected_control_channel(first_gate, 8, false).unwrap();
    let (first_allocator, first_ids, first_graph) = injected_node_id_pair(0);
    let (first_registrar, first_bootstrap) =
        injected_node_lifetime_registry(8, &first_producer, first_ids, first_graph)
            .ok()
            .unwrap();
    let first_constructor =
        InjectedNodeConstructor::new(first_producer.clone(), first_allocator, first_registrar)
            .ok()
            .unwrap();

    let second_gate = InjectedContextAdmissionGate::new();
    let (second_producer, second_lifecycle, second_init) =
        injected_control_channel(second_gate, 8, false).unwrap();
    let (second_allocator, second_ids, second_graph) = injected_node_id_pair(0);
    let (second_registrar, second_bootstrap) =
        injected_node_lifetime_registry(8, &second_producer, second_ids, second_graph)
            .ok()
            .unwrap();
    let second_constructor =
        InjectedNodeConstructor::new(second_producer.clone(), second_allocator, second_registrar)
            .ok()
            .unwrap();

    let handled = Arc::new(AtomicUsize::new(0));
    let handled_by_setup = Arc::clone(&handled);
    let setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                handled_by_setup.fetch_add(1, Ordering::AcqRel);
            })),
        );
    })
    .unwrap();
    let failure = match first_init.build_output_render_thread(
        second_bootstrap,
        48_000.,
        2,
        Arc::new(AtomicU64::new(0)),
        AudioStats::new(),
        setup,
    ) {
        Err(failure) => failure,
        Ok(_) => panic!("foreign node bootstrap must fail before splitting event setup"),
    };

    let (first_renderer, first_binding) = failure
        .init
        .build_output_render_thread(
            first_bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            failure.events,
        )
        .ok()
        .unwrap()
        .bind_output_lifecycle_exact(first_lifecycle)
        .ok()
        .unwrap();
    let first_renderer = initialize_exact_output(first_renderer, first_constructor, first_binding);
    let first_base = first_renderer.base().clone();
    assert_eq!(
        first_base.send_event_with(EventDispatch::sink_change),
        Ok(())
    );
    let handled_deadline = std::time::Instant::now() + TIMEOUT;
    while handled.load(Ordering::Acquire) != 1 {
        assert!(
            std::time::Instant::now() < handled_deadline,
            "returned single-use setup did not reach its original handler"
        );
        thread::sleep(Duration::from_millis(1));
    }

    let (second_renderer, second_binding) = second_init
        .build_output_render_thread(
            failure.node_lifetimes,
            48_000.,
            2,
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            injected_event_dispatch_setup().unwrap(),
        )
        .ok()
        .unwrap()
        .bind_output_lifecycle_exact(second_lifecycle)
        .ok()
        .unwrap();
    let second_renderer =
        initialize_exact_output(second_renderer, second_constructor, second_binding);
    drop(first_producer);
    drop(second_producer);

    for renderer in [first_renderer, second_renderer] {
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
            panic!("returned exact build resources must remain operational")
        };
        assert_eq!(
            confirmed(wait_receipt(cleanup.receipt())).mode(),
            OutputShutdownMode::Silent
        );
    }
    assert_eq!(first_base.state(), AudioContextState::Closed);
}

#[test]
fn exact_concrete_base_mismatch_returns_constructor_and_event_binding_for_reuse() {
    let (first_renderer, first_control, first_constructor) =
        unbound_event_renderer_with_gate(InjectedContextAdmissionGate::new());
    let (second_renderer, second_control, second_constructor) =
        unbound_event_renderer_with_gate(InjectedContextAdmissionGate::new());
    let (first_renderer, first_events) = first_renderer
        .bind_output_lifecycle_exact(first_control)
        .ok()
        .unwrap();
    let (second_renderer, second_events) = second_renderer
        .bind_output_lifecycle_exact(second_control)
        .ok()
        .unwrap();

    let failure = match ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        first_renderer,
        first_constructor,
        second_events,
    ) {
        Err(failure) => failure,
        Ok(_) => panic!("foreign exact event binding must be rejected before base publication"),
    };
    let (first_renderer, first_constructor, second_events) = failure.into_parts();
    let first = initialize_exact_output(first_renderer, first_constructor, first_events);
    let second = initialize_exact_output(second_renderer, second_constructor, second_events);
    let first_base = first.base().clone();
    let second_base = second.base().clone();
    for base in [&first_base, &second_base] {
        assert_eq!(base.state(), AudioContextState::Running);
        assert_eq!(base.send_event_with(EventDispatch::sink_change), Ok(()));
    }

    for (renderer, base) in [(first, first_base), (second, second_base)] {
        let (events, output_events) = AudioOutputEventSink::bounded(8);
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
            panic!("operational retry must start partial cleanup")
        };
        assert_eq!(
            confirmed(wait_receipt(cleanup.receipt())).mode(),
            OutputShutdownMode::Silent
        );
        assert_eq!(base.state(), AudioContextState::Closed);
    }
}

#[test]
fn exact_control_event_rejection_drops_under_admission_and_seal_waits() {
    REJECTED_EVENT_DROPS.store(0, Ordering::Release);
    SEALED_EVENT_FACTORIES.store(0, Ordering::Release);
    let (handler_entered_send, handler_entered) = crossbeam_channel::bounded(1);
    let (handler_release_send, handler_release) = crossbeam_channel::bounded(1);
    let setup = injected_event_dispatch_setup_bounded_for_test(1, move |events| {
        events.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                handler_entered_send.send(()).unwrap();
                handler_release.recv().unwrap();
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
    let surviving_dispatch = base.control_event_dispatch();
    let pump = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(Arc::clone(&pump))),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };

    assert_eq!(base.send_event_with(EventDispatch::sink_change), Ok(()));
    handler_entered.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(
        base.send_event_with(EventDispatch::control_batch_activity),
        Ok(())
    );

    let (admitted_send, admitted) = crossbeam_channel::bounded(1);
    let (admission_release_send, admission_release) = crossbeam_channel::bounded(1);
    surviving_dispatch.set_exact_after_admission_for_test(Arc::new(move || {
        admitted_send.send(()).unwrap();
        admission_release.recv().unwrap();
    }));
    let sending = thread::spawn(move || {
        surviving_dispatch.send_with(|| {
            EventDispatch::message(crate::context::AudioNodeId(91), Box::new(RejectedEventDrop))
        })
    });
    admitted.recv_timeout(TIMEOUT).unwrap();

    let receipt = controller.shutdown_silently();
    let (outcome_send, outcome_recv) = mpsc::sync_channel(1);
    let waiter = thread::spawn(move || outcome_send.send(executor::block_on(receipt)).unwrap());
    thread::sleep(Duration::from_millis(40));
    assert_eq!(outcome_recv.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert!(!pump.shutdown_called.load(Ordering::Acquire));

    admission_release_send.send(()).unwrap();
    assert_eq!(sending.join().unwrap(), ControlEventSendOutcome::Full);
    assert_eq!(REJECTED_EVENT_DROPS.load(Ordering::Acquire), 1);
    let shutdown_deadline = std::time::Instant::now() + TIMEOUT;
    while !pump.shutdown_called.load(Ordering::Acquire) {
        assert!(
            std::time::Instant::now() < shutdown_deadline,
            "endpoint shutdown did not follow admitted payload destruction"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(outcome_recv.try_recv(), Err(mpsc::TryRecvError::Empty));
    handler_release_send.send(()).unwrap();
    let report = confirmed(outcome_recv.recv_timeout(TIMEOUT).unwrap());
    waiter.join().unwrap();
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert_eq!(base.state(), AudioContextState::Closed);
    assert_eq!(
        base.send_event_with(|| {
            SEALED_EVENT_FACTORIES.fetch_add(1, Ordering::AcqRel);
            EventDispatch::sink_change()
        }),
        Err(ControlEventSendOutcome::AdmissionRejected(
            AdmissionError::Sealed
        ))
    );
    assert_eq!(SEALED_EVENT_FACTORIES.load(Ordering::Acquire), 0);
}

#[test]
fn exact_base_cannot_publish_closed_before_physical_retirement() {
    let closed_count = Arc::new(AtomicUsize::new(0));
    let closed_for_handler = Arc::clone(&closed_count);
    let setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |_| {
                closed_for_handler.fetch_add(1, Ordering::AcqRel);
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
    base.set_state(AudioContextState::Closed);
    assert_eq!(base.state(), AudioContextState::Running);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(closed_count.load(Ordering::Acquire), 0);

    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Partial),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Cleanup(cleanup) = lifecycle else {
        panic!("partial start must retain automatic physical cleanup")
    };
    assert_eq!(
        confirmed(wait_receipt(cleanup.receipt())).mode(),
        OutputShutdownMode::Silent
    );
    assert_eq!(base.state(), AudioContextState::Closed);
    assert_eq!(closed_count.load(Ordering::Acquire), 1);
    base.set_state(AudioContextState::Running);
    base.set_state(AudioContextState::Suspended);
    assert_eq!(base.state(), AudioContextState::Closed);
    assert_eq!(closed_count.load(Ordering::Acquire), 1);
}

#[test]
fn suspended_graceful_close_resumes_once_and_retires_exact_graph() {
    // Prepared::start publishes a logically Running endpoint even though the exact graph starts
    // suspended. The lifecycle worker must reconcile native suspension autonomously.
    let control = PumpControl::new(false, true);
    let fixture = lifecycle_fixture_with_suspension(true, false);
    let base = fixture.take_exact_base();
    assert_eq!(base.state(), AudioContextState::Suspended);
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
    assert_eq!(controller.base().address(), base.address());

    wait_until(
        || control.suspend_count.load(Ordering::Acquire) == 1,
        "initial native suspension was not reconciled",
    );
    assert!(!control.running.load(Ordering::Acquire));

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(control.resume_count.load(Ordering::Acquire), 1);
    assert_eq!(control.suspend_count.load(Ordering::Acquire), 1);
    assert!(control.render_count.load(Ordering::Acquire) > 0);
    assert!(control.shutdown_called.load(Ordering::Acquire));
    assert!(report.reclaim_issue().is_none());
    assert!(report.event_issue().is_none());
    assert_eq!(base.state(), AudioContextState::Closed);
}

#[test]
fn suspended_graceful_close_resume_error_is_uncertain_and_never_calls_shutdown() {
    let fixture = lifecycle_fixture_with_event_setup(
        true,
        false,
        8,
        injected_event_dispatch_setup().unwrap(),
    );
    let base = fixture.take_exact_base();
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
    wait_until(
        || control.suspend_count.load(Ordering::Acquire) == 1,
        "initial native suspension did not complete",
    );
    assert_eq!(base.state(), AudioContextState::Suspended);
    control.resume_errors.store(true, Ordering::Release);

    let OutputShutdownOutcome::Unconfirmed { failure, .. } =
        wait_receipt(controller.shutdown_gracefully())
    else {
        panic!("failed Close-observability resume must leave endpoint ownership uncertain");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::EndpointStateTransitionFailed
    );
    assert!(!control.shutdown_called.load(Ordering::Acquire));
}

#[test]
fn running_suspend_resume_orders_exact_state_events_and_native_endpoint() {
    let (state_send, state_recv) = mpsc::channel();
    let setup = injected_event_dispatch_setup_with_handlers(move |event_loop| {
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |payload| {
                if let EventPayload::AudioContextState(state) = payload {
                    let _ = state_send.send(state);
                }
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
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
    let state = controller.state_control();
    wait_until(
        || control.render_count.load(Ordering::Acquire) > 2,
        "running callback did not render",
    );

    assert_eq!(
        wait_state(state.suspend().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    assert_eq!(base.state(), AudioContextState::Suspended);
    assert_eq!(control.suspend_count.load(Ordering::Acquire), 1);
    assert!(!control.running.load(Ordering::Acquire));
    assert_eq!(
        state_recv.recv_timeout(TIMEOUT).unwrap(),
        AudioContextState::Suspended
    );
    let halted = control.render_count.load(Ordering::Acquire);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(control.render_count.load(Ordering::Acquire), halted);

    assert_eq!(
        wait_state(state.resume().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    assert_eq!(base.state(), AudioContextState::Running);
    assert_eq!(control.resume_count.load(Ordering::Acquire), 1);
    assert_eq!(
        state_recv.recv_timeout(TIMEOUT).unwrap(),
        AudioContextState::Running
    );
    wait_until(
        || control.render_count.load(Ordering::Acquire) > halted,
        "resumed callback did not render",
    );

    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
    assert_eq!(base.state(), AudioContextState::Closed);
}

#[test]
fn bounded_state_event_failure_is_terminal_and_silently_closes() {
    let (handler_entered_send, handler_entered) = crossbeam_channel::bounded(1);
    let (handler_release_send, handler_release) = crossbeam_channel::bounded(1);
    let setup = injected_event_dispatch_setup_bounded_for_test(1, move |events| {
        events.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                handler_entered_send.send(()).unwrap();
                handler_release.recv().unwrap();
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
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
    assert_eq!(base.send_event_with(EventDispatch::sink_change), Ok(()));
    handler_entered.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(
        base.send_event_with(EventDispatch::control_batch_activity),
        Ok(())
    );

    let shutdown = controller.receipt();
    assert_eq!(
        wait_state(controller.state_control().suspend().unwrap()),
        InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::EventDelivery)
    );
    // The failed event enqueue first publishes Suspended, then authoritatively latches Silent
    // teardown. Receipt completion and physical retirement run consecutively, so the shared state
    // may already have reached absorbing Closed by the time this observing thread resumes.
    assert!(matches!(
        base.state(),
        AudioContextState::Suspended | AudioContextState::Closed
    ));
    handler_release_send.send(()).unwrap();
    let report = confirmed(wait_receipt(shutdown));
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert_eq!(
        report.reclaim_issue().unwrap().kind(),
        OutputShutdownIssueKind::EventDeliveryDegraded
    );
    assert!(report.event_issue().is_none());
    assert_eq!(base.state(), AudioContextState::Closed);
}

#[test]
fn resume_flushes_all_staged_graph_work_before_state_ack() {
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
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
    let state = controller.state_control();
    assert_eq!(
        wait_state(state.suspend().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    let applied_before = base.applied_control_batch_sequence();
    let first_gain = base
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(gain_payload())
        .unwrap();
    let second_gain = base
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(gain_payload())
        .unwrap();
    assert_eq!(base.applied_control_batch_sequence(), applied_before);

    control.resume_blocked.store(true, Ordering::Release);
    let resumed = state.resume().unwrap();
    wait_until(
        || control.resume_entered.load(Ordering::Acquire),
        "native resume was not invoked before staged flush",
    );
    assert_eq!(base.state(), AudioContextState::Suspended);
    assert_eq!(base.applied_control_batch_sequence(), applied_before);
    control.resume_blocked.store(false, Ordering::Release);
    assert_eq!(wait_state(resumed), InjectedStateChangeOutcome::Applied);
    assert!(base.applied_control_batch_sequence() > applied_before);
    assert_eq!(base.state(), AudioContextState::Running);
    assert!(base.applied_control_batch_sequence() >= applied_before + 2);
    drop(first_gain);
    drop(second_gain);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn suspended_exact_gain_values_flush_fifo_before_b4c_resume_ack() {
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
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
    let state = controller.state_control();
    assert_eq!(
        wait_state(state.suspend().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    let applied_before = base.applied_control_batch_sequence();

    let gain = GainNode::new(&base, GainOptions::default());
    gain.gain().set_value(0.25);
    gain.gain().set_value(0.75);
    assert_eq!(gain.gain().value(), 0.75);
    assert_eq!(base.applied_control_batch_sequence(), applied_before);
    let rendered_before_resume = control.render_count.load(Ordering::Acquire);

    assert_eq!(
        wait_state(state.resume().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    assert!(base.applied_control_batch_sequence() >= applied_before + 3);
    wait_until(
        || control.render_count.load(Ordering::Acquire) > rendered_before_resume,
        "resumed endpoint did not process a post-resume quantum",
    );
    assert_eq!(gain.gain().value(), 0.75);
    drop(gain);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn suspended_exact_connect_then_disconnect_stage_and_flush_fifo_before_resume_ack() {
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
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
    let state = controller.state_control();
    assert_eq!(
        wait_state(state.suspend().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    let applied_before = base.applied_control_batch_sequence();

    let source = GainNode::new(&base, GainOptions::default());
    let destination = GainNode::new(&base, GainOptions::default());
    let constructor = base.injected_node_constructor().unwrap();
    source.connect(&destination);
    assert_eq!(constructor.connection_edge_count_for_test(), 1);
    source.disconnect_dest(&destination);
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
    assert_eq!(base.applied_control_batch_sequence(), applied_before);

    assert_eq!(
        wait_state(state.resume().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    assert!(base.applied_control_batch_sequence() >= applied_before + 4);
    assert_eq!(base.state(), AudioContextState::Running);
    drop((source, destination));
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.reclaim_issue().is_none());
}

#[test]
fn exact_fixed_oscillators_use_ids_eleven_through_thirteen_render_without_allocation_and_end_once()
{
    let fixture = lifecycle_fixture_inner(
        false,
        false,
        64,
        16,
        injected_event_dispatch_setup().unwrap(),
    );
    let base = fixture.take_exact_base();
    let destination = base.destination();
    let ended = Arc::new(AtomicUsize::new(0));
    let mut oscillators = Vec::new();
    let mut completions = Vec::new();
    for (index, type_) in [
        OscillatorType::Sine,
        OscillatorType::Square,
        OscillatorType::Sawtooth,
        OscillatorType::Triangle,
    ]
    .into_iter()
    .enumerate()
    {
        let mut oscillator = OscillatorNode::new(
            &base,
            OscillatorOptions {
                type_,
                frequency: 220. + index as f32 * 55.,
                detune: index as f32 * 10.,
                ..OscillatorOptions::default()
            },
        );
        assert_eq!(
            oscillator.registration().id(),
            AudioNodeId(11 + index as u64 * 3)
        );
        assert_eq!(
            oscillator.frequency().registration().id(),
            AudioNodeId(12 + index as u64 * 3)
        );
        assert_eq!(
            oscillator.detune().registration().id(),
            AudioNodeId(13 + index as u64 * 3)
        );
        assert_eq!(oscillator.type_(), type_);
        oscillator.frequency().set_value(330. + index as f32 * 20.);
        oscillator.detune().set_value(index as f32 * 25.);
        let ended_count = Arc::clone(&ended);
        oscillator.set_onended(move |_| {
            ended_count.fetch_add(1, Ordering::AcqRel);
        });
        oscillator.connect(&destination);
        completions.push(oscillator.completion_token());
        oscillator.start();
        oscillators.push(oscillator);
    }

    let control = PumpControl::new(false, true);
    control
        .deny_next_render_allocation
        .store(true, Ordering::Release);
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
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || control.output_nonzero.load(Ordering::Acquire),
        "fixed exact oscillators did not render nonzero output",
    );

    oscillators[0].set_type(OscillatorType::Square);
    assert_eq!(oscillators[0].type_(), OscillatorType::Square);
    for oscillator in &mut oscillators {
        oscillator.stop();
    }
    wait_until(
        || completions.iter().all(|token| token.is_complete()),
        "fixed exact oscillator completion did not become authoritative",
    );
    wait_until(
        || ended.load(Ordering::Acquire) == oscillators.len(),
        "each fixed exact oscillator did not dispatch exactly one ended event",
    );
    thread::sleep(Duration::from_millis(10));
    assert_eq!(ended.load(Ordering::Acquire), oscillators.len());

    drop(oscillators);
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.reclaim_issue().is_none());
}

#[test]
fn suspended_exact_oscillator_commands_flush_fifo_through_real_b4c_resume() {
    let fixture =
        lifecycle_fixture_inner(true, false, 32, 8, injected_event_dispatch_setup().unwrap());
    let base = fixture.take_exact_base();
    let destination = base.destination();
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let completion = oscillator.completion_token();
    let ended = Arc::new(AtomicUsize::new(0));
    let ended_for_handler = Arc::clone(&ended);
    oscillator.set_onended(move |_| {
        ended_for_handler.fetch_add(1, Ordering::AcqRel);
    });
    oscillator.connect(&destination);
    oscillator.frequency().set_value(330.);
    oscillator.detune().set_value(120.);
    oscillator.set_type(OscillatorType::Triangle);
    oscillator.start_at(0.);
    oscillator.stop_at(1.);
    oscillator.stop_at(0.01);
    let applied_before = base.applied_control_batch_sequence();

    let control = PumpControl::new(false, false);
    control.max_render_count.store(32, Ordering::Release);
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
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || control.suspend_count.load(Ordering::Acquire) == 1,
        "initially suspended endpoint was not natively reconciled",
    );
    assert_eq!(base.state(), AudioContextState::Suspended);
    assert_eq!(base.applied_control_batch_sequence(), applied_before);

    // The endpoint remains idle until Resume flips its native running state. Once it does, the
    // pump can execute at most 32 quanta, providing a deterministic audio-time window.
    control.release.store(true, Ordering::Release);
    assert_eq!(
        wait_state(controller.state_control().resume().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    wait_until(
        || control.render_count.load(Ordering::Acquire) == 32,
        "fixed resumed render window did not complete",
    );
    assert!(
        completion.is_complete(),
        "staged Stop FIFO was reversed: 32 quanta exceed 0.01s but are below 1s"
    );
    wait_until(
        || ended.load(Ordering::Acquire) == 1,
        "resumed exact oscillator did not dispatch ended",
    );
    assert_eq!(oscillator.type_(), OscillatorType::Triangle);
    assert_eq!(oscillator.frequency().value(), 330.);
    assert_eq!(oscillator.detune().value(), 120.);
    assert!(base.applied_control_batch_sequence() >= applied_before + 8);

    control
        .max_render_count
        .store(usize::MAX, Ordering::Release);
    drop(oscillator);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn connected_started_stopped_oscillator_survives_immediate_handle_drop_until_ended_and_reclaim() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 32);
    let base = fixture.take_exact_base();
    let destination = base.destination();
    let ended = Arc::new(AtomicUsize::new(0));
    let ended_for_handler = Arc::clone(&ended);
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let completion = oscillator.completion_token();
    oscillator.set_onended(move |_| {
        ended_for_handler.fetch_add(1, Ordering::AcqRel);
    });
    oscillator.connect(&destination);
    oscillator.start_at(0.);
    oscillator.stop_at(0.01);
    drop(oscillator);

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || completion.is_complete(),
        "dropped connected source did not remain live through its stop",
    );
    wait_until(
        || ended.load(Ordering::Acquire) == 1,
        "dropped connected source did not dispatch ended",
    );
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.reclaim_issue().is_none());
}

#[test]
fn never_started_and_future_started_sources_do_not_synthesize_ended_during_close() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 32);
    let base = fixture.take_exact_base();
    let destination = base.destination();
    let ended = Arc::new(AtomicUsize::new(0));

    let never = OscillatorNode::new(&base, OscillatorOptions::default());
    let never_completion = never.completion_token();
    let never_ended = Arc::clone(&ended);
    never.set_onended(move |_| {
        never_ended.fetch_add(1, Ordering::AcqRel);
    });
    drop(never);

    let mut future = OscillatorNode::new(&base, OscillatorOptions::default());
    let future_completion = future.completion_token();
    let future_ended = Arc::clone(&ended);
    future.set_onended(move |_| {
        future_ended.fetch_add(1, Ordering::AcqRel);
    });
    future.connect(&destination);
    future.start_at(10.);
    drop(future);

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(ended.load(Ordering::Acquire), 0);
    assert!(!never_completion.is_complete());
    assert!(!future_completion.is_complete());
}

#[test]
fn exact_oscillator_validation_precedes_reservation_and_deferred_custom_paths_are_inert() {
    let fixture =
        lifecycle_fixture_inner(false, false, 2, 8, injected_event_dispatch_setup().unwrap());
    let base = fixture.take_exact_base();
    let constructor = base.injected_node_constructor().unwrap();
    let initial_sequence = constructor.last_submitted_batch_sequence();
    let probe = fixture.allocator.try_reserve(3).unwrap();
    assert_eq!(probe.id(0), AudioNodeId(11));
    assert_eq!(probe.id(1), AudioNodeId(12));
    assert_eq!(probe.id(2), AudioNodeId(13));
    drop(probe);

    for options in [
        OscillatorOptions {
            type_: OscillatorType::Custom,
            ..OscillatorOptions::default()
        },
        OscillatorOptions {
            periodic_wave: Some(crate::PeriodicWave::default()),
            ..OscillatorOptions::default()
        },
    ] {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            OscillatorNode::new(&base, options)
        }))
        .is_err());
        assert_eq!(
            constructor.last_submitted_batch_sequence(),
            initial_sequence
        );
        assert_eq!(
            constructor.connection_transport_accounting_for_test(),
            (0, 0, 0, 0)
        );
        let probe = fixture.allocator.try_reserve(3).unwrap();
        assert_eq!(probe.id(0), AudioNodeId(11));
        assert_eq!(probe.id(1), AudioNodeId(12));
        assert_eq!(probe.id(2), AudioNodeId(13));
        drop(probe);
    }

    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let construction_sequence = constructor.last_submitted_batch_sequence();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| oscillator.stop())).is_err());
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        construction_sequence
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.start_at(f64::NAN)
    }))
    .is_err());
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        construction_sequence
    );
    oscillator.start();
    let start_sequence = constructor.last_submitted_batch_sequence();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| oscillator.start())).is_err());
    assert_eq!(constructor.last_submitted_batch_sequence(), start_sequence);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_type(OscillatorType::Custom)
    }))
    .is_err());
    assert_eq!(constructor.last_submitted_batch_sequence(), start_sequence);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_periodic_wave(crate::PeriodicWave::default())
    }))
    .is_err());
    assert_eq!(constructor.last_submitted_batch_sequence(), start_sequence);

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn exact_oscillator_event_nonce_exhaustion_is_typed_terminal_and_never_reused() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let nonce_source = Arc::new(AtomicU64::new(u64::MAX));
    base.injected_events()
        .unwrap()
        .set_nonce_source_for_test(Arc::clone(&nonce_source));

    assert_eq!(
        base.try_begin_injected_oscillator(OscillatorType::Sine)
            .err(),
        Some(InjectedOscillatorConstructionError::EventIdentityExhausted)
    );
    assert_eq!(nonce_source.load(Ordering::Acquire), u64::MAX);
    let ids = fixture.allocator.try_reserve(3).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(14));
    assert_eq!(ids.id(1), AudioNodeId(15));
    assert_eq!(ids.id(2), AudioNodeId(16));
    drop(ids);
    assert!(matches!(
        base.try_begin_injected_oscillator(OscillatorType::Sine),
        Err(InjectedOscillatorConstructionError::Control(
            crate::context::injected_control::InjectedControlError::ProtocolViolation
        ))
    ));
    assert_eq!(nonce_source.load(Ordering::Acquire), u64::MAX);

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_silently())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn exact_oscillator_not_accepted_parser_restores_all_three_ids_from_the_seven_record_batch() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let transaction = base
        .try_begin_injected_oscillator(OscillatorType::Sine)
        .unwrap();
    base.fail_closed_injected_protocol();
    assert_eq!(
        transaction
            .commit(oscillator_payload_for_test(Box::new(SilentProcessor)))
            .err(),
        Some(InjectedOscillatorConstructionError::Control(
            crate::context::injected_control::InjectedControlError::ProtocolViolation
        ))
    );

    // A normal NotAccepted rollback accepts only the exact seven-record construction shape and
    // restores each reclaim token before returning. Magic permanently owns 0..=10.
    let restored = fixture.allocator.try_reserve(3).unwrap();
    assert_eq!(restored.id(0), AudioNodeId(11));
    assert_eq!(restored.id(1), AudioNodeId(12));
    assert_eq!(restored.id(2), AudioNodeId(13));
    drop(restored);

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_silently())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn exact_oscillator_not_accepted_restores_ids_before_hostile_payload_drop_and_close_drain() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let allocator = fixture.allocator.clone();
    let mut transaction = base
        .try_begin_injected_oscillator(OscillatorType::Sine)
        .unwrap();
    let tokens_restored = Arc::new(AtomicBool::new(false));
    transaction.observe_rollback_tokens_restored_for_test(Arc::clone(&tokens_restored));
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
        panic!("pumping endpoint must start running")
    };

    base.fail_closed_injected_protocol();
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    let tokens_restored_in_drop = Arc::clone(&tokens_restored);
    let commit = thread::spawn(move || {
        transaction.commit(oscillator_payload_for_test(Box::new(
            BlockingPanicDropProcessor {
                entered: entered_send,
                release: release_recv,
                tokens_restored: tokens_restored_in_drop,
            },
        )))
    });
    entered_recv.recv_timeout(TIMEOUT).unwrap();
    assert!(tokens_restored.load(Ordering::Acquire));

    let receipt = controller.shutdown_silently();
    let (done_send, done_recv) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = done_send.send(executor::block_on(receipt));
    });
    thread::sleep(Duration::from_millis(20));
    assert!(done_recv.try_recv().is_err());
    assert!(!control.shutdown_called.load(Ordering::Acquire));

    release_send.send(()).unwrap();
    assert_eq!(
        commit.join().unwrap().err(),
        Some(InjectedOscillatorConstructionError::RejectedPayloadPanicked)
    );
    let restored = allocator.try_reserve(3).unwrap();
    assert_eq!(restored.id(0), AudioNodeId(14));
    assert_eq!(restored.id(1), AudioNodeId(15));
    assert_eq!(restored.id(2), AudioNodeId(16));
    drop(restored);
    let report = confirmed(done_recv.recv_timeout(TIMEOUT).unwrap());
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
}

#[test]
fn exact_oscillator_group_finalizer_failure_or_panic_quarantines_all_three_ids() {
    for (ordinal, panics) in [(1, false), (2, false), (3, false), (2, true)] {
        let fixture = lifecycle_fixture_with_capacity(false, false, 8);
        let base = fixture.take_exact_base();
        if panics {
            fixture.registrar.panic_construction_arm_for_test(ordinal);
        } else {
            fixture.registrar.fail_construction_arm_for_test(ordinal);
        }
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            OscillatorNode::new(&base, OscillatorOptions::default())
        }))
        .is_err());

        // Acceptance committed the whole ID group before arming any public registration. No
        // subset is republished after a finalizer fault.
        let next = fixture.allocator.try_reserve(3).unwrap();
        assert_eq!(next.id(0), AudioNodeId(14));
        assert_eq!(next.id(1), AudioNodeId(15));
        assert_eq!(next.id(2), AudioNodeId(16));
        drop(next);

        let control = PumpControl::new(false, true);
        let lifecycle = start(
            TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
            fixture,
        )
        .ok()
        .unwrap();
        let InjectedOutputStart::Running(controller) = lifecycle else {
            panic!("pumping endpoint must start running")
        };
        assert_eq!(
            confirmed(wait_receipt(controller.shutdown_silently())).mode(),
            OutputShutdownMode::Silent
        );
    }
}

#[test]
fn exact_confirmed_close_releases_handler_and_postseal_set_drops_callback_before_panicking() {
    struct CallbackDrop(Arc<AtomicUsize>);

    impl Drop for CallbackDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let drops = Arc::new(AtomicUsize::new(0));
    let retained = CallbackDrop(Arc::clone(&drops));
    oscillator.set_onended(move |_| {
        let _ = &retained;
    });

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
    assert_eq!(drops.load(Ordering::Acquire), 1);

    let rejected = CallbackDrop(Arc::clone(&drops));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_onended(move |_| {
            let _ = &rejected;
        });
    }))
    .is_err());
    assert_eq!(drops.load(Ordering::Acquire), 2);
    oscillator.clear_onended();
    oscillator.clear_onended();
}

#[test]
fn reconciled_oscillator_id_reuse_rejects_stale_target_and_old_event_cannot_hit_replacement() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let allocator = fixture.allocator.clone();
    let registrar = fixture.registrar.clone();
    let destination = base.destination();
    let mut old = OscillatorNode::new(&base, OscillatorOptions::default());
    let old_id = old.registration().id();
    let stale_target = old.injected_control_for_test().ended_target();
    let old_key = stale_target.render_key();
    let old_completion = old.completion_token();
    old.connect(&destination);
    old.start_at(0.);
    old.stop_at(0.);

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || old_completion.is_complete(),
        "first oscillator did not complete before generation reuse",
    );
    drop(old);
    wait_until(
        || registrar.slot_phase_counts_for_test() == Some([8, 0, 0, 0, 0, 0]),
        "first oscillator generation did not reconcile",
    );

    // Reorder the three returned reclaim tokens so the next oscillator's node (slot zero in its
    // atomic reservation) deliberately reuses the old oscillator's numeric id.
    let mut returned = allocator.try_reserve(3).unwrap();
    let old_position = (0..3)
        .find(|index| returned.id(*index) == old_id)
        .expect("reconciled oscillator id must be among its returned three-token group");
    if old_position != 0 {
        let old_token = returned.take_reclaim_node(old_position).unwrap();
        let first_token = returned.take_reclaim_node(0).unwrap();
        returned
            .restore_reclaim_node(old_position, first_token)
            .ok()
            .unwrap();
        returned.restore_reclaim_node(0, old_token).ok().unwrap();
    }
    drop(returned);

    let mut replacement = OscillatorNode::new(&base, OscillatorOptions::default());
    assert_eq!(replacement.registration().id(), old_id);
    let replacement_key = replacement
        .injected_control_for_test()
        .ended_target()
        .render_key();
    assert_ne!(replacement_key, old_key);
    assert_eq!(
        stale_target.try_set_handler(EventHandler::Once(Box::new(|_| {}))),
        Err(crate::events::InjectedExactEndedHandlerError::Inactive)
    );

    let ended = Arc::new(AtomicUsize::new(0));
    let ended_for_handler = Arc::clone(&ended);
    replacement.set_onended(move |_| {
        ended_for_handler.fetch_add(1, Ordering::AcqRel);
    });
    let (barrier_send, barrier_recv) = crossbeam_channel::bounded(1);
    base.set_event_handler(
        EventType::SinkChange,
        EventHandler::Once(Box::new(move |_| barrier_send.send(()).unwrap())),
    );
    assert_eq!(
        base.send_event_with(move || EventDispatch::exact_ended(old_key)),
        Ok(())
    );
    assert_eq!(base.send_event_with(EventDispatch::sink_change), Ok(()));
    barrier_recv.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(ended.load(Ordering::Acquire), 0);

    let replacement_completion = replacement.completion_token();
    replacement.connect(&destination);
    replacement.start_at(0.);
    replacement.stop_at(0.);
    wait_until(
        || replacement_completion.is_complete(),
        "replacement oscillator did not complete",
    );
    wait_until(
        || ended.load(Ordering::Acquire) == 1,
        "replacement exact ended key did not invoke its own handler",
    );
    drop(replacement);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn exact_oscillator_runtime_saturation_precedes_no_host_mutation_and_recovers() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let constructor = base.injected_node_constructor().unwrap();
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let construction_sequence = constructor.last_submitted_batch_sequence();
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || base.applied_control_batch_sequence() >= construction_sequence,
        "oscillator construction did not apply before runtime saturation",
    );
    wait_until(
        || constructor.connection_transport_accounting_for_test() == (0, 0, 0, 0),
        "construction credits did not retire before runtime saturation",
    );

    let held = constructor
        .control()
        .try_begin_operation(crate::message::CONTROL_COMMANDS_PER_CALLBACK)
        .unwrap();
    let submitted_before = constructor.last_submitted_batch_sequence();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_type(OscillatorType::Square);
    }))
    .is_err());
    assert_eq!(oscillator.type_(), OscillatorType::Sine);
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        submitted_before
    );
    drop(held);

    oscillator.set_type(OscillatorType::Square);
    assert_eq!(oscillator.type_(), OscillatorType::Square);
    drop(oscillator);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn exact_oscillator_runtime_not_accepted_rolls_back_without_host_mirror_mutation() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let constructor = base.injected_node_constructor().unwrap();
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let construction_sequence = constructor.last_submitted_batch_sequence();
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || base.applied_control_batch_sequence() >= construction_sequence,
        "oscillator construction did not apply before runtime rejection",
    );
    wait_until(
        || constructor.connection_transport_accounting_for_test() == (0, 0, 0, 0),
        "construction credits did not retire before runtime rejection",
    );
    let submitted_before = constructor.last_submitted_batch_sequence();
    oscillator
        .injected_control_for_test()
        .fail_next_runtime_commit_for_test();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_type(OscillatorType::Square);
    }))
    .is_err());
    assert_eq!(oscillator.type_(), OscillatorType::Sine);
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        submitted_before
    );
    assert_eq!(
        constructor.connection_transport_accounting_for_test(),
        (0, 0, 0, 0)
    );
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_silently())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn accepted_oscillator_runtime_finalizer_panic_keeps_command_queue_owned_and_applied() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let constructor = base.injected_node_constructor().unwrap();
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    oscillator
        .injected_control_for_test()
        .panic_next_runtime_finalizer_for_test();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_type(OscillatorType::Square);
    }))
    .is_err());
    assert_eq!(oscillator.type_(), OscillatorType::Square);
    let accepted_sequence = constructor.last_submitted_batch_sequence();

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || base.applied_control_batch_sequence() >= accepted_sequence,
        "accepted oscillator command was suppressed after its finalizer panic",
    );
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_silently())).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn exact_oscillator_runtime_wrong_processor_latches_before_failed_watermark() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 8);
    let base = fixture.take_exact_base();
    let constructor = base.injected_node_constructor().unwrap();
    let transaction = base
        .try_begin_injected_oscillator(OscillatorType::Sine)
        .unwrap();
    let constructed = transaction
        .commit(oscillator_payload_for_test(Box::new(SilentProcessor)))
        .unwrap();
    let construction_sequence = constructor.last_submitted_batch_sequence();
    constructed.oscillator_control.try_start(0.).unwrap();
    let failed_sequence = constructor.last_submitted_batch_sequence();

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert_eq!(base.applied_control_batch_sequence(), construction_sequence);
    assert!(construction_sequence < failed_sequence);
    assert!(constructor.control().render_protocol_failed_for_test());
    drop(constructed);
}

#[test]
fn saturated_exact_event_queue_cannot_hide_authoritative_oscillator_completion() {
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    let setup = injected_event_dispatch_setup_bounded_for_test(1, move |events| {
        events.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                entered_send.send(()).unwrap();
                release_recv.recv().unwrap();
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 32, setup);
    let base = fixture.take_exact_base();
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let completion = oscillator.completion_token();
    let ended = Arc::new(AtomicUsize::new(0));
    let ended_for_handler = Arc::clone(&ended);
    oscillator.set_onended(move |_| {
        ended_for_handler.fetch_add(1, Ordering::AcqRel);
    });
    oscillator.connect(&base.destination());
    oscillator.start_at(0.);
    oscillator.stop_at(0.);

    assert_eq!(base.send_event_with(EventDispatch::sink_change), Ok(()));
    entered_recv.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(
        base.send_event_with(EventDispatch::control_batch_activity),
        Ok(())
    );

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || completion.is_complete(),
        "event saturation suppressed exact oscillator completion",
    );
    assert_eq!(ended.load(Ordering::Acquire), 0);
    release_send.send(()).unwrap();
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(ended.load(Ordering::Acquire), 0);
}

#[test]
fn disconnected_exact_event_queue_cannot_hide_authoritative_oscillator_completion() {
    static QUEUED_PAYLOAD_DROPS: AtomicUsize = AtomicUsize::new(0);

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    struct PanicDropProbe(Arc<AtomicUsize>);

    impl Drop for PanicDropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
            panic!("hostile retained exact handler destructor");
        }
    }

    struct QueuedPayloadDropProbe(&'static AtomicUsize);

    impl Drop for QueuedPayloadDropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    struct RejectedHandlerDropProbe(Arc<AtomicBool>);

    impl Drop for RejectedHandlerDropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    let setup = injected_event_dispatch_setup().unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 32, setup);
    let base = fixture.take_exact_base();
    let retained_handler_drops = Arc::new(AtomicUsize::new(0));
    let activity_probe = DropProbe(Arc::clone(&retained_handler_drops));
    base.set_event_activity_handler(move || {
        let _ = &activity_probe;
    });
    let mut oscillator = OscillatorNode::new(&base, OscillatorOptions::default());
    let completion = oscillator.completion_token();
    let ended = Arc::new(AtomicUsize::new(0));
    let ended_for_handler = Arc::clone(&ended);
    let ended_probe = PanicDropProbe(Arc::clone(&retained_handler_drops));
    oscillator.set_onended(move |_| {
        let _ = &ended_probe;
        ended_for_handler.fetch_add(1, Ordering::AcqRel);
    });
    oscillator.connect(&base.destination());
    oscillator.start_at(0.);
    oscillator.stop_at(0.);

    QUEUED_PAYLOAD_DROPS.store(0, Ordering::Release);
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    base.set_event_handler(
        EventType::SinkChange,
        EventHandler::Once(Box::new(move |_| {
            entered_send.send(()).unwrap();
            release_recv.recv().unwrap();
        })),
    );
    assert_eq!(base.send_event_with(EventDispatch::sink_change), Ok(()));
    entered_recv.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(
        base.send_event_with(EventDispatch::force_exact_thread_exit_for_test),
        Ok(())
    );
    let queued_payload_drops = &QUEUED_PAYLOAD_DROPS;
    assert_eq!(
        base.send_event_with(move || {
            EventDispatch::message(
                AudioNodeId(9999),
                Box::new(QueuedPayloadDropProbe(queued_payload_drops)),
            )
        }),
        Ok(())
    );
    release_send.send(()).unwrap();
    let deadline = std::time::Instant::now() + TIMEOUT;
    loop {
        match base.send_event_with(EventDispatch::control_batch_activity) {
            Err(ControlEventSendOutcome::Disconnected) => break,
            Ok(()) | Err(ControlEventSendOutcome::Full) => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "forced exact event-thread exit did not disconnect its receiver"
                );
                thread::sleep(Duration::from_millis(1));
            }
            other => panic!("unexpected event-send outcome before Close: {other:?}"),
        }
    }
    let rejected_handler_dropped = Arc::new(AtomicBool::new(false));
    let rejected_probe = RejectedHandlerDropProbe(Arc::clone(&rejected_handler_dropped));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oscillator.set_onended(move |_| {
            let _ = &rejected_probe;
        });
    }))
    .is_err());
    assert!(rejected_handler_dropped.load(Ordering::Acquire));

    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || completion.is_complete(),
        "event disconnect suppressed exact oscillator completion",
    );
    assert_eq!(ended.load(Ordering::Acquire), 0);
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(
        report.event_issue().unwrap().kind(),
        OutputShutdownIssueKind::EventDeliveryDegraded
    );
    // The base and oscillator still retain their exact event capabilities. Confirmed graph
    // retirement plus the observed early thread exit nevertheless clears both callback maps on
    // the lifecycle worker, containing the hostile target-handler destructor before completion.
    assert_eq!(retained_handler_drops.load(Ordering::Acquire), 2);
    assert_eq!(QUEUED_PAYLOAD_DROPS.load(Ordering::Acquire), 1);
}

#[test]
fn public_exact_param_connection_changes_real_destination_output_and_disconnect_silences_it() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 16);
    let base = fixture.take_exact_base();
    let gain = GainNode::new(&base, GainOptions::default());
    let destination = base.destination();
    gain.gain().connect(&destination);
    assert_eq!(
        base.injected_node_constructor()
            .unwrap()
            .connection_edge_count_for_test(),
        1
    );

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
        panic!("pumping endpoint must start running")
    };
    wait_until(
        || control.output_nonzero.load(Ordering::Acquire),
        "exact AudioParam edge never reached the real destination render",
    );

    gain.gain().disconnect_dest(&destination);
    let render_before_disconnect = control.render_count.load(Ordering::Acquire);
    wait_until(
        || {
            control.render_count.load(Ordering::Acquire) > render_before_disconnect
                && !control.output_nonzero.load(Ordering::Acquire)
        },
        "exact disconnect did not silence the next real destination render",
    );
    assert_eq!(
        base.injected_node_constructor()
            .unwrap()
            .connection_edge_count_for_test(),
        0
    );

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.reclaim_issue().is_none());
}

#[test]
fn surviving_public_exact_handles_report_invalid_state_after_confirmed_retirement() {
    let fixture = lifecycle_fixture_with_capacity(false, false, 16);
    let base = fixture.take_exact_base();
    let source = GainNode::new(&base, GainOptions::default());
    let destination = GainNode::new(&base, GainOptions::default());
    let control = PumpControl::new(false, true);
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Pump(control)),
        fixture,
    )
    .ok()
    .unwrap();
    let InjectedOutputStart::Running(controller) = lifecycle else {
        panic!("pumping endpoint must start running")
    };
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.reclaim_issue().is_none());

    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        source.connect(&destination);
    }))
    .expect_err("retired exact endpoint must reject public connect");
    let message = payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .expect("connection panic uses a string payload");
    assert!(message.starts_with("InvalidStateError"));
    assert!(!message.starts_with("InvalidAccessError"));
}

#[test]
fn public_exact_capacity_accepts_256_rejects_257_and_recovers_with_atomic_broad_disconnect() {
    let fixture = lifecycle_fixture_inner(
        false,
        false,
        768,
        256,
        injected_event_dispatch_setup().unwrap(),
    );
    let base = fixture.take_exact_base();
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
        panic!("pumping endpoint must start running")
    };
    let constructor = base.injected_node_constructor().unwrap();
    let mut gains = Vec::with_capacity(128);
    for _ in 0..128 {
        gains.push(GainNode::new(&base, GainOptions::default()));
        let construction_sequence = constructor.last_submitted_batch_sequence();
        wait_until(
            || base.applied_control_batch_sequence() >= construction_sequence,
            "exact Gain batch was not applied before the next construction",
        );
        wait_until(
            || constructor.connection_transport_accounting_for_test() == (0, 0, 0, 0),
            "exact Gain credits did not retire before the next construction",
        );
    }

    let destination = base.destination();
    let listener = base.listener();
    let mut candidates: Vec<&dyn AudioNode> = vec![
        &destination,
        listener.position_x(),
        listener.position_y(),
        listener.position_z(),
        listener.forward_x(),
        listener.forward_y(),
        listener.forward_z(),
        listener.up_x(),
        listener.up_y(),
        listener.up_z(),
    ];
    for gain in gains.iter().skip(1) {
        candidates.push(gain);
        candidates.push(gain.gain());
    }
    assert!(candidates.len() > crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS);
    let source = &gains[0];
    for (index, candidate) in candidates
        .iter()
        .take(crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS)
        .enumerate()
    {
        source.connect(*candidate);
        if index % 16 == 15 {
            wait_until(
                || constructor.connection_transport_accounting_for_test() == (0, 0, 0, 0),
                "connection fill credits did not retire",
            );
        }
    }
    assert_eq!(
        constructor.connection_edge_count_for_test(),
        crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS
    );
    let full_sequence = constructor.last_submitted_batch_sequence();
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        source.connect(candidates[crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS]);
    }))
    .expect_err("the 257th exact explicit edge must be rejected");
    let message = payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .expect("capacity panic uses a string payload");
    assert!(message.starts_with("NotSupportedError"));
    assert!(message.contains("256"));
    assert_eq!(constructor.last_submitted_batch_sequence(), full_sequence);
    assert_eq!(
        constructor.connection_edge_count_for_test(),
        crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS
    );

    source.disconnect_dest(candidates[0]);
    assert_eq!(
        constructor.connection_edge_count_for_test(),
        crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS - 1
    );
    source.connect(candidates[crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS]);
    assert_eq!(
        constructor.connection_edge_count_for_test(),
        crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS
    );
    let replacement_sequence = constructor.last_submitted_batch_sequence();
    wait_until(
        || base.applied_control_batch_sequence() >= replacement_sequence,
        "replacement edge did not apply after capacity recovery",
    );
    wait_until(
        || constructor.connection_transport_accounting_for_test() == (0, 0, 0, 0),
        "replacement edge credits did not retire before broad disconnect",
    );

    source.disconnect();
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
    let broad_disconnect_sequence = constructor.last_submitted_batch_sequence();
    wait_until(
        || base.applied_control_batch_sequence() >= broad_disconnect_sequence,
        "atomic 256-edge broad disconnect did not apply",
    );
    drop(candidates);
    drop(listener);
    drop(destination);
    drop(gains);
    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert!(report.reclaim_issue().is_none());
}

#[test]
fn dropped_caller_does_not_cancel_and_duplicate_requests_coalesce() {
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
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
    let state = controller.state_control();
    drop(state.suspend().unwrap());
    let duplicate = state.suspend().unwrap();
    wait_until(
        || base.state() == AudioContextState::Suspended,
        "dropped Suspend caller cancelled native work",
    );
    assert_eq!(wait_state(duplicate), InjectedStateChangeOutcome::Unchanged);
    assert_eq!(control.suspend_count.load(Ordering::Acquire), 1);
    assert_eq!(
        wait_state(state.resume().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    assert_eq!(control.resume_count.load(Ordering::Acquire), 1);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn close_supersedes_stalled_state_barrier_and_uses_next_sequence() {
    let fixture = lifecycle_fixture();
    let control = PumpControl::new(false, false);
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
    let state = controller.state_control();
    let state_receipt = state.suspend().unwrap();
    thread::sleep(Duration::from_millis(20));
    let shutdown = controller.shutdown_gracefully();
    control.release.store(true, Ordering::Release);
    assert_eq!(
        wait_state(state_receipt),
        InjectedStateChangeOutcome::SupersededByShutdown
    );
    assert_eq!(
        confirmed(wait_receipt(shutdown)).mode(),
        OutputShutdownMode::Graceful
    );
    assert!(control.shutdown_called.load(Ordering::Acquire));
    assert_eq!(
        wait_state(state.resume().unwrap()),
        InjectedStateChangeOutcome::Closed
    );
}

#[test]
fn endpoint_suspend_error_is_uncertain_and_never_reenters_shutdown() {
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
    let control = PumpControl::new(false, true);
    control.suspend_errors.store(true, Ordering::Release);
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
    let shutdown = controller.receipt();
    assert_eq!(
        wait_state(controller.state_control().suspend().unwrap()),
        InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::EndpointUncertain)
    );
    let OutputShutdownOutcome::Unconfirmed { failure, .. } = wait_receipt(shutdown) else {
        panic!("endpoint transition error must not mint graph proof");
    };
    assert_eq!(
        failure.kind(),
        OutputShutdownIssueKind::EndpointStateTransitionFailed
    );
    assert!(!control.shutdown_called.load(Ordering::Acquire));
    // Suspend authoritatively stored Suspended before the endpoint rejected its native method.
    // The forgotten but still-live callback may subsequently consume the fail-closed exact Close
    // and absorbingly store Closed without minting endpoint/whole-graph proof. Running is invalid;
    // the Unconfirmed outcome and absence of endpoint shutdown above remain authoritative.
    assert!(matches!(
        base.state(),
        AudioContextState::Suspended | AudioContextState::Closed
    ));
}

#[test]
fn suspend_event_and_ack_precede_native_call_and_close_latches_during_block() {
    let (state_send, state_recv) = mpsc::sync_channel(1);
    let setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |payload| {
                if let EventPayload::AudioContextState(state) = payload {
                    let _ = state_send.send(state);
                }
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
    let control = PumpControl::new(false, true);
    control.suspend_blocked.store(true, Ordering::Release);
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
    let state_receipt = controller.state_control().suspend().unwrap();
    wait_until(
        || control.suspend_entered.load(Ordering::Acquire),
        "native suspend was not reached",
    );
    assert_eq!(base.state(), AudioContextState::Suspended);
    assert_eq!(
        state_recv.recv_timeout(TIMEOUT).unwrap(),
        AudioContextState::Suspended
    );

    let shutdown = controller.shutdown_silently();
    control.suspend_blocked.store(false, Ordering::Release);
    assert_eq!(
        wait_state(state_receipt),
        InjectedStateChangeOutcome::SupersededByShutdown
    );
    assert_eq!(
        confirmed(wait_receipt(shutdown)).mode(),
        OutputShutdownMode::Silent
    );
    assert!(control.shutdown_called.load(Ordering::Acquire));
}

#[test]
fn opposing_state_requests_are_fifo_serialized() {
    let (state_send, state_recv) = mpsc::channel();
    let setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |payload| {
                if let EventPayload::AudioContextState(state) = payload {
                    let _ = state_send.send(state);
                }
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
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
    let state = controller.state_control();
    let suspend = state.suspend().unwrap();
    let resume = state.resume().unwrap();
    assert_eq!(wait_state(suspend), InjectedStateChangeOutcome::Applied);
    assert_eq!(wait_state(resume), InjectedStateChangeOutcome::Applied);
    assert_eq!(
        state_recv.recv_timeout(TIMEOUT).unwrap(),
        AudioContextState::Suspended
    );
    assert_eq!(
        state_recv.recv_timeout(TIMEOUT).unwrap(),
        AudioContextState::Running
    );
    assert_eq!(control.suspend_count.load(Ordering::Acquire), 1);
    assert_eq!(control.resume_count.load(Ordering::Acquire), 1);
    assert_eq!(base.state(), AudioContextState::Running);
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn state_request_queue_full_cannot_block_out_of_band_close() {
    let fixture = lifecycle_fixture();
    let control = PumpControl::new(false, true);
    control.suspend_blocked.store(true, Ordering::Release);
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
    let state = controller.state_control();
    let first = state.suspend().unwrap();
    wait_until(
        || control.suspend_entered.load(Ordering::Acquire),
        "first state request did not enter native suspend",
    );
    let mut queued = Vec::new();
    for index in 0..STATE_REQUEST_CAPACITY {
        queued.push(if index % 2 == 0 {
            state.resume().unwrap()
        } else {
            state.suspend().unwrap()
        });
    }
    assert_eq!(
        state.resume().err(),
        Some(InjectedControlError::LogicalCommandCredits)
    );
    let shutdown = controller.shutdown_silently();
    control.suspend_blocked.store(false, Ordering::Release);
    assert_eq!(
        wait_state(first),
        InjectedStateChangeOutcome::SupersededByShutdown
    );
    for receipt in queued {
        assert_eq!(
            wait_state(receipt),
            InjectedStateChangeOutcome::SupersededByShutdown
        );
    }
    assert_eq!(
        confirmed(wait_receipt(shutdown)).mode(),
        OutputShutdownMode::Silent
    );
}

#[test]
fn active_preboundary_graph_preparation_delays_suspend_without_blocking_callback() {
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
    let control = PumpControl::new(false, true);
    let transaction = base
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap();
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
    let state = controller.state_control();
    let suspended = state.suspend().unwrap();
    let callbacks_before = control.render_count.load(Ordering::Acquire);
    thread::sleep(Duration::from_millis(30));
    assert_eq!(base.state(), AudioContextState::Running);
    assert!(
        control.render_count.load(Ordering::Acquire) > callbacks_before,
        "pre-boundary preparation must not stop the active callback"
    );
    let gain = transaction.commit(gain_payload()).unwrap();
    assert_eq!(wait_state(suspended), InjectedStateChangeOutcome::Applied);
    assert_eq!(base.state(), AudioContextState::Suspended);
    drop(gain);
    assert_eq!(
        wait_state(state.resume().unwrap()),
        InjectedStateChangeOutcome::Applied
    );
    assert_eq!(
        confirmed(wait_receipt(controller.shutdown_gracefully())).mode(),
        OutputShutdownMode::Graceful
    );
}

#[test]
fn state_barrier_live_stall_has_no_deadline_and_endpoint_death_supersedes() {
    let fixture = lifecycle_fixture();
    let control = PumpControl::new(false, false);
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
    let state_receipt = controller.state_control().suspend().unwrap();
    let (state_send, state_recv) = mpsc::sync_channel(1);
    thread::spawn(move || state_send.send(executor::block_on(state_receipt)).unwrap());
    thread::sleep(Duration::from_millis(50));
    assert_eq!(state_recv.try_recv(), Err(mpsc::TryRecvError::Empty));
    control.stop.store(true, Ordering::Release);
    assert_eq!(
        state_recv.recv_timeout(TIMEOUT).unwrap(),
        InjectedStateChangeOutcome::SupersededByShutdown
    );
    let report = confirmed(wait_receipt(controller.receipt()));
    assert_eq!(report.mode(), OutputShutdownMode::Silent);
    assert!(report.endpoint_death().is_some());
}

#[test]
fn resume_error_and_panic_are_uncertain_state_failures() {
    for panic_on_resume in [false, true] {
        let fixture = lifecycle_fixture();
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
        let state = controller.state_control();
        assert_eq!(
            wait_state(state.suspend().unwrap()),
            InjectedStateChangeOutcome::Applied
        );
        if panic_on_resume {
            control.resume_panics.store(true, Ordering::Release);
        } else {
            control.resume_errors.store(true, Ordering::Release);
        }
        let shutdown = controller.receipt();
        assert_eq!(
            wait_state(state.resume().unwrap()),
            InjectedStateChangeOutcome::Failed(InjectedStateChangeFailure::EndpointUncertain)
        );
        let OutputShutdownOutcome::Unconfirmed { failure, .. } = wait_receipt(shutdown) else {
            panic!("uncertain native resume cannot mint physical proof");
        };
        assert_eq!(
            failure.kind(),
            if panic_on_resume {
                OutputShutdownIssueKind::EndpointMethodPanicked
            } else {
                OutputShutdownIssueKind::EndpointStateTransitionFailed
            }
        );
        assert!(!control.shutdown_called.load(Ordering::Acquire));
    }
}

#[test]
fn initial_native_suspend_error_and_panic_are_uncertain_without_state_event() {
    for panic_on_suspend in [false, true] {
        let state_events = Arc::new(AtomicUsize::new(0));
        let state_events_for_handler = Arc::clone(&state_events);
        let setup = injected_event_dispatch_setup_with_handlers(move |events| {
            events.set_handler(
                EventType::StateChange,
                EventHandler::Multiple(Box::new(move |_| {
                    state_events_for_handler.fetch_add(1, Ordering::AcqRel);
                })),
            );
        })
        .unwrap();
        let fixture = lifecycle_fixture_with_event_setup(true, false, 8, setup);
        let control = PumpControl::new(false, true);
        if panic_on_suspend {
            control.suspend_panics.store(true, Ordering::Release);
        } else {
            control.suspend_errors.store(true, Ordering::Release);
        }
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
            panic!("pumping endpoint must transfer before worker reconciliation");
        };
        let OutputShutdownOutcome::Unconfirmed { failure, .. } = wait_receipt(controller.receipt())
        else {
            panic!("initial endpoint uncertainty cannot mint graph proof");
        };
        assert_eq!(
            failure.kind(),
            if panic_on_suspend {
                OutputShutdownIssueKind::EndpointMethodPanicked
            } else {
                OutputShutdownIssueKind::EndpointStateTransitionFailed
            }
        );
        assert_eq!(state_events.load(Ordering::Acquire), 0);
        assert!(!control.shutdown_called.load(Ordering::Acquire));
    }
}

#[test]
fn graceful_closed_handler_runs_only_after_graph_processors_are_destroyed() {
    let processor_drops = Arc::new(AtomicUsize::new(0));
    let drops_seen_by_handler = Arc::clone(&processor_drops);
    let (closed_send, closed_recv) = mpsc::sync_channel(1);
    let event_pair = injected_event_dispatch_setup_with_handlers(move |event_loop| {
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
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, event_pair);
    let base = fixture.take_exact_base();
    let live_gain = base
        .injected_node_constructor()
        .unwrap()
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
    while base.applied_control_batch_sequence() < 2 {
        assert!(std::time::Instant::now() < deadline, "Gain batch stalled");
        thread::sleep(Duration::from_millis(1));
    }

    let report = confirmed(wait_receipt(controller.shutdown_gracefully()));
    assert_eq!(report.mode(), OutputShutdownMode::Graceful);
    assert_eq!(closed_recv.recv_timeout(TIMEOUT).unwrap(), 2);
    assert_eq!(processor_drops.load(Ordering::Acquire), 2);
    assert_eq!(base.state(), AudioContextState::Closed);
    drop(live_gain); // surviving weak registration is harmless after exact registry retirement
}

#[test]
fn explicit_silent_and_controller_drop_publish_one_terminal_closed_after_reclaim() {
    for explicit in [true, false] {
        let closed_count = Arc::new(AtomicUsize::new(0));
        let closed_count_for_handler = Arc::clone(&closed_count);
        let processor_drops = Arc::new(AtomicUsize::new(0));
        let drops_seen_by_handler = Arc::clone(&processor_drops);
        let event_pair = injected_event_dispatch_setup_with_handlers(move |event_loop| {
            event_loop.set_handler(
                EventType::StateChange,
                EventHandler::Multiple(Box::new(move |_| {
                    assert_eq!(drops_seen_by_handler.load(Ordering::Acquire), 2);
                    closed_count_for_handler.fetch_add(1, Ordering::AcqRel);
                })),
            );
        })
        .unwrap();
        let fixture = lifecycle_fixture_with_event_setup(false, false, 8, event_pair);
        let base = fixture.take_exact_base();
        let live_gain = base
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain()
            .unwrap()
            .commit(gain_payload_with_drop_probe(&processor_drops))
            .unwrap();
        let surviving_base = base.clone();
        assert_eq!(base.state(), AudioContextState::Running);
        let reader_base = base.clone();
        let reader_stop = Arc::new(AtomicBool::new(false));
        let reader_stop_thread = Arc::clone(&reader_stop);
        let saw_closed = Arc::new(AtomicBool::new(false));
        let saw_closed_thread = Arc::clone(&saw_closed);
        let reopened = Arc::new(AtomicBool::new(false));
        let reopened_thread = Arc::clone(&reopened);
        let state_reader = thread::spawn(move || {
            while !reader_stop_thread.load(Ordering::Acquire) {
                let state = reader_base.state();
                if saw_closed_thread.load(Ordering::Acquire) && state != AudioContextState::Closed {
                    reopened_thread.store(true, Ordering::Release);
                }
                if state == AudioContextState::Closed {
                    saw_closed_thread.store(true, Ordering::Release);
                }
                std::hint::spin_loop();
            }
        });
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
        let applied_deadline = std::time::Instant::now() + TIMEOUT;
        while base.applied_control_batch_sequence() < 2 {
            assert!(
                std::time::Instant::now() < applied_deadline,
                "Gain batch did not apply before Silent destruction-order probe"
            );
            thread::sleep(Duration::from_millis(1));
        }
        let receipt = if explicit {
            controller.shutdown_silently()
        } else {
            let receipt = controller.receipt();
            drop(controller);
            receipt
        };
        let report = confirmed(wait_receipt(receipt));
        assert_eq!(report.mode(), OutputShutdownMode::Silent);
        assert_eq!(closed_count.load(Ordering::Acquire), 1);
        assert!(report.event_issue().is_none());
        assert_eq!(surviving_base.state(), AudioContextState::Closed);
        surviving_base.set_state(AudioContextState::Suspended);
        surviving_base.set_state(AudioContextState::Running);
        assert_eq!(surviving_base.state(), AudioContextState::Closed);
        let visibility_deadline = std::time::Instant::now() + TIMEOUT;
        while !saw_closed.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < visibility_deadline,
                "surviving base clone did not observe terminal Closed"
            );
            thread::sleep(Duration::from_millis(1));
        }
        reader_stop.store(true, Ordering::Release);
        state_reader.join().unwrap();
        assert!(!reopened.load(Ordering::Acquire));
        assert_eq!(
            surviving_base.send_event_with(EventDispatch::sink_change),
            Err(ControlEventSendOutcome::AdmissionRejected(
                AdmissionError::Sealed
            ))
        );
        assert_eq!(closed_count.load(Ordering::Acquire), 1);
        assert_eq!(processor_drops.load(Ordering::Acquire), 2);
        drop(live_gain);
    }
}

#[test]
fn endpoint_death_racing_ready_forces_silent_final_classification() {
    let close_rendered = Arc::new(AtomicBool::new(false));
    let closed_count = Arc::new(AtomicUsize::new(0));
    let closed_for_handler = Arc::clone(&closed_count);
    let setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |_| {
                closed_for_handler.fetch_add(1, Ordering::AcqRel);
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
    let lifecycle = start(
        TestPrepared::new(
            format(48_000.),
            PreparedBehavior::LateDeath(Arc::clone(&close_rendered)),
        ),
        fixture,
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
    assert_eq!(base.state(), AudioContextState::Closed);
    assert_eq!(closed_count.load(Ordering::Acquire), 1);
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
    let closed_count = Arc::new(AtomicUsize::new(0));
    let closed_for_handler = Arc::clone(&closed_count);
    let setup = injected_event_dispatch_setup_with_handlers(move |events| {
        events.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |_| {
                closed_for_handler.fetch_add(1, Ordering::AcqRel);
            })),
        );
    })
    .unwrap();
    let fixture = lifecycle_fixture_with_event_setup(false, false, 8, setup);
    let base = fixture.take_exact_base();
    let lifecycle = start(
        TestPrepared::new(format(48_000.), PreparedBehavior::Partial),
        fixture,
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
    assert_eq!(base.state(), AudioContextState::Closed);
    assert_eq!(closed_count.load(Ordering::Acquire), 1);
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
    let fixture = lifecycle_fixture();
    let base = fixture.take_exact_base();
    let constructor = base.injected_node_constructor().unwrap();
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
    let fixture = lifecycle_fixture_with_capacity(false, false, 128);
    let base = Arc::new(fixture.take_exact_base());
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
        let base = Arc::clone(&base);
        let keep_producing = Arc::clone(&keep_producing);
        let produced = Arc::clone(&produced);
        thread::spawn(move || {
            while keep_producing.load(Ordering::Acquire) {
                if let Ok(transaction) = base.injected_node_constructor().unwrap().try_begin_gain()
                {
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
    let expected_base_address = fixture.base.address();
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
        base,
        output_events,
        event_loop,
    } = parts
    else {
        panic!("post-start transfer failure must return running resources");
    };
    assert_eq!(base.address(), expected_base_address);
    let controller = start_running(
        endpoint,
        owner,
        base,
        output_events,
        event_loop,
        &ThreadSpawner,
    )
    .ok()
    .unwrap();
    assert_eq!(controller.base().address(), expected_base_address);
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
