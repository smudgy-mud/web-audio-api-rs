//! Deterministic tests for the private concrete Gain construction boundary.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::injected_connections::{
    InjectedConnectionOperationError, InjectedConnectionOperationOutcome,
    InjectedConnectionOperationTestPoint,
};
use super::injected_control::{
    injected_control_channel, CommitControlOutcome, InjectedControlLifecycleOwner,
    InjectedControlProducer,
};
use super::injected_ids::{injected_node_id_pair, InjectedNodeIdAllocator};
use super::injected_node_construction::{
    InjectedConstructedGain, InjectedGainConstructionError, InjectedGainPayload,
    InjectedNodeConstructor, InjectedNodeConstructorBuildError,
};
use super::injected_node_lifetime::{
    injected_node_lifetime_registry, InjectedNodeLifetimeOwner, InjectedNodeLifetimeRegistrar,
    NodeLifetimeDriveOutcome,
};
use super::{
    AudioContextRegistration, AudioContextState, AudioNodeId, ConcreteBaseAudioContext,
    InjectedContextAdmissionGate,
};
use crate::events::{EventDispatch, EventLoop};
use crate::message::{ControlBatchApplied, ControlBatchSender, ControlMessage};
use crate::node::{
    AudioNode, AudioNodeOptions, ChannelConfigInner, ChannelCountMode, ChannelInterpretation,
    GainNode, GainOptions,
};
use crate::param::{
    injected_audio_param_raw_parts, AudioParam, AudioParamDescriptor, AudioParamInitialValue,
    AutomationRate, InjectedAudioParamProcessor,
};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope, RenderThread,
};
use crate::stats::AudioStats;

const SLOT_CAPACITY: usize = 8;

struct SilentProcessor;

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

struct PanicDropProcessor {
    dropped: Arc<AtomicBool>,
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

impl Drop for PanicDropProcessor {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
        panic!("hostile rejected Gain processor destructor");
    }
}

struct BlockingPanicDropProcessor {
    entered: crossbeam_channel::Sender<()>,
    release: crossbeam_channel::Receiver<()>,
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

impl Drop for BlockingPanicDropProcessor {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.release.recv().unwrap();
        panic!("hostile blocked processor destructor");
    }
}

fn channel_config() -> ChannelConfigInner {
    ChannelConfigInner {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    }
}

fn payload(param: Box<dyn AudioProcessor>, gain: Box<dyn AudioProcessor>) -> InjectedGainPayload {
    payload_with_param(
        InjectedAudioParamProcessor::from_boxed_for_test(param),
        gain,
    )
}

fn payload_with_param(
    param: InjectedAudioParamProcessor,
    gain: Box<dyn AudioProcessor>,
) -> InjectedGainPayload {
    InjectedGainPayload {
        param_processor: param,
        gain_processor: gain,
        param_channel_config: channel_config(),
        gain_channel_config: channel_config(),
        initial_value: AudioParamInitialValue::new(1.),
    }
}

fn gain_descriptor() -> AudioParamDescriptor {
    AudioParamDescriptor {
        name: String::new(),
        automation_rate: AutomationRate::A,
        default_value: 1.,
        min_value: f32::MIN,
        max_value: f32::MAX,
    }
}

fn construct_silent_gain(harness: &Harness) -> InjectedConstructedGain {
    harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(payload(
            Box::new(SilentProcessor),
            Box::new(SilentProcessor),
        ))
        .unwrap()
}

struct Harness {
    producer: InjectedControlProducer,
    lifecycle: Option<InjectedControlLifecycleOwner>,
    allocator: InjectedNodeIdAllocator,
    registrar: InjectedNodeLifetimeRegistrar,
    base: Option<ConcreteBaseAudioContext>,
    renderer: Option<RenderThread>,
    lifetimes: InjectedNodeLifetimeOwner,
    gc: Option<thread::JoinHandle<()>>,
}

impl Harness {
    fn new(ordinary_capacity: usize) -> Self {
        Self::new_with_suspension(ordinary_capacity, false)
    }

    fn new_with_suspension(ordinary_capacity: usize, suspended: bool) -> Self {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, lifecycle, render_init) =
            injected_control_channel(gate, ordinary_capacity, suspended).unwrap();
        let (allocator, node_ids, graph) = injected_node_id_pair(0);
        let (registrar, bootstrap) =
            injected_node_lifetime_registry(SLOT_CAPACITY, &producer, node_ids, graph)
                .ok()
                .unwrap();
        let constructor =
            InjectedNodeConstructor::new(producer.clone(), allocator.clone(), registrar.clone())
                .ok()
                .unwrap();

        let state = Arc::new(AtomicU8::new(if suspended {
            AudioContextState::Suspended as u8
        } else {
            AudioContextState::Running as u8
        }));
        let frames = Arc::new(AtomicU64::new(0));
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let bound = render_init
            .build_render_thread(
                bootstrap,
                48_000.,
                2,
                Arc::clone(&state),
                Arc::clone(&frames),
                AudioStats::new(),
                event_send.clone(),
            )
            .ok()
            .unwrap();
        let (mut renderer, lifetimes) = bound.into_render_thread_for_test();
        let gc = renderer.spawn_joinable_garbage_collector_thread().unwrap();
        let base = ConcreteBaseAudioContext::new_injected_node_construction_base(
            48_000.,
            2,
            state,
            frames,
            constructor,
            event_send,
            event_loop,
            false,
        );
        Self {
            producer,
            lifecycle: Some(lifecycle),
            allocator,
            registrar,
            base: Some(base),
            renderer: Some(renderer),
            lifetimes,
            gc: Some(gc),
        }
    }

    fn base(&self) -> &ConcreteBaseAudioContext {
        self.base.as_ref().unwrap()
    }

    fn callback(&mut self) {
        self.renderer
            .as_mut()
            .unwrap()
            .render(&mut [] as &mut [f32]);
    }

    fn render_quantum(&mut self) {
        let mut output = [0.; crate::RENDER_QUANTUM_SIZE * 2];
        self.renderer.as_mut().unwrap().render(&mut output);
    }

    fn install_persistent_destination_for_recycle_test(&mut self) {
        let mut destination = self.allocator.try_reserve(1).unwrap();
        let id = destination.id(0);
        assert_eq!(id, AudioNodeId(0));
        let reclaim_id = destination.take_reclaim_node(0).unwrap();
        destination.commit().unwrap();
        self.producer
            .try_commit_prevalidated_for_test(vec![ControlMessage::RegisterNode {
                id,
                reclaim_id,
                node: Box::new(SilentProcessor),
                inputs: 1,
                outputs: 1,
                channel_config: channel_config(),
            }])
            .unwrap();
        self.callback();
        self.wait_for_transport_idle();
    }

    fn assert_slots_vacant(&self) {
        assert_eq!(
            self.lifetimes.slot_phase_counts_for_test(),
            [SLOT_CAPACITY, 0, 0, 0, 0, 0]
        );
    }

    fn assert_two_requested(&self) {
        assert_eq!(
            self.lifetimes.slot_phase_counts_for_test(),
            [SLOT_CAPACITY - 2, 0, 0, 2, 0, 0]
        );
    }

    fn assert_two_quarantined(&self) {
        assert_eq!(
            self.lifetimes.slot_phase_counts_for_test(),
            [SLOT_CAPACITY - 2, 0, 0, 0, 2, 0]
        );
    }

    fn wait_for_transport_idle(&self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if self.producer.accounting() == (0, 0, 0, 0) {
                return;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "transport credits did not retire");
            let _ = self
                .lifecycle
                .as_ref()
                .expect("open harness retains lifecycle owner")
                .credit_activity_receiver()
                .recv_timeout(remaining.min(Duration::from_millis(100)));
        }
    }

    fn drive_node_lifetimes_until_vacant(&mut self) {
        self.drive_node_lifetimes_until_counts([SLOT_CAPACITY, 0, 0, 0, 0, 0]);
    }

    fn drive_node_lifetimes_until_counts(&mut self, expected: [usize; 6]) {
        for _ in 0..64 {
            match self.lifetimes.try_drive_once() {
                NodeLifetimeDriveOutcome::Idle
                    if self.lifetimes.slot_phase_counts_for_test() == expected =>
                {
                    return;
                }
                NodeLifetimeDriveOutcome::Submitted { .. } => {
                    self.render_quantum();
                    self.wait_for_transport_idle();
                }
                NodeLifetimeDriveOutcome::Idle | NodeLifetimeDriveOutcome::Reconciled { .. } => {}
                NodeLifetimeDriveOutcome::Retry { .. } => thread::yield_now(),
                outcome => panic!("unexpected lifetime-drive outcome: {outcome:?}"),
            }
        }
        panic!("node lifetimes did not reach {expected:?} within the bounded test drive");
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.base.take());
        drop(self.renderer.take());
        if let Some(gc) = self.gc.take() {
            gc.join().unwrap();
        }
    }
}

#[test]
fn constructor_brand_mismatch_returns_all_three_capabilities_intact() {
    let gate_a = InjectedContextAdmissionGate::new();
    let (control_a, _lifecycle_a, _render_a) = injected_control_channel(gate_a, 1, false).unwrap();
    let (allocator_a, ids_a, graph_a) = injected_node_id_pair(10);
    let (registrar_a, bootstrap_a) = injected_node_lifetime_registry(2, &control_a, ids_a, graph_a)
        .ok()
        .unwrap();

    let gate_b = InjectedContextAdmissionGate::new();
    let (control_b, _lifecycle_b, _render_b) = injected_control_channel(gate_b, 1, false).unwrap();
    let (allocator_b, ids_b, graph_b) = injected_node_id_pair(20);
    let (registrar_b, bootstrap_b) = injected_node_lifetime_registry(2, &control_b, ids_b, graph_b)
        .ok()
        .unwrap();

    let failure = InjectedNodeConstructor::new(control_a, allocator_b, registrar_a)
        .err()
        .unwrap();
    assert_eq!(
        failure.error,
        InjectedNodeConstructorBuildError::MismatchedCapabilities
    );
    // Operational reuse proves every returned weak capability kept its exact brand.
    let exact_a = InjectedNodeConstructor::new(failure.control, allocator_a, failure.lifetimes)
        .ok()
        .unwrap();
    let exact_b = InjectedNodeConstructor::new(control_b, failure.allocator, registrar_b)
        .ok()
        .unwrap();
    drop(exact_a.try_begin_gain().unwrap());
    drop(exact_b.try_begin_gain().unwrap());
    drop((bootstrap_a, bootstrap_b));
}

#[test]
fn event_only_injected_base_keeps_the_legacy_four_command_gain_path() {
    let (render_send, render_recv) = crossbeam_channel::unbounded();
    let (event_send, event_recv) = crossbeam_channel::unbounded::<EventDispatch>();
    let event_loop = EventLoop::new(event_recv);
    let (_node_return, node_consumer) = llq::Queue::new().split();
    let base = ConcreteBaseAudioContext::new_injected(
        48_000.,
        2,
        Arc::new(AtomicU8::new(AudioContextState::Running as u8)),
        Arc::new(AtomicU64::new(0)),
        render_send.clone(),
        ControlBatchSender::new(render_send),
        ControlBatchApplied::default(),
        event_send,
        event_loop,
        true,
        node_consumer,
        InjectedContextAdmissionGate::new(),
    );
    assert!(base.injected_node_constructor().is_none());
    for message in render_recv.try_iter() {
        drop(message); // discard destination/listener initialization
    }

    let gain = GainNode::new(&base, GainOptions::default());
    let messages = render_recv.try_iter().collect::<Vec<_>>();
    assert_eq!(messages.len(), 4);
    let gain_id = gain.registration().id();
    let param_id = gain.gain().registration().id();
    assert!(matches!(
        &messages[0],
        ControlMessage::RegisterNode { id, .. } if *id == param_id
    ));
    assert!(matches!(
        &messages[1],
        ControlMessage::NodeMessage { id, .. } if *id == param_id
    ));
    assert!(matches!(
        &messages[2],
        ControlMessage::RegisterNode { id, .. } if *id == gain_id
    ));
    assert!(matches!(
        &messages[3],
        ControlMessage::ConnectNode { from, to, output: 0, input: usize::MAX }
            if *from == param_id && *to == gain_id
    ));

    gain.gain().set_value(0.25);
    assert_eq!(gain.gain().value(), 0.25);
    assert!(matches!(
        render_recv.recv().unwrap(),
        ControlMessage::NodeMessage { id, .. } if id == param_id
    ));
    gain.gain().set_automation_rate(AutomationRate::K);
    assert_eq!(gain.gain().automation_rate(), AutomationRate::K);
    assert!(matches!(
        render_recv.recv().unwrap(),
        ControlMessage::NodeMessage { id, .. } if id == param_id
    ));
}

#[test]
fn invalid_gain_rolls_back_exact_ids_slots_and_transport_before_any_batch() {
    let mut harness = Harness::new(1);
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        GainNode::new(
            harness.base(),
            GainOptions {
                gain: f32::NAN,
                audio_node_options: AudioNodeOptions::default(),
            },
        )
    }));
    assert!(result.is_err());
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    harness.assert_slots_vacant();
    assert!(harness.lifetimes.request_activity_receiver().is_empty());
    harness.callback();
    assert_eq!(harness.base().applied_control_batch_sequence(), 0);

    let ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(0));
    assert_eq!(ids.id(1), AudioNodeId(1));
}

#[test]
fn admitted_gain_is_one_real_four_command_batch_and_handles_request_lifetime_teardown() {
    let mut harness = Harness::new(1);
    let gain = GainNode::new(
        harness.base(),
        GainOptions {
            gain: 0.375,
            audio_node_options: AudioNodeOptions::default(),
        },
    );
    assert_eq!(gain.registration().id(), AudioNodeId(0));
    assert_eq!(gain.gain().registration().id(), AudioNodeId(1));
    assert_eq!(gain.gain().value(), 0.375);
    assert_eq!(harness.producer.accounting(), (4, 1, 1, 0));
    assert_eq!(harness.base().applied_control_batch_sequence(), 0);

    harness.callback();
    assert_eq!(harness.base().applied_control_batch_sequence(), 1);
    harness.wait_for_transport_idle();
    assert_eq!(
        harness.lifetimes.slot_phase_counts_for_test(),
        [SLOT_CAPACITY - 2, 0, 2, 0, 0, 0]
    );

    drop(gain);
    harness.assert_two_requested();
    assert!(harness
        .lifetimes
        .request_activity_receiver()
        .try_recv()
        .is_ok());
    // Handle Drop only publishes a request hint; the later lifecycle driver owns teardown sends.
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn ordinary_endpoint_attachment_rejects_a_foreign_exact_base_and_fails_both_contexts_closed() {
    let mut first = Harness::new(4);
    let second = Harness::new(4);
    let constructed = construct_silent_gain(&first);
    first.callback();
    first.wait_for_transport_idle();

    let attached = panic::catch_unwind(AssertUnwindSafe(|| {
        AudioContextRegistration::from_injected_with_connection(
            constructed.gain_id,
            second.base().clone(),
            constructed.gain_registration,
            constructed.gain_connection,
            super::InjectedConnectionEndpointKind::AudioNode,
            1,
            1,
        )
    }));
    assert!(attached.is_err());
    for harness in [&first, &second] {
        assert!(matches!(
            harness
                .base()
                .injected_node_constructor()
                .unwrap()
                .try_begin_gain(),
            Err(InjectedGainConstructionError::Control(
                super::injected_control::InjectedControlError::ProtocolViolation
            ))
        ));
    }
    drop(constructed.param_registration);
}

#[test]
fn exact_public_connect_disconnect_remain_not_supported_without_host_or_transport_mutation() {
    let mut harness = Harness::new(8);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let before_sequence = harness.base().applied_control_batch_sequence();

    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        source.connect(&destination);
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        source.disconnect_dest(&destination);
    }))
    .is_err());
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    assert_eq!(
        harness.base().applied_control_batch_sequence(),
        before_sequence
    );
    assert_eq!(
        harness
            .base()
            .injected_node_constructor()
            .unwrap()
            .connection_edge_count_for_test(),
        0
    );
}

#[test]
fn internal_exact_connect_duplicate_and_disconnect_are_fixed_mirror_transactions() {
    let mut harness = Harness::new(8);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();

    assert!(matches!(
        constructor.connect_exact(&source_cap, &destination_cap, 0, 0),
        Ok(InjectedConnectionOperationOutcome::Committed(
            CommitControlOutcome::Enqueued { .. }
        ))
    ));
    assert_eq!(constructor.connection_edge_count_for_test(), 1);
    let duplicate_sequence = constructor.last_submitted_batch_sequence();
    let duplicate_accounting = harness.producer.accounting();
    assert_eq!(
        constructor.connect_exact(&source_cap, &destination_cap, 0, 0),
        Ok(InjectedConnectionOperationOutcome::Noop)
    );
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        duplicate_sequence
    );
    assert_eq!(harness.producer.accounting(), duplicate_accounting);
    assert_eq!(
        constructor.connect_exact(&source_cap, &destination_cap, 1, 0),
        Err(InjectedConnectionOperationError::InvalidPort)
    );
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        duplicate_sequence
    );
    assert_eq!(harness.producer.accounting(), duplicate_accounting);
    harness.callback();
    harness.wait_for_transport_idle();

    assert!(matches!(
        constructor.disconnect_exact(&source_cap, None, Some(&destination_cap), None),
        Ok(InjectedConnectionOperationOutcome::Committed(
            CommitControlOutcome::Enqueued { .. }
        ))
    ));
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(
        constructor.disconnect_exact(&source_cap, None, None, None),
        Ok(InjectedConnectionOperationOutcome::Noop)
    );
    assert_eq!(
        constructor.disconnect_exact(&source_cap, None, Some(&destination_cap), None),
        Err(InjectedConnectionOperationError::Unconnected)
    );
}

#[test]
fn multi_edge_disconnect_not_accepted_restores_every_command_before_admission_releases() {
    let mut harness = Harness::new(12);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination_a = GainNode::new(harness.base(), GainOptions::default());
    let destination_b = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_a_cap = destination_a
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_b_cap = destination_b
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    constructor
        .connect_exact(&source_cap, &destination_a_cap, 0, 0)
        .unwrap();
    constructor
        .connect_exact(&source_cap, &destination_b_cap, 0, 0)
        .unwrap();
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(constructor.connection_edge_count_for_test(), 2);
    let sequence_before_disconnect = constructor.last_submitted_batch_sequence();

    let (commit_send, commit_recv) = crossbeam_channel::bounded(1);
    let (commit_release_send, commit_release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::BeforeCommit,
        commit_send,
        commit_release_recv,
        false,
    );
    let (rollback_send, rollback_recv) = crossbeam_channel::bounded(1);
    let (rollback_release_send, rollback_release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::RejectedRollback,
        rollback_send,
        rollback_release_recv,
        false,
    );
    let operation_base = harness.base().clone();
    let disconnect = thread::spawn(move || {
        operation_base
            .injected_node_constructor()
            .unwrap()
            .disconnect_exact(&source_cap, None, None, None)
    });
    commit_recv.recv().unwrap();

    // Make commit return a healthy, retryable NotAccepted outcome after its exact two-command
    // batch has acquired every bounded authority.
    let (state_entered_send, state_entered_recv) = crossbeam_channel::bounded(1);
    let (state_release_send, state_release_recv) = crossbeam_channel::bounded(1);
    let state_producer = harness.producer.clone();
    let state_holder = thread::spawn(move || {
        state_producer.hold_transport_state_for_test(state_entered_send, state_release_recv);
    });
    state_entered_recv.recv().unwrap();
    commit_release_send.send(()).unwrap();
    rollback_recv.recv().unwrap();
    assert_eq!(harness.producer.accounting(), (2, 1, 1, 0));
    state_release_send.send(()).unwrap();
    state_holder.join().unwrap();

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    rollback_release_send.send(()).unwrap();
    assert_eq!(
        disconnect.join().unwrap(),
        Err(InjectedConnectionOperationError::Control(
            super::injected_control::InjectedControlError::Contended
        ))
    );
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    drop(drained);
    close_thread.join().unwrap();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    assert_eq!(constructor.connection_edge_count_for_test(), 2);
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        sequence_before_disconnect
    );
}

#[test]
fn post_seal_duplicate_and_no_match_are_rejected_before_noop_inspection() {
    let mut harness = Harness::new(12);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    let unconnected = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let unconnected_cap = unconnected
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    constructor
        .connect_exact(&source_cap, &destination_cap, 0, 0)
        .unwrap();
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(
        constructor.connect_exact(&source_cap, &destination_cap, 0, 0),
        Ok(InjectedConnectionOperationOutcome::Noop)
    );
    assert_eq!(
        constructor.disconnect_exact(&unconnected_cap, None, None, None),
        Ok(InjectedConnectionOperationOutcome::Noop)
    );

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let sealed = Err(InjectedConnectionOperationError::Control(
        super::injected_control::InjectedControlError::Sealed,
    ));
    assert_eq!(
        constructor.connect_exact(&source_cap, &destination_cap, 0, 0),
        sealed
    );
    assert_eq!(
        constructor.disconnect_exact(&unconnected_cap, None, None, None),
        sealed
    );
    let (snapshot, drained) = retirement.retire_and_wait();
    assert!(snapshot.is_drained());
    drop(drained);
}

#[test]
fn incident_cleanup_wins_serializer_then_stale_generation_cannot_send_after_id_reuse() {
    let mut harness = Harness::new(8);
    harness.install_persistent_destination_for_recycle_test();
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_id = source.registration().id();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    constructor
        .connect_exact(&source_cap, &destination_cap, 0, 0)
        .unwrap();
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(constructor.connection_edge_count_for_test(), 1);

    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::BeforeSerializer,
        entered_send,
        release_recv,
        false,
    );
    let operation_base = harness.base().clone();
    let stale_source = source_cap.clone();
    let live_destination = destination_cap.clone();
    let operation = thread::spawn(move || {
        operation_base
            .injected_node_constructor()
            .unwrap()
            .connect_exact(&stale_source, &live_destination, 0, 0)
    });
    entered_recv.recv().unwrap();

    drop(source);
    harness.drive_node_lifetimes_until_counts([SLOT_CAPACITY - 2, 0, 2, 0, 0, 0]);
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
    let replacement = GainNode::new(harness.base(), GainOptions::default());
    assert_eq!(replacement.registration().id(), source_id);
    harness.callback();
    harness.wait_for_transport_idle();
    let sequence_before_stale_release = constructor.last_submitted_batch_sequence();

    release_send.send(()).unwrap();
    assert_eq!(
        operation.join().unwrap(),
        Err(InjectedConnectionOperationError::ForeignEndpoint)
    );
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        sequence_before_stale_release
    );
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
}

#[test]
fn incident_cleanup_retries_serializer_contention_and_prevents_id_reuse_until_success() {
    let mut harness = Harness::new(8);
    harness.install_persistent_destination_for_recycle_test();
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_id = source.registration().id();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    constructor
        .connect_exact(&source_cap, &destination_cap, 0, 0)
        .unwrap();
    harness.callback();
    harness.wait_for_transport_idle();

    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::BeforeReserve,
        entered_send,
        release_recv,
        false,
    );
    let operation_base = exact_base.clone();
    let operation_source = source_cap.clone();
    let operation_destination = destination_cap.clone();
    let disconnect = thread::spawn(move || {
        operation_base
            .injected_node_constructor()
            .unwrap()
            .disconnect_exact(&operation_source, None, Some(&operation_destination), None)
    });
    entered_recv.recv().unwrap();
    drop(source);

    let mut observed_retry = false;
    for _ in 0..32 {
        match harness.lifetimes.try_drive_once() {
            NodeLifetimeDriveOutcome::Submitted { .. } => {
                harness.render_quantum();
                harness.wait_for_transport_idle();
            }
            NodeLifetimeDriveOutcome::Retry { .. } => {
                observed_retry = true;
                break;
            }
            NodeLifetimeDriveOutcome::Idle | NodeLifetimeDriveOutcome::Reconciled { .. } => {}
            outcome => panic!("unexpected lifetime-drive outcome: {outcome:?}"),
        }
    }
    assert!(
        observed_retry,
        "incident cleanup never exposed benign contention"
    );
    let while_contended = harness.allocator.try_reserve(1).unwrap();
    assert_ne!(while_contended.id(0), source_id);
    drop(while_contended);

    release_send.send(()).unwrap();
    assert!(matches!(
        disconnect.join().unwrap(),
        Ok(InjectedConnectionOperationOutcome::Committed(_))
    ));
    harness.callback();
    harness.wait_for_transport_idle();
    harness.drive_node_lifetimes_until_counts([SLOT_CAPACITY - 2, 0, 2, 0, 0, 0]);
    let after_cleanup = harness.allocator.try_reserve(3).unwrap();
    assert!((0..3).any(|index| after_cleanup.id(index) == source_id));
}

#[test]
fn preboxed_preparation_panic_latches_before_zero_credit_admission_releases_to_close() {
    let mut harness = Harness::new(8);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::BeforeReserve,
        entered_send,
        release_recv,
        true,
    );
    let operation_base = harness.base().clone();
    let operation = thread::spawn(move || {
        panic::catch_unwind(AssertUnwindSafe(|| {
            operation_base
                .injected_node_constructor()
                .unwrap()
                .connect_exact(&source_cap, &destination_cap, 0, 0)
        }))
    });
    entered_recv.recv().unwrap();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    release_send.send(()).unwrap();
    assert!(operation.join().unwrap().is_err());
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    let sealed = match drained.finish() {
        Ok(sealed) => sealed,
        Err(_) => panic!("drained exact close must retain its Close slot"),
    };
    assert!(sealed.degradation.prior_transport_failure);
    drop(sealed);
    close_thread.join().unwrap();
}

#[test]
fn prepared_batch_panic_retains_fence_after_primary_admission_transfer_and_recovers_credits() {
    let mut harness = Harness::new(8);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::BeforeCommit,
        entered_send,
        release_recv,
        true,
    );
    let operation_base = harness.base().clone();
    let operation = thread::spawn(move || {
        panic::catch_unwind(AssertUnwindSafe(|| {
            operation_base
                .injected_node_constructor()
                .unwrap()
                .connect_exact(&source_cap, &destination_cap, 0, 0)
        }))
    });
    entered_recv.recv().unwrap();
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    release_send.send(()).unwrap();
    assert!(operation.join().unwrap().is_err());
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    let sealed = match drained.finish() {
        Ok(sealed) => sealed,
        Err(_) => panic!("drained exact close must retain its Close slot"),
    };
    assert!(sealed.degradation.prior_transport_failure);
    drop(sealed);
    close_thread.join().unwrap();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn rejected_rollback_panic_keeps_failure_admission_until_fail_closed_latch() {
    let mut harness = Harness::new(8);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    let (commit_send, commit_recv) = crossbeam_channel::bounded(1);
    let (commit_release_send, commit_release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::BeforeCommit,
        commit_send,
        commit_release_recv,
        false,
    );
    let (rollback_send, rollback_recv) = crossbeam_channel::bounded(1);
    let (rollback_release_send, rollback_release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::RejectedRollback,
        rollback_send,
        rollback_release_recv,
        true,
    );
    let operation_base = harness.base().clone();
    let operation = thread::spawn(move || {
        operation_base
            .injected_node_constructor()
            .unwrap()
            .connect_exact(&source_cap, &destination_cap, 0, 0)
    });
    commit_recv.recv().unwrap();
    let (state_entered_send, state_entered_recv) = crossbeam_channel::bounded(1);
    let (state_release_send, state_release_recv) = crossbeam_channel::bounded(1);
    let state_producer = harness.producer.clone();
    let state_holder = thread::spawn(move || {
        state_producer.hold_transport_state_for_test(state_entered_send, state_release_recv);
    });
    state_entered_recv.recv().unwrap();
    commit_release_send.send(()).unwrap();
    rollback_recv.recv().unwrap();
    state_release_send.send(()).unwrap();
    state_holder.join().unwrap();

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    rollback_release_send.send(()).unwrap();
    assert_eq!(
        operation.join().unwrap(),
        Err(InjectedConnectionOperationError::RejectedPayloadPanicked)
    );
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    let sealed = match drained.finish() {
        Ok(sealed) => sealed,
        Err(_) => panic!("drained exact close must retain its Close slot"),
    };
    assert!(sealed.degradation.prior_transport_failure);
    drop(sealed);
    close_thread.join().unwrap();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
}

#[test]
fn accepted_mirror_panic_latches_before_accepted_admission_and_keeps_payload_queue_owned() {
    let mut harness = Harness::new(8);
    let source = GainNode::new(harness.base(), GainOptions::default());
    let destination = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let source_cap = source
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let exact_base = harness.base().clone();
    let constructor = exact_base.injected_node_constructor().unwrap();
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    constructor.hold_connection_operation_for_test(
        InjectedConnectionOperationTestPoint::AcceptedMutation,
        entered_send,
        release_recv,
        true,
    );
    let operation_base = harness.base().clone();
    let operation = thread::spawn(move || {
        panic::catch_unwind(AssertUnwindSafe(|| {
            operation_base
                .injected_node_constructor()
                .unwrap()
                .connect_exact(&source_cap, &destination_cap, 0, 0)
        }))
    });
    entered_recv.recv().unwrap();
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    release_send.send(()).unwrap();
    assert!(operation.join().unwrap().is_err());
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    let sealed = match drained.finish() {
        Ok(sealed) => sealed,
        Err(_) => panic!("drained exact close must retain its Close slot"),
    };
    assert!(sealed.degradation.prior_transport_failure);
    drop(sealed);
    close_thread.join().unwrap();
    // The host mirror never moved, while the accepted renderer payload remains queue-owned until
    // the callback consumes it. No accepted payload is rolled back or dropped on the caller.
    assert_eq!(constructor.connection_edge_count_for_test(), 0);
    harness.callback();
    let deadline = Instant::now() + Duration::from_secs(2);
    while harness.producer.accounting() != (0, 0, 0, 0) {
        assert!(
            Instant::now() < deadline,
            "accepted edge credits did not retire"
        );
        thread::yield_now();
    }
}

#[test]
fn exact_gain_value_is_one_fixed_command_and_survives_the_next_render_quantum() {
    let mut harness = Harness::new(4);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(gain.gain().value(), 1.);

    gain.gain().set_value(0.25);
    assert_eq!(gain.gain().value(), 0.25);
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));
    harness.render_quantum();
    harness.wait_for_transport_idle();
    assert_eq!(harness.base().applied_control_batch_sequence(), 2);
    // The render processor republishes its intrinsic value each quantum, so this proves the fixed
    // command reached the exact parameter instead of merely updating the host mirror.
    assert_eq!(gain.gain().value(), 0.25);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn exact_value_preserves_legacy_host_and_renderer_clamping() {
    let mut harness = Harness::new(4);
    let descriptor = AudioParamDescriptor {
        name: String::new(),
        automation_rate: AutomationRate::A,
        default_value: 0.5,
        min_value: 0.,
        max_value: 1.,
    };
    let (raw_parts, param_processor) = injected_audio_param_raw_parts(descriptor);
    let initial_value = raw_parts.set_initial_value_for_injected(0.5);
    let constructed = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(InjectedGainPayload {
            param_processor,
            gain_processor: Box::new(SilentProcessor),
            param_channel_config: channel_config(),
            gain_channel_config: channel_config(),
            initial_value,
        })
        .unwrap();
    let registration = AudioContextRegistration::from_injected(
        constructed.param_id,
        harness.base().clone(),
        constructed.param_registration,
    );
    let param =
        AudioParam::from_injected_raw_parts(registration, raw_parts, constructed.param_mutation);
    harness.callback();
    harness.wait_for_transport_idle();

    param.set_value(2.);
    assert_eq!(param.value(), 1.);
    harness.render_quantum();
    harness.wait_for_transport_idle();
    assert_eq!(param.value(), 1.);
    drop(param);
    drop(constructed.gain_registration);
}

#[test]
fn more_than_timeline_capacity_fixed_values_coalesce_without_render_allocation() {
    let mut harness = Harness::new(64);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();

    for value in 0..40 {
        gain.gain().set_value(value as f32 / 10.);
    }
    assert_eq!(harness.producer.accounting(), (40, 40, 40, 0));
    alloc_counter::deny_alloc(|| harness.render_quantum());
    harness.wait_for_transport_idle();
    assert_eq!(gain.gain().value(), 3.9);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn exact_processor_mirror_swap_rejects_before_acceptance_and_reuses_ids_slots() {
    let harness = Harness::new(4);
    let transaction = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap();
    let (_raw_a, mut param_a) = injected_audio_param_raw_parts(gain_descriptor());
    let (_raw_b, mut param_b) = injected_audio_param_raw_parts(gain_descriptor());
    param_a.swap_exact_processors_for_test(&mut param_b);

    let error = transaction
        .commit(payload_with_param(param_a, Box::new(SilentProcessor)))
        .err()
        .unwrap();
    assert_eq!(error, InjectedGainConstructionError::ProtocolViolation);
    harness.assert_slots_vacant();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    let ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(0));
    assert_eq!(ids.id(1), AudioNodeId(1));
}

#[test]
fn mirror_mismatch_restores_ids_before_hostile_destructor_and_holds_close_admission() {
    let mut harness = Harness::new(4);
    let transaction = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap();
    let (_raw_a, mut param_a) = injected_audio_param_raw_parts(gain_descriptor());
    let (_raw_b, mut param_b) = injected_audio_param_raw_parts(gain_descriptor());
    param_a.swap_exact_processors_for_test(&mut param_b);
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    let commit_thread = thread::spawn(move || {
        transaction.commit(payload_with_param(
            param_a,
            Box::new(BlockingPanicDropProcessor {
                entered: entered_send,
                release: release_recv,
            }),
        ))
    });
    entered_recv.recv().unwrap();

    harness.assert_slots_vacant();
    let ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(0));
    assert_eq!(ids.id(1), AudioNodeId(1));
    drop(ids);
    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    release_send.send(()).unwrap();
    assert_eq!(
        commit_thread.join().unwrap().err().unwrap(),
        InjectedGainConstructionError::RejectedPayloadPanicked
    );
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    drop(drained);
    close_thread.join().unwrap();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn same_context_foreign_raw_mirror_attachment_fails_closed() {
    let mut harness = Harness::new(4);
    let transaction = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap();
    let (_raw_exact, param_exact) = injected_audio_param_raw_parts(gain_descriptor());
    let (raw_foreign, _param_foreign) = injected_audio_param_raw_parts(gain_descriptor());
    let constructed = transaction
        .commit(payload_with_param(param_exact, Box::new(SilentProcessor)))
        .unwrap();
    let registration = AudioContextRegistration::from_injected(
        constructed.param_id,
        harness.base().clone(),
        constructed.param_registration,
    );
    let attached = panic::catch_unwind(AssertUnwindSafe(|| {
        AudioParam::from_injected_raw_parts(registration, raw_foreign, constructed.param_mutation)
    }));
    assert!(attached.is_err());
    assert!(matches!(
        harness
            .base()
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain(),
        Err(InjectedGainConstructionError::Control(
            super::injected_control::InjectedControlError::ProtocolViolation
        ))
    ));
    drop(constructed.gain_registration);
    harness.callback();
    harness.wait_for_transport_idle();
}

#[test]
fn stale_param_cap_rejects_a_recycled_numeric_id_with_a_new_slot_generation() {
    let mut harness = Harness::new(4);
    // Keep destination 0 permanent so ordinary Gain teardown can render/reclaim without making
    // the test graph itself structurally invalid. Production hosted output gets this from B4b.
    harness.install_persistent_destination_for_recycle_test();
    let first = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let stale_id = first.gain().registration().id();
    let (stale_raw, stale_cap) = first.gain().clone_injected_parts_for_test();

    drop(first);
    harness.drive_node_lifetimes_until_vacant();

    let second = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(payload(
            Box::new(SilentProcessor),
            Box::new(SilentProcessor),
        ))
        .unwrap();
    assert_eq!(second.param_id, stale_id);
    let recycled = AudioContextRegistration::from_injected(
        second.param_id,
        harness.base().clone(),
        second.param_registration,
    );
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        AudioParam::from_injected_raw_parts(recycled, stale_raw, stale_cap)
    }))
    .is_err());
    assert!(matches!(
        harness
            .base()
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain(),
        Err(InjectedGainConstructionError::Control(
            super::injected_control::InjectedControlError::ProtocolViolation
        ))
    ));
    drop(second.gain_registration);
}

#[test]
fn same_control_foreign_allocator_param_brand_fails_closed_before_attachment() {
    let mut harness = Harness::new(4);
    let (raw_parts, param_processor) = injected_audio_param_raw_parts(gain_descriptor());
    let initial_value = raw_parts.set_initial_value_for_injected(1.);
    let mut constructed = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(InjectedGainPayload {
            param_processor,
            gain_processor: Box::new(SilentProcessor),
            param_channel_config: channel_config(),
            gain_channel_config: channel_config(),
            initial_value,
        })
        .unwrap();
    let (foreign_allocator, _foreign_owner, _foreign_graph) = injected_node_id_pair(0);
    constructed
        .param_mutation
        .replace_node_id_identity_for_test(foreign_allocator.identity());
    let registration = AudioContextRegistration::from_injected(
        constructed.param_id,
        harness.base().clone(),
        constructed.param_registration,
    );
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        AudioParam::from_injected_raw_parts(registration, raw_parts, constructed.param_mutation)
    }))
    .is_err());
    assert!(matches!(
        harness
            .base()
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain(),
        Err(InjectedGainConstructionError::Control(
            super::injected_control::InjectedControlError::ProtocolViolation
        ))
    ));
    drop(constructed.gain_registration);
    harness.callback();
    harness.wait_for_transport_idle();
}

#[test]
fn foreign_control_param_cap_is_rejected_while_receiving_context_remains_operational() {
    let mut first = Harness::new(4);
    let mut second = Harness::new(4);
    let (raw_first, processor_first) = injected_audio_param_raw_parts(gain_descriptor());
    let initial_first = raw_first.set_initial_value_for_injected(1.);
    let constructed_first = first
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(InjectedGainPayload {
            param_processor: processor_first,
            gain_processor: Box::new(SilentProcessor),
            param_channel_config: channel_config(),
            gain_channel_config: channel_config(),
            initial_value: initial_first,
        })
        .unwrap();
    let (raw_second, processor_second) = injected_audio_param_raw_parts(gain_descriptor());
    let initial_second = raw_second.set_initial_value_for_injected(1.);
    let constructed_second = second
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap()
        .commit(InjectedGainPayload {
            param_processor: processor_second,
            gain_processor: Box::new(SilentProcessor),
            param_channel_config: channel_config(),
            gain_channel_config: channel_config(),
            initial_value: initial_second,
        })
        .unwrap();
    let registration = AudioContextRegistration::from_injected(
        constructed_first.param_id,
        first.base().clone(),
        constructed_first.param_registration,
    );
    // The foreign cap's mirror is deliberately paired with raw_second, isolating the control
    // identity mismatch from the already-covered mirror mismatch.
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        AudioParam::from_injected_raw_parts(
            registration,
            raw_second,
            constructed_second.param_mutation,
        )
    }))
    .is_err());
    drop(
        first
            .base()
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain()
            .unwrap(),
    );
    assert!(matches!(
        second
            .base()
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain(),
        Err(InjectedGainConstructionError::Control(
            super::injected_control::InjectedControlError::ProtocolViolation
        ))
    ));
    drop((
        raw_first,
        constructed_first.param_mutation,
        constructed_first.gain_registration,
        constructed_second.param_registration,
        constructed_second.gain_registration,
    ));
    first.callback();
    second.callback();
    first.wait_for_transport_idle();
    second.wait_for_transport_idle();
}

#[test]
fn invalid_and_deferred_exact_param_mutations_never_change_host_state_or_poison_rate() {
    let mut harness = Harness::new(4);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();

    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(panic::catch_unwind(AssertUnwindSafe(|| gain.gain().set_value(value))).is_err());
        assert_eq!(gain.gain().value(), 1.);
        assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    }

    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().set_automation_rate(AutomationRate::K)
    }))
    .is_err());
    assert_eq!(gain.gain().automation_rate(), AutomationRate::A);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().set_value_at_time(0.5, 0.)
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().linear_ramp_to_value_at_time(0.5, 1.)
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().exponential_ramp_to_value_at_time(0.5, 1.)
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().set_target_at_time(0.5, 0., 1.)
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().cancel_scheduled_values(0.)
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().cancel_and_hold_at_time(0.)
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        gain.gain().set_value_curve_at_time(&[0., 1., 0.], 0., 1.)
    }))
    .is_err());
    assert_eq!(gain.gain().value(), 1.);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));

    // Expected public panics occur after the serializer is released, so the supported operation
    // remains usable.
    gain.gain().set_value(0.75);
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(gain.gain().value(), 0.75);
}

#[test]
fn surviving_param_clone_is_inert_after_seal() {
    let mut harness = Harness::new(4);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let param = gain.gain().clone();

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (snapshot, drained) = retirement.retire_and_wait();
    assert!(snapshot.is_drained());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| param.set_value(0.5))).is_err());
    assert_eq!(param.value(), 1.);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    drop(drained);
    drop((gain, param));
}

#[test]
fn concurrent_param_clones_serialize_commit_and_finalizer_order_without_idle_credits() {
    let mut harness = Harness::new(4);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let first = gain.gain().clone();
    let second = gain.gain().clone();
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    first.hold_next_injected_value_finalizer_for_test(entered_send, release_recv, false);

    let first_thread = thread::spawn(move || {
        first.set_value(0.25);
    });
    entered_recv.recv().unwrap();
    let (attempted_send, attempted_recv) = crossbeam_channel::bounded(1);
    second.signal_next_injected_serializer_attempt_for_test(attempted_send);
    let second_thread = thread::spawn(move || {
        second.set_value(0.75);
    });
    attempted_recv.recv().unwrap();
    // The second clone waits on the per-param serializer before reserving any transport credit.
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));
    release_send.send(()).unwrap();
    first_thread.join().unwrap();
    second_thread.join().unwrap();
    assert_eq!(gain.gain().value(), 0.75);
    assert_eq!(harness.producer.accounting(), (2, 2, 2, 0));

    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(gain.gain().value(), 0.75);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn accepted_param_finalizer_holds_close_admission_and_panic_is_terminal() {
    for panics in [false, true] {
        let mut harness = Harness::new(4);
        let gain = GainNode::new(harness.base(), GainOptions::default());
        harness.callback();
        harness.wait_for_transport_idle();
        let param = gain.gain().clone();
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        param.hold_next_injected_value_finalizer_for_test(entered_send, release_recv, panics);
        let set_thread = thread::spawn(move || {
            panic::catch_unwind(AssertUnwindSafe(|| param.set_value(0.5))).is_ok()
        });
        entered_recv.recv().unwrap();

        let mut pending_close = None;
        if !panics {
            let retirement = harness
                .lifecycle
                .take()
                .unwrap()
                .try_begin_close()
                .ok()
                .unwrap();
            let (done_send, done_recv) = crossbeam_channel::bounded(1);
            let close_thread = thread::spawn(move || {
                let result = retirement.retire_and_wait();
                done_send.send(result).unwrap();
            });
            assert!(done_recv.try_recv().is_err());
            pending_close = Some((done_recv, close_thread));
        }
        release_send.send(()).unwrap();
        assert_eq!(set_thread.join().unwrap(), !panics);
        assert_eq!(gain.gain().value(), if panics { 1. } else { 0.5 });
        // A real quantum both routes the accepted command and lets the exact processor consume
        // its fixed pending scalar. A failed host finalizer must not suppress renderer ownership.
        harness.render_quantum();
        if panics {
            let retirement = harness
                .lifecycle
                .take()
                .unwrap()
                .try_begin_close()
                .ok()
                .unwrap();
            let (done_send, done_recv) = crossbeam_channel::bounded(1);
            let close_thread = thread::spawn(move || {
                done_send.send(retirement.retire_and_wait()).unwrap();
            });
            pending_close = Some((done_recv, close_thread));
        }
        assert_eq!(gain.gain().value(), 0.5);
        let (done_recv, close_thread) = pending_close.unwrap();
        let (snapshot, drained) = done_recv.recv().unwrap();
        assert!(snapshot.is_drained());
        let sealed = drained.finish().ok().unwrap();
        assert_eq!(sealed.degradation.prior_transport_failure, panics);
        drop(sealed);
        close_thread.join().unwrap();
    }
}

#[test]
fn saturated_and_not_accepted_value_updates_leave_host_unchanged_and_recover_credits() {
    // Capacity failure occurs before host publication and leaves the serializer reusable.
    let mut harness = Harness::new(1);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    let filler = harness
        .producer
        .try_begin_operation(1)
        .unwrap()
        .prepare_with(|| vec![ControlMessage::TestNop])
        .ok()
        .unwrap();
    harness.producer.try_commit(filler).ok().unwrap();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| gain.gain().set_value(0.25))).is_err());
    assert_eq!(gain.gain().value(), 1.);
    harness.callback();
    harness.wait_for_transport_idle();
    gain.gain().set_value(0.25);
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(gain.gain().value(), 0.25);

    let (lock_entered_send, lock_entered_recv) = crossbeam_channel::bounded(1);
    let (lock_release_send, lock_release_recv) = crossbeam_channel::bounded(1);
    let lock_producer = harness.producer.clone();
    let lock_thread = thread::spawn(move || {
        lock_producer.hold_transport_state_for_test(lock_entered_send, lock_release_recv);
    });
    lock_entered_recv.recv().unwrap();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| gain.gain().set_value(0.5))).is_err());
    assert_eq!(gain.gain().value(), 0.25);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    lock_release_send.send(()).unwrap();
    lock_thread.join().unwrap();

    // Force a terminal NotAccepted commit only after its one-command reservation exists.
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    let (rollback_entered_send, rollback_entered_recv) = crossbeam_channel::bounded(1);
    let (rollback_release_send, rollback_release_recv) = crossbeam_channel::bounded(1);
    gain.gain().hold_next_injected_value_rollback_for_test(
        rollback_entered_send,
        rollback_release_recv,
        false,
    );
    harness
        .producer
        .hold_next_audio_param_after_reservation_for_test(entered_send, release_recv);
    let param = gain.gain().clone();
    let update = thread::spawn(move || {
        panic::catch_unwind(AssertUnwindSafe(|| param.set_value(0.75))).is_err()
    });
    entered_recv.recv().unwrap();
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));
    harness.producer.fail_transport();
    release_send.send(()).unwrap();
    rollback_entered_recv.recv().unwrap();
    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    rollback_release_send.send(()).unwrap();
    assert!(update.join().unwrap());
    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    assert!(drained.finish().is_ok());
    close_thread.join().unwrap();
    assert_eq!(gain.gain().value(), 0.25);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn not_accepted_rollback_panic_fails_closed_before_releasing_admission() {
    let mut harness = Harness::new(4);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();

    let (reservation_entered_send, reservation_entered_recv) = crossbeam_channel::bounded(1);
    let (reservation_release_send, reservation_release_recv) = crossbeam_channel::bounded(1);
    harness
        .producer
        .hold_next_audio_param_after_reservation_for_test(
            reservation_entered_send,
            reservation_release_recv,
        );
    let (rollback_entered_send, rollback_entered_recv) = crossbeam_channel::bounded(1);
    let (rollback_release_send, rollback_release_recv) = crossbeam_channel::bounded(1);
    gain.gain().hold_next_injected_value_rollback_for_test(
        rollback_entered_send,
        rollback_release_recv,
        true,
    );
    let param = gain.gain().clone();
    let update = thread::spawn(move || {
        panic::catch_unwind(AssertUnwindSafe(|| param.set_value(0.75))).is_err()
    });
    reservation_entered_recv.recv().unwrap();

    // Contend only the commit after the typed reservation exists. This is an ordinary
    // NotAccepted return whose exact command is recovered before the hostile rollback hook.
    let (state_entered_send, state_entered_recv) = crossbeam_channel::bounded(1);
    let (state_release_send, state_release_recv) = crossbeam_channel::bounded(1);
    let producer = harness.producer.clone();
    let state_thread = thread::spawn(move || {
        producer.hold_transport_state_for_test(state_entered_send, state_release_recv);
    });
    state_entered_recv.recv().unwrap();
    reservation_release_send.send(()).unwrap();
    rollback_entered_recv.recv().unwrap();
    state_release_send.send(()).unwrap();
    state_thread.join().unwrap();

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_send, close_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        close_send.send(retirement.retire_and_wait()).unwrap();
    });
    assert!(close_recv.try_recv().is_err());
    rollback_release_send.send(()).unwrap();
    assert!(update.join().unwrap());

    let (snapshot, drained) = close_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    let sealed = drained.finish().ok().unwrap();
    assert!(sealed.degradation.prior_transport_failure);
    drop(sealed);
    close_thread.join().unwrap();
    assert_eq!(gain.gain().value(), 1.);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn last_gain_handles_request_teardown_before_their_final_base_arc_is_destroyed() {
    let mut harness = Harness::new(1);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    drop(harness.base.take());
    assert_eq!(
        harness.lifetimes.slot_phase_counts_for_test(),
        [SLOT_CAPACITY - 2, 0, 2, 0, 0, 0]
    );
    drop(gain);
    harness.assert_two_requested();
    assert!(harness
        .lifetimes
        .request_activity_receiver()
        .try_recv()
        .is_ok());
}

#[test]
fn last_handle_drop_cannot_cancel_an_already_accepted_param_value() {
    let mut harness = Harness::new(4);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    harness.callback();
    harness.wait_for_transport_idle();
    gain.gain().set_value(0.125);
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));

    drop(gain);
    harness.assert_two_requested();
    harness.callback();
    harness.wait_for_transport_idle();
    assert_eq!(harness.base().applied_control_batch_sequence(), 2);
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn preaccept_rollback_keeps_admission_through_hostile_destructor_and_exact_cleanup() {
    let mut harness = Harness::new(1);
    let transaction = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap();
    let destructor_ran = Arc::new(AtomicBool::new(false));
    let payload = payload(
        Box::new(PanicDropProcessor {
            dropped: Arc::clone(&destructor_ran),
        }),
        Box::new(SilentProcessor),
    );

    let (transport_entered_send, transport_entered_recv) = crossbeam_channel::bounded(1);
    let (transport_release_send, transport_release_recv) = crossbeam_channel::bounded(1);
    let lock_producer = harness.producer.clone();
    let lock_thread = thread::spawn(move || {
        lock_producer.hold_transport_state_for_test(transport_entered_send, transport_release_recv);
    });
    transport_entered_recv.recv().unwrap();

    let (ids_entered_send, ids_entered_recv) = crossbeam_channel::bounded(1);
    let (ids_release_send, ids_release_recv) = crossbeam_channel::bounded(1);
    harness
        .lifetimes
        .set_id_release_hook_for_test(ids_entered_send, ids_release_recv);
    let commit_thread = thread::spawn(move || transaction.commit(payload));
    ids_entered_recv.recv().unwrap();
    assert!(destructor_ran.load(Ordering::Acquire));
    transport_release_send.send(()).unwrap();
    lock_thread.join().unwrap();

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (close_done_send, close_done_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        let (snapshot, drained) = retirement.retire_and_wait();
        drop(drained);
        close_done_send.send(snapshot).unwrap();
    });
    assert!(close_done_recv.try_recv().is_err());
    ids_release_send.send(()).unwrap();

    let result = commit_thread.join().unwrap();
    assert_eq!(
        result.err().unwrap(),
        InjectedGainConstructionError::RejectedPayloadPanicked
    );
    let snapshot = close_done_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    close_thread.join().unwrap();
    harness.assert_slots_vacant();
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    let ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(0));
    assert_eq!(ids.id(1), AudioNodeId(1));
    assert!(harness.lifetimes.request_activity_receiver().is_empty());
}

#[test]
fn staged_gain_is_armed_once_and_close_extracts_one_batch_off_render_thread() {
    let mut harness = Harness::new_with_suspension(1, true);
    let gain = GainNode::new(harness.base(), GainOptions::default());
    assert_eq!(harness.producer.accounting(), (4, 1, 0, 1));
    assert_eq!(
        harness.lifetimes.slot_phase_counts_for_test(),
        [SLOT_CAPACITY - 2, 0, 2, 0, 0, 0]
    );
    assert_eq!(harness.base().applied_control_batch_sequence(), 0);

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (snapshot, drained) = retirement.retire_and_wait();
    assert!(snapshot.is_drained());
    let sealed = drained.finish().ok().unwrap();
    assert_eq!(sealed.payloads.staged_len(), 1);
    assert_eq!(sealed.payloads.last_submitted_batch_sequence(), 0);
    let producer = harness.producer.clone();
    let cleanup = thread::Builder::new()
        .name("injected-gain-close-cleanup".to_owned())
        .spawn(move || {
            let cleanup_thread = thread::current().id();
            drop(sealed.payloads);
            (cleanup_thread, producer.accounting())
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(cleanup.1, (0, 0, 0, 0));
    assert_ne!(cleanup.0, thread::current().id());
    drop(gain);
}

#[test]
fn close_before_gain_begin_mutates_no_id_slot_or_payload_state() {
    let mut harness = Harness::new(1);
    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (snapshot, drained) = retirement.retire_and_wait();
    assert!(snapshot.is_drained());

    let typed = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain();
    assert!(matches!(
        typed,
        Err(InjectedGainConstructionError::Control(
            super::injected_control::InjectedControlError::Sealed
        ))
    ));
    let public = panic::catch_unwind(AssertUnwindSafe(|| {
        GainNode::new(harness.base(), GainOptions::default())
    }));
    assert!(public.is_err());
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    harness.assert_slots_vacant();
    assert!(harness.lifetimes.request_activity_receiver().is_empty());
    let ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(0));
    assert_eq!(ids.id(1), AudioNodeId(1));
    drop(drained);
}

#[test]
fn accepted_arm_blocks_close_drain_until_both_registration_finalizers_finish() {
    let mut harness = Harness::new(1);
    let (arm_entered_send, arm_entered_recv) = crossbeam_channel::bounded(1);
    let (arm_release_send, arm_release_recv) = crossbeam_channel::bounded(1);
    harness
        .lifetimes
        .set_arm_publish_hook(arm_entered_send, arm_release_recv);
    let base = harness.base().clone();
    let gain_thread = thread::spawn(move || GainNode::new(&base, GainOptions::default()));
    arm_entered_recv.recv().unwrap();

    let retirement = harness
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    let (done_send, done_recv) = crossbeam_channel::bounded(1);
    let close_thread = thread::spawn(move || {
        let (snapshot, drained) = retirement.retire_and_wait();
        drop(drained);
        done_send.send(snapshot).unwrap();
    });
    assert!(done_recv.try_recv().is_err());
    arm_release_send.send(()).unwrap();
    let gain = gain_thread.join().unwrap();
    let snapshot = done_recv.recv().unwrap();
    assert!(snapshot.is_drained());
    close_thread.join().unwrap();
    assert_eq!(
        harness.lifetimes.slot_phase_counts_for_test(),
        [SLOT_CAPACITY - 2, 0, 2, 0, 0, 0]
    );
    drop(gain);
    harness.assert_two_requested();
}

#[test]
fn accepted_finalizer_reject_or_panic_never_recycles_either_id() {
    for (ordinal, panics) in [(1, false), (2, false), (2, true)] {
        let mut harness = Harness::new(1);
        if panics {
            harness.registrar.panic_construction_arm_for_test(ordinal);
        } else {
            harness.registrar.fail_construction_arm_for_test(ordinal);
        }
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            GainNode::new(harness.base(), GainOptions::default())
        }));
        assert!(result.is_err());
        harness.assert_two_requested();

        let ids = harness.allocator.try_reserve(2).unwrap();
        assert_eq!(ids.id(0), AudioNodeId(2));
        assert_eq!(ids.id(1), AudioNodeId(3));
        drop(ids);
        assert_eq!(harness.producer.accounting(), (4, 1, 1, 0));
        harness.callback();
        assert_eq!(harness.base().applied_control_batch_sequence(), 1);
        harness.wait_for_transport_idle();
    }
}

#[test]
fn every_internal_id_proof_failure_quarantines_slots_and_both_ids() {
    for failure_point in 1..=3 {
        let harness = Harness::new(1);
        let mut transaction = harness
            .base()
            .injected_node_constructor()
            .unwrap()
            .try_begin_gain()
            .unwrap();
        transaction.corrupt_id_proof_for_test(failure_point);
        let error = transaction
            .commit(payload(
                Box::new(SilentProcessor),
                Box::new(SilentProcessor),
            ))
            .err()
            .unwrap();
        assert_eq!(
            error,
            InjectedGainConstructionError::NodeIds(
                super::injected_ids::ProvisionalNodeIdError::ProtocolViolation
            )
        );
        harness.assert_two_quarantined();
        assert!(harness.lifetimes.request_activity_receiver().is_empty());
        assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
        let ids = harness.allocator.try_reserve(2).unwrap();
        assert_eq!(ids.id(0), AudioNodeId(2));
        assert_eq!(ids.id(1), AudioNodeId(3));
    }
}

#[test]
fn rejected_foreign_reclaim_token_quarantines_both_allocators_without_reuse() {
    let harness = Harness::new(1);
    let mut transaction = harness
        .base()
        .injected_node_constructor()
        .unwrap()
        .try_begin_gain()
        .unwrap();
    let (foreign_allocator, _foreign_owner, _foreign_graph) = injected_node_id_pair(900);
    let mut foreign_ids = foreign_allocator.try_reserve(1).unwrap();
    let foreign = foreign_ids.take_reclaim_node(0).unwrap();
    transaction.replace_reclaim_for_test(1, foreign);
    harness.producer.fail_transport();

    let error = transaction
        .commit(payload(
            Box::new(SilentProcessor),
            Box::new(SilentProcessor),
        ))
        .err()
        .unwrap();
    assert_eq!(error, InjectedGainConstructionError::ProtocolViolation);
    harness.assert_two_quarantined();
    assert!(harness.lifetimes.request_activity_receiver().is_empty());
    assert_eq!(harness.producer.accounting(), (0, 0, 0, 0));
    let exact_ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(exact_ids.id(0), AudioNodeId(2));
    assert_eq!(exact_ids.id(1), AudioNodeId(3));
    drop(foreign_ids);
    let foreign_retry = foreign_allocator.try_reserve(1).unwrap();
    assert_eq!(foreign_retry.id(0), AudioNodeId(901));
}

#[test]
fn full_transport_rejects_before_any_id_or_lifetime_mutation() {
    let mut harness = Harness::new(1);
    let log = Arc::new(Mutex::new(Vec::new()));
    let batch = harness
        .producer
        .try_begin_operation(1)
        .unwrap()
        .prepare_with(|| {
            vec![ControlMessage::TestMarker {
                value: 7,
                log: Arc::clone(&log),
            }]
        })
        .ok()
        .unwrap();
    harness.producer.try_commit(batch).ok().unwrap();
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        GainNode::new(harness.base(), GainOptions::default())
    }));
    assert!(result.is_err());
    assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));
    harness.assert_slots_vacant();
    assert!(harness.lifetimes.request_activity_receiver().is_empty());
    let ids = harness.allocator.try_reserve(2).unwrap();
    assert_eq!(ids.id(0), AudioNodeId(0));
    assert_eq!(ids.id(1), AudioNodeId(1));
    drop(ids);

    harness.callback();
    assert_eq!(*log.lock().unwrap(), [7]);
    harness.wait_for_transport_idle();
}
