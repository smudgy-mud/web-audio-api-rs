//! Deterministic tests for the private concrete Gain construction boundary.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::injected_control::{
    injected_control_channel, InjectedControlLifecycleOwner, InjectedControlProducer,
};
use super::injected_ids::{injected_node_id_pair, InjectedNodeIdAllocator};
use super::injected_node_construction::{
    InjectedGainConstructionError, InjectedGainPayload, InjectedNodeConstructor,
    InjectedNodeConstructorBuildError,
};
use super::injected_node_lifetime::{
    injected_node_lifetime_registry, InjectedNodeLifetimeOwner, InjectedNodeLifetimeRegistrar,
};
use super::{
    AudioContextState, AudioNodeId, ConcreteBaseAudioContext, InjectedContextAdmissionGate,
};
use crate::events::{EventDispatch, EventLoop};
use crate::message::{ControlBatchApplied, ControlBatchSender, ControlMessage};
use crate::node::{
    AudioNode, AudioNodeOptions, ChannelConfigInner, ChannelCountMode, ChannelInterpretation,
    GainNode, GainOptions,
};
use crate::param::AudioParamInitialValue;
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

fn channel_config() -> ChannelConfigInner {
    ChannelConfigInner {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    }
}

fn payload(param: Box<dyn AudioProcessor>, gain: Box<dyn AudioProcessor>) -> InjectedGainPayload {
    InjectedGainPayload {
        param_processor: param,
        gain_processor: gain,
        param_channel_config: channel_config(),
        gain_channel_config: channel_config(),
        initial_value: AudioParamInitialValue::new(1.),
    }
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
