//! Deterministic requested-teardown and reconciliation tests.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use super::*;
use crate::context::injected_control::{
    injected_control_channel, InjectedControlLifecycleOwner, InjectedControlRenderInit,
};
use crate::context::injected_ids::{injected_node_id_pair, InjectedNodeIdAllocator};
use crate::context::{AudioContextState, InjectedContextAdmissionGate};
use crate::events::EventDispatch;
use crate::message::ControlMessage;
use crate::node::{ChannelConfigInner, ChannelCountMode, ChannelInterpretation};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::stats::AudioStats;

struct RetireImmediately;

impl AudioProcessor for RetireImmediately {
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

fn channel_config() -> ChannelConfigInner {
    ChannelConfigInner {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    }
}

#[derive(Clone, Copy)]
enum CleanupBehavior {
    Ok,
    Reject,
    Panic,
    DropPanic,
    RetryTwice,
}

struct CleanupProbe {
    behavior: CleanupBehavior,
    attempts: usize,
    reconciled: Arc<Mutex<Vec<(AudioNodeId, ThreadId)>>>,
    dropped: Arc<Mutex<Vec<ThreadId>>>,
}

impl InjectedNodeReclaimCleanup for CleanupProbe {
    fn reconcile(&mut self, id: AudioNodeId) -> Result<(), NodeReclaimCleanupError> {
        self.attempts += 1;
        self.reconciled
            .lock()
            .unwrap()
            .push((id, thread::current().id()));
        match self.behavior {
            CleanupBehavior::Ok | CleanupBehavior::DropPanic => Ok(()),
            CleanupBehavior::Reject => Err(NodeReclaimCleanupError::Rejected),
            CleanupBehavior::Panic => panic!("reconcile panic"),
            CleanupBehavior::RetryTwice if self.attempts <= 2 => {
                Err(NodeReclaimCleanupError::RetryContended)
            }
            CleanupBehavior::RetryTwice => Ok(()),
        }
    }

    fn reconcile_after_whole_graph(
        &mut self,
        id: AudioNodeId,
    ) -> Result<(), NodeReclaimCleanupError> {
        if matches!(self.behavior, CleanupBehavior::RetryTwice) {
            self.attempts += 1;
            self.reconciled
                .lock()
                .unwrap()
                .push((id, thread::current().id()));
            Ok(())
        } else {
            self.reconcile(id)
        }
    }
}

impl Drop for CleanupProbe {
    fn drop(&mut self) {
        self.dropped.lock().unwrap().push(thread::current().id());
        if matches!(self.behavior, CleanupBehavior::DropPanic) {
            panic!("cleanup drop panic");
        }
    }
}

type ProbeRecords = (
    Arc<Mutex<Vec<(AudioNodeId, ThreadId)>>>,
    Arc<Mutex<Vec<ThreadId>>>,
);

fn probe(behavior: CleanupBehavior) -> (Box<dyn InjectedNodeReclaimCleanup>, ProbeRecords) {
    let reconciled = Arc::new(Mutex::new(Vec::new()));
    let dropped = Arc::new(Mutex::new(Vec::new()));
    (
        Box::new(CleanupProbe {
            behavior,
            attempts: 0,
            reconciled: Arc::clone(&reconciled),
            dropped: Arc::clone(&dropped),
        }),
        (reconciled, dropped),
    )
}

struct Registered {
    id: AudioNodeId,
    key: RegistrationKey,
    live: InjectedNodeRegistration,
    reclaim: llq::Node<AudioNodeId>,
    records: ProbeRecords,
}

struct Foundation {
    producer: InjectedControlProducer,
    lifecycle: Option<InjectedControlLifecycleOwner>,
    render_init: Option<InjectedControlRenderInit>,
    allocator: InjectedNodeIdAllocator,
    registrar: InjectedNodeLifetimeRegistrar,
    owner: Option<InjectedNodeLifetimeOwner>,
    graph: Option<InjectedGraphReclaimInit>,
}

impl Foundation {
    fn new(capacity: usize, first_id: u64, ordinary_capacity: usize, suspended: bool) -> Self {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, lifecycle, render_init) =
            injected_control_channel(gate, ordinary_capacity, suspended).unwrap();
        let (allocator, node_ids, graph) = injected_node_id_pair(first_id);
        let (registrar, bootstrap) =
            injected_node_lifetime_registry(capacity, &producer, node_ids, graph)
                .ok()
                .unwrap();
        let (owner, graph) = bootstrap.into_parts_for_test();
        Self {
            producer,
            lifecycle: Some(lifecycle),
            render_init: Some(render_init),
            allocator,
            registrar,
            owner: Some(owner),
            graph: Some(graph),
        }
    }

    fn register(&mut self, behavior: CleanupBehavior) -> Registered {
        let mut ids = self.allocator.try_reserve(1).unwrap();
        let id = ids.id(0);
        let (cleanup, records) = probe(behavior);
        let provisional = self.registrar.try_register(id, cleanup).ok().unwrap();
        let key = provisional.key;
        provisional.arm_token().arm().unwrap();
        let live = provisional.into_registration().unwrap();
        let reclaim = ids.take_reclaim_node(0).unwrap();
        ids.commit().unwrap();
        Registered {
            id,
            key,
            live,
            reclaim,
            records,
        }
    }

    fn close_and_seal(&mut self) -> (SealedNodeLifetimeRegistry, usize) {
        let retirement = self
            .lifecycle
            .take()
            .unwrap()
            .try_begin_close()
            .ok()
            .unwrap();
        let (_, drained) = retirement.retire_and_wait();
        let registry = self
            .owner
            .take()
            .unwrap()
            .seal_after_control_drain(&drained)
            .ok()
            .unwrap();
        let transport = drained.finish().ok().unwrap();
        let staged = transport.payloads.staged_len();
        drop(transport);
        (registry, staged)
    }
}

fn retire(registry: SealedNodeLifetimeRegistry) -> WholeGraphNodeRetirement {
    let proof = WholeGraphRetired::for_test(&registry);
    registry.retire_after_whole_graph(proof).ok().unwrap()
}

fn set_phase(owner: &InjectedNodeLifetimeOwner, key: RegistrationKey, phase: SlotPhase) {
    let slot = &owner.inner().slots[key.slot];
    let mut word = slot.word.load(Ordering::Acquire);
    loop {
        assert_eq!(generation(word), key.generation.get());
        match slot.word.compare_exchange(
            word,
            with_phase(word, phase),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return,
            Err(observed) => word = observed,
        }
    }
}

fn set_hook(
    hook: &Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
) -> (
    crossbeam_channel::Receiver<()>,
    crossbeam_channel::Sender<()>,
) {
    let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    *hook
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((entered_send, release_recv));
    (entered_recv, release_send)
}

#[test]
fn typed_teardown_stays_out_of_general_batching_and_reuses_only_after_cleanup() {
    assert!(!ControlMessage::ControlHandleDropped { id: AudioNodeId(7) }.is_batchable());
    let mut foundation = Foundation::new(1, 100, 1, true);
    let registered = foundation.register(CleanupBehavior::Ok);
    let Registered {
        id,
        key,
        live,
        reclaim,
        records,
    } = registered;
    drop(live);
    assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));

    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted {
            id,
            outcome: crate::context::injected_control::CommitControlOutcome::Staged,
        }
    );
    let word = foundation.owner.as_ref().unwrap().inner().slots[key.slot]
        .word
        .load(Ordering::Acquire);
    assert_eq!(SlotPhase::from_word(word), SlotPhase::AwaitingReclaim);
    assert!(!has_reclaim(word));
    assert_eq!(foundation.producer.accounting(), (1, 1, 0, 1));

    foundation.graph.as_mut().unwrap().push_for_test(reclaim);
    let owner = foundation.owner.take().unwrap();
    let control = thread::Builder::new()
        .name("b2b-control-reconcile".into())
        .spawn(move || {
            let mut owner = owner;
            let control_thread = thread::current().id();
            let outcome = owner.try_drive_once();
            (owner, control_thread, outcome)
        })
        .unwrap();
    let (owner, control_thread, outcome) = control.join().unwrap();
    foundation.owner = Some(owner);
    assert_eq!(outcome, NodeLifetimeDriveOutcome::Reconciled { id });
    assert_eq!(
        records.0.lock().unwrap().as_slice(),
        &[(id, control_thread)]
    );
    assert_eq!(records.1.lock().unwrap().as_slice(), &[control_thread]);
    let ids = foundation.allocator.try_reserve(1).unwrap();
    assert_eq!(ids.id(0), id);
    drop(ids);

    let (registry, staged) = foundation.close_and_seal();
    assert_eq!(staged, 1);
    assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));
    assert_eq!(retire(registry).cleanup_count, 0);
}

#[test]
fn retryable_cleanup_restores_exact_requested_and_awaiting_state_without_id_reuse() {
    for awaiting in [false, true] {
        let mut foundation = Foundation::new(1, 300 + u64::from(awaiting), 1, true);
        let registered = foundation.register(CleanupBehavior::RetryTwice);
        let id = registered.id;
        let key = registered.key;
        drop(registered.live);
        if awaiting {
            assert!(matches!(
                foundation.owner.as_mut().unwrap().try_drive_once(),
                NodeLifetimeDriveOutcome::Submitted { id: submitted, .. } if submitted == id
            ));
        }
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(registered.reclaim);

        let expected_phase = if awaiting {
            SlotPhase::AwaitingReclaim
        } else {
            SlotPhase::Requested
        };
        for _ in 0..2 {
            assert_eq!(
                foundation.owner.as_mut().unwrap().try_drive_once(),
                NodeLifetimeDriveOutcome::Retry {
                    id,
                    reason: NodeLifetimeRetryReason::Contended,
                }
            );
            let slot = &foundation.owner.as_ref().unwrap().inner().slots[key.slot];
            let word = slot.word.load(Ordering::Acquire);
            assert_eq!(SlotPhase::from_word(word), expected_phase);
            assert!(has_reclaim(word));
            let reserved = foundation.allocator.try_reserve(1).unwrap();
            assert_ne!(reserved.id(0), id);
            drop(reserved);
        }
        assert_eq!(
            foundation.owner.as_mut().unwrap().try_drive_once(),
            NodeLifetimeDriveOutcome::Reconciled { id }
        );
        // The two rejected probe reservations are also in the allocator FIFO. Reserve the whole
        // small set at once so the exact reconciled ID cannot be hidden behind their order.
        let reserved = foundation.allocator.try_reserve(4).unwrap();
        assert!((0..4).any(|index| reserved.id(index) == id));
        drop(reserved);
    }
}

#[test]
fn seal_retains_retryable_cleanup_for_unique_whole_graph_reconciliation() {
    let mut foundation = Foundation::new(1, 400, 1, true);
    let registered = foundation.register(CleanupBehavior::RetryTwice);
    let id = registered.id;
    let reconciled = Arc::clone(&registered.records.0);
    drop(registered.live);
    foundation
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(registered.reclaim);
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Retry {
            id,
            reason: NodeLifetimeRetryReason::Contended,
        }
    );
    let reserved = foundation.allocator.try_reserve(1).unwrap();
    assert_ne!(reserved.id(0), id);
    drop(reserved);

    let (registry, _) = foundation.close_and_seal();
    let report = retire(registry);
    assert_eq!(report.cleanup_count, 1);
    assert!(!report.cleanup_rejected);
    assert_eq!(reconciled.lock().unwrap().len(), 2);
}

#[test]
fn typed_teardown_batch_runs_through_renderer_and_publishes_exact_reclaim() {
    assert!(!ControlMessage::ControlHandleDropped { id: AudioNodeId(7) }.is_batchable());
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) = injected_control_channel(gate, 2, false).unwrap();
    let (allocator, node_ids, graph) = injected_node_id_pair(0);
    let (registrar, bootstrap) = injected_node_lifetime_registry(2, &producer, node_ids, graph)
        .ok()
        .unwrap();

    let mut ids = allocator.try_reserve(2).unwrap();
    let destination_id = ids.id(0);
    assert_eq!(destination_id, AudioNodeId(0));
    let id = ids.id(1);
    let (destination_cleanup, _) = probe(CleanupBehavior::Ok);
    let destination = registrar
        .try_register(destination_id, destination_cleanup)
        .ok()
        .unwrap();
    destination.arm_token().arm().unwrap();
    let destination_live = destination.into_registration().unwrap();
    let (cleanup, records) = probe(CleanupBehavior::Ok);
    let provisional = registrar.try_register(id, cleanup).ok().unwrap();
    provisional.arm_token().arm().unwrap();
    let live = provisional.into_registration().unwrap();
    let destination_reclaim = ids.take_reclaim_node(0).unwrap();
    let reclaim_id = ids.take_reclaim_node(1).unwrap();
    ids.commit().unwrap();

    let (event_sender, _event_receiver) = crossbeam_channel::bounded::<EventDispatch>(4);
    let bound = render_init
        .build_render_thread(
            bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU8::new(AudioContextState::Running as u8)),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            event_sender,
        )
        .ok()
        .unwrap();
    let (mut renderer, mut owner) = bound.into_render_thread_for_test();
    let gc = renderer.spawn_joinable_garbage_collector_thread().unwrap();

    let register = producer
        .try_begin_operation(2)
        .unwrap()
        .prepare_with(|| {
            vec![
                ControlMessage::RegisterNode {
                    id: destination_id,
                    reclaim_id: destination_reclaim,
                    node: Box::new(RetireImmediately),
                    inputs: 0,
                    outputs: 1,
                    channel_config: channel_config(),
                },
                ControlMessage::RegisterNode {
                    id,
                    reclaim_id,
                    node: Box::new(RetireImmediately),
                    inputs: 0,
                    outputs: 0,
                    channel_config: channel_config(),
                },
            ]
        })
        .ok()
        .unwrap();
    assert_eq!(
        producer.try_commit(register).ok().unwrap(),
        crate::context::injected_control::CommitControlOutcome::Enqueued { sequence: 1 }
    );
    renderer.render(&mut [0.; 256]);
    assert_eq!(producer.applied_batch_sequence(), 1);

    drop(live);
    assert_eq!(
        owner.try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted {
            id,
            outcome: crate::context::injected_control::CommitControlOutcome::Enqueued {
                sequence: 2,
            },
        }
    );
    renderer.render(&mut [0.; 256]);
    assert_eq!(producer.applied_batch_sequence(), 2);
    assert_eq!(
        owner.try_drive_once(),
        NodeLifetimeDriveOutcome::Reconciled { id }
    );
    assert_eq!(
        records.0.lock().unwrap().as_slice(),
        &[(id, thread::current().id())]
    );
    assert_eq!(records.1.lock().unwrap().len(), 1);
    let reused = allocator.try_reserve(1).unwrap();
    assert_eq!(reused.id(0), id);
    drop(reused);

    let retirement = lifecycle.try_begin_close().ok().unwrap();
    let (_, drained) = retirement.retire_and_wait();
    let registry = owner.seal_after_control_drain(&drained).ok().unwrap();
    let transport = drained.finish().ok().unwrap();
    assert_eq!(transport.payloads.staged_len(), 0);
    renderer.render(&mut [] as &mut [f32]);
    assert!(transport.close.try_observe_exact().is_ok());
    drop(transport.payloads);
    drop(renderer);
    gc.join().unwrap();
    drop(destination_live);
    assert_eq!(retire(registry).cleanup_count, 1);
}

#[test]
fn every_early_reclaim_phase_preserves_exact_token_and_later_slots_progress() {
    let mut foundation = Foundation::new(5, 100, 5, true);
    let live = foundation.register(CleanupBehavior::Ok);
    let requested = foundation.register(CleanupBehavior::Ok);
    let servicing = foundation.register(CleanupBehavior::Ok);
    let accepted = foundation.register(CleanupBehavior::Ok);
    let awaiting = foundation.register(CleanupBehavior::Ok);

    drop(requested.live);
    drop(servicing.live);
    drop(accepted.live);
    drop(awaiting.live);
    let owner = foundation.owner.as_ref().unwrap();
    set_phase(owner, servicing.key, SlotPhase::Servicing);
    set_phase(owner, accepted.key, SlotPhase::AcceptedPending);
    set_phase(owner, awaiting.key, SlotPhase::AwaitingReclaim);
    for reclaim in [
        live.reclaim,
        requested.reclaim,
        servicing.reclaim,
        accepted.reclaim,
        awaiting.reclaim,
    ] {
        foundation.graph.as_mut().unwrap().push_for_test(reclaim);
    }
    assert_eq!(foundation.owner.as_mut().unwrap().ingest_reclaims(), Ok(5));
    for key in [
        live.key,
        requested.key,
        servicing.key,
        accepted.key,
        awaiting.key,
    ] {
        assert!(has_reclaim(
            foundation.owner.as_ref().unwrap().inner().slots[key.slot]
                .word
                .load(Ordering::Acquire)
        ));
    }

    // Slot zero is Live|R and must not head-of-line block ready later slots.
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Reconciled { id: requested.id }
    );
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Reconciled { id: awaiting.id }
    );
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Idle
    );

    drop(live.live);
    set_phase(
        foundation.owner.as_ref().unwrap(),
        servicing.key,
        SlotPhase::Requested,
    );
    set_phase(
        foundation.owner.as_ref().unwrap(),
        accepted.key,
        SlotPhase::AwaitingReclaim,
    );
    for id in [live.id, servicing.id, accepted.id] {
        assert_eq!(
            foundation.owner.as_mut().unwrap().try_drive_once(),
            NodeLifetimeDriveOutcome::Reconciled { id }
        );
    }
    assert!(foundation
        .owner
        .as_ref()
        .unwrap()
        .inner()
        .slots
        .iter()
        .all(|slot| SlotPhase::from_word(slot.word.load(Ordering::Acquire)) == SlotPhase::Vacant));
}

#[test]
fn not_accepted_destroys_credits_before_requested_is_visible_and_timed_retry_succeeds() {
    let mut foundation = Foundation::new(1, 100, 1, false);
    let registered = foundation.register(CleanupBehavior::Ok);
    let id = registered.id;
    let key = registered.key;
    drop(registered.live);
    let owner = foundation.owner.take().unwrap();
    let (pre_entered, pre_release) = set_hook(&owner.inner().teardown_precommit_hook);
    let (restore_entered, restore_release) = set_hook(&owner.inner().teardown_before_restore_hook);
    let service = thread::spawn(move || {
        let mut owner = owner;
        let outcome = owner.try_drive_once();
        (owner, outcome)
    });
    pre_entered.recv().unwrap();
    assert_eq!(
        SlotPhase::from_word(
            foundation.registrar.inner.upgrade().unwrap().slots[key.slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::AcceptedPending
    );

    let producer = foundation.producer.clone();
    let (locked_send, locked_recv) = crossbeam_channel::bounded(1);
    let (unlock_send, unlock_recv) = crossbeam_channel::bounded(1);
    let lock = thread::spawn(move || {
        producer.hold_transport_state_for_test(locked_send, unlock_recv);
    });
    locked_recv.recv().unwrap();
    pre_release.send(()).unwrap();
    restore_entered.recv().unwrap();
    assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));
    assert_eq!(
        SlotPhase::from_word(
            foundation.registrar.inner.upgrade().unwrap().slots[key.slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::AcceptedPending
    );
    restore_release.send(()).unwrap();
    let (mut owner, outcome) = service.join().unwrap();
    assert_eq!(
        outcome,
        NodeLifetimeDriveOutcome::Retry {
            id,
            reason: NodeLifetimeRetryReason::Contended,
        }
    );
    assert_eq!(
        SlotPhase::from_word(owner.inner().slots[key.slot].word.load(Ordering::Acquire)),
        SlotPhase::Requested
    );
    unlock_send.send(()).unwrap();
    lock.join().unwrap();
    assert!(NODE_LIFETIME_RETRY_INTERVAL > std::time::Duration::ZERO);
    assert!(matches!(
        owner.try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted { id: submitted, .. } if submitted == id
    ));
    foundation.owner = Some(owner);
}

#[test]
fn saturated_transport_retries_after_credit_hint_without_retaining_attempt_credits() {
    let mut foundation = Foundation::new(2, 100, 1, false);
    let first = foundation.register(CleanupBehavior::Ok);
    let second = foundation.register(CleanupBehavior::Ok);
    drop(first.live);
    drop(second.live);
    assert!(matches!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted { id, .. } if id == first.id
    ));
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Retry {
            id: second.id,
            reason: NodeLifetimeRetryReason::Credit,
        }
    );
    assert_eq!(foundation.producer.accounting(), (1, 1, 1, 0));
    while foundation
        .lifecycle
        .as_ref()
        .unwrap()
        .credit_activity_receiver()
        .try_recv()
        .is_ok()
    {}
    assert!(foundation
        .render_init
        .as_ref()
        .unwrap()
        .reclaim_one_ordinary_for_test());
    foundation
        .lifecycle
        .as_ref()
        .unwrap()
        .credit_activity_receiver()
        .recv_timeout(NODE_LIFETIME_RETRY_INTERVAL)
        .unwrap();
    assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));
    assert!(matches!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted { id, .. } if id == second.id
    ));
}

#[test]
fn exact_reclaim_can_arrive_before_accepted_finalizer_returns() {
    let mut foundation = Foundation::new(1, 100, 1, true);
    let registered = foundation.register(CleanupBehavior::Ok);
    let id = registered.id;
    drop(registered.live);
    let owner = foundation.owner.take().unwrap();
    let (entered, release) = set_hook(&owner.inner().teardown_finalizer_hook);
    let service = thread::spawn(move || {
        let mut owner = owner;
        let outcome = owner.try_drive_once();
        (owner, outcome)
    });
    entered.recv().unwrap();
    assert_eq!(
        SlotPhase::from_word(
            foundation.registrar.inner.upgrade().unwrap().slots[registered.key.slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::AcceptedPending
    );
    foundation
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(registered.reclaim);
    release.send(()).unwrap();
    let (mut owner, outcome) = service.join().unwrap();
    assert_eq!(
        outcome,
        NodeLifetimeDriveOutcome::Submitted {
            id,
            outcome: crate::context::injected_control::CommitControlOutcome::Staged,
        }
    );
    assert_eq!(
        owner.try_drive_once(),
        NodeLifetimeDriveOutcome::Reconciled { id }
    );
    foundation.owner = Some(owner);
}

#[test]
fn accepted_finalizer_failure_and_terminal_transport_quarantine_without_idle_credits() {
    for behavior in [1, 2] {
        let mut foundation = Foundation::new(1, behavior as u64 * 100, 1, true);
        let registered = foundation.register(CleanupBehavior::Ok);
        drop(registered.live);
        foundation
            .owner
            .as_ref()
            .unwrap()
            .inner()
            .teardown_finalizer_behavior
            .store(behavior, Ordering::Release);
        assert_eq!(
            foundation.owner.as_mut().unwrap().try_drive_once(),
            NodeLifetimeDriveOutcome::Quarantined {
                id: Some(registered.id),
                reason: NodeLifetimeQuarantineReason::AcceptedFinalizer,
            }
        );
        assert_eq!(
            SlotPhase::from_word(
                foundation.owner.as_ref().unwrap().inner().slots[registered.key.slot]
                    .word
                    .load(Ordering::Acquire)
            ),
            SlotPhase::Quarantined
        );
        let (registry, staged) = foundation.close_and_seal();
        assert_eq!(staged, 1);
        assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));
        assert!(registry.degraded());
        retire(registry);
    }

    let mut foundation = Foundation::new(1, 500, 1, false);
    let registered = foundation.register(CleanupBehavior::Ok);
    drop(registered.live);
    foundation.producer.fail_transport();
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Quarantined {
            id: Some(registered.id),
            reason: NodeLifetimeQuarantineReason::Transport(
                crate::context::injected_control::InjectedControlError::ProtocolViolation
            ),
        }
    );
    assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));
}

#[test]
fn cleanup_failures_and_foreign_token_never_reuse_an_id() {
    for (index, (behavior, reason)) in [
        (
            CleanupBehavior::Reject,
            NodeLifetimeQuarantineReason::CleanupRejected,
        ),
        (
            CleanupBehavior::Panic,
            NodeLifetimeQuarantineReason::CleanupPanicked,
        ),
        (
            CleanupBehavior::DropPanic,
            NodeLifetimeQuarantineReason::CleanupDestructorPanicked,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let mut foundation = Foundation::new(1, index as u64 * 100 + 100, 1, true);
        let registered = foundation.register(behavior);
        let records = (
            Arc::clone(&registered.records.0),
            Arc::clone(&registered.records.1),
        );
        drop(registered.live);
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(registered.reclaim);
        assert_eq!(
            foundation.owner.as_mut().unwrap().try_drive_once(),
            NodeLifetimeDriveOutcome::Quarantined {
                id: Some(registered.id),
                reason,
            }
        );
        let ids = foundation.allocator.try_reserve(1).unwrap();
        assert_ne!(ids.id(0), registered.id);
        drop(ids);
        if matches!(behavior, CleanupBehavior::Reject) {
            assert_eq!(records.0.lock().unwrap().len(), 1);
            assert!(records.1.lock().unwrap().is_empty());
            let (registry, staged) = foundation.close_and_seal();
            assert_eq!(staged, 0);
            let report = retire(registry);
            assert_eq!(report.cleanup_count, 1);
            assert!(report.cleanup_rejected);
            assert_eq!(records.0.lock().unwrap().len(), 2);
            assert!(records.1.lock().unwrap().is_empty());
        }
    }

    let mut exact = Foundation::new(1, 700, 1, true);
    let registered = exact.register(CleanupBehavior::Ok);
    drop(registered.live);
    exact
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(registered.reclaim);
    assert_eq!(exact.owner.as_mut().unwrap().ingest_reclaims(), Ok(1));
    let original = exact.owner.as_ref().unwrap().inner().slots[registered.key.slot]
        .payload
        .lock()
        .unwrap()
        .reclaim
        .take()
        .unwrap();
    std::mem::forget(original);

    let mut foreign = Foundation::new(1, registered.id.0, 1, true);
    let mut foreign_ids = foreign.allocator.try_reserve(1).unwrap();
    foreign
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(foreign_ids.take_reclaim_node(0).unwrap());
    foreign_ids.commit().unwrap();
    let foreign_token = foreign
        .owner
        .as_mut()
        .unwrap()
        .node_ids
        .as_mut()
        .unwrap()
        .try_take_pending_reclaim()
        .unwrap();
    exact.owner.as_ref().unwrap().inner().slots[registered.key.slot]
        .payload
        .lock()
        .unwrap()
        .reclaim = Some(foreign_token);
    assert_eq!(
        exact.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Quarantined {
            id: Some(registered.id),
            reason: NodeLifetimeQuarantineReason::ReclaimBrandMismatch,
        }
    );
    assert!(
        exact.owner.as_ref().unwrap().inner().slots[registered.key.slot]
            .payload
            .lock()
            .unwrap()
            .reclaim
            .is_some()
    );
    let ids = exact.allocator.try_reserve(1).unwrap();
    assert_ne!(ids.id(0), registered.id);
    let foreign_ids = foreign.allocator.try_reserve(1).unwrap();
    assert_ne!(foreign_ids.id(0), registered.id);
}

#[test]
fn post_publication_transition_failure_serializes_registration_and_quarantines() {
    let mut foundation = Foundation::new(1, 100, 1, true);
    let registered = foundation.register(CleanupBehavior::Ok);
    let id = registered.id;
    let key = registered.key;
    drop(registered.live);
    foundation
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(registered.reclaim);
    let owner = foundation.owner.take().unwrap();
    let (entered, release) = set_hook(&owner.inner().teardown_post_publish_hook);
    let reconcile = thread::spawn(move || {
        let mut owner = owner;
        let outcome = owner.try_drive_once();
        (owner, outcome)
    });
    entered.recv().unwrap();

    let ids = foundation.allocator.try_reserve(1).unwrap();
    assert_eq!(ids.id(0), id);
    let failure = foundation
        .registrar
        .try_register(id, probe(CleanupBehavior::Ok).0)
        .err()
        .unwrap();
    assert_eq!(failure.error, NodeRegistrationError::Contended);
    drop(failure);
    drop(ids);
    let inner = foundation.registrar.inner.upgrade().unwrap();
    let word = inner.slots[key.slot].word.load(Ordering::Acquire);
    inner.slots[key.slot].word.store(
        with_phase(word, SlotPhase::AwaitingReclaim),
        Ordering::Release,
    );
    release.send(()).unwrap();
    let (owner, outcome) = reconcile.join().unwrap();
    assert_eq!(
        outcome,
        NodeLifetimeDriveOutcome::Quarantined {
            id: Some(id),
            reason: NodeLifetimeQuarantineReason::ProtocolViolation,
        }
    );
    assert_eq!(
        RegistryPhase::from_u8(owner.inner().phase.load(Ordering::Acquire)),
        RegistryPhase::Quarantined
    );
    let ids = foundation.allocator.try_reserve(1).unwrap();
    assert_eq!(ids.id(0), id);
    assert_eq!(
        foundation
            .registrar
            .try_register(id, probe(CleanupBehavior::Ok).0)
            .err()
            .unwrap()
            .error,
        NodeRegistrationError::Sealed
    );
    foundation.owner = Some(owner);
}

#[test]
fn bounded_wait_selects_each_hint_and_omits_pre_disconnected_receivers() {
    let mut request = Foundation::new(1, 100, 1, true);
    let registered = request.register(CleanupBehavior::Ok);
    let (credit_send, credit_recv) = crossbeam_channel::bounded(1);
    assert_eq!(
        request
            .owner
            .as_ref()
            .unwrap()
            .wait_for_activity(&credit_recv),
        NodeLifetimeActivity::Timeout
    );
    drop(registered.live);
    assert_eq!(
        request
            .owner
            .as_ref()
            .unwrap()
            .wait_for_activity(&credit_recv),
        NodeLifetimeActivity::Request
    );

    let mut reclaim = Foundation::new(1, 300, 1, true);
    let registered = reclaim.register(CleanupBehavior::Ok);
    reclaim
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(registered.reclaim);
    assert_eq!(
        reclaim
            .owner
            .as_ref()
            .unwrap()
            .wait_for_activity(&credit_recv),
        NodeLifetimeActivity::Reclaim
    );

    credit_send.send(()).unwrap();
    assert_eq!(
        reclaim
            .owner
            .as_ref()
            .unwrap()
            .wait_for_activity(&credit_recv),
        NodeLifetimeActivity::Credit
    );

    let mut disconnected = Foundation::new(1, 500, 1, true);
    disconnected
        .owner
        .as_mut()
        .unwrap()
        .disconnect_request_activity_for_test();
    disconnected
        .owner
        .as_mut()
        .unwrap()
        .node_ids
        .as_mut()
        .unwrap()
        .disconnect_reclaim_activity_for_test();
    let (disconnected_send, disconnected_credit) = crossbeam_channel::bounded::<()>(1);
    drop(disconnected_send);
    let started = std::time::Instant::now();
    for _ in 0..3 {
        assert_eq!(
            disconnected
                .owner
                .as_ref()
                .unwrap()
                .wait_for_activity(&disconnected_credit),
            NodeLifetimeActivity::Timeout
        );
    }
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
}

#[test]
fn lossy_full_and_disconnected_wakes_do_not_replace_authoritative_scans() {
    let mut foundation = Foundation::new(2, 100, 2, true);
    let first = foundation.register(CleanupBehavior::Ok);
    let second = foundation.register(CleanupBehavior::Ok);
    drop(first.live);
    drop(second.live);
    foundation
        .owner
        .as_ref()
        .unwrap()
        .request_activity_receiver()
        .recv()
        .unwrap();
    assert!(foundation
        .owner
        .as_ref()
        .unwrap()
        .request_activity_receiver()
        .try_recv()
        .is_err());
    assert!(matches!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted { id, .. } if id == first.id
    ));
    assert!(matches!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted { id, .. } if id == second.id
    ));

    let mut disconnected = Foundation::new(1, 300, 1, true);
    let registered = disconnected.register(CleanupBehavior::Ok);
    let receiver = disconnected
        .owner
        .as_mut()
        .unwrap()
        .request_wake
        .take()
        .unwrap();
    drop(receiver);
    drop(registered.live);
    assert!(matches!(
        disconnected.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Submitted { id, .. } if id == registered.id
    ));

    let mut reclaim = Foundation::new(1, 500, 1, true);
    let registered = reclaim.register(CleanupBehavior::Ok);
    drop(registered.live);
    reclaim
        .owner
        .as_mut()
        .unwrap()
        .node_ids
        .as_mut()
        .unwrap()
        .disconnect_reclaim_activity_for_test();
    assert_eq!(
        reclaim
            .owner
            .as_ref()
            .unwrap()
            .reclaim_activity_receiver()
            .try_recv(),
        Err(crossbeam_channel::TryRecvError::Disconnected)
    );
    reclaim
        .graph
        .as_mut()
        .unwrap()
        .push_for_test(registered.reclaim);
    assert_eq!(
        reclaim.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::Reconciled { id: registered.id }
    );
}

#[test]
fn close_waits_for_accepted_finalizer_extracts_staging_and_seals_surviving_handle() {
    let mut foundation = Foundation::new(2, 100, 1, true);
    let requested = foundation.register(CleanupBehavior::Ok);
    let survivor = foundation.register(CleanupBehavior::Ok);
    drop(requested.live);
    let owner = foundation.owner.take().unwrap();
    let (entered, release) = set_hook(&owner.inner().teardown_finalizer_hook);
    let service = thread::spawn(move || {
        let mut owner = owner;
        let outcome = owner.try_drive_once();
        (owner, outcome)
    });
    entered.recv().unwrap();

    let lifecycle = foundation.lifecycle.take().unwrap();
    let (close_done_send, close_done_recv) = crossbeam_channel::bounded(1);
    let close = thread::spawn(move || {
        let retirement = lifecycle.try_begin_close().ok().unwrap();
        let result = retirement.retire_and_wait();
        close_done_send.send(()).unwrap();
        result
    });
    assert_eq!(
        close_done_recv.try_recv(),
        Err(crossbeam_channel::TryRecvError::Empty)
    );
    release.send(()).unwrap();
    let (owner, outcome) = service.join().unwrap();
    assert!(matches!(
        outcome,
        NodeLifetimeDriveOutcome::Submitted { .. }
    ));
    close_done_recv.recv().unwrap();
    let (_, drained) = close.join().unwrap();
    let registry = owner.seal_after_control_drain(&drained).ok().unwrap();
    let survivor_slot = survivor.key.slot;
    assert_eq!(
        SlotPhase::from_word(
            registry.inner.as_ref().unwrap().slots[survivor_slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::Sealed
    );
    drop(survivor.live);
    assert_eq!(
        SlotPhase::from_word(
            registry.inner.as_ref().unwrap().slots[survivor_slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::Sealed
    );
    let transport = drained.finish().ok().unwrap();
    assert_eq!(transport.payloads.staged_len(), 1);
    drop(transport);
    assert!(retire(registry).cleanup_count >= 1);
}

#[test]
fn proven_close_supersedes_a_requested_teardown_without_quarantining() {
    let mut foundation = Foundation::new(1, 100, 1, true);
    let registered = foundation.register(CleanupBehavior::Ok);
    drop(registered.live);
    let retirement = foundation
        .lifecycle
        .take()
        .unwrap()
        .try_begin_close()
        .ok()
        .unwrap();
    assert_eq!(
        foundation.owner.as_mut().unwrap().try_drive_once(),
        NodeLifetimeDriveOutcome::CloseSuperseded { id: registered.id }
    );
    assert_eq!(foundation.producer.accounting(), (0, 0, 0, 0));
    assert_eq!(
        SlotPhase::from_word(
            foundation.owner.as_ref().unwrap().inner().slots[registered.key.slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::Requested
    );

    let (_, drained) = retirement.retire_and_wait();
    let registry = foundation
        .owner
        .take()
        .unwrap()
        .seal_after_control_drain(&drained)
        .ok()
        .unwrap();
    assert!(!registry.degraded());
    assert_eq!(
        SlotPhase::from_word(
            registry.inner.as_ref().unwrap().slots[registered.key.slot]
                .word
                .load(Ordering::Acquire)
        ),
        SlotPhase::Sealed
    );
    let transport = drained.finish().ok().unwrap();
    assert_eq!(transport.payloads.staged_len(), 0);
    drop(transport);
    drop(registered.reclaim);
    assert_eq!(retire(registry).cleanup_count, 1);
}

#[test]
fn seal_maps_every_residual_phase_without_erasing_reclaim_state() {
    let mut foundation = Foundation::new(8, 100, 1, true);
    let registrations: [Registered; 8] = (0..8)
        .map(|_| foundation.register(CleanupBehavior::Ok))
        .collect::<Vec<_>>()
        .try_into()
        .ok()
        .unwrap();
    let [provisional, live, requested, awaiting, canceling, servicing, accepted, reconciling] =
        registrations;

    drop(requested.live);
    drop(awaiting.live);
    drop(servicing.live);
    drop(accepted.live);
    drop(reconciling.live);
    let owner = foundation.owner.as_ref().unwrap();
    set_phase(owner, provisional.key, SlotPhase::Provisional);
    set_phase(owner, awaiting.key, SlotPhase::AwaitingReclaim);
    set_phase(owner, canceling.key, SlotPhase::Canceling);
    set_phase(owner, servicing.key, SlotPhase::Servicing);
    set_phase(owner, accepted.key, SlotPhase::AcceptedPending);
    set_phase(owner, reconciling.key, SlotPhase::AwaitingReclaim);
    for reclaim in [
        requested.reclaim,
        awaiting.reclaim,
        servicing.reclaim,
        accepted.reclaim,
        reconciling.reclaim,
    ] {
        foundation.graph.as_mut().unwrap().push_for_test(reclaim);
    }
    assert_eq!(foundation.owner.as_mut().unwrap().ingest_reclaims(), Ok(5));
    set_phase(
        foundation.owner.as_ref().unwrap(),
        reconciling.key,
        SlotPhase::Reconciling,
    );

    for key in [
        requested.key,
        awaiting.key,
        servicing.key,
        accepted.key,
        reconciling.key,
    ] {
        assert!(has_reclaim(
            foundation.owner.as_ref().unwrap().inner().slots[key.slot]
                .word
                .load(Ordering::Acquire)
        ));
    }

    let (registry, staged) = foundation.close_and_seal();
    assert_eq!(staged, 0);
    assert!(registry.degraded());
    for key in [provisional.key, live.key, requested.key, awaiting.key] {
        let word = registry.inner.as_ref().unwrap().slots[key.slot]
            .word
            .load(Ordering::Acquire);
        assert_eq!(SlotPhase::from_word(word), SlotPhase::Sealed);
    }
    for key in [canceling.key, servicing.key, accepted.key, reconciling.key] {
        let word = registry.inner.as_ref().unwrap().slots[key.slot]
            .word
            .load(Ordering::Acquire);
        assert_eq!(SlotPhase::from_word(word), SlotPhase::Quarantined);
    }
    for key in [
        requested.key,
        awaiting.key,
        servicing.key,
        accepted.key,
        reconciling.key,
    ] {
        assert!(has_reclaim(
            registry.inner.as_ref().unwrap().slots[key.slot]
                .word
                .load(Ordering::Acquire)
        ));
    }
    for key in [provisional.key, live.key, canceling.key] {
        assert!(!has_reclaim(
            registry.inner.as_ref().unwrap().slots[key.slot]
                .word
                .load(Ordering::Acquire)
        ));
    }

    drop(provisional.live);
    drop(live.live);
    drop(canceling.live);
    drop(provisional.reclaim);
    drop(live.reclaim);
    drop(canceling.reclaim);
    let report = retire(registry);
    assert_eq!(report.cleanup_count, 8);
    assert!(report.pre_retirement_degraded);
}
