//! Deterministic tests for the permanent injected destination/listener transaction.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::{panic, panic::AssertUnwindSafe};

use super::injected_control::{
    injected_control_channel, AcceptedBatchFinalizeError, AcceptedBatchFinalizeFailure,
    CommitControlOutcome, InjectedConcreteEventBinding,
};
use super::injected_ids::{injected_node_id_pair, InjectedNodeIdAllocator};
use super::injected_magic_construction::{
    InjectedMagicConstructionError, MAGIC_TEST_COMMIT_PROOF_FAILURE, MAGIC_TEST_FINALIZER_REJECT,
    MAGIC_TEST_PAYLOAD_PANIC, MAGIC_TEST_REJECTED_SHAPE_CORRUPTION,
    MAGIC_TEST_REJECTED_TOKEN_CORRUPTION, MAGIC_TEST_REJECT_EXACT, MAGIC_TEST_TAKE_PROOF_FAILURE,
};
use super::injected_node_construction::InjectedNodeConstructor;
use super::injected_node_lifetime::{
    injected_node_lifetime_registry, BoundInjectedOutputRenderer, InjectedNodeLifetimeRegistrar,
    InjectedNodeRetireOutcome, InjectedRenderReclaimOutcome,
    MagicInitializedInjectedOutputRenderer,
};
use super::{
    AudioContextRegistration, AudioContextState, AudioNodeId, BaseAudioContext,
    ConcreteBaseAudioContext, InjectedConnectionEndpointKind, InjectedContextAdmissionGate,
};
use crate::events::{injected_event_dispatch_setup, EventDispatch, EventLoop, EventLoopExit};
use crate::message::{ControlBatchApplied, ControlBatchSender, ControlMessage};
use crate::node::{AudioNode, GainNode, GainOptions};
use crate::output::{
    AudioOutputEventSink, AudioRenderFormat, AudioRenderStatus, EndpointShutdownConfirmed,
};
use crate::stats::AudioStats;

const SLOT_CAPACITY: usize = 8;

struct MagicHarness {
    renderer: BoundInjectedOutputRenderer,
    constructor: InjectedNodeConstructor,
    binding: InjectedConcreteEventBinding,
    allocator: InjectedNodeIdAllocator,
    registrar: InjectedNodeLifetimeRegistrar,
}

fn harness(initially_suspended: bool) -> MagicHarness {
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) =
        injected_control_channel(gate, 32, initially_suspended).unwrap();
    let (allocator, node_ids, graph) = injected_node_id_pair(0);
    let (registrar, bootstrap) =
        injected_node_lifetime_registry(SLOT_CAPACITY, &producer, node_ids, graph)
            .ok()
            .unwrap();
    let constructor =
        InjectedNodeConstructor::new(producer.clone(), allocator.clone(), registrar.clone())
            .ok()
            .unwrap();
    let (renderer, binding) = render_init
        .build_output_render_thread(
            bootstrap,
            48_000.,
            2,
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            injected_event_dispatch_setup().unwrap(),
        )
        .ok()
        .unwrap()
        .bind_output_lifecycle_exact(lifecycle)
        .ok()
        .unwrap();
    drop(producer);
    MagicHarness {
        renderer,
        constructor,
        binding,
        allocator,
        registrar,
    }
}

fn initialize(harness: MagicHarness) -> MagicInitializedInjectedOutputRenderer {
    ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        harness.renderer,
        harness.constructor,
        harness.binding,
    )
    .ok()
    .unwrap()
    .try_build()
    .ok()
    .unwrap()
}

fn format() -> AudioRenderFormat {
    AudioRenderFormat::new(48_000., 2, 128).unwrap()
}

fn retire(initialized: MagicInitializedInjectedOutputRenderer, render_once: bool) {
    let (events, watcher) = AudioOutputEventSink::bounded(8);
    let (owner, mut callback, event_loop, base) = initialized
        .try_into_audio_output_pair(format(), events)
        .ok()
        .unwrap();
    if render_once {
        let mut output = [1.; 256];
        assert_eq!(
            callback.render_interleaved_f32(&mut output),
            AudioRenderStatus::Continue
        );
        assert!(output.into_iter().all(|sample| sample == 0.));
    }
    drop(base);
    drop(callback);
    drop(watcher);
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
    let reclaimed = match ready.try_reclaim_after_shutdown(EndpointShutdownConfirmed::new()) {
        InjectedRenderReclaimOutcome::Reclaimed(reclaimed) => reclaimed,
        _ => panic!("unique callback and exact magic graph must reclaim"),
    };
    let retired = match reclaimed.try_retire_nodes() {
        InjectedNodeRetireOutcome::Retired(retired) => retired,
        _ => panic!("magic nodes never occupy ordinary lifetime slots"),
    };
    let event_retirement = event_loop.retire_confirmed(&retired, false).ok().unwrap();
    assert_eq!(
        event_retirement.joined.unwrap(),
        EventLoopExit::TerminalClosed
    );
}

#[test]
fn running_magic_graph_is_applied_before_first_real_render_and_next_id_is_eleven() {
    let harness = harness(false);
    let allocator = harness.allocator.clone();
    let registrar = harness.registrar.clone();
    let initialized = initialize(harness);
    let base = initialized.base().clone();
    assert_eq!(base.state(), AudioContextState::Running);
    assert_eq!(base.sample_rate(), 48_000.);
    assert_eq!(base.max_channel_count(), 2);
    assert!(base.applied_control_batch_sequence() >= 1);
    assert!(initialized.magic_bootstrap_shape_is_exact_for_test());

    let destination = base.destination();
    let listener = base.listener();
    assert_eq!(listener.position_x().value(), 0.);
    assert_eq!(listener.position_y().value(), 0.);
    assert_eq!(listener.position_z().value(), 0.);
    assert_eq!(listener.forward_x().value(), 0.);
    assert_eq!(listener.forward_y().value(), 0.);
    assert_eq!(listener.forward_z().value(), -1.);
    assert_eq!(listener.up_x().value(), 0.);
    assert_eq!(listener.up_y().value(), 1.);
    assert_eq!(listener.up_z().value(), 0.);
    assert_eq!(
        [
            listener.position_x().registration().id(),
            listener.position_y().registration().id(),
            listener.position_z().registration().id(),
            listener.forward_x().registration().id(),
            listener.forward_y().registration().id(),
            listener.forward_z().registration().id(),
            listener.up_x().registration().id(),
            listener.up_y().registration().id(),
            listener.up_z().registration().id(),
        ],
        [2, 3, 4, 5, 6, 7, 8, 9, 10].map(AudioNodeId)
    );
    drop(destination);
    drop(listener);
    assert_eq!(
        registrar.slot_phase_counts_for_test(),
        Some([SLOT_CAPACITY, 0, 0, 0, 0, 0])
    );

    let gain = GainNode::new(&base, GainOptions::default());
    assert_eq!(gain.registration().id(), AudioNodeId(11));
    assert_eq!(gain.gain().registration().id(), AudioNodeId(12));
    let next = allocator.try_reserve(1).unwrap();
    assert_eq!(next.id(0), AudioNodeId(13));
    drop(next);
    drop(base);
    retire(initialized, true);
    drop(gain);
}

#[test]
fn suspended_magic_envelope_is_flushed_and_applied_before_callback_publication() {
    let initialized = initialize(harness(true));
    assert_eq!(initialized.base().state(), AudioContextState::Suspended);
    assert!(initialized.base().applied_control_batch_sequence() >= 1);
    retire(initialized, true);
}

#[test]
fn permanent_magic_listener_param_rejects_runtime_mutation_before_host_change() {
    let initialized = initialize(harness(false));
    let base = initialized.base().clone();
    let listener = base.listener();
    let position_x = listener.position_x();
    assert_eq!(position_x.value(), 0.);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| position_x.set_value(2.))).is_err());
    assert_eq!(position_x.value(), 0.);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        position_x.set_value_curve_at_time(&[0., 1., 0.], 0., 1.)
    }))
    .is_err());
    assert_eq!(position_x.value(), 0.);
    drop(listener);
    drop(base);
    retire(initialized, true);
}

#[test]
fn public_exact_connections_select_gain_param_destination_and_listener_endpoint_roles() {
    let initialized = initialize(harness(false));
    let base = initialized.base().clone();
    let destination = base.destination();
    let listener = base.listener();
    let gain = GainNode::new(&base, GainOptions::default());
    let constructor = base.injected_node_constructor().unwrap();
    assert_eq!(
        constructor.connection_edge_count_for_test(),
        0,
        "magic and Gain parameter-owner edges remain hidden from the public mirror"
    );

    gain.connect(&destination);
    gain.gain().connect(&destination);
    destination.connect(&gain);
    listener.position_x().connect(&gain);
    gain.connect(listener.position_y());
    listener.position_z().connect(listener.forward_x());
    gain.gain().connect(&gain);
    gain.connect(gain.gain());
    assert_eq!(constructor.connection_edge_count_for_test(), 8);
    let sequence_after_unique = constructor.last_submitted_batch_sequence();
    gain.connect(&destination);
    listener.position_z().connect(listener.forward_x());
    assert_eq!(constructor.connection_edge_count_for_test(), 8);
    assert_eq!(
        constructor.last_submitted_batch_sequence(),
        sequence_after_unique
    );

    destination.disconnect_dest(&gain);
    listener.position_x().disconnect();
    gain.disconnect_dest(&destination);
    gain.gain().disconnect_dest(&destination);
    gain.disconnect_dest(listener.position_y());
    listener
        .position_z()
        .disconnect_dest_from_output_to_input(listener.forward_x(), 0, 0);
    gain.gain().disconnect_dest(&gain);
    gain.disconnect_dest(gain.gain());
    assert_eq!(constructor.connection_edge_count_for_test(), 0);

    drop(gain);
    drop(listener);
    drop(destination);
    drop(base);
    retire(initialized, true);
}

#[test]
fn permanent_magic_attachment_rejects_foreign_base_and_misindexed_listener_cap_fail_closed() {
    let first = initialize(harness(false));
    let second = initialize(harness(false));
    let destination = first.base().destination();
    let destination_cap = destination
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    let foreign = panic::catch_unwind(AssertUnwindSafe(|| {
        AudioContextRegistration::from_injected_permanent(
            AudioNodeId(0),
            second.base().clone(),
            destination_cap,
            InjectedConnectionEndpointKind::AudioNode,
            1,
            1,
        )
    }));
    assert!(foreign.is_err());
    for initialized in [&first, &second] {
        assert!(panic::catch_unwind(AssertUnwindSafe(|| {
            GainNode::new(initialized.base(), GainOptions::default())
        }))
        .is_err());
    }

    // A separately initialized context proves the accepted listener-param capability cannot be
    // re-labelled as another magic numeric endpoint even when the semantic kind/ports match.
    let misindexed = initialize(harness(false));
    let listener = misindexed.base().listener();
    let position_x = listener.position_x();
    let param_cap = position_x
        .registration()
        .injected_connection_endpoint()
        .unwrap()
        .clone();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        AudioContextRegistration::from_injected_permanent(
            AudioNodeId(3),
            misindexed.base().clone(),
            param_cap,
            InjectedConnectionEndpointKind::AudioParam,
            1,
            1,
        )
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        GainNode::new(misindexed.base(), GainOptions::default())
    }))
    .is_err());

    // These terminal proof mismatches intentionally leave the exact render owners quarantined.
    std::mem::forget((first, second, misindexed));
}

#[test]
fn payload_panic_returns_exact_bootstrap_for_id_zero_retry() {
    let harness = harness(false);
    let allocator = harness.allocator.clone();
    harness
        .constructor
        .set_magic_behavior_for_test(MAGIC_TEST_PAYLOAD_PANIC);
    let bootstrap = ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        harness.renderer,
        harness.constructor,
        harness.binding,
    )
    .ok()
    .unwrap();
    let failure = match bootstrap.try_build() {
        Err(failure) => failure,
        Ok(_) => panic!("forced payload panic unexpectedly initialized magic graph"),
    };
    assert_eq!(
        failure.error(),
        InjectedMagicConstructionError::PayloadPanicked
    );
    let (renderer, constructor, binding) = failure
        .into_retryable_parts()
        .expect("pre-token payload panic must preserve exact retry ownership");
    let initialized =
        ConcreteBaseAudioContext::try_prepare_exact_injected_base(renderer, constructor, binding)
            .ok()
            .unwrap()
            .try_build()
            .ok()
            .unwrap();
    let next = allocator.try_reserve(1).unwrap();
    assert_eq!(next.id(0), AudioNodeId(11));
    drop(next);
    retire(initialized, false);
}

#[test]
fn every_magic_proof_corruption_is_terminal_and_never_recycles_zero_through_ten() {
    let cases = [
        (
            MAGIC_TEST_FINALIZER_REJECT,
            InjectedMagicConstructionError::AcceptedFinalizer(AcceptedBatchFinalizeFailure {
                outcome: CommitControlOutcome::Enqueued { sequence: 1 },
                error: AcceptedBatchFinalizeError::Rejected,
            }),
        ),
        (
            MAGIC_TEST_TAKE_PROOF_FAILURE,
            InjectedMagicConstructionError::NodeIds(
                super::injected_ids::ProvisionalNodeIdError::ProtocolViolation,
            ),
        ),
        (
            MAGIC_TEST_COMMIT_PROOF_FAILURE,
            InjectedMagicConstructionError::NodeIds(
                super::injected_ids::ProvisionalNodeIdError::ProtocolViolation,
            ),
        ),
        (
            MAGIC_TEST_REJECTED_SHAPE_CORRUPTION,
            InjectedMagicConstructionError::ProtocolViolation,
        ),
        (
            MAGIC_TEST_REJECTED_TOKEN_CORRUPTION,
            InjectedMagicConstructionError::ProtocolViolation,
        ),
    ];
    for (behavior, expected) in cases {
        let harness = harness(false);
        let allocator = harness.allocator.clone();
        harness.constructor.set_magic_behavior_for_test(behavior);
        let bootstrap = ConcreteBaseAudioContext::try_prepare_exact_injected_base(
            harness.renderer,
            harness.constructor,
            harness.binding,
        )
        .ok()
        .unwrap();
        let failure = match bootstrap.try_build() {
            Err(failure) => failure,
            Ok(_) => panic!("forced magic proof corruption unexpectedly succeeded"),
        };
        assert_eq!(failure.error(), expected, "behavior {behavior}");
        assert!(
            failure.into_retryable_parts().is_none(),
            "behavior {behavior}"
        );
        let next = allocator.try_reserve(1).unwrap();
        assert_eq!(next.id(0), AudioNodeId(11), "behavior {behavior}");
        drop(next);
    }
}

#[test]
fn untouched_not_accepted_batch_restores_all_eleven_exact_ids_before_admission_releases() {
    let harness = harness(false);
    let allocator = harness.allocator.clone();
    harness
        .constructor
        .set_magic_behavior_for_test(MAGIC_TEST_REJECT_EXACT);
    let bootstrap = ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        harness.renderer,
        harness.constructor,
        harness.binding,
    )
    .ok()
    .unwrap();
    let failure = match bootstrap.try_build() {
        Err(failure) => failure,
        Ok(_) => panic!("forced NotAccepted magic batch unexpectedly succeeded"),
    };
    assert!(matches!(
        failure.error(),
        InjectedMagicConstructionError::Control(_)
    ));
    assert!(failure.into_retryable_parts().is_none());
    let restored = allocator.try_reserve(11).unwrap();
    assert!((0..11).all(|index| restored.id(index) == AudioNodeId(index as u64)));
    drop(restored);
}

fn run_blocked_not_accepted_destructor(panic_after_release: bool) {
    let harness = harness(false);
    let allocator = harness.allocator.clone();
    let gate = harness.constructor.admission_gate();
    harness
        .constructor
        .set_magic_behavior_for_test(MAGIC_TEST_REJECT_EXACT);
    let (started_send, started_recv) = crossbeam_channel::bounded(1);
    let (release_send, release_recv) = crossbeam_channel::bounded(1);
    let restored = harness.constructor.set_magic_drop_probe_for_test(
        started_send,
        release_recv,
        panic_after_release,
    );
    let bootstrap = ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        harness.renderer,
        harness.constructor,
        harness.binding,
    )
    .ok()
    .unwrap();
    let worker = std::thread::spawn(move || {
        let failure = match bootstrap.try_build() {
            Err(failure) => failure,
            Ok(_) => panic!("forced NotAccepted magic batch unexpectedly succeeded"),
        };
        let error = failure.error();
        let terminal = failure.into_retryable_parts().is_none();
        (error, terminal)
    });

    started_recv.recv().unwrap();
    assert!(
        restored.load(Ordering::Acquire),
        "all exact tokens must be back in their provisional slots before processor Drop"
    );
    let sealed = gate.try_seal().unwrap();
    let (capacity_worker, drain) = sealed.into_parts();
    assert!(capacity_worker.is_none());
    assert_eq!(drain.snapshot().graph_controls, 1);
    assert!(!worker.is_finished());

    release_send.send(()).unwrap();
    let (error, terminal) = worker.join().unwrap();
    assert!(terminal);
    assert!(drain.wait().is_drained());
    if panic_after_release {
        assert_eq!(
            error,
            InjectedMagicConstructionError::RejectedPayloadPanicked
        );
        let next = allocator.try_reserve(1).unwrap();
        assert_eq!(next.id(0), AudioNodeId(11));
        drop(next);
    } else {
        assert!(matches!(error, InjectedMagicConstructionError::Control(_)));
        let restored = allocator.try_reserve(11).unwrap();
        assert!((0..11).all(|index| restored.id(index) == AudioNodeId(index as u64)));
        drop(restored);
    }
}

#[test]
fn not_accepted_magic_processor_drop_stays_admitted_after_all_token_restoration() {
    run_blocked_not_accepted_destructor(false);
    run_blocked_not_accepted_destructor(true);
}

#[test]
fn legacy_listener_publication_remains_lazy_and_exactly_once() {
    let (render_send, render_recv) = crossbeam_channel::unbounded();
    let (event_send, event_recv) = crossbeam_channel::unbounded::<EventDispatch>();
    let (_node_return, node_consumer) = llq::Queue::new().split();
    let base = ConcreteBaseAudioContext::new(
        48_000.,
        2,
        Arc::new(AtomicU8::new(AudioContextState::Running as u8)),
        Arc::new(AtomicU64::new(0)),
        render_send.clone(),
        ControlBatchSender::new(render_send),
        ControlBatchApplied::default(),
        event_send,
        EventLoop::new(event_recv),
        false,
        node_consumer,
    );
    let initial = render_recv.try_iter().collect::<Vec<_>>();
    assert_eq!(initial.len(), 1);
    assert!(matches!(
        &initial[0],
        ControlMessage::RegisterNode {
            id: AudioNodeId(0),
            ..
        }
    ));

    let listener = base.listener();
    let published = render_recv.try_iter().collect::<Vec<_>>();
    assert_eq!(published.len(), 20);
    let legacy_registration_order = [1_u64, 10, 9, 8, 7, 6, 5, 4, 3, 2];
    for (position, expected_id) in legacy_registration_order.into_iter().enumerate() {
        assert!(matches!(
            &published[position],
            ControlMessage::RegisterNode { id, .. } if *id == AudioNodeId(expected_id)
        ));
    }
    for (position, expected_from) in (2_u64..=10).enumerate() {
        assert!(matches!(
            &published[10 + position],
            ControlMessage::ConnectNode { from, to, output: 0, input: usize::MAX }
                if *from == AudioNodeId(expected_from) && *to == AudioNodeId(1)
        ));
    }
    assert!(matches!(
        &published[19],
        ControlMessage::ConnectNode {
            from: AudioNodeId(1),
            to: AudioNodeId(0),
            output: 0,
            input: usize::MAX
        }
    ));
    drop(base.listener());
    assert!(render_recv.try_iter().next().is_none());
    drop(listener);
}

#[test]
fn same_control_foreign_node_owner_is_rejected_before_magic_mutation_and_both_reuse() {
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) = injected_control_channel(gate, 32, false).unwrap();
    let (allocator_a, ids_a, graph_a) = injected_node_id_pair(0);
    let (registrar_a, bootstrap_a) =
        injected_node_lifetime_registry(SLOT_CAPACITY, &producer, ids_a, graph_a)
            .ok()
            .unwrap();
    let constructor_a =
        InjectedNodeConstructor::new(producer.clone(), allocator_a.clone(), registrar_a)
            .ok()
            .unwrap();
    let (allocator_b, ids_b, graph_b) = injected_node_id_pair(0);
    let (registrar_b, bootstrap_b) =
        injected_node_lifetime_registry(SLOT_CAPACITY, &producer, ids_b, graph_b)
            .ok()
            .unwrap();
    let constructor_b =
        InjectedNodeConstructor::new(producer.clone(), allocator_b.clone(), registrar_b)
            .ok()
            .unwrap();
    let (renderer, binding) = render_init
        .build_output_render_thread(
            bootstrap_a,
            48_000.,
            2,
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            injected_event_dispatch_setup().unwrap(),
        )
        .ok()
        .unwrap()
        .bind_output_lifecycle_exact(lifecycle)
        .ok()
        .unwrap();
    drop(producer);

    let failure = match ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        renderer,
        constructor_b,
        binding,
    ) {
        Err(failure) => failure,
        Ok(_) => panic!("same-control foreign node owner passed exact preflight"),
    };
    let (renderer, constructor_b, binding) = failure.into_parts();
    let b_ids = allocator_b.try_reserve(11).unwrap();
    assert!((0..11).all(|index| b_ids.id(index) == AudioNodeId(index as u64)));
    drop(b_ids);
    drop(constructor_b);
    drop(bootstrap_b);

    let initialized =
        ConcreteBaseAudioContext::try_prepare_exact_injected_base(renderer, constructor_a, binding)
            .ok()
            .unwrap()
            .try_build()
            .ok()
            .unwrap();
    let a_next = allocator_a.try_reserve(1).unwrap();
    assert_eq!(a_next.id(0), AudioNodeId(11));
    drop(a_next);
    retire(initialized, false);
}

#[test]
fn prepublication_application_panic_is_typed_terminal_after_id_commit() {
    let mut harness = harness(false);
    let allocator = harness.allocator.clone();
    harness.renderer.panic_magic_apply_for_test();
    let bootstrap = ConcreteBaseAudioContext::try_prepare_exact_injected_base(
        harness.renderer,
        harness.constructor,
        harness.binding,
    )
    .ok()
    .unwrap();
    let failure = match bootstrap.try_build() {
        Err(failure) => failure,
        Ok(_) => panic!("forced graph-application panic unexpectedly published a base"),
    };
    assert_eq!(
        failure.error(),
        InjectedMagicConstructionError::ProtocolViolation
    );
    let renderer = failure
        .into_terminal_renderer()
        .expect("accepted application panic must return only terminal render ownership");
    let next = allocator.try_reserve(1).unwrap();
    assert_eq!(next.id(0), AudioNodeId(11));
    drop(next);
    renderer.quarantine_prepublication_for_test();
}
