//! Deterministic tests for exact injected renderer/output-lifecycle binding and callback bootstrap.

use std::sync::atomic::{AtomicU64, AtomicU8};
use std::sync::Arc;

use super::injected_control::{
    injected_control_channel, BoundInjectedRenderer, InjectedControlLifecycleOwner,
};
use super::injected_ids::injected_node_id_pair;
use super::injected_node_lifetime::{
    injected_node_lifetime_registry, TestBoundInjectedOutputRenderer,
};
use super::{AudioContextState, InjectedContextAdmissionGate};
use crate::events::EventDispatch;
use crate::output::{
    AudioOutputErrorKind, AudioOutputEventSink, AudioRenderFormat, EndpointShutdownConfirmed,
};
use crate::stats::AudioStats;

fn bound_renderer() -> (BoundInjectedRenderer, InjectedControlLifecycleOwner) {
    let gate = InjectedContextAdmissionGate::new();
    let (producer, lifecycle, render_init) = injected_control_channel(gate, 8, false).unwrap();
    let (_allocator, node_ids, graph) = injected_node_id_pair(0);
    let (_registrar, bootstrap) = injected_node_lifetime_registry(8, &producer, node_ids, graph)
        .ok()
        .unwrap();
    let (event_sender, _event_receiver) = crossbeam_channel::unbounded::<EventDispatch>();
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
    (bound, lifecycle)
}

fn output_events() -> AudioOutputEventSink {
    AudioOutputEventSink::bounded(4).0
}

fn install_and_reclaim(bound: TestBoundInjectedOutputRenderer, events: AudioOutputEventSink) {
    let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
    let (owner, mut callback) = bound
        .try_into_audio_render_thread_pair(format, events)
        .ok()
        .unwrap();
    let (render, node_lifetimes, lifecycle) = owner.into_parts_for_test();
    assert!(node_lifetimes
        .control_identity()
        .ptr_eq(&lifecycle.identity()));
    let retirement = lifecycle.try_begin_close().ok().unwrap();
    let (_snapshot, drained) = retirement.retire_and_wait();
    let sealed = node_lifetimes
        .seal_after_control_drain(&drained)
        .ok()
        .unwrap();
    let transport = drained.finish().ok().unwrap();
    assert_eq!(transport.payloads.staged_len(), 0);
    let mut output = [0.; 256];
    let _ = callback.render_interleaved_f32(&mut output);
    let _observed = transport.close.try_observe_exact().ok().unwrap();
    drop(callback);
    render.begin_shutdown();
    render
        .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
        .ok()
        .unwrap()
        .unwrap();
    // B3a has no whole-graph proof yet. The sealed registry uses its existing fail-closed Drop;
    // the mandatory GC thread and exact control close were authoritatively retired above.
    drop(sealed);
}

#[test]
fn foreign_control_bind_returns_both_exact_bundles_operationally_reusable() {
    let (first_renderer, first_control) = bound_renderer();
    let (second_renderer, second_control) = bound_renderer();

    let failure = match first_renderer.bind_output_lifecycle(second_control) {
        Ok(_) => panic!("foreign control owner must not bind"),
        Err(failure) => failure,
    };
    let first = failure
        .renderer
        .bind_output_lifecycle(first_control)
        .ok()
        .unwrap();
    let second = second_renderer
        .bind_output_lifecycle(failure.control)
        .ok()
        .unwrap();

    install_and_reclaim(first, output_events());
    install_and_reclaim(second, output_events());
}

#[test]
fn gc_spawn_failure_returns_exact_bound_bundle_and_event_sink_for_retry() {
    let (renderer, control) = bound_renderer();
    let mut bound = renderer.bind_output_lifecycle(control).ok().unwrap();
    bound.fail_next_gc_spawn_for_test();
    let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
    let (events, watcher) = AudioOutputEventSink::bounded(4);

    let failure = match bound.try_into_audio_render_thread_pair(format, events) {
        Ok(_) => panic!("forced GC spawn failure must return ownership"),
        Err(failure) => failure,
    };
    assert_eq!(failure.error.kind(), AudioOutputErrorKind::BackendSpecific);
    assert!(failure.error.message().contains("forced garbage collector"));
    assert!(watcher.death_reason().is_none());

    install_and_reclaim(failure.renderer, failure.events);
    assert!(watcher.death_reason().is_some());
}

#[test]
fn legacy_joinable_gc_path_still_installs_and_joins() {
    let (renderer, lifecycle) = bound_renderer();
    let (mut renderer, node_lifetimes) = renderer.into_render_thread_for_test();
    let gc = renderer
        .spawn_joinable_garbage_collector_thread()
        .expect("fresh legacy renderer installs its GC sidecar");
    assert!(gc.thread().name().is_none());
    assert!(renderer.spawn_joinable_garbage_collector_thread().is_none());
    drop(renderer);
    gc.join().unwrap();
    drop(node_lifetimes);
    drop(lifecycle);
}
