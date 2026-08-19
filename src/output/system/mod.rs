//! Private system-output foundation.
//!
//! This module deliberately does not reuse `crate::io`: the legacy backends construct and own a
//! second `RenderThread`, while hosted contexts hand an endpoint one exact `AudioRenderCallback`.

#[cfg(all(feature = "cpal", not(feature = "cubeb")))]
mod cpal;
mod none;

use std::cell::UnsafeCell;
use std::fmt;
use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;

use futures_channel::oneshot;

use super::{
    AudioOutputDeathReason, AudioOutputEndpointShutdown, AudioOutputError, AudioOutputErrorKind,
    AudioOutputEventSink, AudioOutputFactory, AudioOutputRequest, AudioRenderCallback,
    AudioRenderStatus, PreparedAudioOutput,
};

const BRIDGE_OPEN: u8 = 0;
const BRIDGE_ACTIVE: u8 = 1;
const BRIDGE_CLOSED: u8 = 2;
const BRIDGE_SUSPENDED: u8 = 4;

/// The future-facing system factory remains crate-private until every feature-selected physical
/// backend has the same callback and thread-retirement proof as the silent endpoint.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SystemAudioOutput;

impl SystemAudioOutput {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl AudioOutputFactory for SystemAudioOutput {
    fn prepare(
        &self,
        request: &AudioOutputRequest,
    ) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
        if request.sink_id() == "none" {
            return none::prepare(request);
        }

        #[cfg(all(feature = "cpal", not(feature = "cubeb")))]
        {
            cpal::prepare(request)
        }

        #[cfg(not(all(feature = "cpal", not(feature = "cubeb"))))]
        {
            Err(AudioOutputError::new(
                AudioOutputErrorKind::NotSupported,
                "physical hosted system output is not enabled in this private foundation",
            ))
        }
    }
}

/// Strong callback ownership retained by the endpoint owner.
///
/// Backend callback closures receive only [`SystemRenderAccess`], which contains a `Weak`. A
/// backend that leaks a failed-init closure therefore cannot retain the proof-critical
/// `AudioRenderCallback`. The callback is manually dropped only by `try_retire`, after the entry
/// gate is closed, no invocation is active, and `Arc::try_unwrap` proves every temporary upgrade
/// has gone away. Weak counts need not be zero: after the sole strong Arc is consumed, even a
/// leaked failed-init closure can only observe `upgrade() == None`. An unexpected owner unwind
/// leaks the callback fail-closed.
struct SystemRenderBridge {
    gate: AtomicU8,
    callback: UnsafeCell<ManuallyDrop<Option<AudioRenderCallback>>>,
    events: AudioOutputEventSink,
}

// SAFETY: callback access is exclusive under `gate`. Owner-side extraction requires a closed,
// inactive gate and Arc uniqueness. The manually managed callback is otherwise intentionally
// leaked, so an unexpected last-Arc drop cannot destroy it on a backend callback thread.
unsafe impl Send for SystemRenderBridge {}
// SAFETY: identical to the `Send` proof; every shared callback-side access first acquires the
// single-entry atomic gate.
unsafe impl Sync for SystemRenderBridge {}

impl SystemRenderBridge {
    fn new(callback: AudioRenderCallback, events: AudioOutputEventSink) -> Arc<Self> {
        Arc::new(Self {
            gate: AtomicU8::new(BRIDGE_OPEN),
            callback: UnsafeCell::new(ManuallyDrop::new(Some(callback))),
            events,
        })
    }

    fn access(this: &Arc<Self>) -> SystemRenderAccess {
        SystemRenderAccess {
            bridge: Arc::downgrade(this),
        }
    }

    fn close(&self) {
        self.gate.fetch_or(BRIDGE_CLOSED, Ordering::AcqRel);
    }

    fn suspend(&self) {
        self.gate.fetch_or(BRIDGE_SUSPENDED, Ordering::AcqRel);
    }

    fn resume(&self) {
        let mut current = self.gate.load(Ordering::Acquire);
        loop {
            if current & BRIDGE_CLOSED != 0 {
                return;
            }
            match self.gate.compare_exchange_weak(
                current,
                current & !BRIDGE_SUSPENDED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn events(&self) -> AudioOutputEventSink {
        self.events.clone()
    }

    fn try_retire(this: Arc<Self>) -> Result<(), Arc<Self>> {
        this.close();
        let gate = this.gate.load(Ordering::Acquire);
        if gate & (BRIDGE_CLOSED | BRIDGE_ACTIVE) != BRIDGE_CLOSED {
            return Err(this);
        }
        let bridge = Arc::try_unwrap(this)?;

        // SAFETY: CLOSED/no-active plus Arc uniqueness proves no backend closure can enter or
        // retain the bridge. This is the sole production extraction of the manually held callback.
        let callback = unsafe { ManuallyDrop::take(&mut *bridge.callback.get()) };
        drop(callback);
        Ok(())
    }
}

#[derive(Clone)]
struct SystemRenderAccess {
    bridge: Weak<SystemRenderBridge>,
}

impl fmt::Debug for SystemRenderAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemRenderAccess")
            .field("available", &self.bridge.strong_count().ne(&0))
            .finish()
    }
}

impl SystemRenderAccess {
    fn is_closed(&self) -> bool {
        self.bridge
            .upgrade()
            .is_none_or(|bridge| bridge.gate.load(Ordering::Acquire) & BRIDGE_CLOSED != 0)
    }

    fn suspend(&self) {
        if let Some(bridge) = self.bridge.upgrade() {
            bridge.suspend();
        }
    }

    fn resume(&self) {
        if let Some(bridge) = self.bridge.upgrade() {
            bridge.resume();
        }
    }

    fn close(&self) {
        if let Some(bridge) = self.bridge.upgrade() {
            bridge.close();
        }
    }

    fn report_endpoint_death(&self, reason: AudioOutputDeathReason) {
        if let Some(bridge) = self.bridge.upgrade() {
            let previous = bridge.gate.fetch_or(BRIDGE_CLOSED, Ordering::AcqRel);
            if previous & BRIDGE_CLOSED == 0 {
                let _ = bridge.events.report_endpoint_death(reason);
            }
        }
    }

    /// Invokes one already-bounded, nonempty, channel-aligned logical callback.
    fn render_interleaved_f32(&self, output: &mut [f32]) -> AudioRenderStatus {
        let Some(bridge) = self.bridge.upgrade() else {
            output.fill(0.);
            return AudioRenderStatus::Stop;
        };

        let mut current = bridge.gate.load(Ordering::Acquire);
        loop {
            if current & BRIDGE_CLOSED != 0 {
                output.fill(0.);
                return AudioRenderStatus::Stop;
            }
            if current & BRIDGE_SUSPENDED != 0 {
                output.fill(0.);
                return AudioRenderStatus::Continue;
            }
            if current & BRIDGE_ACTIVE != 0 {
                let previous = bridge.gate.fetch_or(BRIDGE_CLOSED, Ordering::AcqRel);
                output.fill(0.);
                if previous & BRIDGE_CLOSED == 0 {
                    let _ = bridge
                        .events
                        .report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                }
                return AudioRenderStatus::Stop;
            }
            match bridge.gate.compare_exchange_weak(
                current,
                current | BRIDGE_ACTIVE,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }

        let active = BridgeActive { bridge: &bridge };
        // SAFETY: the successful OPEN->ACTIVE transition grants this invocation the sole mutable
        // callback access. Owner extraction cannot occur until this guard publishes CLOSED.
        let callback = unsafe { &mut *bridge.callback.get() };
        let status = callback
            .as_mut()
            .map_or(AudioRenderStatus::Stop, |callback| {
                callback.render_interleaved_f32(output)
            });
        drop(active);
        status
    }
}

struct BridgeActive<'a> {
    bridge: &'a SystemRenderBridge,
}

impl Drop for BridgeActive<'_> {
    fn drop(&mut self) {
        let previous = self
            .bridge
            .gate
            .fetch_and(!BRIDGE_ACTIVE, Ordering::Release);
        debug_assert!(previous & BRIDGE_ACTIVE != 0);
    }
}

type OwnerResult = Result<(), AudioOutputError>;

/// Single-owner observation of a backend owner that is joined by receipt infrastructure.
struct OwnerCompletion {
    receiver: oneshot::Receiver<OwnerResult>,
}

impl fmt::Debug for OwnerCompletion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnerCompletion").finish_non_exhaustive()
    }
}

impl OwnerCompletion {
    fn into_shutdown(self) -> AudioOutputEndpointShutdown {
        AudioOutputEndpointShutdown::from_future(async move {
            match self.receiver.await {
                Ok(result) => result,
                Err(_) => Err(AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "system output join observation was lost",
                )),
            }
        })
    }

    fn into_bridge_shutdown(self, bridge: Arc<SystemRenderBridge>) -> AudioOutputEndpointShutdown {
        bridge.close();
        AudioOutputEndpointShutdown::from_future(async move {
            let owner_result = match self.receiver.await {
                Ok(result) => result,
                Err(_) => {
                    // Manually-held callback ownership remains fail-closed when this Arc drops;
                    // without join proof it must not be explicitly destroyed.
                    drop(bridge);
                    return Err(AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "system output join observation was lost",
                    ));
                }
            };

            match SystemRenderBridge::try_retire(bridge) {
                Ok(()) => owner_result,
                Err(bridge) => {
                    // A strong backend authority surviving owner join contradicts the endpoint
                    // proof. Permanently quarantine it rather than run the callback destructor.
                    std::mem::forget(bridge);
                    Err(AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "system output callback authority survived owner join",
                    ))
                }
            }
        })
    }
}

struct JoinObserverInstall {
    completion: OwnerCompletion,
    blocking_join: crossbeam_channel::Receiver<OwnerResult>,
}

struct JoinObserverInstallFailure {
    error: AudioOutputError,
    owner: JoinHandle<OwnerResult>,
}

/// Installs a real wake source after the backend owner has actually joined.
///
/// The observer is receipt infrastructure and owns no endpoint resource other than the owner's
/// `JoinHandle`. Its oneshot send wakes the lifecycle poller; no `JoinHandle::is_finished` polling
/// is used. The Arc cell exists solely so observer spawn failure can return the exact handle for a
/// synchronous prepare-time abort/join instead of detaching it.
fn install_join_observer(
    owner: JoinHandle<OwnerResult>,
) -> Result<JoinObserverInstall, JoinObserverInstallFailure> {
    let owner = Arc::new(Mutex::new(Some(owner)));
    let observer_owner = Arc::clone(&owner);
    let (completion_send, completion_recv) = oneshot::channel();
    let (blocking_send, blocking_join) = crossbeam_channel::bounded(1);
    let observer = std::thread::Builder::new()
        .name("web-audio-system-output-join".into())
        .spawn(move || {
            let owner = match observer_owner.lock() {
                Ok(mut slot) => slot.take(),
                Err(poisoned) => poisoned.into_inner().take(),
            };
            let result = owner.map_or_else(
                || {
                    Err(AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "system output owner handle was unavailable to its join observer",
                    ))
                },
                |owner| match owner.join() {
                    Ok(result) => result,
                    Err(payload) => {
                        std::mem::forget(payload);
                        Err(AudioOutputError::new(
                            AudioOutputErrorKind::Shutdown,
                            "system output owner panicked outside its containment boundary",
                        ))
                    }
                },
            );
            let _ = blocking_send.send(result.clone());
            let _ = completion_send.send(result);
        });

    if let Err(error) = observer {
        let owner_handle = match owner.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
        .expect("a failed observer spawn cannot consume the owner handle");
        return Err(JoinObserverInstallFailure {
            error: AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                format!("failed to spawn system output join observer: {error}"),
            ),
            owner: owner_handle,
        });
    }

    Ok(JoinObserverInstall {
        completion: OwnerCompletion {
            receiver: completion_recv,
        },
        blocking_join,
    })
}

fn contained_owner<F>(events: &Mutex<Option<AudioOutputEventSink>>, run: F) -> OwnerResult
where
    F: FnOnce() -> OwnerResult,
{
    match panic::catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => {
            if let Some(events) = match events.lock() {
                Ok(events) => events,
                Err(poisoned) => poisoned.into_inner(),
            }
            .as_ref()
            {
                let _ = events.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
            }
            std::mem::forget(payload);
            Err(AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "system output owner panicked",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future as _;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use futures::executor;
    use futures::task::{waker, ArcWake};

    use super::*;
    use crate::context::{
        AudioContext, AudioContextLatencyCategory, AudioContextOptions,
        AudioContextRenderSizeCategory, AudioContextShutdownOutcome, AudioContextState,
        BaseAudioContext,
    };
    use crate::node::{AudioNode, AudioScheduledSourceNode};
    use crate::output::{
        audio_render_test_pair, AudioOutputConfig, AudioOutputContextId, AudioRenderFormat,
        EndpointShutdownConfirmed,
    };

    fn request(channels: usize, sample_rate: Option<f32>) -> AudioOutputRequest {
        AudioOutputRequest::new(
            AudioOutputContextId::new(1).unwrap(),
            "none",
            sample_rate,
            channels,
            AudioContextLatencyCategory::Interactive,
            AudioContextRenderSizeCategory::Default,
            None,
        )
        .unwrap()
    }

    fn test_callback(
        format: AudioRenderFormat,
        events: AudioOutputEventSink,
        renders: Arc<AtomicUsize>,
    ) -> (crate::output::AudioRenderOwner, AudioRenderCallback) {
        audio_render_test_pair(
            format,
            events,
            move |output| {
                renders.fetch_add(1, AtomicOrdering::AcqRel);
                output.fill(0.25);
            },
            || Ok(()),
        )
    }

    fn reclaim(owner: crate::output::AudioRenderOwner) {
        owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }

    #[test]
    fn bridge_is_exclusive_allocation_free_and_retires_only_after_weak_access() {
        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (events, watcher) = AudioOutputEventSink::bounded(4);
        let renders = Arc::new(AtomicUsize::new(0));
        let (owner, callback) = test_callback(format, events.clone(), Arc::clone(&renders));
        let bridge = SystemRenderBridge::new(callback, events);
        let access = SystemRenderBridge::access(&bridge);
        let mut output = [0.; 256];

        alloc_counter::deny_alloc(|| {
            assert_eq!(
                access.render_interleaved_f32(&mut output),
                AudioRenderStatus::Continue
            );
        });
        assert_eq!(renders.load(AtomicOrdering::Acquire), 1);
        assert!(output.iter().all(|sample| *sample == 0.25));

        owner.begin_shutdown();
        assert_eq!(
            access.render_interleaved_f32(&mut output),
            AudioRenderStatus::Stop
        );
        bridge.close();
        assert!(SystemRenderBridge::try_retire(bridge).is_ok());
        assert!(watcher.death_reason().is_none());
        reclaim(owner);
    }

    #[test]
    fn bridge_refuses_concurrent_entry_and_latches_backend_death() {
        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (events, watcher) = AudioOutputEventSink::bounded(4);
        let renders = Arc::new(AtomicUsize::new(0));
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let renders_for_callback = Arc::clone(&renders);
        let (owner, callback) = audio_render_test_pair(
            format,
            events.clone(),
            move |output| {
                renders_for_callback.fetch_add(1, AtomicOrdering::AcqRel);
                entered_send.send(()).unwrap();
                release_recv.recv().unwrap();
                output.fill(0.5);
            },
            || Ok(()),
        );
        let bridge = SystemRenderBridge::new(callback, events);
        let first_access = SystemRenderBridge::access(&bridge);
        let second_access = first_access.clone();
        let render_thread = std::thread::spawn(move || {
            let mut output = [0.; 256];
            first_access.render_interleaved_f32(&mut output)
        });
        entered_recv.recv_timeout(Duration::from_secs(1)).unwrap();

        let mut refused = [1.; 256];
        assert_eq!(
            second_access.render_interleaved_f32(&mut refused),
            AudioRenderStatus::Stop
        );
        assert!(refused.iter().all(|sample| *sample == 0.));
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::BackendFailure)
        );

        release_send.send(()).unwrap();
        assert_eq!(render_thread.join().unwrap(), AudioRenderStatus::Continue);
        owner.begin_shutdown();
        bridge.close();
        assert!(SystemRenderBridge::try_retire(bridge).is_ok());
        reclaim(owner);
    }

    #[test]
    fn silent_endpoint_abort_and_running_shutdown_are_join_observed() {
        let factory = SystemAudioOutput::new();
        let prepared = factory.prepare(&request(3, Some(44_100.))).unwrap();
        assert_eq!(
            prepared.config(),
            &AudioOutputConfig::new(AudioRenderFormat::new(44_100., 3, 128).unwrap(), "none", 0.)
                .unwrap()
        );
        executor::block_on(prepared.abort()).unwrap();

        let prepared = factory.prepare(&request(2, None)).unwrap();
        let format = prepared.config().format();
        let (events, watcher) = AudioOutputEventSink::bounded(4);
        let renders = Arc::new(AtomicUsize::new(0));
        let (owner, callback) = test_callback(format, events.clone(), Arc::clone(&renders));
        let mut running = prepared.start(callback, events).unwrap();
        let started = std::time::Instant::now();
        while renders.load(AtomicOrdering::Acquire) == 0 {
            assert!(started.elapsed() < Duration::from_secs(1));
            std::thread::yield_now();
        }
        running.suspend().unwrap();
        let suspended = renders.load(AtomicOrdering::Acquire);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(renders.load(AtomicOrdering::Acquire), suspended);
        running.resume().unwrap();

        owner.begin_shutdown();
        executor::block_on(running.shutdown()).unwrap();
        assert!(watcher.death_reason().is_none());
        reclaim(owner);
    }

    #[test]
    fn start_failure_future_retains_callback_until_render_shutdown_begins() {
        let prepared = none::prepare_with_start_failure_for_test(&request(2, None)).unwrap();
        let format = prepared.config().format();
        let (events, watcher) = AudioOutputEventSink::bounded(4);
        let renders = Arc::new(AtomicUsize::new(0));
        let (owner, callback) = test_callback(format, events.clone(), renders);

        let failure = match prepared.start(callback, events) {
            Ok(_) => panic!("forced silent output start unexpectedly succeeded"),
            Err(failure) => failure,
        };
        assert_eq!(
            failure.error().message(),
            "forced silent system output start failure"
        );
        // Returning the failure must not destroy the still-open exact callback. The lifecycle
        // owns the matching render owner and closes it before polling endpoint cleanup.
        assert!(watcher.death_reason().is_none());

        owner.begin_shutdown();
        let (_, shutdown) = failure.into_parts();
        executor::block_on(shutdown).unwrap();
        assert!(watcher.death_reason().is_none());
        reclaim(owner);
    }

    #[test]
    fn join_observer_wakes_only_after_the_backend_owner_really_joins() {
        #[derive(Default)]
        struct WakeCounter(AtomicUsize);
        impl ArcWake for WakeCounter {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.0.fetch_add(1, AtomicOrdering::AcqRel);
            }
        }

        struct OwnerDrop(Arc<AtomicUsize>);
        impl Drop for OwnerDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, AtomicOrdering::AcqRel);
            }
        }

        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let dropped = Arc::new(AtomicUsize::new(0));
        let owner_dropped = Arc::clone(&dropped);
        let owner = std::thread::spawn(move || {
            let _drop = OwnerDrop(owner_dropped);
            release_recv.recv().unwrap();
            Ok(())
        });
        let observed = install_join_observer(owner).map_err(|_| ()).unwrap();
        drop(observed.blocking_join);
        let mut shutdown = Box::pin(observed.completion.into_shutdown());
        let wakes = Arc::new(WakeCounter::default());
        let task_waker = waker(Arc::clone(&wakes));
        let mut cx = Context::from_waker(&task_waker);
        assert!(matches!(shutdown.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(dropped.load(AtomicOrdering::Acquire), 0);

        release_send.send(()).unwrap();
        let started = std::time::Instant::now();
        while wakes.0.load(AtomicOrdering::Acquire) == 0 {
            assert!(started.elapsed() < Duration::from_secs(1));
            std::thread::yield_now();
        }
        assert_eq!(dropped.load(AtomicOrdering::Acquire), 1);
        executor::block_on(shutdown).unwrap();
    }

    #[test]
    fn shutdown_waits_for_an_active_callback_then_wakes_after_destruction_and_join() {
        #[derive(Default)]
        struct WakeCounter(AtomicUsize);
        impl ArcWake for WakeCounter {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.0.fetch_add(1, AtomicOrdering::AcqRel);
            }
        }

        let factory = SystemAudioOutput::new();
        let prepared = factory.prepare(&request(2, None)).unwrap();
        let format = prepared.config().format();
        let (events, watcher) = AudioOutputEventSink::bounded(4);
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let (owner, callback) = audio_render_test_pair(
            format,
            events.clone(),
            move |output| {
                entered_send.send(()).unwrap();
                release_recv.recv().unwrap();
                output.fill(0.5);
            },
            || Ok(()),
        );
        let running = prepared.start(callback, events).unwrap();
        entered_recv.recv_timeout(Duration::from_secs(1)).unwrap();

        owner.begin_shutdown();
        let mut shutdown = Box::pin(running.shutdown());
        let wakes = Arc::new(WakeCounter::default());
        let task_waker = waker(Arc::clone(&wakes));
        let mut cx = Context::from_waker(&task_waker);
        assert!(matches!(shutdown.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(
            Arc::strong_count(owner.slot.as_ref().unwrap()),
            2,
            "the active callback must remain alive while endpoint cleanup is pending"
        );

        release_send.send(()).unwrap();
        let started = std::time::Instant::now();
        while wakes.0.load(AtomicOrdering::Acquire) == 0 {
            assert!(started.elapsed() < Duration::from_secs(1));
            std::thread::yield_now();
        }
        executor::block_on(shutdown).unwrap();
        assert_eq!(
            Arc::strong_count(owner.slot.as_ref().unwrap()),
            1,
            "Ok endpoint cleanup must prove callback destruction"
        );
        assert!(watcher.death_reason().is_none());
        reclaim(owner);
    }

    #[test]
    fn owner_panics_before_and_after_start_response_are_contained_and_quarantined() {
        for after_response in [false, true] {
            let prepared =
                none::prepare_with_start_panic_for_test(&request(2, None), after_response).unwrap();
            let format = prepared.config().format();
            let (events, watcher) = AudioOutputEventSink::bounded(4);
            let renders = Arc::new(AtomicUsize::new(0));
            let (owner, callback) = test_callback(format, events.clone(), renders);

            if after_response {
                let running = prepared.start(callback, events).unwrap();
                owner.begin_shutdown();
                assert!(executor::block_on(running.shutdown()).is_err());
            } else {
                let failure = match prepared.start(callback, events) {
                    Ok(_) => panic!("panicking silent output unexpectedly started"),
                    Err(failure) => failure,
                };
                owner.begin_shutdown();
                let (_, shutdown) = failure.into_parts();
                assert!(executor::block_on(shutdown).is_err());
            }

            assert_eq!(
                watcher.death_reason(),
                Some(AudioOutputDeathReason::BackendFailure)
            );
            assert_eq!(
                Arc::strong_count(owner.slot.as_ref().unwrap()),
                1,
                "joined panic cleanup may report Err only after callback destruction"
            );
            // An Err receipt does not mint EndpointShutdownConfirmed; preserve the render owner
            // through its intentional fail-closed Drop path.
            drop(owner);
        }
    }

    #[test]
    fn private_system_factory_drives_a_real_hosted_none_context() {
        let context = AudioContext::builder(Arc::new(SystemAudioOutput::new()))
            .options(AudioContextOptions {
                sink_id: "none".into(),
                ..AudioContextOptions::default()
            })
            .number_of_channels(2)
            .build()
            .unwrap();
        assert_eq!(context.sample_rate(), 48_000.);
        assert_eq!(context.sink_id(), "none");
        assert_eq!(context.state(), AudioContextState::Running);

        let gain = context.create_gain();
        gain.gain().set_value(0.25);
        let mut oscillator = context.create_oscillator();
        oscillator.connect(&gain);
        gain.connect(&context.destination());
        let ended = Arc::new(AtomicBool::new(false));
        let ended_callback = Arc::clone(&ended);
        oscillator.set_onended(move |_| {
            ended_callback.store(true, AtomicOrdering::Release);
        });
        oscillator.start();
        oscillator.stop_at(context.current_time() + 0.02);
        let started = std::time::Instant::now();
        while !ended.load(AtomicOrdering::Acquire) {
            assert!(started.elapsed() < Duration::from_secs(1));
            std::thread::yield_now();
        }

        executor::block_on(context.suspend());
        assert_eq!(context.state(), AudioContextState::Suspended);
        let suspended_time = context.current_time();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(context.current_time(), suspended_time);
        executor::block_on(context.resume());
        executor::block_on(context.close());
        assert_eq!(context.state(), AudioContextState::Closed);
    }

    #[test]
    fn private_system_contexts_isolate_initial_suspend_resume_and_close_receipts() {
        let suspended = AudioContext::builder(Arc::new(SystemAudioOutput::new()))
            .options(AudioContextOptions {
                sink_id: "none".into(),
                ..AudioContextOptions::default()
            })
            .initially_suspended(true)
            .build()
            .unwrap();
        let running = AudioContext::builder(Arc::new(SystemAudioOutput::new()))
            .options(AudioContextOptions {
                sink_id: "none".into(),
                ..AudioContextOptions::default()
            })
            .build()
            .unwrap();

        assert_eq!(suspended.state(), AudioContextState::Suspended);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(suspended.current_time(), 0.);
        let started = std::time::Instant::now();
        while running.current_time() == 0. {
            assert!(started.elapsed() < Duration::from_secs(1));
            std::thread::yield_now();
        }

        executor::block_on(suspended.resume());
        assert_eq!(suspended.state(), AudioContextState::Running);
        assert_eq!(running.state(), AudioContextState::Running);

        let suspended_close = suspended.request_close().unwrap();
        drop(suspended);
        assert!(matches!(
            suspended_close.wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(running.state(), AudioContextState::Running);
        assert!(matches!(
            running.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(running.state(), AudioContextState::Closed);
    }
}
