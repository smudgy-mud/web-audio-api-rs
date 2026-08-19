//! Admission and sealing foundation for the future injected online context.
//!
//! A private construction seam wires this gate to injected control-side event production and
//! render-capacity workers. No public `AudioContext` constructor selects that seam yet. Legacy and
//! offline contexts therefore keep their existing producer behavior. The gate covers
//! control-owned producers only: raw render-owned `EventDispatch` production remains outside this
//! phase.

#![allow(dead_code)] // Private foundation; public injected-context selection remains deferred.

use std::any::Any;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, TryLockError};

/// Linearized admission failure. No admission call waits for the phase lock or a resource quota.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    /// Another short phase/registry operation currently owns the serialization lock.
    Contended,
    /// The irreversible seal has committed, so no new control-owned producer may start.
    Sealed,
    /// A panic poisoned the phase/registry lock. Callers must fail closed.
    Poisoned,
    /// An in-flight counter or the capacity-worker identifier namespace is exhausted.
    Exhausted,
    /// The single capacity-worker slot is pending, committed, or still exiting.
    CapacityWorkerActive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdmissionPhase {
    Open,
    Sealed,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CapacityWorkerId(NonZeroU64);

impl CapacityWorkerId {
    pub(crate) const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Join failure for one gate-created capacity worker.
#[derive(Debug)]
pub(crate) enum CapacityWorkerJoinError {
    /// The worker panicked; joining still proved that the thread and its producer lease retired.
    Panicked(Box<dyn Any + Send + 'static>),
}

/// Explicit stop/join owner for one started capacity worker.
///
/// This type has no general constructor: it can only be produced by
/// `CapacityWorkerRegistration::start`, which binds the matching id, producer lease, bounded stop
/// authority, and concrete `JoinHandle`. `stop_and_join` may block and must run off RT. Dropping
/// this value requests stop without joining and therefore makes no retirement claim.
#[must_use = "capacity worker retirement must be explicitly run or quarantined"]
pub(crate) struct CapacityWorkerRetirement {
    id: CapacityWorkerId,
    stop: crossbeam_channel::Sender<()>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl CapacityWorkerRetirement {
    pub(crate) const fn id(&self) -> CapacityWorkerId {
        self.id
    }

    fn request_stop(&self) {
        // Capacity one and non-waiting. Full means a stop is already pending; disconnected means
        // the worker has already released its receiver.
        let _ = self.stop.try_send(());
    }

    /// Requests stop and joins on the current, necessarily non-render, thread.
    pub(crate) fn stop_and_join(mut self) -> Result<(), CapacityWorkerJoinError> {
        self.request_stop();
        let join = self
            .join
            .take()
            .expect("private capacity worker is joined at most once");
        join.join().map_err(CapacityWorkerJoinError::Panicked)
    }
}

impl Drop for CapacityWorkerRetirement {
    fn drop(&mut self) {
        self.request_stop();
    }
}

impl std::fmt::Debug for CapacityWorkerRetirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapacityWorkerRetirement")
            .field("id", &self.id)
            .field("joinable", &self.join.is_some())
            .finish()
    }
}

struct AdmissionState {
    phase: AdmissionPhase,
    next_capacity_worker_id: u64,
    capacity_worker: Option<CapacityWorkerRetirement>,
    drain_receiver: Option<crossbeam_channel::Receiver<()>>,
}

impl AdmissionState {
    fn new(drain_receiver: crossbeam_channel::Receiver<()>) -> Self {
        Self {
            phase: AdmissionPhase::Open,
            next_capacity_worker_id: 1,
            capacity_worker: None,
            drain_receiver: Some(drain_receiver),
        }
    }
}

#[derive(Debug, Default)]
struct AdmissionCounters {
    external_events: AtomicUsize,
    graph_controls: AtomicUsize,
    capacity_registrations: AtomicUsize,
    capacity_producers: AtomicUsize,
}

struct AdmissionInner {
    state: Mutex<AdmissionState>,
    counters: AdmissionCounters,
    /// Coalesced wake only; counter snapshots remain authoritative.
    drain_wake: crossbeam_channel::Sender<()>,
}

#[derive(Clone, Copy)]
enum AdmissionKind {
    ExternalEvent,
    GraphControl,
    CapacityRegistration,
    CapacityProducer,
}

impl AdmissionInner {
    fn counter(&self, kind: AdmissionKind) -> &AtomicUsize {
        match kind {
            AdmissionKind::ExternalEvent => &self.counters.external_events,
            AdmissionKind::GraphControl => &self.counters.graph_controls,
            AdmissionKind::CapacityRegistration => &self.counters.capacity_registrations,
            AdmissionKind::CapacityProducer => &self.counters.capacity_producers,
        }
    }

    fn try_increment(&self, kind: AdmissionKind) -> Result<(), AdmissionError> {
        self.counter(kind)
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1)
            })
            .map(|_| ())
            .map_err(|_| AdmissionError::Exhausted)
    }

    fn decrement(&self, kind: AdmissionKind) {
        let previous = self.counter(kind).fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "admission permit count underflow");
        // Bounded, coalesced, and best-effort. It does not wait for channel capacity, but the
        // channel implementation is not claimed to be lock-free.
        let _ = self.drain_wake.try_send(());
    }

    fn snapshot(&self) -> AdmissionSnapshot {
        AdmissionSnapshot {
            external_events: self.counters.external_events.load(Ordering::Acquire),
            graph_controls: self.counters.graph_controls.load(Ordering::Acquire),
            capacity_registrations: self.counters.capacity_registrations.load(Ordering::Acquire),
            capacity_producers: self.counters.capacity_producers.load(Ordering::Acquire),
        }
    }
}

struct AdmissionPermit {
    inner: Arc<AdmissionInner>,
    kind: AdmissionKind,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.inner.decrement(self.kind);
    }
}

/// Short permit held from immediately before constructing/sending one control-owned external
/// event until the send attempt and any caller-side payload cleanup are complete.
#[must_use = "dropping the event permit completes the in-flight admission"]
pub(crate) struct ExternalEventAdmission(AdmissionPermit);

/// Short permit held across one graph/control operation, including mutation of staging queues and
/// the render-channel send attempt.
#[must_use = "dropping the control permit completes the in-flight admission"]
pub(crate) struct GraphControlAdmission(AdmissionPermit);

/// Non-clone, long-lived lease moved into a capacity worker and held until that worker exits.
///
/// Each individual capacity event will also use an `ExternalEventAdmission`; this lease accounts
/// for the producer that could initiate future sends and lets sealing wait for its termination.
#[must_use = "the capacity producer lease must live until the worker exits"]
struct CapacityProducerLease(AdmissionPermit);

/// Pre-spawn reservation for installing one capacity worker into the retirement registry.
#[must_use = "capacity registration must be committed or dropped"]
pub(crate) struct CapacityWorkerRegistration {
    id: CapacityWorkerId,
    permit: AdmissionPermit,
    producer: CapacityProducerLease,
}

impl CapacityWorkerRegistration {
    pub(crate) const fn id(&self) -> CapacityWorkerId {
        self.id
    }

    /// Starts the worker and binds its sole producer lease to the concrete thread lifetime.
    ///
    /// The worker body receives only the capacity-one stop receiver; it cannot extract, drop, or
    /// associate the producer lease with another worker. A spawn failure drops both reservation
    /// counters without creating a running thread.
    pub(crate) fn start<F>(self, worker: F) -> std::io::Result<CapacityStartedWorker>
    where
        F: FnOnce(crossbeam_channel::Receiver<()>) + Send + 'static,
    {
        let Self {
            id,
            permit,
            producer,
        } = self;
        let (stop, stop_receiver) = crossbeam_channel::bounded(1);
        let join = std::thread::Builder::new()
            .name("web-audio-capacity".to_owned())
            .spawn(move || {
                let producer = producer;
                worker(stop_receiver);
                drop(producer);
            })?;
        Ok(CapacityStartedWorker {
            id,
            registration: permit,
            retirement: CapacityWorkerRetirement {
                id,
                stop,
                join: Some(join),
            },
        })
    }
}

/// Started but not yet registry-committed worker. All authorities are bound to the same id.
#[must_use = "started capacity worker must be committed or stopped and joined"]
pub(crate) struct CapacityStartedWorker {
    id: CapacityWorkerId,
    registration: AdmissionPermit,
    retirement: CapacityWorkerRetirement,
}

impl CapacityStartedWorker {
    pub(crate) const fn id(&self) -> CapacityWorkerId {
        self.id
    }

    /// Installs this exact started worker without waiting for registry contention.
    ///
    /// On failure the complete started-worker typestate is returned. A worker that loses the seal
    /// race can therefore still be authoritatively stopped and joined.
    pub(crate) fn try_commit(self) -> Result<CapacityWorkerId, CapacityWorkerCommitFailure> {
        let inner = Arc::clone(&self.registration.inner);
        let mut state = match inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => {
                return Err(CapacityWorkerCommitFailure {
                    error: AdmissionError::Contended,
                    worker: self,
                });
            }
            Err(TryLockError::Poisoned(_)) => {
                return Err(CapacityWorkerCommitFailure {
                    error: AdmissionError::Poisoned,
                    worker: self,
                });
            }
        };

        if state.phase == AdmissionPhase::Sealed {
            drop(state);
            return Err(CapacityWorkerCommitFailure {
                error: AdmissionError::Sealed,
                worker: self,
            });
        }

        debug_assert!(
            state.capacity_worker.is_none(),
            "capacity worker slot is unique"
        );
        let Self {
            id,
            registration,
            retirement,
        } = self;
        state.capacity_worker = Some(retirement);
        drop(state);
        drop(registration);
        Ok(id)
    }

    /// Stops and joins a worker that was not committed or was returned by a failed commit.
    pub(crate) fn stop_and_join(self) -> Result<(), CapacityWorkerJoinError> {
        let Self {
            registration,
            retirement,
            ..
        } = self;
        let result = retirement.stop_and_join();
        drop(registration);
        result
    }
}

/// Commit failure that preserves every live resource for retry or fail-closed cleanup.
#[must_use = "commit failure retains a live registration and worker retirement owner"]
pub(crate) struct CapacityWorkerCommitFailure {
    error: AdmissionError,
    worker: CapacityStartedWorker,
}

impl CapacityWorkerCommitFailure {
    pub(crate) const fn error(&self) -> AdmissionError {
        self.error
    }

    pub(crate) fn into_parts(self) -> (AdmissionError, CapacityStartedWorker) {
        (self.error, self.worker)
    }
}

impl std::fmt::Debug for CapacityWorkerCommitFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapacityWorkerCommitFailure")
            .field("error", &self.error)
            .field("worker_id", &self.worker.id)
            .finish_non_exhaustive()
    }
}

/// Shared admission authority for a future injected online context.
#[derive(Clone)]
pub(crate) struct InjectedContextAdmissionGate {
    inner: Arc<AdmissionInner>,
}

impl InjectedContextAdmissionGate {
    pub(crate) fn new() -> Self {
        let (drain_wake, drain_receiver) = crossbeam_channel::bounded(1);
        Self {
            inner: Arc::new(AdmissionInner {
                state: Mutex::new(AdmissionState::new(drain_receiver)),
                counters: AdmissionCounters::default(),
                drain_wake,
            }),
        }
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    #[cfg(test)]
    pub(crate) fn hold_phase_lock_for_test(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        let _state = self.inner.state.lock().unwrap();
        entered.send(()).unwrap();
        release.recv().unwrap();
    }

    #[cfg(test)]
    pub(crate) fn poison_phase_lock_for_test(&self) {
        let _state = self.inner.state.lock().unwrap();
        panic!("poison injected admission gate for test");
    }

    fn try_short_admission(&self, kind: AdmissionKind) -> Result<AdmissionPermit, AdmissionError> {
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(AdmissionError::Contended),
            Err(TryLockError::Poisoned(_)) => return Err(AdmissionError::Poisoned),
        };
        if state.phase == AdmissionPhase::Sealed {
            return Err(AdmissionError::Sealed);
        }
        self.inner.try_increment(kind)?;
        drop(state);
        Ok(AdmissionPermit {
            inner: Arc::clone(&self.inner),
            kind,
        })
    }

    pub(crate) fn try_external_event(&self) -> Result<ExternalEventAdmission, AdmissionError> {
        self.try_short_admission(AdmissionKind::ExternalEvent)
            .map(ExternalEventAdmission)
    }

    pub(crate) fn try_graph_control(&self) -> Result<GraphControlAdmission, AdmissionError> {
        self.try_short_admission(AdmissionKind::GraphControl)
            .map(GraphControlAdmission)
    }

    /// Reserves one capacity-worker slot.
    ///
    /// The returned registration privately owns the matching producer lease. Only `start` can move
    /// that lease into a concrete worker and create a commit-capable started-worker typestate. The
    /// registry is deliberately bounded to one worker, matching `AudioRenderCapacity::start`'s
    /// stop-before-restart semantics; a racing caller receives `CapacityWorkerActive` rather than
    /// waiting.
    pub(crate) fn try_begin_capacity_worker(
        &self,
    ) -> Result<CapacityWorkerRegistration, AdmissionError> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(AdmissionError::Contended),
            Err(TryLockError::Poisoned(_)) => return Err(AdmissionError::Poisoned),
        };
        if state.phase == AdmissionPhase::Sealed {
            return Err(AdmissionError::Sealed);
        }
        if state.capacity_worker.is_some()
            || self
                .inner
                .counters
                .capacity_registrations
                .load(Ordering::Acquire)
                != 0
            || self
                .inner
                .counters
                .capacity_producers
                .load(Ordering::Acquire)
                != 0
        {
            return Err(AdmissionError::CapacityWorkerActive);
        }
        let Some(id) = NonZeroU64::new(state.next_capacity_worker_id) else {
            return Err(AdmissionError::Exhausted);
        };
        self.inner
            .try_increment(AdmissionKind::CapacityRegistration)?;
        if let Err(error) = self.inner.try_increment(AdmissionKind::CapacityProducer) {
            self.inner.decrement(AdmissionKind::CapacityRegistration);
            return Err(error);
        }
        state.next_capacity_worker_id = id.get().checked_add(1).unwrap_or(0);
        drop(state);

        let id = CapacityWorkerId(id);
        Ok(CapacityWorkerRegistration {
            id,
            permit: AdmissionPermit {
                inner: Arc::clone(&self.inner),
                kind: AdmissionKind::CapacityRegistration,
            },
            producer: CapacityProducerLease(AdmissionPermit {
                inner: Arc::clone(&self.inner),
                kind: AdmissionKind::CapacityProducer,
            }),
        })
    }

    /// Removes a committed worker for an ordinary pre-seal `stop()` operation.
    ///
    /// Sealing and removal use the same phase/registry lock, so exactly one side receives the
    /// retirement owner. The caller must explicitly run or quarantine a returned owner off RT.
    pub(crate) fn try_take_capacity_worker(
        &self,
        id: CapacityWorkerId,
    ) -> Result<Option<CapacityWorkerRetirement>, AdmissionError> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(AdmissionError::Contended),
            Err(TryLockError::Poisoned(_)) => return Err(AdmissionError::Poisoned),
        };
        if state.phase == AdmissionPhase::Sealed {
            return Err(AdmissionError::Sealed);
        }
        if state
            .capacity_worker
            .as_ref()
            .map(CapacityWorkerRetirement::id)
            == Some(id)
        {
            Ok(state.capacity_worker.take())
        } else {
            Ok(None)
        }
    }

    /// Irreversibly seals all control-owned admissions and extracts committed worker owners.
    ///
    /// This method is non-waiting and must run off RT. The private lifecycle worker explicitly
    /// retire the extracted workers and then wait on the returned drain. The resulting drain report
    /// is deliberately not `EventProducersQuiesced`: render-owned events remain outside this gate,
    /// and queued event/control payloads remain owned by their existing channels/staging buffers.
    pub(crate) fn try_seal(&self) -> Result<AdmissionsSealed, AdmissionError> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(AdmissionError::Contended),
            Err(TryLockError::Poisoned(_)) => return Err(AdmissionError::Poisoned),
        };
        if state.phase == AdmissionPhase::Sealed {
            return Err(AdmissionError::Sealed);
        }
        state.phase = AdmissionPhase::Sealed;
        let capacity_worker = state.capacity_worker.take();
        let drain_receiver = state
            .drain_receiver
            .take()
            .expect("the first irreversible seal owns the single drain receiver");
        drop(state);

        Ok(AdmissionsSealed {
            capacity_worker,
            drain: AdmissionDrain {
                inner: Arc::clone(&self.inner),
                receiver: drain_receiver,
            },
        })
    }
}

/// Ownership transferred by the unique successful Open -> Sealed transition.
#[must_use = "sealed admissions retain worker retirement owners and the drain watcher"]
pub(crate) struct AdmissionsSealed {
    capacity_worker: Option<CapacityWorkerRetirement>,
    drain: AdmissionDrain,
}

impl AdmissionsSealed {
    pub(crate) fn into_parts(self) -> (Option<CapacityWorkerRetirement>, AdmissionDrain) {
        (self.capacity_worker, self.drain)
    }
}

/// Authoritative in-flight counts after admission sealing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionSnapshot {
    pub(crate) external_events: usize,
    pub(crate) graph_controls: usize,
    pub(crate) capacity_registrations: usize,
    pub(crate) capacity_producers: usize,
}

impl AdmissionSnapshot {
    pub(crate) const fn is_drained(self) -> bool {
        self.external_events == 0
            && self.graph_controls == 0
            && self.capacity_registrations == 0
            && self.capacity_producers == 0
    }
}

/// Wait authority transferred by sealing.
///
/// `wait` may block and is restricted to the private lifecycle worker. Completion only means all
/// gate-accounted control-owned operations/producers ended. It does not mean render producers are
/// quiescent, staged payloads were reclaimed, or the public event queue was drained.
#[must_use = "the lifecycle worker must observe the sealed admission drain"]
pub(crate) struct AdmissionDrain {
    inner: Arc<AdmissionInner>,
    receiver: crossbeam_channel::Receiver<()>,
}

impl AdmissionDrain {
    pub(crate) fn snapshot(&self) -> AdmissionSnapshot {
        self.inner.snapshot()
    }

    pub(crate) fn wait(self) -> AdmissionSnapshot {
        loop {
            let snapshot = self.snapshot();
            if snapshot.is_drained() {
                return snapshot;
            }
            // Structurally infallible while `self.inner` retains the sole wake sender.
            self.receiver
                .recv()
                .expect("admission drain sender lives as long as the drain");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    fn start_waiting_worker(registration: CapacityWorkerRegistration) -> CapacityStartedWorker {
        registration
            .start(|stop| {
                let _ = stop.recv();
            })
            .unwrap()
    }

    #[test]
    fn admission_types_are_send_sync_and_leases_are_distinct() {
        assert_send_sync::<InjectedContextAdmissionGate>();
        assert_send_sync::<ExternalEventAdmission>();
        assert_send_sync::<GraphControlAdmission>();
        assert_send_sync::<CapacityWorkerRegistration>();
        assert_send_sync::<CapacityStartedWorker>();
        assert_send_sync::<CapacityWorkerRetirement>();
        assert_send::<AdmissionsSealed>();
        assert_send_sync::<AdmissionDrain>();
    }

    #[test]
    fn surviving_clones_observe_one_irreversible_seal_and_raii_drain() {
        let gate = InjectedContextAdmissionGate::new();
        let survivor = gate.clone();
        let event = gate.try_external_event().unwrap();
        let control = survivor.try_graph_control().unwrap();

        let sealed = gate.try_seal().unwrap();
        assert!(matches!(
            survivor.try_external_event(),
            Err(AdmissionError::Sealed)
        ));
        assert!(matches!(
            survivor.try_graph_control(),
            Err(AdmissionError::Sealed)
        ));
        assert!(matches!(gate.try_seal(), Err(AdmissionError::Sealed)));

        let (worker, drain) = sealed.into_parts();
        assert!(worker.is_none());
        assert_eq!(
            drain.snapshot(),
            AdmissionSnapshot {
                external_events: 1,
                graph_controls: 1,
                capacity_registrations: 0,
                capacity_producers: 0,
            }
        );
        drop(event);
        drop(control);
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn contended_admission_and_seal_return_immediately_without_waiting() {
        let gate = InjectedContextAdmissionGate::new();
        let _phase_guard = gate.inner.state.lock().unwrap();
        let start = Instant::now();
        assert!(matches!(
            gate.try_external_event(),
            Err(AdmissionError::Contended)
        ));
        assert!(matches!(
            gate.try_graph_control(),
            Err(AdmissionError::Contended)
        ));
        assert!(matches!(
            gate.try_begin_capacity_worker(),
            Err(AdmissionError::Contended)
        ));
        assert!(matches!(gate.try_seal(), Err(AdmissionError::Contended)));
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn committed_capacity_worker_is_extracted_retired_and_drained_off_thread() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let id = registration.id();
        start_waiting_worker(registration).try_commit().unwrap();

        let sealed = gate.try_seal().unwrap();
        let (worker, drain) = sealed.into_parts();
        let retirement = worker.unwrap();
        assert_eq!(retirement.id(), id);
        assert_eq!(
            drain.snapshot(),
            AdmissionSnapshot {
                external_events: 0,
                graph_controls: 0,
                capacity_registrations: 0,
                capacity_producers: 1,
            }
        );
        let caller_thread = thread::current().id();
        let (retired_send, retired_recv) = mpsc::sync_channel(1);
        let retire_thread = thread::spawn(move || {
            let thread = thread::current().id();
            let result = retirement.stop_and_join();
            retired_send.send((thread, result)).unwrap();
        });
        retire_thread.join().unwrap();
        let (retired_thread, result) = retired_recv.recv().unwrap();
        assert_ne!(retired_thread, caller_thread);
        result.unwrap();
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn capacity_start_losing_seal_race_returns_every_cleanup_owner() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let started = start_waiting_worker(registration);

        let sealed = gate.try_seal().unwrap();
        let (worker, drain) = sealed.into_parts();
        assert!(worker.is_none());
        let failure = started.try_commit().unwrap_err();
        assert_eq!(failure.error(), AdmissionError::Sealed);
        let (error, started) = failure.into_parts();
        assert_eq!(error, AdmissionError::Sealed);

        started.stop_and_join().unwrap();
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn concurrent_capacity_commit_and_seal_transfer_exactly_one_retirement_owner() {
        enum CommitRace {
            Committed(CapacityWorkerId),
            Rejected(AdmissionError, CapacityStartedWorker),
        }

        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let expected_id = registration.id();
        let started = start_waiting_worker(registration);
        let race = Arc::new(Barrier::new(2));
        let commit_race = Arc::clone(&race);
        let commit = thread::spawn(move || {
            commit_race.wait();
            let mut started = started;
            loop {
                match started.try_commit() {
                    Ok(id) => break CommitRace::Committed(id),
                    Err(failure) => {
                        let (error, returned) = failure.into_parts();
                        if error == AdmissionError::Contended {
                            started = returned;
                            thread::yield_now();
                        } else {
                            break CommitRace::Rejected(error, returned);
                        }
                    }
                }
            }
        });

        race.wait();
        let sealed = loop {
            match gate.try_seal() {
                Ok(sealed) => break sealed,
                Err(AdmissionError::Contended) => thread::yield_now(),
                Err(error) => panic!("unexpected seal error: {error:?}"),
            }
        };
        let commit = commit.join().unwrap();
        let (sealed_worker, drain) = sealed.into_parts();

        match commit {
            CommitRace::Committed(id) => {
                assert_eq!(id, expected_id);
                let retirement = sealed_worker.expect("seal owns the committed worker");
                assert_eq!(retirement.id(), expected_id);
                retirement.stop_and_join().unwrap();
            }
            CommitRace::Rejected(error, started) => {
                assert_eq!(error, AdmissionError::Sealed);
                assert_eq!(started.id(), expected_id);
                assert!(sealed_worker.is_none());
                started.stop_and_join().unwrap();
            }
        }
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn ordinary_worker_take_and_seal_have_single_retirement_owner() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let id = registration.id();
        start_waiting_worker(registration).try_commit().unwrap();

        let retirement = gate.try_take_capacity_worker(id).unwrap().unwrap();
        retirement.stop_and_join().unwrap();
        let sealed = gate.try_seal().unwrap();
        let (worker, drain) = sealed.into_parts();
        assert!(worker.is_none());
        assert!(matches!(
            gate.try_take_capacity_worker(id),
            Err(AdmissionError::Sealed)
        ));
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn concurrent_capacity_take_and_seal_transfer_exactly_one_retirement_owner() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let id = registration.id();
        start_waiting_worker(registration).try_commit().unwrap();

        let race = Arc::new(Barrier::new(2));
        let take_gate = gate.clone();
        let take_race = Arc::clone(&race);
        let take = thread::spawn(move || {
            take_race.wait();
            loop {
                match take_gate.try_take_capacity_worker(id) {
                    Err(AdmissionError::Contended) => thread::yield_now(),
                    result => break result,
                }
            }
        });

        race.wait();
        let sealed = loop {
            match gate.try_seal() {
                Ok(sealed) => break sealed,
                Err(AdmissionError::Contended) => thread::yield_now(),
                Err(error) => panic!("unexpected seal error: {error:?}"),
            }
        };
        let take = take.join().unwrap();
        let (sealed_worker, drain) = sealed.into_parts();

        match take {
            Ok(Some(retirement)) => {
                assert_eq!(retirement.id(), id);
                assert!(sealed_worker.is_none());
                retirement.stop_and_join().unwrap();
            }
            Err(AdmissionError::Sealed) => {
                let retirement = sealed_worker.expect("seal owns the committed worker");
                assert_eq!(retirement.id(), id);
                retirement.stop_and_join().unwrap();
            }
            Ok(None) => panic!("the matching worker was lost by the take/seal race"),
            Err(error) => panic!("unexpected take error: {error:?}"),
        }
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn hostile_worker_keeps_its_bound_lease_until_join_really_completes() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let expected_id = registration.id();
        let (ready_send, ready_recv) = mpsc::sync_channel(1);
        let (release_send, release_recv) = mpsc::sync_channel(1);
        let started = registration
            .start(move |_stop| {
                ready_send.send(()).unwrap();
                // Deliberately ignore stop until the test releases this hostile worker. The
                // producer lease is still structurally held by the thread wrapper.
                release_recv.recv().unwrap();
            })
            .unwrap();
        assert_eq!(started.id(), expected_id);
        ready_recv.recv().unwrap();
        assert_eq!(started.try_commit().unwrap(), expected_id);

        // A different id cannot steal or retire this worker.
        let wrong_id = CapacityWorkerId(NonZeroU64::new(expected_id.get() + 1).unwrap());
        assert!(gate.try_take_capacity_worker(wrong_id).unwrap().is_none());

        let sealed = gate.try_seal().unwrap();
        let (worker, drain) = sealed.into_parts();
        let retirement = worker.unwrap();
        assert_eq!(drain.snapshot().capacity_producers, 1);
        let (joined_send, joined_recv) = mpsc::sync_channel(1);
        let joiner = thread::spawn(move || {
            joined_send.send(retirement.stop_and_join()).unwrap();
        });

        assert!(matches!(
            joined_recv.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert_eq!(drain.snapshot().capacity_producers, 1);
        release_send.send(()).unwrap();
        joined_recv.recv().unwrap().unwrap();
        joiner.join().unwrap();
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn capacity_worker_panic_is_surfaced_only_after_authoritative_join() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let started = registration
            .start(|_stop| std::panic::panic_any(73_u8))
            .unwrap();
        started.try_commit().unwrap();
        let sealed = gate.try_seal().unwrap();
        let (worker, drain) = sealed.into_parts();

        let error = worker.unwrap().stop_and_join().unwrap_err();
        let CapacityWorkerJoinError::Panicked(payload) = error;
        assert_eq!(payload.downcast_ref::<u8>(), Some(&73));
        drop(payload);
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn preseal_operations_finish_after_seal_but_no_late_operation_is_admitted() {
        let gate = InjectedContextAdmissionGate::new();
        let operation_started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let thread_gate = gate.clone();
        let thread_started = Arc::clone(&operation_started);
        let thread_release = Arc::clone(&release);
        let operation = thread::spawn(move || {
            let permit = thread_gate.try_graph_control().unwrap();
            thread_started.wait();
            thread_release.wait();
            drop(permit);
        });
        operation_started.wait();

        let sealed = gate.try_seal().unwrap();
        assert!(matches!(
            gate.try_graph_control(),
            Err(AdmissionError::Sealed)
        ));
        let (_, drain) = sealed.into_parts();
        assert_eq!(drain.snapshot().graph_controls, 1);
        release.wait();
        operation.join().unwrap();
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn abandoned_registration_drops_its_bound_unstarted_lease_by_raii() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        let sealed = gate.try_seal().unwrap();
        let (_, drain) = sealed.into_parts();
        assert_eq!(drain.snapshot().capacity_registrations, 1);
        assert_eq!(drain.snapshot().capacity_producers, 1);
        drop(registration);
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn capacity_worker_registry_is_single_slot_and_never_waits_for_restart() {
        let gate = InjectedContextAdmissionGate::new();
        let registration = gate.try_begin_capacity_worker().unwrap();
        assert!(matches!(
            gate.try_begin_capacity_worker(),
            Err(AdmissionError::CapacityWorkerActive)
        ));
        drop(registration);
        let next_registration = gate.try_begin_capacity_worker().unwrap();
        assert_eq!(next_registration.id().get(), 2);
        drop(next_registration);
    }

    #[test]
    fn counter_and_id_exhaustion_are_typed_and_do_not_partially_admit() {
        let gate = InjectedContextAdmissionGate::new();
        gate.inner
            .counters
            .external_events
            .store(usize::MAX, Ordering::Release);
        assert!(matches!(
            gate.try_external_event(),
            Err(AdmissionError::Exhausted)
        ));
        gate.inner
            .counters
            .external_events
            .store(0, Ordering::Release);
        gate.inner.state.lock().unwrap().next_capacity_worker_id = 0;
        assert!(matches!(
            gate.try_begin_capacity_worker(),
            Err(AdmissionError::Exhausted)
        ));
        assert!(gate.inner.snapshot().is_drained());
    }

    #[test]
    fn poisoned_phase_lock_fails_closed_without_partial_counts() {
        let gate = InjectedContextAdmissionGate::new();
        let poison = gate.clone();
        let poisoned = thread::spawn(move || {
            let _state = poison.inner.state.lock().unwrap();
            panic!("poison admission phase");
        });
        assert!(poisoned.join().is_err());
        assert!(matches!(
            gate.try_external_event(),
            Err(AdmissionError::Poisoned)
        ));
        assert!(matches!(
            gate.try_graph_control(),
            Err(AdmissionError::Poisoned)
        ));
        assert!(matches!(
            gate.try_begin_capacity_worker(),
            Err(AdmissionError::Poisoned)
        ));
        assert!(matches!(gate.try_seal(), Err(AdmissionError::Poisoned)));
        assert!(gate.inner.snapshot().is_drained());
    }

    #[test]
    fn many_concurrent_short_permits_are_exactly_drained() {
        let gate = InjectedContextAdmissionGate::new();
        let start = Arc::new(Barrier::new(17));
        let release = Arc::new(Barrier::new(17));
        let active = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for index in 0..16 {
            let gate = gate.clone();
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let active = Arc::clone(&active);
            threads.push(thread::spawn(move || {
                let permit: Box<dyn Send> = loop {
                    let attempt: Result<Box<dyn Send>, AdmissionError> = if index % 2 == 0 {
                        gate.try_external_event()
                            .map(|permit| Box::new(permit) as Box<dyn Send>)
                    } else {
                        gate.try_graph_control()
                            .map(|permit| Box::new(permit) as Box<dyn Send>)
                    };
                    match attempt {
                        Ok(permit) => break permit,
                        Err(AdmissionError::Contended) => thread::yield_now(),
                        Err(error) => panic!("unexpected concurrent admission error: {error:?}"),
                    }
                };
                active.fetch_add(1, Ordering::AcqRel);
                start.wait();
                release.wait();
                drop(permit);
            }));
        }
        start.wait();
        assert_eq!(active.load(Ordering::Acquire), 16);
        let sealed = gate.try_seal().unwrap();
        let (_, drain) = sealed.into_parts();
        assert_eq!(drain.snapshot().external_events, 8);
        assert_eq!(drain.snapshot().graph_controls, 8);
        release.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn raw_render_production_is_not_misrepresented_by_the_drain() {
        // A stand-in proves the contract boundary: arbitrary render-owned work can remain live
        // while every gate-accounted control producer drains. The later lifecycle composes this
        // report with renderer reclamation before constructing EventProducersQuiesced.
        let raw_render_producer_alive = Arc::new(AtomicBool::new(true));
        let gate = InjectedContextAdmissionGate::new();
        let sealed = gate.try_seal().unwrap();
        let (_, drain) = sealed.into_parts();
        assert!(drain.wait().is_drained());
        assert!(raw_render_producer_alive.load(Ordering::Acquire));
    }
}
