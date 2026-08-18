use crossbeam_channel::{Receiver, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::context::{
    AdmissionError, BaseAudioContext, CapacityWorkerId, CapacityWorkerJoinError,
    ConcreteBaseAudioContext, ControlEventDispatch, ControlEventSendOutcome,
    InjectedContextAdmissionGate,
};
use crate::events::{EventDispatch, EventHandler, EventPayload, EventType};
use crate::stats::{AudioStats, AudioStatsSnapshot};
use crate::Event;

/// Options for constructing an `AudioRenderCapacity`
#[derive(Clone, Debug)]
pub struct AudioRenderCapacityOptions {
    /// An update interval (in seconds) for dispatching [`AudioRenderCapacityEvent`]s
    pub update_interval: f64,
}

impl Default for AudioRenderCapacityOptions {
    fn default() -> Self {
        Self {
            update_interval: 1.,
        }
    }
}

/// Performance metrics of the rendering thread
#[derive(Clone, Debug)]
pub struct AudioRenderCapacityEvent {
    /// The start time of the data collection period in terms of the associated AudioContext's currentTime
    pub timestamp: f64,
    /// An average of collected load values over the given update interval
    pub average_load: f64,
    /// A maximum value from collected load values over the given update interval.
    pub peak_load: f64,
    /// A ratio between the number of buffer underruns and the total number of system-level audio callbacks over the given update interval.
    pub underrun_ratio: f64,
    /// Inherits from this base Event
    pub event: Event,
}

impl AudioRenderCapacityEvent {
    fn new(timestamp: f64, average_load: f64, peak_load: f64, underrun_ratio: f64) -> Self {
        // We are limiting the precision here conform
        // https://webaudio.github.io/web-audio-api/#dom-audiorendercapacityevent-averageload
        Self {
            timestamp,
            average_load: (average_load * 100.).round() / 100.,
            peak_load: (peak_load * 100.).round() / 100.,
            underrun_ratio: (underrun_ratio * 100.).ceil() / 100.,
            event: Event {
                type_: "AudioRenderCapacityEvent",
            },
        }
    }
}

/// Provider for rendering performance metrics
///
/// A load value is computed for each system-level audio callback, by dividing its execution
/// duration by the system-level audio callback buffer size divided by the sample rate.
///
/// Ideally the load value is below 1.0, meaning that it took less time to render the audio than it
/// took to play it out. An audio buffer underrun happens when this load value is greater than 1.0: the
/// system could not render audio fast enough for real-time.
#[derive(Clone)]
pub struct AudioRenderCapacity {
    context: ConcreteBaseAudioContext,
    service: Arc<AudioRenderCapacityService>,
}

impl std::fmt::Debug for AudioRenderCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioRenderCapacity")
            .field(
                "context",
                &format!("BaseAudioContext@{}", self.context.address()),
            )
            .finish_non_exhaustive()
    }
}

impl AudioRenderCapacity {
    pub(crate) fn new(context: ConcreteBaseAudioContext, stats: AudioStats) -> Self {
        let runner = MetricsCapacityWorker {
            stats,
            sample_rate: context.sample_rate(),
            frames_played: context.frames_played_counter(),
            event_dispatch: context.control_event_dispatch(),
            #[cfg(test)]
            send_observer: None,
            #[cfg(test)]
            ready_observer: None,
        };

        let service = match runner.event_dispatch.injected_gate() {
            Some(gate) => AudioRenderCapacityService::new_injected(Arc::new(runner), gate),
            None => AudioRenderCapacityService::new(Arc::new(runner)),
        };
        Self {
            context,
            service: Arc::new(service),
        }
    }

    /// Start metric collection and analysis
    #[allow(clippy::missing_panics_doc)]
    pub fn start(&self, options: AudioRenderCapacityOptions) {
        self.service.start(options.update_interval);
    }

    /// Stop metric collection and analysis
    #[allow(clippy::missing_panics_doc)]
    pub fn stop(&self) {
        self.service.stop();
    }

    /// Permanently stops capacity production for a closing context.
    pub(crate) fn close(&self) {
        self.service.close();
    }

    /// The EventHandler for [`AudioRenderCapacityEvent`].
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    pub fn set_onupdate<F: FnMut(AudioRenderCapacityEvent) + Send + 'static>(
        &self,
        mut callback: F,
    ) {
        let callback = move |v| match v {
            EventPayload::RenderCapacity(v) => callback(v),
            _ => unreachable!(),
        };

        self.context.set_event_handler(
            EventType::RenderCapacity,
            EventHandler::Multiple(Box::new(callback)),
        );
    }

    /// Unset the EventHandler for [`AudioRenderCapacityEvent`].
    pub fn clear_onupdate(&self) {
        self.context.clear_event_handler(EventType::RenderCapacity);
    }
}

trait CapacityWorkerRunner: Send + Sync + 'static {
    /// Capture the timestamp, stats snapshot, and peak-reset baseline synchronously with `start`.
    fn prepare(&self) -> Box<dyn PreparedCapacityWorker>;
}

trait PreparedCapacityWorker: Send + 'static {
    fn run(self: Box<Self>, stop: Receiver<()>, update_interval: Duration);
}

struct MetricsCapacityWorker {
    stats: AudioStats,
    sample_rate: f32,
    frames_played: Arc<AtomicU64>,
    event_dispatch: ControlEventDispatch,
    #[cfg(test)]
    send_observer: Option<Sender<CapacityEventSendOutcome>>,
    #[cfg(test)]
    ready_observer: Option<Sender<()>>,
}

impl MetricsCapacityWorker {
    #[allow(clippy::cast_precision_loss)]
    fn current_time(&self) -> f64 {
        self.frames_played.load(Ordering::SeqCst) as f64 / f64::from(self.sample_rate)
    }
}

impl CapacityWorkerRunner for MetricsCapacityWorker {
    fn prepare(&self) -> Box<dyn PreparedCapacityWorker> {
        let timestamp = self.current_time();
        let previous = self.stats.snapshot();
        self.stats.take_peak_load();
        #[cfg(test)]
        if let Some(observer) = &self.ready_observer {
            let _ = observer.send(());
        }

        Box::new(PreparedMetricsCapacityWorker {
            stats: self.stats.clone(),
            sample_rate: self.sample_rate,
            frames_played: Arc::clone(&self.frames_played),
            event_dispatch: self.event_dispatch.clone(),
            timestamp,
            previous,
            #[cfg(test)]
            send_observer: self.send_observer.clone(),
        })
    }
}

struct PreparedMetricsCapacityWorker {
    stats: AudioStats,
    sample_rate: f32,
    frames_played: Arc<AtomicU64>,
    event_dispatch: ControlEventDispatch,
    timestamp: f64,
    previous: AudioStatsSnapshot,
    #[cfg(test)]
    send_observer: Option<Sender<CapacityEventSendOutcome>>,
}

impl PreparedMetricsCapacityWorker {
    #[allow(clippy::cast_precision_loss)]
    fn current_time(&self) -> f64 {
        self.frames_played.load(Ordering::SeqCst) as f64 / f64::from(self.sample_rate)
    }

    #[cfg(test)]
    fn observe_send(&self, outcome: CapacityEventSendOutcome) {
        if let Some(observer) = &self.send_observer {
            let _ = observer.send(outcome);
        }
    }
}

impl PreparedCapacityWorker for PreparedMetricsCapacityWorker {
    fn run(mut self: Box<Self>, stop: Receiver<()>, update_interval: Duration) {
        loop {
            match stop.recv_timeout(update_interval) {
                Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            }

            let next = self.stats.snapshot();
            if next.callback_count == self.previous.callback_count {
                continue;
            }

            let peak_load = self.stats.take_peak_load();
            let timestamp = self.timestamp;
            let previous = self.previous;
            let outcome = match self.event_dispatch.try_send_with(move || {
                EventDispatch::render_capacity(render_capacity_event(
                    timestamp, previous, next, peak_load,
                ))
            }) {
                ControlEventSendOutcome::Delivered => CapacityEventSendOutcome::Delivered,
                ControlEventSendOutcome::Full => CapacityEventSendOutcome::Full,
                ControlEventSendOutcome::Disconnected => CapacityEventSendOutcome::Disconnected,
                ControlEventSendOutcome::AdmissionRejected(AdmissionError::Contended) => {
                    CapacityEventSendOutcome::Dropped
                }
                ControlEventSendOutcome::AdmissionRejected(
                    AdmissionError::Sealed
                    | AdmissionError::Poisoned
                    | AdmissionError::Exhausted
                    | AdmissionError::CapacityWorkerActive,
                ) => CapacityEventSendOutcome::Disconnected,
            };
            if outcome == CapacityEventSendOutcome::Disconnected {
                #[cfg(test)]
                self.observe_send(outcome);
                return;
            }

            // A full queue drops exactly this diagnostic interval. Advancing both baselines keeps
            // later reports bounded to their own interval instead of folding the dropped data into
            // an eventual delivery.
            self.previous = next;
            self.timestamp = self.current_time();
            #[cfg(test)]
            self.observe_send(outcome);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CapacityEventSendOutcome {
    Delivered,
    Full,
    /// A diagnostic tick lost non-waiting admission contention. The interval is intentionally
    /// dropped just like a full queue, and the worker remains live.
    Dropped,
    Disconnected,
}

struct CapacityWorker {
    stop: Sender<()>,
    join: JoinHandle<()>,
}

impl CapacityWorker {
    fn spawn(
        prepared: Box<dyn PreparedCapacityWorker>,
        update_interval: Duration,
    ) -> std::io::Result<Self> {
        let (stop, stop_recv) = crossbeam_channel::bounded(1);
        let join = std::thread::Builder::new()
            .name("web-audio-capacity".to_owned())
            .spawn(move || prepared.run(stop_recv, update_interval))?;
        Ok(Self { stop, join })
    }

    /// Returns true if the joined worker panicked. Any panic payload is already quarantined.
    fn stop_and_join(self) -> bool {
        // Capacity one and non-waiting: Full means stop was already requested. The worker checks
        // this channel while waiting for the interval, so even very large intervals stop promptly.
        let _ = self.stop.try_send(());
        if let Err(payload) = self.join.join() {
            // A panic payload is arbitrary user-controlled `Any`; even dropping it may panic.
            // Capacity has no public error channel, so deliberately quarantine it before any
            // diagnostic work can run.
            std::mem::forget(payload);
            true
        } else {
            false
        }
    }
}

#[derive(Default)]
struct CapacityServiceState {
    transition_in_progress: bool,
    closed: bool,
    worker: Option<CapacityWorkerHandle>,
}

#[derive(Clone)]
enum CapacityServiceMode {
    Legacy,
    /// The gate, rather than this service, owns every committed `JoinHandle`. The service retains
    /// only the exact id needed to atomically take that owner for ordinary stop/restart.
    Injected(InjectedContextAdmissionGate),
}

enum CapacityWorkerHandle {
    Legacy(CapacityWorker),
    Injected(CapacityWorkerId),
}

struct AudioRenderCapacityService {
    state: Mutex<CapacityServiceState>,
    transition_done: Condvar,
    runner: Arc<dyn CapacityWorkerRunner>,
    mode: CapacityServiceMode,
}

enum CapacityTransition {
    Closed,
    Active(Option<CapacityWorkerHandle>),
}

#[derive(Default)]
struct CapacityRetirement {
    panicked: bool,
    /// A seal, poison, or registry mismatch owns/quarantines the retirement authority. The
    /// service must permanently refuse restart rather than guess that no worker remains.
    terminal: bool,
}

enum CapacityStartFailure {
    Spawn(std::io::Error),
    Admission(AdmissionError),
}

struct CapacityStartResult {
    worker: Option<CapacityWorkerHandle>,
    retirement_panicked: bool,
    terminal: bool,
    failure: Option<CapacityStartFailure>,
}

impl AudioRenderCapacityService {
    fn new(runner: Arc<dyn CapacityWorkerRunner>) -> Self {
        Self {
            state: Mutex::new(CapacityServiceState::default()),
            transition_done: Condvar::new(),
            runner,
            mode: CapacityServiceMode::Legacy,
        }
    }

    fn new_injected(
        runner: Arc<dyn CapacityWorkerRunner>,
        gate: InjectedContextAdmissionGate,
    ) -> Self {
        Self {
            state: Mutex::new(CapacityServiceState::default()),
            transition_done: Condvar::new(),
            runner,
            mode: CapacityServiceMode::Injected(gate),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, CapacityServiceState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn begin_transition(&self, close: bool) -> CapacityTransition {
        let mut state = self.lock_state();
        while state.transition_in_progress {
            state = self
                .transition_done
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        if state.closed {
            return CapacityTransition::Closed;
        }

        state.transition_in_progress = true;
        state.closed = close;
        CapacityTransition::Active(state.worker.take())
    }

    fn finish_transition(&self, worker: Option<CapacityWorkerHandle>, force_closed: bool) {
        let mut state = self.lock_state();
        debug_assert!(state.transition_in_progress);
        debug_assert!(state.worker.is_none());
        state.worker = worker;
        state.closed |= force_closed;
        state.transition_in_progress = false;
        drop(state);
        self.transition_done.notify_all();
    }

    fn quarantine_join_error(error: CapacityWorkerJoinError) -> bool {
        let CapacityWorkerJoinError::Panicked(payload) = error;
        // Panic payloads are arbitrary. Forget before logging or touching shared lifecycle state,
        // since even destructing a hostile payload may panic.
        std::mem::forget(payload);
        true
    }

    fn retire(
        mode: &CapacityServiceMode,
        worker: Option<CapacityWorkerHandle>,
    ) -> CapacityRetirement {
        let Some(worker) = worker else {
            return CapacityRetirement::default();
        };
        match (mode, worker) {
            (CapacityServiceMode::Legacy, CapacityWorkerHandle::Legacy(worker)) => {
                CapacityRetirement {
                    panicked: worker.stop_and_join(),
                    terminal: false,
                }
            }
            (CapacityServiceMode::Injected(gate), CapacityWorkerHandle::Injected(id)) => loop {
                match gate.try_take_capacity_worker(id) {
                    Ok(Some(worker)) => {
                        break CapacityRetirement {
                            panicked: worker
                                .stop_and_join()
                                .err()
                                .is_some_and(Self::quarantine_join_error),
                            terminal: false,
                        };
                    }
                    Ok(None) => {
                        // An id mismatch can only be caused by private misuse. No owner is dropped
                        // here; fail closed because another authority may still own a live worker.
                        break CapacityRetirement {
                            panicked: false,
                            terminal: true,
                        };
                    }
                    Err(AdmissionError::Contended) => std::thread::yield_now(),
                    Err(
                        AdmissionError::Sealed
                        | AdmissionError::Poisoned
                        | AdmissionError::Exhausted
                        | AdmissionError::CapacityWorkerActive,
                    ) => {
                        // Sealing extracted the owner; poison retains it in the inaccessible gate
                        // registry. In the poisoned case `stop()` cannot truthfully acknowledge
                        // signaling or joining the worker: its owner stays quarantined for the
                        // process lifetime. In either case this service must never start a
                        // replacement.
                        break CapacityRetirement {
                            panicked: false,
                            terminal: true,
                        };
                    }
                }
            },
            _ => unreachable!("capacity worker ownership mode is immutable"),
        }
    }

    fn start_worker(&self, update_interval: Duration) -> CapacityStartResult {
        match &self.mode {
            CapacityServiceMode::Legacy => {
                let prepared = self.runner.prepare();
                match CapacityWorker::spawn(prepared, update_interval) {
                    Ok(worker) => CapacityStartResult {
                        worker: Some(CapacityWorkerHandle::Legacy(worker)),
                        retirement_panicked: false,
                        terminal: false,
                        failure: None,
                    },
                    Err(error) => CapacityStartResult {
                        worker: None,
                        retirement_panicked: false,
                        terminal: false,
                        failure: Some(CapacityStartFailure::Spawn(error)),
                    },
                }
            }
            CapacityServiceMode::Injected(gate) => {
                let registration = loop {
                    match gate.try_begin_capacity_worker() {
                        Ok(registration) => break registration,
                        Err(AdmissionError::Contended) => std::thread::yield_now(),
                        Err(error) => {
                            return CapacityStartResult {
                                worker: None,
                                retirement_panicked: false,
                                terminal: true,
                                failure: Some(CapacityStartFailure::Admission(error)),
                            };
                        }
                    }
                };

                // The bound registration/producer lease is acquired before baseline preparation.
                // A concurrent seal therefore observes and drains this whole start attempt.
                let prepared = self.runner.prepare();
                let started =
                    match registration.start(move |stop| prepared.run(stop, update_interval)) {
                        Ok(started) => started,
                        Err(error) => {
                            return CapacityStartResult {
                                worker: None,
                                retirement_panicked: false,
                                terminal: false,
                                failure: Some(CapacityStartFailure::Spawn(error)),
                            };
                        }
                    };
                let mut started = started;
                loop {
                    match started.try_commit() {
                        Ok(id) => {
                            return CapacityStartResult {
                                worker: Some(CapacityWorkerHandle::Injected(id)),
                                retirement_panicked: false,
                                terminal: false,
                                failure: None,
                            };
                        }
                        Err(failure) => {
                            let (error, returned) = failure.into_parts();
                            if error == AdmissionError::Contended {
                                started = returned;
                                std::thread::yield_now();
                                continue;
                            }
                            let retirement_panicked = returned
                                .stop_and_join()
                                .err()
                                .is_some_and(Self::quarantine_join_error);
                            return CapacityStartResult {
                                worker: None,
                                retirement_panicked,
                                terminal: true,
                                failure: Some(CapacityStartFailure::Admission(error)),
                            };
                        }
                    }
                }
            }
        }
    }

    fn report_retirement_panic(panicked: bool) {
        if panicked {
            // The transition is already coherent and the hostile payload has already been
            // forgotten, so even an adversarial logger cannot strand the service mutex/state.
            log::error!("AudioRenderCapacity worker panicked");
        }
    }

    fn report_start_failure(failure: Option<CapacityStartFailure>) {
        match failure {
            Some(CapacityStartFailure::Spawn(error)) => {
                log::error!("Failed to start AudioRenderCapacity worker: {error}");
            }
            Some(CapacityStartFailure::Admission(error)) => {
                log::error!("AudioRenderCapacity admission failed closed: {error:?}");
            }
            None => {}
        }
    }

    fn start(&self, update_interval_seconds: f64) {
        let CapacityTransition::Active(previous) = self.begin_transition(false) else {
            return;
        };

        // No service-state lock is held while waiting for the predecessor or spawning its
        // replacement. Other lifecycle calls wait on the condition variable and cannot orphan a
        // handle or overlap a worker.
        let predecessor = Self::retire(&self.mode, previous);
        if predecessor.terminal {
            self.finish_transition(None, true);
            Self::report_retirement_panic(predecessor.panicked);
            return;
        }

        // Preserve legacy ordering: start first retires its predecessor, then applies the 1 ms
        // floor and converts the requested interval. Positive infinity therefore still stops the
        // old worker before panicking. Catch only long enough to restore coherent service state.
        let update_interval = std::panic::catch_unwind(|| {
            Duration::from_secs_f64(update_interval_seconds.max(0.001))
        });
        let update_interval = match update_interval {
            Ok(interval) => interval,
            Err(payload) => {
                self.finish_transition(None, false);
                Self::report_retirement_panic(predecessor.panicked);
                std::panic::resume_unwind(payload);
            }
        };

        let started = self.start_worker(update_interval);
        self.finish_transition(started.worker, started.terminal);
        Self::report_retirement_panic(predecessor.panicked || started.retirement_panicked);
        Self::report_start_failure(started.failure);
    }

    fn stop(&self) {
        let CapacityTransition::Active(worker) = self.begin_transition(false) else {
            return;
        };
        let retired = Self::retire(&self.mode, worker);
        self.finish_transition(None, retired.terminal);
        Self::report_retirement_panic(retired.panicked);
    }

    fn close(&self) {
        let CapacityTransition::Active(worker) = self.begin_transition(true) else {
            return;
        };
        let retired = Self::retire(&self.mode, worker);
        self.finish_transition(None, retired.terminal);
        Self::report_retirement_panic(retired.panicked);
    }
}

impl Drop for AudioRenderCapacityService {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        debug_assert!(!state.transition_in_progress);
        state.closed = true;
        let worker = state.worker.take();
        let retired = Self::retire(&self.mode, worker);
        Self::report_retirement_panic(retired.panicked);
    }
}

fn render_capacity_event(
    timestamp: f64,
    previous: AudioStatsSnapshot,
    next: AudioStatsSnapshot,
    peak_load: f64,
) -> AudioRenderCapacityEvent {
    let callback_count = next
        .callback_count
        .saturating_sub(previous.callback_count)
        .max(1);
    let render_duration = next
        .render_duration_ns_total
        .saturating_sub(previous.render_duration_ns_total);
    let callback_budget = next
        .callback_budget_ns_total
        .saturating_sub(previous.callback_budget_ns_total);
    let underruns = next.underrun_count.saturating_sub(previous.underrun_count);

    let average_load = if callback_budget == 0 {
        0.
    } else {
        render_duration as f64 / callback_budget as f64
    };

    AudioRenderCapacityEvent::new(
        timestamp,
        average_load,
        peak_load,
        underruns as f64 / callback_count as f64,
    )
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{mpsc, Barrier};
    use std::thread;
    use std::time::Instant;

    use super::*;
    use crate::context::{AudioContext, AudioContextOptions};
    use crate::events::EventLoop;
    use crate::io::{self, ControlThreadInit, RenderThreadInit};

    const TEST_TIMEOUT: Duration = Duration::from_secs(2);

    #[derive(Default)]
    struct WorkerProbe {
        starts: AtomicUsize,
        exits: AtomicUsize,
        live: AtomicUsize,
        max_live: AtomicUsize,
        order: Mutex<Vec<(&'static str, usize)>>,
    }

    impl WorkerProbe {
        fn wait_for(&self, predicate: impl Fn(&Self) -> bool) {
            let deadline = Instant::now() + TEST_TIMEOUT;
            while !predicate(self) {
                assert!(Instant::now() < deadline, "capacity worker probe timed out");
                thread::yield_now();
            }
        }
    }

    struct CountingRunner(Arc<WorkerProbe>);

    struct PreparedCountingRunner(Arc<WorkerProbe>);

    impl CapacityWorkerRunner for CountingRunner {
        fn prepare(&self) -> Box<dyn PreparedCapacityWorker> {
            Box::new(PreparedCountingRunner(Arc::clone(&self.0)))
        }
    }

    impl PreparedCapacityWorker for PreparedCountingRunner {
        fn run(self: Box<Self>, stop: Receiver<()>, _update_interval: Duration) {
            let generation = self.0.starts.fetch_add(1, Ordering::AcqRel) + 1;
            let live = self.0.live.fetch_add(1, Ordering::AcqRel) + 1;
            self.0.max_live.fetch_max(live, Ordering::AcqRel);
            self.0.order.lock().unwrap().push(("start", generation));

            struct ExitGuard<'a> {
                probe: &'a WorkerProbe,
                generation: usize,
            }
            impl Drop for ExitGuard<'_> {
                fn drop(&mut self) {
                    self.probe
                        .order
                        .lock()
                        .unwrap()
                        .push(("exit", self.generation));
                    self.probe.live.fetch_sub(1, Ordering::AcqRel);
                    self.probe.exits.fetch_add(1, Ordering::AcqRel);
                }
            }
            let _exit = ExitGuard {
                probe: &self.0,
                generation,
            };
            let _ = stop.recv();
        }
    }

    fn counting_service() -> (Arc<AudioRenderCapacityService>, Arc<WorkerProbe>) {
        let probe = Arc::new(WorkerProbe::default());
        let runner: Arc<dyn CapacityWorkerRunner> = Arc::new(CountingRunner(Arc::clone(&probe)));
        (Arc::new(AudioRenderCapacityService::new(runner)), probe)
    }

    fn counting_service_injected() -> (
        Arc<AudioRenderCapacityService>,
        Arc<WorkerProbe>,
        InjectedContextAdmissionGate,
    ) {
        let probe = Arc::new(WorkerProbe::default());
        let runner: Arc<dyn CapacityWorkerRunner> = Arc::new(CountingRunner(Arc::clone(&probe)));
        let gate = InjectedContextAdmissionGate::new();
        (
            Arc::new(AudioRenderCapacityService::new_injected(
                runner,
                gate.clone(),
            )),
            probe,
            gate,
        )
    }

    fn seal_gate(gate: &InjectedContextAdmissionGate) -> crate::context::AdmissionSnapshot {
        let sealed = loop {
            match gate.try_seal() {
                Ok(sealed) => break sealed,
                Err(AdmissionError::Contended) => thread::yield_now(),
                Err(error) => panic!("unexpected admission seal failure: {error:?}"),
            }
        };
        let (worker, drain) = sealed.into_parts();
        if let Some(worker) = worker {
            if let Err(CapacityWorkerJoinError::Panicked(payload)) = worker.stop_and_join() {
                std::mem::forget(payload);
                panic!("injected capacity worker panicked while sealing");
            }
        }
        drain.wait()
    }

    fn injected_base_harness(
        gate: InjectedContextAdmissionGate,
    ) -> (
        ConcreteBaseAudioContext,
        AudioStats,
        EventLoop,
        RenderThreadInit,
    ) {
        let (control, render) = io::thread_init();
        let ControlThreadInit {
            state,
            frames_played,
            stats,
            ctrl_msg_send,
            control_batch_send,
            control_batch_applied,
            event_send,
            event_recv,
        } = control;
        let event_loop = EventLoop::new(event_recv);
        let (_node_id_producer, node_id_consumer) = llq::Queue::new().split();
        let base = ConcreteBaseAudioContext::new_injected(
            48_000.,
            2,
            state,
            frames_played,
            ctrl_msg_send,
            control_batch_send,
            control_batch_applied,
            event_send,
            event_loop.clone(),
            false,
            node_id_consumer,
            gate,
        );
        (base, stats, event_loop, render)
    }

    #[test]
    fn test_same_instance() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);

        let rc1 = context.render_capacity();
        let rc2 = context.render_capacity();
        let rc3 = rc2.clone();

        // assert all items are actually the same instance
        assert!(Arc::ptr_eq(&rc1.service, &rc2.service));
        assert!(Arc::ptr_eq(&rc1.service, &rc3.service));
    }

    #[test]
    fn injected_base_and_capacity_clones_share_seal_while_render_sender_stays_ungated() {
        let gate = InjectedContextAdmissionGate::new();
        let (base, stats, event_loop, render_init) = injected_base_harness(gate.clone());
        let surviving_base = base.clone();
        let capacity = AudioRenderCapacity::new(base, stats);
        let surviving_capacity = capacity.clone();

        capacity.start(AudioRenderCapacityOptions {
            update_interval: 60.,
        });
        capacity.stop();
        capacity.start(AudioRenderCapacityOptions {
            update_interval: 60.,
        });
        capacity.start(AudioRenderCapacityOptions {
            update_interval: 60.,
        });
        assert!(matches!(
            gate.try_begin_capacity_worker(),
            Err(AdmissionError::CapacityWorkerActive)
        ));

        let sealed = gate.try_seal().unwrap();
        let (worker, drain) = sealed.into_parts();
        let worker = worker.unwrap();
        assert_eq!(worker.id().get(), 3);
        worker.stop_and_join().unwrap();
        assert!(drain.wait().is_drained());

        let factory_called = AtomicBool::new(false);
        assert_eq!(
            surviving_base.send_event_with(|| {
                factory_called.store(true, Ordering::Release);
                EventDispatch::sink_change()
            }),
            Err(ControlEventSendOutcome::AdmissionRejected(
                AdmissionError::Sealed
            ))
        );
        assert!(!factory_called.load(Ordering::Acquire));

        // The service holds only the stale committed id. It cannot recover the sealed worker or
        // install another one, even through a public clone retained past context sealing.
        surviving_capacity.start(AudioRenderCapacityOptions {
            update_interval: 0.001,
        });
        assert!(surviving_capacity.service.lock_state().closed);
        surviving_capacity.stop();

        // Render-owned production is intentionally a separate raw capability. Admission drain is
        // not, and must not later be promoted directly into, EventProducersQuiesced.
        let (handled_send, handled_recv) = mpsc::sync_channel(1);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| handled_send.send(()).unwrap())),
        );
        render_init
            .event_send
            .try_send(EventDispatch::sink_change())
            .unwrap();
        assert!(event_loop.handle_pending_events());
        handled_recv.recv_timeout(TEST_TIMEOUT).unwrap();
    }

    #[test]
    fn injected_base_full_send_holds_drain_through_rejected_payload_drop() {
        struct BlockingDropControl {
            entered: Sender<()>,
            release: Receiver<()>,
        }
        struct BlockingDrop(&'static BlockingDropControl);
        impl Drop for BlockingDrop {
            fn drop(&mut self) {
                self.0.entered.send(()).unwrap();
                self.0.release.recv().unwrap();
            }
        }

        let gate = InjectedContextAdmissionGate::new();
        let (base, _stats, _event_loop, render_init) = injected_base_harness(gate.clone());
        for _ in 0..256 {
            render_init
                .event_send
                .try_send(EventDispatch::sink_change())
                .unwrap();
        }
        let (drop_entered_send, drop_entered_recv) = crossbeam_channel::bounded(1);
        let (drop_release_send, drop_release_recv) = crossbeam_channel::bounded(1);
        let drop_control: &'static BlockingDropControl = Box::leak(Box::new(BlockingDropControl {
            entered: drop_entered_send,
            release: drop_release_recv,
        }));
        let sender = thread::spawn(move || {
            base.send_event_with(|| {
                EventDispatch::message(
                    crate::context::AudioNodeId(92),
                    Box::new(BlockingDrop(drop_control)),
                )
            })
        });
        drop_entered_recv.recv_timeout(TEST_TIMEOUT).unwrap();

        let sealed = gate.try_seal().unwrap();
        let (_, drain) = sealed.into_parts();
        assert_eq!(drain.snapshot().external_events, 1);
        drop_release_send.send(()).unwrap();
        assert_eq!(sender.join().unwrap(), Err(ControlEventSendOutcome::Full));
        assert!(drain.wait().is_drained());
    }

    #[test]
    fn test_stop_when_not_running() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);

        let rc = context.render_capacity();
        rc.stop();
    }

    #[test]
    fn test_render_capacity() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);

        let rc = context.render_capacity();
        let (send, recv) = crossbeam_channel::bounded(1);
        rc.set_onupdate(move |e| send.send(e).unwrap());
        rc.start(AudioRenderCapacityOptions {
            update_interval: 0.05,
        });
        let event = recv.recv().unwrap();

        assert!(event.timestamp >= 0.);
        assert!(event.average_load >= 0.);
        assert!(event.peak_load >= 0.);
        assert!(event.underrun_ratio >= 0.);

        assert!(event.timestamp.is_finite());
        assert!(event.average_load.is_finite());
        assert!(event.peak_load.is_finite());
        assert!(event.underrun_ratio.is_finite());

        assert_eq!(event.event.type_, "AudioRenderCapacityEvent");
    }

    #[test]
    fn test_render_capacity_stops_on_close() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);

        let rc = context.render_capacity();
        let (send, recv) = crossbeam_channel::unbounded();
        rc.set_onupdate(move |e| send.send(e).unwrap());
        rc.start(AudioRenderCapacityOptions {
            update_interval: 0.01,
        });

        recv.recv().unwrap();
        while recv.try_recv().is_ok() {}

        context.close_sync();
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert_eq!(recv.try_iter().count(), 0);

        // `rc` is a surviving public clone, but permanent context close seals only this shared
        // capacity service against restart.
        rc.start(AudioRenderCapacityOptions {
            update_interval: 0.001,
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(recv.try_iter().count(), 0);
    }

    #[test]
    fn idle_stop_is_a_noop_and_close_prevents_restart() {
        let (service, probe) = counting_service();
        let stop = Instant::now();
        service.stop();
        assert!(stop.elapsed() < Duration::from_secs(1));
        assert_eq!(probe.starts.load(Ordering::Acquire), 0);

        service.start(0.001);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);
        service.close();
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);

        service.start(0.001);
        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
        service.stop();
    }

    #[test]
    fn restart_joins_predecessor_before_starting_successor() {
        let (service, probe) = counting_service();
        service.start(1.);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);
        service.start(1.);
        probe.wait_for(|probe| probe.starts.load(Ordering::Acquire) == 2);
        service.stop();

        assert_eq!(probe.max_live.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(
            *probe.order.lock().unwrap(),
            vec![("start", 1), ("exit", 1), ("start", 2), ("exit", 2)]
        );
    }

    #[test]
    fn injected_restart_takes_and_joins_gate_owned_predecessor() {
        let (service, probe, gate) = counting_service_injected();
        service.start(60.);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);
        service.start(60.);
        probe.wait_for(|probe| probe.starts.load(Ordering::Acquire) == 2);

        assert_eq!(probe.max_live.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 1);
        service.stop();
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(probe.exits.load(Ordering::Acquire), 2);
        assert!(seal_gate(&gate).is_drained());
    }

    #[test]
    fn injected_start_losing_seal_race_joins_uncommitted_worker_and_closes_service() {
        struct BlockingPrepareRunner {
            probe: Arc<WorkerProbe>,
            entered: Sender<()>,
            release: Receiver<()>,
        }
        impl CapacityWorkerRunner for BlockingPrepareRunner {
            fn prepare(&self) -> Box<dyn PreparedCapacityWorker> {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
                Box::new(PreparedCountingRunner(Arc::clone(&self.probe)))
            }
        }

        let probe = Arc::new(WorkerProbe::default());
        let gate = InjectedContextAdmissionGate::new();
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let runner: Arc<dyn CapacityWorkerRunner> = Arc::new(BlockingPrepareRunner {
            probe: Arc::clone(&probe),
            entered: entered_send,
            release: release_recv,
        });
        let service = Arc::new(AudioRenderCapacityService::new_injected(
            runner,
            gate.clone(),
        ));
        let starter_service = Arc::clone(&service);
        let starter = thread::spawn(move || starter_service.start(60.));
        entered_recv.recv_timeout(TEST_TIMEOUT).unwrap();

        let sealed = loop {
            match gate.try_seal() {
                Ok(sealed) => break sealed,
                Err(AdmissionError::Contended) => thread::yield_now(),
                Err(error) => panic!("unexpected admission seal failure: {error:?}"),
            }
        };
        let (worker, drain) = sealed.into_parts();
        assert!(worker.is_none(), "the worker has not reached commit");
        assert_eq!(drain.snapshot().capacity_registrations, 1);
        assert_eq!(drain.snapshot().capacity_producers, 1);

        release_send.send(()).unwrap();
        starter.join().unwrap();
        assert!(drain.wait().is_drained());
        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 0);

        // A surviving service clone observes the terminal start result and cannot create a worker.
        service.start(0.001);
        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
        let state = service.lock_state();
        assert!(state.closed);
        assert!(state.worker.is_none());
    }

    #[test]
    fn injected_structural_contention_is_retried_without_losing_worker_authority() {
        let (service, probe, gate) = counting_service_injected();
        let (lock_entered_send, lock_entered_recv) = crossbeam_channel::bounded(1);
        let (lock_release_send, lock_release_recv) = crossbeam_channel::bounded(1);
        let locking_gate = gate.clone();
        let locker = thread::spawn(move || {
            locking_gate.hold_phase_lock_for_test(lock_entered_send, lock_release_recv);
        });
        lock_entered_recv.recv_timeout(TEST_TIMEOUT).unwrap();

        let (start_done_send, start_done_recv) = crossbeam_channel::bounded(1);
        let starting_service = Arc::clone(&service);
        let starter = thread::spawn(move || {
            starting_service.start(60.);
            start_done_send.send(()).unwrap();
        });
        assert!(matches!(
            start_done_recv.recv_timeout(Duration::from_millis(30)),
            Err(crossbeam_channel::RecvTimeoutError::Timeout)
        ));
        assert_eq!(probe.starts.load(Ordering::Acquire), 0);
        lock_release_send.send(()).unwrap();
        locker.join().unwrap();
        start_done_recv.recv_timeout(TEST_TIMEOUT).unwrap();
        starter.join().unwrap();
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);

        service.stop();
        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);
        assert!(seal_gate(&gate).is_drained());
    }

    #[test]
    fn injected_poison_before_start_fails_closed_without_spawning() {
        let (service, probe, gate) = counting_service_injected();
        let poisoning_gate = gate.clone();
        assert!(
            thread::spawn(move || poisoning_gate.poison_phase_lock_for_test())
                .join()
                .is_err()
        );

        service.start(0.001);
        assert_eq!(probe.starts.load(Ordering::Acquire), 0);
        let state = service.lock_state();
        assert!(state.closed);
        assert!(state.worker.is_none());
    }

    #[test]
    fn injected_stop_and_seal_race_transfer_the_single_join_owner() {
        let (service, probe, gate) = counting_service_injected();
        service.start(60.);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);

        let race = Arc::new(Barrier::new(3));
        let stopping_service = Arc::clone(&service);
        let stop_race = Arc::clone(&race);
        let stopping = thread::spawn(move || {
            stop_race.wait();
            stopping_service.stop();
        });
        let sealing_gate = gate.clone();
        let seal_race = Arc::clone(&race);
        let sealing = thread::spawn(move || {
            seal_race.wait();
            seal_gate(&sealing_gate)
        });
        race.wait();
        stopping.join().unwrap();
        assert!(sealing.join().unwrap().is_drained());

        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        service.start(0.001);
        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
    }

    #[test]
    fn injected_final_service_drop_and_seal_race_do_not_detach_worker() {
        let (service, probe, gate) = counting_service_injected();
        service.start(60.);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);

        let race = Arc::new(Barrier::new(3));
        let drop_race = Arc::clone(&race);
        let dropping = thread::spawn(move || {
            drop_race.wait();
            drop(service);
        });
        let seal_race = Arc::clone(&race);
        let sealing_gate = gate.clone();
        let sealing = thread::spawn(move || {
            seal_race.wait();
            seal_gate(&sealing_gate)
        });
        race.wait();
        dropping.join().unwrap();
        assert!(sealing.join().unwrap().is_drained());
        assert_eq!(probe.starts.load(Ordering::Acquire), 1);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
    }

    #[test]
    fn concurrent_starts_are_linearized_without_orphaning_workers() {
        let (service, probe) = counting_service();
        let barrier = Arc::new(Barrier::new(9));
        let mut callers = Vec::new();
        for _ in 0..8 {
            let service = Arc::clone(&service);
            let barrier = Arc::clone(&barrier);
            callers.push(thread::spawn(move || {
                barrier.wait();
                service.start(1.);
            }));
        }
        barrier.wait();
        for caller in callers {
            caller.join().unwrap();
        }

        probe.wait_for(|probe| probe.starts.load(Ordering::Acquire) == 8);
        assert_eq!(probe.starts.load(Ordering::Acquire), 8);
        assert_eq!(probe.max_live.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 1);
        service.stop();
        assert_eq!(probe.exits.load(Ordering::Acquire), 8);
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
    }

    #[test]
    fn concurrent_start_and_stop_leave_no_unjoined_worker() {
        let (service, probe) = counting_service();
        let barrier = Arc::new(Barrier::new(17));
        let mut callers = Vec::new();
        for index in 0..16 {
            let service = Arc::clone(&service);
            let barrier = Arc::clone(&barrier);
            callers.push(thread::spawn(move || {
                barrier.wait();
                if index % 2 == 0 {
                    service.start(1.);
                } else {
                    service.stop();
                }
            }));
        }
        barrier.wait();
        for caller in callers {
            caller.join().unwrap();
        }
        service.stop();

        assert_eq!(probe.max_live.load(Ordering::Acquire), 1);
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(
            probe.starts.load(Ordering::Acquire),
            probe.exits.load(Ordering::Acquire)
        );
    }

    #[test]
    fn huge_interval_stop_is_prompt_and_join_is_authoritative() {
        let (service, probe) = counting_service();
        service.start((365 * 24 * 60 * 60) as f64);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);

        let start = Instant::now();
        service.stop();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);
    }

    #[test]
    fn baseline_is_prepared_synchronously_before_start_returns() {
        let stats = AudioStats::new();
        stats.record_render_callback(10, 10);
        let (event_send, _event_recv) = crossbeam_channel::bounded(1);
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let runner = MetricsCapacityWorker {
            stats: stats.clone(),
            sample_rate: 48_000.,
            frames_played: Arc::new(AtomicU64::new(0)),
            event_dispatch: ControlEventDispatch::legacy_for_test(event_send),
            send_observer: None,
            ready_observer: Some(ready_send),
        };
        let service = AudioRenderCapacityService::new(Arc::new(runner));

        service.start(60.);
        assert_eq!(ready_recv.try_recv(), Ok(()));
        assert_eq!(stats.take_peak_load(), 0.);
        service.stop();
    }

    #[test]
    fn infinite_interval_stops_predecessor_then_panics_without_poisoning_service() {
        let (service, probe) = counting_service();
        service.start(1.);
        probe.wait_for(|probe| probe.live.load(Ordering::Acquire) == 1);

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| service.start(f64::INFINITY)));
        assert!(result.is_err());
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(probe.exits.load(Ordering::Acquire), 1);

        service.start(1.);
        probe.wait_for(|probe| probe.starts.load(Ordering::Acquire) == 2);
        service.stop();
        assert_eq!(probe.live.load(Ordering::Acquire), 0);
        assert_eq!(probe.exits.load(Ordering::Acquire), 2);
    }

    #[test]
    fn full_event_queue_drops_one_interval_and_advances_the_baseline() {
        let stats = AudioStats::new();
        let frames_played = Arc::new(AtomicU64::new(0));
        let (event_send, event_recv) = crossbeam_channel::bounded(1);
        event_send.try_send(EventDispatch::sink_change()).unwrap();
        let (observer_send, observer_recv) = crossbeam_channel::unbounded();
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let runner = MetricsCapacityWorker {
            stats: stats.clone(),
            sample_rate: 48_000.,
            frames_played: Arc::clone(&frames_played),
            event_dispatch: ControlEventDispatch::legacy_for_test(event_send),
            send_observer: Some(observer_send),
            ready_observer: Some(ready_send),
        };
        let service = AudioRenderCapacityService::new(Arc::new(runner));

        service.start(0.001);
        assert_eq!(ready_recv.try_recv(), Ok(()));
        stats.record_render_callback(10, 10);
        assert_eq!(
            observer_recv.recv_timeout(TEST_TIMEOUT).unwrap(),
            CapacityEventSendOutcome::Full
        );
        drop(event_recv.recv_timeout(TEST_TIMEOUT).unwrap());

        frames_played.store(128, Ordering::SeqCst);
        stats.record_render_callback(0, 10);
        assert_eq!(
            observer_recv.recv_timeout(TEST_TIMEOUT).unwrap(),
            CapacityEventSendOutcome::Delivered
        );
        let stop = Instant::now();
        service.stop();
        assert!(stop.elapsed() < Duration::from_secs(1));

        let event_loop = EventLoop::new(event_recv);
        let (result_send, result_recv) = mpsc::sync_channel(1);
        event_loop.set_handler(
            EventType::RenderCapacity,
            EventHandler::Once(Box::new(move |payload| {
                let EventPayload::RenderCapacity(event) = payload else {
                    panic!("unexpected capacity payload");
                };
                result_send.send(event).unwrap();
            })),
        );
        assert!(event_loop.handle_pending_events());
        let event = result_recv.recv_timeout(TEST_TIMEOUT).unwrap();
        assert_eq!(event.average_load, 0.);
        assert_eq!(event.peak_load, 0.);
        assert_eq!(event.timestamp, 0.);
    }

    #[test]
    fn injected_tick_admitted_before_seal_finishes_then_worker_is_joined_and_drained() {
        let stats = AudioStats::new();
        let gate = InjectedContextAdmissionGate::new();
        let (event_send, event_recv) = crossbeam_channel::bounded(1);
        let (admitted_send, admitted_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let after_admission: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            admitted_send.send(()).unwrap();
            release_recv.recv().unwrap();
        });
        let event_dispatch =
            ControlEventDispatch::injected_with_observer(event_send, gate.clone(), after_admission);
        let (observer_send, observer_recv) = crossbeam_channel::unbounded();
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let runner = MetricsCapacityWorker {
            stats: stats.clone(),
            sample_rate: 48_000.,
            frames_played: Arc::new(AtomicU64::new(0)),
            event_dispatch,
            send_observer: Some(observer_send),
            ready_observer: Some(ready_send),
        };
        let service = AudioRenderCapacityService::new_injected(Arc::new(runner), gate.clone());
        service.start(0.001);
        assert_eq!(ready_recv.try_recv(), Ok(()));
        stats.record_render_callback(1, 1);
        admitted_recv.recv_timeout(TEST_TIMEOUT).unwrap();

        let sealed = loop {
            match gate.try_seal() {
                Ok(sealed) => break sealed,
                Err(AdmissionError::Contended) => thread::yield_now(),
                Err(error) => panic!("unexpected admission seal failure: {error:?}"),
            }
        };
        let (worker, drain) = sealed.into_parts();
        assert_eq!(drain.snapshot().external_events, 1);
        assert_eq!(drain.snapshot().capacity_producers, 1);
        release_send.send(()).unwrap();
        assert_eq!(
            observer_recv.recv_timeout(TEST_TIMEOUT).unwrap(),
            CapacityEventSendOutcome::Delivered
        );
        worker.unwrap().stop_and_join().unwrap();
        assert!(drain.wait().is_drained());
        assert!(event_recv.try_recv().is_ok());

        // The service still has a stale id, but the gate owns the irreversible decision. A
        // surviving clone cannot take or replace the worker after sealing.
        service.start(0.001);
        assert!(service.lock_state().closed);
    }

    #[test]
    fn injected_diagnostic_tick_drops_on_contention_and_worker_continues() {
        let stats = AudioStats::new();
        let gate = InjectedContextAdmissionGate::new();
        let (event_send, event_recv) = crossbeam_channel::bounded(2);
        let (observer_send, observer_recv) = crossbeam_channel::unbounded();
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let runner = MetricsCapacityWorker {
            stats: stats.clone(),
            sample_rate: 48_000.,
            frames_played: Arc::new(AtomicU64::new(0)),
            event_dispatch: ControlEventDispatch::injected(event_send, gate.clone()),
            send_observer: Some(observer_send),
            ready_observer: Some(ready_send),
        };
        let service = AudioRenderCapacityService::new_injected(Arc::new(runner), gate.clone());
        service.start(0.001);
        ready_recv.recv_timeout(TEST_TIMEOUT).unwrap();

        let (lock_entered_send, lock_entered_recv) = crossbeam_channel::bounded(1);
        let (lock_release_send, lock_release_recv) = crossbeam_channel::bounded(1);
        let locking_gate = gate.clone();
        let locker = thread::spawn(move || {
            locking_gate.hold_phase_lock_for_test(lock_entered_send, lock_release_recv);
        });
        lock_entered_recv.recv_timeout(TEST_TIMEOUT).unwrap();
        stats.record_render_callback(1, 1);
        assert_eq!(
            observer_recv.recv_timeout(TEST_TIMEOUT).unwrap(),
            CapacityEventSendOutcome::Dropped
        );
        assert!(event_recv.try_recv().is_err());

        lock_release_send.send(()).unwrap();
        locker.join().unwrap();
        stats.record_render_callback(1, 1);
        assert_eq!(
            observer_recv.recv_timeout(TEST_TIMEOUT).unwrap(),
            CapacityEventSendOutcome::Delivered
        );
        assert!(event_recv.recv_timeout(TEST_TIMEOUT).is_ok());
        service.stop();
        assert!(seal_gate(&gate).is_drained());
    }

    #[test]
    fn injected_full_and_disconnect_cleanup_are_admitted_and_terminal_rejection_is_lazy() {
        struct BlockingDropControl {
            entered: Sender<()>,
            release: Receiver<()>,
        }
        struct BlockingDrop(&'static BlockingDropControl);
        impl Drop for BlockingDrop {
            fn drop(&mut self) {
                self.0.entered.send(()).unwrap();
                self.0.release.recv().unwrap();
            }
        }

        let gate = InjectedContextAdmissionGate::new();
        let (event_send, event_recv) = crossbeam_channel::bounded(1);
        event_send.try_send(EventDispatch::sink_change()).unwrap();
        let dispatch = ControlEventDispatch::injected(event_send, gate.clone());
        let (drop_entered_send, drop_entered_recv) = crossbeam_channel::bounded(1);
        let (drop_release_send, drop_release_recv) = crossbeam_channel::bounded(1);
        let drop_control: &'static BlockingDropControl = Box::leak(Box::new(BlockingDropControl {
            entered: drop_entered_send,
            release: drop_release_recv,
        }));
        let sending_dispatch = dispatch.clone();
        let sending = thread::spawn(move || {
            sending_dispatch.try_send_with(|| {
                EventDispatch::message(
                    crate::context::AudioNodeId(91),
                    Box::new(BlockingDrop(drop_control)),
                )
            })
        });
        drop_entered_recv.recv_timeout(TEST_TIMEOUT).unwrap();

        let sealed = gate.try_seal().unwrap();
        let (_, drain) = sealed.into_parts();
        assert_eq!(drain.snapshot().external_events, 1);
        drop_release_send.send(()).unwrap();
        assert_eq!(sending.join().unwrap(), ControlEventSendOutcome::Full);
        assert!(drain.wait().is_drained());

        let factory_called = AtomicBool::new(false);
        assert_eq!(
            dispatch.try_send_with(|| {
                factory_called.store(true, Ordering::Release);
                EventDispatch::sink_change()
            }),
            ControlEventSendOutcome::AdmissionRejected(AdmissionError::Sealed)
        );
        assert!(!factory_called.load(Ordering::Acquire));
        drop(event_recv);

        let disconnected_gate = InjectedContextAdmissionGate::new();
        let (event_send, event_recv) = crossbeam_channel::bounded(1);
        drop(event_recv);
        let disconnected = ControlEventDispatch::injected(event_send, disconnected_gate.clone());
        assert_eq!(
            disconnected.try_send_with(EventDispatch::sink_change),
            ControlEventSendOutcome::Disconnected
        );
        assert!(seal_gate(&disconnected_gate).is_drained());
    }

    #[test]
    fn disconnected_event_receiver_exits_worker_and_is_joined_by_stop() {
        let stats = AudioStats::new();
        let (event_send, event_recv) = crossbeam_channel::bounded(1);
        drop(event_recv);
        let (observer_send, observer_recv) = crossbeam_channel::unbounded();
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let runner = MetricsCapacityWorker {
            stats: stats.clone(),
            sample_rate: 48_000.,
            frames_played: Arc::new(AtomicU64::new(0)),
            event_dispatch: ControlEventDispatch::legacy_for_test(event_send),
            send_observer: Some(observer_send),
            ready_observer: Some(ready_send),
        };
        let service = AudioRenderCapacityService::new(Arc::new(runner));

        service.start(0.001);
        assert_eq!(ready_recv.try_recv(), Ok(()));
        stats.record_render_callback(1, 1);
        assert_eq!(
            observer_recv.recv_timeout(TEST_TIMEOUT).unwrap(),
            CapacityEventSendOutcome::Disconnected
        );
        let start = Instant::now();
        service.stop();
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn worker_panic_is_contained_and_hostile_payload_is_quarantined() {
        struct HostilePayload(Arc<AtomicBool>);
        impl Drop for HostilePayload {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
                panic!("hostile capacity panic payload drop");
            }
        }

        struct PanicOnceRunner {
            first: AtomicBool,
            payload_dropped: Arc<AtomicBool>,
        }
        impl CapacityWorkerRunner for PanicOnceRunner {
            fn prepare(&self) -> Box<dyn PreparedCapacityWorker> {
                struct PreparedPanicOnceRunner {
                    panic: bool,
                    payload_dropped: Arc<AtomicBool>,
                }
                impl PreparedCapacityWorker for PreparedPanicOnceRunner {
                    fn run(self: Box<Self>, stop: Receiver<()>, _update_interval: Duration) {
                        if self.panic {
                            std::panic::panic_any(HostilePayload(self.payload_dropped));
                        }
                        let _ = stop.recv();
                    }
                }

                Box::new(PreparedPanicOnceRunner {
                    panic: self.first.swap(false, Ordering::AcqRel),
                    payload_dropped: Arc::clone(&self.payload_dropped),
                })
            }
        }

        let payload_dropped = Arc::new(AtomicBool::new(false));
        let runner = PanicOnceRunner {
            first: AtomicBool::new(true),
            payload_dropped: Arc::clone(&payload_dropped),
        };
        let service = AudioRenderCapacityService::new(Arc::new(runner));
        service.start(0.001);
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| service.stop())).is_ok());
        assert!(!payload_dropped.load(Ordering::Acquire));

        // The serialized service remains usable after containing and joining the failed worker.
        service.start(0.001);
        service.stop();
    }
}
