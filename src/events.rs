#[cfg(feature = "diagnostics")]
use crate::context::AudioContextDiagnostics;
use crate::context::ConcreteBaseAudioContext;
use crate::context::{
    AdmissionError, AudioContextState, AudioNodeId, InjectedContextAdmissionGate,
    InjectedControlRenderInit,
};
use crate::message::{ControlBatchApplied, ControlMessage};
use crate::render::RenderThread;
use crate::stats::AudioStats;
use crate::{AudioBuffer, AudioRenderCapacityEvent};

use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, TrySendError};

use crate::render::InjectedEventDispatchSender;

type EventActivityHandler = dyn Fn() + Send + Sync + 'static;
#[cfg(test)]
type AfterAdmissionObserver = dyn Fn() + Send + Sync + 'static;

/// The Event interface
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Event {
    pub type_: &'static str,
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub(crate) enum EventType {
    ControlBatchActivity,
    Ended(AudioNodeId),
    SinkChange,
    StateChange,
    RenderCapacity,
    ProcessorError(AudioNodeId),
    #[cfg(feature = "diagnostics")]
    Diagnostics,
    Message(AudioNodeId),
    Complete,
    AudioProcessing(AudioNodeId),
}

/// The Error Event interface
#[non_exhaustive]
#[derive(Debug)]
pub struct ErrorEvent {
    /// The error message
    pub message: String,
    /// The object with which panic was originally invoked.
    pub error: Box<dyn Any + Send>,
    /// Inherits from this base Event
    pub event: Event,
}

/// The AudioProcessingEvent interface
#[non_exhaustive]
#[derive(Debug)]
pub struct AudioProcessingEvent {
    /// The input buffer
    pub input_buffer: AudioBuffer,
    /// The output buffer
    pub output_buffer: AudioBuffer,
    /// The time when the audio will be played in the same time coordinate system as the
    /// AudioContext's currentTime.
    pub playback_time: f64,
    pub(crate) registration: Option<(ConcreteBaseAudioContext, AudioNodeId)>,
}

impl Drop for AudioProcessingEvent {
    fn drop(&mut self) {
        if let Some((context, id)) = self.registration.take() {
            let wrapped = crate::message::ControlMessage::NodeMessage {
                id,
                msg: llq::Node::new(Box::new(self.output_buffer.clone())),
            };
            context.send_control_msg(wrapped);
        }
    }
}

/// The OfflineAudioCompletionEvent Event interface
#[non_exhaustive]
#[derive(Debug)]
pub struct OfflineAudioCompletionEvent {
    /// The rendered AudioBuffer
    pub rendered_buffer: AudioBuffer,
    /// Inherits from this base Event
    pub event: Event,
}

#[derive(Debug)]
pub(crate) enum EventPayload {
    None,
    RenderCapacity(AudioRenderCapacityEvent),
    ProcessorError(ErrorEvent),
    #[cfg(feature = "diagnostics")]
    Diagnostics(AudioContextDiagnostics),
    Message(Box<dyn Any + Send + 'static>),
    AudioContextState(AudioContextState),
    Complete(AudioBuffer),
    AudioProcessing(AudioProcessingEvent),
}

#[derive(Debug)]
pub(crate) struct EventDispatch {
    type_: EventType,
    payload: EventPayload,
}

impl EventDispatch {
    fn is_closed_state_change(&self) -> bool {
        matches!(
            self.payload,
            EventPayload::AudioContextState(AudioContextState::Closed)
        )
    }

    pub(crate) fn control_batch_activity() -> Self {
        EventDispatch {
            type_: EventType::ControlBatchActivity,
            payload: EventPayload::None,
        }
    }

    pub fn ended(id: AudioNodeId) -> Self {
        EventDispatch {
            type_: EventType::Ended(id),
            payload: EventPayload::None,
        }
    }

    pub fn sink_change() -> Self {
        EventDispatch {
            type_: EventType::SinkChange,
            payload: EventPayload::None,
        }
    }

    pub fn state_change(state: AudioContextState) -> Self {
        EventDispatch {
            type_: EventType::StateChange,
            payload: EventPayload::AudioContextState(state),
        }
    }

    pub fn render_capacity(value: AudioRenderCapacityEvent) -> Self {
        EventDispatch {
            type_: EventType::RenderCapacity,
            payload: EventPayload::RenderCapacity(value),
        }
    }

    pub fn processor_error(id: AudioNodeId, value: ErrorEvent) -> Self {
        EventDispatch {
            type_: EventType::ProcessorError(id),
            payload: EventPayload::ProcessorError(value),
        }
    }

    #[cfg(feature = "diagnostics")]
    pub fn diagnostics(value: AudioContextDiagnostics) -> Self {
        EventDispatch {
            type_: EventType::Diagnostics,
            payload: EventPayload::Diagnostics(value),
        }
    }

    pub fn message(id: AudioNodeId, value: Box<dyn Any + Send + 'static>) -> Self {
        EventDispatch {
            type_: EventType::Message(id),
            payload: EventPayload::Message(value),
        }
    }

    pub fn complete(buffer: AudioBuffer) -> Self {
        EventDispatch {
            type_: EventType::Complete,
            payload: EventPayload::Complete(buffer),
        }
    }

    pub fn audio_processing(id: AudioNodeId, value: AudioProcessingEvent) -> Self {
        EventDispatch {
            type_: EventType::AudioProcessing(id),
            payload: EventPayload::AudioProcessing(value),
        }
    }
}

pub(crate) enum EventHandler {
    Once(Box<dyn FnOnce(EventPayload) + Send + 'static>),
    Multiple(Box<dyn FnMut(EventPayload) + Send + 'static>),
}

#[derive(Clone)]
pub(crate) struct EventLoop {
    event_recv: Receiver<EventDispatch>,
    event_handlers: Arc<Mutex<HashMap<EventType, EventHandler>>>,
    event_activity_handler: Arc<Mutex<Option<Arc<EventActivityHandler>>>>,
}

/// Single-use setup authority for one injected context's exact event channel.
///
/// The raw sender, receiver-owning thread, and identity cannot be separated. Only the injected
/// render initializer may consume this value and derive the render, admitted-control, and
/// lifecycle branches. No branch exposes a raw sender or receiver.
pub(crate) struct InjectedEventDispatchSetup {
    sender: Sender<EventDispatch>,
    event_loop: JoinableEventLoop,
    handlers: Arc<Mutex<HashMap<EventType, EventHandler>>>,
    activity: Arc<Mutex<Option<Arc<EventActivityHandler>>>>,
    identity: Arc<()>,
}

/// Non-clone handler-only setup view. It has no event receiver, producer, stop, or join authority.
#[allow(dead_code)] // production handler assembly follows the private B4a ownership seam
pub(crate) struct InjectedEventHandlerSetup<'a> {
    handlers: &'a Arc<Mutex<HashMap<EventType, EventHandler>>>,
}

impl InjectedEventHandlerSetup<'_> {
    #[allow(dead_code)] // production handler assembly follows the private B4a ownership seam
    pub(crate) fn set_handler(&self, event: EventType, callback: EventHandler) {
        self.handlers.lock().unwrap().insert(event, callback);
    }
}

/// Absorbing state shared only by the exact injected renderer, concrete base, and lifecycle owner.
#[derive(Clone)]
pub(crate) struct InjectedContextState(Arc<AtomicU8>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedStateTransition {
    Changed,
    Unchanged,
    ClosedAbsorbing,
    TerminalRejected,
}

impl InjectedContextState {
    fn new(initially_suspended: bool) -> Self {
        Self(Arc::new(AtomicU8::new(if initially_suspended {
            AudioContextState::Suspended as u8
        } else {
            AudioContextState::Running as u8
        })))
    }

    pub(crate) fn load(&self) -> AudioContextState {
        self.0.load(Ordering::Acquire).into()
    }

    /// Changes a live injected state without ever reopening terminal `Closed`.
    pub(crate) fn transition_live(&self, next: AudioContextState) -> InjectedStateTransition {
        if next == AudioContextState::Closed {
            return InjectedStateTransition::TerminalRejected;
        }
        let mut current = self.0.load(Ordering::Acquire);
        loop {
            if current == AudioContextState::Closed as u8 {
                return InjectedStateTransition::ClosedAbsorbing;
            }
            if current == next as u8 {
                return InjectedStateTransition::Unchanged;
            }
            match self.0.compare_exchange_weak(
                current,
                next as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return InjectedStateTransition::Changed,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn transition_render(&self, next: AudioContextState) -> InjectedStateTransition {
        if next == AudioContextState::Closed {
            if self.transition_closed() {
                InjectedStateTransition::Changed
            } else {
                InjectedStateTransition::Unchanged
            }
        } else {
            self.transition_live(next)
        }
    }

    pub(crate) fn atomic_for_render(&self) -> Arc<AtomicU8> {
        Arc::clone(&self.0)
    }

    fn transition_closed(&self) -> bool {
        let mut current = self.0.load(Ordering::Acquire);
        loop {
            if current == AudioContextState::Closed as u8 {
                return false;
            }
            match self.0.compare_exchange_weak(
                current,
                AudioContextState::Closed as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Cloneable control-side producer which always acquires exact external-event admission first.
#[derive(Clone)]
pub(crate) struct InjectedControlEventDispatch {
    sender: Sender<EventDispatch>,
    identity: Arc<()>,
    gate: InjectedContextAdmissionGate,
    state: InjectedContextState,
    handlers: Arc<Mutex<HashMap<EventType, EventHandler>>>,
    activity: Arc<Mutex<Option<Arc<EventActivityHandler>>>>,
    #[cfg(test)]
    after_admission: Arc<Mutex<Option<Arc<AfterAdmissionObserver>>>>,
}

impl InjectedControlEventDispatch {
    pub(crate) fn matches_gate(&self, gate: &InjectedContextAdmissionGate) -> bool {
        self.gate.ptr_eq(gate)
    }

    pub(crate) fn matches_identity(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.identity, identity)
    }

    pub(crate) fn admission_gate(&self) -> InjectedContextAdmissionGate {
        self.gate.clone()
    }

    pub(crate) fn state(&self) -> AudioContextState {
        self.state.load()
    }

    pub(crate) fn transition_live_state(&self, next: AudioContextState) -> InjectedStateTransition {
        self.state.transition_live(next)
    }

    pub(crate) fn try_send_with<F>(
        &self,
        make_event: F,
    ) -> Result<(), InjectedControlEventSendError>
    where
        F: FnOnce() -> EventDispatch + Copy,
    {
        let admission = self
            .gate
            .try_external_event()
            .map_err(InjectedControlEventSendError::Admission)?;
        #[cfg(test)]
        if let Some(observer) = self.after_admission.lock().unwrap().clone() {
            observer();
        }
        let outcome = match self.sender.try_send(make_event()) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                drop(event);
                Err(InjectedControlEventSendError::Full)
            }
            Err(TrySendError::Disconnected(event)) => {
                drop(event);
                Err(InjectedControlEventSendError::Disconnected)
            }
        };
        drop(admission);
        outcome
    }

    pub(crate) fn set_handler(&self, event: EventType, callback: EventHandler) {
        self.handlers.lock().unwrap().insert(event, callback);
    }

    pub(crate) fn clear_handler(&self, event: EventType) {
        self.handlers.lock().unwrap().remove(&event);
    }

    pub(crate) fn set_activity_handler<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        *self.activity.lock().unwrap() = Some(Arc::new(callback));
    }

    pub(crate) fn clear_activity_handler(&self) {
        self.activity.lock().unwrap().take();
    }

    #[cfg(test)]
    pub(crate) fn set_after_admission_for_test(&self, observer: Arc<dyn Fn() + Send + Sync>) {
        *self.after_admission.lock().unwrap() = Some(observer);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedControlEventSendError {
    Full,
    Disconnected,
    Admission(AdmissionError),
}

/// Sole event-consumer retirement authority paired with the exact injected state.
pub(crate) struct InjectedLifecycleEventLoop {
    event_loop: Option<JoinableEventLoop>,
    identity: Arc<()>,
    state: InjectedContextState,
}

#[derive(Clone)]
pub(crate) struct InjectedEventIdentity(Arc<()>);

impl InjectedEventIdentity {
    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedTerminalStateOutcome {
    Published,
    RenderedCloseVerified,
    MissingRenderedClosePublished,
    UnexpectedAlreadyClosed,
}

impl InjectedLifecycleEventLoop {
    pub(crate) fn matches_identity(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.identity, identity)
    }

    pub(crate) fn producer_identity(&self) -> InjectedEventIdentity {
        InjectedEventIdentity(Arc::clone(&self.identity))
    }

    pub(crate) fn retire_confirmed(
        mut self,
        retired: &crate::context::RetiredInjectedGraph,
        graceful: bool,
    ) -> Result<InjectedConfirmedEventRetirement, Self> {
        if !retired.matches_event_identity(&self.producer_identity()) {
            return Err(self);
        }
        let close_applied = retired.close_applied();
        let already_closed = !self.state.transition_closed();
        let outcome = match (close_applied, already_closed) {
            (true, true) => InjectedTerminalStateOutcome::RenderedCloseVerified,
            (true, false) => InjectedTerminalStateOutcome::MissingRenderedClosePublished,
            (false, true) => InjectedTerminalStateOutcome::UnexpectedAlreadyClosed,
            (false, false) => InjectedTerminalStateOutcome::Published,
        };
        let mut event_loop = self.event_loop.take().unwrap();
        let effective_graceful =
            graceful && outcome == InjectedTerminalStateOutcome::RenderedCloseVerified;
        if effective_graceful {
            event_loop.request_graceful_stop();
        } else {
            event_loop.request_terminal_closed_stop();
        }
        Ok(InjectedConfirmedEventRetirement {
            state: outcome,
            graceful: effective_graceful,
            joined: event_loop.join(),
        })
    }
}

pub(crate) struct InjectedConfirmedEventRetirement {
    pub(crate) state: InjectedTerminalStateOutcome,
    pub(crate) graceful: bool,
    pub(crate) joined: Result<EventLoopExit, EventLoopJoinError>,
}

impl Drop for InjectedLifecycleEventLoop {
    fn drop(&mut self) {
        if let Some(event_loop) = self.event_loop.take() {
            // An abandoned exact owner carries no producer-quiescence proof. Its ordinary Drop
            // must not request a stop while a render/control producer can still be live.
            std::mem::forget(event_loop);
        }
    }
}

pub(crate) struct BoundInjectedEventDispatch {
    render: InjectedEventDispatchSender,
    control: InjectedControlEventDispatch,
    lifecycle: InjectedLifecycleEventLoop,
    state: InjectedContextState,
}

impl BoundInjectedEventDispatch {
    /// Consumes the whole exact event bundle while constructing the sole renderer. State cannot be
    /// detached or swapped independently from either producer or the lifecycle consumer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn install_renderer(
        self,
        sample_rate: f32,
        number_of_channels: usize,
        receiver: Receiver<ControlMessage>,
        frames_played: Arc<AtomicU64>,
        stats: AudioStats,
        applied: ControlBatchApplied,
    ) -> (
        RenderThread,
        InjectedControlEventDispatch,
        InjectedLifecycleEventLoop,
    ) {
        let renderer = RenderThread::new_injected(
            sample_rate,
            number_of_channels,
            receiver,
            self.state,
            frames_played,
            stats,
            self.render,
            applied,
        );
        (renderer, self.control, self.lifecycle)
    }
}

impl InjectedEventDispatchSetup {
    pub(crate) fn bind(self, control: &InjectedControlRenderInit) -> BoundInjectedEventDispatch {
        let gate = control.event_admission_gate();
        let initially_suspended = control.initially_suspended();
        let state = InjectedContextState::new(initially_suspended);
        BoundInjectedEventDispatch {
            render: InjectedEventDispatchSender::from_event_setup(
                self.sender.clone(),
                Arc::clone(&self.identity),
            ),
            control: InjectedControlEventDispatch {
                sender: self.sender,
                identity: Arc::clone(&self.identity),
                gate,
                state: state.clone(),
                handlers: self.handlers,
                activity: self.activity,
                #[cfg(test)]
                after_admission: Arc::new(Mutex::new(None)),
            },
            lifecycle: InjectedLifecycleEventLoop {
                event_loop: Some(self.event_loop),
                identity: self.identity,
                state: state.clone(),
            },
            state,
        }
    }
}

#[derive(Clone, Copy)]
enum EventLoopStop {
    Graceful,
    Silent,
    TerminalClosed,
}

/// Confirmed reason that a lifecycle-owned event-loop thread exited.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // consumed by the private lifecycle; public AudioContext wiring is pending
pub(crate) enum EventLoopExit {
    /// Queued records were drained and one final `Closed` state change was dispatched.
    Graceful,
    /// The thread stopped without draining or dispatching another public event.
    Silent,
    /// Queued records were discarded and exactly one final `Closed` state change was dispatched.
    TerminalClosed,
}

/// Join failure for the lifecycle-owned event-loop thread.
#[allow(dead_code)] // consumed by the private lifecycle; public AudioContext wiring is pending
pub(crate) enum EventLoopJoinError {
    /// Joining the current thread would deadlock.
    CurrentThread,
    /// The dedicated stop authority disappeared without selecting a stop mode.
    StopChannelDisconnected,
    /// An event handler panicked. The original payload is retained for lifecycle diagnostics.
    Panicked(Box<dyn Any + Send + 'static>),
}

impl std::fmt::Debug for EventLoopJoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CurrentThread => f.write_str("EventLoopJoinError::CurrentThread"),
            Self::StopChannelDisconnected => {
                f.write_str("EventLoopJoinError::StopChannelDisconnected")
            }
            Self::Panicked(_) => f.write_str("EventLoopJoinError::Panicked(..)"),
        }
    }
}

/// Lifecycle owner for a joinable event-loop thread.
///
/// Stop requests use a dedicated bounded channel, independent of the best-effort Web Audio event
/// queue. Requesting stop is not an acknowledgement; only [`Self::join`] proves that no callback
/// from this event thread remains or can begin. A successful join does not prove destruction of
/// handlers or queued payloads retained by other [`EventLoop`] or receiver clones. This owner is
/// `Send`, so a lifecycle worker may request stop and move it to a different thread for the join.
#[allow(dead_code)] // consumed by the private lifecycle; public AudioContext wiring is pending
pub(crate) struct JoinableEventLoop {
    stop_send: crossbeam_channel::Sender<EventLoopStop>,
    join: Option<JoinHandle<Result<EventLoopExit, EventLoopJoinError>>>,
    stop_requested: bool,
}

/// Creates the inseparable injected event setup without exposing a raw producer or consumer.
#[allow(dead_code)] // selected by the pending private injected AudioContext constructor
pub(crate) fn injected_event_dispatch_setup() -> std::io::Result<InjectedEventDispatchSetup> {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let event_loop = EventLoop::new(receiver);
    finish_injected_event_dispatch_setup(sender, event_loop)
}

#[cfg(test)]
pub(crate) fn injected_event_dispatch_setup_with_handlers(
    setup: impl FnOnce(InjectedEventHandlerSetup<'_>),
) -> std::io::Result<InjectedEventDispatchSetup> {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let event_loop = EventLoop::new(receiver);
    setup(InjectedEventHandlerSetup {
        handlers: &event_loop.event_handlers,
    });
    finish_injected_event_dispatch_setup(sender, event_loop)
}

#[cfg(test)]
pub(crate) fn injected_event_dispatch_setup_bounded_for_test(
    capacity: usize,
    setup: impl FnOnce(InjectedEventHandlerSetup<'_>),
) -> std::io::Result<InjectedEventDispatchSetup> {
    let (sender, receiver) = crossbeam_channel::bounded(capacity);
    let event_loop = EventLoop::new(receiver);
    setup(InjectedEventHandlerSetup {
        handlers: &event_loop.event_handlers,
    });
    finish_injected_event_dispatch_setup(sender, event_loop)
}

fn finish_injected_event_dispatch_setup(
    sender: crossbeam_channel::Sender<EventDispatch>,
    event_loop: EventLoop,
) -> std::io::Result<InjectedEventDispatchSetup> {
    let identity = Arc::new(());
    let handlers = Arc::clone(&event_loop.event_handlers);
    let activity = Arc::clone(&event_loop.event_activity_handler);
    let event_loop = event_loop.run_joinable()?;
    Ok(InjectedEventDispatchSetup {
        sender,
        event_loop,
        handlers,
        activity,
        identity,
    })
}

#[allow(dead_code)] // consumed by private B3b lifecycle; public AudioContext wiring is pending
impl JoinableEventLoop {
    /// Requests an ordered graceful stop.
    ///
    /// Render and control producers must already be quiescent. The event thread drains records
    /// already in the queue, coalescing any real `Closed` records, then dispatches exactly one
    /// final `Closed` state change. The private injected lifecycle establishes producer
    /// quiescence internally; legacy lifecycle callers supply their existing proof. Call
    /// [`Self::join`] for retirement acknowledgement.
    pub(crate) fn request_graceful_stop(&mut self) {
        self.request_stop(EventLoopStop::Graceful);
    }

    /// Requests a silent stop without draining or dispatching another queued or synthetic event.
    ///
    /// Event producers should already be quiescent. Call [`Self::join`] for retirement
    /// acknowledgement.
    pub(crate) fn request_silent_stop(&mut self) {
        self.request_stop(EventLoopStop::Silent);
    }

    /// Discards unrelated queued records and dispatches exactly one terminal `Closed` event.
    /// Producers and exact state publication must already be quiescent/complete.
    fn request_terminal_closed_stop(&mut self) {
        self.request_stop(EventLoopStop::TerminalClosed);
    }

    fn request_stop(&mut self, mode: EventLoopStop) {
        if self.stop_requested {
            return;
        }
        self.stop_requested = true;

        // There is one sender and at most one request, so a live receiver always has capacity.
        // Disconnection means the thread already exited naturally or panicked; join remains the
        // authoritative outcome in either case.
        match self.stop_send.try_send(mode) {
            Ok(())
            | Err(crossbeam_channel::TrySendError::Disconnected(_))
            | Err(crossbeam_channel::TrySendError::Full(_)) => {}
        }
    }

    /// Waits for event-thread retirement and returns its confirmed stop mode.
    pub(crate) fn join(mut self) -> Result<EventLoopExit, EventLoopJoinError> {
        let join = self
            .join
            .take()
            .expect("joinable event loop owns one thread handle");
        if join.thread().id() == std::thread::current().id() {
            self.join = Some(join);
            return Err(EventLoopJoinError::CurrentThread);
        }

        match join.join() {
            Ok(result) => result,
            Err(payload) => Err(EventLoopJoinError::Panicked(payload)),
        }
    }
}

impl Drop for JoinableEventLoop {
    fn drop(&mut self) {
        if self.join.is_some() {
            // This merely commits a best-effort silent retirement request. Dropping JoinHandle
            // detaches the thread; no acknowledgement is claimed by this path.
            self.request_stop(EventLoopStop::Silent);
        }
    }
}

impl EventLoop {
    pub fn new(event_recv: Receiver<EventDispatch>) -> Self {
        Self {
            event_recv,
            event_handlers: Default::default(),
            event_activity_handler: Default::default(),
        }
    }

    fn handle_event(&self, mut event: EventDispatch) -> ControlFlow<()> {
        // Notify native integrations before targeted handler lookup. A dequeued event therefore
        // remains a useful wake even when its target is stale or has no handler.
        let activity_handler = self.event_activity_handler.lock().unwrap().clone();
        if let Some(callback) = activity_handler {
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback())).is_err() {
                log::error!("Event activity handler panicked; disabling it");

                // Do not hold the lock while invoking the callback. If the callback (or another
                // thread) installed a replacement before panicking, preserve that replacement.
                let mut handler = self.event_activity_handler.lock().unwrap();
                if handler
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &callback))
                {
                    handler.take();
                }
            }
        }

        // Terminate the event loop when the audio context is closing
        let mut result = ControlFlow::Continue(());
        if matches!(
            event.payload,
            EventPayload::AudioContextState(AudioContextState::Closed)
        ) {
            event.payload = EventPayload::None; // the statechange handler takes no argument
            result = ControlFlow::Break(());
        }

        // In unit tests, we rethrow panics to avoid missing critical node exceptions
        // https://github.com/orottier/web-audio-api-rs/issues/522
        #[cfg(test)]
        if let EventPayload::ProcessorError(e) = event.payload {
            panic!("Rethrowing exception during tests: {:?}", e);
        }

        let mut event_handler_lock = self.event_handlers.lock().unwrap();
        let callback_option = event_handler_lock.remove(&event.type_);
        drop(event_handler_lock); // release Mutex while running callback

        if let Some(callback) = callback_option {
            match callback {
                EventHandler::Once(f) => (f)(event.payload),
                EventHandler::Multiple(mut f) => {
                    (f)(event.payload);
                    self.event_handlers
                        .lock()
                        .unwrap()
                        .insert(event.type_, EventHandler::Multiple(f));
                }
            };
        }

        result
    }

    #[inline(always)]
    pub fn handle_pending_events(&self) -> bool {
        let mut events_were_handled = false;
        // try_iter will yield all pending events, but does not block
        for event in self.event_recv.try_iter() {
            // we can ignore the return value, it is only useful in the event thread
            let _ = self.handle_event(event);
            events_were_handled = true;
        }
        events_were_handled
    }

    pub fn run_in_thread(&self) {
        log::debug!("Entering event thread");

        // split borrows to help compiler
        let self_clone = self.clone();

        std::thread::spawn(move || {
            self_clone.run();
        });
    }

    /// Starts a lifecycle-owned event thread that can be stopped and explicitly joined.
    ///
    /// This is separate from [`Self::run_in_thread`], whose detached legacy behavior is preserved.
    #[allow(dead_code)] // consumed by the private lifecycle; public AudioContext wiring is pending
    pub(crate) fn run_joinable(&self) -> std::io::Result<JoinableEventLoop> {
        let (stop_send, stop_recv) = crossbeam_channel::bounded(1);
        let event_loop = self.clone();
        let join = std::thread::Builder::new()
            .name("web-audio-event-loop".to_owned())
            .spawn(move || event_loop.run_with_stop(stop_recv))?;
        Ok(JoinableEventLoop {
            stop_send,
            join: Some(join),
            stop_requested: false,
        })
    }

    fn run(&self) {
        // This thread is dedicated to event handling, so we can block.
        for event in self.event_recv.iter() {
            let result = self.handle_event(event);
            if result.is_break() {
                break;
            }
        }

        log::debug!("Event loop has terminated");
    }

    fn run_with_stop(
        &self,
        stop_recv: Receiver<EventLoopStop>,
    ) -> Result<EventLoopExit, EventLoopJoinError> {
        log::debug!("Entering joinable event thread");

        let mut event_queue_connected = true;
        loop {
            if !event_queue_connected {
                let stop = stop_recv
                    .recv()
                    .map_err(|_| EventLoopJoinError::StopChannelDisconnected)?;
                return Ok(self.finish_stop(stop));
            }

            crossbeam_channel::select_biased! {
                recv(stop_recv) -> stop => {
                    let stop = stop
                        .map_err(|_| EventLoopJoinError::StopChannelDisconnected)?;
                    return Ok(self.finish_stop(stop));
                }
                recv(self.event_recv) -> event => {
                    match event {
                        Ok(event) => {
                            // A lossy real Closed record is not lifecycle authority. Coalesce it
                            // and keep draining until the reliable stop channel selects a mode.
                            if !event.is_closed_state_change() {
                                let result = self.handle_event(event);
                                debug_assert!(result.is_continue());
                            }
                        }
                        Err(_) => event_queue_connected = false,
                    }
                }
            }
        }
    }

    fn finish_stop(&self, stop: EventLoopStop) -> EventLoopExit {
        match stop {
            EventLoopStop::Silent => EventLoopExit::Silent,
            EventLoopStop::TerminalClosed => {
                // Silent/controller-drop shutdown does not deliver stale unrelated records. The
                // lifecycle caller has already proved every producer quiescent, so this discard is
                // an authoritative boundary and handler reentrancy cannot race a live producer.
                while let Ok(event) = self.event_recv.try_recv() {
                    drop(event);
                }
                let final_close = EventDispatch::state_change(AudioContextState::Closed);
                let result = self.handle_event(final_close);
                debug_assert!(result.is_break());
                // A terminal handler may enqueue through a surviving but sealed capability. It is
                // rejected before construction; test-only raw reentrancy is discarded here too.
                while let Ok(event) = self.event_recv.try_recv() {
                    drop(event);
                }
                EventLoopExit::TerminalClosed
            }
            EventLoopStop::Graceful => {
                // The lifecycle caller promises that producers are quiescent before requesting
                // graceful stop, so reaching an empty queue is an authoritative drain boundary.
                // Handler-reentrant records are also observed by this repeated try_recv loop.
                while let Ok(event) = self.event_recv.try_recv() {
                    if !event.is_closed_state_change() {
                        let result = self.handle_event(event);
                        debug_assert!(result.is_continue());
                    }
                }

                let final_close = EventDispatch::state_change(AudioContextState::Closed);
                let result = self.handle_event(final_close);
                debug_assert!(result.is_break());
                EventLoopExit::Graceful
            }
        }
    }

    pub fn set_handler(&self, event: EventType, callback: EventHandler) {
        self.event_handlers.lock().unwrap().insert(event, callback);
    }

    pub fn clear_handler(&self, event: EventType) {
        self.event_handlers.lock().unwrap().remove(&event);
    }

    pub fn set_activity_handler<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        *self.event_activity_handler.lock().unwrap() = Some(Arc::new(callback));
    }

    pub fn clear_activity_handler(&self) {
        self.event_activity_handler.lock().unwrap().take();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;
    use crate::context::AudioNodeId;
    use crate::node::ScheduledSourceCompletionToken;
    use crate::render::AudioWorkletGlobalScope;

    #[test]
    fn unrelated_backlog_wakes_reconciliation_after_terminal_send_is_dropped() {
        let (control_init, render_init) = crate::io::thread_init();
        let event_loop = EventLoop::new(control_init.event_recv);
        let completion = ScheduledSourceCompletionToken::new();
        let roots = Arc::new(Mutex::new(vec![completion.clone()]));
        let roots_for_activity = Arc::clone(&roots);
        let activity_count = Arc::new(AtomicUsize::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        event_loop.set_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
            roots_for_activity
                .lock()
                .unwrap()
                .retain(|token| !token.is_complete());
        });

        let mut queued = 0;
        while render_init
            .event_send
            .try_send(EventDispatch::sink_change())
            .is_ok()
        {
            queued += 1;
        }
        assert_eq!(queued, 256);

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48_000.,
            node_id: Cell::new(AudioNodeId(42)),
            event_sender: render_init.event_send.into(),
        };
        completion.mark_complete_and_wake(&scope);

        assert!(completion.is_complete());
        assert_eq!(roots.lock().unwrap().len(), 1);
        assert!(event_loop.handle_pending_events());
        assert!(roots.lock().unwrap().is_empty());
        assert_eq!(activity_count.load(Ordering::Relaxed), queued);
    }

    #[test]
    fn stale_reused_id_event_cannot_consume_context_activity_wake() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let reused_id = AudioNodeId(42);

        let old_completion = ScheduledSourceCompletionToken::new();
        old_completion.mark_complete();
        let new_completion = ScheduledSourceCompletionToken::new();
        let roots = Arc::new(Mutex::new(vec![old_completion, new_completion.clone()]));
        let roots_for_activity = Arc::clone(&roots);
        let activity_count = Arc::new(AtomicUsize::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        event_loop.set_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
            roots_for_activity
                .lock()
                .unwrap()
                .retain(|token| !token.is_complete());
        });

        // Model a new source reusing an id while raw records from the old source remain queued.
        let raw_handler_count = Arc::new(AtomicUsize::new(0));
        let raw_handler_count_clone = Arc::clone(&raw_handler_count);
        event_loop.set_handler(
            EventType::Ended(reused_id),
            EventHandler::Once(Box::new(move |_| {
                raw_handler_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );
        event_send.send(EventDispatch::sink_change()).unwrap();
        event_send.send(EventDispatch::ended(reused_id)).unwrap();

        assert!(event_loop.handle_pending_events());
        assert_eq!(activity_count.load(Ordering::Relaxed), 2);
        assert_eq!(raw_handler_count.load(Ordering::Relaxed), 1);
        assert_eq!(roots.lock().unwrap().len(), 1);

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48_000.,
            node_id: Cell::new(reused_id),
            event_sender: event_send.into(),
        };
        new_completion.mark_complete_and_wake(&scope);

        assert!(event_loop.handle_pending_events());
        assert_eq!(activity_count.load(Ordering::Relaxed), 3);
        assert_eq!(raw_handler_count.load(Ordering::Relaxed), 1);
        assert!(roots.lock().unwrap().is_empty());
    }

    #[test]
    fn close_notifies_activity_before_handler_and_stops_event_loop() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let activity_count = Arc::new(AtomicUsize::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        event_loop.set_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let close_handler_count = Arc::new(AtomicUsize::new(0));
        let close_handler_count_clone = Arc::clone(&close_handler_count);
        let activity_seen_by_handler = Arc::clone(&activity_count);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                assert_eq!(activity_seen_by_handler.load(Ordering::Relaxed), 1);
                close_handler_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );

        event_send
            .send(EventDispatch::state_change(AudioContextState::Closed))
            .unwrap();
        event_send.send(EventDispatch::sink_change()).unwrap();
        let (done_send, done_recv) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            event_loop.run();
            done_send.send(()).unwrap();
        });

        assert!(done_recv.recv_timeout(Duration::from_secs(1)).is_ok());
        assert_eq!(activity_count.load(Ordering::Relaxed), 1);
        assert_eq!(close_handler_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn disconnected_receiver_stops_without_spurious_activity() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let activity_count = Arc::new(AtomicUsize::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        event_loop.set_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let (done_send, done_recv) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            event_loop.run();
            done_send.send(()).unwrap();
        });
        drop(event_send);

        assert!(done_recv.recv_timeout(Duration::from_secs(1)).is_ok());
        assert_eq!(activity_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn panicking_activity_handler_is_disabled_without_stopping_targeted_events() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let activity_count = Arc::new(AtomicUsize::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        event_loop.set_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
            panic!("poisoned integration activity handler");
        });

        let targeted_count = Arc::new(AtomicUsize::new(0));
        let targeted_count_clone = Arc::clone(&targeted_count);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Multiple(Box::new(move |_| {
                targeted_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );
        event_send.send(EventDispatch::sink_change()).unwrap();
        event_send.send(EventDispatch::sink_change()).unwrap();

        assert!(event_loop.handle_pending_events());
        assert_eq!(activity_count.load(Ordering::Relaxed), 1);
        assert_eq!(targeted_count.load(Ordering::Relaxed), 2);
        assert!(event_loop.event_activity_handler.lock().unwrap().is_none());
    }

    #[test]
    fn panicking_activity_handler_does_not_prevent_close() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let activity_count = Arc::new(AtomicUsize::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        event_loop.set_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
            panic!("poisoned integration activity handler");
        });

        let close_handler_count = Arc::new(AtomicUsize::new(0));
        let close_handler_count_clone = Arc::clone(&close_handler_count);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                close_handler_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );
        event_send
            .send(EventDispatch::state_change(AudioContextState::Closed))
            .unwrap();
        event_send.send(EventDispatch::sink_change()).unwrap();

        let (done_send, done_recv) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            event_loop.run();
            done_send.send(()).unwrap();
        });

        assert!(done_recv.recv_timeout(Duration::from_secs(1)).is_ok());
        assert_eq!(activity_count.load(Ordering::Relaxed), 1);
        assert_eq!(close_handler_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn joinable_graceful_stop_synthesizes_lost_closed_exactly_once() {
        let (event_send, event_recv) = crossbeam_channel::bounded(1);
        event_send.send(EventDispatch::sink_change()).unwrap();
        assert!(matches!(
            event_send.try_send(EventDispatch::state_change(AudioContextState::Closed)),
            Err(crossbeam_channel::TrySendError::Full(_))
        ));

        let event_loop = EventLoop::new(event_recv);
        let queued_count = Arc::new(AtomicUsize::new(0));
        let queued_count_clone = Arc::clone(&queued_count);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                queued_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );
        let close_count = Arc::new(AtomicUsize::new(0));
        let close_count_clone = Arc::clone(&close_count);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |_| {
                close_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );

        let mut running = event_loop.run_joinable().unwrap();
        running.request_graceful_stop();
        let outcome = std::thread::spawn(move || running.join())
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(outcome, EventLoopExit::Graceful);

        assert_eq!(queued_count.load(Ordering::Relaxed), 1);
        assert_eq!(close_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn joinable_graceful_stop_drains_queued_events_before_final_closed() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        event_send.send(EventDispatch::sink_change()).unwrap();
        event_send.send(EventDispatch::sink_change()).unwrap();

        let event_loop = EventLoop::new(event_recv);
        let order = Arc::new(Mutex::new(Vec::new()));
        let event_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Multiple(Box::new(move |_| {
                event_order.lock().unwrap().push("event");
            })),
        );
        let close_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                close_order.lock().unwrap().push("closed");
            })),
        );

        let mut running = event_loop.run_joinable().unwrap();
        running.request_graceful_stop();
        assert_eq!(running.join().unwrap(), EventLoopExit::Graceful);

        assert_eq!(*order.lock().unwrap(), ["event", "event", "closed"]);
    }

    #[test]
    fn joinable_closed_is_deferred_until_queued_sentinel_is_dispatched() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        event_send
            .send(EventDispatch::state_change(AudioContextState::Closed))
            .unwrap();
        event_send.send(EventDispatch::sink_change()).unwrap();

        let event_loop = EventLoop::new(event_recv);
        let order = Arc::new(Mutex::new(Vec::new()));
        let sentinel_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                sentinel_order.lock().unwrap().push("sentinel");
            })),
        );
        let close_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                close_order.lock().unwrap().push("closed");
            })),
        );

        let mut running = event_loop.run_joinable().unwrap();
        running.request_graceful_stop();
        assert_eq!(running.join().unwrap(), EventLoopExit::Graceful);

        assert_eq!(*order.lock().unwrap(), ["sentinel", "closed"]);
    }

    #[test]
    fn joinable_duplicate_closed_records_are_coalesced() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        event_send
            .send(EventDispatch::state_change(AudioContextState::Closed))
            .unwrap();
        event_send
            .send(EventDispatch::state_change(AudioContextState::Closed))
            .unwrap();

        let event_loop = EventLoop::new(event_recv);
        let close_count = Arc::new(AtomicUsize::new(0));
        let close_count_clone = Arc::clone(&close_count);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Multiple(Box::new(move |_| {
                close_count_clone.fetch_add(1, Ordering::Relaxed);
            })),
        );

        let mut running = event_loop.run_joinable().unwrap();
        running.request_graceful_stop();
        assert_eq!(running.join().unwrap(), EventLoopExit::Graceful);

        assert_eq!(close_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn joinable_graceful_drain_includes_handler_reentrant_records() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        event_send.send(EventDispatch::sink_change()).unwrap();

        let event_loop = EventLoop::new(event_recv);
        let order = Arc::new(Mutex::new(Vec::new()));
        let reentrant_send = event_send.clone();
        let first_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                first_order.lock().unwrap().push("first");
                reentrant_send
                    .send(EventDispatch::ended(AudioNodeId(42)))
                    .unwrap();
            })),
        );
        let reentrant_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::Ended(AudioNodeId(42)),
            EventHandler::Once(Box::new(move |_| {
                reentrant_order.lock().unwrap().push("reentrant");
            })),
        );
        let close_order = Arc::clone(&order);
        event_loop.set_handler(
            EventType::StateChange,
            EventHandler::Once(Box::new(move |_| {
                close_order.lock().unwrap().push("closed");
            })),
        );

        let mut running = event_loop.run_joinable().unwrap();
        running.request_graceful_stop();
        assert_eq!(running.join().unwrap(), EventLoopExit::Graceful);

        assert_eq!(*order.lock().unwrap(), ["first", "reentrant", "closed"]);
    }

    #[test]
    fn joinable_disconnected_queue_waits_for_explicit_stop_mode() {
        for graceful in [true, false] {
            let (event_send, event_recv) = crossbeam_channel::unbounded();
            let event_loop = EventLoop::new(event_recv);
            let close_count = Arc::new(AtomicUsize::new(0));
            let close_count_clone = Arc::clone(&close_count);
            event_loop.set_handler(
                EventType::StateChange,
                EventHandler::Once(Box::new(move |_| {
                    close_count_clone.fetch_add(1, Ordering::Relaxed);
                })),
            );
            drop(event_send);

            let mut running = event_loop.run_joinable().unwrap();
            let expected = if graceful {
                running.request_graceful_stop();
                EventLoopExit::Graceful
            } else {
                running.request_silent_stop();
                EventLoopExit::Silent
            };
            assert_eq!(running.join().unwrap(), expected);
            assert_eq!(close_count.load(Ordering::Relaxed), usize::from(graceful));
        }
    }

    #[test]
    fn joinable_silent_stop_dispatches_nothing_and_leaves_payload_queued() {
        struct DropProbe(Arc<std::sync::atomic::AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let dispatch_count = Arc::new(AtomicUsize::new(0));
        let dispatch_count_clone = Arc::clone(&dispatch_count);
        event_loop.set_activity_handler(move || {
            dispatch_count_clone.fetch_add(1, Ordering::Relaxed);
        });
        let payload_dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut running = event_loop.run_joinable().unwrap();
        running.request_silent_stop();
        event_send
            .send(EventDispatch::message(
                AudioNodeId(42),
                Box::new(DropProbe(Arc::clone(&payload_dropped))),
            ))
            .unwrap();
        assert_eq!(running.join().unwrap(), EventLoopExit::Silent);

        assert_eq!(dispatch_count.load(Ordering::Relaxed), 0);
        assert!(!payload_dropped.load(Ordering::Acquire));
        drop(event_send);
        drop(event_loop);
        assert!(payload_dropped.load(Ordering::Acquire));
    }

    #[test]
    fn joinable_join_surfaces_targeted_handler_panic() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                panic!("joinable event handler panic");
            })),
        );
        event_send.send(EventDispatch::sink_change()).unwrap();

        let error = event_loop.run_joinable().unwrap().join().unwrap_err();
        let EventLoopJoinError::Panicked(payload) = error else {
            panic!("event loop reported self-join instead of handler panic");
        };
        assert_eq!(
            payload.downcast_ref::<&'static str>(),
            Some(&"joinable event handler panic")
        );
    }

    #[test]
    fn joinable_self_join_is_rejected_without_deadlock() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let owner = Arc::new(Mutex::new(None::<JoinableEventLoop>));
        let owner_for_handler = Arc::clone(&owner);
        let (result_send, result_recv) = crossbeam_channel::bounded(1);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                let running = owner_for_handler.lock().unwrap().take().unwrap();
                result_send
                    .send(matches!(
                        running.join(),
                        Err(EventLoopJoinError::CurrentThread)
                    ))
                    .unwrap();
            })),
        );

        *owner.lock().unwrap() = Some(event_loop.run_joinable().unwrap());
        event_send.send(EventDispatch::sink_change()).unwrap();

        assert_eq!(result_recv.recv_timeout(Duration::from_secs(1)), Ok(true));
    }

    #[test]
    fn dropping_joinable_owner_requests_silent_stop_without_acknowledging_it() {
        let (event_send, event_recv) = crossbeam_channel::unbounded();
        let event_loop = EventLoop::new(event_recv);
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let (finished_send, finished_recv) = crossbeam_channel::bounded(1);
        event_loop.set_handler(
            EventType::SinkChange,
            EventHandler::Once(Box::new(move |_| {
                entered_send.send(()).unwrap();
                release_recv.recv().unwrap();
                finished_send.send(()).unwrap();
            })),
        );
        event_send.send(EventDispatch::sink_change()).unwrap();

        let running = event_loop.run_joinable().unwrap();
        entered_recv.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(running);
        assert!(finished_recv.try_recv().is_err());
        release_send.send(()).unwrap();
        finished_recv.recv_timeout(Duration::from_secs(1)).unwrap();
    }
}
