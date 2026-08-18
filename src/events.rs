#[cfg(feature = "diagnostics")]
use crate::context::AudioContextDiagnostics;
use crate::context::ConcreteBaseAudioContext;
use crate::context::{AudioContextState, AudioNodeId};
use crate::{AudioBuffer, AudioRenderCapacityEvent};

use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;

type EventActivityHandler = dyn Fn() + Send + Sync + 'static;

/// The Event interface
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Event {
    pub type_: &'static str,
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub(crate) enum EventType {
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
            event_sender: render_init.event_send,
        };
        completion.mark_complete_and_wake(&scope);

        assert!(completion.is_complete());
        assert_eq!(roots.lock().unwrap().len(), 1);
        assert!(event_loop.handle_pending_events());
        assert!(roots.lock().unwrap().is_empty());
        assert_eq!(activity_count.load(Ordering::Relaxed), queued as usize);
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
            event_sender: event_send,
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
}
