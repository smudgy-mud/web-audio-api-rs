use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::AudioNode;
use crate::events::{Event, EventHandler, EventType};
use crate::render::AudioWorkletGlobalScope;

/// A cloneable handle for observing a scheduled source's terminal state.
///
/// A token is scoped to one scheduled-source object. It is initially incomplete
/// and changes to complete exactly once, before the render thread attempts to
/// deliver that source's `ended` event. Completion is monotonic, so consumers
/// can use the token to recover from a dropped best-effort event notification.
///
/// Cloning a token does not create a new completion state; all clones observe
/// the same source.
#[derive(Clone, Debug)]
pub struct ScheduledSourceCompletionToken {
    complete: Arc<AtomicBool>,
}

impl ScheduledSourceCompletionToken {
    pub(crate) fn new() -> Self {
        Self {
            complete: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn mark_complete(&self) {
        self.complete.store(true, Ordering::Release);
    }

    pub(crate) fn mark_complete_and_wake(&self, scope: &AudioWorkletGlobalScope) {
        self.mark_complete();
        scope.send_ended_event();
    }

    /// Returns whether the associated source has reached its terminal state.
    ///
    /// Once this method returns `true`, it will return `true` for this token and
    /// every clone for the remainder of their lifetimes.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

/// Interface of source nodes, controlling start and stop times.
/// The node will emit silence before it is started, and after it has ended.
pub trait AudioScheduledSourceNode: AudioNode {
    /// Play immediately
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    fn start(&mut self);

    /// Schedule playback start at given timestamp
    ///
    /// # Panics
    ///
    /// Panics if the source was already started
    fn start_at(&mut self, when: f64);

    /// Stop immediately
    ///
    /// # Panics
    ///
    /// Panics if the source was not started yet
    fn stop(&mut self);

    /// Schedule playback stop at given timestamp
    ///
    /// # Panics
    ///
    /// Panics if the source was not started yet
    fn stop_at(&mut self, when: f64);

    /// Register callback to run when the source node has stopped playing
    ///
    /// For all [`AudioScheduledSourceNode`]s, the ended event is dispatched when the stop time
    /// determined by stop() is reached. For an
    /// [`AudioBufferSourceNode`](crate::node::AudioBufferSourceNode), the event is also dispatched
    /// because the duration has been reached or if the entire buffer has been played.
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    fn set_onended<F: FnOnce(Event) + Send + 'static>(&self, callback: F) {
        let callback = move |_| callback(Event { type_: "ended" });

        self.context().set_event_handler(
            EventType::Ended(self.registration().id()),
            EventHandler::Once(Box::new(callback)),
        );
    }

    /// Unset the callback to run when the source node has stopped playing
    fn clear_onended(&self) {
        self.context()
            .clear_event_handler(EventType::Ended(self.registration().id()));
    }
}

/// Native completion-state access for built-in scheduled source nodes.
///
/// This extension trait is separate from [`AudioScheduledSourceNode`] so that
/// third-party scheduled-source implementations are not required to provide a
/// completion token. It is not part of the Web Audio API.
pub trait AudioScheduledSourceNodeExt: AudioScheduledSourceNode {
    /// Returns a token that tracks this source's terminal state.
    ///
    /// This method does not register or consume an `ended` event handler.
    fn completion_token(&self) -> ScheduledSourceCompletionToken;
}

#[cfg(test)]
mod tests {
    use crate::context::{AudioContextRegistration, BaseAudioContext, OfflineAudioContext};
    use crate::node::{
        AudioNode, AudioScheduledSourceNode, AudioScheduledSourceNodeExt, ChannelConfig,
        ScheduledSourceCompletionToken,
    };

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    enum ConcreteAudioScheduledSourceNode {
        Buffer(crate::node::AudioBufferSourceNode),
        Constant(crate::node::ConstantSourceNode),
        Oscillator(crate::node::OscillatorNode),
    }
    use ConcreteAudioScheduledSourceNode::*;

    impl AudioNode for ConcreteAudioScheduledSourceNode {
        fn registration(&self) -> &AudioContextRegistration {
            match self {
                Buffer(n) => n.registration(),
                Constant(n) => n.registration(),
                Oscillator(n) => n.registration(),
            }
        }

        fn channel_config(&self) -> &ChannelConfig {
            match self {
                Buffer(n) => n.channel_config(),
                Constant(n) => n.channel_config(),
                Oscillator(n) => n.channel_config(),
            }
        }

        fn number_of_inputs(&self) -> usize {
            match self {
                Buffer(n) => n.number_of_inputs(),
                Constant(n) => n.number_of_inputs(),
                Oscillator(n) => n.number_of_inputs(),
            }
        }

        fn number_of_outputs(&self) -> usize {
            match self {
                Buffer(n) => n.number_of_outputs(),
                Constant(n) => n.number_of_outputs(),
                Oscillator(n) => n.number_of_outputs(),
            }
        }
    }

    impl AudioScheduledSourceNode for ConcreteAudioScheduledSourceNode {
        fn start(&mut self) {
            match self {
                Buffer(n) => n.start(),
                Constant(n) => n.start(),
                Oscillator(n) => n.start(),
            }
        }

        fn start_at(&mut self, when: f64) {
            match self {
                Buffer(n) => n.start_at(when),
                Constant(n) => n.start_at(when),
                Oscillator(n) => n.start_at(when),
            }
        }

        fn stop(&mut self) {
            match self {
                Buffer(n) => n.stop(),
                Constant(n) => n.stop(),
                Oscillator(n) => n.stop(),
            }
        }

        fn stop_at(&mut self, when: f64) {
            match self {
                Buffer(n) => n.stop_at(when),
                Constant(n) => n.stop_at(when),
                Oscillator(n) => n.stop_at(when),
            }
        }
    }

    impl AudioScheduledSourceNodeExt for ConcreteAudioScheduledSourceNode {
        fn completion_token(&self) -> ScheduledSourceCompletionToken {
            match self {
                Buffer(n) => n.completion_token(),
                Constant(n) => n.completion_token(),
                Oscillator(n) => n.completion_token(),
            }
        }
    }

    #[test]
    fn completion_mark_and_observe_are_allocation_free() {
        let completion = ScheduledSourceCompletionToken::new();

        alloc_counter::deny_alloc(|| {
            completion.mark_complete();
            assert!(completion.is_complete());
        });
    }

    #[test]
    fn completion_survives_saturated_bounded_event_channel() {
        use std::cell::Cell;

        use crate::context::AudioNodeId;
        use crate::events::EventDispatch;
        use crate::render::AudioWorkletGlobalScope;

        let (control_init, render_init) = crate::io::thread_init();
        let mut queued = 0;
        while render_init
            .event_send
            .try_send(EventDispatch::ended(AudioNodeId(queued)))
            .is_ok()
        {
            queued += 1;
        }
        assert_eq!(queued, 256);

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48_000.,
            node_id: Cell::new(AudioNodeId(queued)),
            event_sender: render_init.event_send.into(),
        };
        let completion = ScheduledSourceCompletionToken::new();

        alloc_counter::deny_alloc(|| completion.mark_complete_and_wake(&scope));

        assert!(completion.is_complete());
        assert_eq!(control_init.event_recv.len(), queued as usize);
    }

    fn run_ended_event(f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode) {
        let mut context = OfflineAudioContext::new(2, 44_100, 44_100.);
        let mut src = f(&context);
        let completion = src.completion_token();
        let callback_completion = completion.clone();
        assert!(!completion.is_complete());
        src.start_at(0.);
        src.stop_at(0.5);

        let ended = Arc::new(AtomicUsize::new(0));
        let ended_clone = Arc::clone(&ended);
        src.set_onended(move |_event| {
            assert!(callback_completion.is_complete());
            ended_clone.fetch_add(1, Ordering::Relaxed);
        });

        let _ = context.start_rendering_sync();
        assert!(completion.is_complete());
        assert!(src.completion_token().is_complete());
        assert_eq!(ended.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_ended_event_constant_source() {
        run_ended_event(|c| Constant(c.create_constant_source()));
    }
    #[test]
    fn test_ended_event_buffer_source() {
        run_ended_event(|c| Buffer(c.create_buffer_source()));
    }
    #[test]
    fn test_ended_event_oscillator() {
        run_ended_event(|c| Oscillator(c.create_oscillator()));
    }

    fn run_no_ended_event(
        f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode,
    ) {
        let mut context = OfflineAudioContext::new(2, 44_100, 44_100.);
        let src = f(&context);
        let completion = src.completion_token();
        assert!(!completion.is_complete());

        // do not start the node

        let ended = Arc::new(AtomicBool::new(false));
        let ended_clone = Arc::clone(&ended);
        src.set_onended(move |_event| {
            ended_clone.store(true, Ordering::Relaxed);
        });

        let _ = context.start_rendering_sync();
        assert!(!ended.load(Ordering::Relaxed)); // should not have triggered
        assert!(!completion.is_complete());
    }

    #[test]
    fn test_no_ended_event_constant_source() {
        run_no_ended_event(|c| Constant(c.create_constant_source()));
    }
    #[test]
    fn test_no_ended_event_buffer_source() {
        run_no_ended_event(|c| Buffer(c.create_buffer_source()));
    }
    #[test]
    fn test_no_ended_event_oscillator() {
        run_no_ended_event(|c| Oscillator(c.create_oscillator()));
    }

    fn run_exact_ended_event(
        f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode,
    ) {
        let mut context = OfflineAudioContext::new(2, 44_100, 44_100.);
        let mut src = f(&context);
        let completion = src.completion_token();
        assert!(!completion.is_complete());
        src.start_at(0.);
        src.stop_at(1.); // end right at the end of the offline buffer

        let ended = Arc::new(AtomicBool::new(false));
        let ended_clone = Arc::clone(&ended);
        src.set_onended(move |_event| {
            ended_clone.store(true, Ordering::Relaxed);
        });

        let _ = context.start_rendering_sync();
        assert!(ended.load(Ordering::Relaxed));
        assert!(completion.is_complete());
    }

    #[test]
    fn test_exact_ended_event_constant_source() {
        run_exact_ended_event(|c| Constant(c.create_constant_source()));
    }
    #[test]
    fn test_exact_ended_event_buffer_source() {
        run_exact_ended_event(|c| Buffer(c.create_buffer_source()));
    }
    #[test]
    fn test_exact_ended_event_oscillator() {
        run_exact_ended_event(|c| Oscillator(c.create_oscillator()));
    }

    fn run_implicit_ended_event(
        f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode,
    ) {
        let mut context = OfflineAudioContext::new(2, 44_100, 44_100.);
        let mut src = f(&context);
        let completion = src.completion_token();
        assert!(!completion.is_complete());
        src.start_at(0.);
        // no explicit stop, so we stop at end of offline context

        let ended = Arc::new(AtomicBool::new(false));
        let ended_clone = Arc::clone(&ended);
        src.set_onended(move |_event| {
            ended_clone.store(true, Ordering::Relaxed);
        });

        let _ = context.start_rendering_sync();
        assert!(ended.load(Ordering::Relaxed));
        assert!(completion.is_complete());
    }

    #[test]
    fn test_implicit_ended_event_constant_source() {
        run_implicit_ended_event(|c| Constant(c.create_constant_source()));
    }

    #[test]
    fn test_implicit_ended_event_buffer_source() {
        run_implicit_ended_event(|c| Buffer(c.create_buffer_source()));
    }

    #[test]
    fn test_implicit_ended_event_oscillator() {
        run_implicit_ended_event(|c| Oscillator(c.create_oscillator()));
    }

    fn run_stop_before_start_time(
        f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode,
    ) {
        let sample_rate = 48_000.;
        let mut context = OfflineAudioContext::new(1, 128, sample_rate);
        let mut src = f(&context);
        let completion = src.completion_token();
        src.start_at(96. / f64::from(sample_rate));
        src.stop_at(32. / f64::from(sample_rate));

        let _ = context.start_rendering_sync();

        assert!(completion.is_complete());
    }

    #[test]
    fn completion_tracks_stop_before_start_time_for_constant_source() {
        run_stop_before_start_time(|c| Constant(c.create_constant_source()));
    }

    #[test]
    fn completion_tracks_stop_before_start_time_for_buffer_source() {
        run_stop_before_start_time(|c| Buffer(c.create_buffer_source()));
    }

    #[test]
    fn completion_tracks_stop_before_start_time_for_oscillator() {
        run_stop_before_start_time(|c| Oscillator(c.create_oscillator()));
    }

    #[test]
    fn completion_is_observable_for_large_terminal_batch() {
        const SOURCE_COUNT: usize = 300;

        let mut context = OfflineAudioContext::new(1, 128, 48_000.);
        let completions: Vec<_> = (0..SOURCE_COUNT)
            .map(|_| {
                let mut source = context.create_oscillator();
                let completion = source.completion_token();
                source.start_at(0.);
                source.stop_at(0.);
                completion
            })
            .collect();

        assert!(completions.iter().all(|token| !token.is_complete()));
        let _ = context.start_rendering_sync();
        assert!(completions.iter().all(|token| token.is_complete()));
    }

    fn run_start_twice(f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode) {
        let context = OfflineAudioContext::new(2, 1, 44_100.);
        let mut src = f(&context);
        src.start();
        src.start();
    }

    #[test]
    #[should_panic]
    fn test_start_twice_constant_source() {
        run_start_twice(|c| Constant(c.create_constant_source()));
    }

    #[test]
    #[should_panic]
    fn test_start_twice_buffer_source() {
        run_start_twice(|c| Buffer(c.create_buffer_source()));
    }

    #[test]
    #[should_panic]
    fn test_start_twice_oscillator() {
        run_start_twice(|c| Oscillator(c.create_oscillator()));
    }

    fn run_stop_before_start(
        f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode,
    ) {
        let context = OfflineAudioContext::new(2, 1, 44_100.);
        let mut src = f(&context);
        src.stop();
    }

    #[test]
    #[should_panic]
    fn test_stop_before_start_constant_source() {
        run_stop_before_start(|c| Constant(c.create_constant_source()));
    }

    #[test]
    #[should_panic]
    fn test_stop_before_start_buffer_source() {
        run_stop_before_start(|c| Buffer(c.create_buffer_source()));
    }

    #[test]
    #[should_panic]
    fn test_stop_before_start_oscillator() {
        run_stop_before_start(|c| Oscillator(c.create_oscillator()));
    }

    fn run_stop_twice(f: impl FnOnce(&OfflineAudioContext) -> ConcreteAudioScheduledSourceNode) {
        // is allowed, see https://github.com/orottier/web-audio-api-rs/issues/579
        let context = OfflineAudioContext::new(2, 1, 44_100.);
        let mut src = f(&context);
        src.start();
        src.stop();
        src.stop();
    }

    #[test]
    fn test_stop_twice_allowed_constant_source() {
        run_stop_twice(|c| Constant(c.create_constant_source()));
    }
    #[test]
    fn test_stop_twice_allowed_buffer_source() {
        run_stop_twice(|c| Buffer(c.create_buffer_source()));
    }
    #[test]
    fn test_stop_twice_allowed_oscillator() {
        run_stop_twice(|c| Oscillator(c.create_oscillator()));
    }
}
