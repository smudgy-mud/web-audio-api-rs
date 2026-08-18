//! Communicates with the control thread and ships audio samples to the hardware

use std::any::Any;
use std::cell::Cell;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use dasp_sample::FromSample;
use futures_channel::{mpsc, oneshot};
use futures_util::StreamExt as _;

use super::AudioRenderQuantum;
use crate::buffer::AudioBuffer;
#[cfg(feature = "diagnostics")]
use crate::context::{AudioContextDiagnostics, AudioRenderThreadDiagnostics};
use crate::context::{
    AudioContextState, AudioNodeId, OfflineAudioContext, OfflineAudioContextCallback,
};
use crate::events::{EventDispatch, EventLoop};
use crate::message::{
    control_batch_storage, control_batch_storage_mut, ControlBatchApplied, ControlBatchNode,
    ControlMessage, CONTROL_COMMANDS_PER_CALLBACK,
};
use crate::node::ChannelInterpretation;
use crate::render::AudioWorkletGlobalScope;
use crate::stats::AudioStats;
use crate::RENDER_QUANTUM_SIZE;

use super::graph::Graph;

/// Operations running off the system-level audio callback
pub(crate) struct RenderThread {
    graph: Option<Graph>,
    sample_rate: f32,
    buffer_size: usize,
    /// number of channels of the backend stream, i.e. sound card number of
    /// channels clamped to MAX_CHANNELS
    number_of_channels: usize,
    suspended: bool,
    state: Arc<AtomicU8>,
    startup_pending: Option<Arc<AtomicBool>>,
    frames_played: Arc<AtomicU64>,
    receiver: Option<Receiver<ControlMessage>>,
    /// Preallocated carrier used to retire the receiver and any queued batches off the callback.
    receiver_retirement: Option<llq::Node<Box<dyn Any + Send>>>,
    buffer_offset: Option<(usize, AudioRenderQuantum)>,
    stats: AudioStats,
    event_sender: Sender<EventDispatch>,
    garbage_collector: Option<llq::Producer<Box<dyn Any + Send>>>,
    /// Preallocated poison record; render-side shutdown must not allocate its GC notification.
    garbage_collector_termination: Option<llq::Node<Box<dyn Any + Send>>>,
    control_batch_applied: ControlBatchApplied,
    pending_control_batch: Option<ControlBatchNode>,
}

// SAFETY:
// The RenderThread is not Send/Sync since it contains `AudioRenderQuantum`s (which use Rc), but
// these are only accessed within the same thread (the render thread). Due to the cpal constraints
// we can neither move the RenderThread object into the render thread, nor can we initialize the
// Rc's in that thread.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}
unsafe impl Send for RenderThread {}
unsafe impl Sync for RenderThread {}

impl std::fmt::Debug for RenderThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderThread")
            .field("sample_rate", &self.sample_rate)
            .field("buffer_size", &self.buffer_size)
            .field("frames_played", &self.frames_played.load(Ordering::Relaxed))
            .field("number_of_channels", &self.number_of_channels)
            .finish_non_exhaustive()
    }
}

impl RenderThread {
    #[cfg(feature = "diagnostics")]
    fn diagnostics(&self) -> AudioRenderThreadDiagnostics {
        AudioRenderThreadDiagnostics {
            sample_rate: self.sample_rate,
            buffer_size: self.buffer_size,
            frames_played: self.frames_played.load(Ordering::Relaxed),
            number_of_channels: self.number_of_channels,
            suspended: self.suspended,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f32,
        number_of_channels: usize,
        receiver: Receiver<ControlMessage>,
        state: Arc<AtomicU8>,
        frames_played: Arc<AtomicU64>,
        stats: AudioStats,
        event_sender: Sender<EventDispatch>,
        control_batch_applied: ControlBatchApplied,
    ) -> Self {
        Self {
            graph: None,
            sample_rate,
            buffer_size: 0,
            number_of_channels,
            suspended: false,
            state,
            startup_pending: None,
            frames_played,
            receiver: Some(receiver),
            receiver_retirement: Some(llq::Node::new(Box::new(ControlReceiverRetirement(None)))),
            buffer_offset: None,
            stats,
            event_sender,
            garbage_collector: None,
            garbage_collector_termination: Some(llq::Node::new(Box::new(
                TerminateGarbageCollectorThread,
            ))),
            control_batch_applied,
            pending_control_batch: None,
        }
    }

    pub(crate) fn set_startup_pending(&mut self, startup_pending: Arc<AtomicBool>) {
        self.startup_pending = Some(startup_pending);
    }

    pub(crate) fn spawn_garbage_collector_thread(&mut self) {
        if self.garbage_collector.is_none() {
            let (gc_producer, gc_consumer) = llq::Queue::new().split();
            spawn_garbage_collector_thread(gc_consumer);
            self.garbage_collector = Some(gc_producer);
        }
    }

    #[inline(always)]
    fn handle_control_messages(&mut self) {
        let mut unlimited = usize::MAX;
        self.handle_control_messages_with_budget(&mut unlimited);
    }

    #[inline(always)]
    fn handle_control_messages_with_budget(&mut self, remaining: &mut usize) {
        while *remaining > 0 && self.receiver.is_some() {
            if self.pending_control_batch.is_none() {
                let Ok(message) = self.receiver.as_ref().unwrap().try_recv() else {
                    return;
                };

                if let ControlMessage::Batch(batch) = message {
                    let storage = control_batch_storage(&batch);
                    let expected = self.control_batch_applied.load().checked_add(1);
                    if storage.is_empty() || Some(storage.sequence()) != expected {
                        self.reclaim_control_batch(batch);
                        continue;
                    }
                    self.pending_control_batch = Some(batch);
                } else {
                    *remaining -= 1;
                    if self.handle_control_message(message).is_break() {
                        return;
                    }
                    continue;
                }
            }

            while *remaining > 0 {
                let message = control_batch_storage_mut(
                    self.pending_control_batch
                        .as_mut()
                        .expect("batch is pending"),
                )
                .take_next()
                .expect("validated batch has a command remaining");
                *remaining -= 1;

                if self.handle_control_message(message).is_break() {
                    let batch = self.pending_control_batch.take().unwrap();
                    self.reclaim_control_batch(batch);
                    return;
                }

                if control_batch_storage(
                    self.pending_control_batch
                        .as_ref()
                        .expect("batch is pending"),
                )
                .is_complete()
                {
                    let batch = self.pending_control_batch.take().unwrap();
                    let sequence = control_batch_storage(&batch).sequence();
                    self.control_batch_applied.publish(sequence);
                    self.event_sender
                        .try_send(EventDispatch::control_batch_activity())
                        .ok();
                    self.reclaim_control_batch(batch);
                    break;
                }
            }
        }
    }

    #[inline]
    fn reclaim_control_batch(&mut self, batch: ControlBatchNode) {
        if let Some(gc) = self.garbage_collector.as_mut() {
            gc.push(batch);
        }
    }

    fn retire_control_receiver(&mut self) {
        let Some(receiver) = self.receiver.take() else {
            return;
        };
        let mut retirement = self
            .receiver_retirement
            .take()
            .expect("control receiver retirement carrier is single-use");
        retirement
            .as_mut()
            .downcast_mut::<ControlReceiverRetirement>()
            .expect("private retirement carrier has the expected type")
            .0 = Some(receiver);
        if let Some(gc) = self.garbage_collector.as_mut() {
            gc.push(retirement);
        }
    }

    fn handle_control_message(&mut self, msg: ControlMessage) -> ControlFlow<()> {
        use ControlMessage::*;

        match msg {
            Batch(_) => unreachable!("batch envelopes are handled before individual commands"),
            RegisterNode {
                id: node_id,
                reclaim_id,
                node,
                inputs,
                outputs,
                channel_config,
            } => {
                self.graph.as_mut().unwrap().add_node(
                    node_id,
                    reclaim_id,
                    node,
                    inputs,
                    outputs,
                    channel_config,
                );
            }
            ConnectNode {
                from,
                to,
                output,
                input,
            } => {
                self.graph
                    .as_mut()
                    .unwrap()
                    .add_edge((from, output), (to, input));
            }
            DisconnectNode {
                from,
                output,
                to,
                input,
            } => {
                self.graph
                    .as_mut()
                    .unwrap()
                    .remove_edge((from, output), (to, input));
            }
            ControlHandleDropped { id } => {
                self.graph.as_mut().unwrap().mark_control_handle_dropped(id);
            }
            MarkCycleBreaker { id } => {
                self.graph.as_mut().unwrap().mark_cycle_breaker(id);
            }
            CloseAndRecycle { sender } => {
                self.set_state(AudioContextState::Suspended);
                let _ = sender.send(self.graph.take().unwrap());
                self.retire_control_receiver();
                return ControlFlow::Break(()); // no further handling of ctrl msgs
            }
            Startup { graph } => {
                debug_assert!(self.graph.is_none());
                self.graph = Some(graph);
                if let Some(startup_pending) = self.startup_pending.as_ref() {
                    startup_pending.store(false, Ordering::Release);
                }
                self.set_state(AudioContextState::Running);
            }
            NodeMessage { id, mut msg } => {
                self.graph.as_mut().unwrap().route_message(id, msg.as_mut());
                if let Some(gc) = self.garbage_collector.as_mut() {
                    gc.push(msg)
                }
            }
            #[cfg(feature = "diagnostics")]
            RunDiagnostics { backend } => {
                let diagnostics = AudioContextDiagnostics {
                    backend,
                    render_thread: self.diagnostics(),
                    graph: self.graph.as_ref().unwrap().diagnostics(),
                };
                self.event_sender
                    .try_send(EventDispatch::diagnostics(diagnostics))
                    .expect("Unable to send diagnostics - channel is full");
            }
            Suspend { notify } => {
                self.suspended = true;
                self.set_state(AudioContextState::Suspended);
                notify.send();
            }
            Resume { notify } => {
                self.suspended = false;
                self.set_state(AudioContextState::Running);
                notify.send();
            }
            Close { notify } => {
                self.suspended = true;
                self.set_state(AudioContextState::Closed);
                notify.send();
                self.retire_control_receiver();
                return ControlFlow::Break(());
            }

            SetChannelCount { id, count } => {
                self.graph.as_mut().unwrap().set_channel_count(id, count);
            }

            SetChannelCountMode { id, mode } => {
                self.graph
                    .as_mut()
                    .unwrap()
                    .set_channel_count_mode(id, mode);
            }

            SetChannelInterpretation { id, interpretation } => {
                self.graph
                    .as_mut()
                    .unwrap()
                    .set_channel_interpretation(id, interpretation);
            }

            #[cfg(test)]
            TestMarker { value, log } => log.lock().unwrap().push(value),

            #[cfg(test)]
            TestNop => {}
        }

        ControlFlow::Continue(()) // continue handling more messages
    }

    // Render method of the `OfflineAudioContext::start_rendering_sync`
    //
    // This method is not spec compliant and obviously marked as synchronous, so we
    // don't launch a thread.
    //
    // cf. https://webaudio.github.io/web-audio-api/#dom-offlineaudiocontext-startrendering
    pub fn render_audiobuffer_sync(
        mut self,
        context: &mut OfflineAudioContext,
        mut suspend_callbacks: Vec<(usize, Box<OfflineAudioContextCallback>)>,
        event_loop: &EventLoop,
    ) -> AudioBuffer {
        let length = context.length();
        let sample_rate = self.sample_rate;

        // construct a properly sized output buffer
        let mut buffer = Vec::with_capacity(self.number_of_channels);
        buffer.resize_with(buffer.capacity(), || Vec::with_capacity(length));

        let num_frames = length.div_ceil(RENDER_QUANTUM_SIZE);

        // Handle initial control messages
        self.handle_control_messages();

        for quantum in 0..num_frames {
            // Suspend at given times and run callbacks
            if suspend_callbacks.first().map(|&(q, _)| q) == Some(quantum) {
                let callback = suspend_callbacks.remove(0).1;
                (callback)(context);

                // Handle any control messages that may have been submitted by the callback
                self.handle_control_messages();
            }

            self.render_offline_quantum(&mut buffer);

            let events_were_handled = event_loop.handle_pending_events();
            if events_were_handled {
                // Handle any control messages that may have been submitted by the handler
                self.handle_control_messages();
            }
        }

        // call destructors of all alive nodes and handle any resulting events
        self.unload_graph();
        event_loop.handle_pending_events();

        AudioBuffer::from(buffer, sample_rate)
    }

    // Render method of the `OfflineAudioContext::start_rendering`
    //
    // This is the async interface, as compared to render_audiobuffer_sync
    //
    // cf. https://webaudio.github.io/web-audio-api/#dom-offlineaudiocontext-startrendering
    pub async fn render_audiobuffer(
        mut self,
        length: usize,
        mut suspend_callbacks: Vec<(usize, oneshot::Sender<()>)>,
        mut resume_receiver: mpsc::Receiver<()>,
        event_loop: &EventLoop,
    ) -> AudioBuffer {
        let sample_rate = self.sample_rate;

        // construct a properly sized output buffer
        let mut buffer = Vec::with_capacity(self.number_of_channels);
        buffer.resize_with(buffer.capacity(), || Vec::with_capacity(length));

        let num_frames = length.div_ceil(RENDER_QUANTUM_SIZE);

        // Handle addition/removal of nodes/edges
        self.handle_control_messages();

        for quantum in 0..num_frames {
            // Suspend at given times and run callbacks
            if suspend_callbacks.first().map(|&(q, _)| q) == Some(quantum) {
                let sender = suspend_callbacks.remove(0).1;
                sender.send(()).unwrap();
                resume_receiver.next().await;

                // Handle addition/removal of nodes/edges
                self.handle_control_messages();
            }

            self.render_offline_quantum(&mut buffer);

            let events_were_handled = event_loop.handle_pending_events();
            if events_were_handled {
                // Handle any control messages that may have been submitted by the handler
                self.handle_control_messages();
            }
        }

        // call destructors of all alive nodes and handle any resulting events
        self.unload_graph();
        event_loop.handle_pending_events();

        AudioBuffer::from(buffer, sample_rate)
    }

    /// Render a single quantum into an AudioBuffer
    fn render_offline_quantum(&mut self, buffer: &mut [Vec<f32>]) {
        // Update time
        let current_frame = self
            .frames_played
            .fetch_add(RENDER_QUANTUM_SIZE as u64, Ordering::Relaxed);
        let current_time = current_frame as f64 / self.sample_rate as f64;

        let scope = AudioWorkletGlobalScope {
            current_frame,
            current_time,
            sample_rate: self.sample_rate,
            event_sender: self.event_sender.clone(),
            node_id: Cell::new(AudioNodeId(0)), // placeholder value
        };

        // Render audio graph
        let graph = self.graph.as_mut().unwrap();

        // For x64 and aarch, process with denormal floats disabled (for performance, #194)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        let rendered = unsafe {
            // SAFETY: potentially risky - "modifying the masking flags, rounding mode, or
            // denormals-are-zero mode flags leads to immediate Undefined Behavior: Rust assumes
            // that these are always in their default state and will optimize accordingly."
            no_denormals::no_denormals(|| graph.render(&scope))
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        let rendered = graph.render(&scope);

        // Use a specialized copyToChannel implementation for performance
        let remaining = (buffer[0].capacity() - buffer[0].len()).min(RENDER_QUANTUM_SIZE);
        let channels = rendered.channels();
        buffer.iter_mut().enumerate().for_each(|(i, b)| {
            let c = channels
                .get(i)
                .map(AsRef::as_ref)
                // When there are no input nodes for the destination, only a single silent channel
                // is emitted. So manually pad the missing channels with silence
                .unwrap_or(&[0.; RENDER_QUANTUM_SIZE]);
            b.extend_from_slice(&c[..remaining]);
        });
    }

    /// Run destructors of all alive nodes in the audio graph
    fn unload_graph(mut self) {
        let current_frame = self.frames_played.load(Ordering::Relaxed);
        let current_time = current_frame as f64 / self.sample_rate as f64;

        let scope = AudioWorkletGlobalScope {
            current_frame,
            current_time,
            sample_rate: self.sample_rate,
            event_sender: self.event_sender.clone(),
            node_id: Cell::new(AudioNodeId(0)), // placeholder value
        };
        self.graph.take().unwrap().before_drop(&scope);
    }

    pub fn render<S: FromSample<f32> + Clone>(&mut self, output_buffer: &mut [S]) {
        // Collect timing information
        let render_start = Instant::now();
        let frames = output_buffer.len() / self.number_of_channels;

        // Perform actual rendering

        // For x64 and aarch, process with denormal floats disabled (for performance, #194)
        let mut control_budget = CONTROL_COMMANDS_PER_CALLBACK;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            // SAFETY: potentially risky - "modifying the masking flags, rounding mode, or
            // denormals-are-zero mode flags leads to immediate Undefined Behavior: Rust assumes
            // that these are always in their default state and will optimize accordingly."
            no_denormals::no_denormals(|| self.render_inner(output_buffer, &mut control_budget))
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        self.render_inner(output_buffer, &mut control_budget);

        let render_duration_ns = render_start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let callback_budget_ns = if self.sample_rate <= 0. {
            0
        } else {
            ((frames as f64 / self.sample_rate as f64) * 1_000_000_000.).round() as u64
        };
        self.stats
            .record_render_callback(render_duration_ns, callback_budget_ns);
    }

    fn render_inner<S: FromSample<f32> + Clone>(
        &mut self,
        mut output_buffer: &mut [S],
        control_budget: &mut usize,
    ) {
        self.buffer_size = output_buffer.len();

        // There may be audio frames left over from the previous render call,
        // if the cpal buffer size did not align with our internal RENDER_QUANTUM_SIZE
        if let Some((offset, prev_rendered)) = self.buffer_offset.take() {
            let leftover_len = (RENDER_QUANTUM_SIZE - offset) * self.number_of_channels;
            // split the leftover frames slice, to fit in `buffer`
            let (first, next) = output_buffer.split_at_mut(leftover_len.min(output_buffer.len()));

            // copy rendered audio into output slice
            for i in 0..self.number_of_channels {
                let output = first.iter_mut().skip(i).step_by(self.number_of_channels);
                let channel = prev_rendered.channel_data(i)[offset..].iter();
                for (sample, input) in output.zip(channel) {
                    let value = S::from_sample_(*input);
                    *sample = value;
                }
            }

            // exit early if we are done filling the buffer with the previously rendered data
            if next.is_empty() {
                self.buffer_offset = Some((
                    offset + first.len() / self.number_of_channels,
                    prev_rendered,
                ));
                return;
            }

            // if there's still space left in the buffer, continue rendering
            output_buffer = next;
        }

        // handle addition/removal of nodes/edges
        self.handle_control_messages_with_budget(control_budget);

        // if the thread is still booting, suspended, or shutting down, fill with silence
        if self.suspended || !self.graph.as_ref().is_some_and(Graph::is_active) {
            output_buffer.fill(S::from_sample_(0.));
            return;
        }

        // The audio graph is rendered in chunks of RENDER_QUANTUM_SIZE frames.  But some audio backends
        // may not be able to emit chunks of this size.
        let chunk_size = RENDER_QUANTUM_SIZE * self.number_of_channels;

        for data in output_buffer.chunks_mut(chunk_size) {
            // update time
            let current_frame = self
                .frames_played
                .fetch_add(RENDER_QUANTUM_SIZE as u64, Ordering::Relaxed);
            let current_time = current_frame as f64 / self.sample_rate as f64;

            let scope = AudioWorkletGlobalScope {
                current_frame,
                current_time,
                sample_rate: self.sample_rate,
                event_sender: self.event_sender.clone(),
                node_id: Cell::new(AudioNodeId(0)), // placeholder value
            };

            // render audio graph, clone it in case we need to mutate/store the value later
            let mut destination_buffer = self.graph.as_mut().unwrap().render(&scope).clone();

            // online AudioContext allows channel count to be less than the number
            // of channels of the backend stream, i.e. number of channels of the
            // soundcard clamped to MAX_CHANNELS.
            if destination_buffer.number_of_channels() < self.number_of_channels {
                destination_buffer.mix(self.number_of_channels, ChannelInterpretation::Discrete);
            }

            // copy rendered audio into output slice
            for i in 0..self.number_of_channels {
                let output = data.iter_mut().skip(i).step_by(self.number_of_channels);
                let channel = destination_buffer.channel_data(i).iter();
                for (sample, input) in output.zip(channel) {
                    let value = S::from_sample_(*input);
                    *sample = value;
                }
            }

            if data.len() != chunk_size {
                // this is the last chunk, and it contained less than RENDER_QUANTUM_SIZE samples
                let channel_offset = data.len() / self.number_of_channels;
                debug_assert!(channel_offset < RENDER_QUANTUM_SIZE);
                self.buffer_offset = Some((channel_offset, destination_buffer));
            }

            // handle addition/removal of nodes/edges
            self.handle_control_messages_with_budget(control_budget);
        }
    }

    fn set_state(&self, state: AudioContextState) {
        self.state.store(state as u8, Ordering::Relaxed);
        self.event_sender
            .try_send(EventDispatch::state_change(state))
            .ok();
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        if let Some(batch) = self.pending_control_batch.take() {
            self.reclaim_control_batch(batch);
        }
        self.retire_control_receiver();
        if let (Some(gc), Some(termination)) = (
            self.garbage_collector.as_mut(),
            self.garbage_collector_termination.take(),
        ) {
            gc.push(termination);
        }
        log::info!("Audio render thread has been dropped");
    }
}

// Controls the polling frequency of the garbage collector thread.
const GARBAGE_COLLECTOR_THREAD_TIMEOUT: Duration = Duration::from_millis(100);

// Poison pill that terminates the garbage collector thread.
#[derive(Debug)]
struct TerminateGarbageCollectorThread;

/// Owns the channel receiver after shutdown so queued envelopes are destroyed by the GC sidecar.
struct ControlReceiverRetirement(Option<Receiver<ControlMessage>>);

// Spawns a sidecar thread of the `RenderThread` for dropping resources.
fn spawn_garbage_collector_thread(consumer: llq::Consumer<Box<dyn Any + Send>>) {
    let _join_handle = std::thread::spawn(move || run_garbage_collector_thread(consumer));
}

fn run_garbage_collector_thread(mut consumer: llq::Consumer<Box<dyn Any + Send>>) {
    log::info!("Entering garbage collector thread");
    loop {
        if let Some(node) = consumer.pop() {
            if node
                .as_ref()
                .downcast_ref::<TerminateGarbageCollectorThread>()
                .is_some()
            {
                log::info!("Terminating garbage collector thread");
                break;
            }
            // Implicitly drop the received node.
        } else {
            std::thread::sleep(GARBAGE_COLLECTOR_THREAD_TIMEOUT);
        }
    }
    log::info!("Exiting garbage collector thread");
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::events::EventLoop;
    use crate::message::{
        ControlBatchSendError, ControlBatchSender, CONTROL_BATCH_CAPACITY,
        CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT,
    };

    struct TestHarness {
        renderer: RenderThread,
        legacy: Sender<ControlMessage>,
        batches: ControlBatchSender,
        applied: ControlBatchApplied,
        events: EventLoop,
        event_sender: Sender<EventDispatch>,
        garbage: llq::Consumer<Box<dyn Any + Send>>,
        retained_receiver: Receiver<ControlMessage>,
    }

    fn harness(control_capacity: usize, event_capacity: usize) -> TestHarness {
        let (legacy, receiver) = crossbeam_channel::bounded(control_capacity);
        let retained_receiver = receiver.clone();
        let batches = ControlBatchSender::new(legacy.clone());
        let applied = ControlBatchApplied::default();
        let (event_sender, event_receiver) = crossbeam_channel::bounded(event_capacity);
        let (garbage_producer, garbage) = llq::Queue::new().split();
        let mut renderer = RenderThread::new(
            48_000.,
            2,
            receiver,
            Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            event_sender.clone(),
            applied.clone(),
        );
        renderer.garbage_collector = Some(garbage_producer);

        TestHarness {
            renderer,
            legacy,
            batches,
            applied,
            events: EventLoop::new(event_receiver),
            event_sender,
            garbage,
            retained_receiver,
        }
    }

    fn run_callback(renderer: &mut RenderThread) {
        renderer.render(&mut [] as &mut [f32]);
    }

    fn marker(value: u16, log: &Arc<Mutex<Vec<u16>>>) -> ControlMessage {
        ControlMessage::TestMarker {
            value,
            log: Arc::clone(log),
        }
    }

    #[test]
    fn batch_occupies_one_physical_channel_slot() {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let batches = ControlBatchSender::new(sender);
        let commands = (0..CONTROL_BATCH_CAPACITY)
            .map(|_| ControlMessage::TestNop)
            .collect();

        assert_eq!(batches.try_send(commands), Ok(1));
        assert_eq!(
            batches.try_send(vec![ControlMessage::TestNop]),
            Err(ControlBatchSendError::QueueFull)
        );
        assert!(matches!(receiver.try_recv(), Ok(ControlMessage::Batch(_))));
        assert_eq!(batches.try_send(vec![ControlMessage::TestNop]), Ok(2));
    }

    #[test]
    fn batch_submission_validates_shape_before_queue_mutation() {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let batches = ControlBatchSender::new(sender);
        assert_eq!(
            batches.try_send(Vec::new()),
            Err(ControlBatchSendError::Empty)
        );
        assert_eq!(
            batches.try_send(
                (0..=CONTROL_BATCH_CAPACITY)
                    .map(|_| ControlMessage::TestNop)
                    .collect()
            ),
            Err(ControlBatchSendError::TooLarge)
        );
        let nested = ControlMessage::Batch(llq::Node::new(Box::new(()) as Box<dyn Any + Send>));
        assert_eq!(
            batches.try_send(vec![nested]),
            Err(ControlBatchSendError::NestedBatch)
        );
        assert!(receiver.is_empty());
        assert_eq!(batches.batch_storage_in_flight(), 0);
        assert_eq!(batches.try_send(vec![ControlMessage::TestNop]), Ok(1));
    }

    #[test]
    fn lifecycle_diagnostic_and_non_graph_maintenance_commands_cannot_be_batched() {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let batches = ControlBatchSender::new(sender);
        let (id_producer, _id_consumer) = llq::Queue::new().split();
        let (notify, _notify_receiver) = crossbeam_channel::bounded(1);
        let (resume_notify, _resume_receiver) = crossbeam_channel::bounded(1);
        let (close_notify, _close_receiver) = crossbeam_channel::bounded(1);
        let (graph_send, _graph_recv) = crossbeam_channel::bounded(1);
        let commands = vec![
            ControlMessage::ControlHandleDropped { id: AudioNodeId(7) },
            ControlMessage::Startup {
                graph: Graph::new(id_producer),
            },
            ControlMessage::Suspend {
                notify: crate::message::OneshotNotify::Sync(notify),
            },
            ControlMessage::Resume {
                notify: crate::message::OneshotNotify::Sync(resume_notify),
            },
            ControlMessage::Close {
                notify: crate::message::OneshotNotify::Sync(close_notify),
            },
            ControlMessage::CloseAndRecycle { sender: graph_send },
        ];

        for command in commands {
            assert_eq!(
                batches.try_send(vec![command]),
                Err(ControlBatchSendError::UnsupportedCommand)
            );
        }

        #[cfg(feature = "diagnostics")]
        assert_eq!(
            batches.try_send(vec![ControlMessage::RunDiagnostics {
                backend: crate::context::AudioBackendDiagnostics {
                    name: String::new(),
                    sink_id: String::new(),
                    output_latency: None,
                },
            }]),
            Err(ControlBatchSendError::UnsupportedCommand)
        );

        assert!(receiver.is_empty());
        assert_eq!(batches.batch_storage_in_flight(), 0);
        assert_eq!(
            batches.try_send(vec![ControlMessage::MarkCycleBreaker {
                id: AudioNodeId(7),
            }]),
            Ok(1)
        );
        assert!(matches!(receiver.try_recv(), Ok(ControlMessage::Batch(_))));
        assert_eq!(batches.try_send(vec![ControlMessage::TestNop]), Ok(2));
    }

    #[test]
    fn partial_batch_preserves_fifo_and_publishes_only_when_complete() {
        let mut test = harness(4, 4);
        let log = Arc::new(Mutex::new(Vec::new()));
        test.legacy.try_send(marker(0, &log)).unwrap();
        test.batches
            .try_send((1..=256).map(|value| marker(value, &log)).collect())
            .unwrap();
        test.legacy.try_send(marker(257, &log)).unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=255).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 0);

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=257).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 1);
    }

    #[test]
    fn sink_swap_replays_partial_and_cached_fifo_without_watermark_gap() {
        let mut old = harness(4, 8);
        let (id_producer, _id_consumer) = llq::Queue::new().split();
        old.renderer.graph = Some(Graph::new(id_producer));
        let log = Arc::new(Mutex::new(Vec::new()));
        old.legacy.try_send(marker(0, &log)).unwrap();
        old.batches
            .try_send((1..=256).map(|value| marker(value, &log)).collect())
            .unwrap();
        old.batches.try_send(vec![marker(257, &log)]).unwrap();
        old.legacy.try_send(marker(258, &log)).unwrap();

        run_callback(&mut old.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=255).collect::<Vec<_>>());
        assert_eq!(old.applied.load(), 0);

        // Mirrors set_sink_id_sync: cache records not yet owned by the old renderer, then place
        // CloseAndRecycle after its partially-applied batch.
        let cached: Vec<_> = old.retained_receiver.try_iter().collect();
        assert_eq!(cached.len(), 2);
        let (graph_send, graph_recv) = crossbeam_channel::bounded(1);
        old.legacy
            .send(ControlMessage::CloseAndRecycle { sender: graph_send })
            .unwrap();
        run_callback(&mut old.renderer);
        let graph = graph_recv.recv().unwrap();
        assert_eq!(&*log.lock().unwrap(), &(0..=256).collect::<Vec<_>>());
        assert_eq!(old.applied.load(), 1);
        drop(old.renderer);

        // New renderer adopts the graph, then replay uses the already-held sender directly in the
        // exact cached order: batch sequence 2 followed by the legacy marker.
        let (gc_producer, _gc_consumer) = llq::Queue::new().split();
        let mut new_renderer = RenderThread::new(
            48_000.,
            2,
            old.retained_receiver.clone(),
            Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            old.event_sender.clone(),
            old.applied.clone(),
        );
        new_renderer.garbage_collector = Some(gc_producer);
        old.legacy.send(ControlMessage::Startup { graph }).unwrap();
        for message in cached {
            old.legacy.send(message).unwrap();
        }

        run_callback(&mut new_renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=258).collect::<Vec<_>>());
        assert_eq!(old.applied.load(), 2);
    }

    #[test]
    fn unrelated_full_event_queue_still_wakes_authoritative_reconciliation() {
        let mut test = harness(2, 256);
        for _ in 0..256 {
            test.event_sender
                .try_send(EventDispatch::sink_change())
                .unwrap();
        }
        let observed = Arc::new(AtomicU64::new(0));
        let observed_clone = Arc::clone(&observed);
        let applied = test.applied.clone();
        test.events.set_activity_handler(move || {
            observed_clone.store(applied.load(), Ordering::Release);
        });
        test.batches
            .try_send(vec![ControlMessage::TestNop])
            .unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(test.applied.load(), 1);
        assert_eq!(observed.load(Ordering::Acquire), 0);
        assert!(test.events.handle_pending_events());
        assert_eq!(observed.load(Ordering::Acquire), 1);
    }

    #[test]
    fn completing_full_batch_allocates_nothing_on_render_thread() {
        let mut test = harness(1, 1);
        test.batches
            .try_send(
                (0..CONTROL_BATCH_CAPACITY)
                    .map(|_| ControlMessage::TestNop)
                    .collect(),
            )
            .unwrap();

        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_eq!(test.applied.load(), 1);
        assert!(test.garbage.pop().is_some());
    }

    #[test]
    fn render_thread_drop_uses_only_preallocated_gc_records() {
        let test = harness(1, 1);
        let renderer = test.renderer;
        alloc_counter::deny_alloc(|| drop(renderer));
    }

    #[test]
    fn completed_batch_storage_is_reclaimed_off_render_thread() {
        let mut test = harness(1, 1);
        let (reclaimed_send, reclaimed_recv) = crossbeam_channel::bounded(1);
        test.batches
            .try_send_with_reclaim_probe(
                vec![ControlMessage::TestNop],
                Arc::new(move || {
                    let _ = reclaimed_send.send(std::thread::current().id());
                }),
            )
            .unwrap();

        let render_thread = std::thread::current().id();
        run_callback(&mut test.renderer);
        assert!(reclaimed_recv.try_recv().is_err());
        let node = test.garbage.pop().unwrap();
        let reclaim_thread = std::thread::spawn(move || drop(node));
        reclaim_thread.join().unwrap();
        assert_ne!(reclaimed_recv.recv().unwrap(), render_thread);
    }

    #[test]
    fn batch_storage_limit_includes_completed_gc_backlog() {
        let mut test = harness(CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT, 256);
        for sequence in 1..=CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT {
            assert_eq!(
                test.batches.try_send(vec![ControlMessage::TestNop]),
                Ok(sequence as u64)
            );
        }
        run_callback(&mut test.renderer);
        assert_eq!(
            test.applied.load(),
            CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT as u64
        );
        assert_eq!(
            test.batches.batch_storage_in_flight(),
            CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT
        );
        assert_eq!(
            test.batches.try_send(vec![ControlMessage::TestNop]),
            Err(ControlBatchSendError::BatchStorageLimit)
        );

        drop(test.garbage.pop().unwrap());
        assert_eq!(
            test.batches.batch_storage_in_flight(),
            CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT - 1
        );
        assert_eq!(
            test.batches.try_send(vec![ControlMessage::TestNop]),
            Ok((CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT + 1) as u64)
        );
    }

    #[test]
    fn dropping_partial_batch_does_not_advance_watermark() {
        let mut test = harness(2, 2);
        test.legacy.try_send(ControlMessage::TestNop).unwrap();
        test.batches
            .try_send(
                (0..CONTROL_BATCH_CAPACITY)
                    .map(|_| ControlMessage::TestNop)
                    .collect(),
            )
            .unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(test.applied.load(), 0);
        assert!(test.renderer.pending_control_batch.is_some());
        drop(test.renderer);
        assert_eq!(test.applied.load(), 0);
        drop(test.garbage.pop().unwrap());
        assert_eq!(test.batches.batch_storage_in_flight(), 0);
    }

    #[test]
    fn queued_batch_is_retired_off_render_thread_without_advancing_watermark() {
        let mut test = harness(2, 2);
        test.batches
            .try_send(
                (0..CONTROL_BATCH_CAPACITY)
                    .map(|_| ControlMessage::TestNop)
                    .collect(),
            )
            .unwrap();
        let (reclaimed_send, reclaimed_recv) = crossbeam_channel::bounded(1);
        test.batches
            .try_send_with_reclaim_probe(
                vec![ControlMessage::TestNop],
                Arc::new(move || {
                    let _ = reclaimed_send.send(std::thread::current().id());
                }),
            )
            .unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(test.applied.load(), 1);
        assert_eq!(test.batches.batch_storage_in_flight(), 2);
        drop(test.renderer);

        let render_thread = std::thread::current().id();
        let mut retired = Vec::new();
        while let Some(node) = test.garbage.pop() {
            retired.push(node);
        }
        let legacy = test.legacy;
        let batches = test.batches;
        let retained_receiver = test.retained_receiver;
        std::thread::spawn(move || {
            drop(retired);
            drop(retained_receiver);
            drop(legacy);
            drop(batches);
        })
        .join()
        .unwrap();
        assert_ne!(
            reclaimed_recv.recv_timeout(Duration::from_secs(1)).unwrap(),
            render_thread
        );
        assert_eq!(test.applied.load(), 1);
    }

    #[test]
    fn permanent_close_allows_retained_receiver_to_reclaim_queued_batch() {
        let mut test = harness(2, 2);
        let (closed_send, closed_recv) = crossbeam_channel::bounded(1);
        test.legacy
            .try_send(ControlMessage::Close {
                notify: crate::message::OneshotNotify::Sync(closed_send),
            })
            .unwrap();
        let (reclaimed_send, reclaimed_recv) = crossbeam_channel::bounded(1);
        test.batches
            .try_send_with_reclaim_probe(
                vec![ControlMessage::TestNop],
                Arc::new(move || {
                    let _ = reclaimed_send.send(std::thread::current().id());
                }),
            )
            .unwrap();

        run_callback(&mut test.renderer);
        closed_recv.recv().unwrap();
        assert!(test.renderer.receiver.is_none());
        assert_eq!(test.applied.load(), 0);
        assert_eq!(test.batches.batch_storage_in_flight(), 1);

        // Mirrors AudioContext::retire_render_thread_init on the permanent-close control path.
        for message in test.retained_receiver.try_iter() {
            drop(message);
        }
        assert_eq!(reclaimed_recv.recv().unwrap(), std::thread::current().id());
        assert_eq!(test.batches.batch_storage_in_flight(), 0);
        assert_eq!(test.applied.load(), 0);
    }

    #[test]
    fn sequence_gap_is_rejected_without_mutation_or_watermark() {
        let mut test = harness(3, 3);
        let log = Arc::new(Mutex::new(Vec::new()));
        test.batches
            .try_send_with_sequence(2, vec![marker(2, &log)])
            .unwrap();
        test.batches.try_send(vec![marker(1, &log)]).unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &[1]);
        assert_eq!(test.applied.load(), 1);

        test.batches.try_send(vec![marker(2, &log)]).unwrap();
        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &[1, 2]);
        assert_eq!(test.applied.load(), 2);
    }

    #[test]
    fn offline_unbounded_legacy_drain_is_unchanged() {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let (event_sender, _event_receiver) = crossbeam_channel::unbounded();
        let log = Arc::new(Mutex::new(Vec::new()));
        for value in 0..300 {
            sender.send(marker(value, &log)).unwrap();
        }
        let mut renderer = RenderThread::new(
            48_000.,
            2,
            receiver,
            Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            event_sender,
            ControlBatchApplied::default(),
        );

        renderer.handle_control_messages();
        assert_eq!(&*log.lock().unwrap(), &(0..300).collect::<Vec<_>>());
    }
}
