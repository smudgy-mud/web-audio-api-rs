//! Communicates with the control thread and ships audio samples to the hardware

use std::any::Any;
use std::cell::Cell;
use std::io;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use dasp_sample::FromSample;
use futures_channel::{mpsc, oneshot};
use futures_util::StreamExt as _;

use super::AudioRenderQuantum;
use crate::buffer::AudioBuffer;
#[cfg(feature = "diagnostics")]
use crate::context::{AudioContextDiagnostics, AudioRenderThreadDiagnostics};
use crate::context::{
    AudioContextState, AudioNodeId, InjectedGraphReclaimInit, OfflineAudioContext,
    OfflineAudioContextCallback,
};
use crate::events::{
    EventDispatch, EventLoop, InjectedContextState, InjectedLifecycleEventLoop,
    InjectedStateTransition,
};
use crate::message::{
    control_batch_storage, control_batch_storage_mut, ControlBatchApplied, ControlBatchNode,
    ControlMessage, GraphLifecycleBarrier, GraphLifecycleOutcome, GraphLifecyclePublisher,
    GraphLifecycleTransition, InjectedExplicitBatchShape, InjectedPhysicalCreditOwners,
    CONTROL_COMMANDS_PER_CALLBACK,
};
use crate::node::ChannelInterpretation;
use crate::render::AudioWorkletGlobalScope;
use crate::stats::AudioStats;
use crate::RENDER_QUANTUM_SIZE;

use super::graph::Graph;

/// Single-use bootstrap producer for one injected context's exact event loop.
///
/// The only raw-unpacking operation is private to this render module. Setup code may construct the
/// opaque value, but crate callers cannot send or recover its sender before `RenderThread` consumes
/// it and owns every internally cloneable render producer.
pub(crate) struct InjectedEventDispatchSender {
    sender: Sender<EventDispatch>,
    identity: Arc<()>,
    event_thread_alive: Arc<AtomicBool>,
}

impl InjectedEventDispatchSender {
    pub(crate) fn from_event_setup(
        sender: Sender<EventDispatch>,
        identity: Arc<()>,
        event_thread_alive: Arc<AtomicBool>,
    ) -> Self {
        Self {
            sender,
            identity,
            event_thread_alive,
        }
    }
}

/// Render-owned producer retaining either the unchanged legacy sender or an exact injected brand.
#[derive(Clone)]
pub(crate) struct EventDispatchSender(EventDispatchSenderKind);

#[derive(Clone)]
enum EventDispatchSenderKind {
    Legacy(Sender<EventDispatch>),
    Injected {
        sender: Sender<EventDispatch>,
        identity: Arc<()>,
        event_thread_alive: Arc<AtomicBool>,
    },
}

impl EventDispatchSender {
    fn from_injected(sender: InjectedEventDispatchSender) -> Self {
        let InjectedEventDispatchSender {
            sender,
            identity,
            event_thread_alive,
        } = sender;
        Self(EventDispatchSenderKind::Injected {
            sender,
            identity,
            event_thread_alive,
        })
    }

    // Returning the rejected event by value keeps the render-thread failure path allocation-free;
    // boxing this intentionally larger exact-event payload would violate that contract.
    #[allow(clippy::result_large_err)]
    pub(crate) fn try_send(&self, event: EventDispatch) -> Result<(), TrySendError<EventDispatch>> {
        match &self.0 {
            EventDispatchSenderKind::Legacy(sender) => sender.try_send(event),
            EventDispatchSenderKind::Injected {
                sender,
                event_thread_alive,
                ..
            } => {
                if event_thread_alive.load(Ordering::Acquire) {
                    sender.try_send(event)
                } else {
                    Err(TrySendError::Disconnected(event))
                }
            }
        }
    }

    fn matches_injected(&self, event_loop: &InjectedLifecycleEventLoop) -> bool {
        matches!(&self.0, EventDispatchSenderKind::Injected { identity, .. } if event_loop.matches_identity(identity))
    }

    fn matches_injected_control(
        &self,
        events: &crate::events::InjectedControlEventDispatch,
    ) -> bool {
        matches!(&self.0, EventDispatchSenderKind::Injected { identity, .. } if events.matches_identity(identity))
    }
}

impl From<Sender<EventDispatch>> for EventDispatchSender {
    fn from(sender: Sender<EventDispatch>) -> Self {
        Self(EventDispatchSenderKind::Legacy(sender))
    }
}

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
    injected_state: Option<InjectedContextState>,
    startup_pending: Option<Arc<AtomicBool>>,
    frames_played: Arc<AtomicU64>,
    receiver: Option<Receiver<ControlMessage>>,
    /// Preallocated carrier used to retire the receiver and any queued batches off the callback.
    receiver_retirement: Option<llq::Node<Box<dyn Any + Send>>>,
    buffer_offset: Option<(usize, AudioRenderQuantum)>,
    stats: AudioStats,
    event_sender: EventDispatchSender,
    garbage_collector: Option<llq::Producer<Box<dyn Any + Send>>>,
    /// Preallocated poison record; render-side shutdown must not allocate its GC notification.
    garbage_collector_termination: Option<llq::Node<Box<dyn Any + Send>>>,
    control_batch_applied: ControlBatchApplied,
    pending_control_batch: Option<ControlBatchNode>,
    /// Keeps both injected physical-credit allocations alive while an injected record can be
    /// dequeued. Consequently releasing a credit on RT is an atomic decrement plus a non-final
    /// `Arc` decrement; allocation destruction remains off RT with this renderer.
    injected_physical_credit_owners: Option<InjectedPhysicalCreditOwners>,
    graph_lifecycle_publisher: Option<GraphLifecyclePublisher>,
    graph_lifecycle_next_sequence: u64,
    #[cfg(test)]
    fail_next_gc_spawn: bool,
    #[cfg(test)]
    fail_reclaim: bool,
    #[cfg(test)]
    disconnect_lifecycle_on_next_render: bool,
    #[cfg(test)]
    panic_magic_apply: bool,
    #[cfg(test)]
    magic_bootstrap_shape: [bool; crate::context::MAGIC_COMMAND_COUNT],
    #[cfg(test)]
    magic_bootstrap_command_count: usize,
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
// Retained for the legacy OfflineAudioContext future's documented Send + Sync contract. The
// injected output seam does not rely on this: AudioRenderCallback remains deliberately !Sync.
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
    pub(crate) fn injected_base_facts(&self) -> (f32, usize, Arc<AtomicU64>) {
        (
            self.sample_rate,
            self.number_of_channels,
            Arc::clone(&self.frames_played),
        )
    }

    pub(crate) fn matches_output_format(
        &self,
        sample_rate: f32,
        number_of_channels: usize,
    ) -> bool {
        self.sample_rate.to_bits() == sample_rate.to_bits()
            && self.number_of_channels == number_of_channels
    }

    /// Installs the only graph accepted by the injected renderer foundation. Consuming the opaque
    /// initializer binds its exact reclaim queue and activity publisher; injected construction
    /// cannot create a renderer while omitting that publisher.
    pub(crate) fn install_injected_graph(
        &mut self,
        graph: InjectedGraphReclaimInit,
    ) -> Result<(), InjectedGraphReclaimInit> {
        if self.graph.is_some() {
            return Err(graph);
        }
        self.graph = Some(graph.into_graph());
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn has_injected_reclaim_publisher(&self) -> bool {
        self.graph
            .as_ref()
            .is_some_and(Graph::has_injected_reclaim_publisher)
    }

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
        event_sender: impl Into<EventDispatchSender>,
        control_batch_applied: ControlBatchApplied,
    ) -> Self {
        Self {
            graph: None,
            sample_rate,
            buffer_size: 0,
            number_of_channels,
            suspended: false,
            state,
            injected_state: None,
            startup_pending: None,
            frames_played,
            receiver: Some(receiver),
            receiver_retirement: Some(llq::Node::new(Box::new(ControlReceiverRetirement(None)))),
            buffer_offset: None,
            stats,
            event_sender: event_sender.into(),
            garbage_collector: None,
            garbage_collector_termination: Some(llq::Node::new(Box::new(
                TerminateGarbageCollectorThread,
            ))),
            control_batch_applied,
            pending_control_batch: None,
            injected_physical_credit_owners: None,
            graph_lifecycle_publisher: None,
            graph_lifecycle_next_sequence: 1,
            #[cfg(test)]
            fail_next_gc_spawn: false,
            #[cfg(test)]
            fail_reclaim: false,
            #[cfg(test)]
            disconnect_lifecycle_on_next_render: false,
            #[cfg(test)]
            panic_magic_apply: false,
            #[cfg(test)]
            magic_bootstrap_shape: [false; crate::context::MAGIC_COMMAND_COUNT],
            #[cfg(test)]
            magic_bootstrap_command_count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_injected(
        sample_rate: f32,
        number_of_channels: usize,
        receiver: Receiver<ControlMessage>,
        state: InjectedContextState,
        frames_played: Arc<AtomicU64>,
        stats: AudioStats,
        event_sender: InjectedEventDispatchSender,
        control_batch_applied: ControlBatchApplied,
    ) -> Self {
        let initially_suspended = state.load() == AudioContextState::Suspended;
        let shared_state = state.atomic_for_render();
        let mut renderer = Self::new(
            sample_rate,
            number_of_channels,
            receiver,
            shared_state,
            frames_played,
            stats,
            EventDispatchSender::from_injected(event_sender),
            control_batch_applied,
        );
        renderer.suspended = initially_suspended;
        renderer.injected_state = Some(state);
        renderer
    }

    pub(crate) fn set_startup_pending(&mut self, startup_pending: Arc<AtomicBool>) {
        self.startup_pending = Some(startup_pending);
    }

    /// Installs the persistent injected lifecycle acknowledgement publisher.
    #[allow(dead_code)] // Used by the private injected-context integration.
    pub(crate) fn set_graph_lifecycle_publisher(
        &mut self,
        publisher: GraphLifecyclePublisher,
    ) -> Result<(), GraphLifecyclePublisher> {
        if self.graph_lifecycle_publisher.is_some() {
            return Err(publisher);
        }
        self.graph_lifecycle_publisher = Some(publisher);
        Ok(())
    }

    pub(crate) fn matches_injected_event_loop(
        &self,
        event_loop: &InjectedLifecycleEventLoop,
    ) -> bool {
        self.event_sender.matches_injected(event_loop)
    }

    pub(crate) fn matches_injected_control_events(
        &self,
        events: &crate::events::InjectedControlEventDispatch,
    ) -> bool {
        self.event_sender.matches_injected_control(events)
    }

    /// Installs the render-side lifetime owners required before injected records can be received.
    #[allow(dead_code)] // Used by the private injected control transport; exercised in tests.
    pub(crate) fn set_injected_physical_credit_owners(
        &mut self,
        owners: InjectedPhysicalCreditOwners,
    ) -> Result<(), InjectedPhysicalCreditOwners> {
        if self.injected_physical_credit_owners.is_some() {
            return Err(owners);
        }
        self.injected_physical_credit_owners = Some(owners);
        Ok(())
    }

    pub(crate) fn spawn_garbage_collector_thread(&mut self) {
        let _detached = self.spawn_joinable_garbage_collector_thread();
    }

    /// Installs the render-side garbage collector and returns its join handle.
    ///
    /// The injected-output lifecycle owns this handle and joins it only after the render callback
    /// has retired and `RenderThread` Drop has enqueued the preallocated poison record. Legacy
    /// backends intentionally continue to use `spawn_garbage_collector_thread`, which detaches the
    /// same sidecar as before.
    pub(crate) fn spawn_joinable_garbage_collector_thread(
        &mut self,
    ) -> Option<std::thread::JoinHandle<()>> {
        self.try_spawn_joinable_garbage_collector_thread()
            .expect("legacy garbage collector thread spawn failed")
    }

    /// Fallible injected-output variant which mutates the renderer only after the sidecar thread
    /// has accepted ownership. On spawn failure the exact renderer remains unchanged and can be
    /// returned through the branded output bootstrap transaction.
    pub(crate) fn try_spawn_joinable_garbage_collector_thread(
        &mut self,
    ) -> io::Result<Option<std::thread::JoinHandle<()>>> {
        if self.garbage_collector.is_none() {
            #[cfg(test)]
            if std::mem::take(&mut self.fail_next_gc_spawn) {
                return Err(io::Error::other("forced garbage collector spawn failure"));
            }
            let (gc_producer, gc_consumer) = llq::Queue::new().split();
            let join = try_spawn_garbage_collector_thread(gc_consumer)?;
            self.garbage_collector = Some(gc_producer);
            Ok(Some(join))
        } else {
            Ok(None)
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_next_gc_spawn_for_test(&mut self) {
        self.fail_next_gc_spawn = true;
    }

    #[cfg(test)]
    pub(crate) fn fail_reclaim_for_test(&mut self) {
        self.fail_reclaim = true;
    }

    #[cfg(test)]
    pub(crate) fn disconnect_lifecycle_on_next_render_for_test(&mut self) {
        self.disconnect_lifecycle_on_next_render = true;
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

                let message = match message {
                    ControlMessage::InjectedBatch { batch, physical } => {
                        debug_assert!(self.injected_physical_credit_owners.is_some());
                        drop(physical);
                        ControlMessage::Batch(batch)
                    }
                    ControlMessage::InjectedGraphLifecycleBarrier { barrier, physical } => {
                        debug_assert!(self.injected_physical_credit_owners.is_some());
                        drop(physical);
                        ControlMessage::GraphLifecycleBarrier(barrier)
                    }
                    message => message,
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

            let batch_commands = control_batch_storage(
                self.pending_control_batch
                    .as_ref()
                    .expect("batch is pending"),
            )
            .remaining_len();
            if batch_commands > *remaining {
                // Keep both the storage cursor and every command untouched. The envelope remains
                // ahead of later direct records and will be reconsidered with the next callback's
                // fresh budget.
                return;
            }

            let exact_shape = control_batch_storage(
                self.pending_control_batch
                    .as_ref()
                    .expect("batch is pending"),
            )
            .classify_injected_explicit_batch();
            match exact_shape {
                InjectedExplicitBatchShape::Invalid => {
                    self.fail_injected_render_protocol();
                }
                InjectedExplicitBatchShape::Connect(value) => {
                    let valid = self.graph.as_ref().is_some_and(|graph| {
                        graph.preflight_injected_explicit_connect(value).is_ok()
                    });
                    if !valid {
                        self.fail_injected_render_protocol();
                    }
                    self.graph
                        .as_mut()
                        .expect("preflight proved an installed exact graph")
                        .apply_injected_explicit_connect(value);
                    let command = control_batch_storage_mut(
                        self.pending_control_batch
                            .as_mut()
                            .expect("batch is pending"),
                    )
                    .take_next();
                    debug_assert!(matches!(
                        command,
                        Some(ControlMessage::InjectedConnectExplicit(_))
                    ));
                    *remaining -= 1;
                }
                InjectedExplicitBatchShape::Disconnect => {
                    let valid = {
                        let (graph, pending) = (&self.graph, &self.pending_control_batch);
                        let storage =
                            control_batch_storage(pending.as_ref().expect("batch is pending"));
                        graph.as_ref().is_some_and(|graph| {
                            graph
                                .preflight_injected_explicit_disconnects(
                                    storage.injected_explicit_disconnects(),
                                )
                                .is_ok()
                        })
                    };
                    if !valid {
                        self.fail_injected_render_protocol();
                    }
                    {
                        let (graph, pending) = (&mut self.graph, &self.pending_control_batch);
                        let storage =
                            control_batch_storage(pending.as_ref().expect("batch is pending"));
                        graph
                            .as_mut()
                            .expect("preflight proved an installed exact graph")
                            .apply_injected_explicit_disconnects(
                                storage.injected_explicit_disconnects(),
                            );
                    }
                    for _ in 0..batch_commands {
                        let command = control_batch_storage_mut(
                            self.pending_control_batch
                                .as_mut()
                                .expect("batch is pending"),
                        )
                        .take_next();
                        debug_assert!(matches!(
                            command,
                            Some(ControlMessage::InjectedDisconnectExplicit(_))
                        ));
                        *remaining -= 1;
                    }
                }
                InjectedExplicitBatchShape::Ordinary => {
                    for _ in 0..batch_commands {
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
                    }
                }
            }

            let batch = self.pending_control_batch.take().unwrap();
            debug_assert!(control_batch_storage(&batch).is_complete());
            let sequence = control_batch_storage(&batch).sequence();
            self.control_batch_applied.publish(sequence);
            self.event_sender
                .try_send(EventDispatch::control_batch_activity())
                .ok();
            self.reclaim_control_batch(batch);
        }
    }

    #[cold]
    #[inline(never)]
    fn fail_injected_render_protocol(&mut self) -> ! {
        // Do not publish the failing envelope's sequence. The absorbing latch is authoritative
        // even when the best-effort event hint is saturated; earlier envelopes may already have
        // changed the Graph before this separately bounded exact operation failed.
        self.control_batch_applied.fail_render_protocol();
        self.event_sender
            .try_send(EventDispatch::control_batch_activity())
            .ok();
        panic!("injected render protocol violation");
    }

    /// Applies a complete injected bootstrap envelope before an audio callback can be published.
    /// The batch capacity is statically bounded by the per-callback command budget, so an envelope
    /// is either left untouched or consumed completely. Success proves both its exact applied
    /// sequence and destination node 0 are present.
    pub(crate) fn apply_injected_magic_before_publication(
        &mut self,
        required_sequence: u64,
    ) -> bool {
        #[cfg(test)]
        if self.panic_magic_apply {
            self.panic_magic_apply = false;
            panic!("forced prepublication magic application panic");
        }
        let mut budget = crate::message::CONTROL_COMMANDS_PER_CALLBACK;
        self.handle_control_messages_with_budget(&mut budget);
        self.pending_control_batch.is_none()
            && self.control_batch_applied.load() >= required_sequence
            && self
                .graph
                .as_ref()
                .is_some_and(|graph| graph.contains_node(crate::context::DESTINATION_NODE_ID))
    }

    #[cfg(test)]
    pub(crate) fn panic_magic_apply_for_test(&mut self) {
        self.panic_magic_apply = true;
    }

    #[cfg(test)]
    pub(crate) fn magic_bootstrap_shape_is_exact_for_test(&self) -> bool {
        self.magic_bootstrap_command_count == crate::context::MAGIC_COMMAND_COUNT
            && self
                .magic_bootstrap_shape
                .into_iter()
                .all(std::convert::identity)
    }

    #[inline]
    fn reclaim_control_batch(&mut self, batch: ControlBatchNode) {
        if let Some(gc) = self.garbage_collector.as_mut() {
            gc.push(batch);
        }
    }

    fn retire_control_receiver(&mut self) -> bool {
        let Some(receiver) = self.receiver.take() else {
            return true;
        };
        let Some(mut retirement) = self.receiver_retirement.take() else {
            self.receiver = Some(receiver);
            return false;
        };
        let Some(retired_receiver) = retirement
            .as_mut()
            .downcast_mut::<ControlReceiverRetirement>()
        else {
            self.receiver = Some(receiver);
            self.receiver_retirement = Some(retirement);
            return false;
        };
        if retired_receiver.0.is_some() {
            self.receiver = Some(receiver);
            self.receiver_retirement = Some(retirement);
            return false;
        }
        retired_receiver.0 = Some(receiver);
        if let Some(gc) = self.garbage_collector.as_mut() {
            gc.push(retirement);
        } else {
            // Retain the receiver in its preallocated carrier until the whole renderer is
            // reclaimed off the callback thread.
            self.receiver_retirement = Some(retirement);
        }
        true
    }

    fn handle_graph_lifecycle_barrier(
        &mut self,
        barrier: GraphLifecycleBarrier,
    ) -> ControlFlow<()> {
        if self.graph_lifecycle_publisher.is_none() {
            return ControlFlow::Continue(());
        }

        let sequence = barrier.controller_sequence();
        let expected = self.graph_lifecycle_next_sequence;
        let observed_batch_sequence = self.control_batch_applied.load();
        if sequence == 0 || expected == 0 || sequence != expected {
            // A forward gap can be reported authoritatively without regressing the commit word.
            // Reuse/regression and zero cannot produce a distinct acknowledgement safely.
            if sequence > expected && expected != 0 {
                self.publish_graph_lifecycle_ack(
                    barrier,
                    observed_batch_sequence,
                    GraphLifecycleOutcome::ControllerSequenceGap,
                );
                self.graph_lifecycle_next_sequence = 0;
            }
            return ControlFlow::Continue(());
        }

        self.graph_lifecycle_next_sequence = sequence.checked_add(1).unwrap_or(0);
        if observed_batch_sequence < barrier.required_batch_sequence() {
            self.publish_graph_lifecycle_ack(
                barrier,
                observed_batch_sequence,
                GraphLifecycleOutcome::RequiredBatchPending,
            );
            return ControlFlow::Continue(());
        }

        let close = match barrier.transition() {
            GraphLifecycleTransition::Suspend => {
                self.suspended = true;
                let outcome =
                    self.transition_injected_state_with_event(AudioContextState::Suspended);
                if outcome != GraphLifecycleOutcome::Applied {
                    self.publish_graph_lifecycle_ack(barrier, observed_batch_sequence, outcome);
                    return ControlFlow::Continue(());
                }
                false
            }
            GraphLifecycleTransition::Resume => {
                self.suspended = false;
                let outcome = self.transition_injected_state_with_event(AudioContextState::Running);
                if outcome != GraphLifecycleOutcome::Applied {
                    self.publish_graph_lifecycle_ack(barrier, observed_batch_sequence, outcome);
                    return ControlFlow::Continue(());
                }
                false
            }
            GraphLifecycleTransition::Close => {
                if !self.retire_control_receiver() {
                    self.publish_graph_lifecycle_ack(
                        barrier,
                        observed_batch_sequence,
                        GraphLifecycleOutcome::ProtocolViolation,
                    );
                    return ControlFlow::Continue(());
                }
                self.suspended = true;
                self.store_state_without_event(AudioContextState::Closed);
                true
            }
        };

        self.publish_graph_lifecycle_ack(
            barrier,
            observed_batch_sequence,
            GraphLifecycleOutcome::Applied,
        );
        if close {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn publish_graph_lifecycle_ack(
        &self,
        barrier: GraphLifecycleBarrier,
        observed_batch_sequence: u64,
        outcome: GraphLifecycleOutcome,
    ) {
        if let Some(publisher) = self.graph_lifecycle_publisher.as_ref() {
            publisher.publish(barrier, observed_batch_sequence, outcome);
        }
    }

    fn handle_control_message(&mut self, msg: ControlMessage) -> ControlFlow<()> {
        use ControlMessage::*;

        #[cfg(test)]
        self.record_magic_bootstrap_shape_for_test(&msg);

        match msg {
            Batch(_) | InjectedBatch { .. } => {
                unreachable!("batch envelopes are handled before individual commands")
            }
            InjectedGraphLifecycleBarrier { .. } => {
                unreachable!("injected barriers are normalized at dequeue")
            }
            GraphLifecycleBarrier(barrier) => {
                return self.handle_graph_lifecycle_barrier(barrier);
            }
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
                let applied = self
                    .graph
                    .as_mut()
                    .is_some_and(|graph| graph.try_add_edge((from, output), (to, input)).is_ok());
                if !applied {
                    self.fail_injected_render_protocol();
                }
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
            InjectedConnectExplicit(_) | InjectedDisconnectExplicit(_) => {
                // Exact explicit records are legal only as a pre-scanned dedicated envelope.
                self.fail_injected_render_protocol();
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
                let _ = self.retire_control_receiver();
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
            AudioParamInitialValue { id, mut value } => {
                self.graph.as_mut().unwrap().route_message(id, &mut value);
            }
            InjectedAudioParamValue { id, mut value } => {
                self.graph.as_mut().unwrap().route_message(id, &mut value);
            }
            InjectedOscillator(value) => {
                let mut message = value.into_render_message();
                let routed = self
                    .graph
                    .as_mut()
                    .is_some_and(|graph| graph.try_route_message(message.id(), &mut message));
                if !routed || !message.was_applied() {
                    self.fail_injected_render_protocol();
                }
            }
            InjectedConstantSource(value) => {
                let mut message = value.into_render_message();
                let routed = self
                    .graph
                    .as_mut()
                    .is_some_and(|graph| graph.try_route_message(message.id(), &mut message));
                if !routed || !message.was_applied() {
                    self.fail_injected_render_protocol();
                }
            }
            InjectedAudioBufferSourceScalar(value) => {
                let mut message = value.into_render_message();
                let routed = self
                    .graph
                    .as_mut()
                    .is_some_and(|graph| graph.try_route_message(message.id(), &mut message));
                if !routed || !message.was_applied() {
                    self.fail_injected_render_protocol();
                }
            }
            InjectedAudioBufferSourceBuffer(mut value) => {
                let id = value.id();
                let Some(mut message) = value.take_render_message() else {
                    self.fail_injected_render_protocol();
                };
                // A corrupt/wrong processor may panic while consuming the authenticated payload.
                // Contain that unwind until the preboxed message has moved to GC; otherwise its
                // AudioBuffer storage lease could be destroyed on the render callback thread.
                let routed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.graph
                        .as_mut()
                        .is_some_and(|graph| graph.try_route_message(id, message.as_mut()))
                }));
                let applied = message
                    .as_ref()
                    .downcast_ref::<crate::context::InjectedAudioBufferSourceBufferRenderMessage>()
                    .is_some_and(
                        crate::context::InjectedAudioBufferSourceBufferRenderMessage::was_applied,
                    );
                if let Some(gc) = self.garbage_collector.as_mut() {
                    gc.push(message);
                } else {
                    // Exact output construction always installs GC. If that invariant is broken,
                    // leak fail-closed rather than destroying an AudioBuffer lease on the RT.
                    std::mem::forget(message);
                }
                match routed {
                    Ok(true) if applied => {}
                    Ok(_) => self.fail_injected_render_protocol(),
                    Err(payload) => {
                        // The payload itself may have a hostile destructor. The fixed protocol
                        // panic below is the only unwind allowed to leave this boundary.
                        std::mem::forget(payload);
                        self.fail_injected_render_protocol();
                    }
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
                let _ = self.retire_control_receiver();
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
            TestGarbage { payload } => {
                if let Some(gc) = self.garbage_collector.as_mut() {
                    gc.push(payload);
                }
            }

            #[cfg(test)]
            TestNop => {}
        }

        ControlFlow::Continue(()) // continue handling more messages
    }

    #[cfg(test)]
    fn record_magic_bootstrap_shape_for_test(&mut self, message: &ControlMessage) {
        use crate::node::{ChannelCountMode, ChannelInterpretation};

        let position = self.magic_bootstrap_command_count;
        if position >= crate::context::MAGIC_COMMAND_COUNT {
            return;
        }
        let exact = match message {
            ControlMessage::RegisterNode {
                id,
                reclaim_id,
                inputs,
                outputs,
                channel_config,
                ..
            } if position <= 10 => {
                let expected_id = crate::context::AudioNodeId(position as u64);
                let (expected_inputs, expected_outputs, expected_count, expected_interpretation) =
                    match position {
                        0 => (
                            1,
                            1,
                            2.min(self.number_of_channels),
                            ChannelInterpretation::Speakers,
                        ),
                        1 => (0, 9, 1, ChannelInterpretation::Discrete),
                        _ => (1, 1, 1, ChannelInterpretation::Discrete),
                    };
                *id == expected_id
                    && **reclaim_id == expected_id
                    && *inputs == expected_inputs
                    && *outputs == expected_outputs
                    && channel_config.count == expected_count
                    && channel_config.count_mode == ChannelCountMode::Explicit
                    && channel_config.interpretation == expected_interpretation
            }
            ControlMessage::ConnectNode {
                from,
                to,
                output,
                input,
            } if (11..20).contains(&position) => {
                *from == crate::context::AudioNodeId((position - 9) as u64)
                    && *to == crate::context::AudioNodeId(1)
                    && *output == 0
                    && *input == usize::MAX
            }
            ControlMessage::ConnectNode {
                from,
                to,
                output,
                input,
            } if position == 20 => {
                *from == crate::context::AudioNodeId(1)
                    && *to == crate::context::AudioNodeId(0)
                    && *output == 0
                    && *input == usize::MAX
            }
            _ => false,
        };
        self.magic_bootstrap_shape[position] = exact;
        self.magic_bootstrap_command_count += 1;
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

    /// Runs node teardown and drops the graph on the calling non-render thread.
    ///
    /// Taking the graph first makes this idempotent even when a node hook panics: unwinding drops
    /// the detached graph here rather than later in `RenderThread::drop`.
    pub(crate) fn prepare_for_reclaim(&mut self) {
        #[cfg(test)]
        if self.fail_reclaim {
            panic!("forced injected renderer reclaim failure");
        }
        let Some(mut graph) = self.graph.take() else {
            return;
        };
        let current_frame = self.frames_played.load(Ordering::Relaxed);
        let current_time = current_frame as f64 / self.sample_rate as f64;

        let scope = AudioWorkletGlobalScope {
            current_frame,
            current_time,
            sample_rate: self.sample_rate,
            event_sender: self.event_sender.clone(),
            node_id: Cell::new(AudioNodeId(0)), // placeholder value
        };
        graph.before_drop(&scope);
    }

    /// Run destructors of all alive nodes in the audio graph.
    fn unload_graph(mut self) {
        self.prepare_for_reclaim();
    }

    pub fn render<S: FromSample<f32> + Clone>(&mut self, output_buffer: &mut [S]) {
        #[cfg(test)]
        if std::mem::take(&mut self.disconnect_lifecycle_on_next_render) {
            self.graph_lifecycle_publisher.take();
        }

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

        let mut rendered_samples = 0;
        let mut rendering_halted = false;
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

            // handle addition/removal of nodes/edges
            self.handle_control_messages_with_budget(control_budget);

            rendered_samples += data.len();
            rendering_halted = self.suspended || self.receiver.is_none() || self.graph.is_none();
            if rendering_halted {
                // A lifecycle acknowledgement may have been published by the control checkpoint.
                // Preserve the quantum that preceded it, but neither render nor replay any audio
                // beyond that authoritative transition.
                break;
            }

            if data.len() != chunk_size {
                // this is the last chunk, and it contained less than RENDER_QUANTUM_SIZE samples
                let channel_offset = data.len() / self.number_of_channels;
                debug_assert!(channel_offset < RENDER_QUANTUM_SIZE);
                self.buffer_offset = Some((channel_offset, destination_buffer));
            }
        }

        if rendering_halted {
            output_buffer[rendered_samples..].fill(S::from_sample_(0.));
        }
    }

    fn set_state(&self, state: AudioContextState) {
        let changed = self.injected_state.as_ref().is_none_or(|injected| {
            injected.transition_render(state) == InjectedStateTransition::Changed
        });
        if self.injected_state.is_none() {
            self.state.store(state as u8, Ordering::Relaxed);
        }
        if changed {
            self.event_sender
                .try_send(EventDispatch::state_change(state))
                .ok();
        }
    }

    /// Injected Suspend/Resume barriers acknowledge only after both the exact shared-state CAS
    /// and the corresponding event record are accepted by the exact event transport. Close is
    /// deliberately excluded: its terminal event remains proof-gated on the lifecycle thread.
    fn transition_injected_state_with_event(
        &self,
        state: AudioContextState,
    ) -> GraphLifecycleOutcome {
        let Some(injected) = self.injected_state.as_ref() else {
            return GraphLifecycleOutcome::ProtocolViolation;
        };
        if injected.transition_render(state) != InjectedStateTransition::Changed {
            return GraphLifecycleOutcome::ProtocolViolation;
        }
        match self
            .event_sender
            .try_send(EventDispatch::state_change(state))
        {
            Ok(()) => GraphLifecycleOutcome::Applied,
            Err(_) => GraphLifecycleOutcome::EventDeliveryFailed,
        }
    }

    fn store_state_without_event(&self, state: AudioContextState) {
        if let Some(injected) = &self.injected_state {
            let _ = injected.transition_render(state);
        } else {
            self.state.store(state as u8, Ordering::Release);
        }
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        if let Some(batch) = self.pending_control_batch.take() {
            self.reclaim_control_batch(batch);
        }
        let _ = self.retire_control_receiver();
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
fn try_spawn_garbage_collector_thread(
    consumer: llq::Consumer<Box<dyn Any + Send>>,
) -> io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new().spawn(move || run_garbage_collector_thread(consumer))
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
    use crossbeam_channel::Sender;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{self, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, ThreadId};

    use super::*;
    use crate::context::{
        injected_node_id_pair, InjectedExplicitConnect, InjectedExplicitDisconnect,
    };
    use crate::events::EventLoop;
    use crate::message::{
        graph_lifecycle_ack_pair, ControlBatchSendError, ControlBatchSender,
        GraphLifecycleSnapshot, GraphLifecycleWatcher, CONTROL_BATCH_CAPACITY,
        CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT,
    };
    use crate::node::{ChannelConfigInner, ChannelCountMode};
    use crate::render::graph::InjectedExplicitEdgeProtocolError;
    use crate::render::{AudioParamValues, AudioProcessor};

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

    struct GarbageCollectorDropProbe(SyncSender<ThreadId>);

    struct BarrierAfterFirstQuantumProcessor {
        control: Sender<ControlMessage>,
        barrier: GraphLifecycleBarrier,
        calls: Arc<AtomicUsize>,
    }

    struct ExplicitEdgeTestProcessor;

    impl AudioProcessor for ExplicitEdgeTestProcessor {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            _outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues<'_>,
            _scope: &AudioWorkletGlobalScope,
        ) -> bool {
            true
        }
    }

    impl AudioProcessor for BarrierAfterFirstQuantumProcessor {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues<'_>,
            _scope: &AudioWorkletGlobalScope,
        ) -> bool {
            outputs[0].set_number_of_channels(2);
            outputs[0]
                .channels_mut()
                .iter_mut()
                .for_each(|channel| channel.fill(0.75));
            if self.calls.fetch_add(1, Ordering::AcqRel) == 0 {
                self.control
                    .try_send(ControlMessage::GraphLifecycleBarrier(self.barrier))
                    .expect("test control channel has capacity");
            }
            true
        }

        fn has_side_effects(&self) -> bool {
            true
        }
    }

    impl Drop for GarbageCollectorDropProbe {
        fn drop(&mut self) {
            let _ = self.0.send(thread::current().id());
        }
    }

    fn harness(control_capacity: usize, event_capacity: usize) -> TestHarness {
        let (legacy, receiver) = crossbeam_channel::bounded(control_capacity);
        let retained_receiver = receiver.clone();
        let batches = ControlBatchSender::new(legacy.clone());
        let applied = ControlBatchApplied::default();
        let (event_sender, event_receiver) = crossbeam_channel::bounded(event_capacity);
        let (garbage_producer, garbage) = llq::Queue::new().split();
        let mut renderer = RenderThread::new_injected(
            48_000.,
            2,
            receiver,
            InjectedContextState::new_for_test(false),
            Arc::new(AtomicU64::new(0)),
            AudioStats::new(),
            InjectedEventDispatchSender::from_event_setup(
                event_sender.clone(),
                Arc::new(()),
                Arc::new(AtomicBool::new(true)),
            ),
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

    fn explicit_edge_test_graph(last_id: u64) -> Graph {
        let (_allocator, _owner, init) = injected_node_id_pair(100);
        let mut graph = init.into_graph();
        for id in 0..=last_id {
            graph.add_node(
                AudioNodeId(id),
                llq::Node::new(AudioNodeId(id)),
                Box::new(ExplicitEdgeTestProcessor),
                1,
                1,
                ChannelConfigInner {
                    count: 1,
                    count_mode: ChannelCountMode::Explicit,
                    interpretation: ChannelInterpretation::Discrete,
                },
            );
        }
        graph
    }

    fn install_explicit_edge_test_graph(renderer: &mut RenderThread) {
        renderer.graph = Some(explicit_edge_test_graph(4));
    }

    fn exact_connect(from: u64, to: u64) -> InjectedExplicitConnect {
        InjectedExplicitConnect::new_for_test(AudioNodeId(from), AudioNodeId(to), 0, 0)
    }

    fn exact_disconnect(from: u64, to: u64) -> InjectedExplicitDisconnect {
        InjectedExplicitDisconnect::new_for_test(AudioNodeId(from), AudioNodeId(to), 0, 0)
    }

    fn assert_exact_protocol_failure(
        graph: Graph,
        commands: Vec<ControlMessage>,
        verify_unchanged: impl FnOnce(&Graph),
    ) {
        let mut test = harness(1, 1);
        test.renderer.graph = Some(graph);
        let (reclaimed_send, reclaimed_recv) = crossbeam_channel::bounded(1);
        test.batches
            .try_send_exact_for_test(
                commands,
                Some(Arc::new(move || {
                    let _ = reclaimed_send.send(thread::current().id());
                })),
            )
            .unwrap();

        let render_thread = thread::current().id();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_callback(&mut test.renderer);
        }))
        .is_err());
        assert_eq!(test.applied.load(), 0, "failed sequence remains unapplied");
        assert!(test.applied.render_protocol_failed());
        assert!(test.renderer.pending_control_batch.is_some());
        verify_unchanged(test.renderer.graph.as_ref().unwrap());

        drop(test.renderer);
        let mut retired = Vec::new();
        while let Some(node) = test.garbage.pop() {
            retired.push(node);
        }
        thread::spawn(move || drop(retired)).join().unwrap();
        assert_ne!(
            reclaimed_recv.recv_timeout(Duration::from_secs(1)).unwrap(),
            render_thread
        );
        assert_eq!(test.batches.batch_storage_in_flight(), 0);
    }

    fn marker(value: u16, log: &Arc<Mutex<Vec<u16>>>) -> ControlMessage {
        ControlMessage::TestMarker {
            value,
            log: Arc::clone(log),
        }
    }

    fn lifecycle_barrier(
        sequence: u64,
        required_batch_sequence: u64,
        transition: GraphLifecycleTransition,
    ) -> GraphLifecycleBarrier {
        GraphLifecycleBarrier::new(
            NonZeroU64::new(sequence).expect("test lifecycle sequence is nonzero"),
            required_batch_sequence,
            transition,
        )
    }

    fn install_lifecycle_watcher(renderer: &mut RenderThread) -> GraphLifecycleWatcher {
        let (publisher, watcher) = graph_lifecycle_ack_pair();
        assert!(renderer.set_graph_lifecycle_publisher(publisher).is_ok());
        watcher
    }

    fn install_barrier_graph(
        renderer: &mut RenderThread,
        reclaim_id_channel: llq::Producer<AudioNodeId>,
        processor: BarrierAfterFirstQuantumProcessor,
    ) {
        let mut graph = Graph::new(reclaim_id_channel);
        graph.add_node(
            AudioNodeId(0),
            llq::Node::new(AudioNodeId(0)),
            Box::new(processor),
            0,
            1,
            ChannelConfigInner {
                count: 1,
                count_mode: ChannelCountMode::Explicit,
                interpretation: ChannelInterpretation::Discrete,
            },
        );
        renderer.graph = Some(graph);
        renderer.suspended = false;
        renderer
            .state
            .store(AudioContextState::Running as u8, Ordering::Release);
    }

    fn assert_applied(
        watcher: &GraphLifecycleWatcher,
        barrier: GraphLifecycleBarrier,
        observed_batch_sequence: u64,
        outcome: GraphLifecycleOutcome,
    ) {
        assert_eq!(
            watcher.snapshot(NonZeroU64::new(barrier.controller_sequence()).unwrap()),
            GraphLifecycleSnapshot::Applied {
                barrier,
                observed_batch_sequence,
                outcome,
            }
        );
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
        let lifecycle_barrier = lifecycle_barrier(1, 0, GraphLifecycleTransition::Suspend);
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
            ControlMessage::GraphLifecycleBarrier(lifecycle_barrier),
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
    fn lifecycle_barrier_wire_types_are_plain_copy_values() {
        fn assert_copy_send_sync<T: Copy + Send + Sync>() {}

        assert_copy_send_sync::<GraphLifecycleTransition>();
        assert_copy_send_sync::<GraphLifecycleBarrier>();
        assert_copy_send_sync::<GraphLifecycleOutcome>();
        assert!(!std::mem::needs_drop::<GraphLifecycleTransition>());
        assert!(!std::mem::needs_drop::<GraphLifecycleBarrier>());
        assert!(!std::mem::needs_drop::<GraphLifecycleOutcome>());
    }

    #[test]
    fn lifecycle_ack_snapshot_survives_empty_full_and_disconnected_wakes_without_allocating() {
        let mut test = harness(8, 1);
        let watcher = install_lifecycle_watcher(&mut test.renderer);

        let first = lifecycle_barrier(1, 0, GraphLifecycleTransition::Suspend);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(first))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_applied(&watcher, first, 0, GraphLifecycleOutcome::Applied);
        assert!(test.events.handle_pending_events());
        assert_eq!(watcher.receiver().try_recv(), Ok(()));

        let second = lifecycle_barrier(2, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(second))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_applied(&watcher, second, 0, GraphLifecycleOutcome::Applied);
        assert!(test.events.handle_pending_events());
        let third = lifecycle_barrier(3, 0, GraphLifecycleTransition::Suspend);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(third))
            .unwrap();
        // The wake for sequence two remains queued, so sequence three observes Full.
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_applied(&watcher, third, 0, GraphLifecycleOutcome::Applied);
        assert!(test.events.handle_pending_events());
        assert_eq!(
            watcher.snapshot(NonZeroU64::new(2).unwrap()),
            GraphLifecycleSnapshot::SequenceAdvanced {
                applied_sequence: 3
            }
        );
        assert_eq!(watcher.receiver().try_recv(), Ok(()));
        assert!(watcher.receiver().try_recv().is_err());

        let watcher = watcher.disconnect_wake_for_test();
        let fourth = lifecycle_barrier(4, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(fourth))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_applied(&watcher, fourth, 0, GraphLifecycleOutcome::Applied);
        assert!(test.events.handle_pending_events());
    }

    #[test]
    fn lifecycle_barrier_follows_deferred_atomic_batch_fifo_across_callback_budgets() {
        let mut test = harness(4, 4);
        let watcher = install_lifecycle_watcher(&mut test.renderer);
        let log = Arc::new(Mutex::new(Vec::new()));
        test.legacy.try_send(marker(0, &log)).unwrap();
        test.batches
            .try_send((1..=256).map(|value| marker(value, &log)).collect())
            .unwrap();
        let barrier = lifecycle_barrier(1, 1, GraphLifecycleTransition::Suspend);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(barrier))
            .unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &[0]);
        assert_eq!(test.applied.load(), 0);
        assert_eq!(
            watcher.snapshot(NonZeroU64::new(1).unwrap()),
            GraphLifecycleSnapshot::Pending
        );

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=256).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 1);
        assert!(!test.renderer.suspended);
        assert_eq!(
            watcher.snapshot(NonZeroU64::new(1).unwrap()),
            GraphLifecycleSnapshot::Pending
        );

        run_callback(&mut test.renderer);
        assert!(test.renderer.suspended);
        assert_applied(&watcher, barrier, 1, GraphLifecycleOutcome::Applied);
    }

    #[test]
    fn lifecycle_required_batch_gap_does_not_transition_and_is_allocation_free() {
        let mut test = harness(2, 1);
        let watcher = install_lifecycle_watcher(&mut test.renderer);
        test.renderer.suspended = true;
        test.renderer.state.store(
            AudioContextState::Suspended as u8,
            std::sync::atomic::Ordering::Release,
        );
        let barrier = lifecycle_barrier(1, 1, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(barrier))
            .unwrap();

        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));

        assert!(test.renderer.suspended);
        assert_eq!(
            test.renderer.state.load(Ordering::Acquire),
            AudioContextState::Suspended as u8
        );
        assert_applied(
            &watcher,
            barrier,
            0,
            GraphLifecycleOutcome::RequiredBatchPending,
        );
    }

    #[test]
    fn lifecycle_suspend_and_resume_transition_before_ack() {
        let mut test = harness(4, 1);
        let watcher = install_lifecycle_watcher(&mut test.renderer);
        test.renderer.state.store(
            AudioContextState::Running as u8,
            std::sync::atomic::Ordering::Relaxed,
        );

        let suspend = lifecycle_barrier(1, 0, GraphLifecycleTransition::Suspend);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(suspend))
            .unwrap();
        let mut output = [1.; 16];
        test.renderer.render(&mut output);
        assert!(test.renderer.suspended);
        assert!(output.iter().all(|sample| *sample == 0.));
        assert_eq!(
            test.renderer.state.load(Ordering::Acquire),
            AudioContextState::Suspended as u8
        );
        assert_applied(&watcher, suspend, 0, GraphLifecycleOutcome::Applied);
        assert!(test.events.handle_pending_events());
        watcher.receiver().try_recv().unwrap();

        let resume = lifecycle_barrier(2, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(resume))
            .unwrap();
        run_callback(&mut test.renderer);
        assert!(!test.renderer.suspended);
        assert_eq!(
            test.renderer.state.load(Ordering::Acquire),
            AudioContextState::Running as u8
        );
        assert_applied(&watcher, resume, 0, GraphLifecycleOutcome::Applied);
    }

    #[test]
    fn injected_state_barrier_reports_full_and_disconnected_event_delivery() {
        let mut full = harness(4, 1);
        let full_watcher = install_lifecycle_watcher(&mut full.renderer);
        full.event_sender
            .try_send(EventDispatch::sink_change())
            .unwrap();
        let suspend = lifecycle_barrier(1, 0, GraphLifecycleTransition::Suspend);
        full.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(suspend))
            .unwrap();
        run_callback(&mut full.renderer);
        assert!(full.renderer.suspended);
        assert_applied(
            &full_watcher,
            suspend,
            0,
            GraphLifecycleOutcome::EventDeliveryFailed,
        );

        let mut disconnected = harness(4, 1);
        let disconnected_watcher = install_lifecycle_watcher(&mut disconnected.renderer);
        drop(disconnected.events);
        let suspend = lifecycle_barrier(1, 0, GraphLifecycleTransition::Suspend);
        disconnected
            .legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(suspend))
            .unwrap();
        run_callback(&mut disconnected.renderer);
        assert!(disconnected.renderer.suspended);
        assert_applied(
            &disconnected_watcher,
            suspend,
            0,
            GraphLifecycleOutcome::EventDeliveryFailed,
        );
    }

    #[test]
    fn mid_callback_suspend_and_close_acknowledge_before_silent_suffix() {
        for transition in [
            GraphLifecycleTransition::Suspend,
            GraphLifecycleTransition::Close,
        ] {
            let (id_producer, _id_consumer) = llq::Queue::new().split();
            let mut test = harness(4, 1);
            let watcher = install_lifecycle_watcher(&mut test.renderer);
            let barrier = lifecycle_barrier(1, 0, transition);
            let calls = Arc::new(AtomicUsize::new(0));
            install_barrier_graph(
                &mut test.renderer,
                id_producer,
                BarrierAfterFirstQuantumProcessor {
                    control: test.legacy.clone(),
                    barrier,
                    calls: Arc::clone(&calls),
                },
            );
            let quantum_samples = RENDER_QUANTUM_SIZE * 2;
            let mut output = vec![-1.; quantum_samples * 3];

            test.renderer.render(&mut output);

            assert_eq!(calls.load(Ordering::Acquire), 1);
            assert!(output[..quantum_samples]
                .iter()
                .all(|sample| *sample == 0.75));
            assert!(output[quantum_samples..].iter().all(|sample| *sample == 0.));
            assert_applied(&watcher, barrier, 0, GraphLifecycleOutcome::Applied);
            assert_eq!(
                test.renderer.state.load(Ordering::Acquire),
                match transition {
                    GraphLifecycleTransition::Suspend => AudioContextState::Suspended as u8,
                    GraphLifecycleTransition::Close => AudioContextState::Closed as u8,
                    GraphLifecycleTransition::Resume => unreachable!(),
                }
            );

            // Once the acknowledgement is observable, no buffered or fresh graph quantum can
            // escape on a later callback either.
            output.fill(-1.);
            test.renderer.render(&mut output);
            assert!(output.iter().all(|sample| *sample == 0.));
            assert_eq!(calls.load(Ordering::Acquire), 1);
        }
    }

    #[test]
    fn lifecycle_close_retires_receiver_before_ack_and_reclaims_queue_off_rt() {
        let mut test = harness(3, 1);
        let watcher = install_lifecycle_watcher(&mut test.renderer);
        let retained = std::mem::replace(
            &mut test.retained_receiver,
            crossbeam_channel::never::<ControlMessage>(),
        );
        drop(retained);
        let close = lifecycle_barrier(1, 0, GraphLifecycleTransition::Close);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(close))
            .unwrap();
        let (dropped_send, dropped_recv) = mpsc::sync_channel(1);
        test.legacy
            .try_send(ControlMessage::NodeMessage {
                id: AudioNodeId(99),
                msg: llq::Node::new(
                    Box::new(GarbageCollectorDropProbe(dropped_send)) as Box<dyn Any + Send>
                ),
            })
            .unwrap();

        let callback_thread = thread::current().id();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert!(test.renderer.receiver.is_none());
        assert!(test.renderer.suspended);
        assert_eq!(
            test.renderer.state.load(Ordering::Acquire),
            AudioContextState::Closed as u8
        );
        assert!(dropped_recv.try_recv().is_err());
        assert_applied(&watcher, close, 0, GraphLifecycleOutcome::Applied);

        let retirement = test.garbage.pop().unwrap();
        // Lifecycle admission sealing drops the remaining control-side senders before GC
        // retirement; crossbeam retains queued payloads while a sender still owns the channel.
        drop(test.legacy);
        drop(test.batches);
        let reclaim_thread = thread::spawn(move || drop(retirement));
        reclaim_thread.join().unwrap();
        assert_ne!(
            dropped_recv.recv_timeout(Duration::from_secs(1)).unwrap(),
            callback_thread
        );
    }

    #[test]
    fn lifecycle_invalid_gap_reuse_and_exhaustion_are_fail_closed_and_allocation_free() {
        let mut missing = harness(2, 1);
        missing.renderer.suspended = true;
        let barrier = lifecycle_barrier(1, 0, GraphLifecycleTransition::Resume);
        missing
            .legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(barrier))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut missing.renderer));
        assert!(missing.renderer.suspended);

        let mut test = harness(8, 1);
        let watcher = install_lifecycle_watcher(&mut test.renderer);
        test.renderer.suspended = true;
        let zero =
            GraphLifecycleBarrier::from_raw_parts_for_test(0, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(zero))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert!(test.renderer.suspended);
        assert_eq!(
            watcher.snapshot(NonZeroU64::new(1).unwrap()),
            GraphLifecycleSnapshot::Pending
        );

        let first = lifecycle_barrier(1, 0, GraphLifecycleTransition::Suspend);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(first))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        watcher.receiver().try_recv().unwrap();
        let reuse = lifecycle_barrier(1, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(reuse))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert!(test.renderer.suspended);
        assert!(watcher.receiver().try_recv().is_err());

        let gap = lifecycle_barrier(3, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(gap))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert!(test.renderer.suspended);
        assert_applied(
            &watcher,
            gap,
            0,
            GraphLifecycleOutcome::ControllerSequenceGap,
        );
        watcher.receiver().try_recv().unwrap();
        let after_gap = lifecycle_barrier(4, 0, GraphLifecycleTransition::Resume);
        test.legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(after_gap))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert!(test.renderer.suspended);
        assert!(watcher.receiver().try_recv().is_err());

        let mut exhausted = harness(4, 1);
        let exhausted_watcher = install_lifecycle_watcher(&mut exhausted.renderer);
        exhausted.renderer.graph_lifecycle_next_sequence = u64::MAX;
        let maximum = lifecycle_barrier(u64::MAX, 0, GraphLifecycleTransition::Suspend);
        exhausted
            .legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(maximum))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut exhausted.renderer));
        assert_applied(
            &exhausted_watcher,
            maximum,
            0,
            GraphLifecycleOutcome::Applied,
        );
        exhausted_watcher.receiver().try_recv().unwrap();
        exhausted
            .legacy
            .try_send(ControlMessage::GraphLifecycleBarrier(maximum))
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut exhausted.renderer));
        assert!(exhausted_watcher.receiver().try_recv().is_err());
    }

    #[test]
    fn lifecycle_ack_slot_remains_consistent_over_many_publications() {
        let mut test = harness(2, 1);
        let watcher = install_lifecycle_watcher(&mut test.renderer);
        for sequence in 1..=512 {
            let transition = if sequence % 2 == 0 {
                GraphLifecycleTransition::Resume
            } else {
                GraphLifecycleTransition::Suspend
            };
            let barrier = lifecycle_barrier(sequence, 0, transition);
            test.legacy
                .try_send(ControlMessage::GraphLifecycleBarrier(barrier))
                .unwrap();
            run_callback(&mut test.renderer);
            assert_applied(&watcher, barrier, 0, GraphLifecycleOutcome::Applied);
            assert!(test.events.handle_pending_events());
            watcher.receiver().try_recv().unwrap();
        }
    }

    #[test]
    fn insufficient_budget_defers_atomic_batch_without_mutation_or_fifo_advance() {
        let mut test = harness(4, 4);
        let log = Arc::new(Mutex::new(Vec::with_capacity(258)));
        test.legacy.try_send(marker(0, &log)).unwrap();
        test.batches
            .try_send((1..=256).map(|value| marker(value, &log)).collect())
            .unwrap();
        test.legacy.try_send(marker(257, &log)).unwrap();

        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_eq!(&*log.lock().unwrap(), &[0]);
        assert_eq!(test.applied.load(), 0);
        let pending = test
            .renderer
            .pending_control_batch
            .as_ref()
            .expect("the deferred batch remains renderer-owned");
        assert_eq!(control_batch_storage(pending).remaining_len(), 256);
        assert_eq!(test.retained_receiver.len(), 1);

        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_eq!(&*log.lock().unwrap(), &(0..=256).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 1);
        assert!(test.renderer.pending_control_batch.is_none());
        assert_eq!(test.retained_receiver.len(), 1);

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=257).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 1);
    }

    #[test]
    fn atomic_batch_applies_when_it_exactly_fits_the_remaining_callback_budget() {
        let mut test = harness(4, 4);
        let log = Arc::new(Mutex::new(Vec::new()));
        test.legacy.try_send(marker(0, &log)).unwrap();
        test.batches
            .try_send((1..=255).map(|value| marker(value, &log)).collect())
            .unwrap();
        test.legacy.try_send(marker(256, &log)).unwrap();

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=255).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 1);
        assert!(test.renderer.pending_control_batch.is_none());
        assert_eq!(test.retained_receiver.len(), 1);

        run_callback(&mut test.renderer);
        assert_eq!(&*log.lock().unwrap(), &(0..=256).collect::<Vec<_>>());
        assert_eq!(test.applied.load(), 1);
    }

    #[test]
    fn sink_swap_replays_deferred_and_cached_fifo_without_watermark_gap() {
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
        assert_eq!(&*log.lock().unwrap(), &[0]);
        assert_eq!(old.applied.load(), 0);

        // Mirrors set_sink_id_sync: cache records not yet owned by the old renderer, then place
        // CloseAndRecycle after its deferred atomic batch.
        let cached: Vec<_> = old.retained_receiver.try_iter().collect();
        assert_eq!(cached.len(), 2);
        let (graph_send, graph_recv) = crossbeam_channel::bounded(1);
        old.legacy
            .send(ControlMessage::CloseAndRecycle { sender: graph_send })
            .unwrap();
        run_callback(&mut old.renderer);
        assert!(graph_recv.try_recv().is_err());
        assert_eq!(&*log.lock().unwrap(), &(0..=256).collect::<Vec<_>>());
        assert_eq!(old.applied.load(), 1);

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
    fn exact_connect_and_atomic_disconnect_batch_apply_without_render_allocation() {
        let mut test = harness(3, 3);
        install_explicit_edge_test_graph(&mut test.renderer);

        for edge in [exact_connect(1, 2), exact_connect(1, 3)] {
            test.batches
                .try_send_exact_for_test(vec![ControlMessage::InjectedConnectExplicit(edge)], None)
                .unwrap();
            alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        }
        assert_eq!(test.applied.load(), 2);
        let graph = test.renderer.graph.as_ref().unwrap();
        assert_eq!(
            graph.preflight_injected_explicit_connect(exact_connect(1, 2)),
            Err(InjectedExplicitEdgeProtocolError::Duplicate)
        );
        assert_eq!(
            graph.preflight_injected_explicit_connect(exact_connect(1, 3)),
            Err(InjectedExplicitEdgeProtocolError::Duplicate)
        );

        test.batches
            .try_send_exact_for_test(
                vec![
                    ControlMessage::InjectedDisconnectExplicit(exact_disconnect(1, 2)),
                    ControlMessage::InjectedDisconnectExplicit(exact_disconnect(1, 3)),
                ],
                None,
            )
            .unwrap();
        alloc_counter::deny_alloc(|| run_callback(&mut test.renderer));
        assert_eq!(test.applied.load(), 3);
        let graph = test.renderer.graph.as_ref().unwrap();
        graph
            .preflight_injected_explicit_connect(exact_connect(1, 2))
            .unwrap();
        graph
            .preflight_injected_explicit_connect(exact_connect(1, 3))
            .unwrap();
    }

    #[test]
    fn exact_records_are_not_general_batchable() {
        let test = harness(1, 1);
        assert_eq!(
            test.batches
                .try_send(vec![ControlMessage::InjectedConnectExplicit(
                    exact_connect(1, 2)
                )]),
            Err(ControlBatchSendError::UnsupportedCommand)
        );
        assert_eq!(
            test.batches
                .try_send(vec![ControlMessage::InjectedDisconnectExplicit(
                    exact_disconnect(1, 2)
                )]),
            Err(ControlBatchSendError::UnsupportedCommand)
        );
    }

    #[test]
    fn every_exact_edge_divergence_latches_before_mutation_and_reclaims_off_rt() {
        let mut duplicate = explicit_edge_test_graph(4);
        duplicate
            .preflight_injected_explicit_connect(exact_connect(1, 2))
            .unwrap();
        duplicate.apply_injected_explicit_connect(exact_connect(1, 2));
        assert_exact_protocol_failure(
            duplicate,
            vec![ControlMessage::InjectedConnectExplicit(exact_connect(1, 2))],
            |graph| {
                assert_eq!(
                    graph.preflight_injected_explicit_connect(exact_connect(1, 2)),
                    Err(InjectedExplicitEdgeProtocolError::Duplicate)
                );
            },
        );

        let mut capacity =
            explicit_edge_test_graph(crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS as u64 + 2);
        for offset in 0..crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS {
            let to = if offset == 0 { 0 } else { offset as u64 + 1 };
            let edge = exact_connect(1, to);
            capacity.preflight_injected_explicit_connect(edge).unwrap();
            capacity.apply_injected_explicit_connect(edge);
        }
        let overflow = exact_connect(
            1,
            crate::context::MAX_INJECTED_EXPLICIT_CONNECTIONS as u64 + 1,
        );
        assert_exact_protocol_failure(
            capacity,
            vec![ControlMessage::InjectedConnectExplicit(overflow)],
            |graph| {
                assert_eq!(
                    graph.preflight_injected_explicit_connect(overflow),
                    Err(InjectedExplicitEdgeProtocolError::Capacity)
                );
                assert_eq!(
                    graph.preflight_injected_explicit_connect(exact_connect(1, 0)),
                    Err(InjectedExplicitEdgeProtocolError::Duplicate)
                );
            },
        );

        for invalid in [
            InjectedExplicitConnect::new_for_test(AudioNodeId(1), AudioNodeId(99), 0, 0),
            InjectedExplicitConnect::new_for_test(AudioNodeId(1), AudioNodeId(2), 1, 0),
            InjectedExplicitConnect::new_for_test(AudioNodeId(1), AudioNodeId(2), 0, usize::MAX),
        ] {
            assert_exact_protocol_failure(
                explicit_edge_test_graph(4),
                vec![ControlMessage::InjectedConnectExplicit(invalid)],
                |graph| {
                    graph
                        .preflight_injected_explicit_connect(exact_connect(1, 2))
                        .unwrap();
                },
            );
        }

        for invalid in [
            InjectedExplicitDisconnect::new_for_test(AudioNodeId(1), AudioNodeId(2), 1, 0),
            InjectedExplicitDisconnect::new_for_test(AudioNodeId(1), AudioNodeId(2), 0, usize::MAX),
        ] {
            let mut graph = explicit_edge_test_graph(4);
            graph
                .preflight_injected_explicit_connect(exact_connect(1, 2))
                .unwrap();
            graph.apply_injected_explicit_connect(exact_connect(1, 2));
            assert_exact_protocol_failure(
                graph,
                vec![ControlMessage::InjectedDisconnectExplicit(invalid)],
                |graph| {
                    assert_eq!(
                        graph.preflight_injected_explicit_connect(exact_connect(1, 2)),
                        Err(InjectedExplicitEdgeProtocolError::Duplicate)
                    );
                },
            );
        }

        let mut missing = explicit_edge_test_graph(4);
        missing
            .preflight_injected_explicit_connect(exact_connect(1, 2))
            .unwrap();
        missing.apply_injected_explicit_connect(exact_connect(1, 2));
        assert_exact_protocol_failure(
            missing,
            vec![ControlMessage::InjectedDisconnectExplicit(
                exact_disconnect(1, 3),
            )],
            |graph| {
                assert_eq!(
                    graph.preflight_injected_explicit_connect(exact_connect(1, 2)),
                    Err(InjectedExplicitEdgeProtocolError::Duplicate)
                );
            },
        );

        let mut duplicate_disconnect = explicit_edge_test_graph(4);
        duplicate_disconnect
            .preflight_injected_explicit_connect(exact_connect(1, 2))
            .unwrap();
        duplicate_disconnect.apply_injected_explicit_connect(exact_connect(1, 2));
        assert_exact_protocol_failure(
            duplicate_disconnect,
            vec![
                ControlMessage::InjectedDisconnectExplicit(exact_disconnect(1, 2)),
                ControlMessage::InjectedDisconnectExplicit(exact_disconnect(1, 2)),
            ],
            |graph| {
                assert_eq!(
                    graph.preflight_injected_explicit_connect(exact_connect(1, 2)),
                    Err(InjectedExplicitEdgeProtocolError::Duplicate)
                );
            },
        );
    }

    #[test]
    fn exact_disconnect_divergence_latches_before_removing_any_present_edge() {
        let mut test = harness(3, 3);
        install_explicit_edge_test_graph(&mut test.renderer);
        for edge in [exact_connect(1, 2), exact_connect(1, 3)] {
            test.batches
                .try_send_exact_for_test(vec![ControlMessage::InjectedConnectExplicit(edge)], None)
                .unwrap();
            run_callback(&mut test.renderer);
        }
        assert_eq!(test.applied.load(), 2);

        test.batches
            .try_send_exact_for_test(
                vec![
                    ControlMessage::InjectedDisconnectExplicit(exact_disconnect(1, 2)),
                    ControlMessage::InjectedDisconnectExplicit(exact_disconnect(2, 3)),
                ],
                None,
            )
            .unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_callback(&mut test.renderer);
        }))
        .is_err());
        assert_eq!(test.applied.load(), 2);
        assert!(test.applied.render_protocol_failed());
        let graph = test.renderer.graph.as_ref().unwrap();
        assert_eq!(
            graph.preflight_injected_explicit_connect(exact_connect(1, 2)),
            Err(InjectedExplicitEdgeProtocolError::Duplicate),
            "the present prefix edge was not removed before the later mismatch"
        );
        assert_eq!(
            graph.preflight_injected_explicit_connect(exact_connect(1, 3)),
            Err(InjectedExplicitEdgeProtocolError::Duplicate)
        );

        drop(test.renderer);
        let mut retired = Vec::new();
        while let Some(node) = test.garbage.pop() {
            retired.push(node);
        }
        thread::spawn(move || drop(retired)).join().unwrap();
    }

    #[test]
    fn mixed_exact_batch_fails_before_ordinary_prefix_and_reclaims_off_render_thread() {
        let mut test = harness(1, 1);
        install_explicit_edge_test_graph(&mut test.renderer);
        let log = Arc::new(Mutex::new(Vec::new()));
        let (reclaimed_send, reclaimed_recv) = crossbeam_channel::bounded(1);
        test.batches
            .try_send_exact_for_test(
                vec![
                    marker(7, &log),
                    ControlMessage::InjectedConnectExplicit(exact_connect(1, 2)),
                ],
                Some(Arc::new(move || {
                    let _ = reclaimed_send.send(thread::current().id());
                })),
            )
            .unwrap();

        let render_thread = thread::current().id();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_callback(&mut test.renderer);
        }))
        .is_err());
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(test.applied.load(), 0, "failed sequence remains unapplied");
        assert!(test.applied.render_protocol_failed());
        assert!(test.renderer.pending_control_batch.is_some());
        assert_eq!(test.batches.batch_storage_in_flight(), 1);

        drop(test.renderer);
        let mut retired = Vec::new();
        while let Some(node) = test.garbage.pop() {
            retired.push(node);
        }
        thread::spawn(move || drop(retired)).join().unwrap();
        assert_ne!(
            reclaimed_recv.recv_timeout(Duration::from_secs(1)).unwrap(),
            render_thread
        );
        assert_eq!(test.batches.batch_storage_in_flight(), 0);
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
    fn dropping_deferred_atomic_batch_does_not_advance_watermark() {
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
        let pending = test.renderer.pending_control_batch.as_ref().unwrap();
        assert_eq!(control_batch_storage(pending).remaining_len(), 256);
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

    #[test]
    fn joinable_garbage_collector_confirms_off_thread_reclamation() {
        let (_sender, receiver) = crossbeam_channel::unbounded();
        let (event_sender, _event_receiver) = crossbeam_channel::unbounded();
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
        let join = renderer.spawn_joinable_garbage_collector_thread().unwrap();
        let collector_thread = join.thread().id();
        let (dropped_send, dropped_recv) = mpsc::sync_channel(1);
        renderer
            .garbage_collector
            .as_mut()
            .unwrap()
            .push(llq::Node::new(
                Box::new(GarbageCollectorDropProbe(dropped_send)) as Box<dyn Any + Send>,
            ));

        drop(renderer);
        join.join().unwrap();
        assert_eq!(dropped_recv.recv().unwrap(), collector_thread);
    }
}
