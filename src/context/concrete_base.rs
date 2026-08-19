//! The `ConcreteBaseAudioContext` type

use crate::context::injected_connections::{
    InjectedConnectionEndpointKind, InjectedMagicConnectionEndpoints,
};
use crate::context::injected_control::InjectedConcreteEventBinding;
use crate::context::injected_magic_construction::{
    InjectedMagicConstructionError, InjectedMagicGraph,
};
use crate::context::injected_node_construction::InjectedNodeConstructor;
use crate::context::injected_node_lifetime::{
    BoundInjectedOutputRenderer, MagicInitializedInjectedOutputRenderer,
};
use crate::context::{
    AdmissionError, AudioContextRegistration, AudioContextState, AudioNodeId, BaseAudioContext,
    InjectedContextAdmissionGate, DESTINATION_NODE_ID, LISTENER_NODE_ID, LISTENER_PARAM_IDS,
};
use crate::events::{
    EventDispatch, EventHandler, EventLoop, EventType, InjectedControlEventDispatch,
    InjectedControlEventSendError, InjectedStateTransition,
};
use crate::message::{ControlBatchApplied, ControlBatchSender, ControlMessage};
use crate::node::{AudioDestinationNode, AudioNode, AudioNodeOptions, ChannelConfig};
use crate::param::AudioParam;
use crate::render::AudioProcessor;
use crate::spatial::AudioListenerParams;

use crate::AudioListener;

use crossbeam_channel::{Sender, TrySendError};
use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

/// Control-side authority for submitting public event records.
///
/// Legacy and offline contexts retain their blocking sender behavior. The private injected form
/// never exposes its raw sender: every attempt first acquires a short admission and holds it until
/// a rejected record has been destroyed on the caller thread. B4a's exact variant is derived from
/// the same single-use setup as the renderer and lifecycle consumer. The raw injected variant is
/// retained only for earlier admission tests.
#[derive(Clone)]
pub(crate) struct ControlEventDispatch {
    mode: ControlEventDispatchMode,
}

#[derive(Clone)]
#[allow(dead_code)] // the exact private variant awaits context assembly; raw injection is test-only
enum ControlEventDispatchMode {
    Legacy(Sender<EventDispatch>),
    Exact(InjectedControlEventDispatch),
    #[cfg(test)]
    Injected {
        sender: Sender<EventDispatch>,
        gate: InjectedContextAdmissionGate,
        #[cfg(test)]
        after_admission: Option<Arc<dyn Fn() + Send + Sync>>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlEventSendOutcome {
    Delivered,
    Full,
    Disconnected,
    AdmissionRejected(AdmissionError),
}

impl ControlEventDispatch {
    fn legacy(sender: Sender<EventDispatch>) -> Self {
        Self {
            mode: ControlEventDispatchMode::Legacy(sender),
        }
    }

    #[cfg(test)]
    pub(crate) fn legacy_for_test(sender: Sender<EventDispatch>) -> Self {
        Self::legacy(sender)
    }

    /// Earlier test-only precursor. Production injection never accepts a raw control-side sender.
    #[cfg(test)]
    pub(crate) fn injected(
        sender: Sender<EventDispatch>,
        gate: InjectedContextAdmissionGate,
    ) -> Self {
        Self {
            mode: ControlEventDispatchMode::Injected {
                sender,
                gate,
                #[cfg(test)]
                after_admission: None,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn injected_with_observer(
        sender: Sender<EventDispatch>,
        gate: InjectedContextAdmissionGate,
        after_admission: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        Self {
            mode: ControlEventDispatchMode::Injected {
                sender,
                gate,
                after_admission: Some(after_admission),
            },
        }
    }

    pub(crate) fn injected_gate(&self) -> Option<InjectedContextAdmissionGate> {
        match &self.mode {
            ControlEventDispatchMode::Legacy(_) => None,
            ControlEventDispatchMode::Exact(events) => Some(events.admission_gate()),
            #[cfg(test)]
            ControlEventDispatchMode::Injected { gate, .. } => Some(gate.clone()),
        }
    }

    /// Submit an ordinary control-side event.
    ///
    /// This preserves the legacy blocking send. Injected contexts are always non-waiting and can
    /// drop an event on phase-lock contention or a full event queue.
    pub(crate) fn send_with<F>(&self, make_event: F) -> ControlEventSendOutcome
    where
        F: FnOnce() -> EventDispatch + Copy,
    {
        match &self.mode {
            ControlEventDispatchMode::Legacy(sender) => match sender.send(make_event()) {
                Ok(()) => ControlEventSendOutcome::Delivered,
                Err(error) => {
                    drop(error.into_inner());
                    ControlEventSendOutcome::Disconnected
                }
            },
            ControlEventDispatchMode::Exact(events) => {
                map_injected_send(events.try_send_with(make_event))
            }
            #[cfg(test)]
            ControlEventDispatchMode::Injected { .. } => self.try_send_injected(make_event),
        }
    }

    /// Submit a diagnostic event without waiting for queue capacity.
    pub(crate) fn try_send_with<F>(&self, make_event: F) -> ControlEventSendOutcome
    where
        F: FnOnce() -> EventDispatch + Copy,
    {
        match &self.mode {
            ControlEventDispatchMode::Legacy(sender) => match sender.try_send(make_event()) {
                Ok(()) => ControlEventSendOutcome::Delivered,
                Err(TrySendError::Full(event)) => {
                    drop(event);
                    ControlEventSendOutcome::Full
                }
                Err(TrySendError::Disconnected(event)) => {
                    drop(event);
                    ControlEventSendOutcome::Disconnected
                }
            },
            ControlEventDispatchMode::Exact(events) => {
                map_injected_send(events.try_send_with(make_event))
            }
            #[cfg(test)]
            ControlEventDispatchMode::Injected { .. } => self.try_send_injected(make_event),
        }
    }

    #[cfg(test)]
    fn try_send_injected<F>(&self, make_event: F) -> ControlEventSendOutcome
    where
        F: FnOnce() -> EventDispatch + Copy,
    {
        let ControlEventDispatchMode::Injected {
            sender,
            gate,
            #[cfg(test)]
            after_admission,
        } = &self.mode
        else {
            unreachable!("injected event submission requires injected capability")
        };

        let admission = match gate.try_external_event() {
            Ok(admission) => admission,
            Err(error) => return ControlEventSendOutcome::AdmissionRejected(error),
        };
        #[cfg(test)]
        if let Some(observer) = after_admission {
            observer();
        }

        // Construct the record only after admission. `Copy` structurally excludes an owned
        // destructor-bearing capture, so rejecting admission can discard the uninvoked factory
        // without hidden cleanup. Event payload ownership begins within the admitted region and
        // rejected channel payloads are destroyed before the permit is released.
        let outcome = match sender.try_send(make_event()) {
            Ok(()) => ControlEventSendOutcome::Delivered,
            Err(TrySendError::Full(event)) => {
                // The short permit deliberately covers caller-side destruction of a rejected
                // payload; sealing cannot claim this producer drained while cleanup is running.
                drop(event);
                ControlEventSendOutcome::Full
            }
            Err(TrySendError::Disconnected(event)) => {
                drop(event);
                ControlEventSendOutcome::Disconnected
            }
        };
        drop(admission);
        outcome
    }

    #[allow(dead_code)] // selected by pending private injected-context assembly
    fn exact(events: InjectedControlEventDispatch) -> Self {
        Self {
            mode: ControlEventDispatchMode::Exact(events),
        }
    }

    fn exact_events(&self) -> Option<&InjectedControlEventDispatch> {
        match &self.mode {
            ControlEventDispatchMode::Exact(events) => Some(events),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn set_exact_after_admission_for_test(&self, observer: Arc<dyn Fn() + Send + Sync>) {
        self.exact_events()
            .expect("exact admission observer requires exact control events")
            .set_after_admission_for_test(observer);
    }
}

fn map_injected_send(
    outcome: Result<(), InjectedControlEventSendError>,
) -> ControlEventSendOutcome {
    match outcome {
        Ok(()) => ControlEventSendOutcome::Delivered,
        Err(InjectedControlEventSendError::Full) => ControlEventSendOutcome::Full,
        Err(InjectedControlEventSendError::Disconnected) => ControlEventSendOutcome::Disconnected,
        Err(InjectedControlEventSendError::Admission(error)) => {
            ControlEventSendOutcome::AdmissionRejected(error)
        }
    }
}

/// This struct assigns new [`AudioNodeId`]s for [`AudioNode`]s
///
/// It reuses the ids of decommissioned nodes to prevent unbounded growth of the audio graphs node
/// list (which is stored in a Vec indexed by the AudioNodeId).
struct AudioNodeIdProvider {
    /// incrementing id
    id_inc: AtomicU64,
    /// receiver for decommissioned AudioNodeIds, which can be reused
    id_consumer: Mutex<llq::Consumer<AudioNodeId>>,
}

impl AudioNodeIdProvider {
    fn new(id_consumer: llq::Consumer<AudioNodeId>) -> Self {
        Self {
            id_inc: AtomicU64::new(0),
            id_consumer: Mutex::new(id_consumer),
        }
    }

    fn get(&self) -> AudioNodeId {
        if let Some(available_id) = self.id_consumer.lock().unwrap().pop() {
            llq::Node::into_inner(available_id)
        } else {
            AudioNodeId(self.id_inc.fetch_add(1, Ordering::Relaxed))
        }
    }
}

/// The struct that corresponds to the Javascript `BaseAudioContext` object.
///
/// This object is returned from the `base()` method on
/// [`AudioContext`](crate::context::AudioContext) and
/// [`OfflineAudioContext`](crate::context::OfflineAudioContext), and the `context()` method on
/// `AudioNode`s.
///
/// The `ConcreteBaseAudioContext` allows for shallow cloning (using an `Arc` internally).
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
#[doc(hidden)]
pub struct ConcreteBaseAudioContext {
    inner: Arc<ConcreteBaseAudioContextInner>,
}

impl PartialEq for ConcreteBaseAudioContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::fmt::Debug for ConcreteBaseAudioContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseAudioContext")
            .field("id", &self.address())
            .field("state", &self.state())
            .field("sample_rate", &self.sample_rate())
            .field("current_time", &self.current_time())
            .field("max_channel_count", &self.max_channel_count())
            .field("offline", &self.offline())
            .finish_non_exhaustive()
    }
}

/// Inner representation of the `ConcreteBaseAudioContext`
///
/// These fields are wrapped inside an `Arc` in the actual `ConcreteBaseAudioContext`.
struct LegacyGraphControl {
    /// provider for new AudioNodeIds
    audio_node_id_provider: AudioNodeIdProvider,
    /// message channel from control to render thread
    render_channel: RwLock<Sender<ControlMessage>>,
    /// Private bounded-batch transport retained by legacy contexts during migration.
    _control_batch_sender: ControlBatchSender,
    /// Authoritative render-side acknowledgement for private control batches.
    control_batch_applied: ControlBatchApplied,
    /// control messages staged while the render thread is suspended
    suspended_messages: Mutex<Option<Vec<ControlMessage>>>,
    /// control messages that cannot be sent immediately
    queued_messages: Mutex<Vec<ControlMessage>>,
    /// control msg to add the AudioListener, to be sent when the first panner is created
    queued_audio_listener_msgs: Mutex<Vec<ControlMessage>>,
    /// Current audio graph connections (from node, output port, to node, input port)
    connections: Mutex<HashSet<(AudioNodeId, usize, AudioNodeId, usize)>>,
}

// The whole base is already behind one Arc; boxing legacy state again would add an allocation to
// every existing context merely to shrink this private discriminated field.
#[allow(clippy::large_enum_variant)]
#[allow(dead_code)] // injected base selection remains test-only until the public context slice
enum ConcreteGraphControl {
    Legacy(LegacyGraphControl),
    /// Exact branded batch/id/lifetime capability; deliberately contains no raw render sender or
    /// legacy ID provider. Only the Gain constructor is migrated at this private boundary.
    Injected(InjectedNodeConstructor),
}

struct ConcreteBaseAudioContextInner {
    /// sample rate in Hertz
    sample_rate: f32,
    /// max number of speaker output channels
    max_channel_count: usize,
    graph_control: ConcreteGraphControl,
    /// destination node's current channel count
    destination_channel_config: ChannelConfig,
    /// number of frames played
    frames_played: Arc<AtomicU64>,
    /// AudioListener fields
    listener_params: Option<AudioListenerParams>,
    /// Accepted permanent destination/listener-param endpoint brands. Legacy and the test-only
    /// magic-less construction harness deliberately carry none.
    magic_connections: Option<InjectedMagicConnectionEndpoints>,
    /// Denotes if this AudioContext is offline or not
    offline: bool,
    /// Current state of the `ConcreteBaseAudioContext`, shared with the RenderThread
    state: ConcreteContextState,
    /// Stores the event handlers
    event_handlers: ConcreteEventHandlers,
    /// Opaque control-side capability for events handled by the EventLoop.
    control_events: ControlEventDispatch,
}

#[allow(dead_code)] // exact variant is selected by pending private injected-context assembly
enum ConcreteContextState {
    Legacy(Arc<AtomicU8>),
    Injected,
}

#[allow(dead_code)] // exact variant is selected by pending private injected-context assembly
enum ConcreteEventHandlers {
    Legacy(EventLoop),
    Injected(InjectedControlEventDispatch),
}

/// Exact, prevalidated base-construction authority.
///
/// Constructing this value validates the event/control gate before the magic transaction may
/// acquire admission, reserve IDs, prewarm HRTF, or allocate graph payloads.
#[must_use]
#[allow(dead_code)] // private prerequisite selected by the deferred injected context builder
pub(crate) struct ExactInjectedBaseBootstrap {
    renderer: BoundInjectedOutputRenderer,
    constructor: InjectedNodeConstructor,
    binding: InjectedConcreteEventBinding,
}

#[allow(dead_code)] // exact mismatch recovery is exercised before public builder wiring
pub(crate) struct PrepareExactInjectedBaseFailure {
    renderer: BoundInjectedOutputRenderer,
    constructor: InjectedNodeConstructor,
    binding: InjectedConcreteEventBinding,
}

#[allow(dead_code)]
impl PrepareExactInjectedBaseFailure {
    pub(crate) fn into_parts(
        self,
    ) -> (
        BoundInjectedOutputRenderer,
        InjectedNodeConstructor,
        InjectedConcreteEventBinding,
    ) {
        (self.renderer, self.constructor, self.binding)
    }
}

#[allow(dead_code)] // private prerequisite selected by the deferred injected context builder
pub(crate) enum ExactInjectedBaseBuildFailure {
    Retryable {
        error: InjectedMagicConstructionError,
        bootstrap: ExactInjectedBaseBootstrap,
    },
    Terminal {
        error: InjectedMagicConstructionError,
        renderer: BoundInjectedOutputRenderer,
    },
}

#[allow(dead_code)]
impl ExactInjectedBaseBuildFailure {
    pub(crate) const fn error(&self) -> InjectedMagicConstructionError {
        match self {
            Self::Retryable { error, .. } | Self::Terminal { error, .. } => *error,
        }
    }

    pub(crate) fn into_retryable_parts(
        self,
    ) -> Option<(
        BoundInjectedOutputRenderer,
        InjectedNodeConstructor,
        InjectedConcreteEventBinding,
    )> {
        match self {
            Self::Retryable { bootstrap, .. } => {
                Some((bootstrap.renderer, bootstrap.constructor, bootstrap.binding))
            }
            Self::Terminal { .. } => None,
        }
    }

    pub(crate) fn into_terminal_renderer(self) -> Option<BoundInjectedOutputRenderer> {
        match self {
            Self::Terminal { renderer, .. } => Some(renderer),
            Self::Retryable { .. } => None,
        }
    }
}

impl BaseAudioContext for ConcreteBaseAudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        self
    }
}

#[allow(dead_code)]
impl ExactInjectedBaseBootstrap {
    /// Atomically installs the permanent magic graph, then assembles the host base from closed
    /// moves only. The only success-side allocation is the final `Arc`; allocation failure aborts
    /// the process and is not a recoverable unwind seam that could detach accepted graph state.
    #[allow(clippy::result_large_err)] // exact recovery returns every unique owner inline
    pub(crate) fn try_build(
        self,
    ) -> Result<MagicInitializedInjectedOutputRenderer, ExactInjectedBaseBuildFailure> {
        let (sample_rate, max_channel_count, frames_played) = self.renderer.injected_base_facts();
        let offline = false;
        let magic = match self.constructor.try_construct_magic_graph(
            sample_rate,
            max_channel_count,
            offline,
        ) {
            Ok(magic) => magic,
            Err(failure) if !failure.retryable => {
                return Err(ExactInjectedBaseBuildFailure::Terminal {
                    error: failure.error,
                    renderer: self.renderer,
                })
            }
            Err(failure) => {
                return Err(ExactInjectedBaseBuildFailure::Retryable {
                    error: failure.error,
                    bootstrap: self,
                })
            }
        };
        if !magic.matches_constructor(&self.constructor) {
            // This is an internal proof mismatch after render ownership was accepted. Never offer
            // either capability as a fresh construction retry.
            drop(magic);
            return Err(ExactInjectedBaseBuildFailure::Terminal {
                error: InjectedMagicConstructionError::ProtocolViolation,
                renderer: self.renderer,
            });
        }
        Self::finish_build(
            self,
            magic,
            sample_rate,
            max_channel_count,
            frames_played,
            offline,
        )
    }

    #[allow(clippy::result_large_err)] // exact terminal failure returns the render owner inline
    fn finish_build(
        self,
        magic: InjectedMagicGraph,
        sample_rate: f32,
        max_channel_count: usize,
        frames_played: Arc<AtomicU64>,
        offline: bool,
    ) -> Result<MagicInitializedInjectedOutputRenderer, ExactInjectedBaseBuildFailure> {
        let Self {
            mut renderer,
            constructor,
            binding,
        } = self;
        let (
            destination_channel_config,
            listener_params,
            outcome,
            magic_connections,
            installed_magic,
        ) = magic.into_host_parts();
        let required_sequence = match outcome {
            crate::context::injected_control::CommitControlOutcome::Enqueued { sequence } => {
                sequence
            }
            crate::context::injected_control::CommitControlOutcome::Staged => {
                let flushed = match constructor.try_flush_staged() {
                    Ok(flushed) => flushed,
                    Err(error) => {
                        return Err(ExactInjectedBaseBuildFailure::Terminal {
                            error: InjectedMagicConstructionError::Control(error),
                            renderer,
                        })
                    }
                };
                if flushed.enqueued != 1 || flushed.remaining_staged != 0 {
                    return Err(ExactInjectedBaseBuildFailure::Terminal {
                        error: InjectedMagicConstructionError::ProtocolViolation,
                        renderer,
                    });
                }
                constructor.last_submitted_batch_sequence()
            }
        };
        let applied = if required_sequence == 0 {
            false
        } else {
            match panic::catch_unwind(AssertUnwindSafe(|| {
                renderer.apply_magic_before_publication(required_sequence)
            })) {
                Ok(applied) => applied,
                Err(payload) => {
                    std::mem::forget(payload);
                    false
                }
            }
        };
        if !applied {
            return Err(ExactInjectedBaseBuildFailure::Terminal {
                error: InjectedMagicConstructionError::ProtocolViolation,
                renderer,
            });
        }
        let events = binding.into_events();
        let base = ConcreteBaseAudioContext {
            inner: Arc::new(ConcreteBaseAudioContextInner {
                sample_rate,
                max_channel_count,
                graph_control: ConcreteGraphControl::Injected(constructor),
                destination_channel_config,
                frames_played,
                listener_params: Some(listener_params),
                magic_connections: Some(magic_connections),
                offline,
                state: ConcreteContextState::Injected,
                event_handlers: ConcreteEventHandlers::Injected(events.clone()),
                control_events: ControlEventDispatch::exact(events),
            }),
        };
        renderer
            .try_finish_magic_initialization(base, installed_magic)
            .map_err(
                |(renderer, _base)| ExactInjectedBaseBuildFailure::Terminal {
                    error: InjectedMagicConstructionError::ProtocolViolation,
                    renderer,
                },
            )
    }
}

impl ConcreteBaseAudioContext {
    /// Validates the exact B4a event branch against the graph constructor without mutating either
    /// transport. A mismatch returns both values intact and operationally reusable.
    #[allow(dead_code)] // private prerequisite selected by the deferred injected context builder
    #[allow(clippy::result_large_err)] // exact mismatch returns all three owners inline
    pub(crate) fn try_prepare_exact_injected_base(
        renderer: BoundInjectedOutputRenderer,
        constructor: InjectedNodeConstructor,
        binding: InjectedConcreteEventBinding,
    ) -> Result<ExactInjectedBaseBootstrap, PrepareExactInjectedBaseFailure> {
        if !binding.matches_constructor(&constructor) || !renderer.matches_constructor(&constructor)
        {
            return Err(PrepareExactInjectedBaseFailure {
                renderer,
                constructor,
                binding,
            });
        }
        Ok(ExactInjectedBaseBootstrap {
            renderer,
            constructor,
            binding,
        })
    }

    #[cfg(test)]
    fn handle_pending_events_for_test(&self) -> bool {
        match &self.inner.event_handlers {
            ConcreteEventHandlers::Legacy(event_loop) => event_loop.handle_pending_events(),
            ConcreteEventHandlers::Injected(_) => {
                panic!("exact injected events are owned by their dedicated event thread")
            }
        }
    }
    /// Creates a `BaseAudioContext` instance
    #[allow(clippy::too_many_arguments)] // TODO refactor with builder pattern
    pub(super) fn new(
        sample_rate: f32,
        max_channel_count: usize,
        state: Arc<AtomicU8>,
        frames_played: Arc<AtomicU64>,
        render_channel: Sender<ControlMessage>,
        control_batch_sender: ControlBatchSender,
        control_batch_applied: ControlBatchApplied,
        event_send: Sender<EventDispatch>,
        event_loop: EventLoop,
        offline: bool,
        node_id_consumer: llq::Consumer<AudioNodeId>,
    ) -> Self {
        Self::new_with_control_events(
            sample_rate,
            max_channel_count,
            state,
            frames_played,
            render_channel,
            control_batch_sender,
            control_batch_applied,
            ControlEventDispatch::legacy(event_send),
            event_loop,
            offline,
            node_id_consumer,
        )
    }

    /// Event-injection seam used by the existing context migration. Graph construction remains
    /// entirely legacy here; in particular the resulting base still owns a raw render sender and
    /// legacy ID provider.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_injected(
        sample_rate: f32,
        max_channel_count: usize,
        state: Arc<AtomicU8>,
        frames_played: Arc<AtomicU64>,
        render_channel: Sender<ControlMessage>,
        control_batch_sender: ControlBatchSender,
        control_batch_applied: ControlBatchApplied,
        event_send: Sender<EventDispatch>,
        event_loop: EventLoop,
        offline: bool,
        node_id_consumer: llq::Consumer<AudioNodeId>,
        gate: InjectedContextAdmissionGate,
    ) -> Self {
        Self::new_with_control_events(
            sample_rate,
            max_channel_count,
            state,
            frames_played,
            render_channel,
            control_batch_sender,
            control_batch_applied,
            ControlEventDispatch::injected(event_send, gate),
            event_loop,
            offline,
            node_id_consumer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_control_events(
        sample_rate: f32,
        max_channel_count: usize,
        state: Arc<AtomicU8>,
        frames_played: Arc<AtomicU64>,
        render_channel: Sender<ControlMessage>,
        control_batch_sender: ControlBatchSender,
        control_batch_applied: ControlBatchApplied,
        control_events: ControlEventDispatch,
        event_loop: EventLoop,
        offline: bool,
        node_id_consumer: llq::Consumer<AudioNodeId>,
    ) -> Self {
        let audio_node_id_provider = AudioNodeIdProvider::new(node_id_consumer);

        let base_inner = ConcreteBaseAudioContextInner {
            sample_rate,
            max_channel_count,
            graph_control: ConcreteGraphControl::Legacy(LegacyGraphControl {
                audio_node_id_provider,
                render_channel: RwLock::new(render_channel),
                _control_batch_sender: control_batch_sender,
                control_batch_applied,
                suspended_messages: Mutex::new(None),
                queued_messages: Mutex::new(Vec::new()),
                queued_audio_listener_msgs: Mutex::new(Vec::new()),
                connections: Mutex::new(HashSet::new()),
            }),
            destination_channel_config: AudioNodeOptions::default().into(),
            frames_played,
            listener_params: None,
            magic_connections: None,
            offline,
            state: ConcreteContextState::Legacy(state),
            event_handlers: ConcreteEventHandlers::Legacy(event_loop),
            control_events,
        };
        let base = Self {
            inner: Arc::new(base_inner),
        };

        // Online AudioContext should start with stereo channels by default
        let initial_channel_count = if offline {
            max_channel_count
        } else {
            2.min(max_channel_count)
        };

        let (listener_params, destination_channel_config) = {
            // Register magical nodes. We should not store the nodes inside our context since that
            // will create a cyclic reference, but we can reconstruct a new instance on the fly
            // when requested
            let dest = AudioDestinationNode::new(&base, initial_channel_count);
            let destination_channel_config = dest.into_channel_config();
            let listener = crate::spatial::AudioListenerNode::new(&base);

            let listener_params = listener.into_fields();
            let AudioListener {
                position_x,
                position_y,
                position_z,
                forward_x,
                forward_y,
                forward_z,
                up_x,
                up_y,
                up_z,
            } = listener_params;

            let listener_params = AudioListenerParams {
                position_x: position_x.into_raw_parts(),
                position_y: position_y.into_raw_parts(),
                position_z: position_z.into_raw_parts(),
                forward_x: forward_x.into_raw_parts(),
                forward_y: forward_y.into_raw_parts(),
                forward_z: forward_z.into_raw_parts(),
                up_x: up_x.into_raw_parts(),
                up_y: up_y.into_raw_parts(),
                up_z: up_z.into_raw_parts(),
            };

            (listener_params, destination_channel_config)
        }; // Nodes will drop now, so base.inner has no copies anymore

        let mut base = base;
        let inner_mut = Arc::get_mut(&mut base.inner).unwrap();
        inner_mut.listener_params = Some(listener_params);
        inner_mut.destination_channel_config = destination_channel_config;

        // Validate if the hardcoded node IDs line up
        debug_assert_eq!(
            base.legacy_graph()
                .audio_node_id_provider
                .id_inc
                .load(Ordering::Relaxed),
            LISTENER_PARAM_IDS.end,
        );

        // For an online AudioContext, pre-create the HRTF-database for panner nodes
        if !offline {
            crate::node::load_hrtf_processor(sample_rate as u32);
        }

        base
    }

    /// Private production boundary for concrete nodes built exclusively through the branded
    /// injected transaction. It deliberately does not create destination/listener graph nodes and
    /// cannot provide legacy graph mutation APIs. Destination and read-only context properties
    /// remain callable, but listener creation, other node constructors, explicit connections, and
    /// post-construction automation panic instead of falling back to a raw sender. A complete
    /// injected AudioContext is deferred, and no public path can select this base in this slice.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_injected_node_construction_base(
        sample_rate: f32,
        max_channel_count: usize,
        state: Arc<AtomicU8>,
        frames_played: Arc<AtomicU64>,
        constructor: InjectedNodeConstructor,
        event_send: Sender<EventDispatch>,
        event_loop: EventLoop,
        offline: bool,
    ) -> Self {
        let control_events =
            ControlEventDispatch::injected(event_send, constructor.admission_gate());
        Self {
            inner: Arc::new(ConcreteBaseAudioContextInner {
                sample_rate,
                max_channel_count,
                graph_control: ConcreteGraphControl::Injected(constructor),
                destination_channel_config: AudioNodeOptions::default().into(),
                frames_played,
                listener_params: None,
                magic_connections: None,
                offline,
                state: ConcreteContextState::Legacy(state),
                event_handlers: ConcreteEventHandlers::Legacy(event_loop),
                control_events,
            }),
        }
    }

    fn legacy_graph(&self) -> &LegacyGraphControl {
        match &self.inner.graph_control {
            ConcreteGraphControl::Legacy(graph) => graph,
            ConcreteGraphControl::Injected(_) => {
                panic!("legacy graph mutation is unavailable on an injected construction base")
            }
        }
    }

    pub(crate) fn injected_node_constructor(&self) -> Option<&InjectedNodeConstructor> {
        match &self.inner.graph_control {
            ConcreteGraphControl::Legacy(_) => None,
            ConcreteGraphControl::Injected(constructor) => Some(constructor),
        }
    }

    pub(crate) fn address(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    #[allow(dead_code)]
    pub(crate) fn applied_control_batch_sequence(&self) -> u64 {
        match &self.inner.graph_control {
            ConcreteGraphControl::Legacy(graph) => graph.control_batch_applied.load(),
            ConcreteGraphControl::Injected(constructor) => constructor.applied_batch_sequence(),
        }
    }

    /// Construct a new pair of [`AudioNode`] and [`AudioProcessor`]
    pub(crate) fn register<
        T: AudioNode,
        F: FnOnce(AudioContextRegistration) -> (T, Box<dyn AudioProcessor>),
    >(
        &self,
        f: F,
    ) -> T {
        // create a unique id for this node
        let id = self.legacy_graph().audio_node_id_provider.get();
        let registration = AudioContextRegistration {
            injected_lifetime: None,
            injected_connection: None,
            id,
            context: self.clone(),
        };

        // create the node and its renderer
        let (node, render) = (f)(registration);

        // pass the renderer to the audio graph
        let message = ControlMessage::RegisterNode {
            id,
            reclaim_id: llq::Node::new(id),
            node: render,
            inputs: node.number_of_inputs(),
            outputs: node.number_of_outputs(),
            channel_config: node.channel_config().inner(),
        };

        // if this is the AudioListener or its params, do not add it to the graph just yet
        if id == LISTENER_NODE_ID || LISTENER_PARAM_IDS.contains(&id.0) {
            let mut queued_audio_listener_msgs = self
                .legacy_graph()
                .queued_audio_listener_msgs
                .lock()
                .unwrap();
            queued_audio_listener_msgs.push(message);
        } else {
            self.send_control_msg(message);
            self.resolve_queued_control_msgs(id);
        }

        node
    }

    /// Send a control message to the render thread
    ///
    /// When the render thread is closed or crashed, the message is discarded and a log warning is
    /// emitted.
    pub(crate) fn send_control_msg(&self, msg: ControlMessage) {
        if self.state() != AudioContextState::Closed {
            let graph = self.legacy_graph();
            let sender = graph.render_channel.read().unwrap();
            // if the context is suspended, buffer the message and don't send it
            if let Some(queued) = graph.suspended_messages.lock().unwrap().as_mut() {
                queued.push(msg);
                return;
            }

            let result = sender.send(msg);
            if result.is_err() {
                log::warn!("Discarding control message - render thread is closed");
            }
        }
    }

    pub(crate) fn suspend_control_msgs(&self, msg: ControlMessage) {
        let graph = self.legacy_graph();
        let sender = graph.render_channel.read().unwrap();
        *graph.suspended_messages.lock().unwrap() = Some(Vec::new());
        if sender.send(msg).is_err() {
            log::warn!("Discarding control message - render thread is closed");
        }
    }

    pub(crate) fn resume_control_msgs(&self, msg: ControlMessage) {
        let graph = self.legacy_graph();
        let sender = graph.render_channel.read().unwrap();
        let messages = self
            .legacy_graph()
            .suspended_messages
            .lock()
            .unwrap()
            .take()
            .unwrap_or_default();

        for msg in messages {
            if sender.send(msg).is_err() {
                log::warn!("Discarding control message - render thread is closed");
                return;
            }
        }

        if sender.send(msg).is_err() {
            log::warn!("Discarding control message - render thread is closed");
        }
    }

    /// Put sink-replay records ahead of mutations already staged while the context is suspended.
    /// The caller holds the render-channel write guard, so no producer can interleave here.
    pub(crate) fn prepend_suspended_control_msgs(&self, mut messages: Vec<ControlMessage>) {
        if messages.is_empty() {
            return;
        }

        let mut suspended = self.legacy_graph().suspended_messages.lock().unwrap();
        let existing = suspended.get_or_insert_with(Vec::new);
        messages.append(existing);
        *existing = messages;
    }

    pub(crate) fn send_event_with<F>(&self, make_event: F) -> Result<(), ControlEventSendOutcome>
    where
        F: FnOnce() -> EventDispatch + Copy,
    {
        match self.inner.control_events.send_with(make_event) {
            ControlEventSendOutcome::Delivered => Ok(()),
            error => Err(error),
        }
    }

    /// Clone only the opaque control-side event capability. In injected contexts this cannot be
    /// converted back into a raw event sender.
    pub(crate) fn control_event_dispatch(&self) -> ControlEventDispatch {
        self.inner.control_events.clone()
    }

    /// Clone the render-frame clock without retaining this entire context in a background metrics
    /// worker.
    pub(crate) fn frames_played_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.inner.frames_played)
    }

    pub(crate) fn set_event_activity_handler<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        match &self.inner.event_handlers {
            ConcreteEventHandlers::Legacy(event_loop) => {
                event_loop.set_activity_handler(callback);
            }
            ConcreteEventHandlers::Injected(events) => events.set_activity_handler(callback),
        }
    }

    pub(crate) fn clear_event_activity_handler(&self) {
        match &self.inner.event_handlers {
            ConcreteEventHandlers::Legacy(event_loop) => event_loop.clear_activity_handler(),
            ConcreteEventHandlers::Injected(events) => events.clear_activity_handler(),
        }
    }

    pub(crate) fn lock_control_msg_sender(&self) -> RwLockWriteGuard<'_, Sender<ControlMessage>> {
        self.legacy_graph().render_channel.write().unwrap()
    }

    pub(super) fn mark_node_dropped(&self, id: AudioNodeId) {
        // Ignore magic nodes
        if id == DESTINATION_NODE_ID || id == LISTENER_NODE_ID || LISTENER_PARAM_IDS.contains(&id.0)
        {
            return;
        }

        // Inform render thread that the control thread AudioNode no longer has any handles
        let message = ControlMessage::ControlHandleDropped { id };
        self.send_control_msg(message);

        // Clear the connection administration for this node, the node id may be recycled later
        self.legacy_graph()
            .connections
            .lock()
            .unwrap()
            .retain(|&(from, _output, to, _input)| from != id && to != id);
    }

    /// Inform render thread that this node can act as a cycle breaker
    #[doc(hidden)]
    pub fn mark_cycle_breaker(&self, reg: &AudioContextRegistration) {
        let id = reg.id();
        let message = ControlMessage::MarkCycleBreaker { id };
        self.send_control_msg(message);
    }

    /// `ChannelConfig` of the `AudioDestinationNode`
    pub(super) fn destination_channel_config(&self) -> ChannelConfig {
        self.inner.destination_channel_config.clone()
    }

    pub(super) fn destination_registration(&self) -> AudioContextRegistration {
        match &self.inner.magic_connections {
            Some(connections) => AudioContextRegistration::from_injected_permanent(
                DESTINATION_NODE_ID,
                self.clone(),
                connections.destination(),
                InjectedConnectionEndpointKind::AudioNode,
                1,
                1,
            ),
            None => AudioContextRegistration {
                injected_lifetime: None,
                injected_connection: None,
                id: DESTINATION_NODE_ID,
                context: self.clone(),
            },
        }
    }

    /// Returns the `AudioListener` which is used for 3D spatialization
    pub(super) fn listener(&self) -> AudioListener {
        // instruct to BaseContext to add the AudioListener if it has not already
        self.base().ensure_audio_listener_present();

        let mut ids = LISTENER_PARAM_IDS
            .into_iter()
            .enumerate()
            .map(|(index, id)| match &self.inner.magic_connections {
                Some(connections) => AudioContextRegistration::from_injected_permanent(
                    AudioNodeId(id),
                    self.clone(),
                    connections.listener_param(index),
                    InjectedConnectionEndpointKind::AudioParam,
                    1,
                    1,
                ),
                None => AudioContextRegistration {
                    injected_lifetime: None,
                    injected_connection: None,
                    id: AudioNodeId(id),
                    context: self.clone(),
                },
            });
        let params = self.inner.listener_params.as_ref().unwrap();

        AudioListener {
            position_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_x.clone()),
            position_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_y.clone()),
            position_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.position_z.clone()),
            forward_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_x.clone()),
            forward_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_y.clone()),
            forward_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.forward_z.clone()),
            up_x: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_x.clone()),
            up_y: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_y.clone()),
            up_z: AudioParam::from_raw_parts(ids.next().unwrap(), params.up_z.clone()),
        }
    }

    /// Returns state of current context
    #[must_use]
    pub(super) fn state(&self) -> AudioContextState {
        match &self.inner.state {
            ConcreteContextState::Legacy(state) => state.load(Ordering::Acquire).into(),
            ConcreteContextState::Injected => {
                self.inner.control_events.exact_events().unwrap().state()
            }
        }
    }

    /// Updates state of current context
    pub(super) fn set_state(&self, state: AudioContextState) {
        // Only used from OfflineAudioContext or suspended AudioContext, otherwise the state
        // changed are spawned from the render thread
        match &self.inner.state {
            ConcreteContextState::Legacy(shared) => {
                let current_state = shared.load(Ordering::Acquire);
                if current_state != state as u8 {
                    shared.store(state as u8, Ordering::Release);
                    let _ = self.send_event_with(|| EventDispatch::state_change(state));
                }
            }
            ConcreteContextState::Injected => {
                let events = self.inner.control_events.exact_events().unwrap();
                if events.transition_live_state(state) == InjectedStateTransition::Changed {
                    let _ = self.send_event_with(|| EventDispatch::state_change(state));
                }
            }
        }
    }

    /// The sample rate (in sample-frames per second) at which the `AudioContext` handles audio.
    #[must_use]
    pub(super) fn sample_rate(&self) -> f32 {
        self.inner.sample_rate
    }

    /// This is the time in seconds of the sample frame immediately following the last sample-frame
    /// in the block of audio most recently processed by the context’s rendering graph.
    #[must_use]
    // web audio api specification requires that `current_time` returns an f64
    // std::sync::AtomicsF64 is not currently implemented in the standard library
    // Currently, we have no other choice than casting an u64 into f64, with possible loss of precision
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn current_time(&self) -> f64 {
        self.inner.frames_played.load(Ordering::SeqCst) as f64 / self.inner.sample_rate as f64
    }

    /// Maximum available channels for the audio destination
    #[must_use]
    pub(crate) fn max_channel_count(&self) -> usize {
        self.inner.max_channel_count
    }

    /// Release queued control messages to the render thread that were blocking on the availability
    /// of the Node with the given `id`
    fn resolve_queued_control_msgs(&self, id: AudioNodeId) {
        // resolve control messages that depend on this registration
        let mut queued = self.legacy_graph().queued_messages.lock().unwrap();
        let mut i = 0; // waiting for Vec::drain_filter to stabilize
        while i < queued.len() {
            if matches!(&queued[i], ControlMessage::ConnectNode {to, ..} if *to == id) {
                let m = queued.remove(i);
                self.send_control_msg(m);
            } else {
                i += 1;
            }
        }
    }

    /// Connects the output of the `from` audio node to the input of the `to` audio node
    pub(crate) fn connect(&self, from: AudioNodeId, to: AudioNodeId, output: usize, input: usize) {
        let inserted = self
            .legacy_graph()
            .connections
            .lock()
            .unwrap()
            .insert((from, output, to, input));

        if !inserted {
            return; // do not allow duplicated edges
        }

        let message = ControlMessage::ConnectNode {
            from,
            to,
            output,
            input,
        };
        self.send_control_msg(message);
    }

    /// Registration-carrying public AudioNode seam. H2a preserves the legacy path while keeping
    /// exact endpoint brands inseparable from their handles; h2b selects the serialized exact
    /// transaction after its overload matrix is frozen.
    pub(crate) fn connect_registrations(
        &self,
        from: &AudioContextRegistration,
        to: &AudioContextRegistration,
        output: usize,
        input: usize,
    ) {
        match &self.inner.graph_control {
            ConcreteGraphControl::Legacy(_) => {
                self.connect(from.id(), to.id(), output, input);
            }
            ConcreteGraphControl::Injected(_) => {
                let _ = (
                    from.injected_connection_endpoint(),
                    to.injected_connection_endpoint(),
                );
                panic!("NotSupportedError - exact AudioNode connections are not selected yet")
            }
        }
    }

    /// Schedule a connection of an `AudioParam` to the `AudioNode` it belongs to
    ///
    /// It is not performed immediately as the `AudioNode` is not registered at this point.
    pub(super) fn queue_audio_param_connect(&self, param: &AudioParam, audio_node: AudioNodeId) {
        // no need to store these type of connections in the legacy explicit mirror

        let message = ControlMessage::ConnectNode {
            from: param.registration().id(),
            to: audio_node,
            output: 0,
            input: usize::MAX, // audio params connect to the 'hidden' input port
        };
        self.legacy_graph()
            .queued_messages
            .lock()
            .unwrap()
            .push(message);
    }

    /// Disconnects outputs of the audio node, possibly filtered by output node, input, output.
    pub(crate) fn disconnect(
        &self,
        from: AudioNodeId,
        output: Option<usize>,
        to: Option<AudioNodeId>,
        input: Option<usize>,
    ) {
        // check if the node was connected, otherwise panic
        let mut has_disconnected = false;
        let mut connections = self.legacy_graph().connections.lock().unwrap();
        connections.retain(|&(c_from, c_output, c_to, c_input)| {
            let retain = c_from != from
                || c_output != output.unwrap_or(c_output)
                || c_to != to.unwrap_or(c_to)
                || c_input != input.unwrap_or(c_input);
            if !retain {
                has_disconnected = true;
                let message = ControlMessage::DisconnectNode {
                    from,
                    to: c_to,
                    input: c_input,
                    output: c_output,
                };
                self.send_control_msg(message);
            }
            retain
        });

        // make sure to drop the MutexGuard before the panic to avoid poisoning
        drop(connections);

        if !has_disconnected && to.is_some() {
            panic!("InvalidAccessError - attempting to disconnect unconnected nodes");
        }
    }

    /// Registration-carrying disconnect seam paired with `connect_registrations`.
    pub(crate) fn disconnect_registrations(
        &self,
        from: &AudioContextRegistration,
        output: Option<usize>,
        to: Option<&AudioContextRegistration>,
        input: Option<usize>,
    ) {
        match &self.inner.graph_control {
            ConcreteGraphControl::Legacy(_) => {
                self.disconnect(
                    from.id(),
                    output,
                    to.map(AudioContextRegistration::id),
                    input,
                );
            }
            ConcreteGraphControl::Injected(_) => {
                let _ = (
                    from.injected_connection_endpoint(),
                    to.and_then(AudioContextRegistration::injected_connection_endpoint),
                );
                panic!("NotSupportedError - exact AudioNode disconnections are not selected yet")
            }
        }
    }

    /// Connect the `AudioListener` to a `PannerNode`
    pub(crate) fn connect_listener_to_panner(&self, panner: AudioNodeId) {
        self.connect(LISTENER_NODE_ID, panner, 0, usize::MAX);
    }

    /// Add the [`AudioListener`] to the audio graph (if not already)
    pub(crate) fn ensure_audio_listener_present(&self) {
        if matches!(&self.inner.graph_control, ConcreteGraphControl::Injected(_)) {
            // The exact injected bootstrap eagerly publishes listener+params in the same atomic
            // envelope as destination. Unlike legacy, there is no deferred raw sender queue.
            return;
        }
        let mut queued_audio_listener_msgs = self
            .legacy_graph()
            .queued_audio_listener_msgs
            .lock()
            .unwrap();
        let mut released = false;
        while let Some(message) = queued_audio_listener_msgs.pop() {
            // add the AudioListenerRenderer to the graph
            self.send_control_msg(message);
            released = true;
        }

        if released {
            // connect the AudioParamRenderers to the Listener
            self.resolve_queued_control_msgs(LISTENER_NODE_ID);

            // hack: Connect the listener to the destination node to force it to render at each
            // quantum. Abuse the magical usize::MAX port so it acts as an AudioParam and has no side
            // effects
            self.connect(LISTENER_NODE_ID, DESTINATION_NODE_ID, 0, usize::MAX);
        }
    }

    /// Returns true if this is `OfflineAudioContext` (false when it is an `AudioContext`)
    pub(crate) fn offline(&self) -> bool {
        self.inner.offline
    }

    pub(crate) fn set_event_handler(&self, event: EventType, callback: EventHandler) {
        match &self.inner.event_handlers {
            ConcreteEventHandlers::Legacy(event_loop) => event_loop.set_handler(event, callback),
            ConcreteEventHandlers::Injected(events) => events.set_handler(event, callback),
        }
    }

    pub(crate) fn clear_event_handler(&self, event: EventType) {
        match &self.inner.event_handlers {
            ConcreteEventHandlers::Legacy(event_loop) => event_loop.clear_handler(event),
            ConcreteEventHandlers::Injected(events) => events.clear_handler(event),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::OfflineAudioContext;

    fn test_marker(value: u16) -> ControlMessage {
        ControlMessage::TestMarker {
            value,
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[test]
    fn sink_replay_is_prepended_to_existing_suspended_fifo() {
        let context = OfflineAudioContext::new(1, 128, 48_000.);
        *context
            .base()
            .legacy_graph()
            .suspended_messages
            .lock()
            .unwrap() = Some(vec![test_marker(3), test_marker(4)]);

        context
            .base()
            .prepend_suspended_control_msgs(vec![test_marker(1), test_marker(2)]);

        let messages = context
            .base()
            .legacy_graph()
            .suspended_messages
            .lock()
            .unwrap()
            .take()
            .unwrap();
        let values: Vec<_> = messages
            .into_iter()
            .map(|message| match message {
                ControlMessage::TestMarker { value, .. } => value,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(values, [1, 2, 3, 4]);
    }

    #[test]
    fn test_provide_node_id() {
        let (mut id_producer, id_consumer) = llq::Queue::new().split();
        let provider = AudioNodeIdProvider::new(id_consumer);
        assert_eq!(provider.get().0, 0); // newly assigned
        assert_eq!(provider.get().0, 1); // newly assigned
        id_producer.push(llq::Node::new(AudioNodeId(0)));
        assert_eq!(provider.get().0, 0); // reused
        assert_eq!(provider.get().0, 2); // newly assigned
    }

    #[test]
    fn test_connect_disconnect() {
        let context = OfflineAudioContext::new(1, 128, 48000.);
        let node1 = context.create_constant_source();
        let node2 = context.create_gain();

        // connection list starts empty
        assert!(context
            .base()
            .legacy_graph()
            .connections
            .lock()
            .unwrap()
            .is_empty());

        node1.disconnect(); // never panic for plain disconnect calls

        node1.connect(&node2);

        // connection should be registered
        assert_eq!(
            context
                .base()
                .legacy_graph()
                .connections
                .lock()
                .unwrap()
                .len(),
            1
        );

        node1.disconnect();
        assert!(context
            .base()
            .legacy_graph()
            .connections
            .lock()
            .unwrap()
            .is_empty());

        node1.connect(&node2);
        assert_eq!(
            context
                .base()
                .legacy_graph()
                .connections
                .lock()
                .unwrap()
                .len(),
            1
        );

        node1.disconnect_dest(&node2);
        assert!(context
            .base()
            .legacy_graph()
            .connections
            .lock()
            .unwrap()
            .is_empty());
    }

    #[test]
    #[should_panic]
    fn test_disconnect_not_existing() {
        let context = OfflineAudioContext::new(1, 128, 48000.);
        let node1 = context.create_constant_source();
        let node2 = context.create_gain();

        node1.disconnect_dest(&node2);
    }

    #[test]
    fn test_mark_node_dropped() {
        let context = OfflineAudioContext::new(1, 128, 48000.);

        let node1 = context.create_constant_source();
        let node2 = context.create_gain();

        node1.connect(&node2);
        context.base().mark_node_dropped(node1.registration().id());

        // dropping should clear connections administration
        assert!(context
            .base()
            .legacy_graph()
            .connections
            .lock()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_event_activity_handler_registration_and_clear() {
        let context = OfflineAudioContext::new(1, 128, 48_000.);
        let activity_count = Arc::new(AtomicU64::new(0));
        let activity_count_clone = Arc::clone(&activity_count);
        context.set_event_activity_handler(move || {
            activity_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        context
            .base()
            .send_event_with(EventDispatch::sink_change)
            .unwrap();
        context.base().handle_pending_events_for_test();
        assert_eq!(activity_count.load(Ordering::Relaxed), 1);

        context.clear_event_activity_handler();
        context
            .base()
            .send_event_with(EventDispatch::sink_change)
            .unwrap();
        context.base().handle_pending_events_for_test();
        assert_eq!(activity_count.load(Ordering::Relaxed), 1);
    }
}
