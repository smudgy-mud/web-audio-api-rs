//! Private bounded graph-control transport for the injected online-context path.
//!
//! The exact private base and output lifecycle consume this transport. It separates a cloneable
//! ordinary producer, a unique Close-only lifecycle owner, and a consuming render initializer.
//! Public builder selection, broader node mutations, and Suspend/Resume barrier wiring remain
//! deferred. No raw sender/receiver escapes.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::num::NonZeroU64;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, TryLockError, Weak};

use crossbeam_channel::{Sender, TrySendError};

use super::injected_admission::{
    AdmissionDrain, AdmissionError, AdmissionSnapshot, CapacityWorkerJoinError,
    CapacityWorkerRetirement, GraphControlAdmission,
};
#[cfg(test)]
use super::injected_node_id_pair;
#[cfg(test)]
use super::injected_node_lifetime::{
    injected_node_lifetime_registry, DEFAULT_NODE_LIFETIME_CAPACITY,
};
use super::injected_node_lifetime::{BoundInjectedOutputRenderer, InjectedNodeLifetimeOwner};
use super::{InjectedContextAdmissionGate, InjectedGraphReclaimInit};
#[cfg(test)]
use crate::events::EventDispatch;
use crate::events::{
    BoundInjectedEventDispatch, InjectedControlEventDispatch, InjectedEventDispatchSetup,
    InjectedLifecycleEventLoop,
};
use crate::message::{
    control_batch_storage_mut, graph_lifecycle_ack_pair, injected_control_batch_node,
    recover_unsubmitted_injected_batch, ControlBatchApplied, ControlBatchNode, ControlBatchPermit,
    ControlBatchStoragePool, ControlMessage, GraphLifecycleBarrier, GraphLifecycleOutcome,
    GraphLifecyclePublisher, GraphLifecycleSnapshot, GraphLifecycleTransition,
    GraphLifecycleWatcher, InjectedCommandCredit, InjectedCommandCreditPool,
    InjectedPhysicalCredit, InjectedPhysicalCreditOwners, InjectedPhysicalCreditPool,
    CONTROL_BATCH_CAPACITY,
};
#[cfg(test)]
use crate::output::audio_render_thread_pair;
use crate::output::{
    try_audio_render_thread_pair, AudioOutputError, AudioOutputEventSink, AudioRenderCallback,
    AudioRenderFormat, AudioRenderOwner, AudioRenderThreadPairFailure,
};
use crate::render::RenderThread;
use crate::stats::AudioStats;

const LOGICAL_COMMAND_LIMIT: usize = CONTROL_BATCH_CAPACITY;
const STAGED_ENVELOPE_LIMIT: usize = CONTROL_BATCH_CAPACITY;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedControlError {
    InvalidOrdinaryCapacity,
    Empty,
    TooLarge,
    UnsupportedCommand,
    LogicalCommandCredits,
    BatchStorageCredits,
    OrdinaryPhysicalCredits,
    StagingFull,
    Contended,
    Poisoned,
    Sealed,
    Disconnected,
    SequenceExhausted,
    ProtocolViolation,
    GatePoisoned,
}

impl From<AdmissionError> for InjectedControlError {
    fn from(error: AdmissionError) -> Self {
        match error {
            AdmissionError::Contended => Self::Contended,
            AdmissionError::Poisoned => Self::GatePoisoned,
            AdmissionError::Sealed => Self::Sealed,
            AdmissionError::Exhausted | AdmissionError::CapacityWorkerActive => {
                Self::ProtocolViolation
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransportPhase {
    Open,
    SealStarted,
    Sealed,
    Failed,
}

#[derive(Clone)]
struct StagingSlotPool(Arc<StagingSlotState>);

struct StagingSlotState {
    in_flight: AtomicUsize,
    activity: Sender<()>,
}

impl StagingSlotPool {
    fn new(activity: Sender<()>) -> Self {
        Self(Arc::new(StagingSlotState {
            in_flight: AtomicUsize::new(0),
            activity,
        }))
    }

    fn try_acquire(&self) -> Option<StagingSlot> {
        self.0
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < STAGED_ENVELOPE_LIMIT).then_some(count + 1)
            })
            .ok()
            .map(|_| StagingSlot(Arc::clone(&self.0)))
    }

    #[cfg(test)]
    fn in_flight(&self) -> usize {
        self.0.in_flight.load(Ordering::Acquire)
    }
}

struct StagingSlot(Arc<StagingSlotState>);

impl Drop for StagingSlot {
    fn drop(&mut self) {
        let previous = self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        let _ = self.0.activity.try_send(());
    }
}

struct StagedControlBatch {
    batch: ControlBatchNode,
    _slot: StagingSlot,
    _sequence: BatchSequenceReservation,
}

struct BatchSequenceReservation(Arc<AtomicUsize>);

impl Drop for BatchSequenceReservation {
    fn drop(&mut self) {
        let previous = self.0.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
    }
}

enum ReservedPlacement {
    Running(InjectedPhysicalCredit),
    Staged(StagingSlot),
}

struct InjectedControlState {
    sender: Sender<ControlMessage>,
    phase: TransportPhase,
    disconnected: bool,
    initially_suspended: bool,
    next_batch_sequence: u64,
    staged: VecDeque<StagedControlBatch>,
}

struct InjectedControlInner {
    gate: InjectedContextAdmissionGate,
    state: Mutex<InjectedControlState>,
    logical_commands: InjectedCommandCreditPool,
    batch_storage: ControlBatchStoragePool,
    ordinary_physical: InjectedPhysicalCreditPool,
    lifecycle_physical: InjectedPhysicalCreditPool,
    staging_slots: StagingSlotPool,
    batch_sequence_reservations: Arc<AtomicUsize>,
    close_in_flight: Arc<AtomicBool>,
    last_submitted_batch_sequence: AtomicU64,
    applied: ControlBatchApplied,
    /// An accepted payload whose mandatory control-side finalizer failed is irrevocably terminal:
    /// the render payload remains owned by the queue/staging, but its paired mirror/lifetime
    /// transition cannot be reported as ordinary success.
    accepted_finalizer_failed: AtomicBool,
}

/// Opaque weak identity used to bind later private lifecycle foundations to this exact transport.
#[derive(Clone)]
pub(crate) struct InjectedControlIdentity(Weak<InjectedControlInner>);

impl InjectedControlIdentity {
    pub(crate) fn matches_drained(&self, drained: &DrainedControlClose) -> bool {
        Weak::ptr_eq(&self.0, &Arc::downgrade(&drained.inner))
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

/// Cloneable ordinary graph producer. Private fields prevent raw sender or lifecycle extraction.
#[derive(Clone)]
pub(crate) struct InjectedControlProducer {
    inner: Arc<InjectedControlInner>,
}

/// Unique, non-clone lifecycle and seal authority.
pub(crate) struct InjectedControlLifecycleOwner {
    inner: Arc<InjectedControlInner>,
    watcher: GraphLifecycleWatcher,
    activity: crossbeam_channel::Receiver<()>,
}

/// Consuming initializer which binds receiver, physical lifetime owners, and lifecycle publisher.
pub(crate) struct InjectedControlRenderInit {
    receiver: crossbeam_channel::Receiver<ControlMessage>,
    physical_owners: InjectedPhysicalCreditOwners,
    lifecycle_publisher: GraphLifecyclePublisher,
    applied: ControlBatchApplied,
    identity: InjectedControlIdentity,
    event_gate: InjectedContextAdmissionGate,
    initially_suspended: bool,
}

impl InjectedControlRenderInit {
    pub(crate) fn event_admission_gate(&self) -> InjectedContextAdmissionGate {
        self.event_gate.clone()
    }

    pub(crate) const fn initially_suspended(&self) -> bool {
        self.initially_suspended
    }
}

/// Inseparable pre-render bundle. Its private fields ensure the exact graph-id publisher and
/// lifetime owner validated together cannot be swapped, and successful render construction keeps
/// returning that owner alongside the render owner/callback.
pub(crate) struct InjectedNodeLifetimeBootstrap {
    owner: InjectedNodeLifetimeOwner,
    graph: InjectedGraphReclaimInit,
}

impl InjectedNodeLifetimeBootstrap {
    #[allow(clippy::result_large_err)] // failure must return both exact unboxed owners intact
    pub(crate) fn new(
        owner: InjectedNodeLifetimeOwner,
        graph: InjectedGraphReclaimInit,
    ) -> Result<Self, (InjectedNodeLifetimeOwner, InjectedGraphReclaimInit)> {
        if !owner.matches_graph_init(&graph) {
            return Err((owner, graph));
        }
        Ok(Self { owner, graph })
    }

    #[cfg(test)]
    pub(crate) fn into_parts_for_test(
        self,
    ) -> (InjectedNodeLifetimeOwner, InjectedGraphReclaimInit) {
        (self.owner, self.graph)
    }
}

pub(crate) struct BuildInjectedRenderFailure {
    pub(crate) init: InjectedControlRenderInit,
    pub(crate) node_lifetimes: InjectedNodeLifetimeBootstrap,
}

/// Opaque renderer with receiver, exact event branches, and node owner already bound. The private
/// lifecycle next consumes it before the sole fallible callback/GC installation.
#[must_use]
pub(crate) struct BoundInjectedRenderer {
    renderer: RenderThread,
    node_lifetimes: InjectedNodeLifetimeOwner,
    events: BoundInjectedRendererEvents,
}

enum BoundInjectedRendererEvents {
    Output {
        control: InjectedControlEventDispatch,
        lifecycle: InjectedLifecycleEventLoop,
    },
    #[cfg(test)]
    Legacy,
}

/// Opaque proof that this renderer and node owner passed the injected build-time graph/control
/// identity checks together. No constructor accepting a raw `RenderThread` is exposed.
pub(crate) struct ExactBoundInjectedRenderer {
    renderer: RenderThread,
    node_lifetimes: InjectedNodeLifetimeOwner,
}

pub(crate) struct ExactInjectedRenderPairFailure {
    pub(crate) error: AudioOutputError,
    pub(crate) renderer: ExactBoundInjectedRenderer,
    pub(crate) events: AudioOutputEventSink,
}

impl ExactBoundInjectedRenderer {
    pub(crate) fn injected_base_facts(&self) -> (f32, usize, Arc<AtomicU64>) {
        self.renderer.injected_base_facts()
    }

    pub(crate) fn matches_constructor(
        &self,
        constructor: &super::injected_node_construction::InjectedNodeConstructor,
    ) -> bool {
        constructor.matches_control_identity(self.node_lifetimes.control_identity())
            && constructor.matches_node_id_identity(self.node_lifetimes.node_id_identity())
    }

    pub(crate) fn matches_magic_graph(
        &self,
        magic: &super::injected_magic_construction::MagicGraphInstalled,
    ) -> bool {
        magic.matches(
            self.node_lifetimes.control_identity(),
            self.node_lifetimes.node_id_identity(),
        )
    }

    pub(crate) fn apply_magic_before_publication(&mut self, required_sequence: u64) -> bool {
        self.renderer
            .apply_injected_magic_before_publication(required_sequence)
    }

    #[cfg(test)]
    pub(crate) fn fail_next_gc_spawn_for_test(&mut self) {
        self.renderer.fail_next_gc_spawn_for_test();
    }

    #[cfg(test)]
    pub(crate) fn disconnect_lifecycle_on_next_render_for_test(&mut self) {
        self.renderer.disconnect_lifecycle_on_next_render_for_test();
    }

    #[cfg(test)]
    pub(crate) fn fail_reclaim_for_test(&mut self) {
        self.renderer.fail_reclaim_for_test();
    }

    #[cfg(test)]
    pub(crate) fn panic_magic_apply_for_test(&mut self) {
        self.renderer.panic_magic_apply_for_test();
    }

    #[cfg(test)]
    pub(crate) fn magic_bootstrap_shape_is_exact_for_test(&self) -> bool {
        self.renderer.magic_bootstrap_shape_is_exact_for_test()
    }

    #[allow(clippy::result_large_err)] // failure returns the exact unboxed renderer and node owner
    pub(crate) fn try_into_audio_render_thread_pair(
        self,
        format: AudioRenderFormat,
        events: AudioOutputEventSink,
    ) -> Result<
        (
            AudioRenderOwner,
            AudioRenderCallback,
            InjectedNodeLifetimeOwner,
        ),
        ExactInjectedRenderPairFailure,
    > {
        if !self
            .renderer
            .matches_output_format(format.sample_rate(), format.number_of_channels())
        {
            return Err(ExactInjectedRenderPairFailure {
                error: AudioOutputError::new(
                    crate::output::AudioOutputErrorKind::InvalidArgument,
                    "prepared output format does not match the injected renderer",
                ),
                renderer: self,
                events,
            });
        }
        match try_audio_render_thread_pair(format, self.renderer, events) {
            Ok((owner, callback)) => Ok((owner, callback, self.node_lifetimes)),
            Err(AudioRenderThreadPairFailure {
                error,
                renderer,
                events,
            }) => Err(ExactInjectedRenderPairFailure {
                error,
                renderer: ExactBoundInjectedRenderer {
                    renderer,
                    node_lifetimes: self.node_lifetimes,
                },
                events,
            }),
        }
    }
}

pub(crate) struct BindInjectedOutputRendererFailure {
    pub(crate) renderer: BoundInjectedRenderer,
    pub(crate) control: InjectedControlLifecycleOwner,
}

pub(crate) struct BindInjectedOutputEventsFailure {
    renderer: Option<BoundInjectedRenderer>,
    control: Option<InjectedControlLifecycleOwner>,
}

/// Opaque control-side event branch after its render producer and sole consumer were both checked
/// against the same event identity and exact control lifecycle.
pub(crate) struct InjectedConcreteEventBinding {
    events: InjectedControlEventDispatch,
    control_identity: InjectedControlIdentity,
}

impl InjectedConcreteEventBinding {
    pub(crate) fn matches_constructor(
        &self,
        constructor: &super::injected_node_construction::InjectedNodeConstructor,
    ) -> bool {
        self.events.matches_gate(&constructor.admission_gate())
            && constructor.matches_control_identity(&self.control_identity)
    }

    pub(crate) fn into_events(self) -> InjectedControlEventDispatch {
        self.events
    }
}

impl BindInjectedOutputEventsFailure {
    pub(crate) fn into_parts(mut self) -> (BoundInjectedRenderer, InjectedControlLifecycleOwner) {
        (self.renderer.take().unwrap(), self.control.take().unwrap())
    }
}

impl BoundInjectedRenderer {
    #[cfg(test)]
    pub(crate) fn swap_control_event_branches_for_test(&mut self, other: &mut Self) {
        let BoundInjectedRendererEvents::Output { control: left, .. } = &mut self.events else {
            panic!("exact event swap requires output renderer")
        };
        let BoundInjectedRendererEvents::Output { control: right, .. } = &mut other.events else {
            panic!("exact event swap requires output renderer")
        };
        std::mem::swap(left, right);
    }

    #[cfg(test)]
    pub(crate) fn into_audio_render_thread_pair(
        self,
        format: AudioRenderFormat,
        events: AudioOutputEventSink,
    ) -> (
        AudioRenderOwner,
        AudioRenderCallback,
        InjectedNodeLifetimeOwner,
    ) {
        let BoundInjectedRendererEvents::Legacy = self.events else {
            unreachable!("raw test pair requires the test-only legacy event branch")
        };
        let (owner, callback) = audio_render_thread_pair(format, self.renderer, events);
        (owner, callback, self.node_lifetimes)
    }

    /// Binds the sole lifecycle owner for this exact transport before callback installation.
    #[cfg(test)]
    #[allow(clippy::result_large_err)]
    pub(crate) fn bind_output_lifecycle(
        self,
        control: InjectedControlLifecycleOwner,
    ) -> Result<
        super::injected_node_lifetime::TestBoundInjectedOutputRenderer,
        BindInjectedOutputRendererFailure,
    > {
        if !matches!(&self.events, BoundInjectedRendererEvents::Legacy) {
            return Err(BindInjectedOutputRendererFailure {
                renderer: self,
                control,
            });
        }
        if !self
            .node_lifetimes
            .control_identity()
            .ptr_eq(&control.identity())
        {
            return Err(BindInjectedOutputRendererFailure {
                renderer: self,
                control,
            });
        }
        Ok(
            super::injected_node_lifetime::TestBoundInjectedOutputRenderer::new(
                ExactBoundInjectedRenderer {
                    renderer: self.renderer,
                    node_lifetimes: self.node_lifetimes,
                },
                control,
            ),
        )
    }

    /// Binds the exact control owner after one consumed setup derived the render, admitted-control,
    /// and sole event-consumer branches together.
    #[allow(clippy::result_large_err)]
    pub(crate) fn bind_output_lifecycle_exact(
        self,
        control: InjectedControlLifecycleOwner,
    ) -> Result<
        (BoundInjectedOutputRenderer, InjectedConcreteEventBinding),
        BindInjectedOutputEventsFailure,
    > {
        if !self
            .node_lifetimes
            .control_identity()
            .ptr_eq(&control.identity())
        {
            return Err(BindInjectedOutputEventsFailure {
                renderer: Some(self),
                control: Some(control),
            });
        }
        let (control_events, lifecycle) = match self.events {
            BoundInjectedRendererEvents::Output { control, lifecycle } => (control, lifecycle),
            #[cfg(test)]
            BoundInjectedRendererEvents::Legacy => {
                unreachable!("exact output binding requires exact output event setup")
            }
        };
        if !self.renderer.matches_injected_event_loop(&lifecycle)
            || !self
                .renderer
                .matches_injected_control_events(&control_events)
            || !control_events.matches_gate(&control.admission_gate())
        {
            return Err(BindInjectedOutputEventsFailure {
                renderer: Some(BoundInjectedRenderer {
                    renderer: self.renderer,
                    node_lifetimes: self.node_lifetimes,
                    events: BoundInjectedRendererEvents::Output {
                        control: control_events,
                        lifecycle,
                    },
                }),
                control: Some(control),
            });
        }
        let concrete_events = InjectedConcreteEventBinding {
            events: control_events,
            control_identity: control.identity(),
        };
        Ok((
            BoundInjectedOutputRenderer::new(
                ExactBoundInjectedRenderer {
                    renderer: self.renderer,
                    node_lifetimes: self.node_lifetimes,
                },
                control,
                lifecycle,
            ),
            concrete_events,
        ))
    }

    #[cfg(test)]
    pub(crate) fn into_render_thread_for_test(self) -> (RenderThread, InjectedNodeLifetimeOwner) {
        let BoundInjectedRendererEvents::Legacy = self.events else {
            unreachable!("raw test extraction requires test-only legacy events")
        };
        (self.renderer, self.node_lifetimes)
    }
}

impl InjectedControlRenderInit {
    #[cfg(test)]
    pub(crate) fn reclaim_one_ordinary_for_test(&self) -> bool {
        match self.receiver.try_recv() {
            Ok(ControlMessage::InjectedBatch { batch, physical }) => {
                drop(physical);
                drop(batch);
                true
            }
            Ok(other) => {
                drop(other);
                panic!("expected one injected ordinary batch")
            }
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::result_large_err)] // retry must return the exact init and bootstrap intact
    #[cfg(test)]
    pub(crate) fn build_render_thread(
        self,
        node_lifetimes: InjectedNodeLifetimeBootstrap,
        sample_rate: f32,
        number_of_channels: usize,
        state: Arc<std::sync::atomic::AtomicU8>,
        frames_played: Arc<AtomicU64>,
        stats: AudioStats,
        event_sender: Sender<EventDispatch>,
    ) -> Result<BoundInjectedRenderer, BuildInjectedRenderFailure> {
        if !self
            .identity
            .ptr_eq(node_lifetimes.owner.control_identity())
        {
            return Err(BuildInjectedRenderFailure {
                init: self,
                node_lifetimes,
            });
        }
        Ok(self.build_render_thread_unchecked(
            node_lifetimes,
            sample_rate,
            number_of_channels,
            frames_played,
            stats,
            InjectedRenderEvents::Legacy {
                sender: event_sender,
                state,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::result_large_err)]
    pub(crate) fn build_output_render_thread(
        self,
        node_lifetimes: InjectedNodeLifetimeBootstrap,
        sample_rate: f32,
        number_of_channels: usize,
        frames_played: Arc<AtomicU64>,
        stats: AudioStats,
        events: InjectedEventDispatchSetup,
    ) -> Result<BoundInjectedRenderer, BuildInjectedOutputRenderFailure> {
        if !self
            .identity
            .ptr_eq(node_lifetimes.owner.control_identity())
        {
            return Err(BuildInjectedOutputRenderFailure {
                init: self,
                node_lifetimes,
                events,
            });
        }
        let events = events.bind(&self);
        let bound = self.build_render_thread_unchecked(
            node_lifetimes,
            sample_rate,
            number_of_channels,
            frames_played,
            stats,
            InjectedRenderEvents::Output(events),
        );
        Ok(bound)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_render_thread_unchecked(
        self,
        node_lifetimes: InjectedNodeLifetimeBootstrap,
        sample_rate: f32,
        number_of_channels: usize,
        frames_played: Arc<AtomicU64>,
        stats: AudioStats,
        event_sender: InjectedRenderEvents,
    ) -> BoundInjectedRenderer {
        let InjectedNodeLifetimeBootstrap {
            owner: node_lifetimes,
            graph,
        } = node_lifetimes;
        let (mut renderer, events) = match event_sender {
            #[cfg(test)]
            InjectedRenderEvents::Legacy {
                sender: event_sender,
                state,
            } => (
                RenderThread::new(
                    sample_rate,
                    number_of_channels,
                    self.receiver,
                    state,
                    frames_played,
                    stats,
                    event_sender,
                    self.applied,
                ),
                BoundInjectedRendererEvents::Legacy,
            ),
            InjectedRenderEvents::Output(event_dispatch) => {
                let (renderer, control, lifecycle) = event_dispatch.install_renderer(
                    sample_rate,
                    number_of_channels,
                    self.receiver,
                    frames_played,
                    stats,
                    self.applied,
                );
                (
                    renderer,
                    BoundInjectedRendererEvents::Output { control, lifecycle },
                )
            }
        };
        if renderer.install_injected_graph(graph).is_err() {
            unreachable!("new injected renderer has no graph");
        }
        if renderer
            .set_injected_physical_credit_owners(self.physical_owners)
            .is_err()
        {
            unreachable!("new renderer has no injected physical owners");
        }
        if renderer
            .set_graph_lifecycle_publisher(self.lifecycle_publisher)
            .is_err()
        {
            unreachable!("new renderer has no lifecycle publisher");
        }
        BoundInjectedRenderer {
            renderer,
            node_lifetimes,
            events,
        }
    }
}

enum InjectedRenderEvents {
    #[cfg(test)]
    Legacy {
        sender: Sender<EventDispatch>,
        state: Arc<std::sync::atomic::AtomicU8>,
    },
    Output(BoundInjectedEventDispatch),
}

pub(crate) struct BuildInjectedOutputRenderFailure {
    pub(crate) init: InjectedControlRenderInit,
    pub(crate) node_lifetimes: InjectedNodeLifetimeBootstrap,
    pub(crate) events: InjectedEventDispatchSetup,
}

/// Constructs an exact `N + 1` channel: N ordinary envelopes plus one Close-only reservation.
pub(crate) fn injected_control_channel(
    gate: InjectedContextAdmissionGate,
    ordinary_capacity: usize,
    initially_suspended: bool,
) -> Result<
    (
        InjectedControlProducer,
        InjectedControlLifecycleOwner,
        InjectedControlRenderInit,
    ),
    InjectedControlError,
> {
    if ordinary_capacity == 0 {
        return Err(InjectedControlError::InvalidOrdinaryCapacity);
    }
    let channel_capacity = ordinary_capacity
        .checked_add(1)
        .ok_or(InjectedControlError::InvalidOrdinaryCapacity)?;
    let (sender, receiver) = crossbeam_channel::bounded(channel_capacity);
    let (activity_send, activity_recv) = crossbeam_channel::bounded(1);
    let ordinary_physical =
        InjectedPhysicalCreditPool::new(ordinary_capacity, activity_send.clone());
    let lifecycle_physical = InjectedPhysicalCreditPool::new(1, activity_send.clone());
    let physical_owners = InjectedPhysicalCreditOwners {
        ordinary: ordinary_physical.clone(),
        lifecycle: lifecycle_physical.clone(),
    };
    let (lifecycle_publisher, watcher) = graph_lifecycle_ack_pair();
    let applied = ControlBatchApplied::default();
    let event_gate = gate.clone();
    let inner = Arc::new(InjectedControlInner {
        gate,
        state: Mutex::new(InjectedControlState {
            sender,
            phase: TransportPhase::Open,
            disconnected: false,
            initially_suspended,
            next_batch_sequence: 1,
            staged: VecDeque::with_capacity(STAGED_ENVELOPE_LIMIT),
        }),
        logical_commands: InjectedCommandCreditPool::new(
            LOGICAL_COMMAND_LIMIT,
            activity_send.clone(),
        ),
        batch_storage: ControlBatchStoragePool::new_with_activity(activity_send.clone()),
        ordinary_physical,
        lifecycle_physical,
        staging_slots: StagingSlotPool::new(activity_send),
        batch_sequence_reservations: Arc::new(AtomicUsize::new(0)),
        close_in_flight: Arc::new(AtomicBool::new(false)),
        last_submitted_batch_sequence: AtomicU64::new(0),
        applied: applied.clone(),
        accepted_finalizer_failed: AtomicBool::new(false),
    });
    Ok((
        InjectedControlProducer {
            inner: Arc::clone(&inner),
        },
        InjectedControlLifecycleOwner {
            inner: Arc::clone(&inner),
            watcher,
            activity: activity_recv,
        },
        InjectedControlRenderInit {
            receiver,
            physical_owners,
            lifecycle_publisher,
            applied,
            identity: InjectedControlIdentity(Arc::downgrade(&inner)),
            event_gate,
            initially_suspended,
        },
    ))
}

fn try_state(
    inner: &InjectedControlInner,
) -> Result<std::sync::MutexGuard<'_, InjectedControlState>, InjectedControlError> {
    match inner.state.try_lock() {
        Ok(state) => Ok(state),
        Err(TryLockError::WouldBlock) => Err(InjectedControlError::Contended),
        Err(TryLockError::Poisoned(_)) => Err(InjectedControlError::Poisoned),
    }
}

/// Every capacity needed by one graph mutation, acquired before its payload factory or mirror
/// mutation runs. The admission remains live through commit or rejected-payload destruction.
#[must_use]
pub(crate) struct ControlBatchReservation {
    inner: Arc<InjectedControlInner>,
    command_count: usize,
    storage: ControlBatchPermit,
    command_credit: InjectedCommandCredit,
    sequence_reservation: BatchSequenceReservation,
    placement: ReservedPlacement,
    admission: GraphControlAdmission,
}

/// Dedicated one-command reservation for node-handle teardown. Keeping this wrapper separate from
/// ordinary batch preparation prevents `ControlHandleDropped` from entering the general mutation
/// whitelist while still using the same bounded transport and admission accounting.
#[must_use]
pub(crate) struct ControlHandleDroppedReservation(ControlBatchReservation);

impl ControlBatchReservation {
    pub(crate) fn prepare_with<F>(
        self,
        factory: F,
    ) -> Result<PreparedControlBatch, PrepareControlFailure>
    where
        F: FnOnce() -> Vec<ControlMessage>,
    {
        let commands = factory();
        let error = if commands.len() != self.command_count {
            Some(if commands.is_empty() {
                InjectedControlError::Empty
            } else {
                InjectedControlError::TooLarge
            })
        } else if commands.iter().any(|command| !command.is_batchable()) {
            Some(InjectedControlError::UnsupportedCommand)
        } else {
            None
        };
        if let Some(error) = error {
            return Err(PrepareControlFailure {
                commands,
                reservation: self,
                error,
            });
        }
        Ok(self.into_prevalidated(commands))
    }

    /// Concrete typed transactions may use this only after proving the reserved count and closed
    /// command set before moving rollback tokens into `commands`.
    pub(super) fn into_prevalidated(self, commands: Vec<ControlMessage>) -> PreparedControlBatch {
        let Self {
            inner,
            storage,
            command_credit,
            sequence_reservation,
            placement,
            admission,
            ..
        } = self;
        PreparedControlBatch {
            commands: commands.into_boxed_slice(),
            inner,
            storage,
            command_credit,
            sequence_reservation,
            placement,
            admission,
        }
    }
}

impl ControlHandleDroppedReservation {
    pub(crate) fn prepare(self, id: super::AudioNodeId) -> PreparedControlBatch {
        self.0
            .into_prevalidated(vec![ControlMessage::ControlHandleDropped { id }])
    }
}

/// Commands precede the reservation so their destructor runs while admission remains live.
pub(crate) struct PrepareControlFailure {
    commands: Vec<ControlMessage>,
    reservation: ControlBatchReservation,
    pub(crate) error: InjectedControlError,
}

impl PrepareControlFailure {
    pub(crate) fn commands_len(&self) -> usize {
        self.commands.len()
    }
}

/// Exact-capacity payload with its original operation admission and all credits still attached.
#[must_use]
pub(crate) struct PreparedControlBatch {
    commands: Box<[ControlMessage]>,
    inner: Arc<InjectedControlInner>,
    storage: ControlBatchPermit,
    command_credit: InjectedCommandCredit,
    sequence_reservation: BatchSequenceReservation,
    placement: ReservedPlacement,
    admission: GraphControlAdmission,
}

impl PreparedControlBatch {
    pub(crate) fn len(&self) -> usize {
        self.commands.len()
    }
}

impl std::fmt::Debug for PreparedControlBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedControlBatch")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

pub(crate) struct CommitControlFailure {
    pub(crate) error: InjectedControlError,
    pub(crate) batch: PreparedControlBatch,
}

struct RejectedControlAuthorities {
    _inner: Arc<InjectedControlInner>,
    _storage: ControlBatchPermit,
    _command_credit: InjectedCommandCredit,
    _sequence_reservation: BatchSequenceReservation,
    _placement: ReservedPlacement,
    _admission: GraphControlAdmission,
}

pub(super) enum RejectedControlRollback<R> {
    Completed(R),
    Panicked,
}

impl CommitControlFailure {
    /// Runs one typed rollback over the owned rejected commands while every transport authority,
    /// credit, and the graph admission remain live. The callback must recover all external
    /// ownership tokens before destroying any other command payload.
    pub(super) fn rollback_with_commands<R>(
        self,
        rollback: impl FnOnce(Box<[ControlMessage]>) -> R,
    ) -> (InjectedControlError, RejectedControlRollback<R>) {
        let Self { error, batch } = self;
        let PreparedControlBatch {
            commands,
            inner,
            storage,
            command_credit,
            sequence_reservation,
            placement,
            admission,
        } = batch;
        let authorities = RejectedControlAuthorities {
            _inner: inner,
            _storage: storage,
            _command_credit: command_credit,
            _sequence_reservation: sequence_reservation,
            _placement: placement,
            _admission: admission,
        };
        let result = match panic::catch_unwind(AssertUnwindSafe(|| rollback(commands))) {
            Ok(result) => RejectedControlRollback::Completed(result),
            Err(payload) => {
                std::mem::forget(payload);
                RejectedControlRollback::Panicked
            }
        };
        drop(authorities);
        (error, result)
    }
}

impl std::fmt::Debug for CommitControlFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitControlFailure")
            .field("error", &self.error)
            .field("batch_len", &self.batch.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommitControlOutcome {
    Enqueued { sequence: u64 },
    Staged,
}

/// Fixed error reported by a mandatory accepted-batch finalizer. The finalizer itself is a
/// monomorphized, stack-owned control-side operation; this seam performs no allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcceptedBatchFinalizeError {
    Rejected,
    Panicked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AcceptedBatchFinalizeFailure {
    pub(crate) outcome: CommitControlOutcome,
    pub(crate) error: AcceptedBatchFinalizeError,
}

pub(crate) enum CommitWithFinalizeFailure {
    NotAccepted(CommitControlFailure),
    AcceptedFinalizer(AcceptedBatchFinalizeFailure),
}

/// Accepted placement whose short operation authorities deliberately remain live until the
/// mandatory finalizer (or ordinary no-op completion) finishes.
struct AcceptedControlCommit {
    inner: Arc<InjectedControlInner>,
    outcome: CommitControlOutcome,
    sequence_reservation: Option<BatchSequenceReservation>,
    admission: GraphControlAdmission,
}

impl AcceptedControlCommit {
    fn complete(self) -> CommitControlOutcome {
        self.outcome
    }

    fn finalize<F>(self, finalizer: F) -> Result<CommitControlOutcome, CommitWithFinalizeFailure>
    where
        F: FnOnce(CommitControlOutcome) -> Result<(), AcceptedBatchFinalizeError> + Copy,
    {
        let finalized = panic::catch_unwind(AssertUnwindSafe(|| finalizer(self.outcome)));
        let failure = match finalized {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(payload) => {
                // A hostile panic payload may panic again from Drop. The accepted render payload
                // and its accounting stay owned by transport; quarantine the payload itself.
                std::mem::forget(payload);
                Some(AcceptedBatchFinalizeError::Panicked)
            }
        };
        if failure.is_some() {
            self.inner
                .accepted_finalizer_failed
                .store(true, Ordering::Release);
        }
        let outcome = self.outcome;
        drop(self.sequence_reservation);
        drop(self.admission);
        match failure {
            None => Ok(outcome),
            Some(error) => Err(CommitWithFinalizeFailure::AcceptedFinalizer(
                AcceptedBatchFinalizeFailure { outcome, error },
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FlushControlOutcome {
    pub(crate) enqueued: usize,
    pub(crate) remaining_staged: usize,
}

impl InjectedControlProducer {
    #[cfg(test)]
    pub(crate) fn try_commit_prevalidated_for_test(
        &self,
        commands: Vec<ControlMessage>,
    ) -> Result<CommitControlOutcome, InjectedControlError> {
        let reservation = self.try_begin_operation(commands.len())?;
        let prepared = reservation.into_prevalidated(commands);
        self.try_commit(prepared).map_err(|failure| failure.error)
    }

    pub(crate) fn admission_gate(&self) -> InjectedContextAdmissionGate {
        self.inner.gate.clone()
    }

    pub(crate) fn identity(&self) -> InjectedControlIdentity {
        InjectedControlIdentity(Arc::downgrade(&self.inner))
    }

    /// Reserves logical, storage, and state-specific placement capacity before caller mutation.
    /// Physical FIFO and sequence order are commit order. A future mirror transaction must
    /// serialize its mirror mutation plus commit if concurrent callers require mutation order.
    pub(crate) fn try_begin_operation(
        &self,
        command_count: usize,
    ) -> Result<ControlBatchReservation, InjectedControlError> {
        if command_count == 0 {
            return Err(InjectedControlError::Empty);
        }
        if command_count > CONTROL_BATCH_CAPACITY {
            return Err(InjectedControlError::TooLarge);
        }
        let admission = self.inner.gate.try_graph_control()?;
        let state = try_state(&self.inner)?;
        if state.disconnected {
            return Err(InjectedControlError::Disconnected);
        }
        if self.inner.accepted_finalizer_failed.load(Ordering::Acquire) {
            return Err(InjectedControlError::ProtocolViolation);
        }
        match state.phase {
            TransportPhase::Open => {}
            TransportPhase::SealStarted | TransportPhase::Sealed => {
                return Err(InjectedControlError::Sealed)
            }
            TransportPhase::Failed => return Err(InjectedControlError::ProtocolViolation),
        }
        let command_credit = self
            .inner
            .logical_commands
            .try_acquire(command_count)
            .ok_or(InjectedControlError::LogicalCommandCredits)?;
        let storage = self
            .inner
            .batch_storage
            .try_acquire()
            .ok_or(InjectedControlError::BatchStorageCredits)?;
        let outstanding = self
            .inner
            .batch_sequence_reservations
            .load(Ordering::Acquire);
        let Some(reserved_sequence) = state.next_batch_sequence.checked_add(outstanding as u64)
        else {
            return Err(InjectedControlError::SequenceExhausted);
        };
        if reserved_sequence == u64::MAX {
            return Err(InjectedControlError::SequenceExhausted);
        }
        self.inner
            .batch_sequence_reservations
            .fetch_add(1, Ordering::AcqRel);
        let sequence_reservation =
            BatchSequenceReservation(Arc::clone(&self.inner.batch_sequence_reservations));
        let placement = if state.initially_suspended || !state.staged.is_empty() {
            ReservedPlacement::Staged(
                self.inner
                    .staging_slots
                    .try_acquire()
                    .ok_or(InjectedControlError::StagingFull)?,
            )
        } else {
            let physical = self
                .inner
                .ordinary_physical
                .try_acquire()
                .ok_or(InjectedControlError::OrdinaryPhysicalCredits)?;
            ReservedPlacement::Running(physical)
        };
        drop(state);
        Ok(ControlBatchReservation {
            inner: Arc::clone(&self.inner),
            command_count,
            storage,
            command_credit,
            sequence_reservation,
            placement,
            admission,
        })
    }

    pub(crate) fn try_begin_control_handle_drop(
        &self,
    ) -> Result<ControlHandleDroppedReservation, InjectedControlError> {
        self.try_begin_operation(1)
            .map(ControlHandleDroppedReservation)
    }

    pub(crate) fn try_commit(
        &self,
        batch: PreparedControlBatch,
    ) -> Result<CommitControlOutcome, CommitControlFailure> {
        self.try_commit_retained(batch)
            .map(AcceptedControlCommit::complete)
    }

    /// Commits an ordinary batch and runs one mandatory finalizer only after its envelope is
    /// physically enqueued or accepted into suspended staging. The operation's graph admission
    /// and running sequence reservation remain live until the finalizer returns.
    ///
    /// The callback contract is fixed, bounded, nonblocking, and allocation-free. `Copy` only
    /// prevents it from directly owning destructor-bearing rollback/registration guards; it does
    /// not prove those runtime properties or prevent indirect mutation through copied references.
    /// Finalizer side effects are not rolled back on failure. A future constructor therefore still
    /// needs a concrete transaction wrapper to disarm provisional ids and arm registrations; this
    /// ordering hook alone must not own those guards.
    ///
    /// A typed failure or panic is terminal: the accepted payload remains owned by
    /// transport/renderer, the panic payload is deliberately forgotten, and this transport
    /// rejects later ordinary work. Such a failure is never reported as ordinary commit success.
    pub(crate) fn try_commit_with_finalize<F>(
        &self,
        batch: PreparedControlBatch,
        finalizer: F,
    ) -> Result<CommitControlOutcome, CommitWithFinalizeFailure>
    where
        F: FnOnce(CommitControlOutcome) -> Result<(), AcceptedBatchFinalizeError> + Copy,
    {
        self.try_commit_retained(batch)
            .map_err(CommitWithFinalizeFailure::NotAccepted)?
            .finalize(finalizer)
    }

    fn try_commit_retained(
        &self,
        batch: PreparedControlBatch,
    ) -> Result<AcceptedControlCommit, CommitControlFailure> {
        if !Arc::ptr_eq(&self.inner, &batch.inner) {
            return Err(CommitControlFailure {
                error: InjectedControlError::ProtocolViolation,
                batch,
            });
        }
        let PreparedControlBatch {
            commands,
            inner,
            storage,
            command_credit,
            sequence_reservation,
            placement,
            admission,
        } = batch;
        let mut state = match try_state(&inner) {
            Ok(state) => state,
            Err(error) => {
                return Err(CommitControlFailure {
                    error,
                    batch: PreparedControlBatch {
                        commands,
                        inner: Arc::clone(&inner),
                        storage,
                        command_credit,
                        sequence_reservation,
                        placement,
                        admission,
                    },
                });
            }
        };
        if state.disconnected {
            drop(state);
            return Err(CommitControlFailure {
                error: InjectedControlError::Disconnected,
                batch: PreparedControlBatch {
                    commands,
                    inner,
                    storage,
                    command_credit,
                    sequence_reservation,
                    placement,
                    admission,
                },
            });
        }
        if inner.accepted_finalizer_failed.load(Ordering::Acquire) {
            drop(state);
            return Err(CommitControlFailure {
                error: InjectedControlError::ProtocolViolation,
                batch: PreparedControlBatch {
                    commands,
                    inner,
                    storage,
                    command_credit,
                    sequence_reservation,
                    placement,
                    admission,
                },
            });
        }
        if matches!(state.phase, TransportPhase::Sealed | TransportPhase::Failed) {
            let error = if state.phase == TransportPhase::Sealed {
                InjectedControlError::Sealed
            } else {
                InjectedControlError::ProtocolViolation
            };
            drop(state);
            return Err(CommitControlFailure {
                error,
                batch: PreparedControlBatch {
                    commands,
                    inner,
                    storage,
                    command_credit,
                    sequence_reservation,
                    placement,
                    admission,
                },
            });
        }
        let mut node = injected_control_batch_node(commands, storage, command_credit);
        match placement {
            ReservedPlacement::Staged(slot) => {
                state.staged.push_back(StagedControlBatch {
                    batch: node,
                    _slot: slot,
                    _sequence: sequence_reservation,
                });
                drop(state);
                Ok(AcceptedControlCommit {
                    inner,
                    outcome: CommitControlOutcome::Staged,
                    sequence_reservation: None,
                    admission,
                })
            }
            ReservedPlacement::Running(physical) => {
                let sequence = state.next_batch_sequence;
                if sequence == u64::MAX
                    || !control_batch_storage_mut(&mut node)
                        .assign_sequence_before_enqueue(sequence)
                {
                    state.phase = TransportPhase::Failed;
                    drop(state);
                    let (commands, storage, command_credit) =
                        recover_unsubmitted_injected_batch(node);
                    return Err(CommitControlFailure {
                        error: InjectedControlError::ProtocolViolation,
                        batch: PreparedControlBatch {
                            commands,
                            inner,
                            storage,
                            command_credit,
                            sequence_reservation,
                            placement: ReservedPlacement::Running(physical),
                            admission,
                        },
                    });
                }
                match state.sender.try_send(ControlMessage::InjectedBatch {
                    batch: node,
                    physical,
                }) {
                    Ok(()) => {
                        state.next_batch_sequence += 1;
                        inner
                            .last_submitted_batch_sequence
                            .store(sequence, Ordering::Release);
                        drop(state);
                        Ok(AcceptedControlCommit {
                            inner,
                            outcome: CommitControlOutcome::Enqueued { sequence },
                            sequence_reservation: Some(sequence_reservation),
                            admission,
                        })
                    }
                    Err(TrySendError::Full(ControlMessage::InjectedBatch {
                        mut batch,
                        physical,
                    })) => {
                        state.phase = TransportPhase::Failed;
                        let _ = control_batch_storage_mut(&mut batch)
                            .clear_unsubmitted_sequence(sequence);
                        drop(state);
                        let (commands, storage, command_credit) =
                            recover_unsubmitted_injected_batch(batch);
                        Err(CommitControlFailure {
                            error: InjectedControlError::ProtocolViolation,
                            batch: PreparedControlBatch {
                                commands,
                                inner,
                                storage,
                                command_credit,
                                sequence_reservation,
                                placement: ReservedPlacement::Running(physical),
                                admission,
                            },
                        })
                    }
                    Err(TrySendError::Disconnected(ControlMessage::InjectedBatch {
                        mut batch,
                        physical,
                    })) => {
                        state.phase = TransportPhase::Failed;
                        state.disconnected = true;
                        let _ = control_batch_storage_mut(&mut batch)
                            .clear_unsubmitted_sequence(sequence);
                        drop(state);
                        let (commands, storage, command_credit) =
                            recover_unsubmitted_injected_batch(batch);
                        Err(CommitControlFailure {
                            error: InjectedControlError::Disconnected,
                            batch: PreparedControlBatch {
                                commands,
                                inner,
                                storage,
                                command_credit,
                                sequence_reservation,
                                placement: ReservedPlacement::Running(physical),
                                admission,
                            },
                        })
                    }
                    Err(_) => unreachable!("private sender returns its submitted variant"),
                }
            }
        }
    }

    /// Flushes as many staged records as currently available ordinary credits permit.
    pub(crate) fn try_flush(&self) -> Result<FlushControlOutcome, InjectedControlError> {
        let _admission = self.inner.gate.try_graph_control()?;
        let mut state = try_state(&self.inner)?;
        if state.disconnected {
            return Err(InjectedControlError::Disconnected);
        }
        if self.inner.accepted_finalizer_failed.load(Ordering::Acquire) {
            return Err(InjectedControlError::ProtocolViolation);
        }
        if state.phase != TransportPhase::Open {
            return Err(InjectedControlError::Sealed);
        }
        let mut enqueued = 0;
        while !state.staged.is_empty() {
            if state.next_batch_sequence == u64::MAX {
                return Err(InjectedControlError::SequenceExhausted);
            }
            let Some(physical) = self.inner.ordinary_physical.try_acquire() else {
                break;
            };
            let sequence = state.next_batch_sequence;
            let mut staged = state.staged.pop_front().expect("staged front exists");
            if !control_batch_storage_mut(&mut staged.batch)
                .assign_sequence_before_enqueue(sequence)
            {
                state.phase = TransportPhase::Failed;
                state.staged.push_front(staged);
                return Err(InjectedControlError::ProtocolViolation);
            }
            match state.sender.try_send(ControlMessage::InjectedBatch {
                batch: staged.batch,
                physical,
            }) {
                Ok(()) => {
                    drop(staged._slot);
                    drop(staged._sequence);
                    state.next_batch_sequence += 1;
                    self.inner
                        .last_submitted_batch_sequence
                        .store(sequence, Ordering::Release);
                    enqueued += 1;
                }
                Err(TrySendError::Full(ControlMessage::InjectedBatch { mut batch, .. })) => {
                    state.phase = TransportPhase::Failed;
                    let _ =
                        control_batch_storage_mut(&mut batch).clear_unsubmitted_sequence(sequence);
                    staged.batch = batch;
                    state.staged.push_front(staged);
                    return Err(InjectedControlError::ProtocolViolation);
                }
                Err(TrySendError::Disconnected(ControlMessage::InjectedBatch {
                    mut batch,
                    ..
                })) => {
                    state.phase = TransportPhase::Failed;
                    state.disconnected = true;
                    let _ =
                        control_batch_storage_mut(&mut batch).clear_unsubmitted_sequence(sequence);
                    staged.batch = batch;
                    state.staged.push_front(staged);
                    return Err(InjectedControlError::Disconnected);
                }
                Err(_) => unreachable!("private sender returns its submitted variant"),
            }
        }
        Ok(FlushControlOutcome {
            enqueued,
            remaining_staged: state.staged.len(),
        })
    }

    pub(crate) fn last_submitted_batch_sequence(&self) -> u64 {
        self.inner
            .last_submitted_batch_sequence
            .load(Ordering::Acquire)
    }

    pub(crate) fn applied_batch_sequence(&self) -> u64 {
        self.inner.applied.load()
    }

    #[cfg(test)]
    pub(crate) fn accounting(&self) -> (usize, usize, usize, usize) {
        (
            self.inner.logical_commands.in_flight(),
            self.inner.batch_storage.in_flight(),
            self.inner.ordinary_physical.in_flight(),
            self.inner.staging_slots.in_flight(),
        )
    }

    #[cfg(test)]
    fn poison_transport(&self) {
        let _state = self.inner.state.lock().unwrap();
        panic!("poison injected transport");
    }

    #[cfg(test)]
    fn set_next_batch_sequence(&self, sequence: u64) {
        self.inner.state.lock().unwrap().next_batch_sequence = sequence;
    }

    #[cfg(test)]
    pub(crate) fn fail_transport(&self) {
        self.inner.state.lock().unwrap().phase = TransportPhase::Failed;
    }

    #[cfg(test)]
    pub(crate) fn hold_transport_state_for_test(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        let _state = self.inner.state.lock().unwrap();
        entered.send(()).unwrap();
        release.recv().unwrap();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ControlCloseDegradation {
    pub(crate) transport_poison_recovered: bool,
    pub(crate) capacity_worker_panicked: bool,
    pub(crate) prior_transport_failure: bool,
}

pub(crate) struct BeginControlCloseFailure {
    pub(crate) error: InjectedControlError,
    pub(crate) owner: InjectedControlLifecycleOwner,
}

/// Admissions are irreversibly sealed. Retirement owns any registered capacity worker and the
/// authoritative drain which includes every pre-seal prepared graph operation.
pub(crate) struct ControlCloseRetirement {
    inner: Arc<InjectedControlInner>,
    watcher: GraphLifecycleWatcher,
    capacity_worker: Option<CapacityWorkerRetirement>,
    drain: AdmissionDrain,
    degradation: ControlCloseDegradation,
}

impl InjectedControlLifecycleOwner {
    pub(crate) fn identity(&self) -> InjectedControlIdentity {
        InjectedControlIdentity(Arc::downgrade(&self.inner))
    }

    pub(crate) fn admission_gate(&self) -> InjectedContextAdmissionGate {
        self.inner.gate.clone()
    }

    /// Best-effort credit-release hint for the private B3b non-RT lifecycle driver. The receiver
    /// is borrowed so no competing consumer can be retained through this API. Callers must retry
    /// from authoritative teardown/transport state; a wake is never an acknowledgement.
    pub(crate) const fn credit_activity_receiver(&self) -> &crossbeam_channel::Receiver<()> {
        &self.activity
    }

    /// Seals the shared gate first, then recovers a poisoned local transport mutex if necessary.
    /// Gate poison cannot establish the irreversible boundary and therefore returns a quarantine
    /// failure retaining the unique owner.
    pub(crate) fn try_begin_close(
        self,
    ) -> Result<ControlCloseRetirement, BeginControlCloseFailure> {
        let admissions = match self.inner.gate.try_seal() {
            Ok(admissions) => admissions,
            Err(error) => {
                return Err(BeginControlCloseFailure {
                    error: error.into(),
                    owner: self,
                });
            }
        };
        let mut degradation = ControlCloseDegradation::default();
        let mut state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                degradation.transport_poison_recovered = true;
                poisoned.into_inner()
            }
        };
        degradation.prior_transport_failure = state.phase == TransportPhase::Failed
            || self.inner.accepted_finalizer_failed.load(Ordering::Acquire);
        state.phase = TransportPhase::SealStarted;
        let (capacity_worker, drain) = admissions.into_parts();
        drop(state);
        Ok(ControlCloseRetirement {
            inner: self.inner,
            watcher: self.watcher,
            capacity_worker,
            drain,
            degradation,
        })
    }
}

/// Unforgeable post-drain authority. It can only be built after capacity retirement and all graph
/// preparation admissions reach zero.
pub(crate) struct DrainedControlClose {
    inner: Arc<InjectedControlInner>,
    watcher: GraphLifecycleWatcher,
    degradation: ControlCloseDegradation,
}

impl ControlCloseRetirement {
    /// Runs off RT. A panicking capacity worker is still authoritatively joined; its hostile panic
    /// payload is forgotten before any later diagnostics and reported as degraded retirement.
    pub(crate) fn retire_and_wait(mut self) -> (AdmissionSnapshot, DrainedControlClose) {
        if let Some(worker) = self.capacity_worker.take() {
            if let Err(CapacityWorkerJoinError::Panicked(payload)) = worker.stop_and_join() {
                std::mem::forget(payload);
                self.degradation.capacity_worker_panicked = true;
            }
        }
        let snapshot = self.drain.wait();
        debug_assert!(snapshot.is_drained());
        // A pre-seal operation may have been inside its accepted finalizer when sealing began.
        // Refresh only after the admission drain proves that finalizer has returned.
        self.degradation.prior_transport_failure |=
            self.inner.accepted_finalizer_failed.load(Ordering::Acquire);
        (
            snapshot,
            DrainedControlClose {
                inner: self.inner,
                watcher: self.watcher,
                degradation: self.degradation,
            },
        )
    }
}

/// Staged graph payloads extracted only after the admission drain. Dropping this value runs their
/// destructors on the caller's non-render thread.
#[must_use]
pub(crate) struct ExtractedControlPayloads {
    staged: VecDeque<StagedControlBatch>,
    last_submitted_batch_sequence: u64,
}

impl ExtractedControlPayloads {
    pub(crate) fn staged_len(&self) -> usize {
        self.staged.len()
    }

    pub(crate) fn last_submitted_batch_sequence(&self) -> u64 {
        self.last_submitted_batch_sequence
    }

    /// Retires every staged envelope on the lifecycle thread while containing hostile payload
    /// destructors. A panic makes later whole-graph proof ineligible because an exact accepted
    /// payload may have been leaked during unwinding.
    pub(crate) fn retire_off_thread(mut self) -> ExtractedPayloadRetirement {
        let mut panicked = false;
        while let Some(batch) = self.staged.pop_front() {
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| drop(batch))) {
                std::mem::forget(payload);
                panicked = true;
            }
        }
        ExtractedPayloadRetirement { panicked }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExtractedPayloadRetirement {
    pub(crate) panicked: bool,
}

struct CloseLifecycleLease {
    state: Arc<AtomicBool>,
}

impl CloseLifecycleLease {
    fn acquire(state: &Arc<AtomicBool>) -> Option<Self> {
        state
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self {
                state: Arc::clone(state),
            })
    }

    fn acknowledge(self) {
        self.state.store(false, Ordering::Release);
    }
}

/// Sole Close observation token. It is non-clone and retains the lifecycle lease after physical
/// dequeue until the exact acknowledgement commit word is acquired.
#[must_use]
pub(crate) struct SubmittedControlClose {
    barrier: GraphLifecycleBarrier,
    watcher: GraphLifecycleWatcher,
    lease: CloseLifecycleLease,
}

impl SubmittedControlClose {
    pub(crate) fn barrier(&self) -> GraphLifecycleBarrier {
        self.barrier
    }

    pub(crate) fn snapshot(&self) -> GraphLifecycleSnapshot {
        self.watcher
            .snapshot(NonZeroU64::new(self.barrier.controller_sequence()).unwrap())
    }

    /// Borrowed, non-cloneable wake hint for lifecycle-worker selection. Wakes are lossy; callers
    /// must always re-read `snapshot` with Acquire ordering. Disconnect is terminal
    /// renderer/publisher retirement, never a successful Close acknowledgement.
    pub(crate) fn wake_receiver(&self) -> &crossbeam_channel::Receiver<()> {
        self.watcher.receiver()
    }

    pub(crate) fn try_observe_exact(
        self,
    ) -> Result<ObservedControlClose, ObserveControlCloseFailure> {
        let snapshot = self.snapshot();
        if matches!(snapshot, GraphLifecycleSnapshot::Pending) {
            return Err(ObserveControlCloseFailure::Pending(self));
        }
        if !matches!(
            &snapshot,
            GraphLifecycleSnapshot::Applied {
                barrier,
                observed_batch_sequence,
                outcome: GraphLifecycleOutcome::Applied,
            } if *barrier == self.barrier
                && *observed_batch_sequence == barrier.required_batch_sequence()
        ) {
            return Err(ObserveControlCloseFailure::Terminal {
                snapshot,
                close: self,
            });
        }
        let Self {
            barrier,
            watcher: _,
            lease,
        } = self;
        lease.acknowledge();
        Ok(ObservedControlClose { barrier, snapshot })
    }
}

/// Pending retains a usable wake/snapshot token. Terminal means the publisher authoritatively
/// committed a non-success result or advanced past this Close; both retain/quarantine the lease.
pub(crate) enum ObserveControlCloseFailure {
    Pending(SubmittedControlClose),
    Terminal {
        snapshot: GraphLifecycleSnapshot,
        close: SubmittedControlClose,
    },
}

impl ObserveControlCloseFailure {
    pub(crate) fn into_close(self) -> SubmittedControlClose {
        match self {
            Self::Pending(close) | Self::Terminal { close, .. } => close,
        }
    }
}

pub(crate) struct ObservedControlClose {
    pub(crate) barrier: GraphLifecycleBarrier,
    pub(crate) snapshot: GraphLifecycleSnapshot,
}

pub(crate) struct SealedControlTransport {
    pub(crate) payloads: ExtractedControlPayloads,
    pub(crate) close: SubmittedControlClose,
    pub(crate) degradation: ControlCloseDegradation,
}

pub(crate) struct FinishControlCloseFailure {
    pub(crate) error: InjectedControlError,
    pub(crate) payloads: ExtractedControlPayloads,
    pub(crate) degradation: ControlCloseDegradation,
}

impl DrainedControlClose {
    /// Recovers local mutex poison again if needed, extracts all unflushed staging, snapshots only
    /// physically submitted batch sequence, then consumes the unique Close-only lease and slot.
    pub(crate) fn finish(mut self) -> Result<SealedControlTransport, FinishControlCloseFailure> {
        let mut state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                self.degradation.transport_poison_recovered = true;
                poisoned.into_inner()
            }
        };
        let last_submitted_batch_sequence = self
            .inner
            .last_submitted_batch_sequence
            .load(Ordering::Acquire);
        let payloads = ExtractedControlPayloads {
            staged: std::mem::take(&mut state.staged),
            last_submitted_batch_sequence,
        };
        let Some(lease) = CloseLifecycleLease::acquire(&self.inner.close_in_flight) else {
            state.phase = TransportPhase::Failed;
            return Err(FinishControlCloseFailure {
                error: InjectedControlError::ProtocolViolation,
                payloads,
                degradation: self.degradation,
            });
        };
        let Some(physical) = self.inner.lifecycle_physical.try_acquire() else {
            state.phase = TransportPhase::Failed;
            return Err(FinishControlCloseFailure {
                error: InjectedControlError::ProtocolViolation,
                payloads,
                degradation: self.degradation,
            });
        };
        let barrier = GraphLifecycleBarrier::new(
            NonZeroU64::new(1).expect("the sole Close uses controller sequence one"),
            last_submitted_batch_sequence,
            GraphLifecycleTransition::Close,
        );
        let send = state
            .sender
            .try_send(ControlMessage::InjectedGraphLifecycleBarrier { barrier, physical });
        match send {
            Ok(()) => {
                state.phase = TransportPhase::Sealed;
                drop(state);
                Ok(SealedControlTransport {
                    payloads,
                    close: SubmittedControlClose {
                        barrier,
                        watcher: self.watcher,
                        lease,
                    },
                    degradation: self.degradation,
                })
            }
            Err(TrySendError::Full(_)) => {
                state.phase = TransportPhase::Failed;
                Err(FinishControlCloseFailure {
                    error: InjectedControlError::ProtocolViolation,
                    payloads,
                    degradation: self.degradation,
                })
            }
            Err(TrySendError::Disconnected(_)) => {
                state.phase = TransportPhase::Failed;
                state.disconnected = true;
                Err(FinishControlCloseFailure {
                    error: InjectedControlError::Disconnected,
                    payloads,
                    degradation: self.degradation,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::panic::panic_any;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, ThreadId};

    use super::*;
    use crate::context::{AudioContextState, AudioNodeId};
    use crate::message::GraphLifecycleOutcome;
    use crate::output::EndpointShutdownConfirmed;

    struct Harness {
        gate: InjectedContextAdmissionGate,
        producer: InjectedControlProducer,
        owner: Option<InjectedControlLifecycleOwner>,
        renderer: Option<RenderThread>,
        gc: Option<std::thread::JoinHandle<()>>,
        _node_lifetimes: InjectedNodeLifetimeOwner,
        _event_receiver: crossbeam_channel::Receiver<EventDispatch>,
    }

    impl Harness {
        fn new(ordinary_capacity: usize, suspended: bool) -> Self {
            let gate = InjectedContextAdmissionGate::new();
            let (producer, owner, init) =
                injected_control_channel(gate.clone(), ordinary_capacity, suspended).unwrap();
            let (event_sender, event_receiver) = crossbeam_channel::bounded(32);
            let (_allocator, node_ids, graph) = injected_node_id_pair(0);
            let (_registrar, node_lifetimes) = injected_node_lifetime_registry(
                DEFAULT_NODE_LIFETIME_CAPACITY,
                &producer,
                node_ids,
                graph,
            )
            .ok()
            .unwrap();
            let bound = init
                .build_render_thread(
                    node_lifetimes,
                    48_000.,
                    2,
                    Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
                    Arc::new(AtomicU64::new(0)),
                    AudioStats::new(),
                    event_sender,
                )
                .ok()
                .unwrap();
            let (mut renderer, node_lifetimes) = bound.into_render_thread_for_test();
            let gc = renderer.spawn_joinable_garbage_collector_thread().unwrap();
            Self {
                gate,
                producer,
                owner: Some(owner),
                renderer: Some(renderer),
                gc: Some(gc),
                _node_lifetimes: node_lifetimes,
                _event_receiver: event_receiver,
            }
        }

        fn callback(&mut self) {
            self.renderer
                .as_mut()
                .unwrap()
                .render(&mut [] as &mut [f32]);
        }

        fn finish_close(&mut self) -> SealedControlTransport {
            let retirement = self.owner.take().unwrap().try_begin_close().ok().unwrap();
            let (snapshot, drained) = retirement.retire_and_wait();
            assert!(snapshot.is_drained());
            drained.finish().ok().unwrap()
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            drop(self.renderer.take());
            if let Some(gc) = self.gc.take() {
                gc.join().unwrap();
            }
        }
    }

    fn prepare_with(
        producer: &InjectedControlProducer,
        commands: Vec<ControlMessage>,
    ) -> PreparedControlBatch {
        producer
            .try_begin_operation(commands.len())
            .unwrap()
            .prepare_with(|| commands)
            .ok()
            .unwrap()
    }

    #[test]
    fn accepted_finalizer_failure_holds_admission_and_reaches_close_degradation() {
        let mut harness = Harness::new(1, false);
        let batch = prepare_with(
            &harness.producer,
            vec![ControlMessage::MarkCycleBreaker { id: AudioNodeId(9) }],
        );
        let producer = harness.producer.clone();
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        thread::scope(|scope| {
            let commit = scope.spawn(move || {
                producer.try_commit_with_finalize(batch, |outcome| {
                    entered_send.send(outcome).unwrap();
                    release_recv.recv().unwrap();
                    Err(AcceptedBatchFinalizeError::Rejected)
                })
            });
            assert_eq!(
                entered_recv.recv().unwrap(),
                CommitControlOutcome::Enqueued { sequence: 1 }
            );

            let retirement = harness
                .owner
                .take()
                .unwrap()
                .try_begin_close()
                .ok()
                .unwrap();
            assert_eq!(retirement.drain.snapshot().graph_controls, 1);
            let (done_send, done_recv) = crossbeam_channel::bounded(1);
            let waiter = scope.spawn(move || {
                done_send.send(retirement.retire_and_wait()).unwrap();
            });
            assert!(done_recv.try_recv().is_err());
            release_send.send(()).unwrap();
            assert!(matches!(
                commit.join().unwrap(),
                Err(CommitWithFinalizeFailure::AcceptedFinalizer(_))
            ));
            let (snapshot, drained) = done_recv.recv().unwrap();
            assert!(snapshot.is_drained());
            assert!(drained.degradation.prior_transport_failure);
            waiter.join().unwrap();
        });
    }

    #[test]
    fn accepted_finalizer_failure_is_terminal_and_never_reports_commit_success() {
        let mut harness = Harness::new(1, true);
        let batch = prepare_with(
            &harness.producer,
            vec![ControlMessage::MarkCycleBreaker { id: AudioNodeId(9) }],
        );
        let sequence_reservations = &harness.producer.inner.batch_sequence_reservations;
        let failure = harness
            .producer
            .try_commit_with_finalize(batch, |outcome| {
                assert_eq!(outcome, CommitControlOutcome::Staged);
                assert_eq!(sequence_reservations.load(Ordering::Acquire), 1);
                Err(AcceptedBatchFinalizeError::Rejected)
            })
            .err()
            .unwrap();
        assert!(matches!(
            failure,
            CommitWithFinalizeFailure::AcceptedFinalizer(AcceptedBatchFinalizeFailure {
                outcome: CommitControlOutcome::Staged,
                error: AcceptedBatchFinalizeError::Rejected,
            })
        ));
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::ProtocolViolation)
        );
        // Staged placement retains the reservation in its queued record; Close extraction owns
        // and later releases it rather than the finalizer path doing so.
        assert_eq!(sequence_reservations.load(Ordering::Acquire), 1);
        let retirement = harness
            .owner
            .take()
            .unwrap()
            .try_begin_close()
            .ok()
            .unwrap();
        assert!(retirement.degradation.prior_transport_failure);
    }

    #[test]
    fn not_accepted_batch_never_runs_finalizer_and_remains_recoverable() {
        let first = Harness::new(1, false);
        let second = Harness::new(1, false);
        let batch = prepare_with(
            &first.producer,
            vec![ControlMessage::MarkCycleBreaker { id: AudioNodeId(9) }],
        );
        let calls = AtomicUsize::new(0);
        let failure = second
            .producer
            .try_commit_with_finalize(batch, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .err()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let CommitWithFinalizeFailure::NotAccepted(failure) = failure else {
            panic!("wrong producer must refuse before acceptance");
        };
        assert_eq!(
            first.producer.try_commit(failure.batch).unwrap(),
            CommitControlOutcome::Enqueued { sequence: 1 }
        );
    }

    #[test]
    fn accepted_finalizer_panic_forgets_hostile_payload_and_terminalizes_transport() {
        struct HostilePayload;
        impl Drop for HostilePayload {
            fn drop(&mut self) {
                panic!("hostile panic payload drop");
            }
        }

        let harness = Harness::new(1, false);
        let batch = prepare_with(
            &harness.producer,
            vec![ControlMessage::MarkCycleBreaker { id: AudioNodeId(9) }],
        );
        let calls = AtomicUsize::new(0);
        let sequence_reservations = &harness.producer.inner.batch_sequence_reservations;
        let failure = harness
            .producer
            .try_commit_with_finalize(batch, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(sequence_reservations.load(Ordering::Acquire), 1);
                panic_any(HostilePayload)
            })
            .err()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(sequence_reservations.load(Ordering::Acquire), 0);
        assert!(matches!(
            failure,
            CommitWithFinalizeFailure::AcceptedFinalizer(AcceptedBatchFinalizeFailure {
                error: AcceptedBatchFinalizeError::Panicked,
                ..
            })
        ));
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::ProtocolViolation)
        );
    }

    #[test]
    fn total_logical_budget_is_exactly_256_across_envelopes_and_mixes() {
        let harness = Harness::new(256, false);
        assert_eq!(harness.producer.applied_batch_sequence(), 0);
        let one_each: Vec<_> = (0..256)
            .map(|_| harness.producer.try_begin_operation(1).unwrap())
            .collect();
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::LogicalCommandCredits)
        );
        assert_eq!(harness.producer.accounting(), (256, 256, 256, 0));
        drop(one_each);

        let first = harness.producer.try_begin_operation(128).unwrap();
        let second = harness.producer.try_begin_operation(127).unwrap();
        assert_eq!(
            harness.producer.try_begin_operation(2).err(),
            Some(InjectedControlError::LogicalCommandCredits)
        );
        let final_one = harness.producer.try_begin_operation(1).unwrap();
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::LogicalCommandCredits)
        );
        drop((first, second, final_one));

        let exact = harness.producer.try_begin_operation(256).unwrap();
        let exact = exact
            .prepare_with(|| (0..256).map(|_| ControlMessage::TestNop).collect())
            .ok()
            .unwrap();
        assert_eq!(exact.len(), 256);
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::LogicalCommandCredits)
        );
    }

    #[test]
    fn refusal_precedes_factory_and_classification_is_closed() {
        let harness = Harness::new(1, false);
        let held = harness.producer.try_begin_operation(256).unwrap();
        let factory_ran = Arc::new(AtomicBool::new(false));
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::LogicalCommandCredits)
        );
        assert!(!factory_ran.load(Ordering::Acquire));
        drop(held);

        let factory_ran_clone = Arc::clone(&factory_ran);
        let failure = harness
            .producer
            .try_begin_operation(1)
            .unwrap()
            .prepare_with(move || {
                factory_ran_clone.store(true, Ordering::Release);
                vec![ControlMessage::GraphLifecycleBarrier(
                    GraphLifecycleBarrier::new(
                        NonZeroU64::new(1).unwrap(),
                        0,
                        GraphLifecycleTransition::Suspend,
                    ),
                )]
            })
            .err()
            .unwrap();
        assert!(factory_ran.load(Ordering::Acquire));
        assert_eq!(failure.error, InjectedControlError::UnsupportedCommand);
        assert_eq!(failure.commands_len(), 1);
    }

    #[test]
    fn running_capacity_is_reserved_before_factory_and_commit_has_no_capacity_failure() {
        let harness = Harness::new(1, false);
        let reservation = harness.producer.try_begin_operation(1).unwrap();
        assert_eq!(harness.producer.accounting(), (1, 1, 1, 0));
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::OrdinaryPhysicalCredits)
        );
        let prepared = reservation
            .prepare_with(|| vec![ControlMessage::TestNop])
            .ok()
            .unwrap();
        assert_eq!(
            harness.producer.try_commit(prepared).unwrap(),
            CommitControlOutcome::Enqueued { sequence: 1 }
        );
    }

    #[test]
    fn suspended_fifo_is_bounded_unsequenced_and_incrementally_flushed() {
        let mut harness = Harness::new(1, true);
        let log = Arc::new(Mutex::new(Vec::new()));
        for value in 1..=3 {
            let command = ControlMessage::TestMarker {
                value,
                log: Arc::clone(&log),
            };
            assert_eq!(
                harness
                    .producer
                    .try_commit(prepare_with(&harness.producer, vec![command]))
                    .unwrap(),
                CommitControlOutcome::Staged
            );
        }
        assert_eq!(harness.producer.last_submitted_batch_sequence(), 0);
        assert_eq!(harness.producer.accounting(), (3, 3, 0, 3));
        for expected in 1..=3 {
            assert_eq!(
                harness.producer.try_flush().unwrap(),
                FlushControlOutcome {
                    enqueued: 1,
                    remaining_staged: 3 - expected
                }
            );
            harness.callback();
            assert_eq!(harness.producer.applied_batch_sequence(), expected as u64);
        }
        assert_eq!(*log.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn suspended_256_envelopes_exhaust_logical_budget_before_a_257th_factory() {
        let harness = Harness::new(1, true);
        let staged: Vec<_> = (0..256)
            .map(|_| {
                harness
                    .producer
                    .try_commit(prepare_with(
                        &harness.producer,
                        vec![ControlMessage::TestNop],
                    ))
                    .unwrap()
            })
            .collect();
        assert!(staged
            .iter()
            .all(|outcome| *outcome == CommitControlOutcome::Staged));
        assert_eq!(harness.producer.accounting(), (256, 256, 0, 256));
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::LogicalCommandCredits)
        );
    }

    #[test]
    fn full_ordinary_queue_still_accepts_reserved_close_and_ack_is_scoped() {
        let mut harness = Harness::new(2, false);
        for _ in 0..2 {
            harness
                .producer
                .try_commit(prepare_with(
                    &harness.producer,
                    vec![ControlMessage::TestNop],
                ))
                .unwrap();
        }
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::OrdinaryPhysicalCredits)
        );
        let sealed = harness.finish_close();
        assert_eq!(sealed.close.barrier().required_batch_sequence(), 2);
        assert!(harness
            .producer
            .inner
            .close_in_flight
            .load(Ordering::Acquire));
        let pending = sealed.close.try_observe_exact().err().unwrap();
        assert!(matches!(pending, ObserveControlCloseFailure::Pending(_)));
        let close = pending.into_close();
        assert!(matches!(close.snapshot(), GraphLifecycleSnapshot::Pending));
        assert!(CloseLifecycleLease::acquire(&harness.producer.inner.close_in_flight).is_none());

        harness.callback();
        assert!(matches!(
            close.snapshot(),
            GraphLifecycleSnapshot::Applied {
                outcome: GraphLifecycleOutcome::Applied,
                ..
            }
        ));
        let observed = close.try_observe_exact().ok().unwrap();
        assert_eq!(observed.barrier.required_batch_sequence(), 2);
        assert!(!harness
            .producer
            .inner
            .close_in_flight
            .load(Ordering::Acquire));
    }

    #[test]
    fn held_preseal_preparation_blocks_close_drain_then_commits_ahead_of_close() {
        let mut harness = Harness::new(1, false);
        let reservation = harness.producer.try_begin_operation(1).unwrap();
        let retirement = harness
            .owner
            .take()
            .unwrap()
            .try_begin_close()
            .ok()
            .unwrap();
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::Sealed)
        );
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (done_send, done_recv) = crossbeam_channel::bounded(1);
        let waiter = thread::spawn(move || {
            entered_send.send(()).unwrap();
            done_send.send(retirement.retire_and_wait()).unwrap();
        });
        entered_recv.recv().unwrap();
        assert!(done_recv.try_recv().is_err());
        let prepared = reservation
            .prepare_with(|| vec![ControlMessage::TestNop])
            .ok()
            .unwrap();
        assert_eq!(
            harness.producer.try_commit(prepared).unwrap(),
            CommitControlOutcome::Enqueued { sequence: 1 }
        );
        let (_, drained) = done_recv.recv().unwrap();
        waiter.join().unwrap();
        let sealed = drained.finish().ok().unwrap();
        assert_eq!(sealed.close.barrier().required_batch_sequence(), 1);
        harness.callback();
    }

    #[test]
    fn close_stops_and_joins_registered_capacity_worker_before_drain() {
        let mut harness = Harness::new(1, false);
        let exited = Arc::new(AtomicBool::new(false));
        let exited_worker = Arc::clone(&exited);
        let worker = harness
            .gate
            .try_begin_capacity_worker()
            .unwrap()
            .start(move |stop| {
                let _ = stop.recv();
                exited_worker.store(true, Ordering::Release);
            })
            .unwrap();
        worker.try_commit().ok().unwrap();
        let retirement = harness
            .owner
            .take()
            .unwrap()
            .try_begin_close()
            .ok()
            .unwrap();
        let (snapshot, drained) = retirement.retire_and_wait();
        assert!(snapshot.is_drained());
        assert!(exited.load(Ordering::Acquire));
        let sealed = drained.finish().ok().unwrap();
        assert!(!sealed.degradation.capacity_worker_panicked);
        harness.callback();
    }

    struct DropThreadProbe(crossbeam_channel::Sender<ThreadId>);

    impl Drop for DropThreadProbe {
        fn drop(&mut self) {
            let _ = self.0.send(thread::current().id());
        }
    }

    #[test]
    fn suspended_extraction_drops_payload_on_explicit_non_rt_thread() {
        let mut harness = Harness::new(1, true);
        let (drop_send, drop_recv) = crossbeam_channel::bounded(1);
        let command = ControlMessage::NodeMessage {
            id: AudioNodeId(123),
            msg: llq::Node::new(Box::new(DropThreadProbe(drop_send)) as Box<dyn Any + Send>),
        };
        harness
            .producer
            .try_commit(prepare_with(&harness.producer, vec![command]))
            .unwrap();
        let sealed = harness.finish_close();
        assert_eq!(sealed.payloads.staged_len(), 1);
        assert_eq!(sealed.payloads.last_submitted_batch_sequence(), 0);
        let worker = thread::spawn(move || drop(sealed.payloads));
        let worker_id = worker.thread().id();
        worker.join().unwrap();
        assert_eq!(drop_recv.recv().unwrap(), worker_id);
        harness.callback();
    }

    #[test]
    fn transport_poison_is_recovered_for_degraded_close() {
        let mut harness = Harness::new(1, false);
        let poison = harness.producer.clone();
        assert!(thread::spawn(move || poison.poison_transport())
            .join()
            .is_err());
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::Poisoned)
        );
        let sealed = harness.finish_close();
        assert!(sealed.degradation.transport_poison_recovered);
        harness.callback();
        assert!(sealed.close.try_observe_exact().is_ok());
    }

    #[test]
    fn prior_transport_failure_survives_as_close_degradation() {
        let mut harness = Harness::new(1, false);
        harness.producer.fail_transport();
        let sealed = harness.finish_close();
        assert!(sealed.degradation.prior_transport_failure);
        harness.callback();
        assert!(sealed.close.try_observe_exact().is_ok());
    }

    #[test]
    fn gate_poison_quarantines_unique_owner_and_all_producers() {
        let mut harness = Harness::new(1, false);
        let gate = harness.gate.clone();
        assert!(thread::spawn(move || gate.poison_phase_lock_for_test())
            .join()
            .is_err());
        assert_eq!(
            harness.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::GatePoisoned)
        );
        let failure = harness
            .owner
            .take()
            .unwrap()
            .try_begin_close()
            .err()
            .unwrap();
        assert_eq!(failure.error, InjectedControlError::GatePoisoned);
        drop(failure.owner);
    }

    #[test]
    fn disconnected_receiver_returns_payload_and_terminal_error() {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, owner, init) = injected_control_channel(gate, 1, false).unwrap();
        drop(init);
        let prepared = prepare_with(&producer, vec![ControlMessage::TestNop]);
        let failure = producer.try_commit(prepared).unwrap_err();
        assert_eq!(failure.error, InjectedControlError::Disconnected);
        assert_eq!(failure.batch.len(), 1);
        assert_eq!(producer.last_submitted_batch_sequence(), 0);
        drop(owner);
    }

    #[test]
    fn renderer_death_before_suspended_flush_latches_disconnect_and_preserves_staging() {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, owner, init) = injected_control_channel(gate, 1, true).unwrap();
        drop(init);
        let (drop_send, drop_recv) = crossbeam_channel::bounded(1);
        let command = ControlMessage::TestGarbage {
            payload: llq::Node::new(Box::new(DropThreadProbe(drop_send)) as Box<dyn Any + Send>),
        };
        assert_eq!(
            producer
                .try_commit(prepare_with(&producer, vec![command]))
                .unwrap(),
            CommitControlOutcome::Staged
        );
        assert_eq!(
            producer.try_flush(),
            Err(InjectedControlError::Disconnected)
        );
        assert_eq!(
            producer.try_flush(),
            Err(InjectedControlError::Disconnected)
        );

        let retirement = owner.try_begin_close().ok().unwrap();
        let (_, drained) = retirement.retire_and_wait();
        let failure = drained.finish().err().unwrap();
        assert_eq!(failure.error, InjectedControlError::Disconnected);
        assert!(failure.degradation.prior_transport_failure);
        assert_eq!(failure.payloads.staged_len(), 1);
        let worker = thread::spawn(move || drop(failure.payloads));
        let worker_id = worker.thread().id();
        worker.join().unwrap();
        assert_eq!(drop_recv.recv().unwrap(), worker_id);
    }

    #[test]
    fn sequence_namespace_is_reserved_before_running_or_staged_factories() {
        let running = Harness::new(2, false);
        running.producer.set_next_batch_sequence(u64::MAX - 2);
        let first = running.producer.try_begin_operation(1).unwrap();
        let second = running.producer.try_begin_operation(1).unwrap();
        assert_eq!(
            running.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::SequenceExhausted)
        );
        let second = second
            .prepare_with(|| vec![ControlMessage::TestNop])
            .ok()
            .unwrap();
        assert_eq!(
            running.producer.try_commit(second).unwrap(),
            CommitControlOutcome::Enqueued {
                sequence: u64::MAX - 2
            }
        );
        let first = first
            .prepare_with(|| vec![ControlMessage::TestNop])
            .ok()
            .unwrap();
        assert_eq!(
            running.producer.try_commit(first).unwrap(),
            CommitControlOutcome::Enqueued {
                sequence: u64::MAX - 1
            }
        );

        let staged = Harness::new(1, true);
        staged.producer.set_next_batch_sequence(u64::MAX - 2);
        let first = staged.producer.try_begin_operation(1).unwrap();
        let second = staged.producer.try_begin_operation(1).unwrap();
        assert_eq!(
            staged.producer.try_begin_operation(1).err(),
            Some(InjectedControlError::SequenceExhausted)
        );
        drop((first, second));
    }

    struct AdmissionCleanupProbe {
        gate: InjectedContextAdmissionGate,
        observed: crossbeam_channel::Sender<(usize, AdmissionDrain)>,
    }

    impl Drop for AdmissionCleanupProbe {
        fn drop(&mut self) {
            let sealed = self.gate.try_seal().unwrap();
            let (worker, drain) = sealed.into_parts();
            assert!(worker.is_none());
            let inside = drain.snapshot().graph_controls;
            self.observed.send((inside, drain)).unwrap();
        }
    }

    #[test]
    fn rejected_payload_cleanup_remains_inside_graph_admission() {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, owner, init) = injected_control_channel(gate.clone(), 1, false).unwrap();
        let (observed_send, observed_recv) = crossbeam_channel::bounded(1);
        let failure = producer
            .try_begin_operation(2)
            .unwrap()
            .prepare_with(|| {
                vec![ControlMessage::TestGarbage {
                    payload: llq::Node::new(Box::new(AdmissionCleanupProbe {
                        gate,
                        observed: observed_send,
                    }) as Box<dyn Any + Send>),
                }]
            })
            .err()
            .unwrap();
        assert_eq!(failure.error, InjectedControlError::TooLarge);
        drop(failure);
        let (inside, drain) = observed_recv.recv().unwrap();
        assert_eq!(inside, 1);
        assert_eq!(drain.snapshot().graph_controls, 0);
        drop((drain, owner, init));
    }

    #[test]
    fn non_applied_close_outcomes_do_not_release_lifecycle_lease() {
        let state = Arc::new(AtomicBool::new(false));
        let lease = CloseLifecycleLease::acquire(&state).unwrap();
        let (publisher, watcher) = graph_lifecycle_ack_pair();
        let barrier = GraphLifecycleBarrier::new(
            NonZeroU64::new(1).unwrap(),
            2,
            GraphLifecycleTransition::Close,
        );
        let close = SubmittedControlClose {
            barrier,
            watcher,
            lease,
        };
        publisher.publish(barrier, 1, GraphLifecycleOutcome::RequiredBatchPending);
        let failure = close.try_observe_exact().err().unwrap();
        assert!(matches!(
            failure,
            ObserveControlCloseFailure::Terminal { .. }
        ));
        let close = failure.into_close();
        assert!(state.load(Ordering::Acquire));
        drop(close);
        assert!(state.load(Ordering::Acquire));
    }

    #[test]
    fn renderer_death_disconnects_close_wake_and_quarantines_pending_lease() {
        let mut harness = Harness::new(1, false);
        let sealed = harness.finish_close();
        let close_flag = Arc::clone(&harness.producer.inner.close_in_flight);
        drop(harness.renderer.take());
        harness.gc.take().unwrap().join().unwrap();
        assert!(sealed.close.wake_receiver().recv().is_err());
        assert!(matches!(
            sealed.close.snapshot(),
            GraphLifecycleSnapshot::Pending
        ));
        drop(sealed.close);
        assert!(close_flag.load(Ordering::Acquire));
    }

    #[test]
    fn injected_dequeue_and_reclaim_are_allocation_free_and_drop_off_rt() {
        let mut harness = Harness::new(1, false);
        let (drop_send, drop_recv) = crossbeam_channel::bounded(1);
        let command = ControlMessage::TestGarbage {
            payload: llq::Node::new(Box::new(DropThreadProbe(drop_send)) as Box<dyn Any + Send>),
        };
        harness
            .producer
            .try_commit(prepare_with(&harness.producer, vec![command]))
            .unwrap();
        let gc_id = harness.gc.as_ref().unwrap().thread().id();
        alloc_counter::deny_alloc(|| harness.callback());
        assert_eq!(drop_recv.recv().unwrap(), gc_id);
        assert_eq!(harness.producer.applied_batch_sequence(), 1);
    }

    #[test]
    fn capability_traits_keep_lifecycle_nonclone_and_sender_opaque() {
        fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
        fn assert_send<T: Send>() {}
        assert_clone_send_sync::<InjectedControlProducer>();
        assert_send::<InjectedControlLifecycleOwner>();
        assert_send::<InjectedControlRenderInit>();
        assert_send::<ControlBatchReservation>();
        assert_send::<PreparedControlBatch>();
        assert_send::<SubmittedControlClose>();
    }

    #[test]
    fn staging_credit_release_emits_coalesced_activity_hint() {
        let (activity, receiver) = crossbeam_channel::bounded(1);
        let pool = StagingSlotPool::new(activity);
        let slot = pool.try_acquire().unwrap();
        assert_eq!(pool.in_flight(), 1);
        drop(slot);
        receiver.recv().unwrap();
        assert_eq!(pool.in_flight(), 0);
    }

    #[test]
    fn production_bound_renderer_can_only_enter_the_owner_callback_gc_pair() {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, owner, init) = injected_control_channel(gate, 1, false).unwrap();
        let (event_sender, _event_receiver) = crossbeam_channel::bounded(1);
        let (_allocator, node_ids, graph) = injected_node_id_pair(0);
        let (_registrar, node_lifetimes) = injected_node_lifetime_registry(
            DEFAULT_NODE_LIFETIME_CAPACITY,
            &producer,
            node_ids,
            graph,
        )
        .ok()
        .unwrap();
        let bound = init
            .build_render_thread(
                node_lifetimes,
                48_000.,
                2,
                Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
                Arc::new(AtomicU64::new(0)),
                AudioStats::new(),
                event_sender,
            )
            .ok()
            .unwrap();
        assert!(bound.renderer.has_injected_reclaim_publisher());
        let (events, _watcher) = AudioOutputEventSink::bounded(1);
        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (render_owner, callback, node_lifetimes) =
            bound.into_audio_render_thread_pair(format, events);
        render_owner.begin_shutdown();
        drop(callback);
        assert!(render_owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .ok()
            .unwrap()
            .is_ok());
        drop((producer, owner, node_lifetimes));
    }
}
