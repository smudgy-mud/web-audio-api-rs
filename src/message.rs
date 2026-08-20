//! Message passing from control to render node

use std::any::Any;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "diagnostics")]
use crate::context::AudioBackendDiagnostics;
use crate::context::{AudioControlBatchReservation, AudioNodeId};
use crate::node::{ChannelConfigInner, ChannelCountMode, ChannelInterpretation};
use crate::render::graph::Graph;
use crate::render::AudioProcessor;

#[allow(dead_code)] // Private staging for the later additive bounded-control API.
pub(crate) const CONTROL_BATCH_CAPACITY: usize = 256;
/// Maximum number of allocated batch-storage envelopes, including the GC backlog.
///
/// Each envelope independently holds at most [`CONTROL_BATCH_CAPACITY`] commands, so this is not
/// a total-command credit limit. The resulting private staging bound is 256 envelopes / 65,536
/// command slots. Heap allocations reachable from individual command payloads are outside this
/// storage-accounting bound.
#[allow(dead_code)] // Private staging for the later additive bounded-control API.
pub(crate) const CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT: usize = 256;
pub(crate) const CONTROL_COMMANDS_PER_CALLBACK: usize = 256;
const _: () = assert!(CONTROL_BATCH_CAPACITY <= CONTROL_COMMANDS_PER_CALLBACK);

pub(crate) type ControlBatchNode = llq::Node<Box<dyn Any + Send>>;

#[derive(Debug, Default)]
struct ControlBatchApplyState {
    sequence: AtomicU64,
    render_protocol_failed: AtomicU8,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ControlBatchApplied {
    inner: Arc<ControlBatchApplyState>,
}

impl ControlBatchApplied {
    pub(crate) fn load(&self) -> u64 {
        self.inner.sequence.load(Ordering::Acquire)
    }

    pub(super) fn publish(&self, sequence: u64) {
        self.inner.sequence.store(sequence, Ordering::Release);
    }

    /// Latches a renderer-side structural failure without claiming the offending envelope's
    /// sequence as applied. The callback subsequently unwinds and exact shutdown reclaims the
    /// still-owned envelope off the render thread.
    pub(super) fn fail_render_protocol(&self) {
        self.inner
            .render_protocol_failed
            .store(1, Ordering::Release);
    }

    pub(crate) fn render_protocol_failed(&self) -> bool {
        self.inner.render_protocol_failed.load(Ordering::Acquire) != 0
    }
}

/// Silent graph transition committed by an injected lifecycle barrier.
#[allow(dead_code)] // Installed by the later injected-context integration; exercised in tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum GraphLifecycleTransition {
    Suspend = 1,
    Resume = 2,
    Close = 3,
}

#[allow(dead_code)]
impl GraphLifecycleTransition {
    const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Suspend),
            2 => Some(Self::Resume),
            3 => Some(Self::Close),
            _ => None,
        }
    }
}

/// Direct, non-batchable lifecycle fence in the controller sequence namespace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GraphLifecycleBarrier {
    controller_sequence: u64,
    required_batch_sequence: u64,
    transition: GraphLifecycleTransition,
}

#[allow(dead_code)]
impl GraphLifecycleBarrier {
    pub(crate) const fn new(
        controller_sequence: NonZeroU64,
        required_batch_sequence: u64,
        transition: GraphLifecycleTransition,
    ) -> Self {
        Self {
            controller_sequence: controller_sequence.get(),
            required_batch_sequence,
            transition,
        }
    }

    pub(crate) const fn controller_sequence(self) -> u64 {
        self.controller_sequence
    }

    pub(crate) const fn required_batch_sequence(self) -> u64 {
        self.required_batch_sequence
    }

    pub(crate) const fn transition(self) -> GraphLifecycleTransition {
        self.transition
    }

    #[cfg(test)]
    pub(crate) const fn from_raw_parts_for_test(
        controller_sequence: u64,
        required_batch_sequence: u64,
        transition: GraphLifecycleTransition,
    ) -> Self {
        Self {
            controller_sequence,
            required_batch_sequence,
            transition,
        }
    }
}

/// Fixed result published for one controller lifecycle sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum GraphLifecycleOutcome {
    Applied = 1,
    ControllerSequenceGap = 2,
    RequiredBatchPending = 3,
    ProtocolViolation = 4,
    EventDeliveryFailed = 5,
}

#[allow(dead_code)]
impl GraphLifecycleOutcome {
    const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Applied),
            2 => Some(Self::ControllerSequenceGap),
            3 => Some(Self::RequiredBatchPending),
            4 => Some(Self::ProtocolViolation),
            5 => Some(Self::EventDeliveryFailed),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct GraphLifecycleAckState {
    /// Release-published commit word; every other field belongs to this sequence.
    applied_sequence: AtomicU64,
    required_batch_sequence: AtomicU64,
    observed_batch_sequence: AtomicU64,
    transition: AtomicU8,
    outcome: AtomicU8,
}

/// Render-owned publisher for the persistent lifecycle acknowledgement slot.
pub(crate) struct GraphLifecyclePublisher {
    state: Arc<GraphLifecycleAckState>,
    wake: crossbeam_channel::Sender<()>,
}

impl GraphLifecyclePublisher {
    /// Publishes fixed-size data before the Release commit word, then emits a best-effort wake.
    ///
    /// The bounded `try_send` never waits for channel capacity. It is deliberately not described
    /// as lock-free because the channel implementation may briefly use internal synchronization.
    pub(crate) fn publish(
        &self,
        barrier: GraphLifecycleBarrier,
        observed_batch_sequence: u64,
        outcome: GraphLifecycleOutcome,
    ) {
        self.state
            .required_batch_sequence
            .store(barrier.required_batch_sequence(), Ordering::Relaxed);
        self.state
            .observed_batch_sequence
            .store(observed_batch_sequence, Ordering::Relaxed);
        self.state
            .transition
            .store(barrier.transition() as u8, Ordering::Relaxed);
        self.state.outcome.store(outcome as u8, Ordering::Relaxed);
        self.state
            .applied_sequence
            .store(barrier.controller_sequence(), Ordering::Release);
        let _ = self.wake.try_send(());
    }
}

/// Acquire snapshot for one expected in-flight lifecycle barrier.
#[allow(dead_code)] // Read by the later injected lifecycle worker; exercised in tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLifecycleSnapshot {
    Pending,
    Applied {
        barrier: GraphLifecycleBarrier,
        observed_batch_sequence: u64,
        outcome: GraphLifecycleOutcome,
    },
    SequenceAdvanced {
        applied_sequence: u64,
    },
}

/// Control-owned watcher for the persistent lifecycle acknowledgement slot.
///
/// The controller must keep exactly one lifecycle barrier in flight: it must observe that
/// barrier's authoritative snapshot before submitting the next sequence. This lets the single
/// fixed-size slot remain allocation-free while preserving a coherent Release/Acquire snapshot.
#[allow(dead_code)] // Owned by the later injected lifecycle worker; exercised in tests.
pub(crate) struct GraphLifecycleWatcher {
    state: Arc<GraphLifecycleAckState>,
    receiver: crossbeam_channel::Receiver<()>,
}

#[allow(dead_code)]
impl GraphLifecycleWatcher {
    pub(crate) fn snapshot(&self, expected_sequence: NonZeroU64) -> GraphLifecycleSnapshot {
        let expected_sequence = expected_sequence.get();
        let applied_sequence = self.state.applied_sequence.load(Ordering::Acquire);
        if applied_sequence < expected_sequence {
            return GraphLifecycleSnapshot::Pending;
        }
        if applied_sequence > expected_sequence {
            return GraphLifecycleSnapshot::SequenceAdvanced { applied_sequence };
        }

        let Some(transition) =
            GraphLifecycleTransition::from_u8(self.state.transition.load(Ordering::Relaxed))
        else {
            return GraphLifecycleSnapshot::SequenceAdvanced { applied_sequence };
        };
        let Some(outcome) =
            GraphLifecycleOutcome::from_u8(self.state.outcome.load(Ordering::Relaxed))
        else {
            return GraphLifecycleSnapshot::SequenceAdvanced { applied_sequence };
        };
        GraphLifecycleSnapshot::Applied {
            barrier: GraphLifecycleBarrier {
                controller_sequence: applied_sequence,
                required_batch_sequence: self.state.required_batch_sequence.load(Ordering::Relaxed),
                transition,
            },
            observed_batch_sequence: self.state.observed_batch_sequence.load(Ordering::Relaxed),
            outcome,
        }
    }

    /// Best-effort wake receiver; callers must always re-read the authoritative snapshot.
    pub(crate) const fn receiver(&self) -> &crossbeam_channel::Receiver<()> {
        &self.receiver
    }

    #[cfg(test)]
    pub(crate) fn disconnect_wake_for_test(self) -> Self {
        let Self { state, receiver } = self;
        drop(receiver);
        Self {
            state,
            receiver: crossbeam_channel::never(),
        }
    }
}

#[allow(dead_code)] // Created by the later injected lifecycle setup; exercised in tests.
pub(crate) fn graph_lifecycle_ack_pair() -> (GraphLifecyclePublisher, GraphLifecycleWatcher) {
    let state = Arc::new(GraphLifecycleAckState::default());
    let (wake, receiver) = crossbeam_channel::bounded(1);
    (
        GraphLifecyclePublisher {
            state: Arc::clone(&state),
            wake,
        },
        GraphLifecycleWatcher { state, receiver },
    )
}

#[allow(dead_code)] // Constructed by the later additive bounded-control API; exercised in tests.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ControlBatchSendError {
    Empty,
    TooLarge,
    NestedBatch,
    UnsupportedCommand,
    BatchStorageLimit,
    QueueFull,
    Disconnected,
    Contended,
    Poisoned,
    SequenceExhausted,
}

#[derive(Debug)]
struct ControlBatchStorageInFlight {
    count: AtomicUsize,
    activity: Option<crossbeam_channel::Sender<()>>,
}

impl Default for ControlBatchStorageInFlight {
    fn default() -> Self {
        Self {
            count: AtomicUsize::new(0),
            activity: None,
        }
    }
}

/// Clone-shared, atomically accounted logical command budget for injected control batches.
#[derive(Clone, Debug)]
pub(crate) struct InjectedCommandCreditPool {
    inner: Arc<InjectedCommandCreditState>,
    limit: usize,
}

#[derive(Debug)]
struct InjectedCommandCreditState {
    in_flight: AtomicUsize,
    activity: crossbeam_channel::Sender<()>,
}

impl InjectedCommandCreditPool {
    pub(crate) fn new(limit: usize, activity: crossbeam_channel::Sender<()>) -> Self {
        Self {
            inner: Arc::new(InjectedCommandCreditState {
                in_flight: AtomicUsize::new(0),
                activity,
            }),
            limit,
        }
    }

    pub(crate) fn try_acquire(&self, count: usize) -> Option<InjectedCommandCredit> {
        self.inner
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(count)
                    .filter(|next| *next <= self.limit)
            })
            .ok()
            .map(|_| InjectedCommandCredit {
                inner: Arc::clone(&self.inner),
                count,
                host_reservation: None,
            })
    }

    #[allow(dead_code)] // Observable by the private injected transport tests.
    pub(crate) fn in_flight(&self) -> usize {
        self.inner.in_flight.load(Ordering::Acquire)
    }
}

/// Logical command ownership. For an enqueued batch this remains inside batch storage until the
/// storage is reclaimed off RT; it is deliberately unrelated to physical channel capacity.
#[derive(Debug)]
pub(crate) struct InjectedCommandCredit {
    inner: Arc<InjectedCommandCreditState>,
    count: usize,
    /// Dropped with the logical command credit. Accepted credits live in batch storage through
    /// off-render-thread reclamation; rejected credits remain under graph admission through
    /// typed rollback.
    host_reservation: Option<AudioControlBatchReservation>,
}

impl InjectedCommandCredit {
    pub(crate) fn attach_host_reservation(
        &mut self,
        reservation: AudioControlBatchReservation,
    ) -> bool {
        if self.host_reservation.is_some() {
            return false;
        }
        self.host_reservation = Some(reservation);
        true
    }
}

impl Drop for InjectedCommandCredit {
    fn drop(&mut self) {
        let previous = self.inner.in_flight.fetch_sub(self.count, Ordering::AcqRel);
        debug_assert!(previous >= self.count);
        let _ = self.inner.activity.try_send(());
    }
}

/// Atomic credit pool for injected physical channel records.
///
/// The render thread retains a pool owner while callbacks can dequeue tokens. Therefore token
/// Drop performs one atomic decrement plus a non-waiting best-effort bounded wake, and cannot
/// destroy the final allocation on RT. The channel wake is not claimed to be lock-free.
#[derive(Clone, Debug)]
pub(crate) struct InjectedPhysicalCreditPool {
    inner: Arc<InjectedPhysicalCreditState>,
    limit: usize,
}

#[derive(Debug)]
struct InjectedPhysicalCreditState {
    in_flight: AtomicUsize,
    activity: crossbeam_channel::Sender<()>,
}

impl InjectedPhysicalCreditPool {
    pub(crate) fn new(limit: usize, activity: crossbeam_channel::Sender<()>) -> Self {
        Self {
            inner: Arc::new(InjectedPhysicalCreditState {
                in_flight: AtomicUsize::new(0),
                activity,
            }),
            limit,
        }
    }

    pub(crate) fn try_acquire(&self) -> Option<InjectedPhysicalCredit> {
        self.inner
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.limit).then_some(current + 1)
            })
            .ok()
            .map(|_| InjectedPhysicalCredit {
                inner: Arc::clone(&self.inner),
            })
    }

    #[allow(dead_code)] // Observable by the private injected transport tests.
    pub(crate) fn in_flight(&self) -> usize {
        self.inner.in_flight.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub(crate) struct InjectedPhysicalCredit {
    inner: Arc<InjectedPhysicalCreditState>,
}

impl Drop for InjectedPhysicalCredit {
    fn drop(&mut self) {
        let previous = self.inner.in_flight.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        // A best-effort, capacity-one hint. The render-side pool owner guarantees this token does
        // not destroy the final allocation on RT; the authoritative credit count is atomic.
        let _ = self.inner.activity.try_send(());
    }
}

/// Render-side owners that make physical-token Drop allocation-free even after every control-side
/// sender clone has been sealed and destroyed.
pub(crate) struct InjectedPhysicalCreditOwners {
    #[allow(dead_code)] // Lifetime owner; deliberately retained rather than read.
    pub(crate) ordinary: InjectedPhysicalCreditPool,
    #[allow(dead_code)] // Lifetime owner; deliberately retained rather than read.
    pub(crate) lifecycle: InjectedPhysicalCreditPool,
}

impl ControlBatchStorageInFlight {
    #[allow(dead_code)]
    fn try_acquire(self: &Arc<Self>) -> Option<ControlBatchPermit> {
        self.count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT).then_some(count + 1)
            })
            .ok()
            .map(|_| ControlBatchPermit(Arc::clone(self)))
    }

    fn load(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub(crate) struct ControlBatchPermit(Arc<ControlBatchStorageInFlight>);

impl Drop for ControlBatchPermit {
    fn drop(&mut self) {
        let previous = self.0.count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if let Some(activity) = &self.0.activity {
            let _ = activity.try_send(());
        }
    }
}

pub(crate) struct ControlBatchStorage {
    sequence: u64,
    /// The first command not yet taken by the renderer.
    ///
    /// Render-side budget admission checks the full remaining length before advancing this field,
    /// so a batch is never partially consumed merely because one callback exhausted its command
    /// budget.
    next: usize,
    commands: Box<[Option<ControlMessage>]>,
    _permit: Option<ControlBatchPermit>,
    _injected_command_credit: Option<InjectedCommandCredit>,
    #[cfg(test)]
    reclaim_probe: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl ControlBatchStorage {
    #[allow(dead_code)]
    fn new(
        sequence: u64,
        commands: Vec<ControlMessage>,
        permit: ControlBatchPermit,
        injected_command_credit: Option<InjectedCommandCredit>,
        #[cfg(test)] reclaim_probe: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self {
            sequence,
            next: 0,
            commands: commands.into_iter().map(Some).collect(),
            _permit: Some(permit),
            _injected_command_credit: injected_command_credit,
            #[cfg(test)]
            reclaim_probe,
        }
    }

    pub(crate) fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Assigns the authoritative sequence immediately before a staged injected envelope is sent.
    /// The storage must still be untouched and unsequenced.
    pub(crate) fn assign_sequence_before_enqueue(&mut self, sequence: u64) -> bool {
        if sequence == 0 || self.sequence != 0 || self.next != 0 {
            return false;
        }
        self.sequence = sequence;
        true
    }

    /// Clears a sequence after a nonblocking channel send returns the untouched envelope.
    pub(crate) fn clear_unsubmitted_sequence(&mut self, sequence: u64) -> bool {
        if self.sequence != sequence || self.next != 0 {
            return false;
        }
        self.sequence = 0;
        true
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.next == self.commands.len()
    }

    pub(crate) fn remaining_len(&self) -> usize {
        self.commands
            .len()
            .checked_sub(self.next)
            .expect("control batch cursor never exceeds its storage")
    }

    pub(crate) fn classify_injected_explicit_batch(&self) -> InjectedExplicitBatchShape {
        if self.next != 0 || self.commands.is_empty() {
            return InjectedExplicitBatchShape::Ordinary;
        }

        let mut connect = None;
        let mut disconnect_count = 0;
        let mut saw_explicit = false;
        let mut saw_ordinary = false;
        for command in &self.commands {
            match command.as_ref() {
                Some(ControlMessage::InjectedConnectExplicit(value)) => {
                    saw_explicit = true;
                    if connect.replace(*value).is_some() {
                        return InjectedExplicitBatchShape::Invalid;
                    }
                }
                Some(ControlMessage::InjectedDisconnectExplicit(_)) => {
                    saw_explicit = true;
                    disconnect_count += 1;
                }
                Some(_) => saw_ordinary = true,
                None => return InjectedExplicitBatchShape::Invalid,
            }
        }

        if !saw_explicit {
            InjectedExplicitBatchShape::Ordinary
        } else if saw_ordinary || (connect.is_some() && disconnect_count != 0) {
            InjectedExplicitBatchShape::Invalid
        } else if let Some(connect) = connect {
            if self.commands.len() == 1 {
                InjectedExplicitBatchShape::Connect(connect)
            } else {
                InjectedExplicitBatchShape::Invalid
            }
        } else if disconnect_count == self.commands.len() {
            InjectedExplicitBatchShape::Disconnect
        } else {
            InjectedExplicitBatchShape::Invalid
        }
    }

    pub(crate) fn injected_explicit_disconnects(
        &self,
    ) -> impl Iterator<Item = crate::context::InjectedExplicitDisconnect> + Clone + '_ {
        self.commands.iter().map(|command| match command.as_ref() {
            Some(ControlMessage::InjectedDisconnectExplicit(value)) => *value,
            _ => unreachable!("caller prevalidated an all-disconnect exact envelope"),
        })
    }

    pub(crate) fn take_next(&mut self) -> Option<ControlMessage> {
        let command = self.commands.get_mut(self.next)?.take();
        self.next += 1;
        command
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedExplicitBatchShape {
    Ordinary,
    Connect(crate::context::InjectedExplicitConnect),
    Disconnect,
    Invalid,
}

/// Clone-shared envelope-storage budget. This is distinct from logical command and physical
/// channel credits and remains held through the off-RT GC backlog.
#[derive(Clone, Debug)]
pub(crate) struct ControlBatchStoragePool {
    inner: Arc<ControlBatchStorageInFlight>,
}

impl ControlBatchStoragePool {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(ControlBatchStorageInFlight::default()),
        }
    }

    pub(crate) fn new_with_activity(activity: crossbeam_channel::Sender<()>) -> Self {
        Self {
            inner: Arc::new(ControlBatchStorageInFlight {
                count: AtomicUsize::new(0),
                activity: Some(activity),
            }),
        }
    }

    pub(crate) fn try_acquire(&self) -> Option<ControlBatchPermit> {
        self.inner.try_acquire()
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.inner.load()
    }
}

pub(crate) fn injected_control_batch_node(
    commands: Box<[ControlMessage]>,
    storage: ControlBatchPermit,
    command_credit: InjectedCommandCredit,
) -> ControlBatchNode {
    let storage = ControlBatchStorage::new(
        0,
        commands.into_vec(),
        storage,
        Some(command_credit),
        #[cfg(test)]
        None,
    );
    llq::Node::new(Box::new(storage) as Box<dyn Any + Send>)
}

/// Recovers an injected envelope that was never accepted by the physical channel.
///
/// This runs only on the control thread. It may allocate while rebuilding exact-capacity prepared
/// storage; both logical and batch-storage credits are returned to their original owners.
pub(crate) fn recover_unsubmitted_injected_batch(
    node: ControlBatchNode,
) -> (
    Box<[ControlMessage]>,
    ControlBatchPermit,
    InjectedCommandCredit,
) {
    let value = llq::Node::into_inner(node);
    let mut storage = value
        .downcast::<ControlBatchStorage>()
        .expect("injected batch node contains private storage");
    assert_eq!(storage.next, 0, "submitted batch was never consumed");
    let commands = storage
        .commands
        .iter_mut()
        .map(|command| command.take().expect("unsubmitted command remains present"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let command_credit = storage
        ._injected_command_credit
        .take()
        .expect("injected batch retains its command credit");
    let permit = storage
        ._permit
        .take()
        .expect("injected batch retains its storage credit");
    drop(storage);
    (commands, permit, command_credit)
}

impl ControlMessage {
    /// Whether this command belongs to the user/graph mutation subset that a future bounded
    /// control API may batch. Lifecycle, diagnostics, and maintenance commands remain legacy-only
    /// so an internal misuse cannot terminate a batch midway and permanently gap its sequence.
    pub(crate) fn is_batchable(&self) -> bool {
        match self {
            Self::RegisterNode { .. }
            | Self::ConnectNode {
                input: usize::MAX, ..
            }
            | Self::MarkCycleBreaker { .. }
            | Self::AudioParamInitialValue { .. }
            | Self::NodeMessage { .. }
            | Self::SetChannelCount { .. }
            | Self::SetChannelCountMode { .. }
            | Self::SetChannelInterpretation { .. } => true,
            #[cfg(test)]
            Self::TestMarker { .. } | Self::TestGarbage { .. } | Self::TestNop => true,
            _ => false,
        }
    }
}

#[cfg(test)]
impl Drop for ControlBatchStorage {
    fn drop(&mut self) {
        if let Some(probe) = self.reclaim_probe.take() {
            probe();
        }
    }
}

pub(crate) fn control_batch_storage(node: &ControlBatchNode) -> &ControlBatchStorage {
    node.as_ref()
        .downcast_ref::<ControlBatchStorage>()
        .expect("control batch node contains the private batch storage")
}

pub(crate) fn control_batch_storage_mut(node: &mut ControlBatchNode) -> &mut ControlBatchStorage {
    node.as_mut()
        .downcast_mut::<ControlBatchStorage>()
        .expect("control batch node contains the private batch storage")
}

struct ControlBatchSubmission {
    #[allow(dead_code)]
    sender: crossbeam_channel::Sender<ControlMessage>,
    #[allow(dead_code)]
    next_sequence: u64,
}

#[derive(Clone)]
// Intentionally private: a later public bounded-control slice must pair this sender with a
// renderer-death/poison receipt. A render panic currently cannot publish a terminal sequence, so
// exposing submission now could leave callers waiting on an applied watermark forever.
pub(crate) struct ControlBatchSender {
    #[allow(dead_code)]
    submission: Arc<Mutex<ControlBatchSubmission>>,
    batch_storage: ControlBatchStoragePool,
}

impl std::fmt::Debug for ControlBatchSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlBatchSender")
            .field("batch_storage_in_flight", &self.batch_storage_in_flight())
            .finish_non_exhaustive()
    }
}

impl ControlBatchSender {
    pub(crate) fn new(sender: crossbeam_channel::Sender<ControlMessage>) -> Self {
        Self {
            submission: Arc::new(Mutex::new(ControlBatchSubmission {
                sender,
                next_sequence: 1,
            })),
            batch_storage: ControlBatchStoragePool::new(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn try_send(
        &self,
        commands: Vec<ControlMessage>,
    ) -> Result<u64, ControlBatchSendError> {
        self.try_send_inner(commands, None, None, false)
    }

    pub(crate) fn batch_storage_in_flight(&self) -> usize {
        self.batch_storage.in_flight()
    }

    #[allow(dead_code)]
    fn try_send_inner(
        &self,
        commands: Vec<ControlMessage>,
        forced_sequence: Option<u64>,
        #[allow(unused_variables)] reclaim_probe: Option<Arc<dyn Fn() + Send + Sync>>,
        allow_unbatchable_for_test: bool,
    ) -> Result<u64, ControlBatchSendError> {
        if commands.is_empty() {
            return Err(ControlBatchSendError::Empty);
        }
        if commands.len() > CONTROL_BATCH_CAPACITY {
            return Err(ControlBatchSendError::TooLarge);
        }
        if commands
            .iter()
            .any(|command| matches!(command, ControlMessage::Batch(_)))
        {
            return Err(ControlBatchSendError::NestedBatch);
        }
        if !allow_unbatchable_for_test && commands.iter().any(|command| !command.is_batchable()) {
            return Err(ControlBatchSendError::UnsupportedCommand);
        }

        let mut submission = match self.submission.try_lock() {
            Ok(submission) => submission,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(ControlBatchSendError::Contended)
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(ControlBatchSendError::Poisoned)
            }
        };
        let permit = self
            .batch_storage
            .try_acquire()
            .ok_or(ControlBatchSendError::BatchStorageLimit)?;
        let sequence = forced_sequence.unwrap_or(submission.next_sequence);
        if forced_sequence.is_none() && sequence == u64::MAX {
            return Err(ControlBatchSendError::SequenceExhausted);
        }

        let storage = ControlBatchStorage::new(
            sequence,
            commands,
            permit,
            None,
            #[cfg(test)]
            reclaim_probe,
        );
        let node = llq::Node::new(Box::new(storage) as Box<dyn Any + Send>);

        match submission.sender.try_send(ControlMessage::Batch(node)) {
            Ok(()) => {
                if forced_sequence.is_none() {
                    submission.next_sequence += 1;
                }
                Ok(sequence)
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => Err(ControlBatchSendError::QueueFull),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(ControlBatchSendError::Disconnected)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn try_send_with_sequence(
        &self,
        sequence: u64,
        commands: Vec<ControlMessage>,
    ) -> Result<u64, ControlBatchSendError> {
        self.try_send_inner(commands, Some(sequence), None, false)
    }

    #[cfg(test)]
    pub(crate) fn try_send_with_reclaim_probe(
        &self,
        commands: Vec<ControlMessage>,
        reclaim_probe: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<u64, ControlBatchSendError> {
        self.try_send_inner(commands, None, Some(reclaim_probe), false)
    }

    #[cfg(test)]
    pub(crate) fn try_send_exact_for_test(
        &self,
        commands: Vec<ControlMessage>,
        reclaim_probe: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Result<u64, ControlBatchSendError> {
        self.try_send_inner(commands, None, reclaim_probe, true)
    }
}

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    /// Private, bounded multi-command envelope.
    ///
    /// Every currently admitted batch is atomic with respect to the per-callback command budget:
    /// the renderer defers the whole envelope when its remaining budget cannot fit every command.
    /// The render thread never drops its storage.
    #[allow(dead_code)]
    Batch(ControlBatchNode),

    /// Injected ordinary envelope. The physical token is released immediately when the renderer
    /// dequeues this record; logical-command and storage credits remain inside `batch` through GC.
    #[allow(dead_code)]
    InjectedBatch {
        batch: ControlBatchNode,
        physical: InjectedPhysicalCredit,
    },

    /// Private injected-only lifecycle fence. It is deliberately never batchable.
    #[allow(dead_code)]
    GraphLifecycleBarrier(GraphLifecycleBarrier),

    /// Injected lifecycle fence using the channel's independently reserved physical slot.
    #[allow(dead_code)]
    InjectedGraphLifecycleBarrier {
        barrier: GraphLifecycleBarrier,
        physical: InjectedPhysicalCredit,
    },

    /// Register a new node in the audio graph
    RegisterNode {
        id: AudioNodeId,
        reclaim_id: llq::Node<AudioNodeId>,
        node: Box<dyn AudioProcessor>,
        inputs: usize,
        outputs: usize,
        channel_config: ChannelConfigInner,
    },

    /// Connect a node to another in the audio graph
    ConnectNode {
        from: AudioNodeId,
        to: AudioNodeId,
        input: usize,
        output: usize,
    },

    /// Clear the connection between two given nodes in the audio graph
    DisconnectNode {
        from: AudioNodeId,
        to: AudioNodeId,
        input: usize,
        output: usize,
    },

    /// Private fixed explicit-edge insertion. No production constructor exists until B5b-h's
    /// dedicated admitted host transaction.
    #[allow(dead_code)] // production constructor is added by the B5b-h host transaction
    InjectedConnectExplicit(crate::context::InjectedExplicitConnect),

    /// One member of a private atomic exact-edge removal envelope. The renderer preflights every
    /// member before removing any edge.
    #[allow(dead_code)] // production constructor is added by the B5b-h host transaction
    InjectedDisconnectExplicit(crate::context::InjectedExplicitDisconnect),

    /// Notify the render thread this node is dropped in the control thread
    ControlHandleDropped { id: AudioNodeId },

    /// Mark node as a cycle breaker (DelayNode only)
    MarkCycleBreaker { id: AudioNodeId },

    /// Shut down and recycle the audio graph
    CloseAndRecycle {
        sender: crossbeam_channel::Sender<Graph>,
    },

    /// Start rendering with given audio graph
    Startup { graph: Graph },

    /// Suspend and pause audio processing
    Suspend { notify: OneshotNotify },

    /// Resume audio processing after suspending
    Resume { notify: OneshotNotify },

    /// Stop audio processing
    Close { notify: OneshotNotify },

    /// Generic message to be handled by AudioProcessor
    NodeMessage {
        id: AudioNodeId,
        msg: llq::Node<Box<dyn Any + Send>>,
    },

    /// Closed, fixed-size initial AudioParam update for injected construction.
    #[allow(dead_code)] // constructed by the pending injected Gain transaction
    AudioParamInitialValue {
        id: AudioNodeId,
        value: crate::param::AudioParamInitialValue,
    },

    /// Closed, fixed-size post-construction value update for an exact injected AudioParam.
    ///
    /// This is deliberately excluded from the general injected batch whitelist. Only its typed
    /// one-command reservation can construct it.
    InjectedAudioParamValue {
        id: AudioNodeId,
        value: crate::param::InjectedAudioParamValue,
    },

    /// Fixed, non-general exact oscillator start/stop/type command.
    InjectedOscillator(crate::context::InjectedOscillatorWireCommand),

    /// Owned, preboxed exact oscillator PeriodicWave command. The renderer always moves the inner
    /// node to off-thread GC after routing or protocol failure.
    InjectedOscillatorPeriodicWave(crate::context::InjectedOscillatorPeriodicWaveWireCommand),

    /// Fixed, non-general exact ConstantSource start/stop command.
    InjectedConstantSource(crate::context::InjectedConstantSourceWireCommand),

    /// Fixed, non-general exact BufferSource start/stop/loop command.
    InjectedAudioBufferSourceScalar(crate::context::InjectedAudioBufferSourceScalarWireCommand),

    /// Owned, preboxed exact BufferSource buffer command. The renderer always moves the inner
    /// node to off-thread GC after routing or protocol failure.
    InjectedAudioBufferSourceBuffer(crate::context::InjectedAudioBufferSourceBufferWireCommand),

    /// Request a diagnostic report of the audio graph
    #[cfg(feature = "diagnostics")]
    RunDiagnostics { backend: AudioBackendDiagnostics },

    /// Update the channel count of a node
    SetChannelCount { id: AudioNodeId, count: usize },

    /// Update the channel count mode of a node
    SetChannelCountMode {
        id: AudioNodeId,
        mode: ChannelCountMode,
    },

    /// Update the channel interpretation of a node
    SetChannelInterpretation {
        id: AudioNodeId,
        interpretation: ChannelInterpretation,
    },

    #[cfg(test)]
    TestMarker {
        value: u16,
        log: Arc<Mutex<Vec<u16>>>,
    },

    #[cfg(test)]
    TestGarbage {
        payload: llq::Node<Box<dyn Any + Send>>,
    },

    #[cfg(test)]
    TestNop,
}

/// Helper object to emit single notification
pub(crate) enum OneshotNotify {
    /// A synchronous oneshot sender
    Sync(crossbeam_channel::Sender<()>),
    /// An asynchronous oneshot sender
    Async(futures_channel::oneshot::Sender<()>),
}

impl OneshotNotify {
    /// Emit the notification
    pub fn send(self) {
        match self {
            Self::Sync(s) => s.send(()).ok(),
            Self::Async(s) => s.send(()).ok(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injected_credit_release_wakes_are_coalesced_and_counts_remain_authoritative() {
        let (activity, receiver) = crossbeam_channel::bounded(1);

        let commands = InjectedCommandCreditPool::new(2, activity.clone());
        let command_credit = commands.try_acquire(2).unwrap();
        drop(command_credit);
        receiver.recv().unwrap();
        assert_eq!(commands.in_flight(), 0);

        let physical = InjectedPhysicalCreditPool::new(1, activity.clone());
        let physical_credit = physical.try_acquire().unwrap();
        drop(physical_credit);
        receiver.recv().unwrap();
        assert_eq!(physical.in_flight(), 0);

        let storage = ControlBatchStoragePool::new_with_activity(activity);
        let first = storage.try_acquire().unwrap();
        let second = storage.try_acquire().unwrap();
        drop(first);
        drop(second);
        // Capacity one deliberately coalesces multiple releases. The atomic count, not wake
        // multiplicity, is authoritative.
        receiver.recv().unwrap();
        assert!(receiver.try_recv().is_err());
        assert_eq!(storage.in_flight(), 0);
    }

    #[test]
    fn physical_credit_wake_is_allocation_free_when_empty_full_or_disconnected() {
        let (empty_send, empty_recv) = crossbeam_channel::bounded(1);
        let empty = InjectedPhysicalCreditPool::new(1, empty_send);
        let token = empty.try_acquire().unwrap();
        alloc_counter::deny_alloc(|| drop(token));
        empty_recv.recv().unwrap();

        let (full_send, full_recv) = crossbeam_channel::bounded(1);
        full_send.try_send(()).unwrap();
        let full = InjectedPhysicalCreditPool::new(1, full_send);
        let token = full.try_acquire().unwrap();
        alloc_counter::deny_alloc(|| drop(token));
        full_recv.recv().unwrap();

        let (disconnected_send, disconnected_recv) = crossbeam_channel::bounded(1);
        drop(disconnected_recv);
        let disconnected = InjectedPhysicalCreditPool::new(1, disconnected_send);
        let token = disconnected.try_acquire().unwrap();
        alloc_counter::deny_alloc(|| drop(token));
    }

    #[test]
    fn injected_initial_param_command_is_closed_and_batchable() {
        assert!(!std::mem::needs_drop::<crate::param::AudioParamInitialValue>());
        assert!(ControlMessage::AudioParamInitialValue {
            id: AudioNodeId(3),
            value: crate::param::AudioParamInitialValue::new(0.5),
        }
        .is_batchable());
        assert!(!std::mem::needs_drop::<crate::param::InjectedAudioParamValue>());
        assert!(!ControlMessage::InjectedAudioParamValue {
            id: AudioNodeId(3),
            value: crate::param::InjectedAudioParamValue::new(0.5),
        }
        .is_batchable());
    }

    #[test]
    fn batch_submission_contention_is_immediate_and_does_not_consume_sequence() {
        let (channel, receiver) = crossbeam_channel::bounded(1);
        let batches = ControlBatchSender::new(channel);
        let guard = batches.submission.lock().unwrap();

        assert_eq!(
            batches.try_send(vec![ControlMessage::TestNop]),
            Err(ControlBatchSendError::Contended)
        );
        assert!(receiver.is_empty());
        assert_eq!(batches.batch_storage_in_flight(), 0);

        drop(guard);
        assert_eq!(batches.try_send(vec![ControlMessage::TestNop]), Ok(1));
    }

    #[test]
    fn poisoned_batch_submission_lock_is_a_typed_error() {
        let (channel, receiver) = crossbeam_channel::bounded(1);
        let batches = ControlBatchSender::new(channel);
        let submission = Arc::clone(&batches.submission);
        std::thread::spawn(move || {
            let _ = std::panic::catch_unwind(|| {
                let _guard = submission.lock().unwrap();
                panic!("poison submission lock");
            });
        })
        .join()
        .unwrap();

        assert_eq!(
            batches.try_send(vec![ControlMessage::TestNop]),
            Err(ControlBatchSendError::Poisoned)
        );
        assert!(receiver.is_empty());
        assert_eq!(batches.batch_storage_in_flight(), 0);
    }

    #[test]
    fn storage_credits_count_envelopes_and_bound_command_slots_exactly() {
        let (channel, receiver) = crossbeam_channel::bounded(CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT);
        let batches = ControlBatchSender::new(channel);
        for sequence in 1..=CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT {
            let full_batch = (0..CONTROL_BATCH_CAPACITY)
                .map(|_| ControlMessage::TestNop)
                .collect();
            assert_eq!(batches.try_send(full_batch), Ok(sequence as u64));
        }
        assert_eq!(
            batches.batch_storage_in_flight(),
            CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT
        );
        assert_eq!(
            batches.try_send(vec![ControlMessage::TestNop]),
            Err(ControlBatchSendError::BatchStorageLimit)
        );

        drop(receiver.try_recv().unwrap());
        assert_eq!(
            batches.batch_storage_in_flight(),
            CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT - 1
        );
        assert_eq!(
            batches.try_send(vec![ControlMessage::TestNop]),
            Ok((CONTROL_BATCH_STORAGE_IN_FLIGHT_LIMIT + 1) as u64)
        );
    }
}
