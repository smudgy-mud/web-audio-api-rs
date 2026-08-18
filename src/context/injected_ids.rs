//! Private rollback-capable node-id ownership for the injected context path.
//!
//! The cloneable allocator is weak. A unique lifecycle-side owner retains the graph's reclaim
//! consumer and the sole strong allocation state, so surviving base/node handles cannot retain
//! reclaim authority after context retirement. Returned graph nodes are not automatically made
//! available: the future teardown registry must first reconcile the matching generation, mirrors,
//! and represented-resource guards, then consume the owner's opaque pending reclaim. Fresh ids use
//! checked arithmetic and never wrap their namespace.
//!
//! The injected transport now provides an accepted-batch ordering finalizer, but deliberately
//! restricts it to `Copy` atomic work. A later concrete Gain transaction wrapper must still own
//! and non-panickingly disarm provisional ids and arm registrations; those destructor-bearing
//! guards cannot be captured by the generic ordering hook. B2 must also tolerate a registered
//! processor rendering and reclaiming before its control-side finalizer returns by representing an
//! accepted-pending state rather than treating that race as corruption.

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError, Weak};

use super::AudioNodeId;
use crate::render::graph::Graph;

pub(crate) const MAX_PROVISIONAL_NODE_IDS: usize = 11;

struct AllocationState {
    next: u64,
    available: llq::Consumer<AudioNodeId>,
}

struct InjectedNodeIdInner {
    allocation: Mutex<AllocationState>,
    available_return: Mutex<llq::Producer<AudioNodeId>>,
    lifecycle: AtomicU8,
    #[cfg(test)]
    release_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AllocatorLifecycle {
    OpenIdle = 0,
    OpenActive = 1,
    RetiredIdle = 2,
    RetiredActive = 3,
}

impl AllocatorLifecycle {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::OpenIdle,
            1 => Self::OpenActive,
            2 => Self::RetiredIdle,
            3 => Self::RetiredActive,
            _ => unreachable!("private allocator lifecycle state"),
        }
    }
}

/// Weak construction capability. An admitted operation may temporarily upgrade it, but a stale
/// base clone cannot keep allocator/reclaim queues alive after the unique owner retires.
#[derive(Clone)]
pub(crate) struct InjectedNodeIdAllocator {
    inner: Weak<InjectedNodeIdInner>,
}

/// Unique lifecycle-side reclaim owner.
pub(crate) struct InjectedNodeIdOwner {
    inner: Arc<InjectedNodeIdInner>,
    graph_reclaims: llq::Consumer<AudioNodeId>,
    pending_reclaim: Option<llq::Node<AudioNodeId>>,
    reclaim_activity: crossbeam_channel::Receiver<()>,
    // While the lifecycle owner is retained, dropping the Graph's publisher cannot destroy the
    // final channel allocation on the render thread. Lifecycle integration must retire the Graph
    // before this unique owner.
    _reclaim_activity_owner: crossbeam_channel::Sender<()>,
}

/// Opaque weak identity shared by one allocator owner, graph initializer, and every exact reclaim
/// token emitted by that graph.
#[derive(Clone)]
pub(crate) struct InjectedNodeIdIdentity(Weak<InjectedNodeIdInner>);

impl InjectedNodeIdIdentity {
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

/// Opaque, consuming graph initializer. It binds one internally matched reclaim producer and wake
/// publisher. The injected node-lifetime bootstrap consumes and validates this initializer with
/// its exact id owner before the render initializer can install it.
pub(crate) struct InjectedGraphReclaimInit {
    graph_reclaims: llq::Producer<AudioNodeId>,
    publisher: InjectedGraphReclaimPublisher,
    identity: InjectedNodeIdIdentity,
}

/// Render-side best-effort wake publisher. The authoritative record remains the exact LLQ node.
/// While its matching lifecycle owner is retained, the channel allocation is lifecycle-owned;
/// `publish` is bounded and non-waiting but the channel implementation is not claimed lock-free.
pub(crate) struct InjectedGraphReclaimPublisher {
    activity: crossbeam_channel::Sender<()>,
}

impl InjectedGraphReclaimPublisher {
    pub(crate) fn publish(&self) {
        let _ = self.activity.try_send(());
    }
}

impl InjectedGraphReclaimInit {
    pub(crate) fn into_graph(self) -> Graph {
        Graph::new_injected(self.graph_reclaims, self.publisher)
    }

    #[cfg(test)]
    pub(crate) fn push_for_test(&mut self, node: llq::Node<AudioNodeId>) {
        self.graph_reclaims.push(node);
        self.publisher.publish();
    }
}

pub(crate) fn injected_node_id_pair(
    first_id: u64,
) -> (
    InjectedNodeIdAllocator,
    InjectedNodeIdOwner,
    InjectedGraphReclaimInit,
) {
    let (graph_reclaims, graph_reclaim_consumer) = llq::Queue::new().split();
    let (reclaim_activity_owner, reclaim_activity) = crossbeam_channel::bounded(1);
    let (available_return, available) = llq::Queue::new().split();
    let inner = Arc::new(InjectedNodeIdInner {
        allocation: Mutex::new(AllocationState {
            next: first_id,
            available,
        }),
        available_return: Mutex::new(available_return),
        lifecycle: AtomicU8::new(AllocatorLifecycle::OpenIdle as u8),
        #[cfg(test)]
        release_hook: Mutex::new(None),
    });
    let identity = InjectedNodeIdIdentity(Arc::downgrade(&inner));
    (
        InjectedNodeIdAllocator {
            inner: Arc::downgrade(&inner),
        },
        InjectedNodeIdOwner {
            inner,
            graph_reclaims: graph_reclaim_consumer,
            pending_reclaim: None,
            reclaim_activity,
            _reclaim_activity_owner: reclaim_activity_owner.clone(),
        },
        InjectedGraphReclaimInit {
            graph_reclaims,
            publisher: InjectedGraphReclaimPublisher {
                activity: reclaim_activity_owner,
            },
            identity,
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProvisionalNodeIdError {
    Empty,
    TooMany,
    Contended,
    OwnerGone,
    Poisoned,
    Exhausted,
    ProtocolViolation,
}

pub(crate) struct ProvisionalNodeRestoreFailure {
    pub(crate) error: ProvisionalNodeIdError,
    pub(crate) node: llq::Node<AudioNodeId>,
}

impl std::fmt::Debug for ProvisionalNodeRestoreFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProvisionalNodeRestoreFailure")
            .field("error", &self.error)
            .field("id", &*self.node)
            .finish()
    }
}

/// Exact provisional reclaim nodes for one serialized constructor transaction.
///
/// A fresh id allocates its reclaim node only after the caller has acquired all higher-level
/// command/resource admission. Nodes may move into prepared `RegisterNode` commands, but a failed
/// commit must restore each node to its original slot before this guard is dropped. Only an
/// accepted atomic batch may call `commit` with every slot moved out.
#[must_use]
pub(crate) struct ProvisionalNodeIds {
    inner: Arc<InjectedNodeIdInner>,
    nodes: arrayvec::ArrayVec<Option<llq::Node<AudioNodeId>>, MAX_PROVISIONAL_NODE_IDS>,
    committed: AtomicBool,
    #[cfg(test)]
    force_commit_failure: bool,
}

/// Copy accepted-finalizer token. Construction proves every exact reclaim node has moved into the
/// prepared batch; committing is therefore one infallible atomic store before registration arms.
#[derive(Clone, Copy)]
pub(crate) struct ProvisionalNodeIdCommit<'a> {
    committed: &'a AtomicBool,
}

impl InjectedNodeIdAllocator {
    pub(crate) fn identity(&self) -> InjectedNodeIdIdentity {
        InjectedNodeIdIdentity(Weak::clone(&self.inner))
    }

    pub(crate) fn try_reserve(
        &self,
        count: usize,
    ) -> Result<ProvisionalNodeIds, ProvisionalNodeIdError> {
        if count == 0 {
            return Err(ProvisionalNodeIdError::Empty);
        }
        if count > MAX_PROVISIONAL_NODE_IDS {
            return Err(ProvisionalNodeIdError::TooMany);
        }
        let inner = self
            .inner
            .upgrade()
            .ok_or(ProvisionalNodeIdError::OwnerGone)?;
        match inner.lifecycle.compare_exchange(
            AllocatorLifecycle::OpenIdle as u8,
            AllocatorLifecycle::OpenActive as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(state) => {
                return Err(match AllocatorLifecycle::from_u8(state) {
                    AllocatorLifecycle::OpenActive => ProvisionalNodeIdError::Contended,
                    AllocatorLifecycle::RetiredIdle | AllocatorLifecycle::RetiredActive => {
                        ProvisionalNodeIdError::OwnerGone
                    }
                    AllocatorLifecycle::OpenIdle => unreachable!("compare_exchange expected idle"),
                });
            }
        }

        let mut reservation = ProvisionalNodeIds {
            inner: Arc::clone(&inner),
            nodes: arrayvec::ArrayVec::new(),
            committed: AtomicBool::new(false),
            #[cfg(test)]
            force_commit_failure: false,
        };
        let mut allocation = match inner.allocation.try_lock() {
            Ok(allocation) => allocation,
            Err(TryLockError::WouldBlock) => return Err(ProvisionalNodeIdError::Contended),
            Err(TryLockError::Poisoned(_)) => return Err(ProvisionalNodeIdError::Poisoned),
        };
        for _ in 0..count {
            let node = if let Some(node) = allocation.available.pop() {
                node
            } else {
                let id = allocation.next;
                allocation.next = allocation
                    .next
                    .checked_add(1)
                    .ok_or(ProvisionalNodeIdError::Exhausted)?;
                llq::Node::new(AudioNodeId(id))
            };
            reservation.nodes.push(Some(node));
        }
        drop(allocation);
        Ok(reservation)
    }
}

impl ProvisionalNodeIds {
    pub(crate) fn identity(&self) -> InjectedNodeIdIdentity {
        InjectedNodeIdIdentity(Arc::downgrade(&self.inner))
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn id(&self, index: usize) -> AudioNodeId {
        **self.nodes[index]
            .as_ref()
            .expect("provisional id node is present before payload construction")
    }

    pub(crate) fn take_reclaim_node(
        &mut self,
        index: usize,
    ) -> Result<llq::Node<AudioNodeId>, ProvisionalNodeIdError> {
        self.nodes
            .get_mut(index)
            .and_then(Option::take)
            .ok_or(ProvisionalNodeIdError::ProtocolViolation)
    }

    pub(crate) fn restore_reclaim_node(
        &mut self,
        index: usize,
        node: llq::Node<AudioNodeId>,
    ) -> Result<(), ProvisionalNodeRestoreFailure> {
        let Some(slot) = self.nodes.get_mut(index) else {
            return Err(ProvisionalNodeRestoreFailure {
                error: ProvisionalNodeIdError::ProtocolViolation,
                node,
            });
        };
        if slot.is_some() {
            return Err(ProvisionalNodeRestoreFailure {
                error: ProvisionalNodeIdError::ProtocolViolation,
                node,
            });
        }
        *slot = Some(node);
        Ok(())
    }

    pub(crate) fn commit_token(
        &self,
    ) -> Result<ProvisionalNodeIdCommit<'_>, ProvisionalNodeIdError> {
        #[cfg(test)]
        if self.force_commit_failure {
            return Err(ProvisionalNodeIdError::ProtocolViolation);
        }
        if self.nodes.iter().any(Option::is_some) {
            return Err(ProvisionalNodeIdError::ProtocolViolation);
        }
        Ok(ProvisionalNodeIdCommit {
            committed: &self.committed,
        })
    }

    #[cfg(test)]
    pub(crate) fn force_commit_failure_for_test(&mut self) {
        self.force_commit_failure = true;
    }

    pub(crate) fn commit(self) -> Result<(), ProvisionalNodeIdError> {
        self.commit_token()?.commit_accepted();
        Ok(())
    }

    /// Internal protocol failure after token movement: never publish any still-owned exact nodes
    /// back to the allocator. Field destruction may destroy them, but they cannot be reused.
    pub(crate) fn retain_unavailable(&self) {
        self.committed.store(true, Ordering::Release);
    }
}

impl ProvisionalNodeIdCommit<'_> {
    pub(crate) fn commit_accepted(self) {
        self.committed.store(true, Ordering::Release);
    }
}

impl Drop for ProvisionalNodeIds {
    fn drop(&mut self) {
        if !self.committed.load(Ordering::Acquire) {
            let mut available = self
                .inner
                .available_return
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Preserve rollback order relative to this transaction. Reconciled graph returns may
            // be interleaved ahead of or behind it by the unique lifecycle owner.
            for node in &mut self.nodes {
                if let Some(node) = node.take() {
                    available.push(node);
                }
            }
        }
        self.inner
            .lifecycle
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                Some(match AllocatorLifecycle::from_u8(state) {
                    AllocatorLifecycle::OpenActive => AllocatorLifecycle::OpenIdle as u8,
                    AllocatorLifecycle::RetiredActive => AllocatorLifecycle::RetiredIdle as u8,
                    AllocatorLifecycle::OpenIdle | AllocatorLifecycle::RetiredIdle => {
                        unreachable!("one provisional reservation owns the active state")
                    }
                })
            })
            .expect("active reservation lifecycle transition cannot fail");
        #[cfg(test)]
        if let Some((entered, release)) = self
            .inner
            .release_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            entered.send(()).unwrap();
            release.recv().unwrap();
        }
    }
}

impl InjectedNodeIdOwner {
    pub(crate) fn identity(&self) -> InjectedNodeIdIdentity {
        InjectedNodeIdIdentity(Arc::downgrade(&self.inner))
    }

    #[cfg(test)]
    pub(crate) fn disconnect_reclaim_activity_for_test(&mut self) {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        drop(sender);
        self.reclaim_activity = receiver;
    }

    pub(crate) fn matches_graph_init(&self, graph: &InjectedGraphReclaimInit) -> bool {
        self.identity().ptr_eq(&graph.identity)
    }

    /// Borrowed, lossy wake hint. The future lifecycle driver must always drain the authoritative
    /// exact-node queue through `try_pending_reclaim`; a wake is never an acknowledgement.
    pub(crate) const fn reclaim_activity_receiver(&self) -> &crossbeam_channel::Receiver<()> {
        &self.reclaim_activity
    }

    /// Borrows one exact graph acknowledgement without exposing its `llq::Node`. If reconciliation
    /// returns an error or panics and this value is dropped, the owner retains the exact pending
    /// node in quarantine and the id cannot be reused. Slice B will match its id+generation and
    /// consume it only after mirrors and represented-resource guards are retired.
    pub(crate) fn try_pending_reclaim(&mut self) -> Option<PendingNodeReclaim<'_>> {
        if self.pending_reclaim.is_none() {
            self.pending_reclaim = self.graph_reclaims.pop();
        }
        self.pending_reclaim.as_ref()?;
        Some(PendingNodeReclaim { owner: self })
    }

    /// Transfers one exact post-cleanup graph acknowledgement to the unique lifecycle driver.
    ///
    /// Unlike [`PendingNodeReclaim`], this token does not borrow the id owner, so a bounded
    /// lifetime registry can retain an early acknowledgement in its matching generation slot and
    /// continue draining later acknowledgements. The token exposes no raw node and deliberately
    /// leaks it if dropped without an explicit successful reconciliation.
    pub(crate) fn try_take_pending_reclaim(&mut self) -> Option<OwnedPendingNodeReclaim> {
        let node = self
            .pending_reclaim
            .take()
            .or_else(|| self.graph_reclaims.pop())?;
        Some(OwnedPendingNodeReclaim {
            node: Some(node),
            identity: self.identity(),
        })
    }

    /// Returns one fully reconciled exact acknowledgement to this allocator. A foreign token is
    /// returned intact, and no queue is modified, so the lifecycle registry can quarantine it and
    /// keep the id permanently unavailable. Once accepted, the infallible queue push is the sole
    /// publication point for ordinary id reuse.
    pub(crate) fn make_reconciled_available(
        &mut self,
        mut pending: OwnedPendingNodeReclaim,
    ) -> Result<(), OwnedPendingNodeReclaim> {
        if !self.identity().ptr_eq(&pending.identity) {
            return Err(pending);
        }
        let node = pending
            .node
            .take()
            .expect("owned pending reclaim is consumed exactly once");
        self.inner
            .available_return
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(node);
        Ok(())
    }

    /// Destroys an exact acknowledgement off RT after whole-graph retirement proves that this
    /// context will never reuse the id.
    pub(crate) fn discard_after_whole_graph(
        &mut self,
        mut pending: OwnedPendingNodeReclaim,
    ) -> Result<(), OwnedPendingNodeReclaim> {
        if !self.identity().ptr_eq(&pending.identity) {
            return Err(pending);
        }
        drop(
            pending
                .node
                .take()
                .expect("owned pending reclaim is consumed exactly once"),
        );
        Ok(())
    }

    #[cfg(test)]
    fn set_next_for_test(&self, next: u64) {
        self.inner
            .allocation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .next = next;
    }

    #[cfg(test)]
    pub(crate) fn set_release_hook_for_test(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        *self
            .inner
            .release_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((entered, release));
    }
}

/// Opaque owned post-cleanup graph acknowledgement.
///
/// Only the unique [`InjectedNodeIdOwner`] can construct one or make its exact intrusive node
/// available. Dropping an unreconciled token intentionally leaks that node, preventing accidental
/// id reuse after a registry mismatch, cleanup failure, or lifecycle quarantine.
#[must_use = "an owned reclaim must be reconciled or explicitly retained in quarantine"]
pub(crate) struct OwnedPendingNodeReclaim {
    node: Option<llq::Node<AudioNodeId>>,
    identity: InjectedNodeIdIdentity,
}

impl OwnedPendingNodeReclaim {
    pub(crate) fn id(&self) -> AudioNodeId {
        **self
            .node
            .as_ref()
            .expect("owned pending reclaim retains its exact node")
    }
}

impl Drop for OwnedPendingNodeReclaim {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            std::mem::forget(node);
        }
    }
}

impl Drop for InjectedNodeIdOwner {
    fn drop(&mut self) {
        self.inner
            .lifecycle
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                Some(match AllocatorLifecycle::from_u8(state) {
                    AllocatorLifecycle::OpenIdle => AllocatorLifecycle::RetiredIdle as u8,
                    AllocatorLifecycle::OpenActive => AllocatorLifecycle::RetiredActive as u8,
                    AllocatorLifecycle::RetiredIdle | AllocatorLifecycle::RetiredActive => {
                        unreachable!("the unique owner retires exactly once")
                    }
                })
            })
            .expect("unique owner retirement transition cannot fail");
    }
}

/// Opaque, owner-borrowing graph acknowledgement. There is no raw-node constructor or extractor.
#[must_use = "a pending reclaim must be reconciled or remains quarantined in its owner"]
pub(crate) struct PendingNodeReclaim<'a> {
    owner: &'a mut InjectedNodeIdOwner,
}

impl PendingNodeReclaim<'_> {
    pub(crate) fn id(&self) -> AudioNodeId {
        **self
            .owner
            .pending_reclaim
            .as_ref()
            .expect("pending reclaim borrow retains its exact node")
    }

    /// Marks reconciliation successful and makes this exact node available. Merely dropping the
    /// opaque value deliberately does not do so.
    pub(crate) fn make_available(self) {
        let node = self
            .owner
            .pending_reclaim
            .take()
            .expect("pending reclaim is consumed exactly once");
        self.owner
            .inner
            .available_return
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(node);
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::*;

    fn pair(
        first: u64,
    ) -> (
        InjectedGraphReclaimInit,
        InjectedNodeIdAllocator,
        InjectedNodeIdOwner,
    ) {
        let (allocator, owner, graph_init) = injected_node_id_pair(first);
        (graph_init, allocator, owner)
    }

    #[test]
    fn reclaim_activity_wake_is_bounded_allocation_free_and_authoritative_queue_wins() {
        let (_allocator, mut owner, graph) = injected_node_id_pair(0);

        alloc_counter::deny_alloc(|| graph.publisher.publish());
        owner.reclaim_activity_receiver().recv().unwrap();
        // Full coalesces without blocking or allocation.
        graph.publisher.publish();
        alloc_counter::deny_alloc(|| graph.publisher.publish());
        owner.reclaim_activity_receiver().recv().unwrap();

        // Disconnect is equally non-waiting. Replace/drop the matching receiver only in this
        // module-private test; production exposes a borrowed receiver and cannot do this.
        let (_replacement_send, replacement_recv) = crossbeam_channel::bounded(1);
        let old = std::mem::replace(&mut owner.reclaim_activity, replacement_recv);
        drop(old);
        alloc_counter::deny_alloc(|| graph.publisher.publish());
        assert!(owner.try_pending_reclaim().is_none());
    }

    #[test]
    fn render_publisher_drop_cannot_destroy_lifecycle_owned_wake_allocation() {
        let (_allocator, owner, graph) = injected_node_id_pair(0);
        std::thread::spawn(move || drop(graph)).join().unwrap();
        assert_eq!(
            owner.reclaim_activity_receiver().try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        );
    }

    #[test]
    fn rollback_preserves_exact_nodes_and_relative_order_after_panic() {
        let (_graph_return, allocator, _owner) = pair(11);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let ids = allocator.try_reserve(2).unwrap();
            assert_eq!(ids.id(0), AudioNodeId(11));
            assert_eq!(ids.id(1), AudioNodeId(12));
            panic!("constructor panic");
        }));
        assert!(result.is_err());

        let ids = allocator.try_reserve(2).unwrap();
        assert_eq!(ids.id(0), AudioNodeId(11));
        assert_eq!(ids.id(1), AudioNodeId(12));
    }

    #[test]
    fn moved_node_must_be_restored_before_rollback_or_commit() {
        let (_graph_return, allocator, _owner) = pair(20);
        let mut ids = allocator.try_reserve(1).unwrap();
        let node = ids.take_reclaim_node(0).unwrap();
        ids.restore_reclaim_node(0, node).unwrap();
        drop(ids);
        let ids = allocator.try_reserve(1).unwrap();
        assert_eq!(ids.id(0), AudioNodeId(20));
    }

    #[test]
    fn graph_reclaim_requires_explicit_reconciliation_before_reuse() {
        let (mut graph_return, allocator, mut owner) = pair(30);
        graph_return.push_for_test(llq::Node::new(AudioNodeId(7)));

        let fresh = allocator.try_reserve(1).unwrap();
        assert_eq!(fresh.id(0), AudioNodeId(30));
        drop(fresh);

        let reclaimed = owner.try_pending_reclaim().unwrap();
        assert_eq!(reclaimed.id(), AudioNodeId(7));
        reclaimed.make_available();
        let reused = allocator.try_reserve(1).unwrap();
        assert_eq!(reused.id(0), AudioNodeId(30));
        drop(reused);
        let reused = allocator.try_reserve(1).unwrap();
        assert_eq!(reused.id(0), AudioNodeId(7));
    }

    #[test]
    fn owner_drop_makes_surviving_allocator_inert() {
        let (_graph_return, allocator, owner) = pair(0);
        drop(owner);
        assert_eq!(
            allocator.try_reserve(1).err(),
            Some(ProvisionalNodeIdError::OwnerGone)
        );
    }

    #[test]
    fn owner_retirement_cannot_reopen_during_final_reservation_drop() {
        let (_graph_return, allocator, owner) = pair(0);
        let reservation = allocator.try_reserve(1).unwrap();
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        owner.set_release_hook_for_test(entered_send, release_recv);
        drop(owner);

        let dropper = std::thread::spawn(move || drop(reservation));
        entered_recv.recv().unwrap();
        // The reservation still owns the final strong Arc here, so Weak::upgrade succeeds. The
        // combined lifecycle word must nevertheless preserve retirement and reject admission.
        assert_eq!(
            allocator.try_reserve(1).err(),
            Some(ProvisionalNodeIdError::OwnerGone)
        );
        release_send.send(()).unwrap();
        dropper.join().unwrap();
        assert_eq!(
            allocator.try_reserve(1).err(),
            Some(ProvisionalNodeIdError::OwnerGone)
        );
    }

    #[test]
    fn allocator_reservation_and_unique_owner_have_worker_safe_traits() {
        fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
        fn assert_send<T: Send>() {}
        assert_clone_send_sync::<InjectedNodeIdAllocator>();
        assert_send::<InjectedNodeIdOwner>();
        assert_send::<ProvisionalNodeIds>();
    }

    #[test]
    fn reservation_is_serial_and_limits_and_exhaustion_are_typed() {
        let (_graph_return, allocator, _owner) = pair(0);
        assert_eq!(
            allocator.try_reserve(0).err(),
            Some(ProvisionalNodeIdError::Empty)
        );
        assert_eq!(
            allocator.try_reserve(MAX_PROVISIONAL_NODE_IDS + 1).err(),
            Some(ProvisionalNodeIdError::TooMany)
        );
        let held = allocator.try_reserve(1).unwrap();
        assert_eq!(
            allocator.try_reserve(1).err(),
            Some(ProvisionalNodeIdError::Contended)
        );
        drop(held);
        let (_graph_return, exhausted_allocator, exhausted_owner) = pair(u64::MAX);
        assert_eq!(
            exhausted_allocator.try_reserve(1).err(),
            Some(ProvisionalNodeIdError::Exhausted)
        );
        exhausted_owner.set_next_for_test(50);
        assert!(exhausted_allocator.try_reserve(1).is_ok());
    }

    #[test]
    fn close_reconciliation_observes_each_exact_reclaim_before_returning_it() {
        let (mut graph_return, allocator, mut owner) = pair(100);
        graph_return.push_for_test(llq::Node::new(AudioNodeId(4)));
        graph_return.push_for_test(llq::Node::new(AudioNodeId(9)));
        let first = owner.try_pending_reclaim().unwrap();
        assert_eq!(first.id(), AudioNodeId(4));
        first.make_available();
        let second = owner.try_pending_reclaim().unwrap();
        assert_eq!(second.id(), AudioNodeId(9));
        second.make_available();

        let ids = allocator.try_reserve(2).unwrap();
        assert_eq!(ids.id(0), AudioNodeId(4));
        assert_eq!(ids.id(1), AudioNodeId(9));
    }

    #[test]
    fn failed_or_panicking_reconciliation_quarantines_exact_node_until_success() {
        let (mut graph_return, allocator, mut owner) = pair(40);
        graph_return.push_for_test(llq::Node::new(AudioNodeId(6)));
        let result = catch_unwind(AssertUnwindSafe(|| {
            let pending = owner.try_pending_reclaim().unwrap();
            assert_eq!(pending.id(), AudioNodeId(6));
            panic!("reconciliation panic");
        }));
        assert!(result.is_err());

        let fresh = allocator.try_reserve(1).unwrap();
        assert_eq!(fresh.id(0), AudioNodeId(40));
        drop(fresh);
        let pending = owner.try_pending_reclaim().unwrap();
        assert_eq!(pending.id(), AudioNodeId(6));
        pending.make_available();
    }

    #[test]
    fn reconciled_return_may_interleave_ahead_of_rollback_without_loss() {
        let (mut graph_return, allocator, mut owner) = pair(11);
        let provisional = allocator.try_reserve(2).unwrap();
        assert_eq!(provisional.id(0), AudioNodeId(11));
        assert_eq!(provisional.id(1), AudioNodeId(12));
        graph_return.push_for_test(llq::Node::new(AudioNodeId(7)));
        owner
            .try_pending_reclaim()
            .expect("graph return")
            .make_available();
        drop(provisional);

        let ids = allocator.try_reserve(3).unwrap();
        assert_eq!(ids.id(0), AudioNodeId(7));
        assert_eq!(ids.id(1), AudioNodeId(11));
        assert_eq!(ids.id(2), AudioNodeId(12));
    }
}
