//! Bounded node-lifetime ownership foundation for a future injected context.
//!
//! This foundation is deliberately not wired to `ConcreteBaseAudioContext`. Cloneable registrars
//! and live registrations retain only a weak registry capability; a unique lifecycle-side owner
//! holds the fixed slot storage, exact graph-reclaim owner, and authoritative wake receivers.
//! No ordinary graph-control credit is retained for a node lifetime. Requested teardown is driven
//! explicitly by the lifecycle caller through bounded, retryable operations; this slice does not
//! create a worker or wire a public context constructor.

#![allow(dead_code)]

use std::num::NonZeroU64;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError, Weak};

use super::injected_control::{
    AcceptedBatchFinalizeError, DrainedControlClose, InjectedControlIdentity,
    InjectedControlProducer, InjectedNodeLifetimeBootstrap,
};
use super::injected_ids::{
    InjectedGraphReclaimInit, InjectedNodeIdAllocator, InjectedNodeIdIdentity, InjectedNodeIdOwner,
    OwnedPendingNodeReclaim,
};
use super::AudioNodeId;

mod teardown;
#[allow(unused_imports)] // consumed by the later concrete lifecycle integration
pub(crate) use teardown::{
    NodeLifetimeActivity, NodeLifetimeDriveOutcome, NodeLifetimeQuarantineReason,
    NodeLifetimeRetryReason, NODE_LIFETIME_RETRY_INTERVAL,
};

#[cfg(test)]
mod teardown_tests;

pub(crate) const DEFAULT_NODE_LIFETIME_CAPACITY: usize = 256;

const PHASE_BITS: u32 = 4;
const RECLAIMED_BIT: u64 = 1 << PHASE_BITS;
const GENERATION_SHIFT: u32 = PHASE_BITS + 1;
const PHASE_MASK: u64 = (1 << PHASE_BITS) - 1;
const MAX_GENERATION: u64 = u64::MAX >> GENERATION_SHIFT;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotPhase {
    Vacant = 0,
    Provisional = 1,
    Live = 2,
    Requested = 3,
    Canceling = 4,
    Servicing = 5,
    AcceptedPending = 6,
    AwaitingReclaim = 7,
    Reconciling = 8,
    Sealed = 9,
    Quarantined = 10,
}

impl SlotPhase {
    fn from_word(word: u64) -> Self {
        match (word & PHASE_MASK) as u8 {
            0 => Self::Vacant,
            1 => Self::Provisional,
            2 => Self::Live,
            3 => Self::Requested,
            4 => Self::Canceling,
            5 => Self::Servicing,
            6 => Self::AcceptedPending,
            7 => Self::AwaitingReclaim,
            8 => Self::Reconciling,
            9 => Self::Sealed,
            10 => Self::Quarantined,
            _ => unreachable!("private node-lifetime slot phase"),
        }
    }
}

fn generation(word: u64) -> u64 {
    word >> GENERATION_SHIFT
}

fn has_reclaim(word: u64) -> bool {
    word & RECLAIMED_BIT != 0
}

fn with_phase(word: u64, phase: SlotPhase) -> u64 {
    (word & !PHASE_MASK) | phase as u64
}

fn slot_word(generation: u64, phase: SlotPhase, reclaimed: bool) -> u64 {
    (generation << GENERATION_SHIFT) | phase as u64 | if reclaimed { RECLAIMED_BIT } else { 0 }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegistryPhase {
    Open = 0,
    Sealed = 1,
    Retired = 2,
    Quarantined = 3,
}

impl RegistryPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Open,
            1 => Self::Sealed,
            2 => Self::Retired,
            3 => Self::Quarantined,
            _ => unreachable!("private node-lifetime registry phase"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeReclaimCleanupError {
    Rejected,
}

/// Control-side represented-resource cleanup run before an exact graph id becomes reusable.
/// Concrete mirror/host-guard implementations are deferred with the injected context wiring.
/// Returning `Rejected` must leave the implementation retry-safe and idempotent: the ordinary
/// path retains it in quarantine, and whole-graph retirement invokes `reconcile` once more before
/// reporting a final rejected cleanup.
pub(crate) trait InjectedNodeReclaimCleanup: Send {
    fn reconcile(&mut self, id: AudioNodeId) -> Result<(), NodeReclaimCleanupError>;
}

struct SlotPayload {
    cleanup: Option<Box<dyn InjectedNodeReclaimCleanup>>,
    reclaim: Option<OwnedPendingNodeReclaim>,
}

struct LifetimeSlot {
    word: AtomicU64,
    id: AtomicU64,
    payload: Mutex<SlotPayload>,
}

struct NodeLifetimeInner {
    phase: AtomicU8,
    allocation: Mutex<()>,
    slots: Box<[LifetimeSlot]>,
    request_wake: crossbeam_channel::Sender<()>,
    control_identity: InjectedControlIdentity,
    node_id_identity: InjectedNodeIdIdentity,
    #[cfg(test)]
    registration_publish_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
    #[cfg(test)]
    arm_publish_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
    #[cfg(test)]
    construction_arm_behavior: AtomicU8,
    #[cfg(test)]
    reclaim_attach_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
    #[cfg(test)]
    teardown_finalizer_behavior: AtomicU8,
    #[cfg(test)]
    teardown_finalizer_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
    #[cfg(test)]
    teardown_precommit_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
    #[cfg(test)]
    teardown_before_restore_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
    #[cfg(test)]
    teardown_post_publish_hook: Mutex<
        Option<(
            crossbeam_channel::Sender<()>,
            crossbeam_channel::Receiver<()>,
        )>,
    >,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RegistrationKey {
    slot: usize,
    generation: NonZeroU64,
}

#[derive(Clone)]
pub(crate) struct InjectedNodeLifetimeRegistrar {
    inner: Weak<NodeLifetimeInner>,
}

pub(crate) struct InjectedNodeLifetimeOwner {
    inner: Option<Arc<NodeLifetimeInner>>,
    request_wake: Option<crossbeam_channel::Receiver<()>>,
    node_ids: Option<InjectedNodeIdOwner>,
    control: Option<InjectedControlProducer>,
    orphan_reclaim: Option<OwnedPendingNodeReclaim>,
}

pub(crate) fn injected_node_lifetime_registry(
    capacity: usize,
    control: &InjectedControlProducer,
    node_ids: InjectedNodeIdOwner,
    graph: InjectedGraphReclaimInit,
) -> Result<(InjectedNodeLifetimeRegistrar, InjectedNodeLifetimeBootstrap), NodeLifetimeBuildFailure>
{
    if capacity == 0 || capacity > DEFAULT_NODE_LIFETIME_CAPACITY {
        return Err(NodeLifetimeBuildFailure {
            error: NodeLifetimeBuildError::InvalidCapacity,
            node_ids,
            graph,
        });
    }
    if !node_ids.matches_graph_init(&graph) {
        return Err(NodeLifetimeBuildFailure {
            error: NodeLifetimeBuildError::MismatchedNodeIdGraph,
            node_ids,
            graph,
        });
    }
    let (request_wake, request_receiver) = crossbeam_channel::bounded(1);
    let slots = (0..capacity)
        .map(|_| LifetimeSlot {
            word: AtomicU64::new(slot_word(0, SlotPhase::Vacant, false)),
            id: AtomicU64::new(0),
            payload: Mutex::new(SlotPayload {
                cleanup: None,
                reclaim: None,
            }),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let inner = Arc::new(NodeLifetimeInner {
        phase: AtomicU8::new(RegistryPhase::Open as u8),
        allocation: Mutex::new(()),
        slots,
        request_wake,
        control_identity: control.identity(),
        node_id_identity: node_ids.identity(),
        #[cfg(test)]
        registration_publish_hook: Mutex::new(None),
        #[cfg(test)]
        arm_publish_hook: Mutex::new(None),
        #[cfg(test)]
        construction_arm_behavior: AtomicU8::new(0),
        #[cfg(test)]
        reclaim_attach_hook: Mutex::new(None),
        #[cfg(test)]
        teardown_finalizer_behavior: AtomicU8::new(0),
        #[cfg(test)]
        teardown_finalizer_hook: Mutex::new(None),
        #[cfg(test)]
        teardown_precommit_hook: Mutex::new(None),
        #[cfg(test)]
        teardown_before_restore_hook: Mutex::new(None),
        #[cfg(test)]
        teardown_post_publish_hook: Mutex::new(None),
    });
    let registrar = InjectedNodeLifetimeRegistrar {
        inner: Arc::downgrade(&inner),
    };
    let owner = InjectedNodeLifetimeOwner {
        inner: Some(inner),
        request_wake: Some(request_receiver),
        node_ids: Some(node_ids),
        control: Some(control.clone()),
        orphan_reclaim: None,
    };
    let node_lifetimes = InjectedNodeLifetimeBootstrap::new(owner, graph)
        .ok()
        .expect("validated owner and graph identities match");
    Ok((registrar, node_lifetimes))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeLifetimeBuildError {
    InvalidCapacity,
    MismatchedNodeIdGraph,
}

pub(crate) struct NodeLifetimeBuildFailure {
    pub(crate) error: NodeLifetimeBuildError,
    pub(crate) node_ids: InjectedNodeIdOwner,
    pub(crate) graph: InjectedGraphReclaimInit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeRegistrationError {
    OwnerGone,
    Sealed,
    Contended,
    Poisoned,
    Full,
    DuplicateId,
    GenerationExhausted,
    ProtocolViolation,
}

pub(crate) struct NodeRegistrationFailure {
    pub(crate) error: NodeRegistrationError,
    pub(crate) cleanup: Box<dyn InjectedNodeReclaimCleanup>,
}

impl InjectedNodeLifetimeRegistrar {
    /// Exact constructor branding: this registry was built from the same control transport and
    /// node-id owner represented by these two weak construction capabilities.
    pub(crate) fn matches_constructor(
        &self,
        control: &InjectedControlProducer,
        allocator: &InjectedNodeIdAllocator,
    ) -> bool {
        let Some(inner) = self.inner.upgrade() else {
            return false;
        };
        inner.control_identity.ptr_eq(&control.identity())
            && inner.node_id_identity.ptr_eq(&allocator.identity())
    }

    #[cfg(test)]
    pub(crate) fn fail_construction_arm_for_test(&self, ordinal: u8) {
        assert!(ordinal > 0 && ordinal < 0x80);
        if let Some(inner) = self.inner.upgrade() {
            inner
                .construction_arm_behavior
                .store(ordinal, Ordering::Release);
        }
    }

    #[cfg(test)]
    pub(crate) fn panic_construction_arm_for_test(&self, ordinal: u8) {
        assert!(ordinal > 0 && ordinal < 0x80);
        if let Some(inner) = self.inner.upgrade() {
            inner
                .construction_arm_behavior
                .store(ordinal | 0x80, Ordering::Release);
        }
    }

    pub(crate) fn try_register(
        &self,
        id: AudioNodeId,
        cleanup: Box<dyn InjectedNodeReclaimCleanup>,
    ) -> Result<ProvisionalNodeRegistration, NodeRegistrationFailure> {
        let Some(inner) = self.inner.upgrade() else {
            return Err(NodeRegistrationFailure {
                error: NodeRegistrationError::OwnerGone,
                cleanup,
            });
        };
        if RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire)) != RegistryPhase::Open {
            return Err(NodeRegistrationFailure {
                error: NodeRegistrationError::Sealed,
                cleanup,
            });
        }
        let _allocation = match inner.allocation.try_lock() {
            Ok(allocation) => allocation,
            Err(TryLockError::WouldBlock) => {
                return Err(NodeRegistrationFailure {
                    error: NodeRegistrationError::Contended,
                    cleanup,
                })
            }
            Err(TryLockError::Poisoned(_)) => {
                return Err(NodeRegistrationFailure {
                    error: NodeRegistrationError::Poisoned,
                    cleanup,
                })
            }
        };
        if RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire)) != RegistryPhase::Open {
            return Err(NodeRegistrationFailure {
                error: NodeRegistrationError::Sealed,
                cleanup,
            });
        }
        #[cfg(test)]
        if let Some((entered, release)) = inner
            .registration_publish_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            entered.send(()).unwrap();
            release.recv().unwrap();
        }
        if inner.slots.iter().any(|slot| {
            SlotPhase::from_word(slot.word.load(Ordering::Acquire)) != SlotPhase::Vacant
                && slot.id.load(Ordering::Acquire) == id.0
        }) {
            return Err(NodeRegistrationFailure {
                error: NodeRegistrationError::DuplicateId,
                cleanup,
            });
        }
        let mut saw_exhausted_vacant = false;
        let available = inner.slots.iter().enumerate().find(|(_, slot)| {
            let word = slot.word.load(Ordering::Acquire);
            if SlotPhase::from_word(word) != SlotPhase::Vacant {
                return false;
            }
            if generation(word) >= MAX_GENERATION {
                saw_exhausted_vacant = true;
                return false;
            }
            true
        });
        let Some((slot_index, slot)) = available else {
            return Err(NodeRegistrationFailure {
                error: if saw_exhausted_vacant {
                    NodeRegistrationError::GenerationExhausted
                } else {
                    NodeRegistrationError::Full
                },
                cleanup,
            });
        };
        let old_word = slot.word.load(Ordering::Acquire);
        let next_generation = generation(old_word) + 1;
        let mut payload = match slot.payload.try_lock() {
            Ok(payload) => payload,
            Err(TryLockError::WouldBlock) => {
                return Err(NodeRegistrationFailure {
                    error: NodeRegistrationError::Contended,
                    cleanup,
                })
            }
            Err(TryLockError::Poisoned(_)) => {
                return Err(NodeRegistrationFailure {
                    error: NodeRegistrationError::Poisoned,
                    cleanup,
                })
            }
        };
        if payload.cleanup.is_some() || payload.reclaim.is_some() {
            return Err(NodeRegistrationFailure {
                error: NodeRegistrationError::ProtocolViolation,
                cleanup,
            });
        }
        payload.cleanup = Some(cleanup);
        slot.id.store(id.0, Ordering::Relaxed);
        slot.word.store(
            slot_word(next_generation, SlotPhase::Provisional, false),
            Ordering::Release,
        );
        drop(payload);
        drop(_allocation);
        Ok(ProvisionalNodeRegistration {
            inner: Arc::downgrade(&inner),
            key: RegistrationKey {
                slot: slot_index,
                generation: NonZeroU64::new(next_generation).unwrap(),
            },
            completed: false,
            accepted: AtomicBool::new(false),
            armed: AtomicBool::new(false),
        })
    }
}

#[must_use = "a provisional registration must be armed after accepted construction or cancelled"]
pub(crate) struct ProvisionalNodeRegistration {
    inner: Weak<NodeLifetimeInner>,
    key: RegistrationKey,
    completed: bool,
    accepted: AtomicBool,
    armed: AtomicBool,
}

#[derive(Clone, Copy)]
pub(crate) struct NodeRegistrationArm<'a> {
    inner: &'a Weak<NodeLifetimeInner>,
    key: RegistrationKey,
    accepted: &'a AtomicBool,
    armed: &'a AtomicBool,
}

impl ProvisionalNodeRegistration {
    /// Copy ordering token suitable for B1's accepted-batch finalizer. It temporarily upgrades
    /// the weak capability; lifecycle retirement either waits for that upgrade or makes arm fail.
    pub(crate) fn arm_token(&self) -> NodeRegistrationArm<'_> {
        NodeRegistrationArm {
            inner: &self.inner,
            key: self.key,
            accepted: &self.accepted,
            armed: &self.armed,
        }
    }

    pub(crate) fn ready_for_registration(&self) -> bool {
        if !self.armed.load(Ordering::Acquire) {
            return false;
        }
        self.inner.upgrade().is_none_or(|inner| {
            let word = inner.slots[self.key.slot].word.load(Ordering::Acquire);
            generation(word) == self.key.generation.get()
                && matches!(
                    SlotPhase::from_word(word),
                    SlotPhase::Live | SlotPhase::Sealed
                )
        })
    }

    pub(crate) fn into_registration(
        mut self,
    ) -> Result<InjectedNodeRegistration, NodeRegistrationError> {
        if !self.ready_for_registration() {
            return Err(NodeRegistrationError::ProtocolViolation);
        }
        self.completed = true;
        Ok(InjectedNodeRegistration {
            inner: Weak::clone(&self.inner),
            key: self.key,
        })
    }
}

impl NodeRegistrationArm<'_> {
    pub(crate) fn mark_accepted(self) {
        self.accepted.store(true, Ordering::Release);
    }

    pub(crate) fn arm_accepted(self) -> Result<(), AcceptedBatchFinalizeError> {
        if !self.accepted.load(Ordering::Acquire) {
            return Err(AcceptedBatchFinalizeError::Rejected);
        }
        let Some(inner) = self.inner.upgrade() else {
            return Err(AcceptedBatchFinalizeError::Rejected);
        };
        #[cfg(test)]
        {
            let behavior = inner.construction_arm_behavior.load(Ordering::Acquire);
            if behavior != 0 {
                let countdown = behavior & 0x7f;
                if countdown == 1 {
                    inner.construction_arm_behavior.store(0, Ordering::Release);
                    if behavior & 0x80 != 0 {
                        panic!("injected construction arm panic");
                    }
                    return Err(AcceptedBatchFinalizeError::Rejected);
                }
                inner
                    .construction_arm_behavior
                    .store((countdown - 1) | (behavior & 0x80), Ordering::Release);
            }
        }
        let _allocation = match inner.allocation.try_lock() {
            Ok(allocation) => allocation,
            Err(TryLockError::WouldBlock | TryLockError::Poisoned(_)) => {
                return Err(AcceptedBatchFinalizeError::Rejected)
            }
        };
        if RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire)) != RegistryPhase::Open {
            return Err(AcceptedBatchFinalizeError::Rejected);
        }
        #[cfg(test)]
        if let Some((entered, release)) = inner
            .arm_publish_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            entered.send(()).unwrap();
            release.recv().unwrap();
        }
        let slot = &inner.slots[self.key.slot];
        let expected = slot_word(self.key.generation.get(), SlotPhase::Provisional, false);
        let reclaimed = expected | RECLAIMED_BIT;
        let result = slot
            .word
            .compare_exchange(
                expected,
                with_phase(expected, SlotPhase::Live),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .or_else(|observed| {
                if observed == reclaimed {
                    slot.word.compare_exchange(
                        reclaimed,
                        with_phase(reclaimed, SlotPhase::Live),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                } else {
                    Err(observed)
                }
            });
        result
            .map(|_| self.armed.store(true, Ordering::Release))
            .map_err(|_| AcceptedBatchFinalizeError::Rejected)
    }

    pub(crate) fn arm(self) -> Result<(), AcceptedBatchFinalizeError> {
        self.mark_accepted();
        self.arm_accepted()
    }

    /// Internal constructor protocol failure. Prefer an explicit quarantined slot; if structural
    /// corruption is proven, this off-RT path serializes with allocation and quarantines both the
    /// exact slot and the registry so later construction cannot proceed.
    pub(crate) fn quarantine_accepted(self) {
        self.mark_accepted();
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        teardown::quarantine_slot(&inner, self.key);
    }
}

impl Drop for ProvisionalNodeRegistration {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        if RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire)) != RegistryPhase::Open {
            return;
        }
        let slot = &inner.slots[self.key.slot];
        loop {
            let word = slot.word.load(Ordering::Acquire);
            if generation(word) != self.key.generation.get() {
                return;
            }
            match SlotPhase::from_word(word) {
                SlotPhase::Provisional => {
                    let mut payload = slot
                        .payload
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let current = slot.word.load(Ordering::Acquire);
                    if generation(current) != self.key.generation.get()
                        || SlotPhase::from_word(current) != SlotPhase::Provisional
                    {
                        drop(payload);
                        continue;
                    }
                    if self.accepted.load(Ordering::Acquire)
                        || payload.reclaim.is_some()
                        || has_reclaim(current)
                    {
                        if slot
                            .word
                            .compare_exchange(
                                current,
                                with_phase(current, SlotPhase::Requested),
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            drop(payload);
                            let _ = inner.request_wake.try_send(());
                            return;
                        }
                        drop(payload);
                        continue;
                    }
                    if slot
                        .word
                        .compare_exchange(
                            current,
                            with_phase(current, SlotPhase::Canceling),
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_err()
                    {
                        drop(payload);
                        continue;
                    }
                    let cleanup = payload.cleanup.take();
                    drop(payload);
                    let clean = cleanup.is_none_or(panic_safe_drop);
                    let phase = if clean {
                        SlotPhase::Vacant
                    } else {
                        SlotPhase::Quarantined
                    };
                    let _ = slot.word.compare_exchange(
                        with_phase(current, SlotPhase::Canceling),
                        slot_word(self.key.generation.get(), phase, false),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    return;
                }
                SlotPhase::Live => {
                    if slot
                        .word
                        .compare_exchange(
                            word,
                            with_phase(word, SlotPhase::Requested),
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        let _ = inner.request_wake.try_send(());
                        return;
                    }
                }
                _ => return,
            }
        }
    }
}

pub(crate) struct InjectedNodeRegistration {
    inner: Weak<NodeLifetimeInner>,
    key: RegistrationKey,
}

impl Drop for InjectedNodeRegistration {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        if RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire)) != RegistryPhase::Open {
            return;
        }
        let slot = &inner.slots[self.key.slot];
        loop {
            let word = slot.word.load(Ordering::Acquire);
            if generation(word) != self.key.generation.get()
                || SlotPhase::from_word(word) != SlotPhase::Live
            {
                return;
            }
            if slot
                .word
                .compare_exchange(
                    word,
                    with_phase(word, SlotPhase::Requested),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let _ = inner.request_wake.try_send(());
                return;
            }
        }
    }
}

fn panic_safe_drop(cleanup: Box<dyn InjectedNodeReclaimCleanup>) -> bool {
    match panic::catch_unwind(AssertUnwindSafe(|| drop(cleanup))) {
        Ok(()) => true,
        Err(payload) => {
            std::mem::forget(payload);
            false
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeReclaimPlacementError {
    ReclaimMismatch,
    DuplicateReclaim,
}

impl InjectedNodeLifetimeOwner {
    fn inner(&self) -> &Arc<NodeLifetimeInner> {
        self.inner
            .as_ref()
            .expect("live node-lifetime owner retains its registry")
    }

    #[cfg(test)]
    pub(crate) fn set_id_release_hook_for_test(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        self.node_ids
            .as_ref()
            .expect("live node-lifetime owner retains its id owner")
            .set_release_hook_for_test(entered, release);
    }

    #[cfg(test)]
    pub(crate) fn slot_phase_counts_for_test(&self) -> [usize; 6] {
        let mut counts = [0; 6];
        for slot in &self.inner().slots {
            let index = match SlotPhase::from_word(slot.word.load(Ordering::Acquire)) {
                SlotPhase::Vacant => 0,
                SlotPhase::Provisional => 1,
                SlotPhase::Live => 2,
                SlotPhase::Requested => 3,
                SlotPhase::Quarantined => 4,
                _ => 5,
            };
            counts[index] += 1;
        }
        counts
    }

    pub(crate) fn control_identity(&self) -> &InjectedControlIdentity {
        &self.inner().control_identity
    }

    pub(crate) fn matches_graph_init(&self, graph: &InjectedGraphReclaimInit) -> bool {
        self.node_ids
            .as_ref()
            .expect("live node-lifetime owner retains its id owner")
            .matches_graph_init(graph)
    }

    #[cfg(test)]
    fn set_registration_publish_hook(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        *self
            .inner()
            .registration_publish_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((entered, release));
    }

    #[cfg(test)]
    pub(crate) fn set_arm_publish_hook(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        *self
            .inner()
            .arm_publish_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((entered, release));
    }

    #[cfg(test)]
    fn set_reclaim_attach_hook(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    ) {
        *self
            .inner()
            .reclaim_attach_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((entered, release));
    }

    fn quarantine_registry(&self) {
        let inner = self.inner();
        let _allocation = inner
            .allocation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = inner.phase.compare_exchange(
            RegistryPhase::Open as u8,
            RegistryPhase::Quarantined as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// Transfers every currently available exact acknowledgement into a matching fixed slot.
    /// A live registration with an already-reclaimed graph node retains the exact token until that
    /// old registration is dropped, preventing ABA reuse after processor failure.
    pub(crate) fn ingest_reclaims(&mut self) -> Result<usize, NodeReclaimPlacementError> {
        if self.orphan_reclaim.is_some() {
            return Err(NodeReclaimPlacementError::ReclaimMismatch);
        }
        if RegistryPhase::from_u8(self.inner().phase.load(Ordering::Acquire)) != RegistryPhase::Open
        {
            return Err(NodeReclaimPlacementError::ReclaimMismatch);
        }
        let mut count = 0;
        loop {
            let pending = {
                self.node_ids
                    .as_mut()
                    .expect("live owner retains id owner")
                    .try_take_pending_reclaim()
            };
            let Some(pending) = pending else {
                return Ok(count);
            };
            let id = pending.id();
            let matches = self
                .inner()
                .slots
                .iter()
                .enumerate()
                .filter_map(|(index, slot)| {
                    let word = slot.word.load(Ordering::Acquire);
                    (SlotPhase::from_word(word) != SlotPhase::Vacant
                        && slot.id.load(Ordering::Acquire) == id.0)
                        .then_some((index, word))
                })
                .collect::<arrayvec::ArrayVec<_, 2>>();
            if matches.len() != 1 {
                self.orphan_reclaim = Some(pending);
                self.quarantine_registry();
                return Err(NodeReclaimPlacementError::ReclaimMismatch);
            }
            let (index, observed) = matches[0];
            let slot = &self.inner().slots[index];
            let mut payload = slot
                .payload
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let word = slot.word.load(Ordering::Acquire);
            if generation(word) != generation(observed)
                || slot.id.load(Ordering::Acquire) != id.0
                || !matches!(
                    SlotPhase::from_word(word),
                    SlotPhase::Provisional
                        | SlotPhase::Live
                        | SlotPhase::Requested
                        | SlotPhase::Servicing
                        | SlotPhase::AcceptedPending
                        | SlotPhase::AwaitingReclaim
                )
            {
                drop(payload);
                self.orphan_reclaim = Some(pending);
                self.quarantine_registry();
                return Err(NodeReclaimPlacementError::ReclaimMismatch);
            }
            if payload.reclaim.is_some() || has_reclaim(word) {
                drop(payload);
                self.orphan_reclaim = Some(pending);
                self.quarantine_registry();
                return Err(NodeReclaimPlacementError::DuplicateReclaim);
            }
            #[cfg(test)]
            if let Some((entered, release)) = self
                .inner()
                .reclaim_attach_hook
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                entered.send(()).unwrap();
                release.recv().unwrap();
            }
            payload.reclaim = Some(pending);
            slot.word.fetch_or(RECLAIMED_BIT, Ordering::Release);
            count += 1;
        }
    }

    /// Irreversibly seals the registry after the exact matching control admission/finalizer drain.
    pub(crate) fn seal_after_control_drain(
        mut self,
        drained: &DrainedControlClose,
    ) -> Result<SealedNodeLifetimeRegistry, NodeLifetimeSealFailure> {
        if !self.inner().control_identity.matches_drained(drained) {
            return Err(NodeLifetimeSealFailure {
                error: NodeLifetimeSealError::ForeignControlDrain,
                owner: self,
            });
        }
        let inner = self.inner();
        let registry_quarantined = match inner.phase.compare_exchange(
            RegistryPhase::Open as u8,
            RegistryPhase::Sealed as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => false,
            Err(phase) if RegistryPhase::from_u8(phase) == RegistryPhase::Quarantined => true,
            Err(_) => {
                return Err(NodeLifetimeSealFailure {
                    error: NodeLifetimeSealError::RegistryNotOpen,
                    owner: self,
                })
            }
        };
        let _allocation = inner
            .allocation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut degraded = registry_quarantined;
        for slot in &inner.slots {
            loop {
                let word = slot.word.load(Ordering::Acquire);
                let next = match SlotPhase::from_word(word) {
                    SlotPhase::Vacant | SlotPhase::Sealed => break,
                    SlotPhase::Quarantined => {
                        degraded = true;
                        break;
                    }
                    SlotPhase::Canceling
                    | SlotPhase::Servicing
                    | SlotPhase::AcceptedPending
                    | SlotPhase::Reconciling => {
                        degraded = true;
                        with_phase(word, SlotPhase::Quarantined)
                    }
                    SlotPhase::Provisional
                    | SlotPhase::Live
                    | SlotPhase::Requested
                    | SlotPhase::AwaitingReclaim => with_phase(word, SlotPhase::Sealed),
                };
                if slot
                    .word
                    .compare_exchange(word, next, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        }
        drop(_allocation);
        Ok(SealedNodeLifetimeRegistry {
            inner: self.inner.take(),
            request_wake: self.request_wake.take(),
            node_ids: self.node_ids.take(),
            control: self.control.take(),
            orphan_reclaim: self.orphan_reclaim.take(),
            degraded,
            registry_quarantined,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeLifetimeSealError {
    ForeignControlDrain,
    RegistryNotOpen,
}

pub(crate) struct NodeLifetimeSealFailure {
    pub(crate) error: NodeLifetimeSealError,
    pub(crate) owner: InjectedNodeLifetimeOwner,
}

impl Drop for InjectedNodeLifetimeOwner {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            {
                let _allocation = inner
                    .allocation
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let phase = RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire));
                if phase == RegistryPhase::Open {
                    inner
                        .phase
                        .store(RegistryPhase::Retired as u8, Ordering::Release);
                }
            }
            std::mem::forget(inner);
        }
        if let Some(node_ids) = self.node_ids.take() {
            std::mem::forget(node_ids);
        }
        if let Some(receiver) = self.request_wake.take() {
            std::mem::forget(receiver);
        }
        self.control.take();
        if let Some(reclaim) = self.orphan_reclaim.take() {
            std::mem::forget(reclaim);
        }
    }
}

/// Sealed registry retaining all bounded payloads for a later teardown driver or whole-graph
/// retirement. Dropping it without whole-graph retirement proof quarantines the remaining
/// resources for process lifetime.
#[must_use]
pub(crate) struct SealedNodeLifetimeRegistry {
    inner: Option<Arc<NodeLifetimeInner>>,
    request_wake: Option<crossbeam_channel::Receiver<()>>,
    node_ids: Option<InjectedNodeIdOwner>,
    control: Option<InjectedControlProducer>,
    orphan_reclaim: Option<OwnedPendingNodeReclaim>,
    degraded: bool,
    registry_quarantined: bool,
}

impl SealedNodeLifetimeRegistry {
    pub(crate) const fn degraded(&self) -> bool {
        self.degraded
    }

    pub(crate) const fn registry_quarantined(&self) -> bool {
        self.registry_quarantined
    }

    /// Releases all remaining bounded slot payloads only after a future lifecycle integration has
    /// proved the renderer, graph, and GC are retired. No production constructor for that proof
    /// exists in this slice.
    #[allow(clippy::result_large_err)] // a retry must retain the exact registry and proof unboxed
    pub(crate) fn retire_after_whole_graph(
        mut self,
        proof: WholeGraphRetired,
    ) -> Result<WholeGraphNodeRetirement, WholeGraphNodeRetireFailure> {
        let inner = self.inner.as_ref().unwrap();
        if !Weak::ptr_eq(&proof.registry_identity, &Arc::downgrade(inner))
            || !proof.control_identity.ptr_eq(&inner.control_identity)
            || !proof.node_id_identity.ptr_eq(&inner.node_id_identity)
        {
            return Err(WholeGraphNodeRetireFailure {
                error: WholeGraphNodeRetireError::ForeignProof,
                registry: self,
                proof,
            });
        }
        let inner = self.inner.take().unwrap();
        let inner = match Arc::try_unwrap(inner) {
            Ok(inner) => inner,
            Err(inner) => {
                self.inner = Some(inner);
                return Err(WholeGraphNodeRetireFailure {
                    error: WholeGraphNodeRetireError::ActiveRegistryUpgrade,
                    registry: self,
                    proof,
                });
            }
        };
        let mut node_ids = self.node_ids.take().unwrap();
        let mut cleanup_count = 0;
        let mut cleanup_panicked = false;
        let mut cleanup_rejected = false;
        let mut reclaim_brand_mismatch = false;
        for slot in &inner.slots {
            let id = AudioNodeId(slot.id.load(Ordering::Acquire));
            let (cleanup, reclaim) = {
                let mut payload = slot
                    .payload
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (payload.cleanup.take(), payload.reclaim.take())
            };
            if let Some(mut cleanup) = cleanup {
                cleanup_count += 1;
                match panic::catch_unwind(AssertUnwindSafe(|| cleanup.reconcile(id))) {
                    Ok(Ok(())) => cleanup_panicked |= !panic_safe_drop(cleanup),
                    Ok(Err(_)) => {
                        cleanup_rejected = true;
                        std::mem::forget(cleanup);
                    }
                    Err(payload) => {
                        std::mem::forget(payload);
                        std::mem::forget(cleanup);
                        cleanup_panicked = true;
                    }
                }
            }
            if let Some(reclaim) = reclaim {
                if let Err(reclaim) = node_ids.discard_after_whole_graph(reclaim) {
                    std::mem::forget(reclaim);
                    reclaim_brand_mismatch = true;
                }
            }
        }
        if let Some(reclaim) = self.orphan_reclaim.take() {
            if let Err(reclaim) = node_ids.discard_after_whole_graph(reclaim) {
                std::mem::forget(reclaim);
                reclaim_brand_mismatch = true;
            }
        }
        while let Some(reclaim) = node_ids.try_take_pending_reclaim() {
            if let Err(reclaim) = node_ids.discard_after_whole_graph(reclaim) {
                std::mem::forget(reclaim);
                reclaim_brand_mismatch = true;
                break;
            }
        }
        self.request_wake.take();
        self.control.take();
        drop(node_ids);
        drop(inner);
        Ok(WholeGraphNodeRetirement {
            cleanup_count,
            cleanup_panicked,
            cleanup_rejected,
            reclaim_brand_mismatch,
            pre_retirement_degraded: self.degraded,
        })
    }
}

impl Drop for SealedNodeLifetimeRegistry {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            std::mem::forget(inner);
        }
        if let Some(node_ids) = self.node_ids.take() {
            std::mem::forget(node_ids);
        }
        if let Some(receiver) = self.request_wake.take() {
            std::mem::forget(receiver);
        }
        self.control.take();
        if let Some(reclaim) = self.orphan_reclaim.take() {
            std::mem::forget(reclaim);
        }
    }
}

/// Future lifecycle proof that the injected renderer, whole graph, and mandatory GC are retired.
/// This slice intentionally provides no production constructor.
pub(crate) struct WholeGraphRetired {
    registry_identity: Weak<NodeLifetimeInner>,
    control_identity: InjectedControlIdentity,
    node_id_identity: InjectedNodeIdIdentity,
}

#[cfg(test)]
impl WholeGraphRetired {
    fn for_test(registry: &SealedNodeLifetimeRegistry) -> Self {
        let inner = registry.inner.as_ref().unwrap();
        Self {
            registry_identity: Arc::downgrade(inner),
            control_identity: inner.control_identity.clone(),
            node_id_identity: inner.node_id_identity.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WholeGraphNodeRetireError {
    ForeignProof,
    ActiveRegistryUpgrade,
}

pub(crate) struct WholeGraphNodeRetireFailure {
    pub(crate) error: WholeGraphNodeRetireError,
    pub(crate) registry: SealedNodeLifetimeRegistry,
    pub(crate) proof: WholeGraphRetired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WholeGraphNodeRetirement {
    pub(crate) cleanup_count: usize,
    pub(crate) cleanup_panicked: bool,
    pub(crate) cleanup_rejected: bool,
    pub(crate) reclaim_brand_mismatch: bool,
    /// Degradation already recorded while the registry was open or sealed. Callers must also
    /// include the cleanup and brand-mismatch fields when computing the final context outcome.
    pub(crate) pre_retirement_degraded: bool,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};
    use std::thread::{self, ThreadId};

    use super::*;
    use crate::context::injected_control::{
        injected_control_channel, InjectedControlLifecycleOwner, InjectedControlRenderInit,
    };
    use crate::context::injected_ids::{injected_node_id_pair, InjectedNodeIdAllocator};
    use crate::context::{AudioContextState, InjectedContextAdmissionGate};
    use crate::events::EventDispatch;
    use crate::output::{AudioOutputEventSink, AudioRenderFormat, EndpointShutdownConfirmed};
    use crate::stats::AudioStats;

    #[derive(Clone, Copy)]
    enum CleanupBehavior {
        Ok,
        Reject,
        Panic,
        DropPanic,
    }

    struct CleanupProbe {
        behavior: CleanupBehavior,
        reconciled: Arc<Mutex<Vec<(AudioNodeId, ThreadId)>>>,
        dropped: Arc<Mutex<Vec<ThreadId>>>,
    }

    type ProbeParts = (
        Box<dyn InjectedNodeReclaimCleanup>,
        Arc<Mutex<Vec<(AudioNodeId, ThreadId)>>>,
        Arc<Mutex<Vec<ThreadId>>>,
    );

    impl InjectedNodeReclaimCleanup for CleanupProbe {
        fn reconcile(&mut self, id: AudioNodeId) -> Result<(), NodeReclaimCleanupError> {
            self.reconciled
                .lock()
                .unwrap()
                .push((id, thread::current().id()));
            match self.behavior {
                CleanupBehavior::Ok | CleanupBehavior::DropPanic => Ok(()),
                CleanupBehavior::Reject => Err(NodeReclaimCleanupError::Rejected),
                CleanupBehavior::Panic => panic!("cleanup panic"),
            }
        }
    }

    impl Drop for CleanupProbe {
        fn drop(&mut self) {
            self.dropped.lock().unwrap().push(thread::current().id());
            if matches!(self.behavior, CleanupBehavior::DropPanic) {
                panic!("cleanup destructor panic");
            }
        }
    }

    fn probe(behavior: CleanupBehavior) -> ProbeParts {
        let reconciled = Arc::new(Mutex::new(Vec::new()));
        let dropped = Arc::new(Mutex::new(Vec::new()));
        (
            Box::new(CleanupProbe {
                behavior,
                reconciled: Arc::clone(&reconciled),
                dropped: Arc::clone(&dropped),
            }),
            reconciled,
            dropped,
        )
    }

    fn noop() -> Box<dyn InjectedNodeReclaimCleanup> {
        probe(CleanupBehavior::Ok).0
    }

    struct Foundation {
        producer: InjectedControlProducer,
        lifecycle: Option<InjectedControlLifecycleOwner>,
        _render_init: InjectedControlRenderInit,
        allocator: InjectedNodeIdAllocator,
        registrar: InjectedNodeLifetimeRegistrar,
        owner: Option<InjectedNodeLifetimeOwner>,
        graph: Option<InjectedGraphReclaimInit>,
    }

    impl Foundation {
        fn new(capacity: usize, first_id: u64) -> Self {
            let gate = InjectedContextAdmissionGate::new();
            let (producer, lifecycle, render_init) =
                injected_control_channel(gate, 4, false).unwrap();
            let (allocator, node_ids, graph) = injected_node_id_pair(first_id);
            let (registrar, bootstrap) =
                injected_node_lifetime_registry(capacity, &producer, node_ids, graph)
                    .ok()
                    .unwrap();
            let (owner, graph) = bootstrap.into_parts_for_test();
            Self {
                producer,
                lifecycle: Some(lifecycle),
                _render_init: render_init,
                allocator,
                registrar,
                owner: Some(owner),
                graph: Some(graph),
            }
        }

        fn drained(&mut self) -> DrainedControlClose {
            let retirement = self
                .lifecycle
                .take()
                .unwrap()
                .try_begin_close()
                .ok()
                .unwrap();
            retirement.retire_and_wait().1
        }

        fn seal(&mut self) -> SealedNodeLifetimeRegistry {
            let drained = self.drained();
            self.owner
                .take()
                .unwrap()
                .seal_after_control_drain(&drained)
                .ok()
                .unwrap()
        }
    }

    fn arm(registrar: &InjectedNodeLifetimeRegistrar, id: AudioNodeId) -> InjectedNodeRegistration {
        let provisional = registrar.try_register(id, noop()).ok().unwrap();
        provisional.arm_token().arm().unwrap();
        provisional.into_registration().unwrap()
    }

    #[test]
    fn bounded_build_returns_resources_and_rejects_mixed_id_graph() {
        let gate = InjectedContextAdmissionGate::new();
        let (producer, _lifecycle, _render) = injected_control_channel(gate, 1, false).unwrap();

        let (_allocator, owner, graph) = injected_node_id_pair(0);
        let failure = injected_node_lifetime_registry(
            DEFAULT_NODE_LIFETIME_CAPACITY + 1,
            &producer,
            owner,
            graph,
        )
        .err()
        .unwrap();
        assert_eq!(failure.error, NodeLifetimeBuildError::InvalidCapacity);
        assert!(failure.node_ids.matches_graph_init(&failure.graph));

        let (_allocator_a, owner_a, graph_a) = injected_node_id_pair(0);
        let (_allocator_b, owner_b, graph_b) = injected_node_id_pair(100);
        let failure = injected_node_lifetime_registry(1, &producer, owner_a, graph_b)
            .err()
            .unwrap();
        assert_eq!(failure.error, NodeLifetimeBuildError::MismatchedNodeIdGraph);
        assert!(!failure.node_ids.matches_graph_init(&failure.graph));
        let (_registrar_a, _bootstrap_a) =
            injected_node_lifetime_registry(1, &producer, failure.node_ids, graph_a)
                .ok()
                .unwrap();
        let (_registrar_b, _bootstrap_b) =
            injected_node_lifetime_registry(1, &producer, owner_b, failure.graph)
                .ok()
                .unwrap();
    }

    #[test]
    fn exhausted_slot_is_skipped_and_stale_generation_cannot_request_reuse() {
        let mut foundation = Foundation::new(2, 100);
        let inner = foundation.registrar.inner.upgrade().unwrap();
        inner.slots[0].word.store(
            slot_word(MAX_GENERATION, SlotPhase::Vacant, false),
            Ordering::Release,
        );

        let first = foundation
            .registrar
            .try_register(AudioNodeId(7), noop())
            .ok()
            .unwrap();
        assert_eq!(first.key.slot, 1);
        let stale_key = first.key;
        drop(first);

        let live = arm(&foundation.registrar, AudioNodeId(8));
        let current = inner.slots[1].word.load(Ordering::Acquire);
        assert_eq!(generation(current), 2);
        let stale = InjectedNodeRegistration {
            inner: Weak::clone(&foundation.registrar.inner),
            key: stale_key,
        };
        drop(stale);
        assert_eq!(
            SlotPhase::from_word(inner.slots[1].word.load(Ordering::Acquire)),
            SlotPhase::Live
        );
        drop(live);
        assert_eq!(
            SlotPhase::from_word(inner.slots[1].word.load(Ordering::Acquire)),
            SlotPhase::Requested
        );
        drop(inner);
        drop(foundation.owner.take());
    }

    #[test]
    fn provisional_cancel_frees_without_teardown_wake() {
        let foundation = Foundation::new(1, 100);
        let (cleanup, _reconciled, dropped) = probe(CleanupBehavior::Ok);
        let provisional = foundation
            .registrar
            .try_register(AudioNodeId(7), cleanup)
            .ok()
            .unwrap();
        let slot = provisional.key.slot;
        drop(provisional);
        assert_eq!(
            foundation
                .owner
                .as_ref()
                .unwrap()
                .request_activity_receiver()
                .try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        );
        assert_eq!(dropped.lock().unwrap().len(), 1);
        let inner = foundation.registrar.inner.upgrade().unwrap();
        assert_eq!(
            SlotPhase::from_word(inner.slots[slot].word.load(Ordering::Acquire)),
            SlotPhase::Vacant
        );
    }

    #[test]
    fn armed_drop_is_allocation_free_coalesced_and_disconnect_safe() {
        let mut foundation = Foundation::new(3, 100);
        let first = arm(&foundation.registrar, AudioNodeId(7));
        let second = arm(&foundation.registrar, AudioNodeId(8));
        alloc_counter::deny_alloc(|| drop(first));
        alloc_counter::deny_alloc(|| drop(second));
        let owner = foundation.owner.as_ref().unwrap();
        owner.request_activity_receiver().recv().unwrap();
        assert_eq!(
            owner.request_activity_receiver().try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        );

        let third = arm(&foundation.registrar, AudioNodeId(9));
        let receiver = foundation
            .owner
            .as_mut()
            .unwrap()
            .request_wake
            .take()
            .unwrap();
        drop(receiver);
        alloc_counter::deny_alloc(|| drop(third));
    }

    #[test]
    fn early_reclaims_are_slot_owned_without_head_of_line_or_reuse() {
        let mut foundation = Foundation::new(2, 100);
        let mut ids = foundation.allocator.try_reserve(2).unwrap();
        let provisional_id = ids.id(0);
        let live_id = ids.id(1);
        let provisional = foundation
            .registrar
            .try_register(provisional_id, noop())
            .ok()
            .unwrap();
        let live = arm(&foundation.registrar, live_id);
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(ids.take_reclaim_node(0).unwrap());
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(ids.take_reclaim_node(1).unwrap());
        ids.commit().unwrap();
        assert_eq!(foundation.owner.as_mut().unwrap().ingest_reclaims(), Ok(2));

        let provisional_word = foundation.owner.as_ref().unwrap().inner().slots
            [provisional.key.slot]
            .word
            .load(Ordering::Acquire);
        assert_eq!(
            SlotPhase::from_word(provisional_word),
            SlotPhase::Provisional
        );
        assert!(has_reclaim(provisional_word));
        provisional.arm_token().arm().unwrap();
        let first = provisional.into_registration().unwrap();
        drop(first);
        drop(live);
        for slot in &foundation.owner.as_ref().unwrap().inner().slots {
            let word = slot.word.load(Ordering::Acquire);
            assert_eq!(SlotPhase::from_word(word), SlotPhase::Requested);
            assert!(has_reclaim(word));
            assert!(slot.payload.lock().unwrap().reclaim.is_some());
        }

        let reserved = foundation.allocator.try_reserve(1).unwrap();
        assert_eq!(reserved.id(0), AudioNodeId(102));
    }

    #[test]
    fn reclaim_attachment_wins_provisional_cancel_without_losing_payload() {
        let mut foundation = Foundation::new(1, 100);
        let mut ids = foundation.allocator.try_reserve(1).unwrap();
        let id = ids.id(0);
        let (cleanup, _reconciled, dropped) = probe(CleanupBehavior::Ok);
        let provisional = foundation.registrar.try_register(id, cleanup).ok().unwrap();
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(ids.take_reclaim_node(0).unwrap());
        ids.commit().unwrap();

        let (attach_entered_send, attach_entered_recv) = crossbeam_channel::bounded(1);
        let (attach_release_send, attach_release_recv) = crossbeam_channel::bounded(1);
        foundation
            .owner
            .as_ref()
            .unwrap()
            .set_reclaim_attach_hook(attach_entered_send, attach_release_recv);
        let mut owner = foundation.owner.take().unwrap();
        let ingest = thread::spawn(move || {
            let result = owner.ingest_reclaims();
            (owner, result)
        });
        attach_entered_recv.recv().unwrap();

        let (cancel_done_send, cancel_done_recv) = crossbeam_channel::bounded(1);
        let cancel = thread::spawn(move || {
            drop(provisional);
            cancel_done_send.send(()).unwrap();
        });
        assert_eq!(
            cancel_done_recv.try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        );
        attach_release_send.send(()).unwrap();
        let (owner, result) = ingest.join().unwrap();
        assert_eq!(result, Ok(1));
        cancel_done_recv.recv().unwrap();
        cancel.join().unwrap();

        let slot = &owner.inner().slots[0];
        let word = slot.word.load(Ordering::Acquire);
        assert_eq!(SlotPhase::from_word(word), SlotPhase::Requested);
        assert!(has_reclaim(word));
        let payload = slot.payload.lock().unwrap();
        assert!(payload.cleanup.is_some());
        assert!(payload.reclaim.is_some());
        assert!(dropped.lock().unwrap().is_empty());
        drop(payload);
        owner.request_activity_receiver().recv().unwrap();
        foundation.owner = Some(owner);
    }

    #[test]
    fn accepted_arm_failure_cannot_recycle_a_sealed_provisional() {
        let mut foundation = Foundation::new(1, 100);
        let provisional = foundation
            .registrar
            .try_register(AudioNodeId(7), noop())
            .ok()
            .unwrap();
        let slot_index = provisional.key.slot;
        let sealed = foundation.seal();
        assert_eq!(
            provisional.arm_token().arm(),
            Err(AcceptedBatchFinalizeError::Rejected)
        );
        drop(provisional);
        let slot = &sealed.inner.as_ref().unwrap().slots[slot_index];
        assert_eq!(
            SlotPhase::from_word(slot.word.load(Ordering::Acquire)),
            SlotPhase::Sealed
        );
        assert!(slot.payload.lock().unwrap().cleanup.is_some());
        let proof = WholeGraphRetired::for_test(&sealed);
        assert!(sealed.retire_after_whole_graph(proof).is_ok());
    }

    #[test]
    fn cancel_vs_seal_preserves_seal_winner_and_armed_conversion_is_inert() {
        struct BlockingDrop {
            entered: crossbeam_channel::Sender<()>,
            release: crossbeam_channel::Receiver<()>,
        }
        impl InjectedNodeReclaimCleanup for BlockingDrop {
            fn reconcile(&mut self, _id: AudioNodeId) -> Result<(), NodeReclaimCleanupError> {
                Ok(())
            }
        }
        impl Drop for BlockingDrop {
            fn drop(&mut self) {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
        }

        let mut foundation = Foundation::new(2, 100);
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        let provisional = foundation
            .registrar
            .try_register(
                AudioNodeId(7),
                Box::new(BlockingDrop {
                    entered: entered_send,
                    release: release_recv,
                }),
            )
            .ok()
            .unwrap();
        let cancel_slot = provisional.key.slot;
        let cancel = thread::spawn(move || drop(provisional));
        entered_recv.recv().unwrap();

        let armed = foundation
            .registrar
            .try_register(AudioNodeId(8), noop())
            .ok()
            .unwrap();
        armed.arm_token().arm().unwrap();
        let drained = foundation.drained();
        let sealed = foundation
            .owner
            .take()
            .unwrap()
            .seal_after_control_drain(&drained)
            .ok()
            .unwrap();
        release_send.send(()).unwrap();
        cancel.join().unwrap();
        assert!(sealed.degraded());
        assert_eq!(
            SlotPhase::from_word(
                sealed.inner.as_ref().unwrap().slots[cancel_slot]
                    .word
                    .load(Ordering::Acquire)
            ),
            SlotPhase::Quarantined
        );
        let inert = armed.into_registration().unwrap();
        drop(inert);
        assert_eq!(
            SlotPhase::from_word(
                sealed.inner.as_ref().unwrap().slots[1]
                    .word
                    .load(Ordering::Acquire)
            ),
            SlotPhase::Sealed
        );
    }

    #[test]
    fn registration_and_quarantine_share_one_linearization_lock() {
        let mut foundation = Foundation::new(2, 100);
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        foundation
            .owner
            .as_ref()
            .unwrap()
            .set_registration_publish_hook(entered_send, release_recv);
        let registrar = foundation.registrar.clone();
        let register = thread::spawn(move || registrar.try_register(AudioNodeId(7), noop()));
        entered_recv.recv().unwrap();

        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(llq::Node::new(AudioNodeId(999)));
        let mut owner = foundation.owner.take().unwrap();
        let quarantine = thread::spawn(move || {
            let result = owner.ingest_reclaims();
            (owner, result)
        });
        release_send.send(()).unwrap();
        let provisional = register.join().unwrap().ok().unwrap();
        let (owner, result) = quarantine.join().unwrap();
        assert_eq!(result, Err(NodeReclaimPlacementError::ReclaimMismatch));
        assert_eq!(
            RegistryPhase::from_u8(owner.inner().phase.load(Ordering::Acquire)),
            RegistryPhase::Quarantined
        );
        assert_eq!(
            foundation
                .registrar
                .try_register(AudioNodeId(8), noop())
                .err()
                .unwrap()
                .error,
            NodeRegistrationError::Sealed
        );
        drop(provisional);
        foundation.owner = Some(owner);
    }

    #[test]
    fn registration_and_arm_linearize_before_owner_retirement() {
        let mut foundation = Foundation::new(2, 100);
        let (register_entered_send, register_entered_recv) = crossbeam_channel::bounded(1);
        let (register_release_send, register_release_recv) = crossbeam_channel::bounded(1);
        foundation
            .owner
            .as_ref()
            .unwrap()
            .set_registration_publish_hook(register_entered_send, register_release_recv);
        let registrar = foundation.registrar.clone();
        let register = thread::spawn(move || registrar.try_register(AudioNodeId(7), noop()));
        register_entered_recv.recv().unwrap();
        let owner = foundation.owner.take().unwrap();
        let (retire_done_send, retire_done_recv) = crossbeam_channel::bounded(1);
        let retire = thread::spawn(move || {
            drop(owner);
            retire_done_send.send(()).unwrap();
        });
        assert_eq!(
            retire_done_recv.try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        );
        register_release_send.send(()).unwrap();
        let provisional = register.join().unwrap().ok().unwrap();
        retire_done_recv.recv().unwrap();
        retire.join().unwrap();
        assert_eq!(
            foundation
                .registrar
                .try_register(AudioNodeId(8), noop())
                .err()
                .unwrap()
                .error,
            NodeRegistrationError::Sealed
        );
        assert_eq!(
            provisional.arm_token().arm(),
            Err(AcceptedBatchFinalizeError::Rejected)
        );
        drop(provisional);

        let mut second = Foundation::new(1, 200);
        let provisional = second
            .registrar
            .try_register(AudioNodeId(9), noop())
            .ok()
            .unwrap();
        let (arm_entered_send, arm_entered_recv) = crossbeam_channel::bounded(1);
        let (arm_release_send, arm_release_recv) = crossbeam_channel::bounded(1);
        second
            .owner
            .as_ref()
            .unwrap()
            .set_arm_publish_hook(arm_entered_send, arm_release_recv);
        let arm = thread::spawn(move || {
            let result = provisional.arm_token().arm();
            (provisional, result)
        });
        arm_entered_recv.recv().unwrap();
        let owner = second.owner.take().unwrap();
        let (retire_done_send, retire_done_recv) = crossbeam_channel::bounded(1);
        let retire = thread::spawn(move || {
            drop(owner);
            retire_done_send.send(()).unwrap();
        });
        assert_eq!(
            retire_done_recv.try_recv(),
            Err(crossbeam_channel::TryRecvError::Empty)
        );
        arm_release_send.send(()).unwrap();
        let (provisional, result) = arm.join().unwrap();
        assert_eq!(result, Ok(()));
        retire_done_recv.recv().unwrap();
        retire.join().unwrap();
        let registration = provisional.into_registration().unwrap();
        alloc_counter::deny_alloc(|| drop(registration));
    }

    #[test]
    fn foreign_control_drain_returns_owner_then_exact_drain_seals() {
        let mut foundation = Foundation::new(1, 100);
        let gate = InjectedContextAdmissionGate::new();
        let (_foreign_producer, foreign_lifecycle, _foreign_render) =
            injected_control_channel(gate, 1, false).unwrap();
        let foreign = foreign_lifecycle
            .try_begin_close()
            .ok()
            .unwrap()
            .retire_and_wait()
            .1;
        let failure = foundation
            .owner
            .take()
            .unwrap()
            .seal_after_control_drain(&foreign)
            .err()
            .unwrap();
        assert_eq!(failure.error, NodeLifetimeSealError::ForeignControlDrain);
        foundation.owner = Some(failure.owner);
        let sealed = foundation.seal();
        assert!(!sealed.degraded());
    }

    #[test]
    fn reclaim_mismatch_remains_quarantined_through_seal() {
        let mut foundation = Foundation::new(1, 100);
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(llq::Node::new(AudioNodeId(999)));
        assert_eq!(
            foundation.owner.as_mut().unwrap().ingest_reclaims(),
            Err(NodeReclaimPlacementError::ReclaimMismatch)
        );
        let sealed = foundation.seal();
        assert!(sealed.degraded());
        assert!(sealed.registry_quarantined());
    }

    #[test]
    fn duplicate_reclaim_quarantines_and_stops_before_later_exact_nodes() {
        let mut foundation = Foundation::new(2, 100);
        let mut ids = foundation.allocator.try_reserve(2).unwrap();
        let first_id = ids.id(0);
        let second_id = ids.id(1);
        let _first = arm(&foundation.registrar, first_id);
        let _second = arm(&foundation.registrar, second_id);

        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(ids.take_reclaim_node(0).unwrap());
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(llq::Node::new(first_id));
        foundation
            .graph
            .as_mut()
            .unwrap()
            .push_for_test(ids.take_reclaim_node(1).unwrap());
        ids.commit().unwrap();

        assert_eq!(
            foundation.owner.as_mut().unwrap().ingest_reclaims(),
            Err(NodeReclaimPlacementError::DuplicateReclaim)
        );
        let owner = foundation.owner.as_ref().unwrap();
        assert_eq!(
            RegistryPhase::from_u8(owner.inner().phase.load(Ordering::Acquire)),
            RegistryPhase::Quarantined
        );
        assert!(owner.orphan_reclaim.is_some());
        let first_slot = owner
            .inner()
            .slots
            .iter()
            .find(|slot| slot.id.load(Ordering::Acquire) == first_id.0)
            .unwrap();
        assert!(has_reclaim(first_slot.word.load(Ordering::Acquire)));
        assert!(first_slot.payload.lock().unwrap().reclaim.is_some());
        let second_slot = owner
            .inner()
            .slots
            .iter()
            .find(|slot| slot.id.load(Ordering::Acquire) == second_id.0)
            .unwrap();
        assert!(!has_reclaim(second_slot.word.load(Ordering::Acquire)));
        assert!(second_slot.payload.lock().unwrap().reclaim.is_none());

        let sealed = foundation.seal();
        assert!(sealed.degraded());
        assert!(sealed.registry_quarantined());
        let proof = WholeGraphRetired::for_test(&sealed);
        assert!(sealed.retire_after_whole_graph(proof).is_ok());
    }

    #[test]
    fn provisional_cleanup_panic_is_reported_as_degraded_at_seal() {
        let mut foundation = Foundation::new(1, 100);
        let (cleanup, _reconciled, dropped) = probe(CleanupBehavior::DropPanic);
        let provisional = foundation
            .registrar
            .try_register(AudioNodeId(7), cleanup)
            .ok()
            .unwrap();
        drop(provisional);
        assert_eq!(dropped.lock().unwrap().len(), 1);
        let sealed = foundation.seal();
        assert!(sealed.degraded());
        let proof = WholeGraphRetired::for_test(&sealed);
        let report = sealed.retire_after_whole_graph(proof).ok().unwrap();
        assert!(report.pre_retirement_degraded);
    }

    #[test]
    fn foreign_whole_graph_proof_returns_registry_and_proof_intact() {
        let mut first = Foundation::new(1, 100);
        let mut second = Foundation::new(1, 200);
        let first = first.seal();
        let second = second.seal();
        let foreign = WholeGraphRetired::for_test(&second);
        let failure = first.retire_after_whole_graph(foreign).err().unwrap();
        assert_eq!(failure.error, WholeGraphNodeRetireError::ForeignProof);
        assert!(Weak::ptr_eq(
            &failure.proof.registry_identity,
            &Arc::downgrade(second.inner.as_ref().unwrap())
        ));
        assert!(!Weak::ptr_eq(
            &failure.proof.registry_identity,
            &Arc::downgrade(failure.registry.inner.as_ref().unwrap())
        ));

        let exact = WholeGraphRetired::for_test(&failure.registry);
        assert!(failure.registry.retire_after_whole_graph(exact).is_ok());
        assert!(second.retire_after_whole_graph(failure.proof).is_ok());
    }

    #[test]
    fn retirement_retries_active_upgrade_then_destroys_on_lifecycle_thread() {
        let mut foundation = Foundation::new(1, 100);
        let (_cleanup, reconciled, dropped) = probe(CleanupBehavior::Ok);
        let provisional = foundation
            .registrar
            .try_register(AudioNodeId(7), _cleanup)
            .ok()
            .unwrap();
        provisional.arm_token().arm().unwrap();
        let _live = provisional.into_registration().unwrap();
        let sealed = foundation.seal();
        let active_upgrade = foundation.registrar.inner.upgrade().unwrap();
        let proof = WholeGraphRetired::for_test(&sealed);
        let failure = sealed.retire_after_whole_graph(proof).err().unwrap();
        assert_eq!(
            failure.error,
            WholeGraphNodeRetireError::ActiveRegistryUpgrade
        );
        drop(active_upgrade);

        let lifecycle_thread = thread::spawn(move || {
            let thread_id = thread::current().id();
            let report = failure
                .registry
                .retire_after_whole_graph(failure.proof)
                .ok()
                .unwrap();
            (thread_id, report)
        });
        let (thread_id, report) = lifecycle_thread.join().unwrap();
        assert_eq!(report.cleanup_count, 1);
        assert_eq!(reconciled.lock().unwrap()[0], (AudioNodeId(7), thread_id));
        assert_eq!(dropped.lock().unwrap().as_slice(), &[thread_id]);
    }

    #[test]
    fn whole_graph_cleanup_reject_panic_and_drop_panic_are_contained_off_thread() {
        let mut foundation = Foundation::new(3, 100);
        let mut records = Vec::new();
        for (index, behavior) in [
            CleanupBehavior::Reject,
            CleanupBehavior::Panic,
            CleanupBehavior::DropPanic,
        ]
        .into_iter()
        .enumerate()
        {
            let (cleanup, reconciled, dropped) = probe(behavior);
            let provisional = foundation
                .registrar
                .try_register(AudioNodeId(index as u64 + 7), cleanup)
                .ok()
                .unwrap();
            provisional.arm_token().arm().unwrap();
            let _ = provisional.into_registration().unwrap();
            records.push((reconciled, dropped));
        }
        let sealed = foundation.seal();
        let proof = WholeGraphRetired::for_test(&sealed);
        let (thread_id, report) = thread::spawn(move || {
            let thread_id = thread::current().id();
            let report = sealed.retire_after_whole_graph(proof).ok().unwrap();
            (thread_id, report)
        })
        .join()
        .unwrap();
        assert_eq!(report.cleanup_count, 3);
        assert!(report.cleanup_rejected);
        assert!(report.cleanup_panicked);
        for (reconciled, _) in &records {
            assert_eq!(reconciled.lock().unwrap()[0].1, thread_id);
        }
        // Rejected and method-panicking hooks are deliberately leaked with their host guards.
        assert!(records[0].1.lock().unwrap().is_empty());
        assert!(records[1].1.lock().unwrap().is_empty());
        assert_eq!(records[2].1.lock().unwrap().as_slice(), &[thread_id]);
    }

    #[test]
    fn owner_retirement_makes_weak_provisional_and_registration_inert() {
        let mut foundation = Foundation::new(2, 100);
        let provisional = foundation
            .registrar
            .try_register(AudioNodeId(7), noop())
            .ok()
            .unwrap();
        let live = arm(&foundation.registrar, AudioNodeId(8));
        drop(foundation.owner.take());
        assert_eq!(
            provisional.arm_token().arm(),
            Err(AcceptedBatchFinalizeError::Rejected)
        );
        assert_eq!(
            foundation
                .registrar
                .try_register(AudioNodeId(9), noop())
                .err()
                .unwrap()
                .error,
            NodeRegistrationError::Sealed
        );
        alloc_counter::deny_alloc(|| drop(live));
        drop(provisional);
    }

    #[test]
    fn render_bootstrap_rejects_foreign_control_and_returns_exact_owner() {
        let gate_a = InjectedContextAdmissionGate::new();
        let (producer_a, _lifecycle_a, init_a) =
            injected_control_channel(gate_a, 1, false).unwrap();
        let gate_b = InjectedContextAdmissionGate::new();
        let (_producer_b, _lifecycle_b, init_b) =
            injected_control_channel(gate_b, 1, false).unwrap();
        let (_allocator, node_ids, graph) = injected_node_id_pair(0);
        let (_registrar, bootstrap) =
            injected_node_lifetime_registry(1, &producer_a, node_ids, graph)
                .ok()
                .unwrap();
        let (event_sender, _event_receiver) = crossbeam_channel::bounded::<EventDispatch>(1);
        let failure = init_b
            .build_render_thread(
                bootstrap,
                48_000.,
                2,
                Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
                Arc::new(AtomicU64::new(0)),
                AudioStats::new(),
                event_sender.clone(),
            )
            .err()
            .unwrap();
        let bound = init_a
            .build_render_thread(
                failure.node_lifetimes,
                48_000.,
                2,
                Arc::new(AtomicU8::new(AudioContextState::Suspended as u8)),
                Arc::new(AtomicU64::new(0)),
                AudioStats::new(),
                event_sender,
            )
            .ok()
            .unwrap();
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
        drop(node_lifetimes);
        drop(failure.init);
    }
}
