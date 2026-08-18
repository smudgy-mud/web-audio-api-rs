//! Caller-driven requested teardown and exact ordinary reclaim reconciliation.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;

use super::*;
use crate::context::injected_control::{
    CommitControlOutcome, CommitWithFinalizeFailure, InjectedControlError,
};

mod reconcile;

/// Maximum idle wait recommended for a lifecycle select loop. Request, reclaim, and credit wakes
/// are deliberately lossy hints; this timeout makes authoritative rescans progress after a full or
/// disconnected wake channel and after structural `try_lock` contention.
pub(crate) const NODE_LIFETIME_RETRY_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeLifetimeActivity {
    Request,
    Reclaim,
    Credit,
    Disconnected,
    Timeout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeLifetimeRetryReason {
    Credit,
    Contended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeLifetimeQuarantineReason {
    ReclaimMismatch,
    DuplicateReclaim,
    Transport(InjectedControlError),
    AcceptedFinalizer,
    ProtocolViolation,
    CleanupRejected,
    CleanupPanicked,
    CleanupDestructorPanicked,
    ReclaimBrandMismatch,
    RegistryPoisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeLifetimeDriveOutcome {
    Idle,
    Submitted {
        id: AudioNodeId,
        outcome: CommitControlOutcome,
    },
    Reconciled {
        id: AudioNodeId,
    },
    Retry {
        id: AudioNodeId,
        reason: NodeLifetimeRetryReason,
    },
    CloseSuperseded {
        id: AudioNodeId,
    },
    Quarantined {
        id: Option<AudioNodeId>,
        reason: NodeLifetimeQuarantineReason,
    },
}

enum ControlErrorClass {
    Credit,
    Contended,
    CloseSuperseded,
    Terminal,
}

fn classify_begin_error(error: InjectedControlError) -> ControlErrorClass {
    match error {
        InjectedControlError::LogicalCommandCredits
        | InjectedControlError::BatchStorageCredits
        | InjectedControlError::OrdinaryPhysicalCredits
        | InjectedControlError::StagingFull => ControlErrorClass::Credit,
        InjectedControlError::Contended => ControlErrorClass::Contended,
        InjectedControlError::Sealed => ControlErrorClass::CloseSuperseded,
        _ => ControlErrorClass::Terminal,
    }
}

fn classify_commit_error(error: InjectedControlError) -> ControlErrorClass {
    match error {
        InjectedControlError::Contended => ControlErrorClass::Contended,
        // Credits were already reserved and a close cannot seal while their admission is live.
        // Every other result is therefore a terminal transport/protocol failure.
        _ => ControlErrorClass::Terminal,
    }
}

impl InjectedNodeLifetimeOwner {
    #[cfg(test)]
    pub(crate) fn disconnect_request_activity_for_test(&mut self) {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        drop(sender);
        self.request_wake = Some(receiver);
    }

    /// Borrowed request hint for a lifecycle select loop. Slot scans remain authoritative.
    pub(crate) fn request_activity_receiver(&self) -> &crossbeam_channel::Receiver<()> {
        self.request_wake
            .as_ref()
            .expect("live node-lifetime owner retains its request wake")
    }

    /// Borrowed graph-reclaim hint for a lifecycle select loop. The exact intrusive-node queue is
    /// authoritative and must be drained by [`Self::try_drive_once`].
    pub(crate) fn reclaim_activity_receiver(&self) -> &crossbeam_channel::Receiver<()> {
        self.node_ids
            .as_ref()
            .expect("live node-lifetime owner retains its id owner")
            .reclaim_activity_receiver()
    }

    /// Waits on all three lossy hints for at most [`NODE_LIFETIME_RETRY_INTERVAL`]. Receivers found
    /// disconnected by the initial probes are omitted, preventing an always-ready disconnected
    /// operation from spinning. [`NodeLifetimeActivity::Disconnected`] reports only a receiver that
    /// disconnects after that probe and wins the select. Callers must run authoritative work both
    /// before and after waiting regardless of the returned hint.
    pub(crate) fn wait_for_activity(
        &self,
        credit: &crossbeam_channel::Receiver<()>,
    ) -> NodeLifetimeActivity {
        let request = self.request_activity_receiver();
        let reclaim = self.reclaim_activity_receiver();
        let request_active = match request.try_recv() {
            Ok(()) => return NodeLifetimeActivity::Request,
            Err(crossbeam_channel::TryRecvError::Empty) => true,
            Err(crossbeam_channel::TryRecvError::Disconnected) => false,
        };
        let reclaim_active = match reclaim.try_recv() {
            Ok(()) => return NodeLifetimeActivity::Reclaim,
            Err(crossbeam_channel::TryRecvError::Empty) => true,
            Err(crossbeam_channel::TryRecvError::Disconnected) => false,
        };
        let credit_active = match credit.try_recv() {
            Ok(()) => return NodeLifetimeActivity::Credit,
            Err(crossbeam_channel::TryRecvError::Empty) => true,
            Err(crossbeam_channel::TryRecvError::Disconnected) => false,
        };

        let mut select = crossbeam_channel::Select::new();
        let request_index = request_active.then(|| select.recv(request));
        let reclaim_index = reclaim_active.then(|| select.recv(reclaim));
        let credit_index = credit_active.then(|| select.recv(credit));
        if request_index.is_none() && reclaim_index.is_none() && credit_index.is_none() {
            std::thread::park_timeout(NODE_LIFETIME_RETRY_INTERVAL);
            return NodeLifetimeActivity::Timeout;
        }
        let Ok(operation) = select.select_timeout(NODE_LIFETIME_RETRY_INTERVAL) else {
            return NodeLifetimeActivity::Timeout;
        };
        let index = operation.index();
        if request_index == Some(index) {
            return operation
                .recv(request)
                .map_or(NodeLifetimeActivity::Disconnected, |_| {
                    NodeLifetimeActivity::Request
                });
        }
        if reclaim_index == Some(index) {
            return operation
                .recv(reclaim)
                .map_or(NodeLifetimeActivity::Disconnected, |_| {
                    NodeLifetimeActivity::Reclaim
                });
        }
        debug_assert_eq!(credit_index, Some(index));
        operation
            .recv(credit)
            .map_or(NodeLifetimeActivity::Disconnected, |_| {
                NodeLifetimeActivity::Credit
            })
    }

    /// Performs one bounded unit of caller-driven lifetime work. This is lifecycle/control-thread
    /// work: reconciliation may invoke arbitrary cleanup code and destructors, so it must never run
    /// on a render or callback thread. Callers should repeat until `Idle` or `Retry`, selecting on
    /// request/reclaim/control-credit hints plus [`NODE_LIFETIME_RETRY_INTERVAL`] before rescanning.
    pub(crate) fn try_drive_once(&mut self) -> NodeLifetimeDriveOutcome {
        if let Err(error) = self.ingest_reclaims() {
            return NodeLifetimeDriveOutcome::Quarantined {
                id: self
                    .orphan_reclaim
                    .as_ref()
                    .map(OwnedPendingNodeReclaim::id),
                reason: match error {
                    NodeReclaimPlacementError::ReclaimMismatch => {
                        NodeLifetimeQuarantineReason::ReclaimMismatch
                    }
                    NodeReclaimPlacementError::DuplicateReclaim => {
                        NodeLifetimeQuarantineReason::DuplicateReclaim
                    }
                },
            };
        }

        match RegistryPhase::from_u8(self.inner().phase.load(Ordering::Acquire)) {
            RegistryPhase::Open => {}
            RegistryPhase::Quarantined => {
                return NodeLifetimeDriveOutcome::Quarantined {
                    id: None,
                    reason: NodeLifetimeQuarantineReason::ProtocolViolation,
                };
            }
            RegistryPhase::Sealed | RegistryPhase::Retired => {
                return NodeLifetimeDriveOutcome::Idle;
            }
        }

        if let Some(key) = self.find_ready_reconcile() {
            return self.reconcile_key(key);
        }
        let Some(key) = self.find_requested() else {
            return NodeLifetimeDriveOutcome::Idle;
        };
        self.service_key(key)
    }

    fn find_ready_reconcile(&self) -> Option<RegistrationKey> {
        self.inner()
            .slots
            .iter()
            .enumerate()
            .find_map(|(slot, entry)| {
                let word = entry.word.load(Ordering::Acquire);
                (has_reclaim(word)
                    && matches!(
                        SlotPhase::from_word(word),
                        SlotPhase::Requested | SlotPhase::AwaitingReclaim
                    ))
                .then(|| RegistrationKey {
                    slot,
                    generation: NonZeroU64::new(generation(word)).unwrap(),
                })
            })
    }

    fn find_requested(&self) -> Option<RegistrationKey> {
        self.inner()
            .slots
            .iter()
            .enumerate()
            .find_map(|(slot, entry)| {
                let word = entry.word.load(Ordering::Acquire);
                (!has_reclaim(word) && SlotPhase::from_word(word) == SlotPhase::Requested).then(
                    || RegistrationKey {
                        slot,
                        generation: NonZeroU64::new(generation(word)).unwrap(),
                    },
                )
            })
    }

    fn service_key(&mut self, key: RegistrationKey) -> NodeLifetimeDriveOutcome {
        let id = AudioNodeId(self.inner().slots[key.slot].id.load(Ordering::Acquire));
        let control = self
            .control
            .as_ref()
            .expect("live lifetime owner retains its exact control producer");
        let reservation = match control.try_begin_control_handle_drop() {
            Ok(reservation) => reservation,
            Err(error) => {
                return match classify_begin_error(error) {
                    ControlErrorClass::Credit => NodeLifetimeDriveOutcome::Retry {
                        id,
                        reason: NodeLifetimeRetryReason::Credit,
                    },
                    ControlErrorClass::Contended => NodeLifetimeDriveOutcome::Retry {
                        id,
                        reason: NodeLifetimeRetryReason::Contended,
                    },
                    ControlErrorClass::CloseSuperseded => {
                        NodeLifetimeDriveOutcome::CloseSuperseded { id }
                    }
                    ControlErrorClass::Terminal => {
                        quarantine_slot(self.inner(), key);
                        NodeLifetimeDriveOutcome::Quarantined {
                            id: Some(id),
                            reason: NodeLifetimeQuarantineReason::Transport(error),
                        }
                    }
                };
            }
        };

        let Some(mut claim) = ServiceClaim::try_new(self.inner(), key) else {
            drop(reservation);
            return NodeLifetimeDriveOutcome::Retry {
                id,
                reason: NodeLifetimeRetryReason::Contended,
            };
        };
        let batch = reservation.prepare(id);
        if !claim.mark_accepted_pending() {
            drop(batch);
            claim.quarantine();
            return NodeLifetimeDriveOutcome::Quarantined {
                id: Some(id),
                reason: NodeLifetimeQuarantineReason::ProtocolViolation,
            };
        }
        // From this point an unwind may have crossed the acceptance boundary. The claim therefore
        // fails closed unless a typed NotAccepted result explicitly restores it.
        claim.fail_closed_on_drop();
        #[cfg(test)]
        claim.run_precommit_hook();
        let finalizer = claim.finalizer();
        let commit = control.try_commit_with_finalize(batch, move |_| finalizer.finish());
        match commit {
            Ok(outcome) => {
                claim.complete_accepted();
                NodeLifetimeDriveOutcome::Submitted { id, outcome }
            }
            Err(CommitWithFinalizeFailure::NotAccepted(failure)) => {
                let error = failure.error;
                match classify_commit_error(error) {
                    ControlErrorClass::Contended => {
                        drop(failure.batch);
                        #[cfg(test)]
                        claim.run_before_restore_hook();
                        claim.restore_requested();
                        NodeLifetimeDriveOutcome::Retry {
                            id,
                            reason: NodeLifetimeRetryReason::Contended,
                        }
                    }
                    _ => {
                        claim.quarantine();
                        drop(failure.batch);
                        NodeLifetimeDriveOutcome::Quarantined {
                            id: Some(id),
                            reason: NodeLifetimeQuarantineReason::Transport(error),
                        }
                    }
                }
            }
            Err(CommitWithFinalizeFailure::AcceptedFinalizer(_)) => {
                claim.quarantine();
                NodeLifetimeDriveOutcome::Quarantined {
                    id: Some(id),
                    reason: NodeLifetimeQuarantineReason::AcceptedFinalizer,
                }
            }
        }
    }
}

enum ClaimDrop {
    Restore,
    Quarantine,
    Complete,
}

struct ServiceClaim {
    inner: Weak<NodeLifetimeInner>,
    key: RegistrationKey,
    on_drop: ClaimDrop,
}

impl ServiceClaim {
    fn try_new(inner: &Arc<NodeLifetimeInner>, key: RegistrationKey) -> Option<Self> {
        let slot = &inner.slots[key.slot];
        let word = slot.word.load(Ordering::Acquire);
        (generation(word) == key.generation.get()
            && !has_reclaim(word)
            && SlotPhase::from_word(word) == SlotPhase::Requested
            && slot
                .word
                .compare_exchange(
                    word,
                    with_phase(word, SlotPhase::Servicing),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok())
        .then(|| Self {
            inner: Arc::downgrade(inner),
            key,
            on_drop: ClaimDrop::Restore,
        })
    }

    fn mark_accepted_pending(&self) -> bool {
        transition_key(
            &self.inner,
            self.key,
            SlotPhase::Servicing,
            SlotPhase::AcceptedPending,
        )
    }

    fn fail_closed_on_drop(&mut self) {
        self.on_drop = ClaimDrop::Quarantine;
    }

    fn finalizer(&self) -> AcceptedTeardownFinalizer<'_> {
        AcceptedTeardownFinalizer {
            inner: &self.inner,
            key: self.key,
        }
    }

    fn restore_requested(&mut self) {
        if !transition_key_any(
            &self.inner,
            self.key,
            &[SlotPhase::Servicing, SlotPhase::AcceptedPending],
            SlotPhase::Requested,
        ) {
            self.quarantine();
        } else {
            self.on_drop = ClaimDrop::Complete;
        }
    }

    fn quarantine(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            quarantine_slot(&inner, self.key);
        }
        self.on_drop = ClaimDrop::Complete;
    }

    fn complete_accepted(&mut self) {
        self.on_drop = ClaimDrop::Complete;
    }

    #[cfg(test)]
    fn run_precommit_hook(&self) {
        self.run_hook(|inner| &inner.teardown_precommit_hook);
    }

    #[cfg(test)]
    fn run_before_restore_hook(&self) {
        self.run_hook(|inner| &inner.teardown_before_restore_hook);
    }

    #[cfg(test)]
    fn run_hook(
        &self,
        select: impl FnOnce(
            &NodeLifetimeInner,
        ) -> &std::sync::Mutex<
            Option<(
                crossbeam_channel::Sender<()>,
                crossbeam_channel::Receiver<()>,
            )>,
        >,
    ) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let hook = select(&inner)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some((entered, release)) = hook {
            entered.send(()).unwrap();
            release.recv().unwrap();
        }
    }
}

impl Drop for ServiceClaim {
    fn drop(&mut self) {
        match self.on_drop {
            ClaimDrop::Restore => self.restore_requested(),
            ClaimDrop::Quarantine => self.quarantine(),
            ClaimDrop::Complete => {}
        }
    }
}

#[derive(Clone, Copy)]
struct AcceptedTeardownFinalizer<'a> {
    inner: &'a Weak<NodeLifetimeInner>,
    key: RegistrationKey,
}

impl AcceptedTeardownFinalizer<'_> {
    fn finish(self) -> Result<(), AcceptedBatchFinalizeError> {
        let Some(inner) = self.inner.upgrade() else {
            return Err(AcceptedBatchFinalizeError::Rejected);
        };
        if RegistryPhase::from_u8(inner.phase.load(Ordering::Acquire)) != RegistryPhase::Open {
            return Err(AcceptedBatchFinalizeError::Rejected);
        }
        #[cfg(test)]
        inner.run_teardown_finalizer_hook()?;
        transition_key(
            self.inner,
            self.key,
            SlotPhase::AcceptedPending,
            SlotPhase::AwaitingReclaim,
        )
        .then_some(())
        .ok_or(AcceptedBatchFinalizeError::Rejected)
    }
}

#[cfg(test)]
impl NodeLifetimeInner {
    fn run_teardown_finalizer_hook(&self) -> Result<(), AcceptedBatchFinalizeError> {
        if let Some((entered, release)) = self
            .teardown_finalizer_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            entered.send(()).unwrap();
            release.recv().unwrap();
        }
        match self.teardown_finalizer_behavior.swap(0, Ordering::AcqRel) {
            0 => Ok(()),
            1 => Err(AcceptedBatchFinalizeError::Rejected),
            2 => panic!("teardown finalizer panic"),
            _ => unreachable!("private teardown finalizer behavior"),
        }
    }
}

fn transition_key(
    inner: &Weak<NodeLifetimeInner>,
    key: RegistrationKey,
    from: SlotPhase,
    to: SlotPhase,
) -> bool {
    transition_key_any(inner, key, &[from], to)
}

fn transition_key_any(
    inner: &Weak<NodeLifetimeInner>,
    key: RegistrationKey,
    from: &[SlotPhase],
    to: SlotPhase,
) -> bool {
    let Some(inner) = inner.upgrade() else {
        return false;
    };
    let slot = &inner.slots[key.slot];
    let mut word = slot.word.load(Ordering::Acquire);
    loop {
        if generation(word) != key.generation.get() || !from.contains(&SlotPhase::from_word(word)) {
            return false;
        }
        match slot.word.compare_exchange(
            word,
            with_phase(word, to),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(observed) => word = observed,
        }
    }
}

fn quarantine_registry_locked(inner: &NodeLifetimeInner) {
    let _ = inner.phase.compare_exchange(
        RegistryPhase::Open as u8,
        RegistryPhase::Quarantined as u8,
        Ordering::AcqRel,
        Ordering::Acquire,
    );
}

pub(super) fn quarantine_slot(inner: &Arc<NodeLifetimeInner>, key: RegistrationKey) {
    let _allocation = inner
        .allocation
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let slot = &inner.slots[key.slot];
    let mut word = slot.word.load(Ordering::Acquire);
    loop {
        if generation(word) != key.generation.get()
            || matches!(
                SlotPhase::from_word(word),
                SlotPhase::Vacant | SlotPhase::Sealed | SlotPhase::Quarantined
            )
        {
            break;
        }
        match slot.word.compare_exchange(
            word,
            with_phase(word, SlotPhase::Quarantined),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,
            Err(observed) => word = observed,
        }
    }
    quarantine_registry_locked(inner);
}
