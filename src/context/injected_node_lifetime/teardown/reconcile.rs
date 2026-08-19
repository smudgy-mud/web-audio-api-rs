//! Exact ordinary reclaim reconciliation, separated from submission state transitions.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::*;

impl InjectedNodeLifetimeOwner {
    pub(super) fn reconcile_key(&mut self, key: RegistrationKey) -> NodeLifetimeDriveOutcome {
        let inner = Arc::clone(self.inner());
        let slot = &inner.slots[key.slot];
        let id = AudioNodeId(slot.id.load(Ordering::Acquire));
        let mut word = slot.word.load(Ordering::Acquire);
        let prior_phase;
        loop {
            if generation(word) != key.generation.get()
                || !has_reclaim(word)
                || !matches!(
                    SlotPhase::from_word(word),
                    SlotPhase::Requested | SlotPhase::AwaitingReclaim
                )
            {
                return NodeLifetimeDriveOutcome::Retry {
                    id,
                    reason: NodeLifetimeRetryReason::Contended,
                };
            }
            match slot.word.compare_exchange(
                word,
                with_phase(word, SlotPhase::Reconciling),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    prior_phase = SlotPhase::from_word(word);
                    break;
                }
                Err(observed) => word = observed,
            }
        }

        let (mut cleanup, reclaim) = {
            let mut payload = slot
                .payload
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let (Some(cleanup), Some(reclaim)) = (payload.cleanup.take(), payload.reclaim.take())
            else {
                drop(payload);
                quarantine_slot(&inner, key);
                return NodeLifetimeDriveOutcome::Quarantined {
                    id: Some(id),
                    reason: NodeLifetimeQuarantineReason::ProtocolViolation,
                };
            };
            (cleanup, reclaim)
        };

        match panic::catch_unwind(AssertUnwindSafe(|| cleanup.reconcile(id))) {
            Ok(Ok(())) => {}
            Ok(Err(NodeReclaimCleanupError::RetryContended)) => {
                if self.restore_reconcile_retry(&inner, key, prior_phase, cleanup, reclaim) {
                    let _ = inner.request_wake.try_send(());
                    return NodeLifetimeDriveOutcome::Retry {
                        id,
                        reason: NodeLifetimeRetryReason::Contended,
                    };
                }
                let _allocation = inner
                    .allocation
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                quarantine_registry_locked(&inner);
                return NodeLifetimeDriveOutcome::Quarantined {
                    id: Some(id),
                    reason: NodeLifetimeQuarantineReason::ProtocolViolation,
                };
            }
            Ok(Err(_)) => {
                self.retain_reconcile_failure(&inner, key, Some(cleanup), reclaim);
                return NodeLifetimeDriveOutcome::Quarantined {
                    id: Some(id),
                    reason: NodeLifetimeQuarantineReason::CleanupRejected,
                };
            }
            Err(payload) => {
                std::mem::forget(payload);
                std::mem::forget(cleanup);
                self.retain_reconcile_failure(&inner, key, None, reclaim);
                return NodeLifetimeDriveOutcome::Quarantined {
                    id: Some(id),
                    reason: NodeLifetimeQuarantineReason::CleanupPanicked,
                };
            }
        }
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| drop(cleanup))) {
            std::mem::forget(payload);
            self.retain_reconcile_failure(&inner, key, None, reclaim);
            return NodeLifetimeDriveOutcome::Quarantined {
                id: Some(id),
                reason: NodeLifetimeQuarantineReason::CleanupDestructorPanicked,
            };
        }

        let allocation = match inner.allocation.lock() {
            Ok(allocation) => allocation,
            Err(poisoned) => {
                drop(poisoned.into_inner());
                self.retain_reconcile_failure(&inner, key, None, reclaim);
                return NodeLifetimeDriveOutcome::Quarantined {
                    id: Some(id),
                    reason: NodeLifetimeQuarantineReason::RegistryPoisoned,
                };
            }
        };
        let node_ids = self
            .node_ids
            .as_mut()
            .expect("live lifetime owner retains its exact id owner");
        if let Err(reclaim) = node_ids.make_reconciled_available(reclaim) {
            self.place_or_orphan_quarantined(&inner, key, None, reclaim);
            quarantine_registry_locked(&inner);
            drop(allocation);
            return NodeLifetimeDriveOutcome::Quarantined {
                id: Some(id),
                reason: NodeLifetimeQuarantineReason::ReclaimBrandMismatch,
            };
        }
        // Publication to the allocator and registry vacancy share `allocation`. A constructor may
        // reserve the id during this tiny interval, but registration sees Contended and rolls its
        // provisional id back; it can never observe a stale duplicate or publish before vacancy.
        #[cfg(test)]
        if let Some((entered, release)) = inner
            .teardown_post_publish_hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            entered.send(()).unwrap();
            release.recv().unwrap();
        }
        let expected = slot_word(key.generation.get(), SlotPhase::Reconciling, true);
        if let Err(observed) = slot.word.compare_exchange(
            expected,
            slot_word(key.generation.get(), SlotPhase::Vacant, false),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            // The id publication cannot be retracted. Keeping the shared allocation lock while
            // quarantining the registry prevents any registrar from accepting a reservation that
            // may transiently pop it from the allocator.
            quarantine_registry_locked(&inner);
            let mut current = observed;
            while generation(current) == key.generation.get()
                && !matches!(
                    SlotPhase::from_word(current),
                    SlotPhase::Vacant | SlotPhase::Sealed | SlotPhase::Quarantined
                )
            {
                match slot.word.compare_exchange(
                    current,
                    with_phase(current, SlotPhase::Quarantined),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(next) => current = next,
                }
            }
            drop(allocation);
            return NodeLifetimeDriveOutcome::Quarantined {
                id: Some(id),
                reason: NodeLifetimeQuarantineReason::ProtocolViolation,
            };
        }
        drop(allocation);
        NodeLifetimeDriveOutcome::Reconciled { id }
    }

    fn restore_reconcile_retry(
        &mut self,
        inner: &Arc<NodeLifetimeInner>,
        key: RegistrationKey,
        prior_phase: SlotPhase,
        cleanup: Box<dyn InjectedNodeReclaimCleanup>,
        reclaim: OwnedPendingNodeReclaim,
    ) -> bool {
        debug_assert!(matches!(
            prior_phase,
            SlotPhase::Requested | SlotPhase::AwaitingReclaim
        ));
        let slot = &inner.slots[key.slot];
        let mut payload = slot
            .payload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let expected = slot_word(key.generation.get(), SlotPhase::Reconciling, true);
        if slot.word.load(Ordering::Acquire) != expected
            || payload.cleanup.is_some()
            || payload.reclaim.is_some()
        {
            std::mem::forget(cleanup);
            if self.orphan_reclaim.is_none() {
                self.orphan_reclaim = Some(reclaim);
            } else {
                std::mem::forget(reclaim);
            }
            return false;
        }
        payload.cleanup = Some(cleanup);
        payload.reclaim = Some(reclaim);
        if slot
            .word
            .compare_exchange(
                expected,
                slot_word(key.generation.get(), prior_phase, true),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            true
        } else {
            let cleanup = payload.cleanup.take().unwrap();
            let reclaim = payload.reclaim.take().unwrap();
            std::mem::forget(cleanup);
            if self.orphan_reclaim.is_none() {
                self.orphan_reclaim = Some(reclaim);
            } else {
                std::mem::forget(reclaim);
            }
            false
        }
    }

    fn retain_reconcile_failure(
        &mut self,
        inner: &Arc<NodeLifetimeInner>,
        key: RegistrationKey,
        cleanup: Option<Box<dyn InjectedNodeReclaimCleanup>>,
        reclaim: OwnedPendingNodeReclaim,
    ) {
        self.place_or_orphan_quarantined(inner, key, cleanup, reclaim);
        let _allocation = inner
            .allocation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        quarantine_registry_locked(inner);
    }

    fn place_or_orphan_quarantined(
        &mut self,
        inner: &Arc<NodeLifetimeInner>,
        key: RegistrationKey,
        cleanup: Option<Box<dyn InjectedNodeReclaimCleanup>>,
        reclaim: OwnedPendingNodeReclaim,
    ) {
        match try_place_quarantined(inner, key, cleanup, reclaim) {
            Ok(()) => {}
            Err((cleanup, reclaim)) => {
                if let Some(cleanup) = cleanup {
                    std::mem::forget(cleanup);
                }
                if self.orphan_reclaim.is_none() {
                    self.orphan_reclaim = Some(reclaim);
                } else {
                    std::mem::forget(reclaim);
                }
            }
        }
    }
}

fn try_place_quarantined(
    inner: &Arc<NodeLifetimeInner>,
    key: RegistrationKey,
    cleanup: Option<Box<dyn InjectedNodeReclaimCleanup>>,
    reclaim: OwnedPendingNodeReclaim,
) -> Result<
    (),
    (
        Option<Box<dyn InjectedNodeReclaimCleanup>>,
        OwnedPendingNodeReclaim,
    ),
> {
    let slot = &inner.slots[key.slot];
    let mut payload = slot
        .payload
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let expected = slot_word(key.generation.get(), SlotPhase::Reconciling, true);
    if slot.word.load(Ordering::Acquire) != expected
        || payload.cleanup.is_some()
        || payload.reclaim.is_some()
    {
        return Err((cleanup, reclaim));
    }
    payload.cleanup = cleanup;
    payload.reclaim = Some(reclaim);
    if slot
        .word
        .compare_exchange(
            expected,
            slot_word(key.generation.get(), SlotPhase::Quarantined, true),
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    {
        Ok(())
    } else {
        Err((payload.cleanup.take(), payload.reclaim.take().unwrap()))
    }
}
