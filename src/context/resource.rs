//! Opaque host resource ownership attached to exact hosted graph and command lifetimes.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

/// A host-owned reservation released with an exact hosted node construction.
///
/// This is an accounting attachment, not graph authority. The hosted Gain and Oscillator
/// constructors move it into their exact lifetime records before allocating node IDs. Successful
/// construction retains it until every node created by that compound constructor has been
/// physically reclaimed (or the whole graph retires). Rejected construction releases it during
/// rollback. A fail-closed lifetime quarantine deliberately retains it.
///
/// The wrapped value must have a nonblocking destructor. A destructor panic is contained rather
/// than allowed to unwind through the audio lifecycle worker.
pub struct AudioNodeLifetimeReservation {
    value: Option<Box<dyn Send + 'static>>,
}

impl AudioNodeLifetimeReservation {
    /// Wraps one or more host accounting guards in a single opaque reservation.
    pub fn new<T>(value: T) -> Self
    where
        T: Send + 'static,
    {
        Self {
            value: Some(Box::new(value)),
        }
    }
}

impl fmt::Debug for AudioNodeLifetimeReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioNodeLifetimeReservation")
            .finish_non_exhaustive()
    }
}

impl Drop for AudioNodeLifetimeReservation {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| drop(value))) {
            // A host accounting destructor cannot be allowed to unwind through exact rollback or
            // physical-reclaim processing. The panic payload may itself have a hostile Drop.
            std::mem::forget(payload);
        }
    }
}

/// A host-owned reservation released with one exact hosted control batch.
///
/// This is an accounting attachment, not graph authority. A hosted operation moves it into the
/// same fixed logical-command credit that owns the submitted batch. Rejected preparation or
/// submission releases it during rollback. Accepted work retains it while staged, queued,
/// rendering, or awaiting off-render-thread batch reclamation. A fail-closed transport quarantine
/// deliberately retains it.
///
/// One reservation may represent every command in the batch; the host remains responsible for
/// reserving the exact batch cost before calling the corresponding hosted operation. The wrapped
/// value must have a nonblocking destructor. A destructor panic is contained rather than allowed
/// to unwind through control rollback or batch reclamation.
pub struct AudioControlBatchReservation {
    value: Option<Box<dyn Send + 'static>>,
}

impl AudioControlBatchReservation {
    /// Wraps one or more host accounting guards in a single opaque reservation.
    pub fn new<T>(value: T) -> Self
    where
        T: Send + 'static,
    {
        Self {
            value: Some(Box::new(value)),
        }
    }
}

impl fmt::Debug for AudioControlBatchReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioControlBatchReservation")
            .finish_non_exhaustive()
    }
}

impl Drop for AudioControlBatchReservation {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| drop(value))) {
            // Accounting destruction cannot unwind through exact rollback, the GC worker, or a
            // renderer teardown fallback. The panic payload may itself have a hostile Drop.
            std::mem::forget(payload);
        }
    }
}

/// A one-shot host callback that reserves accounting for the exact size of a control batch.
///
/// Some hosted operations, notably broad `AudioNode::disconnect` calls, cannot know their
/// command count until the exact connection registry has been serialized and inspected. This
/// provider lets an embedder reserve that count at the transaction boundary without maintaining
/// a second, race-prone graph mirror. It is invoked only when the operation will submit a
/// non-empty batch; duplicate connects and permitted no-match disconnects drop it unused.
///
/// Returning `None` rejects the operation before transport reservation or host-graph mutation.
/// The callback runs on the calling control thread, may be invoked at most once, and must not
/// block. A callback panic is treated as a fail-closed hosted transaction failure.
pub struct AudioControlBatchReservationProvider {
    provider: Box<dyn FnOnce(usize) -> Option<AudioControlBatchReservation> + Send + 'static>,
}

impl AudioControlBatchReservationProvider {
    /// Wraps a one-shot reservation callback.
    pub fn new<F>(provider: F) -> Self
    where
        F: FnOnce(usize) -> Option<AudioControlBatchReservation> + Send + 'static,
    {
        Self {
            provider: Box::new(provider),
        }
    }

    pub(crate) fn reserve(self, command_count: usize) -> Option<AudioControlBatchReservation> {
        debug_assert!(command_count != 0);
        (self.provider)(command_count)
    }
}

impl fmt::Debug for AudioControlBatchReservationProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioControlBatchReservationProvider")
            .finish_non_exhaustive()
    }
}

/// Shared only by the fixed set of lifetime cleanups created in one compound transaction.
/// Keeping the payload behind a mutex permits a merely `Send` host guard to be retained by the
/// `Arc` while cleanup records may move between lifecycle threads.
#[derive(Clone)]
pub(crate) struct SharedAudioNodeLifetimeReservation {
    _inner: Arc<Mutex<AudioNodeLifetimeReservation>>,
}

impl SharedAudioNodeLifetimeReservation {
    pub(crate) fn new(reservation: AudioNodeLifetimeReservation) -> Self {
        Self {
            _inner: Arc::new(Mutex::new(reservation)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("host reservation destructor panic");
        }
    }

    #[test]
    fn opaque_reservation_contains_a_hostile_destructor() {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            drop(AudioNodeLifetimeReservation::new(PanicOnDrop));
        }));
        assert!(result.is_ok());

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            drop(AudioControlBatchReservation::new(PanicOnDrop));
        }));
        assert!(result.is_ok());
    }
}
