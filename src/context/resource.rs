//! Opaque host resource ownership attached to exact hosted graph lifetimes.

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
    }
}
