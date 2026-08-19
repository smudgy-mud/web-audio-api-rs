//! Opaque wire records for the private injected explicit-edge renderer.
//!
//! B5b-r deliberately exposes no production constructor. The later host-registry slice will
//! create these values only through dedicated admitted operations; until then only render tests
//! can mint them. That first producer is scoped to the already hosted magic/Gain endpoints;
//! exact DelayNode/cycle-breaker host semantics require a separate re-audit when such an endpoint
//! becomes constructible. The renderer already preserves fixed records while computing a
//! non-destructive cycle-break ordering, but this slice makes no public Delay capability claim.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;

use super::AudioNodeId;

/// Maximum explicit edges represented by one private injected graph.
///
/// Hidden parameter/owner and listener edges are stored separately and do not consume this
/// capacity. A later outer resource budget must choose a limit no greater than this ceiling.
pub(crate) const MAX_INJECTED_EXPLICIT_CONNECTIONS: usize = 256;

/// Magic nodes 0 through 10 plus every concurrently represented ordinary lifetime slot.
pub(crate) const MAX_INJECTED_GRAPH_NODES: usize =
    11 + super::injected_node_lifetime::DEFAULT_NODE_LIFETIME_CAPACITY;

const _: () = assert!(MAX_INJECTED_EXPLICIT_CONNECTIONS <= u16::MAX as usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InjectedExplicitConnect {
    from: AudioNodeId,
    to: AudioNodeId,
    output: usize,
    input: usize,
}

impl InjectedExplicitConnect {
    pub(crate) const fn edge(self) -> (AudioNodeId, usize, AudioNodeId, usize) {
        (self.from, self.output, self.to, self.input)
    }

    #[cfg(test)]
    pub(crate) const fn new_for_test(
        from: AudioNodeId,
        to: AudioNodeId,
        output: usize,
        input: usize,
    ) -> Self {
        Self {
            from,
            to,
            output,
            input,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InjectedExplicitDisconnect {
    from: AudioNodeId,
    to: AudioNodeId,
    output: usize,
    input: usize,
}

impl InjectedExplicitDisconnect {
    pub(crate) const fn edge(self) -> (AudioNodeId, usize, AudioNodeId, usize) {
        (self.from, self.output, self.to, self.input)
    }

    #[cfg(test)]
    pub(crate) const fn new_for_test(
        from: AudioNodeId,
        to: AudioNodeId,
        output: usize,
        input: usize,
    ) -> Self {
        Self {
            from,
            to,
            output,
            input,
        }
    }
}

// B5b-h1 installs this owner into the exact node-lifetime composite before the production host
// endpoint API is exposed. The fixed record/key shapes are completed by h2; keeping the owner and
// retirement proof here already prevents a later base-local mirror or independently swappable
// cleanup authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ConnectionRegistryPhase {
    Open = 0,
    Sealed = 1,
    Retired = 2,
}

impl ConnectionRegistryPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Open,
            1 => Self::Sealed,
            2 => Self::Retired,
            _ => unreachable!("private injected connection registry phase"),
        }
    }
}

/// Compact host representation completed by the endpoint transaction half of B5b-h.
///
/// It is intentionally distinct from hidden renderer edges. The two endpoint-generation words
/// will encode either one exact ordinary slot generation or an accepted permanent-magic kind.
#[derive(Clone, Copy)]
#[allow(dead_code)] // populated by the immediately following B5b-h endpoint transaction slice
pub(super) struct InjectedHostExplicitConnection {
    pub(super) edge: (AudioNodeId, usize, AudioNodeId, usize),
    pub(super) source_generation: u64,
    pub(super) destination_generation: u64,
}

struct InjectedConnectionRegistryState {
    edges: ArrayVec<InjectedHostExplicitConnection, MAX_INJECTED_EXPLICIT_CONNECTIONS>,
}

/// Sole strong host-registry owner nested in `NodeLifetimeInner`.
///
/// Base/endpoint capabilities in h2 may upgrade only the outer lifetime owner. Incident cleanups
/// receive a Weak to this nested owner, which remains upgradeable after outer Arc uniqueness is
/// proven and while whole-graph slot cleanup is running.
pub(super) struct InjectedConnectionRegistryInner {
    phase: AtomicU8,
    protocol_failed: AtomicBool,
    serializer: Mutex<InjectedConnectionRegistryState>,
}

impl InjectedConnectionRegistryInner {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            phase: AtomicU8::new(ConnectionRegistryPhase::Open as u8),
            protocol_failed: AtomicBool::new(false),
            serializer: Mutex::new(InjectedConnectionRegistryState {
                edges: ArrayVec::new(),
            }),
        })
    }

    /// Admission has already drained, so this absorbing phase transition needs no serializer and
    /// cannot block behind a stale pre-capacity caller.
    pub(super) fn seal_after_control_drain(&self) -> bool {
        let phase = ConnectionRegistryPhase::from_u8(self.phase.load(Ordering::Acquire));
        if phase == ConnectionRegistryPhase::Open {
            let _ = self.phase.compare_exchange(
                ConnectionRegistryPhase::Open as u8,
                ConnectionRegistryPhase::Sealed as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        self.protocol_failed.load(Ordering::Acquire)
    }

    #[allow(dead_code)] // armed by h2's transaction unwind guards
    pub(super) fn fail_closed_protocol(&self) {
        self.protocol_failed.store(true, Ordering::Release);
    }

    #[allow(dead_code)] // read by h2 producer and cleanup preflights
    pub(super) fn is_open(&self) -> bool {
        ConnectionRegistryPhase::from_u8(self.phase.load(Ordering::Acquire))
            == ConnectionRegistryPhase::Open
            && !self.protocol_failed.load(Ordering::Acquire)
    }

    /// Consumes the nested strong owner only after outer lifetime uniqueness and every slot
    /// cleanup. No retryable nested contention may first appear here: an extra strong owner is an
    /// impossible proof failure and is quarantined rather than rewrapped under a new identity.
    pub(super) fn retire_after_slot_cleanup(
        owner: Arc<Self>,
    ) -> InjectedConnectionRegistryRetirement {
        owner
            .phase
            .store(ConnectionRegistryPhase::Retired as u8, Ordering::Release);
        let protocol_failed = owner.protocol_failed.load(Ordering::Acquire);
        let inner = match Arc::try_unwrap(owner) {
            Ok(inner) => inner,
            Err(owner) => {
                std::mem::forget(owner);
                return InjectedConnectionRegistryRetirement {
                    residual_edges_cleared: 0,
                    protocol_failed: true,
                    serializer_poison_recovered: false,
                    ownership_mismatch: true,
                };
            }
        };
        let (mut state, serializer_poison_recovered) = match inner.serializer.into_inner() {
            Ok(state) => (state, false),
            Err(poisoned) => (poisoned.into_inner(), true),
        };
        let residual_edges_cleared = state.edges.len();
        state.edges.clear();
        InjectedConnectionRegistryRetirement {
            residual_edges_cleared,
            protocol_failed,
            serializer_poison_recovered,
            ownership_mismatch: false,
        }
    }

    #[cfg(test)]
    pub(super) fn push_residual_for_test(&self, edge: InjectedHostExplicitConnection) {
        self.serializer.lock().unwrap().edges.push(edge);
    }

    #[cfg(test)]
    pub(super) fn poison_serializer_for_test(&self) {
        let _state = self.serializer.lock().unwrap();
        panic!("poison injected connection serializer");
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct InjectedConnectionRegistryRetirement {
    pub(crate) residual_edges_cleared: usize,
    pub(crate) protocol_failed: bool,
    pub(crate) serializer_poison_recovered: bool,
    pub(crate) ownership_mismatch: bool,
}
