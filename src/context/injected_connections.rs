//! Opaque wire records for the private injected explicit-edge renderer.
//!
//! B5b-r deliberately exposes no production constructor. The later host-registry slice will
//! create these values only through dedicated admitted operations; until then only render tests
//! can mint them. That first producer is scoped to the already hosted magic/Gain endpoints;
//! exact DelayNode/cycle-breaker host semantics require a separate re-audit when such an endpoint
//! becomes constructible. The renderer already preserves fixed records while computing a
//! non-destructive cycle-break ordering, but this slice makes no public Delay capability claim.

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
