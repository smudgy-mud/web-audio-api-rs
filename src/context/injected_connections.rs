//! Opaque wire records for the private injected explicit-edge renderer.
//!
//! B5b-h2 selects the admitted Gain/magic host-registry producer for public `AudioNode`
//! connect/disconnect overloads. Exact DelayNode/cycle-breaker semantics still require a separate
//! re-audit: the renderer preserves fixed records while computing a non-destructive cycle-break
//! ordering, but this module mints no Delay capability.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError, Weak};

#[cfg(test)]
use std::collections::VecDeque;

use arrayvec::ArrayVec;

use super::injected_control::{
    CommitControlOutcome, InjectedControlError, InjectedControlIdentity, InjectedControlProducer,
    RejectedControlRollback,
};
use super::injected_ids::InjectedNodeIdIdentity;
use super::injected_node_lifetime::{
    InjectedNodeReclaimCleanup, InjectedNodeRegistration, InjectedNodeRegistrationStamp,
    NodeLifetimeInner, NodeReclaimCleanupError,
};
use super::{AudioNodeId, DESTINATION_NODE_ID, LISTENER_PARAM_IDS};
use crate::message::ControlMessage;

/// Maximum explicit edges represented by one private injected graph.
///
/// Hidden parameter/owner and listener edges are stored separately and do not consume this
/// capacity. A later outer resource budget must choose a limit no greater than this ceiling.
pub(crate) const MAX_INJECTED_EXPLICIT_CONNECTIONS: usize = 256;

/// Magic nodes 0 through 10 plus every concurrently represented ordinary lifetime slot.
///
/// This is intentionally independent of the smaller explicit-edge ceiling: a
/// graph may retain many disconnected scheduled sources without representing
/// an explicit connection for each one.
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

    const fn new(from: AudioNodeId, to: AudioNodeId, output: usize, input: usize) -> Self {
        Self {
            from,
            to,
            output,
            input,
        }
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

    const fn new(from: AudioNodeId, to: AudioNodeId, output: usize, input: usize) -> Self {
        Self {
            from,
            to,
            output,
            input,
        }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedConnectionEndpointKind {
    AudioNode,
    AudioParam,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PermanentMagicEndpoint {
    Destination,
    ListenerParam(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InjectedConnectionEndpointLifetime {
    Ordinary(InjectedNodeRegistrationStamp),
    PermanentMagic(PermanentMagicEndpoint),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InjectedConnectionEndpointStamp {
    id: AudioNodeId,
    kind: InjectedConnectionEndpointKind,
    lifetime: InjectedConnectionEndpointLifetime,
}

/// Weak, cloneable endpoint brand. It retains no registration, transport credit, or registry
/// owner; every operation must upgrade and validate the exact outer node-lifetime authority.
#[derive(Clone)]
pub(crate) struct InjectedConnectionEndpoint {
    registry: Weak<NodeLifetimeInner>,
    control: InjectedControlIdentity,
    node_ids: InjectedNodeIdIdentity,
    stamp: InjectedConnectionEndpointStamp,
    inputs: usize,
    outputs: usize,
}

impl InjectedConnectionEndpoint {
    pub(super) fn new_ordinary(
        registry: Weak<NodeLifetimeInner>,
        control: InjectedControlIdentity,
        node_ids: InjectedNodeIdIdentity,
        kind: InjectedConnectionEndpointKind,
        inputs: usize,
        outputs: usize,
        registration: InjectedNodeRegistrationStamp,
    ) -> Self {
        Self {
            registry,
            control,
            node_ids,
            stamp: InjectedConnectionEndpointStamp {
                id: registration.id,
                kind,
                lifetime: InjectedConnectionEndpointLifetime::Ordinary(registration),
            },
            inputs,
            outputs,
        }
    }

    fn new_permanent(
        registry: Weak<NodeLifetimeInner>,
        control: InjectedControlIdentity,
        node_ids: InjectedNodeIdIdentity,
        magic: PermanentMagicEndpoint,
    ) -> Self {
        let (id, kind, inputs, outputs) = match magic {
            PermanentMagicEndpoint::Destination => (
                DESTINATION_NODE_ID,
                InjectedConnectionEndpointKind::AudioNode,
                1,
                1,
            ),
            PermanentMagicEndpoint::ListenerParam(index) => (
                AudioNodeId(
                    LISTENER_PARAM_IDS
                        .clone()
                        .nth(usize::from(index))
                        .expect("accepted magic listener endpoint index is exact"),
                ),
                InjectedConnectionEndpointKind::AudioParam,
                1,
                1,
            ),
        };
        Self {
            registry,
            control,
            node_ids,
            stamp: InjectedConnectionEndpointStamp {
                id,
                kind,
                lifetime: InjectedConnectionEndpointLifetime::PermanentMagic(magic),
            },
            inputs,
            outputs,
        }
    }

    pub(crate) fn matches_registration(
        &self,
        registration: &InjectedNodeRegistration,
        id: AudioNodeId,
        kind: InjectedConnectionEndpointKind,
        inputs: usize,
        outputs: usize,
    ) -> bool {
        self.stamp.id == id
            && self.stamp.kind == kind
            && self.inputs == inputs
            && self.outputs == outputs
            && matches!(
                self.stamp.lifetime,
                InjectedConnectionEndpointLifetime::Ordinary(stamp)
                    if registration.matches_connection_stamp(&self.registry, stamp)
            )
    }

    /// Exact attachment proof against the constructor embedded in the public registration's
    /// `ConcreteBaseAudioContext`. This closes same-control foreign allocator/registry swaps.
    pub(crate) fn matches_constructor(
        &self,
        constructor: &super::injected_node_construction::InjectedNodeConstructor,
    ) -> bool {
        Weak::ptr_eq(&self.registry, &constructor.registry_identity())
            && constructor.matches_control_identity(&self.control)
            && constructor.matches_node_id_identity(&self.node_ids)
    }

    /// Terminalizes the original endpoint authorities after an impossible attachment mismatch.
    pub(crate) fn fail_closed_protocol(&self) {
        if let Some(owner) = self.registry.upgrade() {
            owner.connection_registry().fail_closed_protocol();
        }
        self.control.fail_closed_protocol();
    }

    pub(crate) fn matches_permanent_registration(
        &self,
        id: AudioNodeId,
        kind: InjectedConnectionEndpointKind,
        inputs: usize,
        outputs: usize,
    ) -> bool {
        let exact_magic = match self.stamp.lifetime {
            InjectedConnectionEndpointLifetime::PermanentMagic(
                PermanentMagicEndpoint::Destination,
            ) => {
                id == DESTINATION_NODE_ID
                    && kind == InjectedConnectionEndpointKind::AudioNode
                    && inputs == 1
                    && outputs == 1
            }
            InjectedConnectionEndpointLifetime::PermanentMagic(
                PermanentMagicEndpoint::ListenerParam(index),
            ) => {
                LISTENER_PARAM_IDS.clone().nth(usize::from(index)) == Some(id.0)
                    && kind == InjectedConnectionEndpointKind::AudioParam
                    && inputs == 1
                    && outputs == 1
                    && index < 9
            }
            InjectedConnectionEndpointLifetime::Ordinary(_) => false,
        };
        exact_magic
            && self.stamp.id == id
            && self.stamp.kind == kind
            && self.inputs == inputs
            && self.outputs == outputs
    }

    pub(super) fn incident_cleanup(&self) -> Option<Box<dyn InjectedNodeReclaimCleanup>> {
        let InjectedConnectionEndpointLifetime::Ordinary(_) = self.stamp.lifetime else {
            return None;
        };
        let inner = self.registry.upgrade()?;
        Some(Box::new(InjectedIncidentConnectionCleanup {
            id: self.stamp.id,
            endpoint: self.stamp,
            registry: Arc::downgrade(inner.connection_registry()),
        }))
    }

    fn is_current(&self, owner: &NodeLifetimeInner) -> bool {
        match self.stamp.lifetime {
            InjectedConnectionEndpointLifetime::Ordinary(stamp) => {
                owner.connection_registration_is_live(self.stamp.id, stamp)
            }
            InjectedConnectionEndpointLifetime::PermanentMagic(_) => true,
        }
    }
}

#[must_use]
pub(crate) struct InjectedMagicConnectionEndpoints {
    destination: InjectedConnectionEndpoint,
    listener_params: [InjectedConnectionEndpoint; 9],
}

impl InjectedMagicConnectionEndpoints {
    pub(super) fn from_accepted_magic(
        accepted: super::injected_magic_construction::AcceptedMagicConnectionBrand,
    ) -> Self {
        let (registry, control, node_ids) = accepted.into_parts();
        let destination = InjectedConnectionEndpoint::new_permanent(
            Weak::clone(&registry),
            control.clone(),
            node_ids.clone(),
            PermanentMagicEndpoint::Destination,
        );
        let listener_params = std::array::from_fn(|index| {
            InjectedConnectionEndpoint::new_permanent(
                Weak::clone(&registry),
                control.clone(),
                node_ids.clone(),
                PermanentMagicEndpoint::ListenerParam(index as u8),
            )
        });
        Self {
            destination,
            listener_params,
        }
    }

    pub(crate) fn destination(&self) -> InjectedConnectionEndpoint {
        self.destination.clone()
    }

    pub(crate) fn listener_param(&self, index: usize) -> InjectedConnectionEndpoint {
        self.listener_params[index].clone()
    }
}

/// One represented public edge. Hidden magic and AudioParam-owner edges never enter this mirror.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InjectedHostExplicitConnection {
    source: InjectedConnectionEndpointStamp,
    destination: InjectedConnectionEndpointStamp,
    output: usize,
    input: usize,
}

impl InjectedHostExplicitConnection {
    fn wire_connect(self) -> InjectedExplicitConnect {
        InjectedExplicitConnect::new(self.source.id, self.destination.id, self.output, self.input)
    }

    fn wire_disconnect(self) -> InjectedExplicitDisconnect {
        InjectedExplicitDisconnect::new(
            self.source.id,
            self.destination.id,
            self.output,
            self.input,
        )
    }

    fn is_incident_to(self, endpoint: InjectedConnectionEndpointStamp) -> bool {
        self.source == endpoint || self.destination == endpoint
    }
}

struct InjectedConnectionRegistryState {
    edges: ArrayVec<InjectedHostExplicitConnection, MAX_INJECTED_EXPLICIT_CONNECTIONS>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedConnectionOperationError {
    Control(InjectedControlError),
    ForeignEndpoint,
    InactiveEndpoint,
    InvalidPort,
    Capacity,
    Unconnected,
    SerializerPoisoned,
    RejectedPayloadPanicked,
    ProtocolViolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedConnectionOperationOutcome {
    Noop,
    Committed(CommitControlOutcome),
}

/// Closed set of legal public disconnect overloads.
///
/// In particular, an input selector can exist only together with both a destination and output;
/// the exact host registry never accepts independently swappable `Option` fields.
pub(crate) enum InjectedDisconnectSelector<'a> {
    All,
    Destination(&'a InjectedConnectionEndpoint),
    Output(usize),
    DestinationOutput {
        destination: &'a InjectedConnectionEndpoint,
        output: usize,
    },
    Exact {
        destination: &'a InjectedConnectionEndpoint,
        output: usize,
        input: usize,
    },
}

impl<'a> InjectedDisconnectSelector<'a> {
    fn destination(&self) -> Option<&'a InjectedConnectionEndpoint> {
        match self {
            Self::All | Self::Output(_) => None,
            Self::Destination(destination)
            | Self::DestinationOutput { destination, .. }
            | Self::Exact { destination, .. } => Some(destination),
        }
    }

    fn output(&self) -> Option<usize> {
        match self {
            Self::All | Self::Destination(_) => None,
            Self::Output(output)
            | Self::DestinationOutput { output, .. }
            | Self::Exact { output, .. } => Some(*output),
        }
    }

    fn input(&self) -> Option<usize> {
        match self {
            Self::Exact { input, .. } => Some(*input),
            Self::All | Self::Destination(_) | Self::Output(_) | Self::DestinationOutput { .. } => {
                None
            }
        }
    }

    fn matches(
        &self,
        source: InjectedConnectionEndpointStamp,
        edge: &InjectedHostExplicitConnection,
    ) -> bool {
        edge.source == source
            && self.output().is_none_or(|output| output == edge.output)
            && self
                .destination()
                .is_none_or(|destination| destination.stamp == edge.destination)
            && self.input().is_none_or(|input| input == edge.input)
    }

    fn missing_is_error(&self) -> bool {
        self.destination().is_some()
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedConnectionOperationTestPoint {
    BeforeSerializer,
    BeforeReserve,
    BeforeCommit,
    RejectedRollback,
    AcceptedMutation,
}

#[cfg(test)]
struct InjectedConnectionOperationTestHook {
    point: InjectedConnectionOperationTestPoint,
    entered: crossbeam_channel::Sender<()>,
    release: crossbeam_channel::Receiver<()>,
    panics: bool,
}

struct FailClosedConnectionOperation<'a> {
    registry: &'a InjectedConnectionRegistryInner,
    control: &'a InjectedControlProducer,
    armed: bool,
}

impl<'a> FailClosedConnectionOperation<'a> {
    fn new(
        registry: &'a InjectedConnectionRegistryInner,
        control: &'a InjectedControlProducer,
    ) -> Self {
        Self {
            registry,
            control,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for FailClosedConnectionOperation<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.registry.fail_closed_protocol();
            self.control.fail_closed_protocol();
        }
    }
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
    #[cfg(test)]
    operation_hooks: Mutex<VecDeque<InjectedConnectionOperationTestHook>>,
}

impl InjectedConnectionRegistryInner {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            phase: AtomicU8::new(ConnectionRegistryPhase::Open as u8),
            protocol_failed: AtomicBool::new(false),
            serializer: Mutex::new(InjectedConnectionRegistryState {
                edges: ArrayVec::new(),
            }),
            #[cfg(test)]
            operation_hooks: Mutex::new(VecDeque::new()),
        })
    }

    #[cfg(test)]
    pub(super) fn hold_operation_for_test(
        &self,
        point: InjectedConnectionOperationTestPoint,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
        panics: bool,
    ) {
        self.operation_hooks
            .lock()
            .unwrap()
            .push_back(InjectedConnectionOperationTestHook {
                point,
                entered,
                release,
                panics,
            });
    }

    #[cfg(test)]
    fn run_operation_hook_for_test(&self, point: InjectedConnectionOperationTestPoint) {
        let hook = {
            let mut hooks = self.operation_hooks.lock().unwrap();
            hooks
                .iter()
                .position(|hook| hook.point == point)
                .and_then(|index| hooks.remove(index))
        };
        if let Some(hook) = hook {
            hook.entered.send(()).unwrap();
            hook.release.recv().unwrap();
            assert!(!hook.panics, "forced injected connection operation panic");
        }
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

    pub(super) fn fail_closed_protocol(&self) {
        self.protocol_failed.store(true, Ordering::Release);
    }

    pub(super) fn is_open(&self) -> bool {
        ConnectionRegistryPhase::from_u8(self.phase.load(Ordering::Acquire))
            == ConnectionRegistryPhase::Open
            && !self.protocol_failed.load(Ordering::Acquire)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn connect(
        control: &InjectedControlProducer,
        node_ids: &InjectedNodeIdIdentity,
        exact_registry: &Weak<NodeLifetimeInner>,
        source: &InjectedConnectionEndpoint,
        destination: &InjectedConnectionEndpoint,
        output: usize,
        input: usize,
    ) -> Result<InjectedConnectionOperationOutcome, InjectedConnectionOperationError> {
        let owner =
            validate_endpoints(control, node_ids, exact_registry, source, Some(destination))?;
        if output >= source.outputs || input >= destination.inputs {
            return Err(InjectedConnectionOperationError::InvalidPort);
        }
        let admitted = control
            .try_admit_graph_operation()
            .map_err(InjectedConnectionOperationError::Control)?;
        let _admission_fence = admitted.admission_fence();
        let registry = Arc::clone(owner.connection_registry());
        #[cfg(test)]
        registry
            .run_operation_hook_for_test(InjectedConnectionOperationTestPoint::BeforeSerializer);
        let mut state = loop {
            match registry.serializer.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::WouldBlock) => {
                    std::thread::park_timeout(std::time::Duration::from_millis(1));
                }
                Err(TryLockError::Poisoned(_)) => {
                    registry.fail_closed_protocol();
                    control.fail_closed_protocol();
                    return Err(InjectedConnectionOperationError::SerializerPoisoned);
                }
            }
        };
        let mut fail_closed = FailClosedConnectionOperation::new(&registry, control);
        if !registry.is_open() {
            fail_closed.disarm();
            return Err(InjectedConnectionOperationError::Control(
                InjectedControlError::Sealed,
            ));
        }
        if !source.is_current(&owner) || !destination.is_current(&owner) {
            fail_closed.disarm();
            return Err(InjectedConnectionOperationError::InactiveEndpoint);
        }
        let record = InjectedHostExplicitConnection {
            source: source.stamp,
            destination: destination.stamp,
            output,
            input,
        };
        if state.edges.contains(&record) {
            fail_closed.disarm();
            return Ok(InjectedConnectionOperationOutcome::Noop);
        }
        if state.edges.is_full() {
            fail_closed.disarm();
            return Err(InjectedConnectionOperationError::Capacity);
        }
        let expected = record.wire_connect();
        let commands = vec![ControlMessage::InjectedConnectExplicit(expected)].into_boxed_slice();
        #[cfg(test)]
        registry.run_operation_hook_for_test(InjectedConnectionOperationTestPoint::BeforeReserve);
        let reservation = match reserve_after_serialization(admitted, 1) {
            Ok(reservation) => reservation,
            Err(error) => {
                fail_closed.disarm();
                return Err(error);
            }
        };
        let batch = reservation.into_preboxed(commands);
        #[cfg(test)]
        registry.run_operation_hook_for_test(InjectedConnectionOperationTestPoint::BeforeCommit);
        let accepted = match control.try_commit_retained(batch) {
            Ok(accepted) => accepted,
            Err(failure) => {
                let (error, rollback) = failure.rollback_with_commands(|commands| {
                    let mut rollback_guard = FailClosedConnectionOperation::new(&registry, control);
                    let mut commands = commands.into_vec().into_iter();
                    let exact = matches!(
                        (commands.next(), commands.next()),
                        (Some(ControlMessage::InjectedConnectExplicit(value)), None)
                            if value == expected
                    );
                    #[cfg(test)]
                    registry.run_operation_hook_for_test(
                        InjectedConnectionOperationTestPoint::RejectedRollback,
                    );
                    if exact {
                        rollback_guard.disarm();
                    }
                    exact
                });
                fail_closed.disarm();
                return match rollback {
                    RejectedControlRollback::Completed(true) => {
                        Err(InjectedConnectionOperationError::Control(error))
                    }
                    RejectedControlRollback::Completed(false) => {
                        Err(InjectedConnectionOperationError::ProtocolViolation)
                    }
                    RejectedControlRollback::Panicked => {
                        Err(InjectedConnectionOperationError::RejectedPayloadPanicked)
                    }
                };
            }
        };
        fail_closed.disarm();
        drop(fail_closed);
        let mut accepted_mutation = FailClosedConnectionOperation::new(&registry, control);
        #[cfg(test)]
        registry
            .run_operation_hook_for_test(InjectedConnectionOperationTestPoint::AcceptedMutation);
        state.edges.push(record);
        accepted_mutation.disarm();
        Ok(InjectedConnectionOperationOutcome::Committed(
            accepted.complete(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn disconnect(
        control: &InjectedControlProducer,
        node_ids: &InjectedNodeIdIdentity,
        exact_registry: &Weak<NodeLifetimeInner>,
        source: &InjectedConnectionEndpoint,
        selector: InjectedDisconnectSelector<'_>,
    ) -> Result<InjectedConnectionOperationOutcome, InjectedConnectionOperationError> {
        let destination = selector.destination();
        let owner = validate_endpoints(control, node_ids, exact_registry, source, destination)?;
        if selector
            .output()
            .is_some_and(|value| value >= source.outputs)
            || destination.is_some_and(|endpoint| {
                selector
                    .input()
                    .is_some_and(|value| value >= endpoint.inputs)
            })
        {
            return Err(InjectedConnectionOperationError::InvalidPort);
        }
        let admitted = control
            .try_admit_graph_operation()
            .map_err(InjectedConnectionOperationError::Control)?;
        let _admission_fence = admitted.admission_fence();
        let registry = Arc::clone(owner.connection_registry());
        #[cfg(test)]
        registry
            .run_operation_hook_for_test(InjectedConnectionOperationTestPoint::BeforeSerializer);
        let mut state = loop {
            match registry.serializer.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::WouldBlock) => {
                    std::thread::park_timeout(std::time::Duration::from_millis(1));
                }
                Err(TryLockError::Poisoned(_)) => {
                    registry.fail_closed_protocol();
                    control.fail_closed_protocol();
                    return Err(InjectedConnectionOperationError::SerializerPoisoned);
                }
            }
        };
        let mut fail_closed = FailClosedConnectionOperation::new(&registry, control);
        if !registry.is_open() {
            fail_closed.disarm();
            return Err(InjectedConnectionOperationError::Control(
                InjectedControlError::Sealed,
            ));
        }
        if !source.is_current(&owner)
            || destination.is_some_and(|endpoint| !endpoint.is_current(&owner))
        {
            fail_closed.disarm();
            return Err(InjectedConnectionOperationError::InactiveEndpoint);
        }
        let mut removed =
            ArrayVec::<InjectedHostExplicitConnection, MAX_INJECTED_EXPLICIT_CONNECTIONS>::new();
        for edge in &state.edges {
            if selector.matches(source.stamp, edge) {
                removed.push(*edge);
            }
        }
        if removed.is_empty() {
            fail_closed.disarm();
            return if selector.missing_is_error() {
                Err(InjectedConnectionOperationError::Unconnected)
            } else {
                Ok(InjectedConnectionOperationOutcome::Noop)
            };
        }
        let expected: ArrayVec<InjectedExplicitDisconnect, MAX_INJECTED_EXPLICIT_CONNECTIONS> =
            removed
                .iter()
                .copied()
                .map(|edge| edge.wire_disconnect())
                .collect();
        let commands = expected
            .iter()
            .copied()
            .map(ControlMessage::InjectedDisconnectExplicit)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        #[cfg(test)]
        registry.run_operation_hook_for_test(InjectedConnectionOperationTestPoint::BeforeReserve);
        let reservation = match reserve_after_serialization(admitted, removed.len()) {
            Ok(reservation) => reservation,
            Err(error) => {
                fail_closed.disarm();
                return Err(error);
            }
        };
        let batch = reservation.into_preboxed(commands);
        #[cfg(test)]
        registry.run_operation_hook_for_test(InjectedConnectionOperationTestPoint::BeforeCommit);
        let accepted = match control.try_commit_retained(batch) {
            Ok(accepted) => accepted,
            Err(failure) => {
                let (error, rollback) = failure.rollback_with_commands(|commands| {
                    let mut rollback_guard = FailClosedConnectionOperation::new(&registry, control);
                    let mut exact_commands = commands.into_vec().into_iter();
                    let exact = expected.iter().copied().all(|expected| {
                        matches!(
                            exact_commands.next(),
                            Some(ControlMessage::InjectedDisconnectExplicit(value))
                                if value == expected
                        )
                    }) && exact_commands.next().is_none();
                    #[cfg(test)]
                    registry.run_operation_hook_for_test(
                        InjectedConnectionOperationTestPoint::RejectedRollback,
                    );
                    if exact {
                        rollback_guard.disarm();
                    }
                    exact
                });
                fail_closed.disarm();
                return match rollback {
                    RejectedControlRollback::Completed(true) => {
                        Err(InjectedConnectionOperationError::Control(error))
                    }
                    RejectedControlRollback::Completed(false) => {
                        Err(InjectedConnectionOperationError::ProtocolViolation)
                    }
                    RejectedControlRollback::Panicked => {
                        Err(InjectedConnectionOperationError::RejectedPayloadPanicked)
                    }
                };
            }
        };
        fail_closed.disarm();
        drop(fail_closed);
        let mut accepted_mutation = FailClosedConnectionOperation::new(&registry, control);
        #[cfg(test)]
        registry
            .run_operation_hook_for_test(InjectedConnectionOperationTestPoint::AcceptedMutation);
        state.edges.retain(|edge| !removed.contains(edge));
        accepted_mutation.disarm();
        Ok(InjectedConnectionOperationOutcome::Committed(
            accepted.complete(),
        ))
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
    pub(super) fn push_residual_for_test(&self) {
        let source = InjectedConnectionEndpointStamp {
            id: AudioNodeId(2),
            kind: InjectedConnectionEndpointKind::AudioParam,
            lifetime: InjectedConnectionEndpointLifetime::PermanentMagic(
                PermanentMagicEndpoint::ListenerParam(0),
            ),
        };
        let destination = InjectedConnectionEndpointStamp {
            id: AudioNodeId(0),
            kind: InjectedConnectionEndpointKind::AudioNode,
            lifetime: InjectedConnectionEndpointLifetime::PermanentMagic(
                PermanentMagicEndpoint::Destination,
            ),
        };
        self.serializer
            .lock()
            .unwrap()
            .edges
            .push(InjectedHostExplicitConnection {
                source,
                destination,
                output: 0,
                input: 0,
            });
    }

    #[cfg(test)]
    pub(super) fn edge_count_for_test(&self) -> usize {
        self.serializer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .edges
            .len()
    }

    #[cfg(test)]
    pub(super) fn poison_serializer_for_test(&self) {
        let _state = self.serializer.lock().unwrap();
        panic!("poison injected connection serializer");
    }
}

fn validate_endpoints(
    control: &InjectedControlProducer,
    node_ids: &InjectedNodeIdIdentity,
    exact_registry: &Weak<NodeLifetimeInner>,
    source: &InjectedConnectionEndpoint,
    destination: Option<&InjectedConnectionEndpoint>,
) -> Result<Arc<NodeLifetimeInner>, InjectedConnectionOperationError> {
    if !Weak::ptr_eq(exact_registry, &source.registry)
        || !source.control.ptr_eq(&control.identity())
        || !source.node_ids.ptr_eq(node_ids)
        || destination.is_some_and(|destination| {
            !Weak::ptr_eq(exact_registry, &destination.registry)
                || !destination.control.ptr_eq(&source.control)
                || !destination.node_ids.ptr_eq(&source.node_ids)
        })
    {
        return Err(InjectedConnectionOperationError::ForeignEndpoint);
    }
    let owner = exact_registry
        .upgrade()
        .ok_or(InjectedConnectionOperationError::InactiveEndpoint)?;
    if !owner.matches_connection_brands(&source.control, &source.node_ids) {
        return Err(InjectedConnectionOperationError::ForeignEndpoint);
    }
    if !source.is_current(&owner)
        || destination.is_some_and(|destination| !destination.is_current(&owner))
    {
        return Err(InjectedConnectionOperationError::InactiveEndpoint);
    }
    Ok(owner)
}

fn reserve_after_serialization(
    mut admitted: super::injected_control::AdmittedGraphOperation,
    command_count: usize,
) -> Result<super::injected_control::ControlBatchReservation, InjectedConnectionOperationError> {
    loop {
        match admitted.reserve_commands(command_count) {
            Ok(reservation) => return Ok(reservation),
            Err(failure) if failure.error == InjectedControlError::Contended => {
                admitted = failure.operation;
                std::thread::park_timeout(std::time::Duration::from_millis(1));
            }
            Err(failure) => {
                return Err(InjectedConnectionOperationError::Control(failure.error));
            }
        }
    }
}

struct InjectedIncidentConnectionCleanup {
    id: AudioNodeId,
    endpoint: InjectedConnectionEndpointStamp,
    registry: Weak<InjectedConnectionRegistryInner>,
}

impl InjectedNodeReclaimCleanup for InjectedIncidentConnectionCleanup {
    fn reconcile(&mut self, id: AudioNodeId) -> Result<(), NodeReclaimCleanupError> {
        if id != self.id || self.endpoint.id != id {
            return Err(NodeReclaimCleanupError::Rejected);
        }
        let Some(registry) = self.registry.upgrade() else {
            return Err(NodeReclaimCleanupError::Rejected);
        };
        let mut state = match registry.serializer.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(NodeReclaimCleanupError::RetryContended),
            Err(TryLockError::Poisoned(_)) => {
                registry.fail_closed_protocol();
                return Err(NodeReclaimCleanupError::Rejected);
            }
        };
        state
            .edges
            .retain(|edge| !edge.is_incident_to(self.endpoint));
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct InjectedConnectionRegistryRetirement {
    pub(crate) residual_edges_cleared: usize,
    pub(crate) protocol_failed: bool,
    pub(crate) serializer_poison_recovered: bool,
    pub(crate) ownership_mismatch: bool,
}
