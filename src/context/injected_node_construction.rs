//! Branded, private injected-node construction transactions.
//!
//! This is a production capability but not a public context constructor. It binds the exact
//! control transport, node-id allocator, and lifetime registry, then implements the first concrete
//! two-node Gain transaction without exposing any of those authorities separately.

use std::panic::{self, AssertUnwindSafe};
#[cfg(test)]
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;

use super::injected_connections::{
    InjectedConnectionEndpoint, InjectedConnectionEndpointKind, InjectedConnectionOperationError,
    InjectedConnectionOperationOutcome, InjectedConnectionRegistryInner,
    InjectedDisconnectSelector,
};
use super::injected_control::{
    AcceptedBatchFinalizeFailure, CommitControlOutcome, CommitWithFinalizeFailure,
    ControlBatchReservation, InjectedControlError, InjectedControlIdentity,
    InjectedControlProducer, RejectedControlRollback,
};
use super::injected_ids::{
    InjectedNodeIdAllocator, InjectedNodeIdIdentity, ProvisionalNodeIdError, ProvisionalNodeIds,
};
use super::injected_node_lifetime::{
    InjectedNodeLifetimeRegistrar, InjectedNodeReclaimCleanup, InjectedNodeRegistration,
    InjectedNodeRegistrationIdentity, NodeReclaimCleanupError, NodeRegistrationError,
    ProvisionalNodeRegistration,
};
use super::{AudioContextRegistration, AudioNodeId};
use crate::message::ControlMessage;
use crate::node::ChannelConfigInner;
use crate::param::{
    AudioParamInitialValue, AudioParamInner, InjectedAudioParamMirror, InjectedAudioParamProcessor,
    InjectedAudioParamValue,
};
use crate::render::AudioProcessor;

const GAIN_COMMAND_COUNT: usize = 4;
const GAIN_NODE_COUNT: usize = 2;
const GAIN_ID_INDEX: usize = 0;
const PARAM_ID_INDEX: usize = 1;

/// Opaque capability created only after exact control, allocator, and lifetime identities match.
pub(crate) struct InjectedNodeConstructor {
    control: InjectedControlProducer,
    allocator: InjectedNodeIdAllocator,
    lifetimes: InjectedNodeLifetimeRegistrar,
    #[cfg(test)]
    magic_behavior: AtomicU8,
    #[cfg(test)]
    magic_drop_probe: Mutex<Option<MagicPayloadDropProbe>>,
}

#[cfg(test)]
pub(super) struct MagicPayloadDropProbe {
    pub(super) started: crossbeam_channel::Sender<()>,
    pub(super) release: crossbeam_channel::Receiver<()>,
    pub(super) tokens_restored: Arc<std::sync::atomic::AtomicBool>,
    pub(super) panic_after_release: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // returned by the private builder before a public injected context is wired
pub(crate) enum InjectedNodeConstructorBuildError {
    MismatchedCapabilities,
}

#[allow(dead_code)] // exact failure ownership is exercised at the private integration boundary
pub(crate) struct InjectedNodeConstructorBuildFailure {
    pub(crate) error: InjectedNodeConstructorBuildError,
    pub(crate) control: InjectedControlProducer,
    pub(crate) allocator: InjectedNodeIdAllocator,
    pub(crate) lifetimes: InjectedNodeLifetimeRegistrar,
}

impl InjectedNodeConstructor {
    pub(super) const fn control(&self) -> &InjectedControlProducer {
        &self.control
    }

    pub(super) const fn allocator(&self) -> &InjectedNodeIdAllocator {
        &self.allocator
    }

    pub(super) fn registry_identity(
        &self,
    ) -> std::sync::Weak<super::injected_node_lifetime::NodeLifetimeInner> {
        self.lifetimes.registry_identity()
    }

    pub(crate) fn matches_node_id_identity(
        &self,
        identity: &super::injected_ids::InjectedNodeIdIdentity,
    ) -> bool {
        self.allocator.identity().ptr_eq(identity)
    }

    #[cfg(test)]
    pub(super) fn take_magic_behavior_for_test(&self) -> u8 {
        self.magic_behavior.swap(0, Ordering::AcqRel)
    }

    #[cfg(test)]
    pub(crate) fn set_magic_behavior_for_test(&self, behavior: u8) {
        self.magic_behavior.store(behavior, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn set_magic_drop_probe_for_test(
        &self,
        started: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
        panic_after_release: bool,
    ) -> Arc<std::sync::atomic::AtomicBool> {
        let tokens_restored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        *self.magic_drop_probe.lock().unwrap() = Some(MagicPayloadDropProbe {
            started,
            release,
            tokens_restored: Arc::clone(&tokens_restored),
            panic_after_release,
        });
        tokens_restored
    }

    #[cfg(test)]
    pub(super) fn take_magic_drop_probe_for_test(&self) -> Option<MagicPayloadDropProbe> {
        self.magic_drop_probe.lock().unwrap().take()
    }

    #[allow(dead_code)] // selected by the deferred public injected context constructor
    #[allow(clippy::result_large_err)] // exact weak capabilities must be returned intact
    pub(crate) fn new(
        control: InjectedControlProducer,
        allocator: InjectedNodeIdAllocator,
        lifetimes: InjectedNodeLifetimeRegistrar,
    ) -> Result<Self, InjectedNodeConstructorBuildFailure> {
        if !lifetimes.matches_constructor(&control, &allocator) {
            return Err(InjectedNodeConstructorBuildFailure {
                error: InjectedNodeConstructorBuildError::MismatchedCapabilities,
                control,
                allocator,
                lifetimes,
            });
        }
        Ok(Self {
            control,
            allocator,
            lifetimes,
            #[cfg(test)]
            magic_behavior: AtomicU8::new(0),
            #[cfg(test)]
            magic_drop_probe: Mutex::new(None),
        })
    }

    pub(crate) fn try_begin_gain(
        &self,
    ) -> Result<InjectedGainConstruction, InjectedGainConstructionError> {
        // This admission must precede every ID, lifetime-slot, mirror, or payload mutation.
        let reservation = self
            .control
            .try_begin_operation(GAIN_COMMAND_COUNT)
            .map_err(InjectedGainConstructionError::Control)?;
        // Allocate the post-construction serializer while every later Gain resource is still
        // rollback-owned by this admitted transaction.
        let param_serializer = Arc::new(Mutex::new(()));
        #[cfg(test)]
        let param_finalizer_hook = Arc::new(Mutex::new(None));
        #[cfg(test)]
        let param_rollback_hook = Arc::new(Mutex::new(None));
        #[cfg(test)]
        let param_serializer_attempt = Arc::new(Mutex::new(None));
        let ids = self
            .allocator
            .try_reserve(GAIN_NODE_COUNT)
            .map_err(InjectedGainConstructionError::NodeIds)?;
        let gain_id = ids.id(GAIN_ID_INDEX);
        let param_id = ids.id(PARAM_ID_INDEX);

        let (gain, gain_connection) =
            self.register_endpoint(gain_id, InjectedConnectionEndpointKind::AudioNode, 1, 1)?;
        let (param, param_connection) = match self.register_endpoint(
            param_id,
            InjectedConnectionEndpointKind::AudioParam,
            1,
            1,
        ) {
            Ok(param) => param,
            Err(error) => {
                drop(gain);
                return Err(error);
            }
        };
        Ok(InjectedGainConstruction {
            control: self.control.clone(),
            ids,
            gain,
            param,
            gain_connection,
            param_connection,
            gain_id,
            param_id,
            param_serializer,
            #[cfg(test)]
            param_finalizer_hook,
            #[cfg(test)]
            param_rollback_hook,
            #[cfg(test)]
            param_serializer_attempt,
            #[cfg(test)]
            id_corruption: 0,
            #[cfg(test)]
            foreign_reclaim: None,
            // Admission is deliberately declared last: ordinary field destruction therefore
            // cancels both lifetime provisionals and returns both exact IDs before it releases
            // graph admission, including unwinds while the caller constructs the payload.
            reservation: Some(reservation),
        })
    }

    fn register_endpoint(
        &self,
        id: AudioNodeId,
        kind: InjectedConnectionEndpointKind,
        inputs: usize,
        outputs: usize,
    ) -> Result<
        (ProvisionalNodeRegistration, InjectedConnectionEndpoint),
        InjectedGainConstructionError,
    > {
        let cleanup: Box<dyn InjectedNodeReclaimCleanup> =
            Box::new(DeferredIncidentConnectionCleanup { id });
        let provisional = self
            .lifetimes
            .try_register(id, cleanup)
            .map_err(|failure| {
                let error = failure.error;
                drop(failure);
                InjectedGainConstructionError::Registration(error)
            })?;
        let endpoint = InjectedConnectionEndpoint::new_ordinary(
            self.lifetimes.registry_identity(),
            self.control.identity(),
            self.allocator.identity(),
            kind,
            inputs,
            outputs,
            provisional.stamp(),
        );
        let Some(cleanup) = endpoint.incident_cleanup() else {
            drop(provisional);
            return Err(InjectedGainConstructionError::Registration(
                NodeRegistrationError::OwnerGone,
            ));
        };
        let previous = provisional
            .replace_cleanup_before_acceptance(cleanup)
            .map_err(|cleanup| {
                drop(cleanup);
                InjectedGainConstructionError::Registration(
                    NodeRegistrationError::ProtocolViolation,
                )
            })?;
        drop(previous);
        Ok((provisional, endpoint))
    }

    pub(crate) fn applied_batch_sequence(&self) -> u64 {
        self.control.applied_batch_sequence()
    }

    #[allow(dead_code)] // selected by the private magic bootstrap before public builder wiring
    pub(crate) fn try_flush_staged(
        &self,
    ) -> Result<super::injected_control::FlushControlOutcome, InjectedControlError> {
        self.control.try_flush()
    }

    #[allow(dead_code)] // selected by the private magic bootstrap before public builder wiring
    pub(crate) fn last_submitted_batch_sequence(&self) -> u64 {
        self.control.last_submitted_batch_sequence()
    }

    pub(crate) fn admission_gate(&self) -> super::InjectedContextAdmissionGate {
        self.control.admission_gate()
    }

    pub(crate) fn matches_control_identity(&self, identity: &InjectedControlIdentity) -> bool {
        self.control.identity().ptr_eq(identity)
    }

    pub(super) fn fail_closed_protocol(&self) {
        self.control.fail_closed_protocol();
        if let Some(owner) = self.lifetimes.registry_identity().upgrade() {
            owner.connection_registry().fail_closed_protocol();
        }
    }

    #[cfg(test)]
    pub(crate) fn hold_connection_operation_for_test(
        &self,
        point: super::injected_connections::InjectedConnectionOperationTestPoint,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
        panics: bool,
    ) {
        self.lifetimes
            .registry_identity()
            .upgrade()
            .expect("test retains exact lifetime owner")
            .connection_registry()
            .hold_operation_for_test(point, entered, release, panics);
    }

    #[cfg(test)]
    pub(crate) fn connection_edge_count_for_test(&self) -> usize {
        self.lifetimes
            .registry_identity()
            .upgrade()
            .expect("test retains exact lifetime owner")
            .connection_registry()
            .edge_count_for_test()
    }

    #[cfg(test)]
    pub(crate) fn connection_transport_accounting_for_test(&self) -> (usize, usize, usize, usize) {
        self.control.accounting()
    }

    pub(crate) fn connect_exact(
        &self,
        source: &InjectedConnectionEndpoint,
        destination: &InjectedConnectionEndpoint,
        output: usize,
        input: usize,
    ) -> Result<InjectedConnectionOperationOutcome, InjectedConnectionOperationError> {
        InjectedConnectionRegistryInner::connect(
            &self.control,
            &self.allocator.identity(),
            &self.lifetimes.registry_identity(),
            source,
            destination,
            output,
            input,
        )
    }

    pub(crate) fn disconnect_exact(
        &self,
        source: &InjectedConnectionEndpoint,
        selector: InjectedDisconnectSelector<'_>,
    ) -> Result<InjectedConnectionOperationOutcome, InjectedConnectionOperationError> {
        InjectedConnectionRegistryInner::disconnect(
            &self.control,
            &self.allocator.identity(),
            &self.lifetimes.registry_identity(),
            source,
            selector,
        )
    }
}

/// Short-lived exact-ID cleanup used only while a Gain registration remains provisional.
///
/// `register_endpoint` replaces this value with the exact incident-edge cleanup before accepted
/// publication. It may therefore run only during rollback; it must never survive in a live Gain
/// slot. The hidden parameter edge remains deliberately outside the public connection mirror.
struct DeferredIncidentConnectionCleanup {
    id: AudioNodeId,
}

impl InjectedNodeReclaimCleanup for DeferredIncidentConnectionCleanup {
    fn reconcile(&mut self, id: AudioNodeId) -> Result<(), NodeReclaimCleanupError> {
        if id != self.id {
            return Err(NodeReclaimCleanupError::Rejected);
        }
        Ok(())
    }
}

pub(crate) struct InjectedGainPayload {
    pub(crate) param_processor: InjectedAudioParamProcessor,
    pub(crate) gain_processor: Box<dyn AudioProcessor>,
    pub(crate) param_channel_config: ChannelConfigInner,
    pub(crate) gain_channel_config: ChannelConfigInner,
    pub(crate) initial_value: AudioParamInitialValue,
}

pub(crate) struct InjectedConstructedGain {
    pub(crate) gain_id: AudioNodeId,
    pub(crate) param_id: AudioNodeId,
    pub(crate) gain_registration: InjectedNodeRegistration,
    pub(crate) param_registration: InjectedNodeRegistration,
    pub(crate) gain_connection: InjectedConnectionEndpoint,
    pub(crate) param_connection: InjectedConnectionEndpoint,
    pub(crate) param_mutation: InjectedAudioParamMutation,
    pub(crate) outcome: CommitControlOutcome,
}

/// Exact post-construction capability for the one supported injected AudioParam mutation.
///
/// The serializer is allocated while the Gain transaction can still roll back. Clones share that
/// ordering, the exact host mirror, and weak control/allocator/slot-generation brands, but retain
/// no transport, admission, or live-registration credit.
#[derive(Clone)]
pub(crate) struct InjectedAudioParamMutation {
    control: InjectedControlProducer,
    node_ids: InjectedNodeIdIdentity,
    param_id: AudioNodeId,
    lifetime: InjectedNodeRegistrationIdentity,
    serializer: Arc<Mutex<()>>,
    mirror: InjectedAudioParamMirror,
    #[cfg(test)]
    finalizer_hook: Arc<Mutex<Option<AudioParamFinalizerHook>>>,
    #[cfg(test)]
    rollback_hook: Arc<Mutex<Option<AudioParamRollbackHook>>>,
    #[cfg(test)]
    serializer_attempt: Arc<Mutex<Option<crossbeam_channel::Sender<()>>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedAudioParamMutationError {
    Control(InjectedControlError),
    AcceptedFinalizer(AcceptedBatchFinalizeFailure),
    SerializerPoisoned,
    RejectedPayloadPanicked,
    ProtocolViolation,
}

impl InjectedAudioParamMutation {
    pub(crate) fn matches_registration(
        &self,
        registration: &AudioContextRegistration,
        constructor: &InjectedNodeConstructor,
        raw_parts: &AudioParamInner,
    ) -> bool {
        self.param_id == registration.id()
            && constructor.matches_control_identity(&self.control.identity())
            && constructor.matches_node_id_identity(&self.node_ids)
            && registration.matches_injected_lifetime_identity(&self.lifetime)
            && self.mirror.matches_inner(raw_parts)
    }

    /// Commits one fixed value update. Expected transport rejection is returned only after the
    /// serializer guard is gone, so the public panic-shaped API cannot poison this lock.
    pub(crate) fn try_set_value(
        &self,
        value: InjectedAudioParamValue,
        clamped: f32,
    ) -> Result<CommitControlOutcome, InjectedAudioParamMutationError> {
        let result = {
            #[cfg(test)]
            if let Some(attempted) = self.serializer_attempt.lock().unwrap().take() {
                attempted.send(()).unwrap();
            }
            let _serialized = self
                .serializer
                .lock()
                .map_err(|_| InjectedAudioParamMutationError::SerializerPoisoned)?;
            let reservation = self
                .control
                .try_begin_audio_param_value()
                .map_err(InjectedAudioParamMutationError::Control)?;
            let expected_bits = value.get().to_bits();
            let prepared = reservation.prepare(self.param_id, value);
            let mirror = &self.mirror;
            #[cfg(test)]
            let finalizer_hook = &self.finalizer_hook;
            match self.control.try_commit_with_finalize(prepared, move |_| {
                #[cfg(test)]
                if let Some(hook) = finalizer_hook.lock().unwrap().take() {
                    hook.entered.send(()).unwrap();
                    hook.release.recv().unwrap();
                    assert!(!hook.panics, "forced AudioParam accepted-finalizer panic");
                }
                mirror.store(clamped);
                Ok(())
            }) {
                Ok(outcome) => Ok(outcome),
                Err(CommitWithFinalizeFailure::AcceptedFinalizer(failure)) => {
                    Err(InjectedAudioParamMutationError::AcceptedFinalizer(failure))
                }
                Err(CommitWithFinalizeFailure::NotAccepted(failure)) => {
                    let control = self.control.clone();
                    let param_id = self.param_id;
                    #[cfg(test)]
                    let rollback_hook = &self.rollback_hook;
                    let (error, rollback) = failure.rollback_with_commands(move |commands| {
                        let mut fail_closed = FailClosedParamRollback::new(control);
                        let mut commands = commands.into_vec().into_iter();
                        let exact = matches!(
                            (commands.next(), commands.next()),
                            (
                                Some(ControlMessage::InjectedAudioParamValue { id, value }),
                                None
                            ) if id == param_id && value.get().to_bits() == expected_bits
                        );
                        #[cfg(test)]
                        if let Some(hook) = rollback_hook.lock().unwrap().take() {
                            hook.entered.send(()).unwrap();
                            hook.release.recv().unwrap();
                            assert!(!hook.panics, "forced AudioParam rejected rollback panic");
                        }
                        if exact {
                            fail_closed.disarm();
                        }
                        exact
                    });
                    match rollback {
                        RejectedControlRollback::Completed(true) => {
                            Err(InjectedAudioParamMutationError::Control(error))
                        }
                        RejectedControlRollback::Completed(false) => {
                            Err(InjectedAudioParamMutationError::ProtocolViolation)
                        }
                        RejectedControlRollback::Panicked => {
                            Err(InjectedAudioParamMutationError::RejectedPayloadPanicked)
                        }
                    }
                }
            }
        };
        result
    }

    pub(crate) fn fail_closed_protocol(&self) {
        self.control.fail_closed_protocol();
    }

    #[cfg(test)]
    pub(crate) fn hold_next_finalizer_for_test(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
        panics: bool,
    ) {
        *self.finalizer_hook.lock().unwrap() = Some(AudioParamFinalizerHook {
            entered,
            release,
            panics,
        });
    }

    #[cfg(test)]
    pub(crate) fn hold_next_rollback_for_test(
        &self,
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
        panics: bool,
    ) {
        *self.rollback_hook.lock().unwrap() = Some(AudioParamRollbackHook {
            entered,
            release,
            panics,
        });
    }

    #[cfg(test)]
    pub(crate) fn signal_next_serializer_attempt_for_test(
        &self,
        attempted: crossbeam_channel::Sender<()>,
    ) {
        *self.serializer_attempt.lock().unwrap() = Some(attempted);
    }

    #[cfg(test)]
    pub(crate) fn replace_node_id_identity_for_test(&mut self, identity: InjectedNodeIdIdentity) {
        self.node_ids = identity;
    }
}

#[cfg(test)]
struct AudioParamFinalizerHook {
    entered: crossbeam_channel::Sender<()>,
    release: crossbeam_channel::Receiver<()>,
    panics: bool,
}

#[cfg(test)]
struct AudioParamRollbackHook {
    entered: crossbeam_channel::Sender<()>,
    release: crossbeam_channel::Receiver<()>,
    panics: bool,
}

struct FailClosedParamRollback {
    control: InjectedControlProducer,
    armed: bool,
}

impl FailClosedParamRollback {
    fn new(control: InjectedControlProducer) -> Self {
        Self {
            control,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for FailClosedParamRollback {
    fn drop(&mut self) {
        if self.armed {
            self.control.fail_closed_protocol();
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedGainConstructionError {
    Control(InjectedControlError),
    NodeIds(ProvisionalNodeIdError),
    Registration(NodeRegistrationError),
    AcceptedFinalizer(AcceptedBatchFinalizeFailure),
    RejectedPayloadPanicked,
    ProtocolViolation,
}

/// Owns one admitted construction from ID reservation through accepted finalization or complete
/// preaccept rollback. It never escapes into a public node.
pub(crate) struct InjectedGainConstruction {
    control: InjectedControlProducer,
    ids: ProvisionalNodeIds,
    gain: ProvisionalNodeRegistration,
    param: ProvisionalNodeRegistration,
    gain_connection: InjectedConnectionEndpoint,
    param_connection: InjectedConnectionEndpoint,
    gain_id: AudioNodeId,
    param_id: AudioNodeId,
    param_serializer: Arc<Mutex<()>>,
    #[cfg(test)]
    param_finalizer_hook: Arc<Mutex<Option<AudioParamFinalizerHook>>>,
    #[cfg(test)]
    param_rollback_hook: Arc<Mutex<Option<AudioParamRollbackHook>>>,
    #[cfg(test)]
    param_serializer_attempt: Arc<Mutex<Option<crossbeam_channel::Sender<()>>>>,
    #[cfg(test)]
    id_corruption: u8,
    #[cfg(test)]
    foreign_reclaim: Option<(usize, llq::Node<AudioNodeId>)>,
    reservation: Option<ControlBatchReservation>,
}

impl InjectedGainConstruction {
    pub(crate) const fn gain_id(&self) -> AudioNodeId {
        self.gain_id
    }

    pub(crate) const fn param_id(&self) -> AudioNodeId {
        self.param_id
    }

    #[cfg(test)]
    pub(crate) fn corrupt_id_proof_for_test(&mut self, failure_point: u8) {
        assert!((1..=3).contains(&failure_point));
        self.id_corruption = failure_point;
        if failure_point == 1 {
            let token = self.ids.take_reclaim_node(PARAM_ID_INDEX).unwrap();
            std::mem::forget(token);
        } else if failure_point == 2 {
            let token = self.ids.take_reclaim_node(GAIN_ID_INDEX).unwrap();
            std::mem::forget(token);
        }
    }

    #[cfg(test)]
    pub(crate) fn replace_reclaim_for_test(
        &mut self,
        index: usize,
        foreign: llq::Node<AudioNodeId>,
    ) {
        assert!(index < GAIN_NODE_COUNT);
        self.foreign_reclaim = Some((index, foreign));
    }

    pub(crate) fn commit(
        mut self,
        payload: InjectedGainPayload,
    ) -> Result<InjectedConstructedGain, InjectedGainConstructionError> {
        let InjectedGainPayload {
            param_processor,
            gain_processor,
            param_channel_config,
            gain_channel_config,
            initial_value,
        } = payload;
        // Validate the exact renderer/host mirror while the whole Gain transaction can still roll
        // back, then perform the only processor boxing allocation before moving reclaim tokens.
        let (param_processor, mirror) = match param_processor.into_boxed_prevalidated() {
            Ok(parts) => parts,
            Err(mismatch) => {
                let destructor_panicked =
                    self.rollback_param_processor_mismatch(mismatch, gain_processor);
                return Err(if destructor_panicked {
                    InjectedGainConstructionError::RejectedPayloadPanicked
                } else {
                    InjectedGainConstructionError::ProtocolViolation
                });
            }
        };
        let param_mutation = InjectedAudioParamMutation {
            control: self.control.clone(),
            node_ids: self.ids.identity(),
            param_id: self.param_id,
            lifetime: self.param.identity(),
            serializer: Arc::clone(&self.param_serializer),
            mirror,
            #[cfg(test)]
            finalizer_hook: Arc::clone(&self.param_finalizer_hook),
            #[cfg(test)]
            rollback_hook: Arc::clone(&self.param_rollback_hook),
            #[cfg(test)]
            serializer_attempt: Arc::clone(&self.param_serializer_attempt),
        };
        // Allocate command storage before moving either exact reclaim node. Every later command
        // construction is a closed struct move with the exact capacity already reserved.
        let mut commands = Vec::with_capacity(GAIN_COMMAND_COUNT);
        let param_reclaim = match self.ids.take_reclaim_node(PARAM_ID_INDEX) {
            Ok(node) => node,
            Err(error) => {
                self.fail_closed_id_proof();
                return Err(InjectedGainConstructionError::NodeIds(error));
            }
        };
        let gain_reclaim = match self.ids.take_reclaim_node(GAIN_ID_INDEX) {
            Ok(node) => node,
            Err(error) => {
                if let Err(failure) = self.ids.restore_reclaim_node(PARAM_ID_INDEX, param_reclaim) {
                    std::mem::forget(failure.node);
                }
                self.fail_closed_id_proof();
                return Err(InjectedGainConstructionError::NodeIds(error));
            }
        };
        commands.push(ControlMessage::RegisterNode {
            id: self.param_id,
            reclaim_id: param_reclaim,
            node: param_processor,
            inputs: 1,
            outputs: 1,
            channel_config: param_channel_config,
        });
        commands.push(ControlMessage::AudioParamInitialValue {
            id: self.param_id,
            value: initial_value,
        });
        commands.push(ControlMessage::RegisterNode {
            id: self.gain_id,
            reclaim_id: gain_reclaim,
            node: gain_processor,
            inputs: 1,
            outputs: 1,
            channel_config: gain_channel_config,
        });
        commands.push(ControlMessage::ConnectNode {
            from: self.param_id,
            to: self.gain_id,
            output: 0,
            input: usize::MAX,
        });

        #[cfg(test)]
        if self.id_corruption == 3 {
            self.ids
                .restore_reclaim_node(GAIN_ID_INDEX, llq::Node::new(self.gain_id))
                .unwrap();
        }
        #[cfg(test)]
        if let Some((index, foreign)) = self.foreign_reclaim.take() {
            let command_index = if index == PARAM_ID_INDEX { 0 } else { 2 };
            let ControlMessage::RegisterNode { reclaim_id, .. } = &mut commands[command_index]
            else {
                unreachable!("fixed Gain command index is RegisterNode")
            };
            let exact = std::mem::replace(reclaim_id, foreign);
            std::mem::forget(exact);
        }

        let id_commit = match self.ids.commit_token() {
            Ok(commit) => commit,
            Err(error) => {
                self.fail_closed_id_proof();
                return Err(InjectedGainConstructionError::NodeIds(error));
            }
        };
        let gain_arm = self.gain.arm_token();
        let param_arm = self.param.arm_token();
        let batch = self
            .reservation
            .take()
            .expect("one Gain transaction owns one reservation")
            .into_prevalidated(commands);
        let committed = self.control.try_commit_with_finalize(batch, move |_| {
            // Mark both graph-owned provisionals non-cancelable before either arm can fail.
            param_arm.mark_accepted();
            gain_arm.mark_accepted();
            id_commit.commit_accepted();
            param_arm.arm_accepted()?;
            gain_arm.arm_accepted()
        });
        match committed {
            Ok(outcome) => self.finish_accepted(outcome, param_mutation),
            Err(CommitWithFinalizeFailure::AcceptedFinalizer(failure)) => {
                Err(InjectedGainConstructionError::AcceptedFinalizer(failure))
            }
            Err(CommitWithFinalizeFailure::NotAccepted(failure)) => {
                self.rollback_not_accepted(failure, param_mutation)
            }
        }
    }

    fn finish_accepted(
        self,
        outcome: CommitControlOutcome,
        param_mutation: InjectedAudioParamMutation,
    ) -> Result<InjectedConstructedGain, InjectedGainConstructionError> {
        let Self {
            control: _,
            ids,
            gain,
            param,
            gain_connection,
            param_connection,
            gain_id,
            param_id,
            param_serializer: _,
            #[cfg(test)]
                param_finalizer_hook: _,
            #[cfg(test)]
                param_rollback_hook: _,
            #[cfg(test)]
                param_serializer_attempt: _,
            #[cfg(test)]
                id_corruption: _,
            #[cfg(test)]
                foreign_reclaim: _,
            reservation: _,
        } = self;
        drop(ids);
        if !param.ready_for_registration() || !gain.ready_for_registration() {
            param.arm_token().quarantine_accepted();
            gain.arm_token().quarantine_accepted();
            return Err(InjectedGainConstructionError::ProtocolViolation);
        }
        let param_registration = param
            .into_registration()
            .expect("preflighted accepted param registration remains exact");
        let gain_registration = gain
            .into_registration()
            .expect("preflighted accepted Gain registration remains exact");
        Ok(InjectedConstructedGain {
            gain_id,
            param_id,
            gain_registration,
            param_registration,
            gain_connection,
            param_connection,
            param_mutation,
            outcome,
        })
    }

    fn rollback_not_accepted(
        self,
        failure: super::injected_control::CommitControlFailure,
        param_mutation: InjectedAudioParamMutation,
    ) -> Result<InjectedConstructedGain, InjectedGainConstructionError> {
        let Self {
            control: _,
            mut ids,
            gain,
            param,
            gain_connection: _,
            param_connection: _,
            gain_id,
            param_id,
            param_serializer: _,
            #[cfg(test)]
                param_finalizer_hook: _,
            #[cfg(test)]
                param_rollback_hook: _,
            #[cfg(test)]
                param_serializer_attempt: _,
            #[cfg(test)]
                id_corruption: _,
            #[cfg(test)]
                foreign_reclaim: _,
            reservation: _,
        } = self;
        let (control_error, rollback) = failure.rollback_with_commands(move |commands| {
            let recovery = recover_gain_commands(commands, &mut ids, gain_id, param_id);
            if !recovery.exact {
                gain.arm_token().quarantine_accepted();
                param.arm_token().quarantine_accepted();
                ids.retain_unavailable();
            }
            // These destructor-bearing guards and the ID reservation must complete while the
            // rejected batch still owns graph admission.
            drop(param);
            drop(gain);
            drop(param_mutation);
            drop(ids);
            recovery
        });
        match rollback {
            RejectedControlRollback::Panicked => {
                Err(InjectedGainConstructionError::RejectedPayloadPanicked)
            }
            RejectedControlRollback::Completed(recovery) if !recovery.exact => {
                Err(InjectedGainConstructionError::ProtocolViolation)
            }
            RejectedControlRollback::Completed(recovery) if recovery.destructor_panicked => {
                Err(InjectedGainConstructionError::RejectedPayloadPanicked)
            }
            RejectedControlRollback::Completed(_) => {
                Err(InjectedGainConstructionError::Control(control_error))
            }
        }
    }

    fn fail_closed_id_proof(&self) {
        self.gain.arm_token().quarantine_accepted();
        self.param.arm_token().quarantine_accepted();
        self.ids.retain_unavailable();
    }

    fn rollback_param_processor_mismatch(
        self,
        param_processor: InjectedAudioParamProcessor,
        gain_processor: Box<dyn AudioProcessor>,
    ) -> bool {
        let Self {
            control,
            ids,
            gain,
            param,
            gain_connection: _,
            param_connection: _,
            gain_id: _,
            param_id: _,
            param_serializer,
            #[cfg(test)]
            param_finalizer_hook,
            #[cfg(test)]
            param_rollback_hook,
            #[cfg(test)]
            param_serializer_attempt,
            #[cfg(test)]
                id_corruption: _,
            #[cfg(test)]
                foreign_reclaim: _,
            reservation,
        } = self;
        // Exact IDs and provisional lifetime slots roll back before either processor destructor;
        // the original admitted reservation remains the last released authority.
        drop(param);
        drop(gain);
        drop(ids);
        let param_panicked = match panic::catch_unwind(AssertUnwindSafe(|| drop(param_processor))) {
            Ok(()) => false,
            Err(payload) => {
                std::mem::forget(payload);
                true
            }
        };
        let gain_panicked = match panic::catch_unwind(AssertUnwindSafe(|| drop(gain_processor))) {
            Ok(()) => false,
            Err(payload) => {
                std::mem::forget(payload);
                true
            }
        };
        drop(param_serializer);
        #[cfg(test)]
        drop(param_finalizer_hook);
        #[cfg(test)]
        drop(param_rollback_hook);
        #[cfg(test)]
        drop(param_serializer_attempt);
        drop(control);
        drop(reservation);
        param_panicked || gain_panicked
    }
}

#[derive(Clone, Copy)]
struct RejectedGainRecovery {
    exact: bool,
    destructor_panicked: bool,
}

fn recover_gain_commands(
    commands: Box<[ControlMessage]>,
    ids: &mut ProvisionalNodeIds,
    gain_id: AudioNodeId,
    param_id: AudioNodeId,
) -> RejectedGainRecovery {
    let exact_len = commands.len() == GAIN_COMMAND_COUNT;
    let mut exact = exact_len;
    let mut saw_gain = false;
    let mut saw_param = false;
    let mut processors: ArrayVec<Box<dyn AudioProcessor>, GAIN_COMMAND_COUNT> = ArrayVec::new();
    let mut other: ArrayVec<ControlMessage, GAIN_COMMAND_COUNT> = ArrayVec::new();

    for (index, command) in commands.into_vec().into_iter().enumerate() {
        match command {
            ControlMessage::RegisterNode {
                id,
                reclaim_id,
                node,
                inputs,
                outputs,
                channel_config: _,
            } => {
                let restore = if id == param_id && !saw_param {
                    saw_param = true;
                    exact &= index == 0 && inputs == 1 && outputs == 1;
                    if *reclaim_id == param_id {
                        Some(ids.restore_reclaim_node(PARAM_ID_INDEX, reclaim_id))
                    } else {
                        exact = false;
                        std::mem::forget(reclaim_id);
                        None
                    }
                } else if id == gain_id && !saw_gain {
                    saw_gain = true;
                    exact &= index == 2 && inputs == 1 && outputs == 1;
                    if *reclaim_id == gain_id {
                        Some(ids.restore_reclaim_node(GAIN_ID_INDEX, reclaim_id))
                    } else {
                        exact = false;
                        std::mem::forget(reclaim_id);
                        None
                    }
                } else {
                    std::mem::forget(reclaim_id);
                    exact = false;
                    None
                };
                if let Some(restore) = restore {
                    match restore {
                        Ok(()) => {}
                        Err(failure) => {
                            exact = false;
                            std::mem::forget(failure.node);
                        }
                    }
                }
                if let Err(error) = processors.try_push(node) {
                    exact = false;
                    std::mem::forget(error.element());
                }
            }
            command @ ControlMessage::AudioParamInitialValue { id, .. } => {
                exact &= index == 1 && id == param_id;
                if let Err(error) = other.try_push(command) {
                    exact = false;
                    std::mem::forget(error.element());
                }
            }
            command @ ControlMessage::ConnectNode {
                from,
                to,
                output,
                input,
            } => {
                exact &= index == 3
                    && from == param_id
                    && to == gain_id
                    && output == 0
                    && input == usize::MAX;
                if let Err(error) = other.try_push(command) {
                    exact = false;
                    std::mem::forget(error.element());
                }
            }
            command => {
                exact = false;
                if let Err(error) = other.try_push(command) {
                    std::mem::forget(error.element());
                }
            }
        }
    }
    exact &= saw_gain && saw_param;

    // Both exact reclaim nodes are restored before any processor or other command destructor.
    let destructor_panicked = match panic::catch_unwind(AssertUnwindSafe(|| {
        drop(other);
        drop(processors);
    })) {
        Ok(()) => false,
        Err(payload) => {
            std::mem::forget(payload);
            true
        }
    };
    RejectedGainRecovery {
        exact,
        destructor_panicked,
    }
}
