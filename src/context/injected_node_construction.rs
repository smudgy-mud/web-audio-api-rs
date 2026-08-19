//! Branded, private injected-node construction transactions.
//!
//! This is a production capability but not a public context constructor. It binds the exact
//! control transport, node-id allocator, and lifetime registry, then implements the first concrete
//! two-node Gain transaction without exposing any of those authorities separately.

use std::panic::{self, AssertUnwindSafe};

use arrayvec::ArrayVec;

use super::injected_control::{
    AcceptedBatchFinalizeFailure, CommitControlOutcome, CommitWithFinalizeFailure,
    ControlBatchReservation, InjectedControlError, InjectedControlProducer,
    RejectedControlRollback,
};
use super::injected_ids::{InjectedNodeIdAllocator, ProvisionalNodeIdError, ProvisionalNodeIds};
use super::injected_node_lifetime::{
    InjectedNodeLifetimeRegistrar, InjectedNodeReclaimCleanup, InjectedNodeRegistration,
    NodeReclaimCleanupError, NodeRegistrationError, ProvisionalNodeRegistration,
};
use super::AudioNodeId;
use crate::message::ControlMessage;
use crate::node::ChannelConfigInner;
use crate::param::AudioParamInitialValue;
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
        let ids = self
            .allocator
            .try_reserve(GAIN_NODE_COUNT)
            .map_err(InjectedGainConstructionError::NodeIds)?;
        let gain_id = ids.id(GAIN_ID_INDEX);
        let param_id = ids.id(PARAM_ID_INDEX);

        let gain = self.register(gain_id)?;
        let param = match self.register(param_id) {
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
            gain_id,
            param_id,
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

    fn register(
        &self,
        id: AudioNodeId,
    ) -> Result<ProvisionalNodeRegistration, InjectedGainConstructionError> {
        let cleanup: Box<dyn InjectedNodeReclaimCleanup> =
            Box::new(DeferredIncidentConnectionCleanup { id });
        self.lifetimes.try_register(id, cleanup).map_err(|failure| {
            let error = failure.error;
            drop(failure);
            InjectedGainConstructionError::Registration(error)
        })
    }

    pub(crate) fn applied_batch_sequence(&self) -> u64 {
        self.control.applied_batch_sequence()
    }

    pub(crate) fn admission_gate(&self) -> super::InjectedContextAdmissionGate {
        self.control.admission_gate()
    }
}

/// Exact-ID cleanup placeholder for the private construction boundary.
///
/// The hidden parameter edge is deliberately not mirrored by legacy construction either. The
/// public injected context and its real explicit-connection administration are deferred; that
/// later integration must replace this placeholder with cleanup against the actual base mirror.
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
    pub(crate) param_processor: Box<dyn AudioProcessor>,
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
    pub(crate) outcome: CommitControlOutcome,
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
    gain_id: AudioNodeId,
    param_id: AudioNodeId,
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
            node: payload.param_processor,
            inputs: 1,
            outputs: 1,
            channel_config: payload.param_channel_config,
        });
        commands.push(ControlMessage::AudioParamInitialValue {
            id: self.param_id,
            value: payload.initial_value,
        });
        commands.push(ControlMessage::RegisterNode {
            id: self.gain_id,
            reclaim_id: gain_reclaim,
            node: payload.gain_processor,
            inputs: 1,
            outputs: 1,
            channel_config: payload.gain_channel_config,
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
            Ok(outcome) => self.finish_accepted(outcome),
            Err(CommitWithFinalizeFailure::AcceptedFinalizer(failure)) => {
                Err(InjectedGainConstructionError::AcceptedFinalizer(failure))
            }
            Err(CommitWithFinalizeFailure::NotAccepted(failure)) => {
                self.rollback_not_accepted(failure)
            }
        }
    }

    fn finish_accepted(
        self,
        outcome: CommitControlOutcome,
    ) -> Result<InjectedConstructedGain, InjectedGainConstructionError> {
        let Self {
            control: _,
            ids,
            gain,
            param,
            gain_id,
            param_id,
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
            outcome,
        })
    }

    fn rollback_not_accepted(
        self,
        failure: super::injected_control::CommitControlFailure,
    ) -> Result<InjectedConstructedGain, InjectedGainConstructionError> {
        let Self {
            control: _,
            mut ids,
            gain,
            param,
            gain_id,
            param_id,
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
