//! Closed one-node transaction for the native PCM queue source.

use super::*;

pub(crate) struct InjectedPcmSourceConstruction {
    control: InjectedControlProducer,
    ids: ProvisionalNodeIds,
    node: ProvisionalNodeRegistration,
    connection: InjectedConnectionEndpoint,
    queue_lifetime: Option<SharedAudioNodeLifetimeReservation>,
    // Keep admission until all provisional fields have rolled back on unwind.
    reservation: Option<ControlBatchReservation>,
}

impl InjectedNodeConstructor {
    pub(crate) fn try_begin_pcm_source(
        &self,
        lifetime: Option<AudioNodeLifetimeReservation>,
        control: Option<AudioControlBatchReservation>,
    ) -> Result<InjectedPcmSourceConstruction, InjectedGainConstructionError> {
        let reservation = match control {
            Some(control) => self
                .control
                .try_begin_operation_with_host_reservation(1, control),
            None => self.control.try_begin_operation(1),
        }
        .map_err(InjectedGainConstructionError::Control)?;
        let ids = self
            .allocator
            .try_reserve(1)
            .map_err(InjectedGainConstructionError::NodeIds)?;
        let queue_lifetime = lifetime.map(SharedAudioNodeLifetimeReservation::new);
        let (node, connection) = self.register_endpoint(
            ids.id(0),
            InjectedConnectionEndpointKind::AudioNode,
            0,
            1,
            queue_lifetime.clone(),
        )?;
        Ok(InjectedPcmSourceConstruction {
            control: self.control.clone(),
            ids,
            node,
            connection,
            queue_lifetime,
            reservation: Some(reservation),
        })
    }
}

impl InjectedPcmSourceConstruction {
    pub(crate) fn queue_lifetime(&self) -> Option<SharedAudioNodeLifetimeReservation> {
        self.queue_lifetime.clone()
    }
    pub(crate) fn commit(
        mut self,
        // A concrete processor type prevents this seam becoming a general worklet escape hatch.
        processor: crate::node::PcmSourceRenderer,
        channel_config: ChannelConfigInner,
    ) -> Result<
        (
            AudioNodeId,
            InjectedNodeRegistration,
            InjectedConnectionEndpoint,
        ),
        InjectedGainConstructionError,
    > {
        let id = self.ids.id(0);
        let processor: Box<dyn AudioProcessor> = Box::new(processor);
        let mut commands = Vec::with_capacity(1);
        let reclaim_id = self.ids.take_reclaim_node(0).map_err(|error| {
            self.node.arm_token().quarantine_accepted();
            self.ids.retain_unavailable();
            InjectedGainConstructionError::NodeIds(error)
        })?;
        commands.push(ControlMessage::RegisterNode {
            id,
            reclaim_id,
            node: processor,
            inputs: 0,
            outputs: 1,
            channel_config,
        });
        let commit = self.ids.commit_token().map_err(|error| {
            self.node.arm_token().quarantine_accepted();
            self.ids.retain_unavailable();
            InjectedGainConstructionError::NodeIds(error)
        })?;
        let arm = self.node.arm_token();
        let batch = self
            .reservation
            .take()
            .expect("PCM construction owns admission")
            .into_prevalidated(commands);
        match self.control.try_commit_with_finalize(batch, move |_| {
            arm.mark_accepted();
            commit.commit_accepted();
            arm.arm_accepted()
        }) {
            Ok(_) => {
                if !self.node.ready_for_registration() {
                    self.node.arm_token().quarantine_accepted();
                    return Err(InjectedGainConstructionError::ProtocolViolation);
                }
                Ok((
                    id,
                    self.node
                        .into_registration()
                        .expect("accepted PCM registration"),
                    self.connection,
                ))
            }
            Err(CommitWithFinalizeFailure::AcceptedFinalizer(error)) => {
                Err(InjectedGainConstructionError::AcceptedFinalizer(error))
            }
            Err(CommitWithFinalizeFailure::NotAccepted(failure)) => {
                let (error, rollback) = failure.rollback_with_commands(move |commands| {
                    let mut exact = commands.len() == 1;
                    let mut panicked = false;
                    for command in commands.into_vec() {
                        match command {
                            ControlMessage::RegisterNode {
                                id: actual,
                                reclaim_id,
                                node,
                                inputs,
                                outputs,
                                ..
                            } => {
                                exact &= actual == id && inputs == 0 && outputs == 1;
                                if let Err(failure) = self.ids.restore_reclaim_node(0, reclaim_id) {
                                    exact = false;
                                    std::mem::forget(failure.node);
                                }
                                panicked |= catch_individual_payload_drop(node);
                            }
                            other => {
                                exact = false;
                                panicked |= catch_individual_payload_drop(other);
                            }
                        }
                    }
                    if !exact {
                        self.node.arm_token().quarantine_accepted();
                        self.ids.retain_unavailable();
                        self.control.fail_closed_protocol();
                    }
                    drop(self); // Return lifetime slot and ID while rejection retains admission.
                    (exact, panicked)
                });
                match rollback {
                    RejectedControlRollback::Completed((true, false)) => {
                        Err(InjectedGainConstructionError::Control(error))
                    }
                    RejectedControlRollback::Completed((false, _)) => {
                        Err(InjectedGainConstructionError::ProtocolViolation)
                    }
                    _ => Err(InjectedGainConstructionError::RejectedPayloadPanicked),
                }
            }
        }
    }
}
