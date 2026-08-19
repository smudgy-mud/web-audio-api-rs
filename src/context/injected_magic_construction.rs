//! Exact private construction transaction for the permanent destination/listener namespace.
//!
//! Legacy contexts deliberately keep their lazy listener publication. The injected graph instead
//! publishes every magic node and hidden edge in one 21-command envelope. Eager publication is an
//! internal performance difference: it avoids retaining admission/transport credits until a
//! future infallible `listener()` call, and the renderer cannot observe a partially installed
//! magic graph.

#![allow(dead_code)] // private prerequisite; selected by the later injected context builder

use std::panic::{self, AssertUnwindSafe};

use arrayvec::ArrayVec;

#[cfg(test)]
use super::injected_control::AcceptedBatchFinalizeError;
use super::injected_control::{
    AcceptedBatchFinalizeFailure, CommitControlOutcome, CommitWithFinalizeFailure,
    InjectedControlError, RejectedControlRollback,
};
use super::injected_ids::{InjectedNodeIdIdentity, ProvisionalNodeIdError, ProvisionalNodeIds};
use super::injected_node_construction::InjectedNodeConstructor;
use super::{AudioNodeId, DESTINATION_NODE_ID, LISTENER_NODE_ID, LISTENER_PARAM_IDS};
use crate::message::ControlMessage;
use crate::node::{
    destination_raw_parts, ChannelConfig, ChannelConfigInner, ChannelCountMode,
    ChannelInterpretation,
};
use crate::render::AudioProcessor;
use crate::spatial::{audio_listener_raw_parts, AudioListenerParams};

#[cfg(test)]
use super::injected_node_construction::MagicPayloadDropProbe;

pub(crate) const MAGIC_NODE_COUNT: usize = 11;
pub(crate) const MAGIC_COMMAND_COUNT: usize = 21;

const DESTINATION_INDEX: usize = 0;
const LISTENER_INDEX: usize = 1;
const FIRST_PARAM_INDEX: usize = 2;

#[cfg(test)]
pub(crate) const MAGIC_TEST_PAYLOAD_PANIC: u8 = 1;
#[cfg(test)]
pub(crate) const MAGIC_TEST_FINALIZER_REJECT: u8 = 2;
#[cfg(test)]
pub(crate) const MAGIC_TEST_TAKE_PROOF_FAILURE: u8 = 3;
#[cfg(test)]
pub(crate) const MAGIC_TEST_COMMIT_PROOF_FAILURE: u8 = 4;
#[cfg(test)]
pub(crate) const MAGIC_TEST_REJECTED_SHAPE_CORRUPTION: u8 = 5;
#[cfg(test)]
pub(crate) const MAGIC_TEST_REJECTED_TOKEN_CORRUPTION: u8 = 6;
#[cfg(test)]
pub(crate) const MAGIC_TEST_REJECT_EXACT: u8 = 7;

/// Accepted host representation of the permanent magic graph.
///
/// It owns no renderer or reclaim token: those moved into the accepted graph envelope. The
/// identities prevent a foreign accepted graph from being attached to a different exact base.
#[must_use]
pub(crate) struct InjectedMagicGraph {
    control_identity: super::injected_control::InjectedControlIdentity,
    node_id_identity: InjectedNodeIdIdentity,
    pub(crate) destination_channel_config: ChannelConfig,
    pub(crate) listener_params: AudioListenerParams,
    pub(crate) outcome: CommitControlOutcome,
}

/// One-shot proof that the exact control/allocator accepted the permanent magic namespace.
/// Private fields prevent a base or renderer from being marked startable by caller convention.
pub(crate) struct MagicGraphInstalled {
    control_identity: super::injected_control::InjectedControlIdentity,
    node_id_identity: InjectedNodeIdIdentity,
}

impl MagicGraphInstalled {
    pub(crate) fn matches(
        &self,
        control: &super::injected_control::InjectedControlIdentity,
        node_ids: &InjectedNodeIdIdentity,
    ) -> bool {
        self.control_identity.ptr_eq(control) && self.node_id_identity.ptr_eq(node_ids)
    }
}

impl InjectedMagicGraph {
    pub(crate) fn matches_constructor(&self, constructor: &InjectedNodeConstructor) -> bool {
        constructor.matches_control_identity(&self.control_identity)
            && constructor.matches_node_id_identity(&self.node_id_identity)
    }

    pub(crate) fn into_host_parts(
        self,
    ) -> (
        ChannelConfig,
        AudioListenerParams,
        CommitControlOutcome,
        MagicGraphInstalled,
    ) {
        (
            self.destination_channel_config,
            self.listener_params,
            self.outcome,
            MagicGraphInstalled {
                control_identity: self.control_identity,
                node_id_identity: self.node_id_identity,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedMagicConstructionError {
    Control(InjectedControlError),
    NodeIds(ProvisionalNodeIdError),
    UnexpectedMagicIdNamespace,
    PayloadPanicked,
    RejectedPayloadPanicked,
    AcceptedFinalizer(AcceptedBatchFinalizeFailure),
    ProtocolViolation,
}

pub(crate) struct InjectedMagicConstructionFailure {
    pub(crate) error: InjectedMagicConstructionError,
    /// True only while every exact ID proof remains available for this bootstrap to retry.
    /// Transport acceptance is sufficient but not necessary to make a failure terminal.
    pub(crate) retryable: bool,
}

struct MagicHostParts {
    destination_channel_config: ChannelConfig,
    listener_params: AudioListenerParams,
}

struct MagicPayload {
    host: MagicHostParts,
    destination_processor: Box<dyn AudioProcessor>,
    listener_processor: Box<dyn AudioProcessor>,
    param_processors: [Box<dyn AudioProcessor>; 9],
}

/// One admitted magic construction. The reservation is declared last so every provisional ID
/// and partially built payload is retired before graph admission on ordinary unwind.
struct InjectedMagicConstruction {
    control: super::injected_control::InjectedControlProducer,
    ids: ProvisionalNodeIds,
    #[cfg(test)]
    test_behavior: u8,
    #[cfg(test)]
    test_drop_probe: Option<MagicPayloadDropProbe>,
    reservation: Option<super::injected_control::ControlBatchReservation>,
}

impl InjectedNodeConstructor {
    /// Builds the complete permanent namespace under one admitted transaction.
    ///
    /// Exact event/control binding must be validated by the base bootstrap before calling this
    /// method. The command reservation precedes ID allocation, HRTF prewarm, and every host/render
    /// payload allocation.
    pub(crate) fn try_construct_magic_graph(
        &self,
        sample_rate: f32,
        max_channel_count: usize,
        offline: bool,
    ) -> Result<InjectedMagicGraph, InjectedMagicConstructionFailure> {
        let reservation = self
            .control()
            .try_begin_operation(MAGIC_COMMAND_COUNT)
            .map_err(|error| InjectedMagicConstructionFailure {
                error: InjectedMagicConstructionError::Control(error),
                retryable: control_error_is_retryable(error),
            })?;
        let ids = self
            .allocator()
            .try_reserve(MAGIC_NODE_COUNT)
            .map_err(|error| InjectedMagicConstructionFailure {
                error: InjectedMagicConstructionError::NodeIds(error),
                retryable: node_id_error_is_retryable(error),
            })?;
        if (0..MAGIC_NODE_COUNT).any(|index| ids.id(index).0 != index as u64) {
            return Err(InjectedMagicConstructionFailure {
                error: InjectedMagicConstructionError::UnexpectedMagicIdNamespace,
                retryable: false,
            });
        }
        InjectedMagicConstruction {
            control: self.control().clone(),
            ids,
            #[cfg(test)]
            test_behavior: self.take_magic_behavior_for_test(),
            #[cfg(test)]
            test_drop_probe: self.take_magic_drop_probe_for_test(),
            reservation: Some(reservation),
        }
        .commit(sample_rate, max_channel_count, offline)
    }
}

impl InjectedMagicConstruction {
    fn commit(
        mut self,
        sample_rate: f32,
        max_channel_count: usize,
        offline: bool,
    ) -> Result<InjectedMagicGraph, InjectedMagicConstructionFailure> {
        let payload = match panic::catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            if self.test_behavior == MAGIC_TEST_PAYLOAD_PANIC {
                panic!("forced magic payload panic");
            }
            let initial_channel_count = if offline {
                max_channel_count
            } else {
                2.min(max_channel_count)
            };
            let (destination_channel_config, destination_processor) =
                destination_raw_parts(initial_channel_count);
            let listener = audio_listener_raw_parts();

            // Legacy online construction eagerly warms this global performance cache after
            // creating its magic nodes. Preserve that timing before any exact reclaim token is
            // irreversibly transferred.
            if !offline {
                drop(crate::node::load_hrtf_processor(sample_rate as u32));
            }
            MagicPayload {
                host: MagicHostParts {
                    destination_channel_config,
                    listener_params: listener.params,
                },
                destination_processor,
                listener_processor: listener.listener_processor,
                param_processors: listener.param_processors,
            }
        })) {
            Ok(payload) => payload,
            Err(panic_payload) => {
                std::mem::forget(panic_payload);
                return Err(InjectedMagicConstructionFailure {
                    error: InjectedMagicConstructionError::PayloadPanicked,
                    retryable: true,
                });
            }
        };

        // Allocate all command storage before moving the first exact reclaim node.
        let mut commands = Vec::with_capacity(MAGIC_COMMAND_COUNT);
        let mut reclaim_nodes: ArrayVec<llq::Node<AudioNodeId>, MAGIC_NODE_COUNT> = ArrayVec::new();
        #[cfg(test)]
        if self.test_behavior == MAGIC_TEST_TAKE_PROOF_FAILURE {
            let missing = self
                .ids
                .take_reclaim_node(0)
                .expect("forced corruption removes the first exact token");
            std::mem::forget(missing);
        }
        for index in 0..MAGIC_NODE_COUNT {
            match self.ids.take_reclaim_node(index) {
                Ok(node) => reclaim_nodes.push(node),
                Err(error) => {
                    restore_taken_nodes(&mut self.ids, reclaim_nodes);
                    // The failing slot itself is missing, even when every earlier local token was
                    // restored. Never republish a partial magic namespace.
                    self.ids.retain_unavailable();
                    return Err(InjectedMagicConstructionFailure {
                        error: InjectedMagicConstructionError::NodeIds(error),
                        retryable: false,
                    });
                }
            }
        }

        let MagicPayload {
            host,
            destination_processor,
            listener_processor,
            param_processors,
        } = payload;
        #[cfg(test)]
        let mut param_processors = param_processors;
        #[cfg(test)]
        let restored_probe = self
            .test_drop_probe
            .as_ref()
            .map(|probe| std::sync::Arc::clone(&probe.tokens_restored));
        #[cfg(test)]
        if let Some(probe) = self.test_drop_probe.take() {
            let inner = std::mem::replace(&mut param_processors[0], Box::new(NoopTestProcessor));
            param_processors[0] = Box::new(ProbeDropProcessor {
                inner: Some(inner),
                probe,
            });
        }
        let destination_config = host.destination_channel_config.inner();
        commands.push(ControlMessage::RegisterNode {
            id: DESTINATION_NODE_ID,
            reclaim_id: reclaim_nodes.remove(0),
            node: destination_processor,
            inputs: 1,
            outputs: 1,
            channel_config: destination_config,
        });
        commands.push(ControlMessage::RegisterNode {
            id: LISTENER_NODE_ID,
            reclaim_id: reclaim_nodes.remove(0),
            node: listener_processor,
            inputs: 0,
            outputs: 9,
            channel_config: listener_channel_config(),
        });
        for (offset, processor) in param_processors.into_iter().enumerate() {
            let id = AudioNodeId((FIRST_PARAM_INDEX + offset) as u64);
            commands.push(ControlMessage::RegisterNode {
                id,
                reclaim_id: reclaim_nodes.remove(0),
                node: processor,
                inputs: 1,
                outputs: 1,
                channel_config: param_channel_config(),
            });
        }
        debug_assert!(reclaim_nodes.is_empty());
        for id in LISTENER_PARAM_IDS.map(AudioNodeId) {
            commands.push(ControlMessage::ConnectNode {
                from: id,
                to: LISTENER_NODE_ID,
                output: 0,
                input: usize::MAX,
            });
        }
        commands.push(ControlMessage::ConnectNode {
            from: LISTENER_NODE_ID,
            to: DESTINATION_NODE_ID,
            output: 0,
            input: usize::MAX,
        });
        debug_assert_eq!(commands.len(), MAGIC_COMMAND_COUNT);

        #[cfg(test)]
        match self.test_behavior {
            MAGIC_TEST_COMMIT_PROOF_FAILURE => self.ids.force_commit_failure_for_test(),
            MAGIC_TEST_REJECTED_SHAPE_CORRUPTION => {
                let ControlMessage::RegisterNode { inputs, .. } = &mut commands[0] else {
                    unreachable!("first magic command is destination registration")
                };
                *inputs = 2;
                self.control.fail_transport();
            }
            MAGIC_TEST_REJECTED_TOKEN_CORRUPTION => {
                let ControlMessage::RegisterNode { reclaim_id, .. } = &mut commands[0] else {
                    unreachable!("first magic command is destination registration")
                };
                let exact = std::mem::replace(reclaim_id, llq::Node::new(AudioNodeId(99)));
                std::mem::forget(exact);
                self.control.fail_transport();
            }
            MAGIC_TEST_REJECT_EXACT => self.control.fail_transport(),
            _ => {}
        }

        let id_commit = match self.ids.commit_token() {
            Ok(commit) => commit,
            Err(error) => {
                self.ids.retain_unavailable();
                return Err(InjectedMagicConstructionFailure {
                    error: InjectedMagicConstructionError::NodeIds(error),
                    retryable: false,
                });
            }
        };
        let node_id_identity = self.ids.identity();
        let batch = self
            .reservation
            .take()
            .expect("one magic transaction owns one reservation")
            .into_prevalidated(commands);

        #[cfg(not(test))]
        let committed = self.control.try_commit_with_finalize(batch, move |_| {
            // This infallible ID publication is the finalizer's first and only production action.
            id_commit.commit_accepted();
            Ok(())
        });
        #[cfg(test)]
        let committed = {
            let fail = self.test_behavior == MAGIC_TEST_FINALIZER_REJECT;
            self.control.try_commit_with_finalize(batch, move |_| {
                // Test-only failure injection still commits first, so an accepted graph can never
                // recycle a magic ID.
                id_commit.commit_accepted();
                if fail {
                    Err(AcceptedBatchFinalizeError::Rejected)
                } else {
                    Ok(())
                }
            })
        };

        match committed {
            Ok(outcome) => {
                drop(self.ids);
                Ok(InjectedMagicGraph {
                    control_identity: self.control.identity(),
                    node_id_identity,
                    destination_channel_config: host.destination_channel_config,
                    listener_params: host.listener_params,
                    outcome,
                })
            }
            Err(CommitWithFinalizeFailure::AcceptedFinalizer(failure)) => {
                self.ids.retain_unavailable();
                Err(InjectedMagicConstructionFailure {
                    error: InjectedMagicConstructionError::AcceptedFinalizer(failure),
                    retryable: false,
                })
            }
            Err(CommitWithFinalizeFailure::NotAccepted(failure)) => self.rollback_not_accepted(
                failure,
                host,
                #[cfg(test)]
                restored_probe,
            ),
        }
    }

    fn rollback_not_accepted(
        self,
        failure: super::injected_control::CommitControlFailure,
        host: MagicHostParts,
        #[cfg(test)] restored_probe: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<InjectedMagicGraph, InjectedMagicConstructionFailure> {
        let Self {
            control: _,
            ids,
            #[cfg(test)]
                test_behavior: _,
            #[cfg(test)]
                test_drop_probe: _,
            reservation: _,
        } = self;
        let destination_config = host.destination_channel_config.inner();
        let (control_error, rollback) = failure.rollback_with_commands(move |commands| {
            let mut ids = FailClosedMagicIds::new(ids);
            let exact = recover_magic_commands(
                commands,
                ids.get_mut(),
                &destination_config,
                #[cfg(test)]
                restored_probe,
            );
            // The parser has restored every recoverable exact node before any of these raw host
            // parts, processors, commands, or the ID guard can run a destructor.
            drop(host);
            if exact {
                // Disarm only after every destructor-bearing command and host part completed.
                drop(ids.into_retryable());
            }
            exact
        });
        match rollback {
            RejectedControlRollback::Panicked => Err(InjectedMagicConstructionFailure {
                error: InjectedMagicConstructionError::RejectedPayloadPanicked,
                retryable: false,
            }),
            RejectedControlRollback::Completed(false) => Err(InjectedMagicConstructionFailure {
                error: InjectedMagicConstructionError::ProtocolViolation,
                retryable: false,
            }),
            RejectedControlRollback::Completed(true) => Err(InjectedMagicConstructionFailure {
                error: InjectedMagicConstructionError::Control(control_error),
                retryable: control_error_is_retryable(control_error),
            }),
        }
    }
}

fn control_error_is_retryable(error: InjectedControlError) -> bool {
    matches!(
        error,
        InjectedControlError::LogicalCommandCredits
            | InjectedControlError::BatchStorageCredits
            | InjectedControlError::OrdinaryPhysicalCredits
            | InjectedControlError::StagingFull
            | InjectedControlError::Contended
    )
}

fn node_id_error_is_retryable(error: ProvisionalNodeIdError) -> bool {
    matches!(error, ProvisionalNodeIdError::Contended)
}

fn restore_taken_nodes(
    ids: &mut ProvisionalNodeIds,
    nodes: ArrayVec<llq::Node<AudioNodeId>, MAGIC_NODE_COUNT>,
) -> bool {
    let mut exact = true;
    for (index, node) in nodes.into_iter().enumerate() {
        if let Err(failure) = ids.restore_reclaim_node(index, node) {
            ids.retain_unavailable();
            std::mem::forget(failure.node);
            exact = false;
        }
    }
    exact
}

/// Rejected recovery defaults to permanent retention. Any parser/processor/host destructor panic
/// runs this guard while the rejected batch still owns admission, so a terminal classification can
/// never accidentally republish a partially recovered magic namespace.
struct FailClosedMagicIds {
    ids: Option<ProvisionalNodeIds>,
}

impl FailClosedMagicIds {
    fn new(ids: ProvisionalNodeIds) -> Self {
        Self { ids: Some(ids) }
    }

    fn get_mut(&mut self) -> &mut ProvisionalNodeIds {
        self.ids.as_mut().unwrap()
    }

    fn into_retryable(mut self) -> ProvisionalNodeIds {
        self.ids.take().unwrap()
    }
}

impl Drop for FailClosedMagicIds {
    fn drop(&mut self) {
        if let Some(ids) = self.ids.as_ref() {
            ids.retain_unavailable();
        }
    }
}

fn recover_magic_commands(
    commands: Box<[ControlMessage]>,
    ids: &mut ProvisionalNodeIds,
    destination_config: &ChannelConfigInner,
    #[cfg(test)] restored_probe: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> bool {
    let mut exact = commands.len() == MAGIC_COMMAND_COUNT;
    let mut restored = [false; MAGIC_NODE_COUNT];
    let mut processors: ArrayVec<Box<dyn AudioProcessor>, MAGIC_NODE_COUNT> = ArrayVec::new();
    let mut other: ArrayVec<ControlMessage, MAGIC_COMMAND_COUNT> = ArrayVec::new();

    for (position, command) in commands.into_vec().into_iter().enumerate() {
        match command {
            ControlMessage::RegisterNode {
                id,
                reclaim_id,
                node,
                inputs,
                outputs,
                channel_config,
            } => {
                let Some(index) = usize::try_from(id.0)
                    .ok()
                    .filter(|index| *index < MAGIC_NODE_COUNT)
                else {
                    exact = false;
                    std::mem::forget(reclaim_id);
                    if let Err(error) = processors.try_push(node) {
                        std::mem::forget(error.element());
                    }
                    continue;
                };
                let expected_shape = match index {
                    DESTINATION_INDEX => {
                        position == 0
                            && inputs == 1
                            && outputs == 1
                            && config_matches(&channel_config, destination_config)
                    }
                    LISTENER_INDEX => {
                        position == 1
                            && inputs == 0
                            && outputs == 9
                            && config_matches(&channel_config, &listener_channel_config())
                    }
                    _ => {
                        position == index
                            && inputs == 1
                            && outputs == 1
                            && config_matches(&channel_config, &param_channel_config())
                    }
                };
                exact &= expected_shape && !restored[index] && *reclaim_id == id;
                if !restored[index] && *reclaim_id == id {
                    restored[index] = true;
                    if let Err(failure) = ids.restore_reclaim_node(index, reclaim_id) {
                        exact = false;
                        std::mem::forget(failure.node);
                    }
                } else {
                    std::mem::forget(reclaim_id);
                }
                if let Err(error) = processors.try_push(node) {
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
                let expected = if (11..20).contains(&position) {
                    let param = AudioNodeId((position - 9) as u64);
                    from == param && to == LISTENER_NODE_ID && output == 0 && input == usize::MAX
                } else {
                    position == 20
                        && from == LISTENER_NODE_ID
                        && to == DESTINATION_NODE_ID
                        && output == 0
                        && input == usize::MAX
                };
                exact &= expected;
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
    exact &= restored.into_iter().all(std::convert::identity);

    // All exact reclaim nodes have been restored before processor/raw-command destruction.
    #[cfg(test)]
    if let Some(restored_probe) = restored_probe {
        restored_probe.store(exact, std::sync::atomic::Ordering::Release);
    }
    drop(other);
    drop(processors);
    exact
}

#[cfg(test)]
struct ProbeDropProcessor {
    inner: Option<Box<dyn AudioProcessor>>,
    probe: MagicPayloadDropProbe,
}

#[cfg(test)]
struct NoopTestProcessor;

#[cfg(test)]
impl AudioProcessor for NoopTestProcessor {
    fn process(
        &mut self,
        _inputs: &[crate::render::AudioRenderQuantum],
        _outputs: &mut [crate::render::AudioRenderQuantum],
        _params: crate::render::AudioParamValues<'_>,
        _scope: &crate::worklet::AudioWorkletGlobalScope,
    ) -> bool {
        false
    }
}

#[cfg(test)]
impl AudioProcessor for ProbeDropProcessor {
    fn process(
        &mut self,
        inputs: &[crate::render::AudioRenderQuantum],
        outputs: &mut [crate::render::AudioRenderQuantum],
        params: crate::render::AudioParamValues<'_>,
        scope: &crate::worklet::AudioWorkletGlobalScope,
    ) -> bool {
        self.inner
            .as_mut()
            .expect("probe processor retains its wrapped processor")
            .process(inputs, outputs, params, scope)
    }
}

#[cfg(test)]
impl Drop for ProbeDropProcessor {
    fn drop(&mut self) {
        assert!(
            self.probe
                .tokens_restored
                .load(std::sync::atomic::Ordering::Acquire),
            "all eleven exact reclaim tokens must be restored before processor destruction"
        );
        self.probe.started.send(()).unwrap();
        self.probe.release.recv().unwrap();
        if self.probe.panic_after_release {
            panic!("forced magic processor destructor panic");
        }
        drop(self.inner.take());
    }
}

fn listener_channel_config() -> ChannelConfigInner {
    ChannelConfigInner {
        count: 1,
        count_mode: ChannelCountMode::Explicit,
        interpretation: ChannelInterpretation::Discrete,
    }
}

fn param_channel_config() -> ChannelConfigInner {
    listener_channel_config()
}

fn config_matches(left: &ChannelConfigInner, right: &ChannelConfigInner) -> bool {
    left.count == right.count
        && left.count_mode == right.count_mode
        && left.interpretation == right.interpretation
}
