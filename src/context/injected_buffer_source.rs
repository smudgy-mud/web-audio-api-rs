//! Exact hosted `AudioBufferSourceNode` construction and runtime commands.
//!
//! Construction reuses the already-proven three-node scheduled-source transaction, but the
//! accepted control capability and every runtime wire are distinct BufferSource brands. Scalar
//! commands remain Copy. The owned buffer command carries a preboxed GC node so no AudioBuffer or
//! external storage lease is ever destroyed on the render thread.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
#[cfg(test)]
use std::sync::atomic::AtomicU8;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::injected_connections::InjectedConnectionEndpoint;
use super::injected_control::{
    AcceptedBatchFinalizeFailure, CommitControlOutcome, CommitWithFinalizeFailure,
    InjectedControlError, RejectedControlRollback,
};
use super::injected_node_construction::{
    InjectedNodeConstructor, InjectedOscillatorConstruction, InjectedOscillatorConstructionError,
    InjectedOscillatorPayload,
};
use super::{
    AudioContextRegistration, AudioControlBatchReservation, AudioNodeId,
    AudioNodeLifetimeReservation, InjectedAudioParamMutation, InjectedNodeRegistration,
};
use crate::buffer::AudioBuffer;
use crate::events::{ExactEndedEventKey, InjectedExactEndedEventTarget};
use crate::message::ControlMessage;
use crate::node::{ChannelConfigInner, OscillatorType};
use crate::param::{AudioParamInitialValue, InjectedAudioParamProcessor};
use crate::render::AudioProcessor;
use crate::AtomicF64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum InjectedAudioBufferSourceScalarCommand {
    StartWithOffsetAndDuration {
        start: f64,
        offset: f64,
        duration: f64,
    },
    Stop(f64),
    Loop(bool),
    LoopStart(f64),
    LoopEnd(f64),
}

/// Fixed scalar wire constructible only by an accepted BufferSource capability.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InjectedAudioBufferSourceScalarWireCommand {
    id: AudioNodeId,
    key: ExactEndedEventKey,
    command: InjectedAudioBufferSourceScalarCommand,
}

/// Stack-only scalar renderer dispatch wrapper.
pub(crate) struct InjectedAudioBufferSourceScalarRenderMessage {
    wire: InjectedAudioBufferSourceScalarWireCommand,
    applied: bool,
}

impl InjectedAudioBufferSourceScalarWireCommand {
    pub(crate) fn into_render_message(self) -> InjectedAudioBufferSourceScalarRenderMessage {
        InjectedAudioBufferSourceScalarRenderMessage {
            wire: self,
            applied: false,
        }
    }
}

impl InjectedAudioBufferSourceScalarRenderMessage {
    pub(crate) const fn id(&self) -> AudioNodeId {
        self.wire.id
    }

    pub(crate) fn apply_to(
        &mut self,
        expected: ExactEndedEventKey,
    ) -> Option<InjectedAudioBufferSourceScalarCommand> {
        if self.wire.key != expected || self.applied {
            return None;
        }
        self.applied = true;
        Some(self.wire.command)
    }

    pub(crate) const fn was_applied(&self) -> bool {
        self.applied
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InjectedAudioBufferPayloadToken(u64);

/// Preboxed owned-buffer message routed to the exact processor and then always moved to GC.
pub(crate) struct InjectedAudioBufferSourceBufferRenderMessage {
    key: ExactEndedEventKey,
    token: InjectedAudioBufferPayloadToken,
    buffer: Option<AudioBuffer>,
    applied: bool,
}

impl std::fmt::Debug for InjectedAudioBufferSourceBufferRenderMessage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InjectedAudioBufferSourceBufferRenderMessage")
            .field("key", &self.key)
            .field("token", &self.token)
            .field("has_buffer", &self.buffer.is_some())
            .field("applied", &self.applied)
            .finish()
    }
}

impl InjectedAudioBufferSourceBufferRenderMessage {
    pub(crate) fn apply_to(
        &mut self,
        expected: ExactEndedEventKey,
        current: &mut Option<AudioBuffer>,
    ) -> bool {
        if self.key != expected || self.applied || self.buffer.is_none() || current.is_some() {
            return false;
        }
        std::mem::swap(current, &mut self.buffer);
        self.applied = true;
        true
    }

    pub(crate) const fn was_applied(&self) -> bool {
        self.applied
    }

    fn matches(&self, key: ExactEndedEventKey, token: InjectedAudioBufferPayloadToken) -> bool {
        self.key == key && self.token == token && !self.applied && self.buffer.is_some()
    }
}

/// One owned exact BufferSource buffer command. Its node is taken exactly once by the renderer.
pub(crate) struct InjectedAudioBufferSourceBufferWireCommand {
    id: AudioNodeId,
    message: Option<llq::Node<Box<dyn Any + Send>>>,
}

impl std::fmt::Debug for InjectedAudioBufferSourceBufferWireCommand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InjectedAudioBufferSourceBufferWireCommand")
            .field("id", &self.id)
            .field("has_message", &self.message.is_some())
            .finish()
    }
}

impl InjectedAudioBufferSourceBufferWireCommand {
    pub(crate) const fn id(&self) -> AudioNodeId {
        self.id
    }

    pub(crate) fn take_render_message(&mut self) -> Option<llq::Node<Box<dyn Any + Send>>> {
        self.message.take()
    }

    fn matches(
        &self,
        id: AudioNodeId,
        key: ExactEndedEventKey,
        token: InjectedAudioBufferPayloadToken,
    ) -> bool {
        self.id == id
            && self.message.as_ref().is_some_and(|message| {
                message
                    .as_ref()
                    .downcast_ref::<InjectedAudioBufferSourceBufferRenderMessage>()
                    .is_some_and(|message| message.matches(key, token))
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedAudioBufferSourceMutationError {
    Control(InjectedControlError),
    AcceptedFinalizer(AcceptedBatchFinalizeFailure),
    DuplicateStart,
    StopBeforeStart,
    BufferAlreadySet,
    PayloadIdentityExhausted,
    Inactive,
    SerializerPoisoned,
    RejectedPayloadPanicked,
    ProtocolViolation,
}

/// Weak post-construction command capability for one exact BufferSource generation.
pub(crate) struct InjectedAudioBufferSourceControl {
    control: super::injected_control::InjectedControlProducer,
    node_ids: super::injected_ids::InjectedNodeIdIdentity,
    id: AudioNodeId,
    lifetime: super::InjectedNodeRegistrationIdentity,
    ended: InjectedExactEndedEventTarget,
    serializer: Arc<Mutex<()>>,
    has_start: Arc<AtomicBool>,
    buffer: Arc<OnceLock<AudioBuffer>>,
    loop_enabled: Arc<AtomicBool>,
    loop_start: Arc<AtomicF64>,
    loop_end: Arc<AtomicF64>,
    next_payload_token: AtomicU64,
    #[cfg(test)]
    runtime_behavior: Arc<AtomicU8>,
}

impl std::fmt::Debug for InjectedAudioBufferSourceControl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InjectedAudioBufferSourceControl")
            .field("id", &self.id)
            .field("has_start", &self.has_start())
            .field("has_buffer", &self.buffer().is_some())
            .field("loop", &self.loop_())
            .finish_non_exhaustive()
    }
}

impl InjectedAudioBufferSourceControl {
    #[cfg(test)]
    pub(crate) fn fail_next_scalar_commit_for_test(&self) {
        self.runtime_behavior.store(1, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn panic_next_scalar_finalizer_for_test(&self) {
        self.runtime_behavior.store(2, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn fail_next_buffer_commit_for_test(&self) {
        self.runtime_behavior.store(3, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn panic_next_buffer_finalizer_for_test(&self) {
        self.runtime_behavior.store(4, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn exhaust_payload_tokens_for_test(&self) {
        self.next_payload_token.store(u64::MAX, Ordering::Release);
    }

    pub(crate) fn ended_target(&self) -> InjectedExactEndedEventTarget {
        self.ended.clone()
    }

    pub(crate) fn matches_registration(
        &self,
        registration: &AudioContextRegistration,
        constructor: &InjectedNodeConstructor,
    ) -> bool {
        self.id == registration.id()
            && constructor.matches_control_identity(&self.control.identity())
            && constructor.matches_node_id_identity(&self.node_ids)
            && registration.matches_injected_lifetime_identity(&self.lifetime)
    }

    pub(crate) fn has_start(&self) -> bool {
        self.has_start.load(Ordering::Acquire)
    }

    pub(crate) fn buffer(&self) -> Option<&AudioBuffer> {
        self.buffer.get()
    }

    pub(crate) fn loop_(&self) -> bool {
        self.loop_enabled.load(Ordering::Acquire)
    }

    pub(crate) fn loop_start(&self) -> f64 {
        self.loop_start.load(Ordering::Acquire)
    }

    pub(crate) fn loop_end(&self) -> f64 {
        self.loop_end.load(Ordering::Acquire)
    }

    pub(crate) fn try_start(
        &self,
        start: f64,
        offset: f64,
        duration: f64,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::StartWithOffsetAndDuration {
                start,
                offset,
                duration,
            },
            None,
        )
    }

    pub(crate) fn try_start_with_host_reservation(
        &self,
        start: f64,
        offset: f64,
        duration: f64,
        reservation: AudioControlBatchReservation,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::StartWithOffsetAndDuration {
                start,
                offset,
                duration,
            },
            Some(reservation),
        )
    }

    pub(crate) fn try_stop(
        &self,
        when: f64,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(InjectedAudioBufferSourceScalarCommand::Stop(when), None)
    }

    pub(crate) fn try_stop_with_host_reservation(
        &self,
        when: f64,
        reservation: AudioControlBatchReservation,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::Stop(when),
            Some(reservation),
        )
    }

    pub(crate) fn try_set_loop(
        &self,
        value: bool,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(InjectedAudioBufferSourceScalarCommand::Loop(value), None)
    }

    pub(crate) fn try_set_loop_with_host_reservation(
        &self,
        value: bool,
        reservation: AudioControlBatchReservation,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::Loop(value),
            Some(reservation),
        )
    }

    pub(crate) fn try_set_loop_start(
        &self,
        value: f64,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::LoopStart(value),
            None,
        )
    }

    pub(crate) fn try_set_loop_start_with_host_reservation(
        &self,
        value: f64,
        reservation: AudioControlBatchReservation,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::LoopStart(value),
            Some(reservation),
        )
    }

    pub(crate) fn try_set_loop_end(
        &self,
        value: f64,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(InjectedAudioBufferSourceScalarCommand::LoopEnd(value), None)
    }

    pub(crate) fn try_set_loop_end_with_host_reservation(
        &self,
        value: f64,
        reservation: AudioControlBatchReservation,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_scalar_command(
            InjectedAudioBufferSourceScalarCommand::LoopEnd(value),
            Some(reservation),
        )
    }

    fn try_scalar_command(
        &self,
        command: InjectedAudioBufferSourceScalarCommand,
        host_reservation: Option<AudioControlBatchReservation>,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        let _serialized = self
            .serializer
            .lock()
            .map_err(|_| InjectedAudioBufferSourceMutationError::SerializerPoisoned)?;
        match command {
            InjectedAudioBufferSourceScalarCommand::StartWithOffsetAndDuration { .. }
                if self.has_start() =>
            {
                return Err(InjectedAudioBufferSourceMutationError::DuplicateStart)
            }
            InjectedAudioBufferSourceScalarCommand::Stop(_) if !self.has_start() => {
                return Err(InjectedAudioBufferSourceMutationError::StopBeforeStart)
            }
            _ => {}
        }
        let reservation = match host_reservation {
            Some(host_reservation) => self
                .control
                .try_begin_operation_with_host_reservation(1, host_reservation),
            None => self.control.try_begin_operation(1),
        }
        .map_err(InjectedAudioBufferSourceMutationError::Control)?;
        if !self.lifetime.is_live_for(self.id) {
            return Err(InjectedAudioBufferSourceMutationError::Inactive);
        }
        let wire = InjectedAudioBufferSourceScalarWireCommand {
            id: self.id,
            key: self.ended.render_key(),
            command,
        };
        let batch = reservation
            .into_prevalidated(vec![ControlMessage::InjectedAudioBufferSourceScalar(wire)]);
        #[cfg(test)]
        if self
            .runtime_behavior
            .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.control.fail_closed_protocol();
        }
        let has_start = &self.has_start;
        let loop_enabled = &self.loop_enabled;
        let loop_start = &self.loop_start;
        let loop_end = &self.loop_end;
        #[cfg(test)]
        let runtime_behavior = &self.runtime_behavior;
        match self.control.try_commit_with_finalize(batch, move |_| {
            match command {
                InjectedAudioBufferSourceScalarCommand::StartWithOffsetAndDuration { .. } => {
                    has_start.store(true, Ordering::Release);
                }
                InjectedAudioBufferSourceScalarCommand::Stop(_) => {}
                InjectedAudioBufferSourceScalarCommand::Loop(value) => {
                    loop_enabled.store(value, Ordering::Release);
                }
                InjectedAudioBufferSourceScalarCommand::LoopStart(value) => {
                    loop_start.store(value, Ordering::Release);
                }
                InjectedAudioBufferSourceScalarCommand::LoopEnd(value) => {
                    loop_end.store(value, Ordering::Release);
                }
            }
            #[cfg(test)]
            if runtime_behavior
                .compare_exchange(2, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                panic!("forced exact BufferSource scalar finalizer panic");
            }
            Ok(())
        }) {
            Ok(outcome) => Ok(outcome),
            Err(CommitWithFinalizeFailure::AcceptedFinalizer(failure)) => Err(
                InjectedAudioBufferSourceMutationError::AcceptedFinalizer(failure),
            ),
            Err(CommitWithFinalizeFailure::NotAccepted(failure)) => {
                let control = self.control.clone();
                let id = self.id;
                let key = self.ended.render_key();
                let (error, rollback) = failure.rollback_with_commands(move |commands| {
                    let mut fail_closed = FailClosedBufferSourceRollback::new(control);
                    let mut commands = commands.into_vec().into_iter();
                    let exact = matches!(
                        (commands.next(), commands.next()),
                        (Some(ControlMessage::InjectedAudioBufferSourceScalar(value)), None)
                            if value.id == id
                                && value.key == key
                                && scalar_commands_match(value.command, command)
                    );
                    if exact {
                        fail_closed.disarm();
                    }
                    exact
                });
                match rollback {
                    RejectedControlRollback::Completed(true) => {
                        Err(InjectedAudioBufferSourceMutationError::Control(error))
                    }
                    RejectedControlRollback::Completed(false) => {
                        Err(InjectedAudioBufferSourceMutationError::ProtocolViolation)
                    }
                    RejectedControlRollback::Panicked => {
                        Err(InjectedAudioBufferSourceMutationError::RejectedPayloadPanicked)
                    }
                }
            }
        }
    }

    pub(crate) fn try_set_buffer(
        &self,
        audio_buffer: AudioBuffer,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_set_buffer_inner(audio_buffer, None)
    }

    pub(crate) fn try_set_buffer_with_host_reservation(
        &self,
        audio_buffer: AudioBuffer,
        reservation: AudioControlBatchReservation,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        self.try_set_buffer_inner(audio_buffer, Some(reservation))
    }

    fn try_set_buffer_inner(
        &self,
        audio_buffer: AudioBuffer,
        host_reservation: Option<AudioControlBatchReservation>,
    ) -> Result<CommitControlOutcome, InjectedAudioBufferSourceMutationError> {
        let _serialized = self
            .serializer
            .lock()
            .map_err(|_| InjectedAudioBufferSourceMutationError::SerializerPoisoned)?;
        if self.buffer().is_some() {
            return Err(InjectedAudioBufferSourceMutationError::BufferAlreadySet);
        }

        let admitted = self
            .control
            .try_admit_graph_operation()
            .map_err(InjectedAudioBufferSourceMutationError::Control)?;
        let _admission_fence = admitted.admission_fence();
        let mut fail_closed = FailClosedBufferSourceRollback::new(self.control.clone());
        if !self.lifetime.is_live_for(self.id) {
            fail_closed.disarm();
            return Err(InjectedAudioBufferSourceMutationError::Inactive);
        }
        let token = match self
            .next_payload_token
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map(InjectedAudioBufferPayloadToken)
        {
            Ok(token) => token,
            Err(_) => {
                // Exhaustion is terminal and the guard deliberately remains armed. Destroy the
                // caller-owned buffer while the cloned admission fence still keeps Close pending;
                // an external storage lease may have an adversarial destructor.
                drop(audio_buffer);
                drop(fail_closed);
                return Err(InjectedAudioBufferSourceMutationError::PayloadIdentityExhausted);
            }
        };
        let render_message = InjectedAudioBufferSourceBufferRenderMessage {
            key: self.ended.render_key(),
            token,
            buffer: Some(audio_buffer.clone()),
            applied: false,
        };
        let wire = InjectedAudioBufferSourceBufferWireCommand {
            id: self.id,
            message: Some(llq::Node::new(
                Box::new(render_message) as Box<dyn Any + Send>
            )),
        };
        let commands: Box<[ControlMessage]> =
            Box::new([ControlMessage::InjectedAudioBufferSourceBuffer(wire)]);
        let reservation = match admitted.reserve_commands(1) {
            Ok(reservation) => reservation,
            Err(failure) => {
                fail_closed.disarm();
                drop(audio_buffer);
                drop(commands);
                drop(failure.operation);
                return Err(InjectedAudioBufferSourceMutationError::Control(
                    failure.error,
                ));
            }
        };
        let reservation = match host_reservation {
            Some(host_reservation) => match reservation.with_host_reservation(host_reservation) {
                Ok(reservation) => reservation,
                Err(error) => {
                    fail_closed.disarm();
                    drop(audio_buffer);
                    drop(commands);
                    return Err(InjectedAudioBufferSourceMutationError::Control(error));
                }
            },
            None => reservation,
        };
        let batch = reservation.into_preboxed(commands);
        #[cfg(test)]
        if self
            .runtime_behavior
            .compare_exchange(3, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.control.fail_closed_protocol();
        }
        let accepted = match self.control.try_commit_retained(batch) {
            Ok(accepted) => accepted,
            Err(failure) => {
                let control = self.control.clone();
                let id = self.id;
                let key = self.ended.render_key();
                let (error, rollback) = failure.rollback_with_commands(move |commands| {
                    let mut rollback_guard = FailClosedBufferSourceRollback::new(control);
                    drop(audio_buffer);
                    let exact = commands.len() == 1
                        && matches!(
                            &commands[0],
                            ControlMessage::InjectedAudioBufferSourceBuffer(value)
                                if value.matches(id, key, token)
                        );
                    let destructor_panicked = drop_commands_individually(commands);
                    if exact && !destructor_panicked {
                        rollback_guard.disarm();
                    }
                    exact && !destructor_panicked
                });
                fail_closed.disarm();
                return match rollback {
                    RejectedControlRollback::Completed(true) => {
                        Err(InjectedAudioBufferSourceMutationError::Control(error))
                    }
                    RejectedControlRollback::Completed(false) => {
                        Err(InjectedAudioBufferSourceMutationError::ProtocolViolation)
                    }
                    RejectedControlRollback::Panicked => {
                        Err(InjectedAudioBufferSourceMutationError::RejectedPayloadPanicked)
                    }
                };
            }
        };
        fail_closed.disarm();
        drop(fail_closed);
        let mut accepted_guard = FailClosedBufferSourceRollback::new(self.control.clone());
        #[cfg(test)]
        if self
            .runtime_behavior
            .compare_exchange(4, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            panic!("forced exact BufferSource buffer finalizer panic");
        }
        match self.buffer.set(audio_buffer) {
            Ok(()) => {
                accepted_guard.disarm();
                Ok(accepted.complete())
            }
            Err(buffer) => {
                std::mem::forget(buffer);
                Err(InjectedAudioBufferSourceMutationError::ProtocolViolation)
            }
        }
    }
}

fn scalar_commands_match(
    left: InjectedAudioBufferSourceScalarCommand,
    right: InjectedAudioBufferSourceScalarCommand,
) -> bool {
    match (left, right) {
        (
            InjectedAudioBufferSourceScalarCommand::StartWithOffsetAndDuration {
                start: left_start,
                offset: left_offset,
                duration: left_duration,
            },
            InjectedAudioBufferSourceScalarCommand::StartWithOffsetAndDuration {
                start: right_start,
                offset: right_offset,
                duration: right_duration,
            },
        ) => {
            left_start.to_bits() == right_start.to_bits()
                && left_offset.to_bits() == right_offset.to_bits()
                && left_duration.to_bits() == right_duration.to_bits()
        }
        (
            InjectedAudioBufferSourceScalarCommand::Stop(left),
            InjectedAudioBufferSourceScalarCommand::Stop(right),
        )
        | (
            InjectedAudioBufferSourceScalarCommand::LoopStart(left),
            InjectedAudioBufferSourceScalarCommand::LoopStart(right),
        )
        | (
            InjectedAudioBufferSourceScalarCommand::LoopEnd(left),
            InjectedAudioBufferSourceScalarCommand::LoopEnd(right),
        ) => left.to_bits() == right.to_bits(),
        (
            InjectedAudioBufferSourceScalarCommand::Loop(left),
            InjectedAudioBufferSourceScalarCommand::Loop(right),
        ) => left == right,
        _ => false,
    }
}

fn drop_commands_individually(commands: Box<[ControlMessage]>) -> bool {
    let mut panicked = false;
    for command in commands.into_vec() {
        panicked |= panic::catch_unwind(AssertUnwindSafe(|| drop(command))).map_or_else(
            |payload| {
                std::mem::forget(payload);
                true
            },
            |()| false,
        );
    }
    panicked
}

struct FailClosedBufferSourceRollback {
    control: super::injected_control::InjectedControlProducer,
    armed: bool,
}

impl FailClosedBufferSourceRollback {
    fn new(control: super::injected_control::InjectedControlProducer) -> Self {
        Self {
            control,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for FailClosedBufferSourceRollback {
    fn drop(&mut self) {
        if self.armed {
            self.control.fail_closed_protocol();
        }
    }
}

pub(crate) struct InjectedAudioBufferSourcePayload {
    pub(crate) playback_rate_processor: InjectedAudioParamProcessor,
    pub(crate) detune_processor: InjectedAudioParamProcessor,
    pub(crate) source_processor: Box<dyn AudioProcessor>,
    pub(crate) param_channel_config: ChannelConfigInner,
    pub(crate) source_channel_config: ChannelConfigInner,
    pub(crate) playback_rate_initial_value: AudioParamInitialValue,
    pub(crate) detune_initial_value: AudioParamInitialValue,
    pub(crate) loop_enabled: bool,
    pub(crate) loop_start: f64,
    pub(crate) loop_end: f64,
}

pub(crate) struct InjectedConstructedAudioBufferSource {
    pub(crate) source_id: AudioNodeId,
    pub(crate) playback_rate_id: AudioNodeId,
    pub(crate) detune_id: AudioNodeId,
    pub(crate) source_registration: InjectedNodeRegistration,
    pub(crate) playback_rate_registration: InjectedNodeRegistration,
    pub(crate) detune_registration: InjectedNodeRegistration,
    pub(crate) source_connection: InjectedConnectionEndpoint,
    pub(crate) playback_rate_connection: InjectedConnectionEndpoint,
    pub(crate) detune_connection: InjectedConnectionEndpoint,
    pub(crate) playback_rate_mutation: InjectedAudioParamMutation,
    pub(crate) detune_mutation: InjectedAudioParamMutation,
    pub(crate) source_control: InjectedAudioBufferSourceControl,
    pub(crate) outcome: CommitControlOutcome,
}

pub(crate) type InjectedAudioBufferSourceConstructionError = InjectedOscillatorConstructionError;

/// Distinct constructor brand wrapping the common three-node scheduled-source transaction.
pub(crate) struct InjectedAudioBufferSourceConstruction {
    inner: InjectedOscillatorConstruction,
}

impl InjectedNodeConstructor {
    pub(super) fn try_begin_audio_buffer_source_with_reservations(
        &self,
        events: &crate::events::InjectedControlEventDispatch,
        lifetime: Option<AudioNodeLifetimeReservation>,
        control: Option<AudioControlBatchReservation>,
    ) -> Result<InjectedAudioBufferSourceConstruction, InjectedAudioBufferSourceConstructionError>
    {
        self.try_begin_oscillator_with_reservations(events, OscillatorType::Sine, lifetime, control)
            .map(|inner| InjectedAudioBufferSourceConstruction { inner })
    }
}

impl InjectedAudioBufferSourceConstruction {
    pub(crate) const fn source_id(&self) -> AudioNodeId {
        self.inner.oscillator_id()
    }

    pub(crate) const fn playback_rate_id(&self) -> AudioNodeId {
        self.inner.detune_id()
    }

    pub(crate) const fn detune_id(&self) -> AudioNodeId {
        self.inner.frequency_id()
    }

    pub(crate) fn completion_key(&self) -> ExactEndedEventKey {
        self.inner.completion_key()
    }

    pub(crate) fn commit(
        self,
        payload: InjectedAudioBufferSourcePayload,
    ) -> Result<InjectedConstructedAudioBufferSource, InjectedAudioBufferSourceConstructionError>
    {
        let buffer = Arc::new(OnceLock::new());
        let loop_enabled = Arc::new(AtomicBool::new(payload.loop_enabled));
        let loop_start = Arc::new(AtomicF64::new(payload.loop_start));
        let loop_end = Arc::new(AtomicF64::new(payload.loop_end));
        let constructed = self.inner.commit(InjectedOscillatorPayload {
            frequency_processor: payload.detune_processor,
            detune_processor: payload.playback_rate_processor,
            oscillator_processor: payload.source_processor,
            param_channel_config: payload.param_channel_config,
            oscillator_channel_config: payload.source_channel_config,
            frequency_initial_value: payload.detune_initial_value,
            detune_initial_value: payload.playback_rate_initial_value,
        })?;
        let parts = constructed.oscillator_control.into_scheduled_source_parts();
        let source_control = InjectedAudioBufferSourceControl {
            control: parts.control,
            node_ids: parts.node_ids,
            id: parts.id,
            lifetime: parts.lifetime,
            ended: parts.ended,
            serializer: parts.serializer,
            has_start: parts.has_start,
            buffer,
            loop_enabled,
            loop_start,
            loop_end,
            next_payload_token: AtomicU64::new(1),
            #[cfg(test)]
            runtime_behavior: Arc::new(AtomicU8::new(0)),
        };
        Ok(InjectedConstructedAudioBufferSource {
            source_id: constructed.oscillator_id,
            playback_rate_id: constructed.detune_id,
            detune_id: constructed.frequency_id,
            source_registration: constructed.oscillator_registration,
            playback_rate_registration: constructed.detune_registration,
            detune_registration: constructed.frequency_registration,
            source_connection: constructed.oscillator_connection,
            playback_rate_connection: constructed.detune_connection,
            detune_connection: constructed.frequency_connection,
            playback_rate_mutation: constructed.detune_mutation,
            detune_mutation: constructed.frequency_mutation,
            source_control,
            outcome: constructed.outcome,
        })
    }
}
