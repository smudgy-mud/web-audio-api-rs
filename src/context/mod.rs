//! The `BaseAudioContext` interface and the `AudioContext` and `OfflineAudioContext` types

use std::{any::Any, ops::Range};

mod base;
pub use base::*;

mod concrete_base;
pub use concrete_base::*;

mod injected_admission;
#[cfg(test)]
pub(crate) use injected_admission::AdmissionSnapshot;
pub(crate) use injected_admission::{
    AdmissionError, CapacityWorkerId, CapacityWorkerJoinError, InjectedContextAdmissionGate,
};

mod injected_control;
pub(crate) use injected_control::{CommitControlOutcome, InjectedControlRenderInit};

mod injected_connections;
pub(crate) use injected_connections::{
    InjectedConnectionEndpointKind, InjectedExplicitConnect, InjectedExplicitDisconnect,
    MAX_INJECTED_EXPLICIT_CONNECTIONS, MAX_INJECTED_GRAPH_NODES,
};

mod injected_ids;
#[cfg(test)]
pub(crate) use injected_ids::injected_node_id_pair;
pub(crate) use injected_ids::{InjectedGraphReclaimInit, InjectedGraphReclaimPublisher};

mod injected_magic_construction;
#[cfg(test)]
pub(crate) use injected_magic_construction::MAGIC_COMMAND_COUNT;
#[cfg(test)]
mod injected_magic_construction_tests;

mod injected_node_lifetime;
pub(crate) use injected_node_lifetime::{
    InjectedNodeRegistration, InjectedNodeRegistrationIdentity, RetiredInjectedGraph,
};

mod injected_node_construction;
pub(crate) use injected_node_construction::{
    InjectedAudioParamMutation, InjectedGainPayload, InjectedOscillatorCommandKind,
    InjectedOscillatorControl, InjectedOscillatorEventMint, InjectedOscillatorMutationError,
    InjectedOscillatorPayload, InjectedOscillatorRenderMessage, InjectedOscillatorWireCommand,
};
#[cfg(test)]
mod injected_node_construction_tests;

#[cfg(feature = "diagnostics")]
mod diagnostics;
#[cfg(feature = "diagnostics")]
pub use diagnostics::*;

mod offline;
pub use offline::*;

mod online;
pub use online::*;

mod resource;
pub use resource::AudioNodeLifetimeReservation;
pub(crate) use resource::SharedAudioNodeLifetimeReservation;

mod hosted;
pub use hosted::{
    AudioContextBuildError, AudioContextBuildErrorKind, AudioContextBuilder,
    AudioContextLifecycleError, AudioContextShutdownIssue, AudioContextShutdownIssueKind,
    AudioContextShutdownMode, AudioContextShutdownOutcome, AudioContextShutdownReceipt,
    AudioContextShutdownReport, AudioContextStateChangeOutcome, AudioContextStateChangeReceipt,
};

mod output_lifecycle;

#[cfg(test)]
mod injected_output_bootstrap_tests;

// magic node values
/// Destination node id is always at index 0
pub(crate) const DESTINATION_NODE_ID: AudioNodeId = AudioNodeId(0);
/// listener node id is always at index 1
const LISTENER_NODE_ID: AudioNodeId = AudioNodeId(1);
/// listener audio parameters ids are always at index 2 through 10
const LISTENER_PARAM_IDS: Range<u64> = 2..11;
/// listener audio parameters ids are always at index 2 through 10
pub(crate) const LISTENER_AUDIO_PARAM_IDS: [AudioParamId; 9] = [
    AudioParamId(2),
    AudioParamId(3),
    AudioParamId(4),
    AudioParamId(5),
    AudioParamId(6),
    AudioParamId(7),
    AudioParamId(8),
    AudioParamId(9),
    AudioParamId(10),
];

/// Unique identifier for audio nodes.
///
/// Used for internal bookkeeping.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct AudioNodeId(pub u64);

impl std::fmt::Debug for AudioNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AudioNodeId({})", self.0)
    }
}

/// Unique identifier for audio params.
///
/// Store these in your `AudioProcessor` to get access to `AudioParam` values.
#[derive(Debug)]
pub struct AudioParamId(u64);

impl AudioParamId {
    pub(crate) const fn from_node_id(id: AudioNodeId) -> Self {
        Self(id.0)
    }
}

// bit contrived, but for type safety only the context mod can access the inner u64
impl From<&AudioParamId> for AudioNodeId {
    fn from(i: &AudioParamId) -> Self {
        Self(i.0)
    }
}

/// Describes the current state of the `AudioContext`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AudioContextState {
    /// This context is currently suspended (context time is not proceeding,
    /// audio hardware may be powered down/released).
    Suspended,
    /// Audio is being processed.
    Running,
    /// This context has been released, and can no longer be used to process audio.
    /// All system audio resources have been released.
    Closed,
}

impl From<u8> for AudioContextState {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Suspended,
            1 => Self::Running,
            2 => Self::Closed,
            _ => unreachable!(),
        }
    }
}

/// Handle of the [`AudioNode`](crate::node::AudioNode) to its associated [`BaseAudioContext`].
///
/// Only when implementing the AudioNode trait manually, this struct is of any concern.
///
/// This object allows for communication with the render thread and dynamic lifetime management.
// Legacy nodes receive this from [`BaseAudioContext::register`]; the private injected constructor
// attaches an exact live-registration only after its whole batch is accepted.
// This struct should not derive Clone because of the Drop handler.
pub struct AudioContextRegistration {
    /// Injected lifetime handle, explicitly dropped while `context` is still alive.
    injected_lifetime: Option<injected_node_lifetime::InjectedNodeRegistration>,
    /// Exact public-edge endpoint brand. Permanent magic registrations carry this without an
    /// ordinary lifetime slot; ordinary Gain registrations carry both capabilities.
    injected_connection: Option<injected_connections::InjectedConnectionEndpoint>,
    /// Exact scheduled-source event key, present only after accepted oscillator publication.
    injected_ended: Option<crate::events::InjectedExactEndedEventTarget>,
    /// the audio context in which nodes and connections lives
    context: ConcreteBaseAudioContext,
    /// identify a specific `AudioNode`
    id: AudioNodeId,
}

impl std::fmt::Debug for AudioContextRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioContextRegistration")
            .field("id", &self.id)
            .field(
                "context",
                &format!("BaseAudioContext@{}", self.context.address()),
            )
            .finish()
    }
}

impl AudioContextRegistration {
    #[cfg(test)]
    pub(crate) fn from_injected(
        id: AudioNodeId,
        context: ConcreteBaseAudioContext,
        lifetime: injected_node_lifetime::InjectedNodeRegistration,
    ) -> Self {
        Self {
            injected_lifetime: Some(lifetime),
            injected_connection: None,
            injected_ended: None,
            context,
            id,
        }
    }

    pub(crate) fn from_injected_with_connection(
        id: AudioNodeId,
        context: ConcreteBaseAudioContext,
        lifetime: injected_node_lifetime::InjectedNodeRegistration,
        connection: injected_connections::InjectedConnectionEndpoint,
        kind: injected_connections::InjectedConnectionEndpointKind,
        inputs: usize,
        outputs: usize,
    ) -> Self {
        let matches_context = context
            .injected_node_constructor()
            .is_some_and(|constructor| connection.matches_constructor(constructor));
        if !matches_context
            || !connection.matches_registration(&lifetime, id, kind, inputs, outputs)
        {
            connection.fail_closed_protocol();
            if let Some(constructor) = context.injected_node_constructor() {
                constructor.fail_closed_protocol();
            }
            panic!("exact connection endpoint does not match its context/live registration");
        }
        Self {
            injected_lifetime: Some(lifetime),
            injected_connection: Some(connection),
            injected_ended: None,
            context,
            id,
        }
    }

    pub(crate) fn from_injected_permanent(
        id: AudioNodeId,
        context: ConcreteBaseAudioContext,
        connection: injected_connections::InjectedConnectionEndpoint,
        kind: injected_connections::InjectedConnectionEndpointKind,
        inputs: usize,
        outputs: usize,
    ) -> Self {
        let matches_context = context
            .injected_node_constructor()
            .is_some_and(|constructor| connection.matches_constructor(constructor));
        if !matches_context || !connection.matches_permanent_registration(id, kind, inputs, outputs)
        {
            connection.fail_closed_protocol();
            if let Some(constructor) = context.injected_node_constructor() {
                constructor.fail_closed_protocol();
            }
            panic!("exact permanent endpoint does not match its context/magic registration");
        }
        Self {
            injected_lifetime: None,
            injected_connection: Some(connection),
            injected_ended: None,
            context,
            id,
        }
    }

    /// Get the audio node id of the registration
    #[must_use]
    pub(crate) fn id(&self) -> AudioNodeId {
        self.id
    }

    /// Get the [`BaseAudioContext`] concrete type associated with this `AudioContext`
    #[must_use]
    pub(crate) fn context(&self) -> &ConcreteBaseAudioContext {
        &self.context
    }

    pub(crate) fn matches_injected_lifetime_identity(
        &self,
        identity: &injected_node_lifetime::InjectedNodeRegistrationIdentity,
    ) -> bool {
        self.injected_lifetime
            .as_ref()
            .is_some_and(|lifetime| lifetime.matches_identity(identity))
    }

    pub(crate) fn injected_connection_endpoint(
        &self,
    ) -> Option<&injected_connections::InjectedConnectionEndpoint> {
        self.injected_connection.as_ref()
    }

    pub(crate) fn from_injected_oscillator(
        id: AudioNodeId,
        context: ConcreteBaseAudioContext,
        lifetime: injected_node_lifetime::InjectedNodeRegistration,
        connection: injected_connections::InjectedConnectionEndpoint,
        ended: crate::events::InjectedExactEndedEventTarget,
    ) -> Self {
        let matches_context = context
            .injected_node_constructor()
            .is_some_and(|constructor| connection.matches_constructor(constructor));
        let matches_events = context
            .injected_events()
            .is_some_and(|events| ended.matches_attachment(events, &lifetime, id));
        if !matches_context
            || !matches_events
            || !connection.matches_registration(
                &lifetime,
                id,
                injected_connections::InjectedConnectionEndpointKind::AudioNode,
                0,
                1,
            )
        {
            connection.fail_closed_protocol();
            if let Some(constructor) = context.injected_node_constructor() {
                constructor.fail_closed_protocol();
            }
            panic!("exact oscillator capabilities do not match their context/live registration");
        }
        Self {
            injected_lifetime: Some(lifetime),
            injected_connection: Some(connection),
            injected_ended: Some(ended),
            context,
            id,
        }
    }

    pub(crate) fn set_ended_handler(&self, callback: crate::events::EventHandler) {
        if let Some(ended) = &self.injected_ended {
            if let Err(error) = ended.try_set_handler(callback) {
                panic!("InvalidStateError - exact ended handler installation failed: {error:?}");
            }
            return;
        }
        if self.context.injected_node_constructor().is_some() {
            drop(callback);
            panic!("NotSupportedError - this exact scheduled source has no ended capability");
        }
        self.context
            .set_event_handler(crate::events::EventType::Ended(self.id), callback);
    }

    pub(crate) fn clear_ended_handler(&self) {
        if let Some(ended) = &self.injected_ended {
            ended.clear_handler();
            return;
        }
        if self.context.injected_node_constructor().is_none() {
            self.context
                .clear_event_handler(crate::events::EventType::Ended(self.id));
        }
    }

    /// Send a message to the corresponding audio processor of this node
    ///
    /// The message will be handled by
    /// [`AudioProcessor::onmessage`](crate::render::AudioProcessor::onmessage).
    pub(crate) fn post_message<M: Any + Send + 'static>(&self, msg: M) {
        let wrapped = crate::message::ControlMessage::NodeMessage {
            id: self.id,
            msg: llq::Node::new(Box::new(msg)),
        };
        self.context.send_control_msg(wrapped);
    }
}

impl Drop for AudioContextRegistration {
    fn drop(&mut self) {
        if let Some(lifetime) = self.injected_lifetime.take() {
            // The exact constructor/lifetime capabilities remain alive through `context` while
            // the injected handle publishes its teardown request.
            drop(lifetime);
        } else if self.injected_connection.is_some() {
            // Permanent magic handles are reconstructed freely and never request ordinary node
            // teardown. Whole-graph retirement owns their renderer and host-edge cleanup.
        } else {
            self.context.mark_node_dropped(self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::AudioNode;

    fn require_send_sync_static<T: Send + Sync + 'static>(_: T) {}

    #[test]
    fn test_audio_context_registration_traits() {
        let context = OfflineAudioContext::new(1, 1, 44100.);
        let registration = context.mock_registration();

        // we want to be able to ship AudioNodes to another thread, so the Registration should be
        // Send, Sync and 'static
        require_send_sync_static(registration);
    }

    #[test]
    fn test_offline_audio_context_send_sync() {
        let context = OfflineAudioContext::new(1, 1, 44100.);
        require_send_sync_static(context);
    }

    #[test]
    fn test_online_audio_context_send_sync() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);
        require_send_sync_static(context);
    }

    #[test]
    fn test_context_equals() {
        let context = OfflineAudioContext::new(1, 48000, 96000.);
        let dest = context.destination();
        assert!(dest.context() == context.base());
    }
}
