//! The `AudioContext` type and constructor options
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "diagnostics")]
use crate::context::{AudioBackendDiagnostics, AudioContextDiagnostics};
use crate::context::{
    AudioContextState, AudioControlBatchReservation, AudioNodeLifetimeReservation,
    BaseAudioContext, ConcreteBaseAudioContext,
};
#[cfg(feature = "diagnostics")]
use crate::events::EventPayload;
use crate::events::{EventDispatch, EventHandler, EventLoop, EventType};
use crate::io::{self, AudioBackendManager, ControlThreadInit, NoneBackend, RenderThreadInit};
use crate::media_devices::{enumerate_devices_sync, MediaDeviceInfoKind};
use crate::media_streams::{MediaStream, MediaStreamTrack};
use crate::message::{ControlMessage, OneshotNotify};
use crate::node::{self, AudioNodeOptions};
use crate::render::graph::Graph;
use crate::MediaElement;
use crate::{is_valid_sample_rate, AudioPlaybackStats, AudioRenderCapacity, Event};

use super::hosted::{
    hosted_shutdown_outcome_is_success, hosted_state_outcome_is_success, AudioContextBuilder,
    AudioContextLifecycleError, AudioContextShutdownReceipt, AudioContextStateChangeReceipt,
    HostedAudioContextMode,
};
use crate::output::AudioOutputFactory;

use futures_channel::oneshot;

/// Check if the provided sink_id is available for playback
///
/// It should be "", "none" or a valid output `sinkId` returned from [`enumerate_devices_sync`]
fn is_valid_sink_id(sink_id: &str) -> bool {
    if sink_id.is_empty() || sink_id == "none" {
        true
    } else {
        enumerate_devices_sync()
            .into_iter()
            .filter(|d| d.kind() == MediaDeviceInfoKind::AudioOutput)
            .any(|d| d.device_id() == sink_id)
    }
}

#[derive(Debug)]
enum AudioContextError {
    SinkNotFound { sink_id: String },
    InvalidSampleRate { sample_rate: f32 },
    Backend { error: io::AudioBackendError },
}

impl std::fmt::Display for AudioContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SinkNotFound { sink_id } => {
                write!(f, "NotFoundError - Invalid sinkId: {sink_id:?}")
            }
            Self::InvalidSampleRate { sample_rate } => {
                write!(
                    f,
                    "NotSupportedError - Invalid sample rate: {sample_rate}, should be in the range [3000.0, 768000.0]"
                )
            }
            Self::Backend { error } => write!(f, "InvalidStateError - {error}"),
        }
    }
}

impl Error for AudioContextError {}

impl From<io::AudioBackendError> for AudioContextError {
    fn from(error: io::AudioBackendError) -> Self {
        Self::Backend { error }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SinkSwapControlError {
    Disconnected { operation: &'static str },
}

impl std::fmt::Display for SinkSwapControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected { operation } => write!(
                f,
                "InvalidStateError - render thread disconnected while {operation}"
            ),
        }
    }
}

impl Error for SinkSwapControlError {}

/// Install the recycled graph and replay records that the old renderer had not dequeued.
///
/// A suspended replacement cannot consume concurrently. Its 256-slot channel therefore holds
/// `Startup` plus at most 255 cached records; any remaining suffix is prepended to mutations that
/// were already staged while suspended. The normal resume path submits that suffix in FIFO order.
fn replay_sink_swap_control_messages(
    base: &ConcreteBaseAudioContext,
    sender: &crossbeam_channel::Sender<ControlMessage>,
    graph: Graph,
    pending_msgs: Vec<ControlMessage>,
    original_state: AudioContextState,
) -> Result<(), SinkSwapControlError> {
    sender
        .send(ControlMessage::Startup { graph })
        .map_err(|_| SinkSwapControlError::Disconnected {
            operation: "installing the recycled graph",
        })?;

    if original_state == AudioContextState::Suspended {
        let mut pending = pending_msgs.into_iter();
        while let Some(message) = pending.next() {
            match sender.try_send(message) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(message)) => {
                    let mut deferred = Vec::with_capacity(1 + pending.len());
                    deferred.push(message);
                    deferred.extend(pending);
                    base.prepend_suspended_control_msgs(deferred);
                    break;
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                    return Err(SinkSwapControlError::Disconnected {
                        operation: "replaying cached control messages",
                    });
                }
            }
        }
    } else {
        for message in pending_msgs {
            sender
                .send(message)
                .map_err(|_| SinkSwapControlError::Disconnected {
                    operation: "replaying cached control messages",
                })?;
        }
    }

    Ok(())
}

/// Identify the type of playback, which affects tradeoffs
/// between audio output latency and power consumption
#[derive(Copy, Clone, Debug, Default)]
pub enum AudioContextLatencyCategory {
    /// Balance audio output latency and power consumption.
    Balanced,
    /// Provide the lowest audio output latency possible without glitching. This is the default.
    #[default]
    Interactive,
    /// Prioritize sustained playback without interruption over audio output latency.
    ///
    /// Lowest power consumption.
    Playback,
    /// Specify the number of seconds of latency
    ///
    /// This latency is not guaranteed to be applied, it depends on the audio hardware capabilities
    Custom(f64),
}

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
/// This allows users to ask for a particular render quantum size.
///
/// Currently, only the default value is available
#[derive(Default)]
pub enum AudioContextRenderSizeCategory {
    /// The default value of 128 frames
    #[default]
    Default,
}

/// Specify the playback configuration for the [`AudioContext`] constructor.
///
/// All fields are optional and will default to the value best suited for interactive playback on
/// your hardware configuration.
///
/// For future compatibility, it is best to construct a default implementation of this struct and
/// set the fields you would like to override:
/// ```
/// use web_audio_api::context::AudioContextOptions;
///
/// // Request a sample rate of 44.1 kHz, leave other fields to their default values
/// let opts = AudioContextOptions {
///     sample_rate: Some(44100.),
///     ..AudioContextOptions::default()
/// };
#[derive(Clone, Debug, Default)]
pub struct AudioContextOptions {
    /// Identify the type of playback, which affects tradeoffs between audio output latency and
    /// power consumption.
    pub latency_hint: AudioContextLatencyCategory,

    /// Sample rate of the audio context and audio output hardware. Use `None` for a default value.
    pub sample_rate: Option<f32>,

    /// The audio output device
    /// - use `""` for the default audio output device
    /// - use `"none"` to process the audio graph without playing through an audio output device.
    /// - use `"sinkId"` to use the specified audio sink id, obtained with [`enumerate_devices_sync`]
    pub sink_id: String,

    /// Option to request a default, optimized or specific render quantum size. It is a hint that might not be honored.
    pub render_size_hint: AudioContextRenderSizeCategory,
}

/// This interface represents an audio graph whose `AudioDestinationNode` is routed to a real-time
/// output device that produces a signal directed at the user.
// the naming comes from the web audio specification
#[allow(clippy::module_name_repetitions)]
pub struct AudioContext {
    /// represents the underlying `BaseAudioContext`
    base: ConcreteBaseAudioContext,
    /// Provider for rendering performance metrics
    render_capacity: AudioRenderCapacity,
    /// Provider for playback statistics
    playback_stats: AudioPlaybackStats,
    mode: AudioContextMode,
}

enum AudioContextMode {
    Legacy(LegacyAudioContextMode),
    Hosted(HostedAudioContextMode),
}

struct LegacyAudioContextMode {
    /// audio backend (play/pause functionality)
    backend_manager: Mutex<Box<dyn AudioBackendManager>>,
    /// true while the render thread has not yet processed its initial Startup message
    startup_pending: std::sync::Arc<AtomicBool>,
    /// Initializer for the render thread (when restart is required)
    render_thread_init: Mutex<Option<RenderThreadInit>>,
}

impl std::fmt::Debug for AudioContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioContext")
            .field("sink_id", &self.sink_id())
            .field("base_latency", &self.base_latency())
            .field("output_latency", &self.output_latency())
            .field("base", &self.base())
            .finish_non_exhaustive()
    }
}

impl Drop for AudioContext {
    fn drop(&mut self) {
        let state = self.base.state();
        match &mut self.mode {
            AudioContextMode::Legacy(legacy) => {
                // Continue playing the stream if the legacy AudioContext goes out of scope.
                if state == AudioContextState::Running {
                    let tombstone = Box::new(NoneBackend::void());
                    let original =
                        std::mem::replace(legacy.backend_manager.get_mut().unwrap(), tombstone);
                    Box::leak(original);
                }
            }
            AudioContextMode::Hosted(hosted) => {
                self.render_capacity.close();
                hosted.request_silent_on_drop();
            }
        }
    }
}

impl BaseAudioContext for AudioContext {
    fn base(&self) -> &ConcreteBaseAudioContext {
        &self.base
    }
}

impl Default for AudioContext {
    fn default() -> Self {
        Self::new(AudioContextOptions::default())
    }
}

impl AudioContext {
    /// Starts construction of an exact hosted context using the supplied output factory.
    #[must_use]
    pub fn builder(output: Arc<dyn AudioOutputFactory>) -> AudioContextBuilder {
        AudioContextBuilder::new(output)
    }

    /// Constructs a hosted `GainNode` while attaching host accounting to both exact graph nodes.
    ///
    /// The reservation is released only after the Gain node and its AudioParam are physically
    /// reclaimed, or after rejected construction has fully rolled back. This operation is
    /// available only on contexts returned by [`AudioContext::builder`].
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_gain_with_lifetime_reservation(
        &self,
        reservation: AudioNodeLifetimeReservation,
    ) -> node::GainNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - lifetime reservations require an exact hosted AudioContext"
        );
        node::GainNode::new_injected_with_lifetime(
            &self.base,
            node::GainOptions::default(),
            Some(reservation),
        )
    }

    /// Constructs a hosted `GainNode` with exact graph-lifetime and four-command reservations.
    ///
    /// The command reservation remains held through suspension, renderer application, and
    /// off-render-thread batch reclamation. Rejected construction releases both reservations only
    /// after rollback. This operation is available only on hosted contexts.
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_gain_with_reservations(
        &self,
        lifetime: AudioNodeLifetimeReservation,
        control: AudioControlBatchReservation,
    ) -> node::GainNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - resource reservations require an exact hosted AudioContext"
        );
        node::GainNode::new_injected_with_reservations(
            &self.base,
            node::GainOptions::default(),
            Some(lifetime),
            Some(control),
        )
    }

    /// Constructs a hosted `ConstantSourceNode` while attaching host accounting to the source
    /// and its offset parameter.
    ///
    /// The reservation is released only after both exact graph nodes are physically reclaimed,
    /// or after rejected construction has fully rolled back. This operation is available only on
    /// contexts returned by [`AudioContext::builder`].
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_constant_source_with_lifetime_reservation(
        &self,
        reservation: AudioNodeLifetimeReservation,
    ) -> node::ConstantSourceNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - lifetime reservations require an exact hosted AudioContext"
        );
        node::ConstantSourceNode::new_injected_with_lifetime(
            &self.base,
            node::ConstantSourceOptions::default(),
            Some(reservation),
        )
    }

    /// Constructs a hosted `ConstantSourceNode` with exact graph-lifetime and four-command
    /// reservations.
    ///
    /// The command reservation remains held through suspension, renderer application, and
    /// off-render-thread batch reclamation. Rejected construction releases both reservations only
    /// after rollback. This operation is available only on hosted contexts.
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_constant_source_with_reservations(
        &self,
        lifetime: AudioNodeLifetimeReservation,
        control: AudioControlBatchReservation,
    ) -> node::ConstantSourceNode {
        self.create_constant_source_with_options_and_reservations(
            node::ConstantSourceOptions::default(),
            lifetime,
            control,
        )
    }

    /// Constructs a hosted `ConstantSourceNode` with caller-selected options and exact
    /// graph/control reservations.
    ///
    /// The initial offset is part of the atomic four-command construction transaction. The
    /// command reservation remains held through suspension, renderer application, and off-render
    /// reclamation, while the graph reservation follows both exact nodes through physical reclaim.
    ///
    /// # Panics
    ///
    /// Panics on a legacy context, an invalid offset, or rejected exact construction.
    pub fn create_constant_source_with_options_and_reservations(
        &self,
        options: node::ConstantSourceOptions,
        lifetime: AudioNodeLifetimeReservation,
        control: AudioControlBatchReservation,
    ) -> node::ConstantSourceNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - resource reservations require an exact hosted AudioContext"
        );
        node::ConstantSourceNode::new_injected_with_reservations(
            &self.base,
            options,
            Some(lifetime),
            Some(control),
        )
    }

    /// Constructs a hosted fixed-wave `OscillatorNode` while attaching host accounting to all
    /// three exact graph nodes (oscillator, frequency, and detune).
    ///
    /// The reservation is released only after all three nodes are physically reclaimed, or after
    /// rejected construction has fully rolled back. This operation is available only on contexts
    /// returned by [`AudioContext::builder`].
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_oscillator_with_lifetime_reservation(
        &self,
        reservation: AudioNodeLifetimeReservation,
    ) -> node::OscillatorNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - lifetime reservations require an exact hosted AudioContext"
        );
        node::OscillatorNode::new_injected_with_lifetime(
            &self.base,
            node::OscillatorOptions::default(),
            Some(reservation),
        )
    }

    /// Constructs a hosted fixed-wave `OscillatorNode` with exact graph-lifetime and
    /// seven-command reservations.
    ///
    /// The command reservation remains held through suspension, renderer application, and
    /// off-render-thread batch reclamation. Rejected construction releases both reservations only
    /// after rollback. This operation is available only on hosted contexts.
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_oscillator_with_reservations(
        &self,
        lifetime: AudioNodeLifetimeReservation,
        control: AudioControlBatchReservation,
    ) -> node::OscillatorNode {
        self.create_oscillator_with_options_and_reservations(
            node::OscillatorOptions::default(),
            lifetime,
            control,
        )
    }

    /// Constructs a hosted `OscillatorNode` with caller-selected options and exact graph/control
    /// reservations.
    ///
    /// A custom oscillator must carry a [`crate::PeriodicWave`] created for this same context. Its
    /// fixed native table and host lease move into the atomic seven-command construction and remain
    /// owned through renderer retirement.
    ///
    /// # Panics
    ///
    /// Panics on a legacy context, an invalid custom-wave combination, a foreign wave, or rejected
    /// exact construction.
    pub fn create_oscillator_with_options_and_reservations(
        &self,
        options: node::OscillatorOptions,
        lifetime: AudioNodeLifetimeReservation,
        control: AudioControlBatchReservation,
    ) -> node::OscillatorNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - resource reservations require an exact hosted AudioContext"
        );
        node::OscillatorNode::new_injected_with_reservations(
            &self.base,
            options,
            Some(lifetime),
            Some(control),
        )
    }

    /// Constructs a hosted `AudioBufferSourceNode` while attaching host accounting to the source,
    /// detune, and playback-rate graph nodes.
    ///
    /// The reservation is released only after all three nodes are physically reclaimed, or after
    /// rejected construction has fully rolled back. This operation is available only on contexts
    /// returned by [`AudioContext::builder`].
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_buffer_source_with_lifetime_reservation(
        &self,
        reservation: AudioNodeLifetimeReservation,
    ) -> node::AudioBufferSourceNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - lifetime reservations require an exact hosted AudioContext"
        );
        node::AudioBufferSourceNode::new_injected_with_lifetime(
            &self.base,
            node::AudioBufferSourceOptions::default(),
            Some(reservation),
        )
    }

    /// Constructs a hosted `AudioBufferSourceNode` with exact graph-lifetime and seven-command
    /// reservations.
    ///
    /// The command reservation remains held through suspension, renderer application, and
    /// off-render-thread batch reclamation. Rejected construction releases both reservations only
    /// after rollback. This operation is available only on hosted contexts.
    ///
    /// # Panics
    ///
    /// Panics when called on a legacy context or when exact construction is rejected.
    pub fn create_buffer_source_with_reservations(
        &self,
        lifetime: AudioNodeLifetimeReservation,
        control: AudioControlBatchReservation,
    ) -> node::AudioBufferSourceNode {
        assert!(
            self.is_hosted(),
            "NotSupportedError - resource reservations require an exact hosted AudioContext"
        );
        node::AudioBufferSourceNode::new_injected_with_reservations(
            &self.base,
            node::AudioBufferSourceOptions::default(),
            Some(lifetime),
            Some(control),
        )
    }

    pub(super) fn from_hosted_parts(
        base: ConcreteBaseAudioContext,
        render_capacity: AudioRenderCapacity,
        playback_stats: AudioPlaybackStats,
        hosted: HostedAudioContextMode,
    ) -> Self {
        Self {
            base,
            render_capacity,
            playback_stats,
            mode: AudioContextMode::Hosted(hosted),
        }
    }

    fn legacy(&self) -> &LegacyAudioContextMode {
        match &self.mode {
            AudioContextMode::Legacy(legacy) => legacy,
            AudioContextMode::Hosted(_) => {
                unreachable!("legacy-only AudioContext path selected for hosted context")
            }
        }
    }

    fn is_hosted(&self) -> bool {
        matches!(self.mode, AudioContextMode::Hosted(_))
    }

    #[cfg(test)]
    fn set_hosted_state_request_hook_for_test(
        &self,
        observer: Arc<dyn Fn() + Send + Sync + 'static>,
    ) {
        let AudioContextMode::Hosted(hosted) = &self.mode else {
            panic!("test hook requires hosted context")
        };
        hosted.set_state_request_hook_for_test(observer);
    }

    /// Creates and returns a new `AudioContext` object.
    ///
    /// This will play live audio on the default output device.
    ///
    /// ```no_run
    /// use web_audio_api::context::{AudioContext, AudioContextOptions};
    ///
    /// // Request a sample rate of 44.1 kHz and default latency (buffer size 128, if available)
    /// let opts = AudioContextOptions {
    ///     sample_rate: Some(44100.),
    ///     ..AudioContextOptions::default()
    /// };
    ///
    /// // Setup the audio context that will emit to your speakers
    /// let context = AudioContext::new(opts);
    ///
    /// // Alternatively, use the default constructor to get the best settings for your hardware
    /// // let context = AudioContext::default();
    /// ```
    ///
    /// # Panics
    ///
    /// The `AudioContext` constructor will panic when an invalid `sinkId` is provided in the
    /// `AudioContextOptions`, when the sample rate is outside the valid range [3000.0, 768000.0],
    /// or when the selected audio backend cannot create or start the output stream. Use
    /// [`Self::try_new`] to handle these errors without panicking.
    #[must_use]
    pub fn new(options: AudioContextOptions) -> Self {
        Self::try_new_inner(options).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Creates and returns a new `AudioContext` object.
    ///
    /// This will play live audio on the requested output device and returns backend errors instead
    /// of panicking when the stream cannot be created.
    ///
    /// # Errors
    ///
    /// Returns an error when the sink id is invalid, the sample rate is outside the valid range
    /// [3000.0, 768000.0], or when the selected audio backend cannot create or start the output
    /// stream.
    pub fn try_new(options: AudioContextOptions) -> Result<Self, Box<dyn Error>> {
        Self::try_new_inner(options).map_err(Into::into)
    }

    fn try_new_inner(options: AudioContextOptions) -> Result<Self, AudioContextError> {
        // https://webaudio.github.io/web-audio-api/#validating-sink-identifier
        if !is_valid_sink_id(&options.sink_id) {
            return Err(AudioContextError::SinkNotFound {
                sink_id: options.sink_id,
            });
        }

        // Validate sample_rate if provided
        // https://webaudio.github.io/web-audio-api/#sample-rates
        if let Some(sample_rate) = options.sample_rate {
            if !is_valid_sample_rate(sample_rate) {
                return Err(AudioContextError::InvalidSampleRate { sample_rate });
            }
        }

        // Set up the audio output thread
        let (control_thread_init, render_thread_init) = io::thread_init();
        let startup_pending = Arc::clone(&render_thread_init.startup_pending);
        let backend = io::build_output(options, render_thread_init.clone())?;

        let ControlThreadInit {
            state,
            frames_played,
            stats,
            ctrl_msg_send,
            control_batch_send,
            control_batch_applied,
            event_send,
            event_recv,
        } = control_thread_init;

        // Construct the audio Graph and hand it to the render thread
        let (node_id_producer, node_id_consumer) = llq::Queue::new().split();
        let graph = Graph::new(node_id_producer);
        let message = ControlMessage::Startup { graph };
        ctrl_msg_send.send(message).unwrap();

        // Set up the event loop thread that handles the events spawned by the render thread
        let event_loop = EventLoop::new(event_recv);

        // Put everything together in the BaseAudioContext (shared with offline context)
        let base = ConcreteBaseAudioContext::new(
            backend.sample_rate(),
            backend.number_of_channels(),
            state,
            frames_played,
            ctrl_msg_send,
            control_batch_send,
            control_batch_applied,
            event_send,
            event_loop.clone(),
            false,
            node_id_consumer,
        );

        // Setup AudioRenderCapacity for this context
        let render_capacity = AudioRenderCapacity::new(base.clone(), stats.clone());
        let playback_stats = AudioPlaybackStats::new(base.clone(), stats);

        // As the final step, spawn a thread for the event loop. If we do this earlier we may miss
        // event handling of the initial events that are emitted right after render thread
        // construction.
        event_loop.run_in_thread();

        Ok(Self {
            base,
            render_capacity,
            playback_stats,
            mode: AudioContextMode::Legacy(LegacyAudioContextMode {
                backend_manager: Mutex::new(backend),
                startup_pending,
                render_thread_init: Mutex::new(Some(render_thread_init)),
            }),
        })
    }

    /// This represents the number of seconds of processing latency incurred by
    /// the `AudioContext` passing the audio from the `AudioDestinationNode`
    /// to the audio subsystem.
    // We don't do any buffering between rendering the audio and sending
    // it to the audio subsystem, so this value is zero. (see Gecko)
    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn base_latency(&self) -> f64 {
        0.
    }

    /// The estimation in seconds of audio output latency, i.e., the interval
    /// between the time the UA requests the host system to play a buffer and
    /// the time at which the first sample in the buffer is actually processed
    /// by the audio output device.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn output_latency(&self) -> f64 {
        self.try_output_latency()
            .unwrap_or_else(|e| panic!("InvalidStateError - {e}"))
    }

    /// The estimation in seconds of audio output latency.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected audio backend cannot query the output latency.
    fn try_output_latency(&self) -> Result<f64, Box<dyn Error>> {
        match &self.mode {
            AudioContextMode::Legacy(legacy) => {
                Ok(legacy.backend_manager.lock().unwrap().output_latency()?)
            }
            AudioContextMode::Hosted(hosted) => Ok(hosted.config().output_latency()),
        }
    }

    /// Identifier or the information of the current audio output device.
    ///
    /// The initial value is `""`, which means the default audio output device.
    #[allow(clippy::missing_panics_doc)]
    pub fn sink_id(&self) -> String {
        match &self.mode {
            AudioContextMode::Legacy(legacy) => {
                legacy.backend_manager.lock().unwrap().sink_id().to_owned()
            }
            AudioContextMode::Hosted(hosted) => hosted.config().accepted_sink_id().to_owned(),
        }
    }

    /// Returns an [`AudioRenderCapacity`] instance associated with an AudioContext.
    #[must_use]
    pub fn render_capacity(&self) -> AudioRenderCapacity {
        self.render_capacity.clone()
    }

    /// Returns an [`AudioPlaybackStats`] instance associated with this `AudioContext`.
    #[must_use]
    pub fn playback_stats(&self) -> AudioPlaybackStats {
        self.playback_stats.clone()
    }

    /// Update the current audio output device.
    ///
    /// The provided `sink_id` string must match a device name `enumerate_devices_sync`.
    ///
    /// Supplying `"none"` for the `sink_id` will process the audio graph without playing through an
    /// audio output device.
    ///
    /// This function operates synchronously and might block the current thread. An async version
    /// is currently not implemented.
    #[allow(clippy::needless_collect, clippy::missing_panics_doc)]
    pub fn set_sink_id_sync(&self, sink_id: String) -> Result<(), Box<dyn Error>> {
        if self.is_hosted() {
            if self.sink_id() == sink_id {
                return Ok(());
            }
            return Err("NotSupportedError: hosted output replacement is not available".into());
        }
        log::debug!("SinkChange requested");
        if self.sink_id() == sink_id {
            log::debug!("SinkChange: no-op");
            return Ok(()); // sink is already active
        }

        if !is_valid_sink_id(&sink_id) {
            Err(format!("NotFoundError: invalid sinkId {sink_id}"))?;
        };

        log::debug!("SinkChange: locking backend manager");
        let legacy = self.legacy();
        let mut backend_manager_guard = legacy.backend_manager.lock().unwrap();
        let original_state = self.state();
        if original_state == AudioContextState::Closed {
            log::debug!("SinkChange: context is closed");
            return Ok(());
        }

        // Acquire exclusive lock on ctrl msg sender
        log::debug!("SinkChange: locking message channel");
        let ctrl_msg_send = self.base.lock_control_msg_sender();

        // Flush out the ctrl msg receiver, cache
        let render_thread_init = legacy
            .render_thread_init
            .lock()
            .unwrap()
            .as_ref()
            .cloned()
            .expect("open context retains its render-thread initializer");
        let mut pending_msgs: Vec<_> = render_thread_init.ctrl_msg_recv.try_iter().collect();

        // Acquire the active audio graph from the current render thread, shutting it down
        let graph = if matches!(pending_msgs.first(), Some(ControlMessage::Startup { .. })) {
            // Handle the edge case where the previous backend was suspended for its entire lifetime.
            // In this case, the `Startup` control message was never processed.
            log::debug!("SinkChange: recover unstarted graph");

            let msg = pending_msgs.remove(0);
            match msg {
                ControlMessage::Startup { graph } => graph,
                _ => unreachable!(),
            }
        } else {
            // Acquire the audio graph from the current render thread, shutting it down
            log::debug!("SinkChange: recover graph from render thread");

            let (graph_send, graph_recv) = crossbeam_channel::bounded(1);
            let message = ControlMessage::CloseAndRecycle { sender: graph_send };
            ctrl_msg_send
                .send(message)
                .map_err(|_| SinkSwapControlError::Disconnected {
                    operation: "requesting graph recycle",
                })?;
            if original_state == AudioContextState::Suspended {
                // We must wake up the render thread to be able to handle the shutdown.
                // No new audio will be produced because it will receive the shutdown command first.
                backend_manager_guard.resume()?;
            }
            graph_recv
                .recv()
                .map_err(|_| SinkSwapControlError::Disconnected {
                    operation: "waiting for graph recycle",
                })?
        };

        log::debug!("SinkChange: closing audio stream");
        backend_manager_guard.close()?;

        // hotswap the backend
        let options = AudioContextOptions {
            sample_rate: Some(self.sample_rate()),
            latency_hint: AudioContextLatencyCategory::default(), // todo reuse existing setting
            sink_id,
            render_size_hint: AudioContextRenderSizeCategory::default(), // todo reuse existing setting
        };
        log::debug!("SinkChange: starting audio stream");
        *backend_manager_guard = io::build_output(options, render_thread_init)?;

        // if the previous backend state was suspend, suspend the new one before shipping the graph
        if original_state == AudioContextState::Suspended {
            log::debug!("SinkChange: suspending audio stream");
            backend_manager_guard.suspend()?;
        }

        // Replay through the sender whose write guard is already held. Calling
        // `ConcreteBaseAudioContext::send_control_msg` here would try to acquire a read guard on
        // the same RwLock and self-deadlock whenever the cached backlog is non-empty.
        replay_sink_swap_control_messages(
            &self.base,
            &ctrl_msg_send,
            graph,
            pending_msgs,
            original_state,
        )?;

        // Explicitly release both serialization locks after Startup and the cached FIFO are sent
        // or, for a full suspended channel, staged ahead of later suspended mutations.
        drop(ctrl_msg_send);
        drop(backend_manager_guard);

        // trigger event when all the work is done
        let _ = self.base.send_event_with(EventDispatch::sink_change);

        log::debug!("SinkChange: done");
        Ok(())
    }

    /// Register callback to run when the audio sink has changed
    ///
    /// Only a single event handler is active at any time. Calling this method multiple times will
    /// override the previous event handler.
    pub fn set_onsinkchange<F: FnMut(Event) + Send + 'static>(&self, mut callback: F) {
        let callback = move |_| {
            callback(Event {
                type_: "sinkchange",
            })
        };

        self.base().set_event_handler(
            EventType::SinkChange,
            EventHandler::Multiple(Box::new(callback)),
        );
    }

    /// Unset the callback to run when the audio sink has changed
    pub fn clear_onsinkchange(&self) {
        self.base().clear_event_handler(EventType::SinkChange);
    }

    /// Request a structured diagnostic report of the audio context.
    ///
    /// The report is collected asynchronously: backend details are captured on the control thread,
    /// while render thread and graph details are captured in the realtime render thread. The
    /// callback is invoked once on the event loop thread.
    ///
    /// This API is available with the `diagnostics` crate feature.
    #[cfg(feature = "diagnostics")]
    #[allow(clippy::missing_panics_doc)]
    pub fn run_diagnostics<F: Fn(AudioContextDiagnostics) + Send + 'static>(&self, callback: F) {
        if self.is_hosted() {
            drop(callback);
            panic!("NotSupportedError: hosted diagnostics are not available");
        }
        let backend = {
            let backend = self.legacy().backend_manager.lock().unwrap();
            AudioBackendDiagnostics {
                name: backend.name().to_string(),
                sink_id: backend.sink_id().to_string(),
                output_latency: backend.output_latency().ok(),
            }
        };

        let callback = move |v| match v {
            EventPayload::Diagnostics(v) => {
                callback(v);
            }
            _ => unreachable!(),
        };

        self.base().set_event_handler(
            EventType::Diagnostics,
            EventHandler::Once(Box::new(callback)),
        );

        self.base()
            .send_control_msg(ControlMessage::RunDiagnostics { backend });
    }

    /// Suspends the progression of time in the audio context.
    ///
    /// This will temporarily halt audio hardware access and reducing CPU/battery usage in the
    /// process.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    /// * The hosted context's fixed lifecycle request capacity is temporarily contended or full.
    ///   Call [`Self::request_suspend`] to handle that bounded condition without panicking. A
    ///   downstream isolate must provide its own wake/retry policy rather than blocking here.
    pub async fn suspend(&self) {
        if let AudioContextMode::Hosted(hosted) = &self.mode {
            let receipt = hosted
                .request_suspend()
                .unwrap_or_else(|error| panic!("InvalidStateError - {error:?}"));
            let outcome = receipt.await;
            if !hosted_state_outcome_is_success(outcome) {
                panic!("InvalidStateError - hosted suspend failed: {outcome:?}");
            }
            return;
        }
        // Don't lock the backend manager because we can't hold is across the await point
        log::debug!("Suspend called");

        let state = self.state();
        if state == AudioContextState::Closed {
            log::debug!("Suspend no-op - context is closed");
            return;
        }

        if state != AudioContextState::Running
            && !self.legacy().startup_pending.load(Ordering::Acquire)
        {
            log::debug!("Suspend no-op - context is not running");
            return;
        }

        // Pause rendering via a control message
        let (sender, receiver) = oneshot::channel();
        let notify = OneshotNotify::Async(sender);
        self.base
            .suspend_control_msgs(ControlMessage::Suspend { notify });

        // Wait for the render thread to have processed the suspend message.
        // The AudioContextState will be updated by the render thread.
        log::debug!("Suspending audio graph, waiting for signal..");
        receiver.await.unwrap();

        // Then ask the audio host to suspend the stream
        log::debug!("Suspended audio graph. Suspending audio stream..");
        self.legacy()
            .backend_manager
            .lock()
            .unwrap()
            .suspend()
            .unwrap_or_else(|e| panic!("InvalidStateError - {e}"));

        log::debug!("Suspended audio stream");
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    /// * The hosted context's fixed lifecycle request capacity is temporarily contended or full.
    ///   Call [`Self::request_resume`] to handle that bounded condition without panicking. A
    ///   downstream isolate must provide its own wake/retry policy rather than blocking here.
    pub async fn resume(&self) {
        if let AudioContextMode::Hosted(hosted) = &self.mode {
            let receipt = hosted
                .request_resume()
                .unwrap_or_else(|error| panic!("InvalidStateError - {error:?}"));
            let outcome = receipt.await;
            if !hosted_state_outcome_is_success(outcome) {
                panic!("InvalidStateError - hosted resume failed: {outcome:?}");
            }
            return;
        }
        let (sender, receiver) = oneshot::channel();

        {
            // Lock the backend manager mutex to avoid concurrent calls
            log::debug!("Resume called, locking backend manager");
            let backend_manager_guard = self.legacy().backend_manager.lock().unwrap();

            if self.state() != AudioContextState::Suspended {
                log::debug!("Resume no-op - context is not suspended");
                return;
            }

            // Ask the audio host to resume the stream
            backend_manager_guard
                .resume()
                .unwrap_or_else(|e| panic!("InvalidStateError - {e}"));

            // Then, ask to resume rendering via a control message
            log::debug!("Resumed audio stream, waking audio graph");
            let notify = OneshotNotify::Async(sender);
            self.base
                .resume_control_msgs(ControlMessage::Resume { notify });

            // Drop the Mutex guard so we won't hold it across an await point
        }

        // Wait for the render thread to have processed the resume message
        // The AudioContextState will be updated by the render thread.
        receiver.await.unwrap();
        log::debug!("Resumed audio graph");
    }

    /// Closes the `AudioContext`, releasing the system resources being used.
    ///
    /// This will not automatically release all `AudioContext`-created objects, but will suspend
    /// the progression of the currentTime, and stop processing audio data.
    ///
    /// # Panics
    ///
    /// Hosted close is idempotent, but panics when awaited from this context's event thread or
    /// when lifecycle retirement is unconfirmed/controller-terminated. A confirmed degraded
    /// report still completes this convenience method; [`Self::request_close`] exposes its typed
    /// details. Legacy close preserves its backend error behavior.
    pub async fn close(&self) {
        if let AudioContextMode::Hosted(hosted) = &self.mode {
            if hosted.is_event_thread() {
                panic!("InvalidStateError - EventThread");
            }
            self.render_capacity.close();
            let outcome = hosted
                .request_close()
                .unwrap_or_else(|error| panic!("InvalidStateError - {error:?}"))
                .await;
            if !hosted_shutdown_outcome_is_success(&outcome) {
                panic!("InvalidStateError - hosted close failed: {outcome:?}");
            }
            return;
        }
        // Don't lock the backend manager because we can't hold is across the await point
        log::debug!("Close called");

        if self.state() == AudioContextState::Closed {
            log::debug!("Close no-op - context is already closed");
            return;
        }

        // Permanently stop AudioRenderCapacity before closing so surviving public clones cannot
        // restart event production during shutdown.
        self.render_capacity.close();

        if self.state() == AudioContextState::Running {
            // First, stop rendering via a control message
            let (sender, receiver) = oneshot::channel();
            let notify = OneshotNotify::Async(sender);
            self.base.send_control_msg(ControlMessage::Close { notify });

            // Wait for the render thread to have processed the suspend message.
            // The AudioContextState will be updated by the render thread.
            log::debug!("Suspending audio graph, waiting for signal..");
            receiver.await.unwrap();
        } else {
            // if the context is not running, change the state manually
            self.base.set_state(AudioContextState::Closed);
        }

        // Then ask the audio host to close the stream
        log::debug!("Suspended audio graph. Closing audio stream..");
        self.legacy()
            .backend_manager
            .lock()
            .unwrap()
            .close()
            .unwrap_or_else(|e| panic!("InvalidStateError - {e}"));
        self.retire_render_thread_init();

        log::debug!("Closed audio stream");
    }

    /// Suspends the progression of time in the audio context.
    ///
    /// This will temporarily halt audio hardware access and reducing CPU/battery usage in the
    /// process.
    ///
    /// This function operates synchronously and blocks the current thread until the audio thread
    /// has stopped processing.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    /// * The hosted context's fixed lifecycle request capacity is temporarily contended or full;
    ///   [`Self::request_suspend`] exposes that condition as a typed error.
    pub fn suspend_sync(&self) {
        if let AudioContextMode::Hosted(hosted) = &self.mode {
            let receipt = hosted
                .request_suspend()
                .unwrap_or_else(|error| panic!("InvalidStateError - {error:?}"));
            let outcome = receipt.wait();
            if !hosted_state_outcome_is_success(outcome) {
                panic!("InvalidStateError - hosted suspend failed: {outcome:?}");
            }
            return;
        }
        // Lock the backend manager mutex to avoid concurrent calls
        log::debug!("Suspend_sync called, locking backend manager");
        let backend_manager_guard = self.legacy().backend_manager.lock().unwrap();

        let state = self.state();
        if state == AudioContextState::Closed {
            log::debug!("Suspend_sync no-op - context is closed");
            return;
        }

        if state != AudioContextState::Running
            && !self.legacy().startup_pending.load(Ordering::Acquire)
        {
            log::debug!("Suspend_sync no-op - context is not running");
            return;
        }

        // Pause rendering via a control message
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let notify = OneshotNotify::Sync(sender);
        self.base
            .suspend_control_msgs(ControlMessage::Suspend { notify });

        // Wait for the render thread to have processed the suspend message.
        // The AudioContextState will be updated by the render thread.
        log::debug!("Suspending audio graph, waiting for signal..");
        receiver.recv().ok();

        // Then ask the audio host to suspend the stream
        log::debug!("Suspended audio graph. Suspending audio stream..");
        backend_manager_guard
            .suspend()
            .unwrap_or_else(|e| panic!("InvalidStateError - {e}"));

        log::debug!("Suspended audio stream");
    }

    /// Resumes the progression of time in an audio context that has previously been
    /// suspended/paused.
    ///
    /// This function operates synchronously and blocks the current thread until the audio thread
    /// has started processing again.
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * The audio device is not available
    /// * For a `BackendSpecificError`
    /// * The hosted context's fixed lifecycle request capacity is temporarily contended or full;
    ///   [`Self::request_resume`] exposes that condition as a typed error.
    pub fn resume_sync(&self) {
        if let AudioContextMode::Hosted(hosted) = &self.mode {
            let receipt = hosted
                .request_resume()
                .unwrap_or_else(|error| panic!("InvalidStateError - {error:?}"));
            let outcome = receipt.wait();
            if !hosted_state_outcome_is_success(outcome) {
                panic!("InvalidStateError - hosted resume failed: {outcome:?}");
            }
            return;
        }
        // Lock the backend manager mutex to avoid concurrent calls
        log::debug!("Resume_sync called, locking backend manager");
        let backend_manager_guard = self.legacy().backend_manager.lock().unwrap();

        if self.state() != AudioContextState::Suspended {
            log::debug!("Resume no-op - context is not suspended");
            return;
        }

        // Ask the audio host to resume the stream
        backend_manager_guard
            .resume()
            .unwrap_or_else(|e| panic!("InvalidStateError - {e}"));

        // Then, ask to resume rendering via a control message
        log::debug!("Resumed audio stream, waking audio graph");
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let notify = OneshotNotify::Sync(sender);
        self.base
            .resume_control_msgs(ControlMessage::Resume { notify });

        // Wait for the render thread to have processed the resume message
        // The AudioContextState will be updated by the render thread.
        receiver.recv().ok();
        log::debug!("Resumed audio graph");
    }

    /// Closes the `AudioContext`, releasing the system resources being used.
    ///
    /// This will not automatically release all `AudioContext`-created objects, but will suspend
    /// the progression of the currentTime, and stop processing audio data.
    ///
    /// This function operates synchronously and blocks the current thread until the audio thread
    /// has stopped processing.
    ///
    /// # Panics
    ///
    /// Hosted close is idempotent, but panics when called from this context's event thread or when
    /// lifecycle retirement is unconfirmed/controller-terminated. A confirmed degraded report
    /// still completes this convenience method; [`Self::request_close`] exposes its typed details.
    /// Legacy close preserves its backend error behavior.
    pub fn close_sync(&self) {
        if let AudioContextMode::Hosted(hosted) = &self.mode {
            if hosted.is_event_thread() {
                panic!(
                    "InvalidStateError: cannot synchronously close a context from its event thread"
                );
            }
            self.render_capacity.close();
            let outcome = hosted
                .request_close()
                .unwrap_or_else(|error| panic!("InvalidStateError - {error:?}"))
                .wait();
            if !hosted_shutdown_outcome_is_success(&outcome) {
                panic!("InvalidStateError - hosted close failed: {outcome:?}");
            }
            return;
        }
        // Lock the backend manager mutex to avoid concurrent calls
        log::debug!("Close_sync called, locking backend manager");
        let backend_manager_guard = self.legacy().backend_manager.lock().unwrap();

        if self.state() == AudioContextState::Closed {
            log::debug!("Close no-op - context is already closed");
            return;
        }

        // Permanently stop AudioRenderCapacity before closing so surviving public clones cannot
        // restart event production during shutdown.
        self.render_capacity.close();

        // First, stop rendering via a control message
        if self.state() == AudioContextState::Running {
            let (sender, receiver) = crossbeam_channel::bounded(0);
            let notify = OneshotNotify::Sync(sender);
            self.base.send_control_msg(ControlMessage::Close { notify });

            // Wait for the render thread to have processed the suspend message.
            // The AudioContextState will be updated by the render thread.
            log::debug!("Suspending audio graph, waiting for signal..");
            receiver.recv().ok();
        } else {
            // if the context is not running, change the state manually
            self.base.set_state(AudioContextState::Closed);
        }

        // Then ask the audio host to close the stream
        log::debug!("Suspended audio graph. Closing audio stream..");
        backend_manager_guard
            .close()
            .unwrap_or_else(|e| panic!("InvalidStateError - {e}"));
        self.retire_render_thread_init();

        log::debug!("Closed audio stream");
    }

    /// Drop the context-owned receiver clone and synchronously clear any records that the closing
    /// renderer did not dequeue. This runs on the control side after permanent backend close; sink
    /// replacement deliberately keeps and reuses the initializer instead.
    fn retire_render_thread_init(&self) {
        if let Some(init) = self.legacy().render_thread_init.lock().unwrap().take() {
            for message in init.ctrl_msg_recv.try_iter() {
                drop(message);
            }
        }
    }

    /// Creates a [`MediaStreamAudioSourceNode`](node::MediaStreamAudioSourceNode) from a
    /// [`MediaStream`]
    #[must_use]
    pub fn create_media_stream_source(
        &self,
        media: &MediaStream,
    ) -> node::MediaStreamAudioSourceNode {
        self.reject_hosted_media_node();
        let opts = node::MediaStreamAudioSourceOptions {
            media_stream: media,
        };
        node::MediaStreamAudioSourceNode::new(self, opts)
    }

    /// Creates a [`MediaStreamAudioDestinationNode`](node::MediaStreamAudioDestinationNode)
    #[must_use]
    pub fn create_media_stream_destination(&self) -> node::MediaStreamAudioDestinationNode {
        self.reject_hosted_media_node();
        let opts = AudioNodeOptions::default();
        node::MediaStreamAudioDestinationNode::new(self, opts)
    }

    /// Creates a [`MediaStreamTrackAudioSourceNode`](node::MediaStreamTrackAudioSourceNode) from a
    /// [`MediaStreamTrack`]
    #[must_use]
    pub fn create_media_stream_track_source(
        &self,
        media: &MediaStreamTrack,
    ) -> node::MediaStreamTrackAudioSourceNode {
        self.reject_hosted_media_node();
        let opts = node::MediaStreamTrackAudioSourceOptions {
            media_stream_track: media,
        };
        node::MediaStreamTrackAudioSourceNode::new(self, opts)
    }

    /// Creates a [`MediaElementAudioSourceNode`](node::MediaElementAudioSourceNode) from a
    /// [`MediaElement`]
    #[must_use]
    pub fn create_media_element_source(
        &self,
        media_element: &mut MediaElement,
    ) -> node::MediaElementAudioSourceNode {
        self.reject_hosted_media_node();
        let opts = node::MediaElementAudioSourceOptions { media_element };
        node::MediaElementAudioSourceNode::new(self, opts)
    }

    fn reject_hosted_media_node(&self) {
        if self.is_hosted() {
            panic!("NotSupportedError: hosted media-source nodes are not available");
        }
    }

    /// Requests a cancellation-independent hosted suspend without converting failures to panics.
    ///
    /// Admission is immediate and bounded; contention or capacity is returned to the caller. Once
    /// accepted, dropping the receipt does not cancel native work and there is no deadline.
    pub fn request_suspend(
        &self,
    ) -> Result<AudioContextStateChangeReceipt, AudioContextLifecycleError> {
        match &self.mode {
            AudioContextMode::Hosted(hosted) => hosted.request_suspend(),
            AudioContextMode::Legacy(_) => Err(AudioContextLifecycleError::LegacyContext),
        }
    }

    /// Requests a cancellation-independent hosted resume without converting failures to panics.
    ///
    /// Admission is immediate and bounded; contention or capacity is returned to the caller. Once
    /// accepted, dropping the receipt does not cancel native work and there is no deadline.
    pub fn request_resume(
        &self,
    ) -> Result<AudioContextStateChangeReceipt, AudioContextLifecycleError> {
        match &self.mode {
            AudioContextMode::Hosted(hosted) => hosted.request_resume(),
            AudioContextMode::Legacy(_) => Err(AudioContextLifecycleError::LegacyContext),
        }
    }

    /// Latches idempotent hosted shutdown and returns its shared observation receipt.
    ///
    /// Initiation may briefly block while closing/joining the internal render-capacity producer
    /// and serializing the lifecycle latch. It never waits for endpoint, graph, or event-thread
    /// retirement, so it is safe to initiate from an event callback; move the returned receipt to
    /// another thread before polling or waiting for confirmation.
    pub fn request_close(&self) -> Result<AudioContextShutdownReceipt, AudioContextLifecycleError> {
        match &self.mode {
            AudioContextMode::Hosted(hosted) => {
                self.render_capacity.close();
                hosted.request_close()
            }
            AudioContextMode::Legacy(_) => Err(AudioContextLifecycleError::LegacyContext),
        }
    }

    /// Observes eventual hosted shutdown without initiating it.
    ///
    /// Dropping this receipt never cancels lifecycle work. Legacy contexts do not have an exact
    /// shutdown controller and return [`AudioContextLifecycleError::LegacyContext`].
    pub fn shutdown_receipt(
        &self,
    ) -> Result<AudioContextShutdownReceipt, AudioContextLifecycleError> {
        match &self.mode {
            AudioContextMode::Hosted(hosted) => Ok(hosted.shutdown_receipt()),
            AudioContextMode::Legacy(_) => Err(AudioContextLifecycleError::LegacyContext),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "diagnostics")]
    use crate::context::DESTINATION_NODE_ID;
    use crate::context::{
        AudioContextBuildErrorKind, AudioContextShutdownIssueKind, AudioContextShutdownMode,
        AudioContextShutdownOutcome, AudioContextStateChangeOutcome,
        AudioControlBatchReservationProvider, AudioExplicitConnectionReservation,
        AudioExplicitConnectionReservationProvider, AudioGraphConnectionReservation, AudioNodeId,
    };
    use crate::message::ControlBatchSender;
    use crate::node::{
        AudioNode, AudioNodeDisconnectSelector, AudioScheduledSourceNode,
        AudioScheduledSourceNodeExt,
    };
    use crate::node::{ChannelCountMode, ChannelInterpretation};
    use crate::output::{
        audio_render_thread_pair, AudioOutputConfig, AudioOutputContextId,
        AudioOutputEndpointShutdown, AudioOutputError, AudioOutputErrorKind, AudioOutputEventSink,
        AudioOutputFactory, AudioOutputRequest, AudioOutputStartFailure, AudioRenderCallback,
        AudioRenderFormat, AudioRenderOwner, AudioRenderStatus, EndpointShutdownConfirmed,
        PreparedAudioOutput, RunningAudioOutput,
    };
    use crate::render::RenderThread;
    use futures::executor;
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
    use std::thread::{self, JoinHandle, ThreadId};
    use std::time::{Duration, Instant};

    fn marker(value: u16, log: &Arc<Mutex<Vec<u16>>>) -> ControlMessage {
        ControlMessage::TestMarker {
            value,
            log: Arc::clone(log),
        }
    }

    fn panic_message(result: std::thread::Result<()>) -> String {
        let payload = result.expect_err("operation unexpectedly succeeded");
        payload
            .downcast_ref::<&'static str>()
            .map(|message| (*message).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_owned())
    }

    const INJECTED_TEST_RATE: f32 = 48_000.;
    const INJECTED_TEST_CHANNELS: usize = 2;
    const INJECTED_TEST_FRAMES: usize = 128;
    const INJECTED_TEST_TIMEOUT: Duration = Duration::from_secs(2);
    static NEXT_INJECTED_TEST_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

    #[derive(Default)]
    struct InjectedEndpointProbe {
        render_count: AtomicU64,
        saw_nonzero: AtomicBool,
        callback_destroyed: AtomicBool,
        shutdown_joined: AtomicBool,
        abort_completed: AtomicBool,
        suspend_calls: AtomicU64,
        resume_calls: AtomicU64,
        config_calls: AtomicU64,
        abort_calls: AtomicU64,
        abort_thread: Mutex<Option<ThreadId>>,
        abort_future_dropped: AtomicBool,
        context_id: Mutex<Option<AudioOutputContextId>>,
        format: Mutex<Option<AudioRenderFormat>>,
        callback_thread: Mutex<Option<ThreadId>>,
    }

    impl InjectedEndpointProbe {
        fn render_count(&self) -> u64 {
            self.render_count.load(AtomicOrdering::Acquire)
        }
    }

    enum InjectedEndpointCommand {
        Resume(crossbeam_channel::Sender<()>),
        Suspend(crossbeam_channel::Sender<()>),
        Shutdown,
    }

    struct TrackedRenderCallback {
        callback: Option<AudioRenderCallback>,
        probe: Arc<InjectedEndpointProbe>,
    }

    impl TrackedRenderCallback {
        fn callback(&mut self) -> &mut AudioRenderCallback {
            self.callback
                .as_mut()
                .expect("tracked callback is present until endpoint retirement")
        }
    }

    impl Drop for TrackedRenderCallback {
        fn drop(&mut self) {
            drop(self.callback.take());
            self.probe
                .callback_destroyed
                .store(true, AtomicOrdering::Release);
        }
    }

    fn run_injected_endpoint(
        mut tracked: TrackedRenderCallback,
        receiver: crossbeam_channel::Receiver<InjectedEndpointCommand>,
        mut output: Vec<f32>,
    ) {
        let format = tracked.callback().format();
        let interval = Duration::from_secs_f64(
            format.max_frames_per_callback() as f64 / f64::from(format.sample_rate()),
        );
        let probe = Arc::clone(&tracked.probe);
        *probe.callback_thread.lock().unwrap() = Some(thread::current().id());
        let mut running = true;

        loop {
            match receiver.recv_timeout(interval) {
                Ok(InjectedEndpointCommand::Resume(ack)) => {
                    running = true;
                    let _ = ack.send(());
                }
                Ok(InjectedEndpointCommand::Suspend(ack)) => {
                    running = false;
                    let _ = ack.send(());
                }
                Ok(InjectedEndpointCommand::Shutdown) => {
                    drop(tracked);
                    return;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) if running => {
                    let status = tracked.callback().render_interleaved_f32(&mut output);
                    if status == AudioRenderStatus::Stop {
                        return;
                    }
                    probe.render_count.fetch_add(1, AtomicOrdering::Release);
                    if output.iter().any(|sample| sample.abs() > f32::EPSILON) {
                        probe.saw_nonzero.store(true, AtomicOrdering::Release);
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
            }
        }
    }

    struct InjectedTestFactory {
        probe: Arc<InjectedEndpointProbe>,
        fail_start: bool,
        fail_start_cleanup: bool,
        configured_sample_rate: Option<f32>,
        fail_prepare: bool,
        panic_prepare: bool,
        panic_config: bool,
        panic_start: bool,
        output_latency: f64,
        abort_behavior: u8,
        abort_release: Option<Arc<Mutex<Option<futures_channel::oneshot::Receiver<()>>>>>,
        suspend_release: Option<crossbeam_channel::Receiver<()>>,
    }

    impl InjectedTestFactory {
        fn new(fail_start: bool) -> Self {
            Self {
                probe: Arc::new(InjectedEndpointProbe::default()),
                fail_start,
                fail_start_cleanup: false,
                configured_sample_rate: None,
                fail_prepare: false,
                panic_prepare: false,
                panic_config: false,
                panic_start: false,
                output_latency: 0.,
                abort_behavior: 0,
                abort_release: None,
                suspend_release: None,
            }
        }

        fn with_failed_start_cleanup() -> Self {
            Self {
                fail_start_cleanup: true,
                ..Self::new(true)
            }
        }

        fn with_configured_sample_rate(sample_rate: f32) -> Self {
            Self {
                configured_sample_rate: Some(sample_rate),
                ..Self::new(false)
            }
        }
    }

    impl AudioOutputFactory for InjectedTestFactory {
        fn prepare(
            &self,
            request: &AudioOutputRequest,
        ) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
            *self.probe.context_id.lock().unwrap() = Some(request.context_id());
            if self.panic_prepare {
                panic!("injected test factory prepare panic");
            }
            if self.fail_prepare {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::DeviceUnavailable,
                    "injected test factory rejected prepare",
                ));
            }
            let format = AudioRenderFormat::new(
                self.configured_sample_rate.unwrap_or_else(|| {
                    request
                        .requested_sample_rate()
                        .unwrap_or(INJECTED_TEST_RATE)
                }),
                request.number_of_channels(),
                INJECTED_TEST_FRAMES,
            )?;
            let config = AudioOutputConfig::new(format, request.sink_id(), self.output_latency)?;
            Ok(Box::new(InjectedPreparedOutput {
                config,
                probe: Arc::clone(&self.probe),
                fail_start: self.fail_start,
                fail_start_cleanup: self.fail_start_cleanup,
                panic_config: self.panic_config,
                panic_start: self.panic_start,
                abort_behavior: self.abort_behavior,
                abort_release: self.abort_release.clone(),
                suspend_release: self.suspend_release.clone(),
            }))
        }
    }

    struct InjectedPreparedOutput {
        config: AudioOutputConfig,
        probe: Arc<InjectedEndpointProbe>,
        fail_start: bool,
        fail_start_cleanup: bool,
        panic_config: bool,
        panic_start: bool,
        abort_behavior: u8,
        abort_release: Option<Arc<Mutex<Option<futures_channel::oneshot::Receiver<()>>>>>,
        suspend_release: Option<crossbeam_channel::Receiver<()>>,
    }

    struct HostilePreparedAbortFuture {
        behavior: u8,
        probe: Arc<InjectedEndpointProbe>,
    }

    impl std::future::Future for HostilePreparedAbortFuture {
        type Output = Result<(), AudioOutputError>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            match self.behavior {
                1 => std::task::Poll::Ready(Err(AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "forced prepared abort error",
                ))),
                3 => panic!("forced prepared abort future panic"),
                _ => unreachable!("hostile prepared abort future behavior"),
            }
        }
    }

    impl Drop for HostilePreparedAbortFuture {
        fn drop(&mut self) {
            self.probe
                .abort_future_dropped
                .store(true, AtomicOrdering::Release);
        }
    }

    impl PreparedAudioOutput for InjectedPreparedOutput {
        fn config(&self) -> &AudioOutputConfig {
            self.probe.config_calls.fetch_add(1, AtomicOrdering::AcqRel);
            if self.panic_config {
                panic!("injected test prepared config panic");
            }
            &self.config
        }

        fn start(
            self: Box<Self>,
            callback: AudioRenderCallback,
            _events: AudioOutputEventSink,
        ) -> Result<Box<dyn RunningAudioOutput>, AudioOutputStartFailure> {
            let format = callback.format();
            *self.probe.format.lock().unwrap() = Some(format);
            let tracked = TrackedRenderCallback {
                callback: Some(callback),
                probe: Arc::clone(&self.probe),
            };
            if self.panic_start {
                panic!("injected test prepared start panic");
            }
            if self.fail_start {
                drop(tracked);
                let shutdown = if self.fail_start_cleanup {
                    Err(AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected test endpoint failed partial-start cleanup",
                    ))
                } else {
                    Ok(())
                };
                return Err(AudioOutputStartFailure::new(
                    AudioOutputError::new(
                        AudioOutputErrorKind::BackendSpecific,
                        "injected test endpoint rejected startup",
                    ),
                    AudioOutputEndpointShutdown::ready(shutdown),
                ));
            }

            let (sender, receiver) = crossbeam_channel::bounded(8);
            let output = vec![0.; format.max_frames_per_callback() * format.number_of_channels()];
            let join = thread::Builder::new()
                .name("injected-test-output".to_owned())
                .spawn(move || run_injected_endpoint(tracked, receiver, output))
                .map_err(|error| {
                    AudioOutputStartFailure::new(
                        AudioOutputError::new(
                            AudioOutputErrorKind::BackendSpecific,
                            format!("failed to spawn injected test endpoint: {error}"),
                        ),
                        AudioOutputEndpointShutdown::ready(Ok(())),
                    )
                })?;

            Ok(Box::new(InjectedRunningOutput {
                sender,
                join: Some(join),
                probe: Arc::clone(&self.probe),
                suspend_release: self.suspend_release.clone(),
            }))
        }

        fn abort(self: Box<Self>) -> AudioOutputEndpointShutdown {
            let probe = Arc::clone(&self.probe);
            probe.abort_calls.fetch_add(1, AtomicOrdering::AcqRel);
            *probe.abort_thread.lock().unwrap() = Some(thread::current().id());
            match self.abort_behavior {
                0 => AudioOutputEndpointShutdown::from_future(async move {
                    probe.abort_completed.store(true, AtomicOrdering::Release);
                    Ok(())
                }),
                1 => AudioOutputEndpointShutdown::from_future(HostilePreparedAbortFuture {
                    behavior: 1,
                    probe,
                }),
                2 => panic!("forced prepared abort method panic"),
                3 => AudioOutputEndpointShutdown::from_future(HostilePreparedAbortFuture {
                    behavior: 3,
                    probe,
                }),
                4 => {
                    let release = self
                        .abort_release
                        .expect("pending abort behavior requires a release channel")
                        .lock()
                        .unwrap()
                        .take()
                        .expect("pending abort release is single-use");
                    AudioOutputEndpointShutdown::from_future(async move {
                        release.await.map_err(|_| {
                            AudioOutputError::new(
                                AudioOutputErrorKind::Shutdown,
                                "pending abort release sender disconnected",
                            )
                        })?;
                        probe.abort_completed.store(true, AtomicOrdering::Release);
                        Ok(())
                    })
                }
                _ => unreachable!("unknown injected prepared abort behavior"),
            }
        }
    }

    struct InjectedRunningOutput {
        sender: crossbeam_channel::Sender<InjectedEndpointCommand>,
        join: Option<JoinHandle<()>>,
        probe: Arc<InjectedEndpointProbe>,
        suspend_release: Option<crossbeam_channel::Receiver<()>>,
    }

    impl InjectedRunningOutput {
        fn acknowledged_command(
            &self,
            command: impl FnOnce(crossbeam_channel::Sender<()>) -> InjectedEndpointCommand,
        ) -> Result<(), AudioOutputError> {
            let (ack_send, ack_recv) = crossbeam_channel::bounded(1);
            self.sender.send(command(ack_send)).map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "injected test endpoint callback thread disconnected",
                )
            })?;
            ack_recv.recv_timeout(INJECTED_TEST_TIMEOUT).map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "injected test endpoint command was not acknowledged",
                )
            })
        }
    }

    impl RunningAudioOutput for InjectedRunningOutput {
        fn resume(&mut self) -> Result<(), AudioOutputError> {
            self.probe.resume_calls.fetch_add(1, AtomicOrdering::AcqRel);
            self.acknowledged_command(InjectedEndpointCommand::Resume)
        }

        fn suspend(&mut self) -> Result<(), AudioOutputError> {
            self.probe
                .suspend_calls
                .fetch_add(1, AtomicOrdering::AcqRel);
            if let Some(release) = self.suspend_release.take() {
                release.recv().map_err(|_| {
                    AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected test native suspend gate disconnected",
                    )
                })?;
            }
            self.acknowledged_command(InjectedEndpointCommand::Suspend)
        }

        fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
            let _ = self.sender.send(InjectedEndpointCommand::Shutdown);
            let join_result = self.join.take().unwrap().join().map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "injected test endpoint callback thread panicked",
                )
            });
            self.probe
                .shutdown_joined
                .store(true, AtomicOrdering::Release);
            let result = join_result.and_then(|()| {
                let callback_destroyed =
                    self.probe.callback_destroyed.load(AtomicOrdering::Acquire);
                if callback_destroyed {
                    Ok(())
                } else {
                    Err(AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected endpoint joined without confirming callback destruction",
                    ))
                }
            });
            AudioOutputEndpointShutdown::ready(result)
        }
    }

    #[test]
    fn public_hosted_builder_renders_exact_surface_and_closes_idempotently() {
        let factory = Arc::new(InjectedTestFactory::new(false));
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(factory)
            .options(AudioContextOptions {
                sample_rate: Some(INJECTED_TEST_RATE),
                ..AudioContextOptions::default()
            })
            .diagnostic_label("public hosted smoke")
            .build()
            .unwrap();

        assert_eq!(context.state(), AudioContextState::Running);
        assert_eq!(context.sample_rate(), INJECTED_TEST_RATE);
        assert_eq!(context.sink_id(), "");
        assert_eq!(context.output_latency(), 0.);
        assert_eq!(context.set_sink_id_sync(String::new()).unwrap(), ());

        let gain = context.create_gain();
        gain.gain().set_value(0.25);
        let mut oscillator = context.create_oscillator();
        oscillator.connect(&gain);
        gain.connect(&context.destination());
        let ended = Arc::new(AtomicBool::new(false));
        let ended_callback = Arc::clone(&ended);
        oscillator.set_onended(move |_| ended_callback.store(true, AtomicOrdering::Release));
        oscillator.start();
        oscillator.stop_at(context.current_time() + 0.02);

        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while !ended.load(AtomicOrdering::Acquire) {
            assert!(Instant::now() < deadline, "hosted oscillator never ended");
            thread::yield_now();
        }
        assert!(probe.saw_nonzero.load(AtomicOrdering::Acquire));

        probe.saw_nonzero.store(false, AtomicOrdering::Release);
        let mut constant = context.create_constant_source();
        constant.offset().set_value(0.125);
        constant.connect(&context.destination());
        let constant_ended = Arc::new(AtomicBool::new(false));
        let constant_ended_callback = Arc::clone(&constant_ended);
        constant.set_onended(move |_| constant_ended_callback.store(true, AtomicOrdering::Release));
        constant.start();
        constant.stop_at(context.current_time() + 0.02);
        while !constant_ended.load(AtomicOrdering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "hosted ConstantSource never ended"
            );
            thread::yield_now();
        }
        assert!(constant.completion_token().is_complete());
        assert!(probe.saw_nonzero.load(AtomicOrdering::Acquire));

        assert_eq!(
            context.request_suspend().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        assert_eq!(context.state(), AudioContextState::Suspended);
        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        assert_eq!(context.state(), AudioContextState::Running);

        let first = context.request_close().unwrap();
        let second = context.request_close().unwrap();
        let first = first.wait();
        assert_eq!(second.wait(), first);
        assert!(matches!(first, AudioContextShutdownOutcome::Confirmed(_)));
        assert_eq!(context.state(), AudioContextState::Closed);
        assert!(probe.callback_destroyed.load(AtomicOrdering::Acquire));
        assert!(probe.shutdown_joined.load(AtomicOrdering::Acquire));
    }

    #[test]
    fn hosted_node_reservations_survive_wrapper_drop_until_staged_graph_reclaim() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }

        let (suspend_release, suspend_wait) = crossbeam_channel::bounded(1);
        let mut factory = InjectedTestFactory::new(false);
        factory.suspend_release = Some(suspend_wait);
        let factory = Arc::new(factory);
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(factory)
            .initially_suspended(true)
            .build()
            .unwrap();

        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while probe.suspend_calls.load(AtomicOrdering::Acquire) == 0 {
            assert!(
                Instant::now() < deadline,
                "initial native suspension was never entered"
            );
            thread::yield_now();
        }

        let gain_dropped = Arc::new(AtomicBool::new(false));
        let oscillator_dropped = Arc::new(AtomicBool::new(false));
        let oscillator_wave_dropped = Arc::new(AtomicBool::new(false));
        let constant_source_dropped = Arc::new(AtomicBool::new(false));
        let buffer_source_dropped = Arc::new(AtomicBool::new(false));
        let gain_commands_dropped = Arc::new(AtomicBool::new(false));
        let oscillator_commands_dropped = Arc::new(AtomicBool::new(false));
        let constant_source_commands_dropped = Arc::new(AtomicBool::new(false));
        let buffer_source_commands_dropped = Arc::new(AtomicBool::new(false));
        let buffer_payload_command_dropped = Arc::new(AtomicBool::new(false));
        let buffer_start_command_dropped = Arc::new(AtomicBool::new(false));
        let buffer_stop_command_dropped = Arc::new(AtomicBool::new(false));
        let start_command_dropped = Arc::new(AtomicBool::new(false));
        let stop_command_dropped = Arc::new(AtomicBool::new(false));
        let constant_start_command_dropped = Arc::new(AtomicBool::new(false));
        let constant_stop_command_dropped = Arc::new(AtomicBool::new(false));
        let gain = context.create_gain_with_reservations(
            AudioNodeLifetimeReservation::new(DropProbe(Arc::clone(&gain_dropped))),
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&gain_commands_dropped))),
        );
        let periodic_wave = crate::PeriodicWave::new_with_storage_lease(
            &context,
            crate::PeriodicWaveOptions {
                real: Some(vec![0., 0.]),
                imag: Some(vec![0., 1.]),
                disable_normalization: false,
            },
            crate::PeriodicWaveStorageLease::new(DropProbe(Arc::clone(&oscillator_wave_dropped))),
        );
        let mut oscillator = context.create_oscillator_with_options_and_reservations(
            node::OscillatorOptions {
                periodic_wave: Some(periodic_wave),
                ..node::OscillatorOptions::default()
            },
            AudioNodeLifetimeReservation::new(DropProbe(Arc::clone(&oscillator_dropped))),
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&oscillator_commands_dropped))),
        );
        assert_eq!(oscillator.type_(), node::OscillatorType::Custom);
        oscillator.start_at_with_control_reservation(
            0.,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&start_command_dropped))),
        );
        oscillator.stop_at_with_control_reservation(
            0.01,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&stop_command_dropped))),
        );
        let mut constant_source = context.create_constant_source_with_options_and_reservations(
            node::ConstantSourceOptions { offset: 0.375 },
            AudioNodeLifetimeReservation::new(DropProbe(Arc::clone(&constant_source_dropped))),
            AudioControlBatchReservation::new(DropProbe(Arc::clone(
                &constant_source_commands_dropped,
            ))),
        );
        assert_eq!(constant_source.offset().value(), 0.375);
        constant_source.start_at_with_control_reservation(
            0.,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(
                &constant_start_command_dropped,
            ))),
        );
        constant_source.stop_at_with_control_reservation(
            0.01,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(
                &constant_stop_command_dropped,
            ))),
        );
        let mut buffer_source = context.create_buffer_source_with_reservations(
            AudioNodeLifetimeReservation::new(DropProbe(Arc::clone(&buffer_source_dropped))),
            AudioControlBatchReservation::new(DropProbe(Arc::clone(
                &buffer_source_commands_dropped,
            ))),
        );
        buffer_source.set_buffer_with_control_reservation(
            crate::AudioBuffer::from(vec![vec![0.25; 256]], INJECTED_TEST_RATE),
            AudioControlBatchReservation::new(DropProbe(Arc::clone(
                &buffer_payload_command_dropped,
            ))),
        );
        buffer_source.start_at_with_offset_and_duration_with_control_reservation(
            0.,
            0.,
            f64::MAX,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&buffer_start_command_dropped))),
        );
        buffer_source.stop_at_with_control_reservation(
            0.01,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&buffer_stop_command_dropped))),
        );
        drop(gain);
        drop(oscillator);
        drop(constant_source);
        drop(buffer_source);

        // Construction is staged and the renderer has not been allowed to observe either graph
        // insertion or teardown. Wrapper destruction alone must not release host accounting.
        assert!(!gain_dropped.load(AtomicOrdering::Acquire));
        assert!(!oscillator_dropped.load(AtomicOrdering::Acquire));
        assert!(!oscillator_wave_dropped.load(AtomicOrdering::Acquire));
        assert!(!constant_source_dropped.load(AtomicOrdering::Acquire));
        assert!(!buffer_source_dropped.load(AtomicOrdering::Acquire));
        assert!(!gain_commands_dropped.load(AtomicOrdering::Acquire));
        assert!(!oscillator_commands_dropped.load(AtomicOrdering::Acquire));
        assert!(!constant_source_commands_dropped.load(AtomicOrdering::Acquire));
        assert!(!buffer_source_commands_dropped.load(AtomicOrdering::Acquire));
        assert!(!buffer_payload_command_dropped.load(AtomicOrdering::Acquire));
        assert!(!buffer_start_command_dropped.load(AtomicOrdering::Acquire));
        assert!(!buffer_stop_command_dropped.load(AtomicOrdering::Acquire));
        assert!(!start_command_dropped.load(AtomicOrdering::Acquire));
        assert!(!stop_command_dropped.load(AtomicOrdering::Acquire));
        assert!(!constant_start_command_dropped.load(AtomicOrdering::Acquire));
        assert!(!constant_stop_command_dropped.load(AtomicOrdering::Acquire));

        suspend_release.send(()).unwrap();
        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        while !gain_dropped.load(AtomicOrdering::Acquire)
            || !oscillator_dropped.load(AtomicOrdering::Acquire)
            || !oscillator_wave_dropped.load(AtomicOrdering::Acquire)
            || !constant_source_dropped.load(AtomicOrdering::Acquire)
            || !buffer_source_dropped.load(AtomicOrdering::Acquire)
            || !gain_commands_dropped.load(AtomicOrdering::Acquire)
            || !oscillator_commands_dropped.load(AtomicOrdering::Acquire)
            || !constant_source_commands_dropped.load(AtomicOrdering::Acquire)
            || !buffer_source_commands_dropped.load(AtomicOrdering::Acquire)
            || !buffer_payload_command_dropped.load(AtomicOrdering::Acquire)
            || !buffer_start_command_dropped.load(AtomicOrdering::Acquire)
            || !buffer_stop_command_dropped.load(AtomicOrdering::Acquire)
            || !start_command_dropped.load(AtomicOrdering::Acquire)
            || !stop_command_dropped.load(AtomicOrdering::Acquire)
            || !constant_start_command_dropped.load(AtomicOrdering::Acquire)
            || !constant_stop_command_dropped.load(AtomicOrdering::Acquire)
        {
            assert!(
                Instant::now() < deadline,
                "node lifetime reservation was not released after physical reclaim"
            );
            thread::yield_now();
        }

        let whole_graph_dropped = Arc::new(AtomicBool::new(false));
        let surviving_oscillator = context.create_oscillator_with_lifetime_reservation(
            AudioNodeLifetimeReservation::new(DropProbe(Arc::clone(&whole_graph_dropped))),
        );
        assert!(!whole_graph_dropped.load(AtomicOrdering::Acquire));
        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert!(whole_graph_dropped.load(AtomicOrdering::Acquire));
        drop(surviving_oscillator);

        let rejected_dropped = Arc::new(AtomicBool::new(false));
        let rejected_commands_dropped = Arc::new(AtomicBool::new(false));
        let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            context.create_gain_with_reservations(
                AudioNodeLifetimeReservation::new(DropProbe(Arc::clone(&rejected_dropped))),
                AudioControlBatchReservation::new(DropProbe(Arc::clone(
                    &rejected_commands_dropped,
                ))),
            );
        }));
        assert!(rejected.is_err());
        assert!(rejected_dropped.load(AtomicOrdering::Acquire));
        assert!(rejected_commands_dropped.load(AtomicOrdering::Acquire));
    }

    #[test]
    fn hosted_param_and_connection_reservations_follow_exact_nonempty_batches() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }

        fn provider(
            expected: usize,
            called: Arc<AtomicBool>,
            dropped: Arc<AtomicBool>,
        ) -> AudioControlBatchReservationProvider {
            AudioControlBatchReservationProvider::new(move |command_count| {
                assert_eq!(command_count, expected);
                called.store(true, AtomicOrdering::Release);
                Some(AudioControlBatchReservation::new(DropProbe(dropped)))
            })
        }

        fn explicit_provider(
            called: Arc<AtomicBool>,
            command_dropped: Arc<AtomicBool>,
            graph_dropped: Arc<AtomicBool>,
        ) -> AudioExplicitConnectionReservationProvider {
            AudioExplicitConnectionReservationProvider::new(move || {
                called.store(true, AtomicOrdering::Release);
                Some(AudioExplicitConnectionReservation::new(
                    AudioGraphConnectionReservation::new(DropProbe(graph_dropped)),
                    AudioControlBatchReservation::new(DropProbe(command_dropped)),
                ))
            })
        }

        let (suspend_release, suspend_wait) = crossbeam_channel::bounded(1);
        let mut factory = InjectedTestFactory::new(false);
        factory.suspend_release = Some(suspend_wait);
        let factory = Arc::new(factory);
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(factory)
            .initially_suspended(true)
            .build()
            .unwrap();

        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while probe.suspend_calls.load(AtomicOrdering::Acquire) == 0 {
            assert!(
                Instant::now() < deadline,
                "initial suspension never entered"
            );
            thread::yield_now();
        }

        let gain = context.create_gain();
        let oscillator = context.create_oscillator();
        let destination = context.destination();

        let param_dropped = Arc::new(AtomicBool::new(false));
        gain.gain().set_value_with_control_reservation(
            0.5,
            AudioControlBatchReservation::new(DropProbe(Arc::clone(&param_dropped))),
        );

        let first_called = Arc::new(AtomicBool::new(false));
        let first_dropped = Arc::new(AtomicBool::new(false));
        let first_graph_dropped = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_reservations(
            &gain,
            0,
            0,
            explicit_provider(
                Arc::clone(&first_called),
                Arc::clone(&first_dropped),
                Arc::clone(&first_graph_dropped),
            ),
        );
        assert!(first_called.load(AtomicOrdering::Acquire));

        let duplicate_called = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_reservations(
            &gain,
            0,
            0,
            AudioExplicitConnectionReservationProvider::new({
                let duplicate_called = Arc::clone(&duplicate_called);
                move || {
                    duplicate_called.store(true, AtomicOrdering::Release);
                    None
                }
            }),
        );
        assert!(!duplicate_called.load(AtomicOrdering::Acquire));

        let second_called = Arc::new(AtomicBool::new(false));
        let second_dropped = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_control_reservation(
            &destination,
            0,
            0,
            provider(1, Arc::clone(&second_called), Arc::clone(&second_dropped)),
        );
        assert!(second_called.load(AtomicOrdering::Acquire));

        let disconnect_called = Arc::new(AtomicBool::new(false));
        let disconnect_dropped = Arc::new(AtomicBool::new(false));
        oscillator.disconnect_with_control_reservation(
            AudioNodeDisconnectSelector::All,
            provider(
                2,
                Arc::clone(&disconnect_called),
                Arc::clone(&disconnect_dropped),
            ),
        );
        assert!(disconnect_called.load(AtomicOrdering::Acquire));

        let no_match_called = Arc::new(AtomicBool::new(false));
        oscillator.disconnect_with_control_reservation(
            AudioNodeDisconnectSelector::All,
            AudioControlBatchReservationProvider::new({
                let no_match_called = Arc::clone(&no_match_called);
                move |_| {
                    no_match_called.store(true, AtomicOrdering::Release);
                    None
                }
            }),
        );
        assert!(!no_match_called.load(AtomicOrdering::Acquire));

        assert!(!param_dropped.load(AtomicOrdering::Acquire));
        assert!(!first_dropped.load(AtomicOrdering::Acquire));
        assert!(!first_graph_dropped.load(AtomicOrdering::Acquire));
        assert!(!second_dropped.load(AtomicOrdering::Acquire));
        assert!(!disconnect_dropped.load(AtomicOrdering::Acquire));

        suspend_release.send(()).unwrap();
        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        while !param_dropped.load(AtomicOrdering::Acquire)
            || !first_dropped.load(AtomicOrdering::Acquire)
            || !first_graph_dropped.load(AtomicOrdering::Acquire)
            || !second_dropped.load(AtomicOrdering::Acquire)
            || !disconnect_dropped.load(AtomicOrdering::Acquire)
        {
            assert!(
                Instant::now() < deadline,
                "accepted mutation reservation was not reclaimed"
            );
            thread::yield_now();
        }

        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn hosted_connection_reservation_refusal_is_nonterminal_and_preserves_edges() {
        fn accepted_provider(
            expected: usize,
            called: Arc<AtomicBool>,
        ) -> AudioControlBatchReservationProvider {
            AudioControlBatchReservationProvider::new(move |command_count| {
                assert_eq!(command_count, expected);
                called.store(true, AtomicOrdering::Release);
                Some(AudioControlBatchReservation::new(()))
            })
        }

        let context = AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
            .build()
            .unwrap();
        let oscillator = context.create_oscillator();
        let gain = context.create_gain();
        let destination = context.destination();

        let rejected_connect_called = Arc::new(AtomicBool::new(false));
        let rejection = panic_message(std::panic::catch_unwind(AssertUnwindSafe({
            let rejected_connect_called = Arc::clone(&rejected_connect_called);
            || {
                oscillator.connect_from_output_to_input_with_reservations(
                    &gain,
                    0,
                    0,
                    AudioExplicitConnectionReservationProvider::new(move || {
                        rejected_connect_called.store(true, AtomicOrdering::Release);
                        None
                    }),
                );
            }
        })));
        assert!(rejection.contains("QuotaExceededError"));
        assert!(rejected_connect_called.load(AtomicOrdering::Acquire));

        // Refusal happened before commit and did not terminalize the graph: this is a new edge,
        // so the accepted provider must run rather than being discarded as a duplicate.
        let first_connect_called = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_control_reservation(
            &gain,
            0,
            0,
            accepted_provider(1, Arc::clone(&first_connect_called)),
        );
        assert!(first_connect_called.load(AtomicOrdering::Acquire));

        let second_connect_called = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_control_reservation(
            &destination,
            0,
            0,
            accepted_provider(1, Arc::clone(&second_connect_called)),
        );
        assert!(second_connect_called.load(AtomicOrdering::Acquire));

        let rejected_disconnect_called = Arc::new(AtomicBool::new(false));
        let rejection = panic_message(std::panic::catch_unwind(AssertUnwindSafe({
            let rejected_disconnect_called = Arc::clone(&rejected_disconnect_called);
            || {
                oscillator.disconnect_with_control_reservation(
                    AudioNodeDisconnectSelector::All,
                    AudioControlBatchReservationProvider::new(move |command_count| {
                        assert_eq!(command_count, 2);
                        rejected_disconnect_called.store(true, AtomicOrdering::Release);
                        None
                    }),
                );
            }
        })));
        assert!(rejection.contains("QuotaExceededError"));
        assert!(rejected_disconnect_called.load(AtomicOrdering::Acquire));

        // A refused broad disconnect leaves both represented edges in the mirror. Duplicate
        // connects therefore remain zero-command operations and discard their providers unused.
        let duplicate_gain_called = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_control_reservation(
            &gain,
            0,
            0,
            AudioControlBatchReservationProvider::new({
                let duplicate_gain_called = Arc::clone(&duplicate_gain_called);
                move |_| {
                    duplicate_gain_called.store(true, AtomicOrdering::Release);
                    None
                }
            }),
        );
        let duplicate_destination_called = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_control_reservation(
            &destination,
            0,
            0,
            AudioControlBatchReservationProvider::new({
                let duplicate_destination_called = Arc::clone(&duplicate_destination_called);
                move |_| {
                    duplicate_destination_called.store(true, AtomicOrdering::Release);
                    None
                }
            }),
        );
        assert!(!duplicate_gain_called.load(AtomicOrdering::Acquire));
        assert!(!duplicate_destination_called.load(AtomicOrdering::Acquire));

        let accepted_disconnect_called = Arc::new(AtomicBool::new(false));
        oscillator.disconnect_with_control_reservation(
            AudioNodeDisconnectSelector::All,
            accepted_provider(2, Arc::clone(&accepted_disconnect_called)),
        );
        assert!(accepted_disconnect_called.load(AtomicOrdering::Acquire));

        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn hosted_whole_graph_retirement_releases_residual_explicit_connection_reservation() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }

        let context = AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
            .build()
            .unwrap();
        let oscillator = context.create_oscillator();
        let gain = context.create_gain();
        let graph_dropped = Arc::new(AtomicBool::new(false));
        oscillator.connect_from_output_to_input_with_reservations(
            &gain,
            0,
            0,
            AudioExplicitConnectionReservationProvider::new({
                let graph_dropped = Arc::clone(&graph_dropped);
                move || {
                    Some(AudioExplicitConnectionReservation::new(
                        AudioGraphConnectionReservation::new(DropProbe(graph_dropped)),
                        AudioControlBatchReservation::new(()),
                    ))
                }
            }),
        );
        assert!(!graph_dropped.load(AtomicOrdering::Acquire));

        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert!(graph_dropped.load(AtomicOrdering::Acquire));
    }

    #[test]
    fn hosted_param_reservation_is_released_when_closed_rejects_mutation() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }

        let context = AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
            .build()
            .unwrap();
        let gain = context.create_gain();
        let initial_value = gain.gain().value();
        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));

        let dropped = Arc::new(AtomicBool::new(false));
        let rejection = panic_message(std::panic::catch_unwind(AssertUnwindSafe(|| {
            gain.gain().set_value_with_control_reservation(
                0.25,
                AudioControlBatchReservation::new(DropProbe(Arc::clone(&dropped))),
            );
        })));
        assert!(rejection.contains("InvalidStateError"));
        assert!(dropped.load(AtomicOrdering::Acquire));
        assert_eq!(gain.gain().value(), initial_value);
    }

    #[test]
    fn hosted_connection_provider_panic_fails_closed_before_later_reservation() {
        struct DropProbe(Arc<AtomicBool>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }

        let context = AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
            .build()
            .unwrap();
        let oscillator = context.create_oscillator();
        let gain = context.create_gain();
        let panic_probe_dropped = Arc::new(AtomicBool::new(false));

        let failure = std::panic::catch_unwind(AssertUnwindSafe({
            let panic_probe_dropped = Arc::clone(&panic_probe_dropped);
            || {
                oscillator.connect_from_output_to_input_with_control_reservation(
                    &gain,
                    0,
                    0,
                    AudioControlBatchReservationProvider::new(move |command_count| {
                        let _probe = DropProbe(panic_probe_dropped);
                        assert_eq!(command_count, 1);
                        panic!("forced host reservation provider panic");
                    }),
                );
            }
        }));
        assert!(failure.is_err());
        assert!(panic_probe_dropped.load(AtomicOrdering::Acquire));

        let later_provider_called = Arc::new(AtomicBool::new(false));
        let later = std::panic::catch_unwind(AssertUnwindSafe({
            let later_provider_called = Arc::clone(&later_provider_called);
            || {
                oscillator.connect_from_output_to_input_with_control_reservation(
                    &gain,
                    0,
                    0,
                    AudioControlBatchReservationProvider::new(move |_| {
                        later_provider_called.store(true, AtomicOrdering::Release);
                        Some(AudioControlBatchReservation::new(()))
                    }),
                );
            }
        }));
        assert!(later.is_err());
        assert!(!later_provider_called.load(AtomicOrdering::Acquire));

        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn public_hosted_builder_preserves_initial_suspension_and_staged_fifo() {
        let (suspend_release, suspend_wait) = crossbeam_channel::bounded(1);
        let mut factory = InjectedTestFactory::new(false);
        factory.suspend_release = Some(suspend_wait);
        let factory = Arc::new(factory);
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(factory)
            .initially_suspended(true)
            .build()
            .unwrap();
        assert_eq!(context.state(), AudioContextState::Suspended);
        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while probe.suspend_calls.load(AtomicOrdering::Acquire) == 0 {
            assert!(
                Instant::now() < deadline,
                "initial native suspension was never reconciled"
            );
            thread::yield_now();
        }
        while probe.render_count() < 2 {
            assert!(
                Instant::now() < deadline,
                "test endpoint stopped invoking callbacks during native suspend"
            );
            thread::yield_now();
        }
        assert_eq!(context.current_time(), 0.);
        assert!(!probe.saw_nonzero.load(AtomicOrdering::Acquire));

        let source = context.create_gain();
        let destination = context.create_gain();
        source.connect(&destination);
        source.disconnect();
        source.connect(&destination);
        let resume = context.request_resume().unwrap();
        let close = context.request_close().unwrap();
        suspend_release.send(()).unwrap();
        assert!(matches!(
            resume.wait(),
            AudioContextStateChangeOutcome::SupersededByShutdown
                | AudioContextStateChangeOutcome::Closed
        ));
        assert!(matches!(
            close.wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(context.state(), AudioContextState::Closed);
    }

    #[test]
    fn public_hosted_staged_connections_and_source_commands_flush_on_resume() {
        let factory = Arc::new(InjectedTestFactory::new(false));
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(factory)
            .initially_suspended(true)
            .build()
            .unwrap();
        let gain = context.create_gain();
        let mut oscillator = context.create_oscillator();
        oscillator.connect(&gain);
        oscillator.disconnect();
        oscillator.connect(&gain);
        gain.connect(&context.destination());
        let ended = Arc::new(AtomicBool::new(false));
        let ended_callback = Arc::clone(&ended);
        oscillator.set_onended(move |_| ended_callback.store(true, AtomicOrdering::Release));
        oscillator.start();
        oscillator.stop_at(0.02);

        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while !ended.load(AtomicOrdering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "staged source never completed after Resume"
            );
            thread::yield_now();
        }
        assert!(probe.saw_nonzero.load(AtomicOrdering::Acquire));
        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn hosted_event_thread_rejects_blocking_close_but_can_handoff_receipt() {
        let factory = Arc::new(InjectedTestFactory::new(false));
        let context = Arc::new(AudioContext::builder(factory).build().unwrap());
        let (result_send, result_recv) = crossbeam_channel::bounded(1);
        let callback_context = Arc::clone(&context);
        context.set_onstatechange(move |_| {
            let panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
                executor::block_on(callback_context.close());
            }))
            .is_err();
            result_send.send(panicked).unwrap();
        });
        assert_eq!(
            context.request_suspend().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        assert!(result_recv.recv_timeout(INJECTED_TEST_TIMEOUT).unwrap());
        context.clear_onstatechange();
        // The rejected close did not close render capacity or latch lifecycle shutdown.
        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );

        let (receipt_send, receipt_recv) = crossbeam_channel::bounded(1);
        let callback_context = Arc::clone(&context);
        context.set_onstatechange(move |_| {
            let receipt = callback_context.request_close().unwrap();
            let panicked =
                std::panic::catch_unwind(AssertUnwindSafe(|| receipt.clone().wait())).is_err();
            receipt_send.send((receipt, panicked)).unwrap();
        });
        let state_receipt = context.request_suspend().unwrap();
        let (receipt, panicked) = receipt_recv.recv_timeout(INJECTED_TEST_TIMEOUT).unwrap();
        assert!(panicked);
        let _ = state_receipt.wait();
        assert!(matches!(
            receipt.wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn hosted_live_handler_panic_is_deferred_to_confirmed_retirement() {
        let factory = Arc::new(InjectedTestFactory::new(false));
        let context = AudioContext::builder(factory).build().unwrap();
        context.set_onstatechange(|_| panic!("hostile hosted state handler"));
        assert_eq!(
            context.request_suspend().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        let AudioContextShutdownOutcome::Confirmed(report) =
            context.request_close().unwrap().wait()
        else {
            panic!("handler panic does not prevent physical shutdown proof");
        };
        assert_eq!(
            report.event_issue().unwrap().kind(),
            AudioContextShutdownIssueKind::EventDelivery
        );
    }

    #[test]
    fn hosted_builder_classifies_factory_config_and_start_failures_with_cleanup() {
        let mut prepare_error = InjectedTestFactory::new(false);
        prepare_error.fail_prepare = true;
        let error = AudioContext::builder(Arc::new(prepare_error))
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), AudioContextBuildErrorKind::OutputRejected);
        assert!(error.cleanup_receipt().is_none());

        let mut prepare_panic = InjectedTestFactory::new(false);
        prepare_panic.panic_prepare = true;
        let error = AudioContext::builder(Arc::new(prepare_panic))
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), AudioContextBuildErrorKind::OutputPanicked);
        assert!(error.cleanup_receipt().is_none());

        let mut config_panic = InjectedTestFactory::new(false);
        config_panic.panic_config = true;
        let probe = Arc::clone(&config_panic.probe);
        let error = AudioContext::builder(Arc::new(config_panic))
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), AudioContextBuildErrorKind::OutputPanicked);
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(probe.config_calls.load(AtomicOrdering::Acquire), 1);
        assert!(probe.abort_completed.load(AtomicOrdering::Acquire));

        let mismatch = InjectedTestFactory::with_configured_sample_rate(44_100.);
        let probe = Arc::clone(&mismatch.probe);
        let error = AudioContext::builder(Arc::new(mismatch))
            .options(AudioContextOptions {
                sample_rate: Some(INJECTED_TEST_RATE),
                ..AudioContextOptions::default()
            })
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::InvalidConfiguration
        );
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(probe.config_calls.load(AtomicOrdering::Acquire), 1);

        let start_error = InjectedTestFactory::new(true);
        let probe = Arc::clone(&start_error.probe);
        let error = AudioContext::builder(Arc::new(start_error))
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), AudioContextBuildErrorKind::StartFailed);
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert!(probe.callback_destroyed.load(AtomicOrdering::Acquire));

        let mut start_panic = InjectedTestFactory::new(false);
        start_panic.panic_start = true;
        let probe = Arc::clone(&start_panic.probe);
        let error = AudioContext::builder(Arc::new(start_panic))
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), AudioContextBuildErrorKind::StartFailed);
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Unconfirmed { .. }
        ));
        assert!(probe.callback_destroyed.load(AtomicOrdering::Acquire));
    }

    #[test]
    fn hosted_prepared_abort_failures_and_stall_are_owned_off_the_builder_thread() {
        let builder_thread = thread::current().id();
        for (behavior, expected) in [
            (1, AudioContextShutdownIssueKind::OutputFuture),
            (2, AudioContextShutdownIssueKind::OutputMethod),
            (3, AudioContextShutdownIssueKind::OutputFuture),
        ] {
            let mut factory = InjectedTestFactory::with_configured_sample_rate(44_100.);
            factory.abort_behavior = behavior;
            let probe = Arc::clone(&factory.probe);
            let error = AudioContext::builder(Arc::new(factory))
                .options(AudioContextOptions {
                    sample_rate: Some(INJECTED_TEST_RATE),
                    ..AudioContextOptions::default()
                })
                .build()
                .unwrap_err();
            let AudioContextShutdownOutcome::Unconfirmed { failure, .. } =
                error.cleanup_receipt().unwrap().wait()
            else {
                panic!("hostile prepared abort must not manufacture confirmation");
            };
            assert_eq!(failure.kind(), expected);
            assert_ne!(*probe.abort_thread.lock().unwrap(), Some(builder_thread));
            if matches!(behavior, 1 | 3) {
                assert!(
                    !probe.abort_future_dropped.load(AtomicOrdering::Acquire),
                    "Err/panicking abort future must retain unsafe leases in quarantine"
                );
            }
        }

        let (release_send, release_recv) = futures_channel::oneshot::channel();
        let mut factory = InjectedTestFactory::with_configured_sample_rate(44_100.);
        factory.abort_behavior = 4;
        factory.abort_release = Some(Arc::new(Mutex::new(Some(release_recv))));
        let probe = Arc::clone(&factory.probe);
        let error = AudioContext::builder(Arc::new(factory))
            .options(AudioContextOptions {
                sample_rate: Some(INJECTED_TEST_RATE),
                ..AudioContextOptions::default()
            })
            .build()
            .unwrap_err();
        let receipt = error.cleanup_receipt().unwrap();
        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while probe.abort_calls.load(AtomicOrdering::Acquire) == 0 {
            assert!(
                Instant::now() < deadline,
                "abort worker never acquired Prepared"
            );
            thread::yield_now();
        }
        assert_ne!(*probe.abort_thread.lock().unwrap(), Some(builder_thread));
        assert!(!probe.abort_completed.load(AtomicOrdering::Acquire));

        let (outcome_send, outcome_recv) = crossbeam_channel::bounded(1);
        thread::spawn(move || outcome_send.send(receipt.wait()).unwrap());
        assert!(outcome_recv.try_recv().is_err());
        release_send.send(()).unwrap();
        assert!(matches!(
            outcome_recv.recv_timeout(INJECTED_TEST_TIMEOUT).unwrap(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert!(probe.abort_completed.load(AtomicOrdering::Acquire));
    }

    #[test]
    fn hosted_bootstrap_transfer_recovers_running_partial_and_uncertain_owners() {
        super::super::hosted::force_next_prepared_start_failure_for_test();
        let prepared = InjectedTestFactory::new(false);
        let prepared_probe = Arc::clone(&prepared.probe);
        let error = AudioContext::builder(Arc::new(prepared))
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::LifecycleUnavailable
        );
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Unconfirmed { .. }
        ));
        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while prepared_probe.abort_calls.load(AtomicOrdering::Acquire) == 0 {
            assert!(
                Instant::now() < deadline,
                "prepared fallback never started abort"
            );
            thread::yield_now();
        }

        super::super::hosted::fail_next_worker_transfer_for_test();
        let running = InjectedTestFactory::new(false);
        let running_probe = Arc::clone(&running.probe);
        let error = AudioContext::builder(Arc::new(running))
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::LifecycleUnavailable
        );
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert!(running_probe
            .callback_destroyed
            .load(AtomicOrdering::Acquire));
        assert!(running_probe.shutdown_joined.load(AtomicOrdering::Acquire));

        super::super::hosted::fail_next_worker_transfer_for_test();
        let partial = InjectedTestFactory::new(true);
        let partial_probe = Arc::clone(&partial.probe);
        let error = AudioContext::builder(Arc::new(partial))
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::LifecycleUnavailable
        );
        assert_eq!(
            error.output_error().unwrap().kind(),
            AudioOutputErrorKind::BackendSpecific
        );
        assert!(error
            .to_string()
            .contains("injected test endpoint rejected startup"));
        assert!(std::error::Error::source(&error).is_some());
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert!(partial_probe
            .callback_destroyed
            .load(AtomicOrdering::Acquire));

        super::super::hosted::fail_next_worker_transfer_for_test();
        let mut uncertain = InjectedTestFactory::new(false);
        uncertain.panic_start = true;
        let uncertain_probe = Arc::clone(&uncertain.probe);
        let error = AudioContext::builder(Arc::new(uncertain))
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::LifecycleUnavailable
        );
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Unconfirmed { .. }
        ));
        assert!(uncertain_probe
            .callback_destroyed
            .load(AtomicOrdering::Acquire));

        super::super::hosted::fail_next_worker_transfer_for_test();
        super::super::hosted::fail_next_fallback_worker_spawn_for_test();
        let running = InjectedTestFactory::new(false);
        let error = AudioContext::builder(Arc::new(running))
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::LifecycleUnavailable
        );
        assert!(matches!(
            error.cleanup_receipt().unwrap().wait(),
            AudioContextShutdownOutcome::Unconfirmed { .. }
        ));
    }

    #[test]
    fn hosted_output_config_is_observed_once_and_seeds_playback_latency() {
        let mut factory = InjectedTestFactory::new(false);
        factory.output_latency = 0.0125;
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(Arc::new(factory)).build().unwrap();
        assert_eq!(probe.config_calls.load(AtomicOrdering::Acquire), 1);
        assert_eq!(context.output_latency(), 0.0125);
        assert_eq!(context.playback_stats().average_latency(), 0.0125);
        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn invalid_options_and_worker_failure_do_not_consume_hosted_context_identity() {
        let identities = Arc::new(AtomicU64::new(77));
        super::super::hosted::set_context_id_source_for_test(Some(Arc::clone(&identities)));

        let invalid_rate = InjectedTestFactory::new(false);
        let invalid_rate_probe = Arc::clone(&invalid_rate.probe);
        let error = AudioContext::builder(Arc::new(invalid_rate))
            .options(AudioContextOptions {
                sample_rate: Some(1.),
                ..AudioContextOptions::default()
            })
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::InvalidConfiguration
        );
        assert_eq!(identities.load(AtomicOrdering::Acquire), 77);
        assert!(invalid_rate_probe.context_id.lock().unwrap().is_none());

        let invalid_channels = InjectedTestFactory::new(false);
        let invalid_channels_probe = Arc::clone(&invalid_channels.probe);
        let error = AudioContext::builder(Arc::new(invalid_channels))
            .number_of_channels(0)
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::InvalidConfiguration
        );
        assert_eq!(identities.load(AtomicOrdering::Acquire), 77);
        assert!(invalid_channels_probe.context_id.lock().unwrap().is_none());

        let invalid_latency = InjectedTestFactory::new(false);
        let invalid_latency_probe = Arc::clone(&invalid_latency.probe);
        let error = AudioContext::builder(Arc::new(invalid_latency))
            .options(AudioContextOptions {
                latency_hint: AudioContextLatencyCategory::Custom(0.),
                ..AudioContextOptions::default()
            })
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::InvalidConfiguration
        );
        assert_eq!(identities.load(AtomicOrdering::Acquire), 77);
        assert!(invalid_latency_probe.context_id.lock().unwrap().is_none());

        super::super::hosted::fail_next_worker_spawn_for_test();
        let error = AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
            .build()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            AudioContextBuildErrorKind::LifecycleUnavailable
        );
        assert_eq!(identities.load(AtomicOrdering::Acquire), 77);

        super::super::hosted::set_context_id_source_for_test(None);
    }

    #[test]
    fn hosted_unsupported_mutators_reject_before_host_or_callback_retention() {
        struct DropProbe(Arc<AtomicBool>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }

        let context = AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
            .build()
            .unwrap();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = context.create_biquad_filter();
        }))
        .is_err());
        let gain = context.create_gain();
        assert_eq!(gain.registration().id(), AudioNodeId(11));
        assert_eq!(gain.gain().registration().id(), AudioNodeId(12));
        let original_count = gain.channel_count();
        let original_mode = gain.channel_count_mode();
        let original_interpretation = gain.channel_interpretation();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            gain.set_channel_count(1);
        }))
        .is_err());
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            gain.set_channel_count_mode(ChannelCountMode::Explicit);
        }))
        .is_err());
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            gain.set_channel_interpretation(ChannelInterpretation::Discrete);
        }))
        .is_err());
        assert_eq!(gain.channel_count(), original_count);
        assert_eq!(gain.channel_count_mode(), original_mode);
        assert_eq!(gain.channel_interpretation(), original_interpretation);

        let processor_callback_dropped = Arc::new(AtomicBool::new(false));
        let probe = DropProbe(Arc::clone(&processor_callback_dropped));
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            gain.set_onprocessorerror(Box::new(move |_| {
                let _ = &probe;
            }));
        }))
        .is_err());
        assert!(processor_callback_dropped.load(AtomicOrdering::Acquire));

        assert!(matches!(
            context.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        let sink_callback_dropped = Arc::new(AtomicBool::new(false));
        let probe = DropProbe(Arc::clone(&sink_callback_dropped));
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            context.set_onsinkchange(move |_| {
                let _ = &probe;
            });
        }))
        .is_err());
        assert!(sink_callback_dropped.load(AtomicOrdering::Acquire));
        context.clear_onsinkchange();
    }

    #[test]
    fn hosted_rejections_do_not_change_legacy_channel_or_event_behavior() {
        let context = AudioContext::new(AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        });
        let gain = context.create_gain();
        gain.set_channel_count(1);
        gain.set_channel_count_mode(ChannelCountMode::Explicit);
        gain.set_channel_interpretation(ChannelInterpretation::Discrete);
        assert_eq!(gain.channel_count(), 1);
        assert_eq!(gain.channel_count_mode(), ChannelCountMode::Explicit);
        assert_eq!(
            gain.channel_interpretation(),
            ChannelInterpretation::Discrete
        );

        let dropped = Arc::new(AtomicBool::new(false));
        struct LegacyDropProbe(Arc<AtomicBool>);
        impl Drop for LegacyDropProbe {
            fn drop(&mut self) {
                self.0.store(true, AtomicOrdering::Release);
            }
        }
        let probe = LegacyDropProbe(Arc::clone(&dropped));
        context.set_onsinkchange(move |_| {
            let _ = &probe;
        });
        context.clear_onsinkchange();
        assert!(dropped.load(AtomicOrdering::Acquire));
        context.close_sync();
    }

    #[test]
    fn public_hosted_contexts_isolate_factory_identity_graph_state_and_close() {
        let factory_a = Arc::new(InjectedTestFactory::new(false));
        let factory_b = Arc::new(InjectedTestFactory::new(false));
        let probe_a = Arc::clone(&factory_a.probe);
        let probe_b = Arc::clone(&factory_b.probe);
        let context_a = AudioContext::builder(factory_a).build().unwrap();
        let context_b = AudioContext::builder(factory_b).build().unwrap();
        assert_ne!(
            *probe_a.context_id.lock().unwrap(),
            *probe_b.context_id.lock().unwrap()
        );

        let gain_a = context_a.create_gain();
        gain_a.gain().set_value(0.25);
        assert_eq!(context_b.create_gain().gain().value(), 1.);
        assert_eq!(
            context_a.request_suspend().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );
        assert_eq!(context_a.state(), AudioContextState::Suspended);
        assert_eq!(context_b.state(), AudioContextState::Running);

        assert!(matches!(
            context_a.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(context_a.state(), AudioContextState::Closed);
        assert_eq!(context_b.state(), AudioContextState::Running);
        assert!(probe_b.render_count() > 0);
        assert!(matches!(
            context_b.request_close().unwrap().wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn hosted_drop_and_dropped_receipts_do_not_cancel_native_lifecycle() {
        let factory = Arc::new(InjectedTestFactory::new(false));
        let probe = Arc::clone(&factory.probe);
        let context = AudioContext::builder(factory).build().unwrap();

        drop(context.request_suspend().unwrap());
        let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
        while context.state() != AudioContextState::Suspended
            || probe.suspend_calls.load(AtomicOrdering::Acquire) == 0
        {
            assert!(
                Instant::now() < deadline,
                "dropped suspend receipt cancelled work"
            );
            thread::yield_now();
        }
        assert_eq!(
            context.request_resume().unwrap().wait(),
            AudioContextStateChangeOutcome::Applied
        );

        let gain = context.create_gain();
        let observer = context.shutdown_receipt().unwrap();
        drop(context);
        let AudioContextShutdownOutcome::Confirmed(report) = observer.wait() else {
            panic!("hosted Drop must autonomously retire the exact owner set");
        };
        assert_eq!(report.mode(), AudioContextShutdownMode::Silent);
        assert_eq!(gain.context().state(), AudioContextState::Closed);
        assert!(probe.callback_destroyed.load(AtomicOrdering::Acquire));
        assert!(probe.shutdown_joined.load(AtomicOrdering::Acquire));

        let factory = Arc::new(InjectedTestFactory::new(false));
        let context = AudioContext::builder(factory).build().unwrap();
        let observer = context.shutdown_receipt().unwrap();
        drop(context.request_close().unwrap());
        assert!(matches!(
            observer.wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
    }

    #[test]
    fn hosted_state_contention_is_typed_and_close_dominates_admitted_request() {
        let context = Arc::new(
            AudioContext::builder(Arc::new(InjectedTestFactory::new(false)))
                .build()
                .unwrap(),
        );
        let (entered_send, entered_recv) = crossbeam_channel::bounded(1);
        let (release_send, release_recv) = crossbeam_channel::bounded(1);
        context.set_hosted_state_request_hook_for_test(Arc::new(move || {
            entered_send.send(()).unwrap();
            release_recv.recv().unwrap();
        }));

        let request_context = Arc::clone(&context);
        let (state_send, state_recv) = crossbeam_channel::bounded(1);
        thread::spawn(move || state_send.send(request_context.request_suspend()).unwrap());
        entered_recv.recv_timeout(INJECTED_TEST_TIMEOUT).unwrap();
        assert_eq!(
            context.request_resume().unwrap_err(),
            AudioContextLifecycleError::Contended
        );
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| context.resume_sync())).is_err());

        let close_context = Arc::clone(&context);
        let (close_send, close_recv) = crossbeam_channel::bounded(1);
        thread::spawn(move || close_send.send(close_context.request_close()).unwrap());
        assert!(close_recv.try_recv().is_err());
        release_send.send(()).unwrap();

        let state_receipt = state_recv
            .recv_timeout(INJECTED_TEST_TIMEOUT)
            .unwrap()
            .unwrap();
        let close_receipt = close_recv
            .recv_timeout(INJECTED_TEST_TIMEOUT)
            .unwrap()
            .unwrap();
        assert!(matches!(
            state_receipt.wait(),
            AudioContextStateChangeOutcome::Applied
                | AudioContextStateChangeOutcome::SupersededByShutdown
        ));
        assert!(matches!(
            close_receipt.wait(),
            AudioContextShutdownOutcome::Confirmed(_)
        ));
        assert_eq!(context.state(), AudioContextState::Closed);
    }

    struct InjectedContextHarness {
        base: ConcreteBaseAudioContext,
        event_loop: EventLoop,
        output_events: crate::output::AudioOutputEventWatcher,
        owner: Option<AudioRenderOwner>,
        running: Option<Box<dyn RunningAudioOutput>>,
    }

    impl BaseAudioContext for InjectedContextHarness {
        fn base(&self) -> &ConcreteBaseAudioContext {
            &self.base
        }
    }

    impl InjectedContextHarness {
        fn new(factory: &dyn AudioOutputFactory) -> Result<Self, AudioOutputError> {
            let context_id = NEXT_INJECTED_TEST_CONTEXT_ID.fetch_add(1, AtomicOrdering::Relaxed);
            let request = AudioOutputRequest::new(
                AudioOutputContextId::new(context_id).unwrap(),
                "injected-test",
                Some(INJECTED_TEST_RATE),
                INJECTED_TEST_CHANNELS,
                AudioContextLatencyCategory::Interactive,
                AudioContextRenderSizeCategory::Default,
                Some(format!("injected test context {context_id}")),
            )?;
            let prepared = factory.prepare(&request)?;
            let config = prepared.config().clone();
            if let Err(error) = request.validate_config(&config) {
                executor::block_on(prepared.abort())?;
                return Err(error);
            }
            let format = config.format();

            let (control_init, render_init) = io::thread_init();
            let ControlThreadInit {
                state,
                frames_played,
                stats: _,
                ctrl_msg_send,
                control_batch_send,
                control_batch_applied,
                event_send,
                event_recv,
            } = control_init;
            let RenderThreadInit {
                state: render_state,
                startup_pending,
                frames_played: render_frames_played,
                stats: render_stats,
                ctrl_msg_recv,
                control_batch_applied: render_control_batch_applied,
                event_send: render_event_send,
            } = render_init;

            let (node_id_producer, node_id_consumer) = llq::Queue::new().split();
            ctrl_msg_send
                .send(ControlMessage::Startup {
                    graph: Graph::new(node_id_producer),
                })
                .map_err(|_| {
                    AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected test control channel disconnected during startup",
                    )
                })?;

            let mut renderer = RenderThread::new(
                format.sample_rate(),
                format.number_of_channels(),
                ctrl_msg_recv,
                render_state,
                render_frames_played,
                render_stats,
                render_event_send,
                render_control_batch_applied,
            );
            renderer.set_startup_pending(startup_pending);
            let (output_sink, output_events) = AudioOutputEventSink::bounded(8);
            let (owner, callback) = audio_render_thread_pair(format, renderer, output_sink.clone());
            let running = match prepared.start(callback, output_sink) {
                Ok(running) => running,
                Err(failure) => {
                    owner.begin_shutdown();
                    let (error, shutdown) = failure.into_parts();
                    executor::block_on(shutdown)?;
                    let reclaim = owner
                        .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
                        .map_err(|_| {
                            AudioOutputError::new(
                                AudioOutputErrorKind::Shutdown,
                                "injected endpoint retained its callback after failed startup",
                            )
                        })?;
                    reclaim?;
                    return Err(error);
                }
            };

            let event_loop = EventLoop::new(event_recv);
            let base = ConcreteBaseAudioContext::new(
                format.sample_rate(),
                format.number_of_channels(),
                state,
                frames_played,
                ctrl_msg_send,
                control_batch_send,
                control_batch_applied,
                event_send,
                event_loop.clone(),
                false,
                node_id_consumer,
            );

            Ok(Self {
                base,
                event_loop,
                output_events,
                owner: Some(owner),
                running: Some(running),
            })
        }

        fn wait_until(&self, predicate: impl Fn() -> bool) {
            let deadline = Instant::now() + INJECTED_TEST_TIMEOUT;
            while !predicate() {
                self.event_loop.handle_pending_events();
                assert_eq!(self.output_events.death_reason(), None);
                assert!(
                    Instant::now() < deadline,
                    "injected context did not reach the expected state"
                );
                thread::sleep(Duration::from_millis(1));
            }
            self.event_loop.handle_pending_events();
        }

        fn acknowledged_graph_message(
            &self,
            submit: impl FnOnce(OneshotNotify) -> ControlMessage,
            suspended: bool,
        ) -> Result<(), AudioOutputError> {
            let (ack_send, ack_recv) = crossbeam_channel::bounded(1);
            let message = submit(OneshotNotify::Sync(ack_send));
            if suspended {
                self.base.suspend_control_msgs(message);
            } else {
                self.base.resume_control_msgs(message);
            }
            ack_recv.recv_timeout(INJECTED_TEST_TIMEOUT).map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::Shutdown,
                    "injected graph lifecycle command was not acknowledged",
                )
            })
        }

        fn suspend(&mut self) -> Result<(), AudioOutputError> {
            self.acknowledged_graph_message(|notify| ControlMessage::Suspend { notify }, true)?;
            self.running.as_mut().unwrap().suspend()
        }

        fn resume(&mut self) -> Result<(), AudioOutputError> {
            self.running.as_mut().unwrap().resume()?;
            self.acknowledged_graph_message(|notify| ControlMessage::Resume { notify }, false)
        }

        fn close(&mut self) -> Result<(), AudioOutputError> {
            if self.owner.is_none() {
                return Ok(());
            }

            if self.state() == AudioContextState::Running {
                let (ack_send, ack_recv) = crossbeam_channel::bounded(1);
                self.base.send_control_msg(ControlMessage::Close {
                    notify: OneshotNotify::Sync(ack_send),
                });
                ack_recv.recv_timeout(INJECTED_TEST_TIMEOUT).map_err(|_| {
                    AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected graph close was not acknowledged",
                    )
                })?;
            } else {
                self.base.set_state(AudioContextState::Closed);
            }

            let owner = self.owner.take().unwrap();
            owner.begin_shutdown();
            let shutdown = self.running.take().unwrap().shutdown();
            executor::block_on(shutdown)?;
            let reclaim = owner
                .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
                .map_err(|_| {
                    AudioOutputError::new(
                        AudioOutputErrorKind::Shutdown,
                        "injected endpoint retained its callback after shutdown",
                    )
                })?;
            reclaim?;
            self.event_loop.handle_pending_events();
            Ok(())
        }
    }

    #[test]
    fn injected_outputs_render_isolated_oscillators_and_retire_joinably() {
        let factory_a = InjectedTestFactory::new(false);
        let factory_b = InjectedTestFactory::new(false);
        let mut context_a = InjectedContextHarness::new(&factory_a).unwrap();
        let mut context_b = InjectedContextHarness::new(&factory_b).unwrap();

        context_a.wait_until(|| context_a.state() == AudioContextState::Running);
        context_b.wait_until(|| context_b.state() == AudioContextState::Running);
        let expected_format = AudioRenderFormat::new(
            INJECTED_TEST_RATE,
            INJECTED_TEST_CHANNELS,
            INJECTED_TEST_FRAMES,
        )
        .unwrap();
        assert_eq!(context_a.sample_rate(), INJECTED_TEST_RATE);
        assert_eq!(context_b.sample_rate(), INJECTED_TEST_RATE);
        assert_eq!(
            context_a.destination().max_channel_count(),
            INJECTED_TEST_CHANNELS
        );
        assert_eq!(
            context_b.destination().max_channel_count(),
            INJECTED_TEST_CHANNELS
        );
        assert_eq!(
            *factory_a.probe.format.lock().unwrap(),
            Some(expected_format)
        );
        assert_eq!(
            *factory_b.probe.format.lock().unwrap(),
            Some(expected_format)
        );
        context_a.wait_until(|| factory_a.probe.render_count() > 0);
        context_b.wait_until(|| factory_b.probe.render_count() > 0);
        assert!(!factory_a.probe.saw_nonzero.load(AtomicOrdering::Acquire));
        assert!(!factory_b.probe.saw_nonzero.load(AtomicOrdering::Acquire));

        let mut oscillator_a = context_a.create_oscillator();
        oscillator_a.connect(&context_a.destination());
        oscillator_a.start();
        let mut oscillator_b = context_b.create_oscillator();
        oscillator_b.connect(&context_b.destination());
        oscillator_b.start();

        context_a.wait_until(|| factory_a.probe.saw_nonzero.load(AtomicOrdering::Acquire));
        context_b.wait_until(|| factory_b.probe.saw_nonzero.load(AtomicOrdering::Acquire));
        assert_ne!(
            *factory_a.probe.callback_thread.lock().unwrap(),
            *factory_b.probe.callback_thread.lock().unwrap()
        );

        context_a.suspend().unwrap();
        assert_eq!(context_a.state(), AudioContextState::Suspended);
        let suspended_a_count = factory_a.probe.render_count();
        let running_b_count = factory_b.probe.render_count();
        context_b.wait_until(|| factory_b.probe.render_count() > running_b_count);
        assert_eq!(factory_a.probe.render_count(), suspended_a_count);

        context_a.resume().unwrap();
        assert_eq!(context_a.state(), AudioContextState::Running);
        context_a.wait_until(|| factory_a.probe.render_count() > suspended_a_count);

        let b_count_before_a_close = factory_b.probe.render_count();
        context_a.close().unwrap();
        assert_eq!(context_a.state(), AudioContextState::Closed);
        assert!(factory_a
            .probe
            .callback_destroyed
            .load(AtomicOrdering::Acquire));
        assert!(factory_a
            .probe
            .shutdown_joined
            .load(AtomicOrdering::Acquire));
        context_b.wait_until(|| factory_b.probe.render_count() > b_count_before_a_close);
        assert_eq!(context_b.state(), AudioContextState::Running);
        assert!(!factory_b
            .probe
            .shutdown_joined
            .load(AtomicOrdering::Acquire));

        context_b.close().unwrap();
        assert!(factory_b
            .probe
            .callback_destroyed
            .load(AtomicOrdering::Acquire));
        assert!(factory_b
            .probe
            .shutdown_joined
            .load(AtomicOrdering::Acquire));
    }

    #[test]
    fn injected_start_failure_destroys_callback_before_ready_and_reclaims_renderer() {
        let factory = InjectedTestFactory::new(true);
        let error = match InjectedContextHarness::new(&factory) {
            Ok(_) => panic!("injected test endpoint unexpectedly started"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), AudioOutputErrorKind::BackendSpecific);
        assert!(factory
            .probe
            .callback_destroyed
            .load(AtomicOrdering::Acquire));
    }

    #[test]
    fn injected_start_failure_surfaces_endpoint_cleanup_error() {
        // A failed endpoint retirement cannot authorize renderer reclamation. The owner therefore
        // takes its deliberate fail-closed quarantine path in this negative test.
        let factory = InjectedTestFactory::with_failed_start_cleanup();
        let error = match InjectedContextHarness::new(&factory) {
            Ok(_) => panic!("injected test endpoint unexpectedly started"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), AudioOutputErrorKind::Shutdown);
        assert_eq!(
            error.message(),
            "injected test endpoint failed partial-start cleanup"
        );
    }

    #[test]
    fn injected_mismatched_config_is_aborted_before_validation_error() {
        let factory = InjectedTestFactory::with_configured_sample_rate(44_100.);
        let error = match InjectedContextHarness::new(&factory) {
            Ok(_) => panic!("injected test endpoint accepted a mismatched format"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), AudioOutputErrorKind::NotSupported);
        assert!(factory.probe.abort_completed.load(AtomicOrdering::Acquire));
        assert_eq!(factory.probe.render_count(), 0);
    }

    #[test]
    fn suspended_sink_replay_saturates_exact_capacity_then_resumes_fifo() {
        let (control_init, render_init) = io::thread_init();
        let ControlThreadInit {
            state,
            frames_played,
            stats: _,
            ctrl_msg_send,
            control_batch_send,
            control_batch_applied,
            event_send,
            event_recv,
        } = control_init;
        let (node_id_producer, node_id_consumer) = llq::Queue::new().split();
        let base = ConcreteBaseAudioContext::new(
            48_000.,
            2,
            state,
            frames_played,
            ctrl_msg_send.clone(),
            control_batch_send,
            control_batch_applied,
            event_send,
            EventLoop::new(event_recv),
            false,
            node_id_consumer,
        );

        // Discard constructor traffic, then model an already-suspended context with one newer
        // mutation in its staging FIFO. Cached sink-swap records must be inserted ahead of it.
        render_init.ctrl_msg_recv.try_iter().for_each(drop);
        base.suspend_control_msgs(ControlMessage::TestNop);
        assert!(matches!(
            render_init.ctrl_msg_recv.recv().unwrap(),
            ControlMessage::TestNop
        ));
        let log = Arc::new(Mutex::new(Vec::new()));
        base.send_control_msg(marker(257, &log));
        assert!(render_init.ctrl_msg_recv.is_empty());

        // Build 256 cached batch envelopes without placing them in the replacement channel yet.
        let (cache_send, cache_recv) = crossbeam_channel::bounded(256);
        let cache_batches = ControlBatchSender::new(cache_send);
        for value in 1..=256 {
            assert_eq!(
                cache_batches.try_send(vec![marker(value, &log)]),
                Ok(value.into())
            );
        }
        let cached: Vec<_> = cache_recv.try_iter().collect();
        assert_eq!(cached.len(), 256);

        replay_sink_swap_control_messages(
            &base,
            &ctrl_msg_send,
            Graph::new(node_id_producer),
            cached,
            AudioContextState::Suspended,
        )
        .unwrap();

        // The replacement cannot consume while suspended: Startup plus exactly 255 cached
        // envelopes fill all physical slots. Sequence 256 was prepended to staging, without a
        // blocking send, ahead of the newer legacy marker 257.
        assert_eq!(render_init.ctrl_msg_recv.len(), 256);
        let applied = render_init.control_batch_applied.clone();
        let mut renderer = RenderThread::new(
            48_000.,
            2,
            render_init.ctrl_msg_recv.clone(),
            Arc::clone(&render_init.state),
            Arc::clone(&render_init.frames_played),
            render_init.stats.clone(),
            render_init.event_send.clone(),
            applied.clone(),
        );
        renderer.render(&mut [] as &mut [f32]);
        assert_eq!(applied.load(), 255);
        assert_eq!(&*log.lock().unwrap(), &(1..=255).collect::<Vec<_>>());
        assert!(render_init.ctrl_msg_recv.is_empty());

        let (resume_send, resume_recv) = crossbeam_channel::bounded(1);
        base.resume_control_msgs(ControlMessage::Resume {
            notify: OneshotNotify::Sync(resume_send),
        });
        renderer.render(&mut [] as &mut [f32]);
        assert_eq!(resume_recv.try_recv(), Ok(()));
        assert_eq!(applied.load(), 256);
        assert_eq!(&*log.lock().unwrap(), &(1..=257).collect::<Vec<_>>());
    }

    #[test]
    fn disconnected_sink_replay_returns_typed_error() {
        let context = crate::context::OfflineAudioContext::new(1, 128, 48_000.);
        let (sender, receiver) = crossbeam_channel::bounded(0);
        let receiver_thread = std::thread::spawn(move || {
            assert!(matches!(
                receiver.recv().unwrap(),
                ControlMessage::Startup { .. }
            ));
            // Dropping the rendezvous receiver after Startup makes the cached replay fail.
        });
        let (node_id_producer, _node_id_consumer) = llq::Queue::new().split();

        let error = replay_sink_swap_control_messages(
            context.base(),
            &sender,
            Graph::new(node_id_producer),
            vec![ControlMessage::TestNop],
            AudioContextState::Running,
        )
        .unwrap_err();
        receiver_thread.join().unwrap();

        assert_eq!(
            error,
            SinkSwapControlError::Disconnected {
                operation: "replaying cached control messages",
            }
        );
    }

    #[test]
    fn test_suspend_resume_close() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };

        // construct with 'none' sink_id
        let context = AudioContext::new(options);

        // Ensure startup has been processed before testing suspend/resume transitions.
        executor::block_on(context.resume());
        assert_eq!(context.state(), AudioContextState::Running);

        executor::block_on(context.suspend());
        assert_eq!(context.state(), AudioContextState::Suspended);
        let time1 = context.current_time();
        assert!(time1 >= 0.);

        // allow some time to progress
        std::thread::sleep(std::time::Duration::from_millis(1));
        let time2 = context.current_time();
        assert_eq!(time1, time2); // no progression of time

        executor::block_on(context.resume());
        assert_eq!(context.state(), AudioContextState::Running);

        // allow some time to progress
        std::thread::sleep(std::time::Duration::from_millis(1));

        let time3 = context.current_time();
        assert!(time3 > time2); // time is progressing

        executor::block_on(context.close());
        assert_eq!(context.state(), AudioContextState::Closed);
        assert!(context
            .legacy()
            .render_thread_init
            .lock()
            .unwrap()
            .is_none());

        let time4 = context.current_time();

        // allow some time to progress
        std::thread::sleep(std::time::Duration::from_millis(1));

        let time5 = context.current_time();
        assert_eq!(time5, time4); // no progression of time
    }

    #[test]
    fn test_suspend_during_startup() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };

        let context = AudioContext::new(options);

        executor::block_on(context.suspend());
        assert_eq!(context.state(), AudioContextState::Suspended);

        let time1 = context.current_time();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let time2 = context.current_time();
        assert_eq!(time1, time2);
    }

    #[test]
    fn test_suspend_sync_during_startup() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };

        let context = AudioContext::new(options);

        context.suspend_sync();
        assert_eq!(context.state(), AudioContextState::Suspended);

        let time1 = context.current_time();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let time2 = context.current_time();
        assert_eq!(time1, time2);
    }

    fn require_send_sync<T: Send + Sync>(_: T) {}

    #[test]
    fn test_all_futures_thread_safe() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);

        require_send_sync(context.suspend());
        require_send_sync(context.resume());
        require_send_sync(context.close());
    }

    #[test]
    fn test_try_new_invalid_sample_rate() {
        let options = AudioContextOptions {
            sample_rate: Some(0.),
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };

        let result = AudioContext::try_new(options);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid sample rate"));
    }

    #[test]
    #[should_panic]
    fn test_invalid_sink_id() {
        let options = AudioContextOptions {
            sink_id: "invalid".into(),
            ..AudioContextOptions::default()
        };
        let _ = AudioContext::new(options);
    }

    #[test]
    fn test_try_new_invalid_sink_id() {
        let options = AudioContextOptions {
            sink_id: "invalid".into(),
            ..AudioContextOptions::default()
        };

        let error = AudioContext::try_new(options).unwrap_err();
        assert_eq!(
            error.to_string(),
            "NotFoundError - Invalid sinkId: \"invalid\""
        );
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn test_run_diagnostics_returns_structured_output() {
        let options = AudioContextOptions {
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        };
        let context = AudioContext::new(options);
        let (sender, receiver) = std::sync::mpsc::channel();

        context.run_diagnostics(move |diagnostics| {
            sender.send(diagnostics).unwrap();
        });

        let diagnostics = receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();

        assert!(diagnostics.backend.name.contains("NoneBackend"));
        assert_eq!(diagnostics.backend.sink_id, "none");
        assert_eq!(diagnostics.backend.output_latency, Some(0.));
        assert_eq!(diagnostics.render_thread.sample_rate, context.sample_rate());
        assert_eq!(
            diagnostics.render_thread.number_of_channels,
            crate::MAX_CHANNELS
        );
        assert!(diagnostics.graph.active);
        assert_eq!(diagnostics.graph.node_count, diagnostics.graph.nodes.len());
        assert_eq!(diagnostics.graph.edge_count, 0);
        assert!(diagnostics.graph.in_cycle.is_empty());
        assert!(diagnostics.graph.cycle_breakers.is_empty());
        assert!(diagnostics
            .graph
            .nodes
            .iter()
            .any(|node| node.id == DESTINATION_NODE_ID.0
                && node.inputs == node.input_channels.len()
                && node.outputs == node.output_channels.len()));
    }
}
