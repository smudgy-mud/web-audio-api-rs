//! Native, fixed-capacity PCM ingress. This is a host API, not a Web Audio interface.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use rtrb::{Consumer, Producer, RingBuffer};

use super::{AudioNode, ChannelConfig};
use crate::context::{
    AudioContextRegistration, AudioContextState, AudioControlBatchReservation,
    AudioNodeLifetimeReservation, BaseAudioContext, InjectedConnectionEndpointKind,
};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::RENDER_QUANTUM_SIZE;

/// Fixed queue capacity in stereo sample frames (16 KiB of PCM).
pub const PCM_SOURCE_CAPACITY: usize = 2048;

#[derive(Default)]
struct State {
    _queue_lifetime: Option<crate::context::SharedAudioNodeLifetimeReservation>,
    started: AtomicBool,
    stopped: AtomicBool,
    producer_done: AtomicBool,
    ended: AtomicBool,
    rendered: AtomicU64,
    underruns: AtomicU64,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PcmSourceState").finish_non_exhaustive()
    }
}

/// Backpressured writer for one native source. Samples must be at the context's
/// sample rate. Mono sources should duplicate their channel into stereo frames.
/// Drop publishes EOF; already queued frames drain before the source ends.
/// Keep decoding and all I/O on a worker, and retain its join handle in the host.
#[derive(Debug)]
pub struct PcmSourceWriter {
    producer: Producer<[f32; 2]>,
    state: Arc<State>,
}

impl PcmSourceWriter {
    /// Writes as many frames as currently fit without waiting. Zero means full or stopped.
    /// The caller retains unconsumed input. This method never grows the queue.
    pub fn write(&mut self, frames: &[[f32; 2]]) -> usize {
        if self.is_stopped() {
            return 0;
        }
        let count = self.producer.slots().min(frames.len());
        for frame in &frames[..count] {
            // One producer owns these slots. The consumer can only make more available.
            if self.producer.push(*frame).is_err() {
                unreachable!("reserved PCM queue slot")
            }
        }
        count
    }

    /// True after stop or physical processor destruction. The worker should exit and be joined.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.state.stopped.load(Ordering::Acquire)
    }

    /// Current free queue capacity. This is a lower bound while render consumes concurrently.
    #[must_use]
    pub fn available_frames(&mut self) -> usize {
        self.producer.slots()
    }
}

impl Drop for PcmSourceWriter {
    fn drop(&mut self) {
        self.state.producer_done.store(true, Ordering::Release);
    }
}

/// Native stereo source backed by a fixed PCM queue. Supported in legacy and hosted
/// contexts; hosted construction admits exactly one node and one control command.
///
/// This node does not open files, spawn workers, grant script permissions, or expose
/// a general render callback. The host owns source authorization and decoder jobs.
#[derive(Debug)]
pub struct PcmSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    state: Arc<State>,
}

impl AudioNode for PcmSourceNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }
    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }
    fn number_of_inputs(&self) -> usize {
        0
    }
    fn number_of_outputs(&self) -> usize {
        1
    }
}

/// Rejected native source construction.
#[derive(Debug)]
pub struct PcmSourceBuildError(String);
impl std::fmt::Display for PcmSourceBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PcmSourceBuildError {}

impl PcmSourceNode {
    /// Creates a paused source and its unique writer.
    ///
    /// # Errors
    /// Returns an error if the context is closed or hosted graph admission fails.
    pub fn new(
        context: &impl BaseAudioContext,
    ) -> Result<(Self, PcmSourceWriter), PcmSourceBuildError> {
        Self::with_reservations(context, None, None)
    }

    /// Creates a source with host accounting guards for one node and one command.
    /// The node guard must also cover queue storage and stays held until physical
    /// graph reclamation AND all queue/state handles are released. Decoder jobs
    /// require their own host-side reservation.
    ///
    /// # Errors
    /// Returns an error for a closed context, failed admission, or reservations on
    /// a legacy context. Rejected construction drops all transferred guards.
    pub fn with_reservations(
        context: &impl BaseAudioContext,
        lifetime: Option<AudioNodeLifetimeReservation>,
        control: Option<AudioControlBatchReservation>,
    ) -> Result<(Self, PcmSourceWriter), PcmSourceBuildError> {
        if context.state() == AudioContextState::Closed {
            return Err(PcmSourceBuildError("audio context is closed".into()));
        }
        let base = context.base();
        let transaction = if let Some(constructor) = base.injected_node_constructor() {
            Some(
                constructor
                    .try_begin_pcm_source(lifetime, control)
                    .map_err(|error| {
                        PcmSourceBuildError(format!("PCM source admission failed: {error:?}"))
                    })?,
            )
        } else {
            if lifetime.is_some() || control.is_some() {
                return Err(PcmSourceBuildError(
                    "host reservations require a hosted context".into(),
                ));
            }
            None
        };
        // Queue and shared state are allocated only after hosted admission succeeds.
        let (producer, consumer) = RingBuffer::new(PCM_SOURCE_CAPACITY);
        let state = Arc::new(State {
            _queue_lifetime: transaction.as_ref().and_then(|t| t.queue_lifetime()),
            ..State::default()
        });
        let writer = PcmSourceWriter {
            producer,
            state: Arc::clone(&state),
        };
        let renderer = PcmSourceRenderer {
            consumer,
            state: Arc::clone(&state),
        };
        let channel_config = ChannelConfig::default();
        let node = if let Some(transaction) = transaction {
            let (id, lifetime, connection) = transaction
                .commit(renderer, channel_config.inner())
                .map_err(|error| {
                PcmSourceBuildError(format!("PCM source construction failed: {error:?}"))
            })?;
            let registration = AudioContextRegistration::from_injected_with_connection(
                id,
                base.clone(),
                lifetime,
                connection,
                InjectedConnectionEndpointKind::AudioNode,
                0,
                1,
            );
            Self {
                registration,
                channel_config,
                state,
            }
        } else {
            base.register(move |registration| {
                (
                    Self {
                        registration,
                        channel_config,
                        state,
                    },
                    Box::new(renderer),
                )
            })
        };
        Ok((node, writer))
    }

    /// Starts consumption at the next render quantum. Idempotent; a stopped source
    /// cannot restart. This native scalar flag does not schedule a graph mutation.
    pub fn start(&self) {
        self.state.started.store(true, Ordering::Release);
    }

    /// Permanently silences this source at the next render quantum and cancels its writer.
    /// The host must separately join its decoder worker off the render thread.
    pub fn stop(&self) {
        self.state.stopped.store(true, Ordering::Release);
    }

    /// True after render observes stop or EOF with an empty queue.
    #[must_use]
    pub fn ended(&self) -> bool {
        self.state.ended.load(Ordering::Acquire)
    }

    /// Number of actual source frames consumed, excluding starvation silence.
    #[must_use]
    pub fn rendered_frames(&self) -> u64 {
        self.state.rendered.load(Ordering::Acquire)
    }

    /// Number of started quanta starved of input before EOF.
    #[must_use]
    pub fn underrun_quanta(&self) -> u64 {
        self.state.underruns.load(Ordering::Acquire)
    }
}

pub(crate) struct PcmSourceRenderer {
    consumer: Consumer<[f32; 2]>,
    state: Arc<State>,
}

impl AudioProcessor for PcmSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        let output = &mut outputs[0];
        output.make_silent();
        if self.state.stopped.load(Ordering::Acquire) {
            self.state.ended.store(true, Ordering::Release);
            return false;
        }
        if !self.state.started.load(Ordering::Acquire) {
            return true;
        }
        let mut frames = [[0.0; RENDER_QUANTUM_SIZE]; 2];
        let mut count = 0;
        let mut terminal = false;
        let [left, right] = &mut frames;
        for (left, right) in left.iter_mut().zip(right) {
            let frame = match self.consumer.pop() {
                Ok(frame) => Some(frame),
                Err(_) if self.state.producer_done.load(Ordering::Acquire) => {
                    // Acquire EOF before rechecking, so a racing final push is never discarded.
                    let frame = self.consumer.pop().ok();
                    terminal = frame.is_none();
                    frame
                }
                Err(_) => None,
            };
            let Some(frame) = frame else { break };
            *left = frame[0];
            *right = frame[1];
            count += 1;
        }
        if count > 0 {
            output.set_number_of_channels(2);
            for (output, input) in output.channels_mut().iter_mut().zip(&frames) {
                output.copy_from_slice(input);
            }
            self.state.rendered.fetch_add(count, Ordering::Release);
        }
        if terminal {
            self.state.ended.store(true, Ordering::Release);
        } else if count < RENDER_QUANTUM_SIZE as u64 {
            self.state.underruns.fetch_add(1, Ordering::Relaxed);
        }
        // Keep a partial final quantum alive until its samples have traversed the graph.
        count > 0 || !terminal
    }
}

impl Drop for PcmSourceRenderer {
    fn drop(&mut self) {
        self.state.stopped.store(true, Ordering::Release);
    }
}
