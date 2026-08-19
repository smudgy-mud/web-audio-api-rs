use crate::context::{
    AudioContextRegistration, AudioControlBatchReservation, AudioNodeLifetimeReservation,
    AudioParamId, BaseAudioContext, ConcreteBaseAudioContext,
};
use crate::param::{injected_audio_param_raw_parts, AudioParam, AudioParamDescriptor};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};

use super::{AudioNode, AudioNodeOptions, ChannelConfig, ChannelCountMode, ChannelInterpretation};

/// Options for constructing a [`GainNode`]
// dictionary GainOptions : AudioNodeOptions {
//   float gain = 1.0;
// };
#[derive(Clone, Debug)]
pub struct GainOptions {
    pub gain: f32,
    pub audio_node_options: AudioNodeOptions,
}

impl Default for GainOptions {
    fn default() -> Self {
        Self {
            gain: 1.,
            audio_node_options: AudioNodeOptions::default(),
        }
    }
}

/// Applies a single gain (volume) value to its incoming audio signal.
///
/// The value is exposed as an [`AudioParam`] so it can be automated over time.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/GainNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#GainNode>
/// - see also: [`BaseAudioContext::create_gain`]
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::default();
///
/// // Build an oscillator that we want to fade in.
/// let mut osc = context.create_oscillator();
/// osc.frequency().set_value(440.);
///
/// // The gain node sits between the source and the destination.
/// let gain = context.create_gain();
/// gain.gain().set_value(0.);
/// gain.gain()
///     .linear_ramp_to_value_at_time(1., context.current_time() + 1.);
///
/// osc.connect(&gain);
/// gain.connect(&context.destination());
/// osc.start();
/// ```
///
/// # Examples
///
/// - `cargo run --release --example amplitude_modulation`
///
#[derive(Debug)]
pub struct GainNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Infos about audio node channel configuration
    channel_config: ChannelConfig,
    /// Multiplier applied to every sample. Defaults to `1.0` (pass-through).
    gain: AudioParam,
}

impl AudioNode for GainNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

impl GainNode {
    /// Constructs a new `GainNode` from explicit options.
    ///
    /// [`BaseAudioContext::create_gain`] is an alternative that applies the
    /// spec defaults (`gain = 1.0`).
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live
    /// * `options` - initial value of the gain parameter and channel config
    pub fn new<C: BaseAudioContext>(context: &C, options: GainOptions) -> Self {
        if context.base().injected_node_constructor().is_some() {
            return Self::new_injected(context.base(), options);
        }

        // Keep the public legacy path and its four individual graph mutations unchanged.
        context.base().register(move |registration| {
            let param_opts = AudioParamDescriptor {
                name: String::new(),
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: crate::param::AutomationRate::A,
            };
            let (param, proc) = context.create_audio_param(param_opts, &registration);

            param.set_value(options.gain);

            let render = GainRenderer { gain: proc };

            let node = GainNode {
                registration,
                channel_config: options.audio_node_options.into(),
                gain: param,
            };

            (node, Box::new(render))
        })
    }

    fn new_injected(context: &ConcreteBaseAudioContext, options: GainOptions) -> Self {
        Self::new_injected_with_lifetime(context, options, None)
    }

    pub(crate) fn new_injected_with_lifetime(
        context: &ConcreteBaseAudioContext,
        options: GainOptions,
        lifetime: Option<AudioNodeLifetimeReservation>,
    ) -> Self {
        Self::new_injected_with_reservations(context, options, lifetime, None)
    }

    pub(crate) fn new_injected_with_reservations(
        context: &ConcreteBaseAudioContext,
        options: GainOptions,
        lifetime: Option<AudioNodeLifetimeReservation>,
        control: Option<AudioControlBatchReservation>,
    ) -> Self {
        let transaction = context
            .injected_node_constructor()
            .expect("injected Gain selection requires an injected construction base")
            .try_begin_gain_with_reservations(lifetime, control)
            .unwrap_or_else(|error| panic!("injected Gain admission failed: {error:?}"));

        // All potentially panicking validation and payload construction happens after one exact
        // admission. Unwind drops the transaction's slots and IDs before releasing admission.
        let descriptor = AudioParamDescriptor {
            name: String::new(),
            min_value: f32::MIN,
            max_value: f32::MAX,
            default_value: 1.,
            automation_rate: crate::param::AutomationRate::A,
        };
        let (param_raw_parts, param_processor) = injected_audio_param_raw_parts(descriptor);
        let initial_value = param_raw_parts.set_initial_value_for_injected(options.gain);
        let channel_config: ChannelConfig = options.audio_node_options.into();
        let param_channel_config: ChannelConfig = AudioNodeOptions {
            channel_count: 1,
            channel_count_mode: ChannelCountMode::Explicit,
            channel_interpretation: ChannelInterpretation::Discrete,
        }
        .into();
        let gain_id = transaction.gain_id();
        let param_id = transaction.param_id();
        let constructed = transaction
            .commit(crate::context::InjectedGainPayload {
                param_processor,
                gain_processor: Box::new(GainRenderer {
                    gain: AudioParamId::from_node_id(param_id),
                }),
                param_channel_config: param_channel_config.inner(),
                gain_channel_config: channel_config.inner(),
                initial_value,
            })
            .unwrap_or_else(|error| panic!("injected Gain construction failed: {error:?}"));
        debug_assert_eq!(constructed.gain_id, gain_id);
        debug_assert_eq!(constructed.param_id, param_id);
        let _accepted_placement = constructed.outcome;

        // Public handles are created only after the graph accepted the entire four-command batch
        // and both exact lifetime slots were armed by its mandatory finalizer.
        let param_registration = AudioContextRegistration::from_injected_with_connection(
            param_id,
            context.clone(),
            constructed.param_registration,
            constructed.param_connection,
            crate::context::InjectedConnectionEndpointKind::AudioParam,
            1,
            1,
        );
        let registration = AudioContextRegistration::from_injected_with_connection(
            gain_id,
            context.clone(),
            constructed.gain_registration,
            constructed.gain_connection,
            crate::context::InjectedConnectionEndpointKind::AudioNode,
            1,
            1,
        );
        let gain = AudioParam::from_injected_raw_parts(
            param_registration,
            param_raw_parts,
            constructed.param_mutation,
        );

        Self {
            registration,
            channel_config,
            gain,
        }
    }

    /// Returns the gain `AudioParam`.
    ///
    /// The default value is `1.0` (pass-through). Setting `0.0` mutes the
    /// signal; values greater than `1.0` boost it (and may clip downstream
    /// nodes if uncompensated). On legacy contexts, this `a-rate` parameter can
    /// be scheduled with the full automation API such as
    /// [`AudioParam::linear_ramp_to_value_at_time`] for fades. The private exact
    /// injected context currently admits only scalar [`AudioParam::set_value`]
    /// updates; its bounded scheduled-automation transport is deferred.
    #[must_use]
    pub fn gain(&self) -> &AudioParam {
        &self.gain
    }
}

pub(crate) struct GainRenderer {
    pub(crate) gain: AudioParamId,
}

impl AudioProcessor for GainRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        if input.is_silent() {
            output.make_silent();
            return false;
        }

        let gain = params.get(&self.gain);

        // very fast track for mute or pass-through
        if gain.len() == 1 {
            // 1e-6 is -120 dB when close to 0 and ±8.283506e-6 dB when close to 1
            // very probably small enough to not be audible
            let threshold = 1e-6;

            let diff_to_zero = gain[0].abs();
            if diff_to_zero <= threshold {
                output.make_silent();
                return false;
            }

            let diff_to_one = (1. - gain[0]).abs();
            if diff_to_one <= threshold {
                *output = input.clone();
                return false;
            }
        }

        *output = input.clone();

        if gain.len() == 1 {
            let g = gain[0];

            output.channels_mut().iter_mut().for_each(|channel| {
                channel.iter_mut().for_each(|o| *o *= g);
            });
        } else {
            output.channels_mut().iter_mut().for_each(|channel| {
                channel
                    .iter_mut()
                    .zip(gain.iter().cycle())
                    .for_each(|(o, g)| *o *= g);
            });
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::OfflineAudioContext;
    use float_eq::assert_float_eq;

    #[test]
    fn test_audioparam_value_applies_immediately() {
        let context = OfflineAudioContext::new(1, 128, 48000.);
        let options = GainOptions {
            gain: 0.12,
            ..Default::default()
        };
        let src = GainNode::new(&context, options);
        assert_float_eq!(src.gain.value(), 0.12, abs_all <= 0.);
    }
}
