use std::any::Any;

use crate::context::{
    AudioContextRegistration, AudioControlBatchReservation, AudioNodeLifetimeReservation,
    AudioParamId, BaseAudioContext, ConcreteBaseAudioContext, InjectedConstantSourceControl,
    InjectedConstantSourceMutationError, InjectedConstantSourcePayload,
};
use crate::param::{
    injected_audio_param_raw_parts, AudioParam, AudioParamDescriptor, AutomationRate,
};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::{assert_valid_time_value, RENDER_QUANTUM_SIZE};

use super::{
    AudioNode, AudioNodeOptions, AudioScheduledSourceNode, AudioScheduledSourceNodeExt,
    ChannelConfig, ChannelCountMode, ChannelInterpretation, ScheduledSourceCompletionToken,
};

/// Options for constructing an [`ConstantSourceNode`]
// dictionary ConstantSourceOptions {
//   float offset = 1;
// };
// https://webaudio.github.io/web-audio-api/#ConstantSourceOptions
//
// @note - Does not extend AudioNodeOptions because AudioNodeOptions are
// useless for source nodes, because they instruct how to upmix the inputs.
// This is a common source of confusion, see e.g. mdn/content#18472
#[derive(Clone, Debug)]
pub struct ConstantSourceOptions {
    /// Initial parameter value of the constant signal
    pub offset: f32,
}

impl Default for ConstantSourceOptions {
    fn default() -> Self {
        Self { offset: 1. }
    }
}

/// Instructions to start or stop processing
#[derive(Debug, Copy, Clone)]
enum Schedule {
    Start(f64),
    Stop(f64),
}

/// Audio source whose output is nominally a constant value.
///
/// Can be used as a constructible `AudioParam` by automating the value of its offset.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/ConstantSourceNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#ConstantSourceNode>
/// - see also: [`BaseAudioContext::create_constant_source`]
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::node::AudioNode;
///
/// let audio_context = AudioContext::default();
///
/// let gain1 = audio_context.create_gain();
/// gain1.gain().set_value(0.);
///
/// let gain2 = audio_context.create_gain();
/// gain2.gain().set_value(0.);
///
/// let automation = audio_context.create_constant_source();
/// automation.offset().set_value(0.);
/// automation.connect(gain1.gain());
/// automation.connect(gain2.gain());
///
/// // control both `GainNode`s with 1 automation
/// automation.offset().set_target_at_time(1., audio_context.current_time(), 0.1);
/// ```
///
/// # Example
///
/// - `cargo run --release --example constant_source`
///
#[derive(Debug)]
pub struct ConstantSourceNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    offset: AudioParam,
    has_start: bool,
    completion: ScheduledSourceCompletionToken,
    injected_control: Option<InjectedConstantSourceControl>,
}

impl AudioNode for ConstantSourceNode {
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

impl AudioScheduledSourceNode for ConstantSourceNode {
    fn start(&mut self) {
        let when = self.registration.context().current_time();
        self.start_at(when);
    }

    fn start_at(&mut self, when: f64) {
        assert_valid_time_value(when);
        if let Some(control) = &self.injected_control {
            finish_exact_constant_source_mutation(control.try_start(when));
            return;
        }
        assert!(
            !self.has_start,
            "InvalidStateError - Cannot call `start` twice"
        );

        self.has_start = true;
        self.registration.post_message(Schedule::Start(when));
    }

    fn stop(&mut self) {
        let when = self.registration.context().current_time();
        self.stop_at(when);
    }

    fn stop_at(&mut self, when: f64) {
        assert_valid_time_value(when);
        if let Some(control) = &self.injected_control {
            finish_exact_constant_source_mutation(control.try_stop(when));
            return;
        }
        assert!(
            self.has_start,
            "InvalidStateError - cannot stop before start"
        );

        self.registration.post_message(Schedule::Stop(when));
    }
}

impl AudioScheduledSourceNodeExt for ConstantSourceNode {
    fn completion_token(&self) -> ScheduledSourceCompletionToken {
        self.completion.clone()
    }
}

impl ConstantSourceNode {
    /// Constructs a new `ConstantSourceNode` from explicit options.
    ///
    /// [`BaseAudioContext::create_constant_source`] is an alternative that
    /// applies the spec defaults (`offset = 1.0`).
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live
    /// * `options` - initial value of the offset parameter
    pub fn new<C: BaseAudioContext>(context: &C, options: ConstantSourceOptions) -> Self {
        if context.base().injected_node_constructor().is_some() {
            return Self::new_injected(context.base(), options);
        }
        context.base().register(move |registration| {
            let ConstantSourceOptions { offset } = options;

            let param_options = AudioParamDescriptor {
                name: String::new(),
                min_value: f32::MIN,
                max_value: f32::MAX,
                default_value: 1.,
                automation_rate: AutomationRate::A,
            };
            let (param, proc) = context.create_audio_param(param_options, &registration);
            param.set_value(offset);

            let completion = ScheduledSourceCompletionToken::new();

            let render = ConstantSourceRenderer {
                offset: proc,
                start_time: f64::MAX,
                stop_time: f64::MAX,
                ended_triggered: false,
                completion: completion.clone(),
                exact_key: None,
            };

            let node = ConstantSourceNode {
                registration,
                channel_config: ChannelConfig::default(),
                offset: param,
                has_start: false,
                completion,
                injected_control: None,
            };

            (node, Box::new(render))
        })
    }

    fn new_injected(context: &ConcreteBaseAudioContext, options: ConstantSourceOptions) -> Self {
        Self::new_injected_with_lifetime(context, options, None)
    }

    pub(crate) fn new_injected_with_lifetime(
        context: &ConcreteBaseAudioContext,
        options: ConstantSourceOptions,
        lifetime: Option<AudioNodeLifetimeReservation>,
    ) -> Self {
        Self::new_injected_with_reservations(context, options, lifetime, None)
    }

    pub(crate) fn new_injected_with_reservations(
        context: &ConcreteBaseAudioContext,
        options: ConstantSourceOptions,
        lifetime: Option<AudioNodeLifetimeReservation>,
        control: Option<AudioControlBatchReservation>,
    ) -> Self {
        let transaction = context
            .try_begin_injected_constant_source_with_reservations(lifetime, control)
            .unwrap_or_else(|error| panic!("injected ConstantSource admission failed: {error:?}"));
        let source_id = transaction.source_id();
        let offset_id = transaction.offset_id();
        let completion = ScheduledSourceCompletionToken::new_exact(transaction.completion_key());

        let descriptor = AudioParamDescriptor {
            name: String::new(),
            min_value: f32::MIN,
            max_value: f32::MAX,
            default_value: 1.,
            automation_rate: AutomationRate::A,
        };
        let (offset_raw, offset_processor) = injected_audio_param_raw_parts(descriptor);
        let offset_initial_value = offset_raw.set_initial_value_for_injected(options.offset);
        let channel_config = ChannelConfig::default();
        let param_channel_config: ChannelConfig = AudioNodeOptions {
            channel_count: 1,
            channel_count_mode: ChannelCountMode::Explicit,
            channel_interpretation: ChannelInterpretation::Discrete,
        }
        .into();
        let renderer = Box::new(ConstantSourceRenderer {
            offset: AudioParamId::from_node_id(offset_id),
            start_time: f64::MAX,
            stop_time: f64::MAX,
            ended_triggered: false,
            completion: completion.clone(),
            exact_key: Some(transaction.completion_key()),
        });
        let constructed = transaction
            .commit(InjectedConstantSourcePayload {
                offset_processor,
                source_processor: renderer,
                param_channel_config: param_channel_config.inner(),
                source_channel_config: channel_config.inner(),
                offset_initial_value,
            })
            .unwrap_or_else(|error| {
                panic!("injected ConstantSource construction failed: {error:?}")
            });
        debug_assert_eq!(constructed.source_id, source_id);
        debug_assert_eq!(constructed.offset_id, offset_id);
        let _accepted_placement = constructed.outcome;

        let offset_registration = AudioContextRegistration::from_injected_with_connection(
            offset_id,
            context.clone(),
            constructed.offset_registration,
            constructed.offset_connection,
            crate::context::InjectedConnectionEndpointKind::AudioParam,
            1,
            1,
        );
        let registration = AudioContextRegistration::from_injected_scheduled_source(
            source_id,
            context.clone(),
            constructed.source_registration,
            constructed.source_connection,
            constructed.source_control.ended_target(),
        );
        let constructor = context
            .injected_node_constructor()
            .expect("exact ConstantSource context retains constructor");
        if !constructed
            .source_control
            .matches_registration(&registration, constructor)
        {
            context.fail_closed_injected_protocol();
            panic!("exact ConstantSource control does not match its registration");
        }
        let offset = AudioParam::from_injected_raw_parts(
            offset_registration,
            offset_raw,
            constructed.offset_mutation,
        );

        Self {
            registration,
            channel_config,
            offset,
            has_start: false,
            completion,
            injected_control: Some(constructed.source_control),
        }
    }

    /// Starts an exact hosted ConstantSource while attaching one host reservation to the
    /// submitted one-command batch.
    ///
    /// # Panics
    ///
    /// Panics for an invalid time, a legacy context, duplicate start, or rejected exact control.
    pub fn start_at_with_control_reservation(
        &mut self,
        when: f64,
        reservation: AudioControlBatchReservation,
    ) {
        assert_valid_time_value(when);
        let control = self.injected_control.as_ref().unwrap_or_else(|| {
            panic!(
                "NotSupportedError - control reservations require an exact hosted ConstantSource"
            )
        });
        finish_exact_constant_source_mutation(
            control.try_start_with_host_reservation(when, reservation),
        );
    }

    /// Stops an exact hosted ConstantSource while attaching one host reservation to the
    /// submitted one-command batch.
    ///
    /// # Panics
    ///
    /// Panics for an invalid time, a legacy context, stop-before-start, or rejected exact control.
    pub fn stop_at_with_control_reservation(
        &mut self,
        when: f64,
        reservation: AudioControlBatchReservation,
    ) {
        assert_valid_time_value(when);
        let control = self.injected_control.as_ref().unwrap_or_else(|| {
            panic!(
                "NotSupportedError - control reservations require an exact hosted ConstantSource"
            )
        });
        finish_exact_constant_source_mutation(
            control.try_stop_with_host_reservation(when, reservation),
        );
    }

    /// Returns the offset `AudioParam`. Default is `1.0`.
    ///
    /// Useful as a constructible `AudioParam`: connect this once to several
    /// sink params and automate it to drive them all in lockstep. Legacy contexts support the
    /// full automation timeline; exact hosted contexts currently admit scalar `set_value`
    /// updates through their bounded control transport.
    #[must_use]
    pub fn offset(&self) -> &AudioParam {
        &self.offset
    }

    #[cfg(test)]
    pub(crate) fn injected_control_for_test(&self) -> &InjectedConstantSourceControl {
        self.injected_control
            .as_ref()
            .expect("test requires an exact injected ConstantSource")
    }
}

fn finish_exact_constant_source_mutation(
    result: Result<crate::context::CommitControlOutcome, InjectedConstantSourceMutationError>,
) {
    match result {
        Ok(_) => {}
        Err(InjectedConstantSourceMutationError::DuplicateStart) => {
            panic!("InvalidStateError - Cannot call `start` twice")
        }
        Err(InjectedConstantSourceMutationError::StopBeforeStart) => {
            panic!("InvalidStateError - cannot stop before start")
        }
        Err(InjectedConstantSourceMutationError::Inactive) => {
            panic!("InvalidStateError - exact ConstantSource is no longer active")
        }
        Err(InjectedConstantSourceMutationError::Control(error)) => {
            panic!("InvalidStateError - exact ConstantSource command was rejected: {error:?}")
        }
        Err(error) => {
            panic!("InvalidStateError - exact ConstantSource transaction failed: {error:?}")
        }
    }
}

struct ConstantSourceRenderer {
    offset: AudioParamId,
    start_time: f64,
    stop_time: f64,
    ended_triggered: bool,
    completion: ScheduledSourceCompletionToken,
    exact_key: Option<crate::events::ExactEndedEventKey>,
}

impl ConstantSourceRenderer {
    fn trigger_ended(&mut self, scope: &AudioWorkletGlobalScope) {
        if !self.ended_triggered {
            self.ended_triggered = true;
            self.completion.mark_complete_and_wake(scope);
        }
    }
}

impl AudioProcessor for ConstantSourceRenderer {
    fn process(
        &mut self,
        _inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        params: AudioParamValues<'_>,
        scope: &AudioWorkletGlobalScope,
    ) -> bool {
        // single output node
        let output = &mut outputs[0];

        let dt = 1. / scope.sample_rate as f64;
        let next_block_time = scope.current_time + dt * RENDER_QUANTUM_SIZE as f64;

        if self.start_time >= next_block_time {
            output.make_silent();

            if self.stop_time <= next_block_time {
                self.trigger_ended(scope);

                return false;
            }

            // #462 AudioScheduledSourceNodes that have not been scheduled to start can safely
            // return tail_time false in order to be collected if their control handle drops.
            return self.start_time != f64::MAX;
        }

        output.force_mono();

        let offset = params.get(&self.offset);
        let output_channel = output.channel_data_mut(0);

        // fast path
        if offset.len() == 1
            && self.start_time <= scope.current_time
            && self.stop_time >= next_block_time
        {
            output_channel.fill(offset[0]);
        } else {
            // sample accurate path
            let mut current_time = scope.current_time;

            output_channel
                .iter_mut()
                .zip(offset.iter().cycle())
                .for_each(|(o, &value)| {
                    if current_time < self.start_time || current_time >= self.stop_time {
                        *o = 0.;
                    } else {
                        // as we pick values directly from the offset param which is already
                        // computed at sub-sample accuracy, we don't need to do more than
                        // copying the values to their right place.
                        *o = value;
                    }

                    current_time += dt;
                });
        }

        // tail_time false when output has ended this quantum
        let still_running = self.stop_time > next_block_time;

        if !still_running {
            // @note: we need this check because this is called a until the program
            // ends, such as if the node was never removed from the graph
            self.trigger_ended(scope);
        }

        still_running
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(message) =
            msg.downcast_mut::<crate::context::InjectedConstantSourceRenderMessage>()
        {
            let Some(key) = self.exact_key else {
                return;
            };
            let Some(command) = message.apply_to(key) else {
                return;
            };
            match command {
                crate::context::InjectedConstantSourceCommandKind::Start(value) => {
                    self.start_time = value
                }
                crate::context::InjectedConstantSourceCommandKind::Stop(value) => {
                    self.stop_time = value
                }
            }
            return;
        }

        if let Some(schedule) = msg.downcast_ref::<Schedule>() {
            match *schedule {
                Schedule::Start(v) => self.start_time = v,
                Schedule::Stop(v) => self.stop_time = v,
            }
            return;
        }

        log::warn!("ConstantSourceRenderer: Dropping incoming message {msg:?}");
    }

    fn before_drop(&mut self, scope: &AudioWorkletGlobalScope) {
        if !self.ended_triggered
            && (scope.current_time >= self.start_time || scope.current_time >= self.stop_time)
        {
            self.trigger_ended(scope);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioNode, AudioScheduledSourceNode};

    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_audioparam_value_applies_immediately() {
        let context = OfflineAudioContext::new(1, 128, 48000.);
        let options = ConstantSourceOptions { offset: 12. };
        let src = ConstantSourceNode::new(&context, options);
        assert_float_eq!(src.offset.value(), 12., abs_all <= 0.);
    }

    #[test]
    fn test_start_stop() {
        let sample_rate = 48000.;
        let start_in_samples = (128 + 1) as f64; // start rendering in 2d block
        let stop_in_samples = (256 + 1) as f64; // stop rendering of 3rd block
        let mut context = OfflineAudioContext::new(1, 128 * 4, sample_rate);

        let mut src = context.create_constant_source();
        src.connect(&context.destination());

        src.start_at(start_in_samples / sample_rate as f64);
        src.stop_at(stop_in_samples / sample_rate as f64);

        let buffer = context.start_rendering_sync();
        let channel = buffer.get_channel_data(0);

        // 1rst block should be silence
        assert_float_eq!(channel[0..128], vec![0.; 128][..], abs_all <= 0.);

        // 2d block - start at second frame
        let mut res = vec![1.; 128];
        res[0] = 0.;
        assert_float_eq!(channel[128..256], res[..], abs_all <= 0.);

        // 3rd block - stop at second frame
        let mut res = vec![0.; 128];
        res[0] = 1.;
        assert_float_eq!(channel[256..384], res[..], abs_all <= 0.);

        // 4th block is silence
        assert_float_eq!(channel[384..512], vec![0.; 128][..], abs_all <= 0.);
    }

    #[test]
    fn test_start_in_the_past() {
        let sample_rate = 48000.;
        let mut context = OfflineAudioContext::new(1, 2 * 128, sample_rate);

        context.suspend_sync((128. / sample_rate).into(), |context| {
            let mut src = context.create_constant_source();
            src.connect(&context.destination());
            src.start_at(0.);
        });

        let buffer = context.start_rendering_sync();
        let channel = buffer.get_channel_data(0);

        // 1rst block should be silence
        assert_float_eq!(channel[0..128], vec![0.; 128][..], abs_all <= 0.);
        assert_float_eq!(channel[128..], vec![1.; 128][..], abs_all <= 0.);
    }

    #[test]
    fn test_start_in_the_future_while_dropped() {
        let sample_rate = 48000.;
        let mut context = OfflineAudioContext::new(1, 4 * 128, sample_rate);

        let mut src = context.create_constant_source();
        src.connect(&context.destination());
        src.start_at(258. / sample_rate as f64); // in 3rd block
        drop(src); // explicit drop

        let buffer = context.start_rendering_sync();
        let channel = buffer.get_channel_data(0);

        assert_float_eq!(channel[0..258], vec![0.; 258][..], abs_all <= 0.);
        assert_float_eq!(channel[258..], vec![1.; 254][..], abs_all <= 0.);
    }
}
