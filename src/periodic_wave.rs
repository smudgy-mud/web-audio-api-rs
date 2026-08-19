//! PeriodicWave interface

use std::f32::consts::PI;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use crate::context::{
    BaseAudioContext, InjectedControlIdentity, InjectedNodeConstructor, InjectedPeriodicWaveContext,
};

/// Options for constructing a [`PeriodicWave`]
#[derive(Debug, Default, Clone)]
pub struct PeriodicWaveOptions {
    /// The real parameter represents an array of cosine terms of Fourier series.
    ///
    /// The first element (index 0) represents the DC-offset.
    /// This offset has to be given but will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and
    /// harmonics of the periodic waveform.
    pub real: Option<Vec<f32>>,
    /// The imag parameter represents an array of sine terms of Fourier series.
    ///
    /// The first element (index 0) will not be taken into account
    /// to build the custom periodic waveform.
    ///
    /// The following elements (index 1 and more) represent the fundamental and
    /// harmonics of the periodic waveform.
    pub imag: Option<Vec<f32>>,
    /// By default PeriodicWave is build with normalization enabled (disable_normalization = false).
    /// In this case, a peak normalization is applied to the given custom periodic waveform.
    ///
    /// If disable_normalization is enabled (disable_normalization = true), the normalization is
    /// defined by the periodic waveform characteristics (img, and real fields).
    pub disable_normalization: bool,
}

/// `PeriodicWave` represents an arbitrary periodic waveform to be used with an `OscillatorNode`.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/PeriodicWave>
/// - specification: <https://webaudio.github.io/web-audio-api/#PeriodicWave>
/// - see also: [`BaseAudioContext::create_periodic_wave`]
/// - see also: [`OscillatorNode`](crate::node::OscillatorNode)
///
/// # Usage
///
/// ```no_run
/// use web_audio_api::context::{BaseAudioContext, AudioContext};
/// use web_audio_api::{PeriodicWave, PeriodicWaveOptions};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};
///
/// let context = AudioContext::default();
///
/// // generate a simple waveform with 2 harmonics
/// let options = PeriodicWaveOptions {
///   real: Some(vec![0., 0., 0.]),
///   imag: Some(vec![0., 0.5, 0.5]),
///   disable_normalization: false,
/// };
///
/// let periodic_wave = PeriodicWave::new(&context, options);
///
/// let mut osc = context.create_oscillator();
/// osc.set_periodic_wave(periodic_wave);
/// osc.connect(&context.destination());
/// osc.start();
/// ```
/// # Examples
///
/// - `cargo run --release --example oscillators`
///
/// Opaque host accounting retained by every native clone of one periodic-wave table.
///
/// The wrapped guard must have a nonblocking destructor. A destructor panic is contained so it
/// cannot unwind through exact command rollback or the renderer's off-real-time GC worker.
pub struct PeriodicWaveStorageLease {
    value: Option<Box<dyn Send + 'static>>,
}

impl PeriodicWaveStorageLease {
    /// Wraps host accounting for the fixed native wavetable allocation.
    pub fn new<T>(value: T) -> Self
    where
        T: Send + 'static,
    {
        Self {
            value: Some(Box::new(value)),
        }
    }
}

impl fmt::Debug for PeriodicWaveStorageLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PeriodicWaveStorageLease")
            .finish_non_exhaustive()
    }
}

impl Drop for PeriodicWaveStorageLease {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| drop(value))) {
            std::mem::forget(payload);
        }
    }
}

struct PeriodicWaveStorage {
    wavetable: Vec<f32>,
    // Mutex makes a merely Send host lease safe to share with cheap native clones. The engine
    // never locks it; it exists solely to carry destruction to the last clone's thread.
    _lease: Option<Mutex<PeriodicWaveStorageLease>>,
}

// Basically a wrapper around Arc-backed fixed storage, so `PeriodicWave`s are cheap to clone.
#[derive(Clone)]
pub struct PeriodicWave {
    storage: Arc<PeriodicWaveStorage>,
    injected_context: Option<InjectedPeriodicWaveContext>,
}

/// Maximum Fourier component count accepted by this implementation.
pub const MAX_PERIODIC_WAVE_COMPONENTS: usize = 8192;

/// Number of samples in every generated native periodic-wave table.
const PERIODIC_WAVE_TABLE_LENGTH: usize = 8192;

/// Persistent sample payload bytes owned by every generated native periodic-wave table.
pub const PERIODIC_WAVE_TABLE_BYTES: usize =
    PERIODIC_WAVE_TABLE_LENGTH * std::mem::size_of::<f32>();

impl Default for PeriodicWave {
    fn default() -> Self {
        Self {
            storage: Arc::new(PeriodicWaveStorage {
                wavetable: Vec::new(),
                _lease: None,
            }),
            injected_context: None,
        }
    }
}

impl fmt::Debug for PeriodicWave {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PeriodicWave")
            .field("table_length", &self.storage.wavetable.len())
            .field("exact_context", &self.injected_context.is_some())
            .field("has_storage_lease", &self.storage._lease.is_some())
            .finish()
    }
}

impl PeriodicWave {
    /// Returns a `PeriodicWave`
    ///
    /// # Arguments
    ///
    /// * `real` - The real parameter represents an array of cosine terms of Fourier series.
    /// * `imag` - The imag parameter represents an array of sine terms of Fourier series.
    /// * `constraints` - The constraints parameter specifies the normalization mode of the `PeriodicWave`
    ///
    /// # Panics
    ///
    /// Will panic if:
    ///
    /// * `real` is defined and its length is less than 2
    /// * `imag` is defined and its length is less than 2
    /// * `real` and `imag` are defined and theirs lengths are not equal
    /// * `PeriodicWave` is more than 8192 components
    //
    // @notes:
    // - Current implementation is very naive and could be improved using inverse
    // FFT or table lookup on SINETABLE. Such performance improvements should be
    // however tested also against this implementation.
    // - Built-in types of the `OscillatorNode` should use periodic waves
    // c.f. https://webaudio.github.io/web-audio-api/#oscillator-coefficients
    // - The question of bandlimited oscillators should also be handled
    // e.g. https://www.dafx12.york.ac.uk/papers/dafx12_submission_69.pdf
    pub fn new<C: BaseAudioContext>(context: &C, options: PeriodicWaveOptions) -> Self {
        Self::new_inner(context, options, None)
    }

    /// Builds a `PeriodicWave` while attaching opaque accounting to its native wavetable storage.
    ///
    /// The caller must acquire the reservation before invoking this constructor. The lease follows
    /// cheap clones into an exact renderer and is released only after the last clone is destroyed.
    ///
    /// # Panics
    ///
    /// Panics under the same invalid coefficient-length conditions as [`PeriodicWave::new`].
    pub fn new_with_storage_lease<C: BaseAudioContext>(
        context: &C,
        options: PeriodicWaveOptions,
        lease: PeriodicWaveStorageLease,
    ) -> Self {
        Self::new_inner(context, options, Some(lease))
    }

    fn new_inner<C: BaseAudioContext>(
        context: &C,
        options: PeriodicWaveOptions,
        lease: Option<PeriodicWaveStorageLease>,
    ) -> Self {
        let PeriodicWaveOptions {
            real,
            imag,
            disable_normalization,
        } = options;

        let (real, imag) = match (real, imag) {
            (Some(r), Some(i)) => {
                assert_eq!(
                    r.len(),
                    i.len(),
                    "IndexSizeError - `real` and `imag` length should be equal"
                );
                assert!(
                    r.len() >= 2,
                    "IndexSizeError - `real` and `imag` length should at least 2"
                );
                assert!(
                    r.len() <= MAX_PERIODIC_WAVE_COMPONENTS,
                    "NotSupportedError - `PeriodicWave` supports at most {MAX_PERIODIC_WAVE_COMPONENTS} components"
                );

                (r, i)
            }
            (Some(r), None) => {
                assert!(
                    r.len() >= 2,
                    "IndexSizeError - `real` and `imag` length should at least 2"
                );
                assert!(
                    r.len() <= MAX_PERIODIC_WAVE_COMPONENTS,
                    "NotSupportedError - `PeriodicWave` supports at most {MAX_PERIODIC_WAVE_COMPONENTS} components"
                );

                let len = r.len();
                (r, vec![0.; len])
            }
            (None, Some(i)) => {
                assert!(
                    i.len() >= 2,
                    "IndexSizeError - `real` and `imag` length should at least 2"
                );
                assert!(
                    i.len() <= MAX_PERIODIC_WAVE_COMPONENTS,
                    "NotSupportedError - `PeriodicWave` supports at most {MAX_PERIODIC_WAVE_COMPONENTS} components"
                );

                let len = i.len();
                (vec![0.; len], i)
            }
            // Defaults to sine wave
            // [spec] Note: When setting this PeriodicWave on an OscillatorNode,
            // this is equivalent to using the built-in type "sine".
            _ => (vec![0., 0.], vec![0., 1.]),
        };

        let normalize = !disable_normalization;
        // [spec] A conforming implementation MUST support PeriodicWave up to at least 8192 elements.
        let wavetable =
            Self::generate_wavetable(&real, &imag, normalize, PERIODIC_WAVE_TABLE_LENGTH);

        let injected_context = context
            .base()
            .injected_node_constructor()
            .map(InjectedNodeConstructor::periodic_wave_context);
        Self {
            storage: Arc::new(PeriodicWaveStorage {
                wavetable,
                _lease: lease.map(Mutex::new),
            }),
            injected_context,
        }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.storage.wavetable
    }

    pub(crate) fn matches_injected_constructor(
        &self,
        constructor: &InjectedNodeConstructor,
    ) -> bool {
        self.injected_context
            .as_ref()
            .is_some_and(|identity| identity.matches_constructor(constructor))
    }

    pub(crate) fn matches_injected_control(&self, identity: &InjectedControlIdentity) -> bool {
        self.injected_context
            .as_ref()
            .is_some_and(|context| context.matches_control(identity))
    }

    pub(crate) fn is_injected_context_bound(&self) -> bool {
        self.injected_context.is_some()
    }

    // cf. https://webaudio.github.io/web-audio-api/#waveform-generation
    fn generate_wavetable(reals: &[f32], imags: &[f32], normalize: bool, size: usize) -> Vec<f32> {
        let mut wavetable = Vec::with_capacity(size);
        let pi_2 = 2. * PI;

        for i in 0..size {
            let mut sample = 0.;
            let phase = pi_2 * i as f32 / size as f32;

            for j in 1..reals.len() {
                let freq = j as f32;
                let real = reals[j];
                let imag = imags[j];
                let rad = phase * freq;
                let contrib = real * rad.cos() + imag * rad.sin();
                sample += contrib;
            }

            wavetable.push(sample);
        }

        if normalize {
            Self::normalize(&mut wavetable);
        }

        wavetable
    }

    fn normalize(wavetable: &mut [f32]) {
        let mut max = 0.;

        for sample in wavetable.iter() {
            let abs = sample.abs();
            if abs > max {
                max = abs;
            }
        }

        // prevent division by 0. (nothing to normalize anyway...)
        if max > 0. {
            let norm_factor = 1. / max;

            for sample in wavetable.iter_mut() {
                *sample *= norm_factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use std::f32::consts::PI;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{
        PeriodicWave, PeriodicWaveOptions, PeriodicWaveStorageLease, MAX_PERIODIC_WAVE_COMPONENTS,
        PERIODIC_WAVE_TABLE_LENGTH,
    };
    use crate::context::OfflineAudioContext;

    fn context() -> OfflineAudioContext {
        OfflineAudioContext::new(1, 1, 48_000.)
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_real_is_defined_and_too_short() {
        let context = context();

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: None,
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_only_imag_is_defined_and_too_short() {
        let context = context();

        let options = PeriodicWaveOptions {
            real: None,
            imag: Some(vec![0.]),
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_not_equal_length() {
        let context = context();

        let options = PeriodicWaveOptions {
            real: Some(vec![0., 0., 0.]),
            imag: Some(vec![0., 0.]),
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn fails_to_build_when_imag_and_real_too_shorts() {
        let context = context();

        let options = PeriodicWaveOptions {
            real: Some(vec![0.]),
            imag: Some(vec![0.]),
            disable_normalization: false,
        };

        let _periodic_wave = PeriodicWave::new(&context, options);
    }

    #[test]
    fn rejects_component_counts_above_the_fixed_native_limit_before_table_generation() {
        let context = context();
        for options in [
            PeriodicWaveOptions {
                real: Some(vec![0.; MAX_PERIODIC_WAVE_COMPONENTS + 1]),
                imag: None,
                disable_normalization: false,
            },
            PeriodicWaveOptions {
                real: None,
                imag: Some(vec![0.; MAX_PERIODIC_WAVE_COMPONENTS + 1]),
                disable_normalization: false,
            },
            PeriodicWaveOptions {
                real: Some(vec![0.; MAX_PERIODIC_WAVE_COMPONENTS + 1]),
                imag: Some(vec![0.; MAX_PERIODIC_WAVE_COMPONENTS + 1]),
                disable_normalization: false,
            },
        ] {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                PeriodicWave::new(&context, options)
            }))
            .is_err());
        }
    }

    #[test]
    fn storage_lease_follows_all_native_clones_and_contains_a_hostile_destructor() {
        struct DropProbe(Arc<AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }
        struct PanicOnDrop;
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("hostile PeriodicWave storage lease destructor");
            }
        }

        let context = context();
        let drops = Arc::new(AtomicUsize::new(0));
        let wave = PeriodicWave::new_with_storage_lease(
            &context,
            PeriodicWaveOptions {
                real: Some(vec![0., 0.]),
                imag: Some(vec![0., 1.]),
                disable_normalization: false,
            },
            PeriodicWaveStorageLease::new(DropProbe(Arc::clone(&drops))),
        );
        let clone = wave.clone();
        drop(wave);
        assert_eq!(drops.load(Ordering::Acquire), 0);
        drop(clone);
        assert_eq!(drops.load(Ordering::Acquire), 1);

        let hostile = PeriodicWave::new_with_storage_lease(
            &context,
            PeriodicWaveOptions {
                real: Some(vec![0., 0.]),
                imag: Some(vec![0., 1.]),
                disable_normalization: false,
            },
            PeriodicWaveStorageLease::new(PanicOnDrop),
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(hostile))).is_ok());
    }

    #[test]
    fn wavetable_generate_sine() {
        let reals = [0., 0.];
        let imags = [0., 1.];

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, true, PERIODIC_WAVE_TABLE_LENGTH);
        let mut expected = Vec::new();

        for i in 0..PERIODIC_WAVE_TABLE_LENGTH {
            let sample = (i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();
            expected.push(sample);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn wavetable_generate_2f_not_norm() {
        let reals = [0., 0., 0.];
        let imags = [0., 0.5, 0.5];

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, false, PERIODIC_WAVE_TABLE_LENGTH);
        let mut expected = Vec::new();

        for i in 0..PERIODIC_WAVE_TABLE_LENGTH {
            let mut sample = 0.;
            // fundamental frequency
            sample += 0.5 * (1. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();
            // 1rst partial
            sample += 0.5 * (2. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();

            expected.push(sample);
        }

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-6);
    }

    #[test]
    fn normalize() {
        {
            let mut signal = [-0.5, 0.2];
            PeriodicWave::normalize(&mut signal);
            let expected = [-1., 0.4];

            assert_float_eq!(signal[..], expected[..], abs_all <= 0.);
        }

        {
            let mut signal = [0.5, -0.2];
            PeriodicWave::normalize(&mut signal);
            let expected = [1., -0.4];

            assert_float_eq!(signal[..], expected[..], abs_all <= 0.);
        }
    }

    #[test]
    fn wavetable_generate_2f_norm() {
        let reals = [0., 0., 0.];
        let imags = [0., 0.5, 0.5];

        let result =
            PeriodicWave::generate_wavetable(&reals, &imags, true, PERIODIC_WAVE_TABLE_LENGTH);
        let mut expected = Vec::new();

        for i in 0..PERIODIC_WAVE_TABLE_LENGTH {
            let mut sample = 0.;
            // fundamental frequency
            sample += 0.5 * (1. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();
            // 1rst partial
            sample += 0.5 * (2. * i as f32 / PERIODIC_WAVE_TABLE_LENGTH as f32 * 2. * PI).sin();

            expected.push(sample);
        }

        PeriodicWave::normalize(&mut expected);

        assert_float_eq!(result[..], expected[..], abs_all <= 1e-6);
    }
}
