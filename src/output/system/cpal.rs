//! Hosted CPAL endpoint adapter.
//!
//! The crate-owned owner thread constructs, starts, controls, and drops the CPAL `Stream`. Native
//! data/error closures retain only `SystemRenderAccess` (a Weak capability), so CPAL-internal
//! auxiliary delivery that outlives `Stream` RAII cannot retain the logical render callback. A
//! temporary callback upgrade surviving owner join makes callback retirement fail closed.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, Data, Device, Error as CpalError, ErrorKind as CpalErrorKind, FromSample, Sample,
    SampleFormat, SizedSample, Stream, StreamConfig, SupportedBufferSize, SupportedStreamConfig,
    SupportedStreamConfigRange, I24, U24,
};

use super::{
    contained_owner, install_join_observer, OwnerCompletion, OwnerResult, SystemRenderAccess,
    SystemRenderBridge,
};
use crate::context::AudioContextLatencyCategory;
use crate::media_devices::MediaDeviceInfoKind;
use crate::output::{
    AudioOutputConfig, AudioOutputDeathReason, AudioOutputEndpointShutdown, AudioOutputError,
    AudioOutputErrorKind, AudioOutputEventSink, AudioOutputRequest, AudioOutputStartFailure,
    AudioRenderCallback, AudioRenderFormat, AudioRenderStatus, PreparedAudioOutput,
    RunningAudioOutput, MAX_AUDIO_OUTPUT_CALLBACK_FRAMES,
};
use crate::RENDER_QUANTUM_SIZE;

pub(super) fn prepare(
    request: &AudioOutputRequest,
) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
    let request = request.clone();
    let (command_send, command_recv) = crossbeam_channel::bounded(1);
    let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
    let panic_events = Arc::new(Mutex::new(None));
    let owner_panic_events = Arc::clone(&panic_events);
    let owner = thread::Builder::new()
        .name("web-audio-system-output-cpal".into())
        .spawn(move || {
            let panic_ready = ready_send.clone();
            let result = contained_owner(&owner_panic_events, || {
                let prepared = match prepare_device(&request) {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        let _ = ready_send.send(Err(error.clone()));
                        return Err(error);
                    }
                };
                if ready_send.send(Ok(prepared.config.clone())).is_err() {
                    return Ok(());
                }
                run_owner(prepared, command_recv, &owner_panic_events)
            });
            if result.is_err() {
                let _ = panic_ready.try_send(result.clone().map(|_| unreachable!()));
            }
            result
        })
        .map_err(|error| {
            AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                format!("failed to spawn CPAL system output owner: {error}"),
            )
        })?;

    let observed = match install_join_observer(owner) {
        Ok(observed) => observed,
        Err(failure) => {
            let _ = command_send.try_send(CpalCommand::Abort);
            if let Err(payload) = failure.owner.join() {
                std::mem::forget(payload);
            }
            return Err(failure.error);
        }
    };

    let config = match ready_recv.recv() {
        Ok(Ok(config)) => config,
        Ok(Err(error)) => {
            let _ = observed.blocking_join.recv();
            return Err(error);
        }
        Err(_) => {
            let _ = command_send.try_send(CpalCommand::Abort);
            let joined = observed.blocking_join.recv().ok();
            return Err(joined.and_then(Result::err).unwrap_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output owner exited before preparation completed",
                )
            }));
        }
    };

    Ok(Box::new(CpalPrepared {
        config,
        command_send: Some(command_send),
        completion: Some(observed.completion),
    }))
}

struct PreparedDevice {
    device: Device,
    stream: StreamConfig,
    sample_format: SampleFormat,
    config: AudioOutputConfig,
}

fn prepare_device(request: &AudioOutputRequest) -> Result<PreparedDevice, AudioOutputError> {
    let host = preferred_host()?;
    let device = select_device(&host, request.sink_id())?;
    let default = device
        .default_output_config()
        .map_err(|error| map_cpal_error("query default output configuration", error))?;
    let supported = device
        .supported_output_configs()
        .map_err(|error| map_cpal_error("enumerate output configurations", error))?
        .collect::<Vec<_>>();
    let chosen = choose_stream_config(
        &supported,
        default,
        request.number_of_channels(),
        request.requested_sample_rate(),
    )?;
    let sample_rate = chosen.sample_rate();
    let (buffer_size, estimated_frames) = choose_buffer_size(
        chosen.buffer_size(),
        request.latency_hint(),
        sample_rate as f32,
    )?;
    let stream = StreamConfig {
        channels: chosen.channels(),
        sample_rate,
        buffer_size,
    };
    let accepted_sink = if request.sink_id().is_empty() {
        String::new()
    } else {
        request.sink_id().to_owned()
    };
    let format = AudioRenderFormat::new(
        sample_rate as f32,
        usize::from(chosen.channels()),
        MAX_AUDIO_OUTPUT_CALLBACK_FRAMES,
    )?;
    let config = AudioOutputConfig::new(
        format,
        accepted_sink,
        f64::from(estimated_frames) / f64::from(sample_rate),
    )?;
    request.validate_config(&config)?;

    Ok(PreparedDevice {
        device,
        stream,
        sample_format: chosen.sample_format(),
        config,
    })
}

fn preferred_host() -> Result<cpal::Host, AudioOutputError> {
    #[cfg(all(feature = "cpal-pipewire", target_os = "linux"))]
    if let Some(host) = available_preferred_host(cpal::HostId::PipeWire) {
        return Ok(host);
    }

    #[cfg(feature = "cpal-jack")]
    if let Some(host) = available_preferred_host(cpal::HostId::Jack) {
        return Ok(host);
    }

    Ok(cpal::default_host())
}

#[cfg(any(
    feature = "cpal-jack",
    all(feature = "cpal-pipewire", target_os = "linux")
))]
fn available_preferred_host(id: cpal::HostId) -> Option<cpal::Host> {
    let id = cpal::available_hosts()
        .into_iter()
        .find(|item| *item == id)?;
    let host = cpal::host_from_id(id).ok()?;
    host.devices().ok()?.next().map(|_| host)
}

fn select_device(host: &cpal::Host, sink_id: &str) -> Result<Device, AudioOutputError> {
    if sink_id.is_empty() {
        return host.default_output_device().ok_or_else(|| {
            AudioOutputError::new(
                AudioOutputErrorKind::DeviceUnavailable,
                "CPAL has no default output device",
            )
        });
    }

    let devices = host
        .output_devices()
        .map_err(|error| map_cpal_error("enumerate output devices", error))?;
    let mut stable_ids = Vec::new();
    for device in devices {
        let native_matches = device
            .id()
            .is_ok_and(|native| native.to_string() == sink_id);
        if native_matches {
            return Ok(device);
        }
        let channels = device
            .default_output_config()
            .ok()
            .map_or(0, |config| config.channels());
        let stable = stable_device_id(&device, channels, &stable_ids)?;
        if stable == sink_id {
            return Ok(device);
        }
        stable_ids.push(stable);
    }

    Err(AudioOutputError::new(
        AudioOutputErrorKind::DeviceUnavailable,
        format!("CPAL output device {sink_id:?} is unavailable"),
    ))
}

fn stable_device_id(
    device: &Device,
    channels: u16,
    seen: &[String],
) -> Result<String, AudioOutputError> {
    let name = device
        .description()
        .map_err(|error| map_cpal_error("read output device description", error))?
        .to_string();
    let mut index = 0;
    loop {
        let id = crate::media_devices::DeviceId::as_string(
            MediaDeviceInfoKind::AudioOutput,
            "cpal".to_owned(),
            name.clone(),
            channels,
            index,
        );
        if !seen.iter().any(|seen| seen == &id) {
            return Ok(id);
        }
        index += 1;
    }
}

fn choose_stream_config(
    supported: &[SupportedStreamConfigRange],
    default: SupportedStreamConfig,
    channels: usize,
    requested_rate: Option<f32>,
) -> Result<SupportedStreamConfig, AudioOutputError> {
    let channels = u16::try_from(channels).map_err(|_| {
        AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "requested channel count cannot be represented by CPAL",
        )
    })?;
    let requested_rate = requested_rate.map(exact_sample_rate).transpose()?;

    if requested_rate.is_none()
        && default.channels() == channels
        && pcm_format_rank(default.sample_format()).is_some()
    {
        return Ok(default);
    }

    let mut choices = supported
        .iter()
        .copied()
        .filter(|range| range.channels() == channels)
        .filter_map(|range| {
            let rank = pcm_format_rank(range.sample_format())?;
            let configured = requested_rate.map_or_else(
                || {
                    [48_000, 44_100]
                        .into_iter()
                        .find_map(|rate| range.try_with_sample_rate(rate))
                        .or_else(|| Some(range.with_max_sample_rate()))
                },
                |rate| range.try_with_sample_rate(rate),
            )?;
            Some((rank, configured))
        })
        .collect::<Vec<_>>();
    choices.sort_by_key(|(rank, _)| *rank);
    choices
        .into_iter()
        .next()
        .map(|(_, config)| config)
        .ok_or_else(|| {
            AudioOutputError::new(
                AudioOutputErrorKind::NotSupported,
                match requested_rate {
                    Some(rate) => {
                        format!("CPAL device does not support {channels} PCM channels at {rate} Hz")
                    }
                    None => format!("CPAL device does not support {channels} PCM channels"),
                },
            )
        })
}

fn exact_sample_rate(rate: f32) -> Result<u32, AudioOutputError> {
    if rate.fract() != 0. || !(1. ..=u32::MAX as f32).contains(&rate) {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("CPAL requires an integer physical sample rate, got {rate}"),
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let integer = rate as u32;
    if integer as f32 != rate {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("CPAL cannot exactly represent physical sample rate {rate}"),
        ));
    }
    Ok(integer)
}

fn pcm_format_rank(format: SampleFormat) -> Option<u8> {
    Some(match format {
        SampleFormat::F32 => 0,
        SampleFormat::F64 => 1,
        SampleFormat::I32 => 2,
        SampleFormat::I24 => 3,
        SampleFormat::I16 => 4,
        SampleFormat::I8 => 5,
        SampleFormat::I64 => 6,
        SampleFormat::U32 => 7,
        SampleFormat::U24 => 8,
        SampleFormat::U16 => 9,
        SampleFormat::U8 => 10,
        SampleFormat::U64 => 11,
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => return None,
        _ => return None,
    })
}

fn choose_buffer_size(
    supported: &SupportedBufferSize,
    latency: AudioContextLatencyCategory,
    sample_rate: f32,
) -> Result<(BufferSize, u32), AudioOutputError> {
    let desired = match latency {
        AudioContextLatencyCategory::Interactive => RENDER_QUANTUM_SIZE,
        AudioContextLatencyCategory::Balanced => RENDER_QUANTUM_SIZE * 4,
        AudioContextLatencyCategory::Playback => RENDER_QUANTUM_SIZE * 8,
        AudioContextLatencyCategory::Custom(seconds) => {
            let frames = seconds * f64::from(sample_rate);
            if !frames.is_finite() || frames <= 0. || frames > f64::from(u32::MAX) {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::NotSupported,
                    "requested CPAL output latency cannot be represented",
                ));
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let frames = frames as u32;
            let frames = frames.max(1).checked_next_power_of_two().ok_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::NotSupported,
                    "requested CPAL output buffer size is too large",
                )
            })?;
            usize::try_from(frames).map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::NotSupported,
                    "requested CPAL output buffer size is too large",
                )
            })?
        }
    };
    let desired = u32::try_from(desired).map_err(|_| {
        AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "requested CPAL output buffer size is too large",
        )
    })?;
    let frames = match supported {
        SupportedBufferSize::Range { min, max } if *min > 0 && min <= max => {
            desired.clamp(*min, *max)
        }
        SupportedBufferSize::Range { .. } => {
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::NotSupported,
                "CPAL reported an invalid output buffer range",
            ));
        }
        SupportedBufferSize::Unknown => desired,
    };
    let use_default = matches!(supported, SupportedBufferSize::Unknown)
        || (cfg!(target_os = "android")
            && matches!(
                latency,
                AudioContextLatencyCategory::Interactive | AudioContextLatencyCategory::Balanced
            ));
    Ok((
        if use_default {
            BufferSize::Default
        } else {
            BufferSize::Fixed(frames)
        },
        frames,
    ))
}

enum CpalCommand {
    Start {
        access: SystemRenderAccess,
        events: AudioOutputEventSink,
        ready: crossbeam_channel::Sender<OwnerResult>,
        accept: crossbeam_channel::Receiver<()>,
        started: crossbeam_channel::Sender<()>,
    },
    Suspend(crossbeam_channel::Sender<OwnerResult>),
    Resume(crossbeam_channel::Sender<OwnerResult>),
    Shutdown,
    Abort,
}

fn run_owner(
    prepared: PreparedDevice,
    command_recv: crossbeam_channel::Receiver<CpalCommand>,
    panic_events: &Mutex<Option<AudioOutputEventSink>>,
) -> OwnerResult {
    run_owner_with(command_recv, panic_events, move |access| {
        build_stream(&prepared, access)
    })
}

fn run_owner_with<S, F>(
    command_recv: crossbeam_channel::Receiver<CpalCommand>,
    panic_events: &Mutex<Option<AudioOutputEventSink>>,
    build: F,
) -> OwnerResult
where
    S: CpalStreamControl,
    F: FnOnce(SystemRenderAccess) -> Result<S, AudioOutputError>,
{
    let first = match command_recv.recv() {
        Ok(command) => command,
        Err(_) => return Ok(()),
    };
    let CpalCommand::Start {
        access,
        events,
        ready,
        accept,
        started,
    } = first
    else {
        return match first {
            CpalCommand::Abort | CpalCommand::Shutdown => Ok(()),
            CpalCommand::Suspend(response) | CpalCommand::Resume(response) => {
                let error = AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output received state work before start",
                );
                let _ = response.send(Err(error.clone()));
                Err(error)
            }
            CpalCommand::Start { .. } => unreachable!(),
        };
    };
    match panic_events.lock() {
        Ok(mut retained) => *retained = Some(events),
        Err(poisoned) => *poisoned.into_inner() = Some(events),
    }
    let close_guard = AccessCloseGuard::new(access);

    let stream = match build(close_guard.access().clone()) {
        Ok(stream) => stream,
        Err(error) => {
            close_guard.close();
            let _ = ready.send(Err(error));
            clear_panic_events(panic_events);
            return Ok(());
        }
    };
    match panic::catch_unwind(AssertUnwindSafe(|| stream.play())) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let error = map_cpal_error("start output stream", error);
            close_guard.close();
            drop(stream);
            let _ = ready.send(Err(error));
            clear_panic_events(panic_events);
            return Ok(());
        }
        Err(payload) => {
            close_guard
                .access()
                .report_endpoint_death(AudioOutputDeathReason::BackendFailure);
            std::mem::forget(payload);
            drop(stream);
            clear_panic_events(panic_events);
            return Err(AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "CPAL output panicked while starting its stream",
            ));
        }
    }
    if close_guard.access().is_closed() {
        close_guard.close();
        drop(stream);
        let _ = ready.send(Err(AudioOutputError::new(
            AudioOutputErrorKind::BackendSpecific,
            "CPAL output failed before startup completed",
        )));
        clear_panic_events(panic_events);
        return Ok(());
    }
    if ready.send(Ok(())).is_err() || accept.recv().is_err() {
        close_guard.close();
        drop(stream);
        clear_panic_events(panic_events);
        return Ok(());
    }
    // The caller's acceptance is the start boundary. Native callbacks stay on the software-silence
    // gate throughout build/play, then the owner opens them before acknowledging Running.
    close_guard.access().resume();
    let _ = started.send(());

    let result = match drive_started(stream, command_recv, close_guard) {
        StartedOutcome::Retired(result) => result,
        StartedOutcome::Quarantine(stream) => quarantine_stream(stream),
    };
    clear_panic_events(panic_events);
    result
}

struct AccessCloseGuard {
    access: SystemRenderAccess,
}

impl AccessCloseGuard {
    fn new(access: SystemRenderAccess) -> Self {
        Self { access }
    }

    fn access(&self) -> &SystemRenderAccess {
        &self.access
    }

    fn close(&self) {
        self.access.close();
    }
}

impl Drop for AccessCloseGuard {
    fn drop(&mut self) {
        self.access.close();
    }
}

trait CpalStreamControl: 'static {
    fn play(&self) -> Result<(), CpalError>;
    fn pause(&self) -> Result<(), CpalError>;
}

impl CpalStreamControl for Stream {
    fn play(&self) -> Result<(), CpalError> {
        StreamTrait::play(self)
    }

    fn pause(&self) -> Result<(), CpalError> {
        StreamTrait::pause(self)
    }
}

fn clear_panic_events(events: &Mutex<Option<AudioOutputEventSink>>) {
    match events.lock() {
        Ok(mut events) => *events = None,
        Err(poisoned) => *poisoned.into_inner() = None,
    }
}

fn build_stream(
    prepared: &PreparedDevice,
    access: SystemRenderAccess,
) -> Result<Stream, AudioOutputError> {
    let channels = usize::from(prepared.stream.channels);
    let expected_format = prepared.sample_format;
    let mut scratch = vec![0.; MAX_AUDIO_OUTPUT_CALLBACK_FRAMES * channels];
    let error_access = access.clone();
    prepared
        .device
        .build_output_stream_raw(
            prepared.stream,
            expected_format,
            move |data, _| {
                process_output_data(data, expected_format, channels, &mut scratch, &access);
            },
            move |error| {
                handle_stream_error(&error_access, error.kind());
            },
            None,
        )
        .map_err(|error| map_cpal_error("build output stream", error))
}

fn handle_stream_error(access: &SystemRenderAccess, kind: CpalErrorKind) {
    access.report_endpoint_death(death_reason(kind));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PauseMode {
    Running,
    Hardware,
    Software,
}

enum StartedOutcome<S> {
    Retired(OwnerResult),
    Quarantine(S),
}

fn drive_started<S: CpalStreamControl>(
    stream: S,
    command_recv: crossbeam_channel::Receiver<CpalCommand>,
    close_guard: AccessCloseGuard,
) -> StartedOutcome<S> {
    let access = close_guard.access();
    let mut mode = PauseMode::Running;
    loop {
        match command_recv.recv() {
            Ok(CpalCommand::Suspend(response)) => match mode {
                PauseMode::Hardware | PauseMode::Software => {
                    let _ = response.send(Ok(()));
                }
                PauseMode::Running => {
                    let pause = panic::catch_unwind(AssertUnwindSafe(|| stream.pause()));
                    match pause {
                        Ok(Ok(())) => {
                            mode = PauseMode::Hardware;
                            let _ = response.send(Ok(()));
                        }
                        Ok(Err(error)) if error.kind() == CpalErrorKind::UnsupportedOperation => {
                            access.suspend();
                            mode = PauseMode::Software;
                            let _ = response.send(Ok(()));
                        }
                        Ok(Err(error)) => {
                            let mapped = map_cpal_error("suspend output stream", error.clone());
                            access.report_endpoint_death(death_reason(error.kind()));
                            let _ = response.send(Err(mapped));
                            return StartedOutcome::Quarantine(stream);
                        }
                        Err(payload) => {
                            access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                            std::mem::forget(payload);
                            let _ = response.send(Err(AudioOutputError::new(
                                AudioOutputErrorKind::BackendSpecific,
                                "CPAL output panicked while suspending its stream",
                            )));
                            return StartedOutcome::Quarantine(stream);
                        }
                    }
                }
            },
            Ok(CpalCommand::Resume(response)) => match mode {
                PauseMode::Running => {
                    let _ = response.send(Ok(()));
                }
                PauseMode::Software => {
                    access.resume();
                    mode = PauseMode::Running;
                    let _ = response.send(Ok(()));
                }
                PauseMode::Hardware => {
                    let play = panic::catch_unwind(AssertUnwindSafe(|| stream.play()));
                    match play {
                        Ok(Ok(())) => {
                            mode = PauseMode::Running;
                            let _ = response.send(Ok(()));
                        }
                        Ok(Err(error)) => {
                            let mapped = map_cpal_error("resume output stream", error.clone());
                            access.report_endpoint_death(death_reason(error.kind()));
                            let _ = response.send(Err(mapped));
                            return StartedOutcome::Quarantine(stream);
                        }
                        Err(payload) => {
                            access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                            std::mem::forget(payload);
                            let _ = response.send(Err(AudioOutputError::new(
                                AudioOutputErrorKind::BackendSpecific,
                                "CPAL output panicked while resuming its stream",
                            )));
                            return StartedOutcome::Quarantine(stream);
                        }
                    }
                }
            },
            Ok(CpalCommand::Shutdown | CpalCommand::Abort) => {
                access.close();
                drop(stream);
                return StartedOutcome::Retired(Ok(()));
            }
            Ok(CpalCommand::Start { ready, .. }) => {
                let error = AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output received a duplicate start",
                );
                let _ = ready.send(Err(error.clone()));
                access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                drop(stream);
                return StartedOutcome::Retired(Err(error));
            }
            Err(_) => {
                access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                drop(stream);
                return StartedOutcome::Retired(Err(AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output controller disconnected",
                )));
            }
        }
    }
}

fn quarantine_stream<S: CpalStreamControl>(stream: S) -> ! {
    let _stream = std::mem::ManuallyDrop::new(stream);
    loop {
        thread::park();
    }
}

fn process_output_data(
    data: &mut Data,
    expected: SampleFormat,
    channels: usize,
    scratch: &mut [f32],
    access: &SystemRenderAccess,
) {
    if data.len() == 0 {
        access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        return;
    }
    if data.sample_format() != expected {
        fill_data_equilibrium(data);
        access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        return;
    }
    macro_rules! process {
        ($ty:ty) => {{
            let Some(output) = data.as_slice_mut::<$ty>() else {
                fill_data_equilibrium(data);
                access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
                return;
            };
            process_typed_output(output, channels, scratch, access);
        }};
    }
    match expected {
        SampleFormat::F32 => {
            let Some(output) = data.as_slice_mut::<f32>() else {
                fill_data_equilibrium(data);
                access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
                return;
            };
            process_f32_output(output, channels, access);
        }
        SampleFormat::F64 => process!(f64),
        SampleFormat::I8 => process!(i8),
        SampleFormat::I16 => process!(i16),
        SampleFormat::I24 => process!(I24),
        SampleFormat::I32 => process!(i32),
        SampleFormat::I64 => process!(i64),
        SampleFormat::U8 => process!(u8),
        SampleFormat::U16 => process!(u16),
        SampleFormat::U24 => process!(U24),
        SampleFormat::U32 => process!(u32),
        SampleFormat::U64 => process!(u64),
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => {
            fill_data_equilibrium(data);
            access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        }
        _ => {
            fill_data_equilibrium(data);
            access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        }
    }
}

fn process_f32_output(output: &mut [f32], channels: usize, access: &SystemRenderAccess) {
    if channels == 0 || output.len() % channels != 0 {
        output.fill(f32::EQUILIBRIUM);
        access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        return;
    }
    let max_samples = MAX_AUDIO_OUTPUT_CALLBACK_FRAMES * channels;
    let mut chunks = output.chunks_mut(max_samples);
    while let Some(chunk) = chunks.next() {
        if access.render_interleaved_f32(chunk) == AudioRenderStatus::Stop {
            for remaining in chunks {
                remaining.fill(f32::EQUILIBRIUM);
            }
            return;
        }
    }
}

fn process_typed_output<T>(
    output: &mut [T],
    channels: usize,
    scratch: &mut [f32],
    access: &SystemRenderAccess,
) where
    T: Sample + SizedSample + FromSample<f32>,
{
    if channels == 0
        || output.len() % channels != 0
        || scratch.len() < MAX_AUDIO_OUTPUT_CALLBACK_FRAMES * channels
    {
        output.fill(T::EQUILIBRIUM);
        access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        return;
    }
    let max_samples = MAX_AUDIO_OUTPUT_CALLBACK_FRAMES * channels;
    let mut chunks = output.chunks_mut(max_samples);
    while let Some(chunk) = chunks.next() {
        let logical = &mut scratch[..chunk.len()];
        let status = access.render_interleaved_f32(logical);
        for (physical, logical) in chunk.iter_mut().zip(logical.iter().copied()) {
            *physical = T::from_sample(logical);
        }
        if status == AudioRenderStatus::Stop {
            for remaining in chunks {
                remaining.fill(T::EQUILIBRIUM);
            }
            return;
        }
    }
}

fn fill_data_equilibrium(data: &mut Data) {
    macro_rules! fill {
        ($ty:ty) => {
            if let Some(output) = data.as_slice_mut::<$ty>() {
                output.fill(<$ty as Sample>::EQUILIBRIUM);
            }
        };
    }
    match data.sample_format() {
        SampleFormat::F32 => fill!(f32),
        SampleFormat::F64 => fill!(f64),
        SampleFormat::I8 => fill!(i8),
        SampleFormat::I16 => fill!(i16),
        SampleFormat::I24 => fill!(I24),
        SampleFormat::I32 => fill!(i32),
        SampleFormat::I64 => fill!(i64),
        SampleFormat::U8 => fill!(u8),
        SampleFormat::U16 => fill!(u16),
        SampleFormat::U24 => fill!(U24),
        SampleFormat::U32 => fill!(u32),
        SampleFormat::U64 => fill!(u64),
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => {
            data.bytes_mut().fill(0x69);
        }
        _ => data.bytes_mut().fill(0),
    }
}

fn death_reason(kind: CpalErrorKind) -> AudioOutputDeathReason {
    match kind {
        CpalErrorKind::DeviceNotAvailable | CpalErrorKind::HostUnavailable => {
            AudioOutputDeathReason::DeviceUnavailable
        }
        _ => AudioOutputDeathReason::BackendFailure,
    }
}

fn map_cpal_error(operation: &str, error: CpalError) -> AudioOutputError {
    let kind = match error.kind() {
        CpalErrorKind::DeviceNotAvailable | CpalErrorKind::HostUnavailable => {
            AudioOutputErrorKind::DeviceUnavailable
        }
        CpalErrorKind::UnsupportedConfig | CpalErrorKind::UnsupportedOperation => {
            AudioOutputErrorKind::NotSupported
        }
        CpalErrorKind::InvalidInput => AudioOutputErrorKind::InvalidArgument,
        _ => AudioOutputErrorKind::BackendSpecific,
    };
    AudioOutputError::new(kind, format!("CPAL failed to {operation}: {error}"))
}

struct CpalPrepared {
    config: AudioOutputConfig,
    command_send: Option<crossbeam_channel::Sender<CpalCommand>>,
    completion: Option<OwnerCompletion>,
}

impl fmt::Debug for CpalPrepared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpalPrepared")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PreparedAudioOutput for CpalPrepared {
    fn config(&self) -> &AudioOutputConfig {
        &self.config
    }

    fn start(
        mut self: Box<Self>,
        callback: AudioRenderCallback,
        events: AudioOutputEventSink,
    ) -> Result<Box<dyn RunningAudioOutput>, AudioOutputStartFailure> {
        let command_send = self
            .command_send
            .take()
            .expect("prepared CPAL output is single-use");
        let completion = self
            .completion
            .take()
            .expect("prepared CPAL output is single-use");
        let bridge = SystemRenderBridge::new(callback, events);
        let access = SystemRenderBridge::access(&bridge);
        // CPAL implementations may invoke their data closure during stream construction or play.
        // Keep those provisional calls silent until the owner publishes a successful Start.
        access.suspend();
        let owner_events = bridge.events();
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let (accept_send, accept_recv) = crossbeam_channel::bounded(1);
        let (started_send, started_recv) = crossbeam_channel::bounded(1);
        if let Err(error) = command_send.send(CpalCommand::Start {
            access,
            events: owner_events,
            ready: ready_send,
            accept: accept_recv,
            started: started_send,
        }) {
            bridge.close();
            drop(error.into_inner());
            return Err(AudioOutputStartFailure::new(
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output owner disappeared during start",
                ),
                completion.into_bridge_shutdown(bridge),
            ));
        }
        match ready_recv.recv() {
            Ok(Ok(())) => {
                if accept_send.send(()).is_ok() && started_recv.recv().is_ok() {
                    Ok(Box::new(CpalRunning {
                        command_send: Some(command_send),
                        completion: Some(completion),
                        bridge: Some(bridge),
                    }))
                } else {
                    bridge.close();
                    Err(AudioOutputStartFailure::new(
                        AudioOutputError::new(
                            AudioOutputErrorKind::BackendSpecific,
                            "CPAL system output owner failed at the accepted start boundary",
                        ),
                        completion.into_bridge_shutdown(bridge),
                    ))
                }
            }
            Ok(Err(error)) => {
                bridge.close();
                Err(AudioOutputStartFailure::new(
                    error,
                    completion.into_bridge_shutdown(bridge),
                ))
            }
            Err(_) => {
                bridge.close();
                Err(AudioOutputStartFailure::new(
                    AudioOutputError::new(
                        AudioOutputErrorKind::BackendSpecific,
                        "CPAL system output owner exited during start",
                    ),
                    completion.into_bridge_shutdown(bridge),
                ))
            }
        }
    }

    fn abort(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CpalCommand::Abort);
        }
        self.completion.take().map_or_else(
            || missing_completion("prepared abort"),
            OwnerCompletion::into_shutdown,
        )
    }
}

impl Drop for CpalPrepared {
    fn drop(&mut self) {
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CpalCommand::Abort);
        }
    }
}

struct CpalRunning {
    command_send: Option<crossbeam_channel::Sender<CpalCommand>>,
    completion: Option<OwnerCompletion>,
    bridge: Option<Arc<SystemRenderBridge>>,
}

impl fmt::Debug for CpalRunning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpalRunning").finish_non_exhaustive()
    }
}

impl CpalRunning {
    fn state_command(
        &self,
        command: impl FnOnce(crossbeam_channel::Sender<OwnerResult>) -> CpalCommand,
    ) -> OwnerResult {
        let (response_send, response_recv) = crossbeam_channel::bounded(1);
        self.command_send
            .as_ref()
            .ok_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output no longer owns its controller",
                )
            })?
            .send(command(response_send))
            .map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "CPAL system output owner disconnected",
                )
            })?;
        response_recv.recv().map_err(|_| {
            AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "CPAL system output owner exited during a state transition",
            )
        })?
    }
}

impl RunningAudioOutput for CpalRunning {
    fn resume(&mut self) -> OwnerResult {
        self.state_command(CpalCommand::Resume)
    }

    fn suspend(&mut self) -> OwnerResult {
        self.state_command(CpalCommand::Suspend)
    }

    fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        if let Some(bridge) = self.bridge.as_ref() {
            bridge.close();
        }
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CpalCommand::Shutdown);
        }
        let bridge = self.bridge.take();
        self.completion.take().map_or_else(
            || missing_completion("running shutdown"),
            |completion| {
                bridge.map_or_else(
                    || missing_completion("running callback retirement"),
                    |bridge| completion.into_bridge_shutdown(bridge),
                )
            },
        )
    }
}

impl Drop for CpalRunning {
    fn drop(&mut self) {
        if let Some(bridge) = self.bridge.as_ref() {
            bridge.close();
        }
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CpalCommand::Shutdown);
        }
        self.bridge.take();
    }
}

fn missing_completion(operation: &str) -> AudioOutputEndpointShutdown {
    AudioOutputEndpointShutdown::ready(Err(AudioOutputError::new(
        AudioOutputErrorKind::Shutdown,
        format!("CPAL system output lost completion ownership during {operation}"),
    )))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use super::*;
    use crate::output::{
        audio_render_test_pair, AudioOutputEventWatcher, AudioRenderOwner,
        EndpointShutdownConfirmed,
    };
    use futures::executor;
    use futures::task::{waker, ArcWake};

    fn test_bridge(
        format: AudioRenderFormat,
        render: impl Fn(&mut [f32]) + Send + 'static,
    ) -> (
        AudioRenderOwner,
        Arc<SystemRenderBridge>,
        SystemRenderAccess,
        AudioOutputEventWatcher,
    ) {
        let (events, watcher) = AudioOutputEventSink::bounded(8);
        let (owner, callback) = audio_render_test_pair(format, events.clone(), render, || Ok(()));
        let bridge = SystemRenderBridge::new(callback, events);
        let access = SystemRenderBridge::access(&bridge);
        (owner, bridge, access, watcher)
    }

    fn retire(owner: AudioRenderOwner, bridge: Arc<SystemRenderBridge>) {
        owner.begin_shutdown();
        bridge.close();
        SystemRenderBridge::try_retire(bridge)
            .map_err(|_| ())
            .unwrap();
        owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }

    fn raw_callback(
        format: AudioRenderFormat,
    ) -> (
        AudioRenderOwner,
        AudioRenderCallback,
        AudioOutputEventSink,
        AudioOutputEventWatcher,
    ) {
        let (events, watcher) = AudioOutputEventSink::bounded(8);
        let (owner, callback) = audio_render_test_pair(format, events.clone(), |_| {}, || Ok(()));
        (owner, callback, events, watcher)
    }

    fn counted_raw_callback(
        format: AudioRenderFormat,
        renders: Arc<AtomicUsize>,
    ) -> (
        AudioRenderOwner,
        AudioRenderCallback,
        AudioOutputEventSink,
        AudioOutputEventWatcher,
    ) {
        let (events, watcher) = AudioOutputEventSink::bounded(8);
        let (owner, callback) = audio_render_test_pair(
            format,
            events.clone(),
            move |_| {
                renders.fetch_add(1, Ordering::AcqRel);
            },
            || Ok(()),
        );
        (owner, callback, events, watcher)
    }

    #[derive(Clone, Copy, Debug)]
    enum FakeAction {
        Ok,
        Unsupported,
        DeviceUnavailable,
        Panic,
        ReportDeviceDeath,
        RenderThenDeviceUnavailable,
        RenderThenPanic,
    }

    #[derive(Default)]
    struct FakeProbes {
        plays: AtomicUsize,
        pauses: AtomicUsize,
        drops: AtomicUsize,
        access: Mutex<Option<SystemRenderAccess>>,
    }

    struct FakeDropBlock {
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
    }

    struct FakeStream {
        access: SystemRenderAccess,
        play: Mutex<VecDeque<FakeAction>>,
        pause: Mutex<VecDeque<FakeAction>>,
        probes: Arc<FakeProbes>,
        drop_block: Option<FakeDropBlock>,
    }

    impl FakeStream {
        fn new(
            access: SystemRenderAccess,
            play: impl IntoIterator<Item = FakeAction>,
            pause: impl IntoIterator<Item = FakeAction>,
            probes: Arc<FakeProbes>,
            drop_block: Option<FakeDropBlock>,
        ) -> Self {
            match probes.access.lock() {
                Ok(mut slot) => *slot = Some(access.clone()),
                Err(poisoned) => *poisoned.into_inner() = Some(access.clone()),
            }
            Self {
                access,
                play: Mutex::new(play.into_iter().collect()),
                pause: Mutex::new(pause.into_iter().collect()),
                probes,
                drop_block,
            }
        }

        fn apply(&self, action: FakeAction) -> Result<(), CpalError> {
            match action {
                FakeAction::Ok => Ok(()),
                FakeAction::Unsupported => Err(CpalError::new(CpalErrorKind::UnsupportedOperation)),
                FakeAction::DeviceUnavailable => {
                    Err(CpalError::new(CpalErrorKind::DeviceNotAvailable))
                }
                FakeAction::Panic => panic!("forced fake CPAL method panic"),
                FakeAction::ReportDeviceDeath => {
                    handle_stream_error(&self.access, CpalErrorKind::DeviceNotAvailable);
                    Ok(())
                }
                FakeAction::RenderThenDeviceUnavailable => {
                    let mut output = [1.; 256];
                    assert_eq!(
                        self.access.render_interleaved_f32(&mut output),
                        AudioRenderStatus::Continue
                    );
                    assert!(output.iter().all(|sample| *sample == 0.));
                    Err(CpalError::new(CpalErrorKind::DeviceNotAvailable))
                }
                FakeAction::RenderThenPanic => {
                    let mut output = [1.; 256];
                    assert_eq!(
                        self.access.render_interleaved_f32(&mut output),
                        AudioRenderStatus::Continue
                    );
                    assert!(output.iter().all(|sample| *sample == 0.));
                    panic!("forced fake CPAL method panic after provisional render")
                }
            }
        }
    }

    impl CpalStreamControl for FakeStream {
        fn play(&self) -> Result<(), CpalError> {
            self.probes.plays.fetch_add(1, Ordering::AcqRel);
            let action = match self.play.lock() {
                Ok(mut actions) => actions.pop_front(),
                Err(poisoned) => poisoned.into_inner().pop_front(),
            }
            .unwrap_or(FakeAction::Ok);
            self.apply(action)
        }

        fn pause(&self) -> Result<(), CpalError> {
            self.probes.pauses.fetch_add(1, Ordering::AcqRel);
            let action = match self.pause.lock() {
                Ok(mut actions) => actions.pop_front(),
                Err(poisoned) => poisoned.into_inner().pop_front(),
            }
            .unwrap_or(FakeAction::Ok);
            self.apply(action)
        }
    }

    impl Drop for FakeStream {
        fn drop(&mut self) {
            assert!(
                self.access.is_closed(),
                "CPAL stream destruction must observe a closed callback bridge"
            );
            self.probes.drops.fetch_add(1, Ordering::AcqRel);
            if let Some(block) = self.drop_block.take() {
                let _ = block.entered.send(());
                let _ = block.release.recv();
            }
        }
    }

    enum FakeBuild {
        Stream {
            play: Vec<FakeAction>,
            pause: Vec<FakeAction>,
            drop_block: Option<FakeDropBlock>,
        },
        Error,
        Panic,
        RenderThenError,
        RenderThenPanic,
    }

    fn fake_prepared(build: FakeBuild, probes: Arc<FakeProbes>) -> CpalPrepared {
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let panic_events = Arc::new(Mutex::new(None));
        let owner_events = Arc::clone(&panic_events);
        let owner = thread::spawn(move || {
            contained_owner(&owner_events, || {
                run_owner_with(command_recv, &owner_events, move |access| {
                    match probes.access.lock() {
                        Ok(mut slot) => *slot = Some(access.clone()),
                        Err(poisoned) => *poisoned.into_inner() = Some(access.clone()),
                    }
                    match build {
                        FakeBuild::Stream {
                            play,
                            pause,
                            drop_block,
                        } => Ok(FakeStream::new(access, play, pause, probes, drop_block)),
                        FakeBuild::Error => Err(AudioOutputError::new(
                            AudioOutputErrorKind::BackendSpecific,
                            "forced fake CPAL build failure",
                        )),
                        FakeBuild::Panic => panic!("forced fake CPAL build panic"),
                        FakeBuild::RenderThenError => {
                            let mut output = [1.; 256];
                            assert_eq!(
                                access.render_interleaved_f32(&mut output),
                                AudioRenderStatus::Continue
                            );
                            assert!(output.iter().all(|sample| *sample == 0.));
                            Err(AudioOutputError::new(
                                AudioOutputErrorKind::BackendSpecific,
                                "forced fake CPAL build failure after provisional render",
                            ))
                        }
                        FakeBuild::RenderThenPanic => {
                            let mut output = [1.; 256];
                            assert_eq!(
                                access.render_interleaved_f32(&mut output),
                                AudioRenderStatus::Continue
                            );
                            assert!(output.iter().all(|sample| *sample == 0.));
                            panic!("forced fake CPAL build panic after provisional render")
                        }
                    }
                })
            })
        });
        let observed = install_join_observer(owner).map_err(|_| ()).unwrap();
        drop(observed.blocking_join);
        CpalPrepared {
            config: AudioOutputConfig::new(
                AudioRenderFormat::new(48_000., 2, MAX_AUDIO_OUTPUT_CALLBACK_FRAMES).unwrap(),
                "",
                0.,
            )
            .unwrap(),
            command_send: Some(command_send),
            completion: Some(observed.completion),
        }
    }

    fn data<T: SizedSample>(slice: &mut [T]) -> Data {
        // SAFETY: the returned `Data` borrows the typed slice for the duration of the test call.
        unsafe { Data::from_parts(slice.as_mut_ptr().cast(), slice.len(), T::FORMAT) }
    }

    #[test]
    fn planner_is_exact_and_rejects_fractional_and_dsd_only_devices() {
        let ranges = [
            SupportedStreamConfigRange::new(
                2,
                44_100,
                96_000,
                SupportedBufferSize::Range { min: 64, max: 512 },
                SampleFormat::I16,
            ),
            SupportedStreamConfigRange::new(
                2,
                48_000,
                48_000,
                SupportedBufferSize::Unknown,
                SampleFormat::F32,
            ),
        ];
        let default =
            SupportedStreamConfig::new(6, 48_000, SupportedBufferSize::Unknown, SampleFormat::F32);
        let chosen = choose_stream_config(&ranges, default, 2, Some(48_000.)).unwrap();
        assert_eq!(chosen.sample_format(), SampleFormat::F32);
        assert_eq!(chosen.channels(), 2);
        assert_eq!(chosen.sample_rate(), 48_000);
        assert_eq!(
            choose_stream_config(&ranges, default, 1, Some(48_000.))
                .unwrap_err()
                .kind(),
            AudioOutputErrorKind::NotSupported
        );
        assert_eq!(
            choose_stream_config(&ranges, default, 2, Some(48_000.5))
                .unwrap_err()
                .kind(),
            AudioOutputErrorKind::NotSupported
        );
        let dsd = [SupportedStreamConfigRange::new(
            2,
            44_100,
            96_000,
            SupportedBufferSize::Unknown,
            SampleFormat::DsdU8,
        )];
        assert!(choose_stream_config(&dsd, default, 2, Some(48_000.)).is_err());
        let overflowing_latency = f64::from((1_u32 << 31) + 1) / 48_000.;
        assert_eq!(
            choose_buffer_size(
                &SupportedBufferSize::Unknown,
                AudioContextLatencyCategory::Custom(overflowing_latency),
                48_000.,
            )
            .unwrap_err()
            .kind(),
            AudioOutputErrorKind::NotSupported
        );
        assert_eq!(
            choose_buffer_size(
                &SupportedBufferSize::Range { min: 512, max: 64 },
                AudioContextLatencyCategory::Interactive,
                48_000.,
            )
            .unwrap_err()
            .kind(),
            AudioOutputErrorKind::NotSupported
        );
    }

    #[test]
    fn callback_preflights_whole_slice_and_chunks_without_allocation() {
        let renders = Arc::new(AtomicUsize::new(0));
        let callback_renders = Arc::clone(&renders);
        let format = AudioRenderFormat::new(48_000., 2, MAX_AUDIO_OUTPUT_CALLBACK_FRAMES).unwrap();
        let (owner, bridge, access, watcher) = test_bridge(format, move |output| {
            callback_renders.fetch_add(1, Ordering::AcqRel);
            output.fill(0.5);
        });
        let frames = MAX_AUDIO_OUTPUT_CALLBACK_FRAMES + 17;
        let mut output = vec![0_f32; frames * 2];
        let mut raw = data(&mut output);
        let mut scratch = vec![0.; MAX_AUDIO_OUTPUT_CALLBACK_FRAMES * 2];
        alloc_counter::deny_alloc(|| {
            process_output_data(&mut raw, SampleFormat::F32, 2, &mut scratch, &access);
        });
        assert_eq!(renders.load(Ordering::Acquire), 2);
        assert!(output.iter().all(|sample| *sample == 0.5));
        assert!(watcher.death_reason().is_none());
        retire(owner, bridge);

        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (owner, bridge, access, watcher) = test_bridge(format, |_| {
            panic!("misaligned physical data must not reach the logical callback")
        });
        let mut invalid = [1_f32; 3];
        let mut raw = data(&mut invalid);
        let mut scratch = [0.; 256];
        process_output_data(&mut raw, SampleFormat::F32, 2, &mut scratch, &access);
        assert_eq!(invalid, [0.; 3]);
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackProtocolViolation)
        );
        retire(owner, bridge);

        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (owner, bridge, access, watcher) = test_bridge(format, |_| {
            panic!("a mismatched physical format must not reach the logical callback")
        });
        let mut mismatched = [0_u8; 4];
        let mut raw = data(&mut mismatched);
        process_output_data(&mut raw, SampleFormat::F32, 2, &mut scratch, &access);
        assert_eq!(mismatched, [u8::EQUILIBRIUM; 4]);
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackProtocolViolation)
        );
        retire(owner, bridge);

        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (owner, bridge, access, watcher) = test_bridge(format, |_| {
            panic!("an empty physical callback must not reach the logical callback")
        });
        let mut empty = Vec::<f32>::new();
        let mut raw = data(&mut empty);
        process_output_data(&mut raw, SampleFormat::F32, 2, &mut scratch, &access);
        assert_eq!(
            watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackProtocolViolation)
        );
        retire(owner, bridge);
    }

    fn assert_conversion<T>()
    where
        T: Sample + SizedSample + FromSample<f32> + fmt::Debug + PartialEq,
    {
        let format = AudioRenderFormat::new(48_000., 1, 128).unwrap();
        let (owner, bridge, access, watcher) = test_bridge(format, |output| {
            output.copy_from_slice(&[-1., 0., 1.]);
        });
        let mut output = [T::EQUILIBRIUM; 3];
        let mut scratch = [0.; MAX_AUDIO_OUTPUT_CALLBACK_FRAMES];
        alloc_counter::deny_alloc(|| {
            process_typed_output(&mut output, 1, &mut scratch, &access);
        });
        assert_eq!(
            output,
            [T::from_sample(-1.), T::from_sample(0.), T::from_sample(1.)]
        );
        assert!(watcher.death_reason().is_none());
        retire(owner, bridge);
    }

    #[test]
    fn every_pcm_format_converts_extrema_and_unsigned_equilibrium() {
        assert_conversion::<f64>();
        assert_conversion::<i8>();
        assert_conversion::<i16>();
        assert_conversion::<I24>();
        assert_conversion::<i32>();
        assert_conversion::<i64>();
        assert_conversion::<u8>();
        assert_conversion::<u16>();
        assert_conversion::<U24>();
        assert_conversion::<u32>();
        assert_conversion::<u64>();
        assert_eq!(u8::EQUILIBRIUM, 128);
        assert_eq!(u16::EQUILIBRIUM, 32_768);
        assert_eq!(U24::EQUILIBRIUM, U24::new(1 << 23).unwrap());
    }

    #[test]
    fn software_suspend_is_silent_and_late_error_does_not_degrade_close() {
        let renders = Arc::new(AtomicUsize::new(0));
        let callback_renders = Arc::clone(&renders);
        let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
        let (owner, bridge, access, watcher) = test_bridge(format, move |output| {
            callback_renders.fetch_add(1, Ordering::AcqRel);
            output.fill(0.25);
        });
        access.suspend();
        let mut output = [1.; 256];
        alloc_counter::deny_alloc(|| {
            assert_eq!(
                access.render_interleaved_f32(&mut output),
                AudioRenderStatus::Continue
            );
        });
        assert_eq!(renders.load(Ordering::Acquire), 0);
        assert!(output.iter().all(|sample| *sample == 0.));
        access.resume();
        assert_eq!(
            access.render_interleaved_f32(&mut output),
            AudioRenderStatus::Continue
        );
        assert_eq!(renders.load(Ordering::Acquire), 1);

        owner.begin_shutdown();
        bridge.close();
        access.report_endpoint_death(AudioOutputDeathReason::DeviceUnavailable);
        assert!(watcher.death_reason().is_none());
        SystemRenderBridge::try_retire(bridge)
            .map_err(|_| ())
            .unwrap();
        owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }

    #[test]
    fn fake_owner_contains_build_play_and_early_error_failures() {
        let cases = [
            (FakeBuild::Error, true, None, 0),
            (
                FakeBuild::Panic,
                false,
                Some(AudioOutputDeathReason::BackendFailure),
                0,
            ),
            (FakeBuild::RenderThenError, true, None, 0),
            (
                FakeBuild::RenderThenPanic,
                false,
                Some(AudioOutputDeathReason::BackendFailure),
                0,
            ),
            (
                FakeBuild::Stream {
                    play: vec![FakeAction::DeviceUnavailable],
                    pause: Vec::new(),
                    drop_block: None,
                },
                true,
                None,
                1,
            ),
            (
                FakeBuild::Stream {
                    play: vec![FakeAction::Panic],
                    pause: Vec::new(),
                    drop_block: None,
                },
                false,
                Some(AudioOutputDeathReason::BackendFailure),
                1,
            ),
            (
                FakeBuild::Stream {
                    play: vec![FakeAction::RenderThenDeviceUnavailable],
                    pause: Vec::new(),
                    drop_block: None,
                },
                true,
                None,
                1,
            ),
            (
                FakeBuild::Stream {
                    play: vec![FakeAction::RenderThenPanic],
                    pause: Vec::new(),
                    drop_block: None,
                },
                false,
                Some(AudioOutputDeathReason::BackendFailure),
                1,
            ),
            (
                FakeBuild::Stream {
                    play: vec![FakeAction::ReportDeviceDeath],
                    pause: Vec::new(),
                    drop_block: None,
                },
                true,
                Some(AudioOutputDeathReason::DeviceUnavailable),
                1,
            ),
        ];

        for (build, cleanup_ok, expected_death, expected_drops) in cases {
            let probes = Arc::new(FakeProbes::default());
            let prepared = fake_prepared(build, Arc::clone(&probes));
            let renders = Arc::new(AtomicUsize::new(0));
            let (owner, callback, events, watcher) =
                counted_raw_callback(prepared.config().format(), Arc::clone(&renders));
            let failure = match Box::new(prepared).start(callback, events) {
                Ok(_) => panic!("forced CPAL startup failure unexpectedly succeeded"),
                Err(failure) => failure,
            };
            let access = match probes.access.lock() {
                Ok(access) => access.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            }
            .expect("the fake builder must observe exact callback access");
            assert!(
                access.is_closed(),
                "failure must close callback access before publishing its response"
            );
            assert_eq!(
                renders.load(Ordering::Acquire),
                0,
                "provisional CPAL build/play callbacks must stay silent"
            );
            assert_eq!(
                Arc::strong_count(owner.slot.as_ref().unwrap()),
                2,
                "start failure must retain the callback until cleanup is polled"
            );
            owner.begin_shutdown();
            let (_, shutdown) = failure.into_parts();
            assert_eq!(executor::block_on(shutdown).is_ok(), cleanup_ok);
            assert_eq!(watcher.death_reason(), expected_death);
            assert_eq!(probes.drops.load(Ordering::Acquire), expected_drops);
            assert_eq!(
                Arc::strong_count(owner.slot.as_ref().unwrap()),
                1,
                "joined CPAL cleanup must destroy the callback even when the owner panicked"
            );
            if cleanup_ok {
                owner
                    .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
                    .map_err(|_| ())
                    .unwrap()
                    .unwrap();
            } else {
                drop(owner);
            }
        }
    }

    #[test]
    fn fake_running_shutdown_wakes_only_after_stream_drop_and_callback_retirement() {
        #[derive(Default)]
        struct WakeCounter(AtomicUsize);
        impl ArcWake for WakeCounter {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.0.fetch_add(1, Ordering::AcqRel);
            }
        }

        let (drop_entered_send, drop_entered_recv) = crossbeam_channel::bounded(1);
        let (drop_release_send, drop_release_recv) = crossbeam_channel::bounded(1);
        let probes = Arc::new(FakeProbes::default());
        let prepared = fake_prepared(
            FakeBuild::Stream {
                play: vec![FakeAction::Ok],
                pause: Vec::new(),
                drop_block: Some(FakeDropBlock {
                    entered: drop_entered_send,
                    release: drop_release_recv,
                }),
            },
            Arc::clone(&probes),
        );
        let renders = Arc::new(AtomicUsize::new(0));
        let (owner, callback, events, watcher) =
            counted_raw_callback(prepared.config().format(), Arc::clone(&renders));
        let running = Box::new(prepared).start(callback, events).unwrap();
        let access = match probes.access.lock() {
            Ok(access) => access.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
        .unwrap();
        let mut output = [0.; 256];
        assert_eq!(
            access.render_interleaved_f32(&mut output),
            AudioRenderStatus::Continue
        );
        assert_eq!(renders.load(Ordering::Acquire), 1);
        owner.begin_shutdown();
        let mut shutdown = Box::pin(running.shutdown());
        let wakes = Arc::new(WakeCounter::default());
        let task_waker = waker(Arc::clone(&wakes));
        let mut cx = Context::from_waker(&task_waker);
        assert!(matches!(shutdown.as_mut().poll(&mut cx), Poll::Pending));
        drop_entered_recv
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(probes.drops.load(Ordering::Acquire), 1);
        assert_eq!(Arc::strong_count(owner.slot.as_ref().unwrap()), 2);
        assert!(matches!(shutdown.as_mut().poll(&mut cx), Poll::Pending));

        drop_release_send.send(()).unwrap();
        let started = std::time::Instant::now();
        while wakes.0.load(Ordering::Acquire) == 0 {
            assert!(started.elapsed() < Duration::from_secs(1));
            thread::yield_now();
        }
        executor::block_on(shutdown).unwrap();
        assert_eq!(Arc::strong_count(owner.slot.as_ref().unwrap()), 1);
        assert!(watcher.death_reason().is_none());
        owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }

    #[test]
    fn fake_state_driver_covers_hardware_and_software_pause() {
        for software in [false, true] {
            let probes = Arc::new(FakeProbes::default());
            let prepared = fake_prepared(
                FakeBuild::Stream {
                    play: if software {
                        vec![FakeAction::Ok]
                    } else {
                        vec![FakeAction::Ok, FakeAction::Ok]
                    },
                    pause: vec![if software {
                        FakeAction::Unsupported
                    } else {
                        FakeAction::Ok
                    }],
                    drop_block: None,
                },
                Arc::clone(&probes),
            );
            let (owner, callback, events, watcher) = raw_callback(prepared.config().format());
            let mut running = Box::new(prepared).start(callback, events).unwrap();
            running.suspend().unwrap();
            assert_eq!(probes.pauses.load(Ordering::Acquire), 1);
            if software {
                let access = match probes.access.lock() {
                    Ok(access) => access.clone(),
                    Err(poisoned) => poisoned.into_inner().clone(),
                }
                .unwrap();
                let mut output = [1.; 256];
                assert_eq!(
                    access.render_interleaved_f32(&mut output),
                    AudioRenderStatus::Continue
                );
                assert!(output.iter().all(|sample| *sample == 0.));
            }
            running.resume().unwrap();
            assert_eq!(
                probes.plays.load(Ordering::Acquire),
                if software { 1 } else { 2 }
            );
            owner.begin_shutdown();
            executor::block_on(running.shutdown()).unwrap();
            assert!(watcher.death_reason().is_none());
            owner
                .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
                .map_err(|_| ())
                .unwrap()
                .unwrap();
        }
    }

    #[test]
    fn fake_pause_and_resume_errors_return_stream_for_quarantine() {
        for fail_resume in [false, true] {
            for failure in [FakeAction::DeviceUnavailable, FakeAction::Panic] {
                let format = AudioRenderFormat::new(48_000., 2, 128).unwrap();
                let (owner, bridge, access, watcher) = test_bridge(format, |_| {});
                let probes = Arc::new(FakeProbes::default());
                let stream = FakeStream::new(
                    access.clone(),
                    if fail_resume {
                        vec![failure]
                    } else {
                        Vec::new()
                    },
                    vec![if fail_resume { FakeAction::Ok } else { failure }],
                    Arc::clone(&probes),
                    None,
                );
                let (command_send, command_recv) = crossbeam_channel::unbounded();
                let (suspend_send, suspend_recv) = crossbeam_channel::bounded(1);
                command_send
                    .send(CpalCommand::Suspend(suspend_send))
                    .unwrap();
                let resume_recv = if fail_resume {
                    let (resume_send, resume_recv) = crossbeam_channel::bounded(1);
                    command_send.send(CpalCommand::Resume(resume_send)).unwrap();
                    Some(resume_recv)
                } else {
                    None
                };
                drop(command_send);
                let outcome = drive_started(stream, command_recv, AccessCloseGuard::new(access));
                if fail_resume {
                    suspend_recv.recv().unwrap().unwrap();
                    assert!(resume_recv.unwrap().recv().unwrap().is_err());
                } else {
                    assert!(suspend_recv.recv().unwrap().is_err());
                }
                let StartedOutcome::Quarantine(stream) = outcome else {
                    panic!(
                        "uncertain CPAL state error retired the stream instead of quarantining it"
                    )
                };
                assert_eq!(
                    watcher.death_reason(),
                    Some(if matches!(failure, FakeAction::Panic) {
                        AudioOutputDeathReason::BackendFailure
                    } else {
                        AudioOutputDeathReason::DeviceUnavailable
                    })
                );
                drop(stream);
                assert_eq!(probes.drops.load(Ordering::Acquire), 1);
                owner.begin_shutdown();
                bridge.close();
                SystemRenderBridge::try_retire(bridge)
                    .map_err(|_| ())
                    .unwrap();
                owner
                    .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
                    .map_err(|_| ())
                    .unwrap()
                    .unwrap();
            }
        }
    }
}
