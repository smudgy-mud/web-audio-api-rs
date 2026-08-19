//! Hosted Cubeb endpoint adapter.
//!
//! The crate owner thread retains Cubeb's thread-affine context and stream. Cubeb callback boxes
//! capture only Weak render access. This is essential because Cubeb 0.34's high-level stream
//! builder does not reclaim its callback box when native stream initialization fails; even such a
//! leaked closure cannot retain the exact `AudioRenderCallback` or its event producer authority.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread;

use cubeb::{
    ChannelLayout, Context, DeviceId, DeviceState, DeviceType, Error as CubebError, State, Stream,
    StreamBuilder, StreamParams, StreamParamsBuilder,
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

const MAX_CUBEB_CHANNELS: usize = 8;
const MIN_CUBEB_SAMPLE_RATE: u32 = 1_000;
const MAX_CUBEB_SAMPLE_RATE: u32 = 384_000;

#[cfg(target_os = "windows")]
struct CubebOwnerPlatformGuard;

#[cfg(target_os = "windows")]
impl CubebOwnerPlatformGuard {
    fn enter() -> Result<Self, AudioOutputError> {
        use std::ffi::c_void;

        const COINIT_MULTITHREADED: u32 = 0;
        const S_OK: i32 = 0;
        const S_FALSE: i32 = 1;

        #[link(name = "ole32")]
        extern "system" {
            fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> i32;
        }

        // SAFETY: this owner thread has not called Cubeb yet. Both successful HRESULTs require
        // one matching CoUninitialize, owned by this non-Clone same-thread guard.
        let result = unsafe { CoInitializeEx(std::ptr::null_mut(), COINIT_MULTITHREADED) };
        if matches!(result, S_OK | S_FALSE) {
            Ok(Self)
        } else {
            Err(AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                format!("Cubeb output owner could not enter the Windows MTA (HRESULT {result:#x})"),
            ))
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for CubebOwnerPlatformGuard {
    fn drop(&mut self) {
        #[link(name = "ole32")]
        extern "system" {
            fn CoUninitialize();
        }

        // SAFETY: `enter` succeeded on this same owner thread. The guard is declared before every
        // Cubeb owner local, so Stream and Context destruction precede this matching uninitialize.
        unsafe { CoUninitialize() };
    }
}

#[cfg(not(target_os = "windows"))]
struct CubebOwnerPlatformGuard;

#[cfg(not(target_os = "windows"))]
impl CubebOwnerPlatformGuard {
    fn enter() -> Result<Self, AudioOutputError> {
        Ok(Self)
    }
}

pub(super) fn prepare(
    request: &AudioOutputRequest,
) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
    let request = request.clone();
    let (command_send, command_recv) = crossbeam_channel::bounded(1);
    let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
    let panic_events = Arc::new(Mutex::new(None));
    let owner_panic_events = Arc::clone(&panic_events);
    let owner = thread::Builder::new()
        .name("web-audio-system-output-cubeb".into())
        .spawn(move || {
            let panic_ready = ready_send.clone();
            let result = contained_owner(&owner_panic_events, || {
                let _platform = match CubebOwnerPlatformGuard::enter() {
                    Ok(platform) => platform,
                    Err(error) => {
                        let _ = ready_send.send(Err(error.clone()));
                        return Err(error);
                    }
                };
                let prepared = match prepare_context(&request) {
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
                format!("failed to spawn Cubeb system output owner: {error}"),
            )
        })?;

    let observed = match install_join_observer(owner) {
        Ok(observed) => observed,
        Err(failure) => {
            let _ = command_send.try_send(CubebCommand::Abort);
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
            let _ = command_send.try_send(CubebCommand::Abort);
            let joined = observed.blocking_join.recv().ok();
            return Err(joined.and_then(Result::err).unwrap_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "Cubeb system output owner exited before preparation completed",
                )
            }));
        }
    };

    Ok(Box::new(CubebPrepared {
        config,
        command_send: Some(command_send),
        completion: Some(observed.completion),
    }))
}

struct PreparedContext {
    context: Context,
    params: StreamParams,
    latency_frames: u32,
    device: Option<DeviceId>,
    config: AudioOutputConfig,
}

fn prepare_context(request: &AudioOutputRequest) -> Result<PreparedContext, AudioOutputError> {
    let context = Context::init(None, None)
        .map_err(|error| map_cubeb_error("initialize output context", error))?;
    let channels = cubeb_channel_count(request.number_of_channels())?;

    let selected = select_device(&context, request.sink_id())?;
    let maximum = selected
        .as_ref()
        .map_or_else(
            || context.max_channel_count(),
            |facts| Ok(facts.max_channels),
        )
        .map_err(|error| map_cubeb_error("query maximum output channels", error))?;
    if channels > maximum {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("Cubeb output supports at most {maximum} channels, requested {channels}"),
        ));
    }

    let sample_rate = match request.requested_sample_rate() {
        Some(rate) => {
            let rate = supported_sample_rate(rate)?;
            if selected
                .as_ref()
                .is_some_and(|facts| rate < facts.min_rate || rate > facts.max_rate)
            {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::NotSupported,
                    format!("Cubeb does not support requested physical sample rate {rate} Hz"),
                ));
            }
            rate
        }
        None => selected.as_ref().map_or_else(
            || {
                context
                    .preferred_sample_rate()
                    .map_err(|error| map_cubeb_error("query preferred sample rate", error))
            },
            |facts| Ok(facts.default_rate),
        )?,
    };
    if sample_rate == 0 {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "Cubeb reported a zero output sample rate",
        ));
    }
    if !(MIN_CUBEB_SAMPLE_RATE..=MAX_CUBEB_SAMPLE_RATE).contains(&sample_rate) {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("Cubeb reported unsupported output sample rate {sample_rate} Hz"),
        ));
    }

    let params = StreamParamsBuilder::new()
        .format(cubeb::SampleFormat::Float32NE)
        .rate(sample_rate)
        .channels(channels)
        .layout(channel_layout(channels))
        .take();
    let requested_latency = latency_frames(request.latency_hint(), sample_rate)?;
    let minimum = context
        .min_latency(&params)
        .map_err(|error| map_cubeb_error("query minimum output latency", error))?;
    if !(1..=96_000).contains(&minimum) {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("Cubeb reported unsupported minimum output latency {minimum} frames"),
        ));
    }
    let latency_frames = requested_latency.max(minimum);
    let format = AudioRenderFormat::new(
        sample_rate as f32,
        request.number_of_channels(),
        MAX_AUDIO_OUTPUT_CALLBACK_FRAMES,
    )?;
    let accepted_sink = if request.sink_id().is_empty() {
        String::new()
    } else {
        request.sink_id().to_owned()
    };
    let config = AudioOutputConfig::new(
        format,
        accepted_sink,
        f64::from(latency_frames) / f64::from(sample_rate),
    )?;
    request.validate_config(&config)?;

    Ok(PreparedContext {
        context,
        params,
        latency_frames,
        device: selected.map(|selected| selected.id),
        config,
    })
}

fn cubeb_channel_count(channels: usize) -> Result<u32, AudioOutputError> {
    if !(1..=MAX_CUBEB_CHANNELS).contains(&channels) {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "Cubeb supports hosted output channel counts from 1 through 8",
        ));
    }
    u32::try_from(channels).map_err(|_| {
        AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "requested channel count cannot be represented by Cubeb",
        )
    })
}

struct DeviceFacts {
    id: DeviceId,
    max_channels: u32,
    default_rate: u32,
    min_rate: u32,
    max_rate: u32,
}

fn select_device(
    context: &Context,
    sink_id: &str,
) -> Result<Option<DeviceFacts>, AudioOutputError> {
    if sink_id.is_empty() {
        return Ok(None);
    }
    let devices = context
        .enumerate_devices(DeviceType::OUTPUT)
        .map_err(|error| map_cubeb_error("enumerate output devices", error))?;
    let mut seen = Vec::new();
    for device in devices.iter() {
        let id = stable_device_id(device, &seen)?;
        if id == sink_id || device.device_id() == Some(sink_id) {
            if device.state() != DeviceState::Enabled {
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::DeviceUnavailable,
                    format!("Cubeb output device {sink_id:?} is not enabled"),
                ));
            }
            return Ok(Some(DeviceFacts {
                id: device.devid(),
                max_channels: device.max_channels(),
                default_rate: device.default_rate(),
                min_rate: device.min_rate(),
                max_rate: device.max_rate(),
            }));
        }
        seen.push(id);
    }
    Err(AudioOutputError::new(
        AudioOutputErrorKind::DeviceUnavailable,
        format!("Cubeb output device {sink_id:?} is unavailable"),
    ))
}

fn stable_device_id(
    device: &cubeb::DeviceInfo,
    seen: &[String],
) -> Result<String, AudioOutputError> {
    let name = device.friendly_name().ok_or_else(|| {
        AudioOutputError::new(
            AudioOutputErrorKind::BackendSpecific,
            "Cubeb output device has no friendly name",
        )
    })?;
    let channels = u16::try_from(device.max_channels()).map_err(|_| {
        AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "Cubeb output device channel count exceeds the public identifier format",
        )
    })?;
    let mut index = 0;
    loop {
        let id = crate::media_devices::DeviceId::as_string(
            MediaDeviceInfoKind::AudioOutput,
            "cubeb".to_owned(),
            name.to_owned(),
            channels,
            index,
        );
        if !seen.iter().any(|seen| seen == &id) {
            return Ok(id);
        }
        index += 1;
    }
}

fn exact_sample_rate(rate: f32) -> Result<u32, AudioOutputError> {
    if rate.fract() != 0. || !(1. ..=u32::MAX as f32).contains(&rate) {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("Cubeb requires an integer physical sample rate, got {rate}"),
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let integer = rate as u32;
    if integer as f32 != rate {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("Cubeb cannot exactly represent physical sample rate {rate}"),
        ));
    }
    Ok(integer)
}

fn supported_sample_rate(rate: f32) -> Result<u32, AudioOutputError> {
    let rate = exact_sample_rate(rate)?;
    if !(MIN_CUBEB_SAMPLE_RATE..=MAX_CUBEB_SAMPLE_RATE).contains(&rate) {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            format!("Cubeb does not support requested physical sample rate {rate} Hz"),
        ));
    }
    Ok(rate)
}

fn latency_frames(
    latency: AudioContextLatencyCategory,
    sample_rate: u32,
) -> Result<u32, AudioOutputError> {
    let frames = match latency {
        AudioContextLatencyCategory::Interactive => RENDER_QUANTUM_SIZE as f64,
        AudioContextLatencyCategory::Balanced => (RENDER_QUANTUM_SIZE * 4) as f64,
        AudioContextLatencyCategory::Playback => (RENDER_QUANTUM_SIZE * 8) as f64,
        AudioContextLatencyCategory::Custom(seconds) => seconds * f64::from(sample_rate),
    };
    if !frames.is_finite() || frames <= 0. || frames > 96_000. {
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "requested Cubeb output latency is outside the supported frame range",
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let frames = frames.ceil() as u32;
    Ok(frames.max(1))
}

const fn channel_layout(channels: u32) -> ChannelLayout {
    match channels {
        1 => ChannelLayout::MONO,
        2 => ChannelLayout::STEREO,
        4 => ChannelLayout::QUAD,
        _ => ChannelLayout::UNDEFINED,
    }
}

enum CubebCommand {
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
    prepared: PreparedContext,
    command_recv: crossbeam_channel::Receiver<CubebCommand>,
    panic_events: &Mutex<Option<AudioOutputEventSink>>,
) -> OwnerResult {
    let PreparedContext {
        context,
        params,
        latency_frames,
        device,
        ..
    } = prepared;
    run_owner_with(command_recv, panic_events, move |access| {
        build_stream(context, params, latency_frames, device, access)
    })
}

fn run_owner_with<S, F>(
    command_recv: crossbeam_channel::Receiver<CubebCommand>,
    panic_events: &Mutex<Option<AudioOutputEventSink>>,
    build: F,
) -> OwnerResult
where
    S: CubebStreamControl,
    F: FnOnce(SystemRenderAccess) -> Result<S, AudioOutputError>,
{
    let first = match command_recv.recv() {
        Ok(command) => command,
        Err(_) => return Ok(()),
    };
    let CubebCommand::Start {
        access,
        events,
        ready,
        accept,
        started,
    } = first
    else {
        return match first {
            CubebCommand::Abort | CubebCommand::Shutdown => Ok(()),
            CubebCommand::Suspend(response) | CubebCommand::Resume(response) => {
                let error = AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "Cubeb system output received state work before start",
                );
                let _ = response.send(Err(error.clone()));
                Err(error)
            }
            CubebCommand::Start { .. } => unreachable!(),
        };
    };
    retain_panic_events(panic_events, events);
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
    match panic::catch_unwind(AssertUnwindSafe(|| stream.start())) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let mapped = map_cubeb_error("start output stream", error);
            close_guard.close();
            drop(stream);
            let _ = ready.send(Err(mapped));
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
                "Cubeb output panicked while starting its stream",
            ));
        }
    }
    if close_guard.access().is_closed() {
        close_guard.close();
        drop(stream);
        let _ = ready.send(Err(AudioOutputError::new(
            AudioOutputErrorKind::BackendSpecific,
            "Cubeb output failed before startup completed",
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
    if close_guard.access().is_closed() {
        drop(stream);
        clear_panic_events(panic_events);
        return Err(AudioOutputError::new(
            AudioOutputErrorKind::BackendSpecific,
            "Cubeb output failed at the accepted start boundary",
        ));
    }
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

trait CubebStreamControl: 'static {
    fn start(&self) -> Result<(), CubebError>;
    fn stop(&self) -> Result<(), CubebError>;
}

impl<F: 'static> CubebStreamControl for Stream<F> {
    fn start(&self) -> Result<(), CubebError> {
        cubeb::StreamRef::start(self)
    }

    fn stop(&self) -> Result<(), CubebError> {
        cubeb::StreamRef::stop(self)
    }
}

fn build_stream(
    context: Context,
    params: StreamParams,
    latency_frames: u32,
    device: Option<DeviceId>,
    access: SystemRenderAccess,
) -> Result<NativeCubebStream, AudioOutputError> {
    macro_rules! build {
        ($channels:literal) => {
            build_stream_for_channels::<$channels>(
                &context,
                &params,
                latency_frames,
                device,
                access.clone(),
            )
        };
    }
    let stream = match params.channels() {
        1 => build!(1),
        2 => build!(2),
        3 => build!(3),
        4 => build!(4),
        5 => build!(5),
        6 => build!(6),
        7 => build!(7),
        8 => build!(8),
        _ => Err(AudioOutputError::new(
            AudioOutputErrorKind::NotSupported,
            "Cubeb stream channel count is outside the hosted limit",
        )),
    }?;
    Ok(NativeCubebStream {
        stream,
        _context: context,
    })
}

fn build_stream_for_channels<const N: usize>(
    context: &Context,
    params: &StreamParams,
    latency_frames: u32,
    device: Option<DeviceId>,
    access: SystemRenderAccess,
) -> Result<BoxedCubebStream, AudioOutputError> {
    let mut builder = StreamBuilder::<[f32; N]>::new();
    match device {
        Some(device) => builder.output(device, params),
        None => builder.default_output(params),
    };
    let state_access = access.clone();
    builder
        .name("Cubeb web_audio_api hosted")
        .latency(latency_frames)
        .data_callback(move |_input, output| process_output::<N>(output, &access))
        .state_callback(move |value| handle_state(value, &state_access));
    let stream = builder
        .init(context)
        .map_err(|error| map_cubeb_error("initialize output stream", error))?;
    Ok(BoxedCubebStream(Box::new(stream)))
}

fn handle_state(state: State, access: &SystemRenderAccess) {
    match state {
        // Cubeb state delivery is asynchronous, so a delayed Stopped cannot be safely paired to a
        // particular owner transition without a backend acknowledgement generation. Owner RPC
        // results are authoritative for Start/Stop; only terminal Cubeb states latch endpoint death.
        State::Started | State::Stopped => {}
        State::Drained | State::Error => {
            access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
        }
    }
}

fn process_output<const N: usize>(output: &mut [[f32; N]], access: &SystemRenderAccess) -> isize {
    if output.is_empty() {
        access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        return 0;
    }
    let requested = output.len();
    let mut rendered = 0usize;
    for frames in output.chunks_mut(MAX_AUDIO_OUTPUT_CALLBACK_FRAMES) {
        let samples = unsafe {
            // SAFETY: arrays are contiguous and this slice spans exactly `frames.len() * N`
            // initialized `f32` elements belonging to the mutable frame slice.
            std::slice::from_raw_parts_mut(frames.as_mut_ptr().cast(), frames.len() * N)
        };
        match access.render_interleaved_f32(samples) {
            AudioRenderStatus::Continue => rendered += frames.len(),
            AudioRenderStatus::Stop => {
                rendered += frames.len();
                break;
            }
        }
    }
    for remaining in output[rendered..].iter_mut() {
        remaining.fill(0.);
    }
    isize::try_from(requested).unwrap_or_else(|_| {
        access.report_endpoint_death(AudioOutputDeathReason::CallbackProtocolViolation);
        0
    })
}

struct BoxedCubebStream(Box<dyn CubebStreamControl>);

impl CubebStreamControl for BoxedCubebStream {
    fn start(&self) -> Result<(), CubebError> {
        self.0.start()
    }

    fn stop(&self) -> Result<(), CubebError> {
        self.0.stop()
    }
}

struct NativeCubebStream {
    // Field order is the destruction proof: Stream::drop performs stop+destroy and reclaims the
    // non-leaked callback box before the thread-affine Context is destroyed.
    stream: BoxedCubebStream,
    _context: Context,
}

impl CubebStreamControl for NativeCubebStream {
    fn start(&self) -> Result<(), CubebError> {
        self.stream.start()
    }

    fn stop(&self) -> Result<(), CubebError> {
        self.stream.stop()
    }
}

enum StartedOutcome<S> {
    Retired(OwnerResult),
    Quarantine(S),
}

fn drive_started<S: CubebStreamControl>(
    stream: S,
    command_recv: crossbeam_channel::Receiver<CubebCommand>,
    close_guard: AccessCloseGuard,
) -> StartedOutcome<S> {
    let access = close_guard.access();
    let mut running = true;
    loop {
        match command_recv.recv() {
            Ok(CubebCommand::Suspend(response)) if !running => {
                let _ = response.send(Ok(()));
            }
            Ok(CubebCommand::Suspend(response)) => {
                // Silence logical callbacks before Cubeb begins its asynchronous stop transition.
                access.suspend();
                match panic::catch_unwind(AssertUnwindSafe(|| stream.stop())) {
                    Ok(Ok(())) => {
                        running = false;
                        let _ = response.send(Ok(()));
                    }
                    Ok(Err(error)) => {
                        let mapped = map_cubeb_error("suspend output stream", error);
                        access.report_endpoint_death(death_reason(error));
                        let _ = response.send(Err(mapped));
                        return StartedOutcome::Quarantine(stream);
                    }
                    Err(payload) => {
                        access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                        std::mem::forget(payload);
                        let _ = response.send(Err(AudioOutputError::new(
                            AudioOutputErrorKind::BackendSpecific,
                            "Cubeb output panicked while suspending its stream",
                        )));
                        return StartedOutcome::Quarantine(stream);
                    }
                }
            }
            Ok(CubebCommand::Resume(response)) if running => {
                let _ = response.send(Ok(()));
            }
            Ok(CubebCommand::Resume(response)) => {
                match panic::catch_unwind(AssertUnwindSafe(|| stream.start())) {
                    Ok(Ok(())) => {
                        access.resume();
                        running = true;
                        let _ = response.send(Ok(()));
                    }
                    Ok(Err(error)) => {
                        let mapped = map_cubeb_error("resume output stream", error);
                        access.report_endpoint_death(death_reason(error));
                        let _ = response.send(Err(mapped));
                        return StartedOutcome::Quarantine(stream);
                    }
                    Err(payload) => {
                        access.report_endpoint_death(AudioOutputDeathReason::BackendFailure);
                        std::mem::forget(payload);
                        let _ = response.send(Err(AudioOutputError::new(
                            AudioOutputErrorKind::BackendSpecific,
                            "Cubeb output panicked while resuming its stream",
                        )));
                        return StartedOutcome::Quarantine(stream);
                    }
                }
            }
            Ok(CubebCommand::Shutdown | CubebCommand::Abort) => {
                access.close();
                drop(stream);
                return StartedOutcome::Retired(Ok(()));
            }
            Ok(CubebCommand::Start { ready, .. }) => {
                let error = AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "Cubeb system output received a duplicate start",
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
                    "Cubeb system output controller disconnected",
                )));
            }
        }
    }
}

fn quarantine_stream<S: CubebStreamControl>(stream: S) -> ! {
    let _stream = std::mem::ManuallyDrop::new(stream);
    loop {
        thread::park();
    }
}

fn retain_panic_events(slot: &Mutex<Option<AudioOutputEventSink>>, events: AudioOutputEventSink) {
    match slot.lock() {
        Ok(mut retained) => *retained = Some(events),
        Err(poisoned) => *poisoned.into_inner() = Some(events),
    }
}

fn clear_panic_events(events: &Mutex<Option<AudioOutputEventSink>>) {
    match events.lock() {
        Ok(mut events) => *events = None,
        Err(poisoned) => *poisoned.into_inner() = None,
    }
}

fn death_reason(error: CubebError) -> AudioOutputDeathReason {
    match error {
        CubebError::DeviceUnavailable => AudioOutputDeathReason::DeviceUnavailable,
        _ => AudioOutputDeathReason::BackendFailure,
    }
}

fn map_cubeb_error(operation: &str, error: CubebError) -> AudioOutputError {
    let kind = match error {
        CubebError::DeviceUnavailable => AudioOutputErrorKind::DeviceUnavailable,
        CubebError::NotSupported | CubebError::InvalidFormat => AudioOutputErrorKind::NotSupported,
        CubebError::InvalidParameter => AudioOutputErrorKind::InvalidArgument,
        CubebError::Error => AudioOutputErrorKind::BackendSpecific,
    };
    AudioOutputError::new(kind, format!("Cubeb failed to {operation}: {error}"))
}

struct CubebPrepared {
    config: AudioOutputConfig,
    command_send: Option<crossbeam_channel::Sender<CubebCommand>>,
    completion: Option<OwnerCompletion>,
}

impl fmt::Debug for CubebPrepared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CubebPrepared")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PreparedAudioOutput for CubebPrepared {
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
            .expect("prepared Cubeb output is single-use");
        let completion = self
            .completion
            .take()
            .expect("prepared Cubeb output is single-use");
        let bridge = SystemRenderBridge::new(callback, events);
        let access = SystemRenderBridge::access(&bridge);
        access.suspend();
        let owner_events = bridge.events();
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let (accept_send, accept_recv) = crossbeam_channel::bounded(1);
        let (started_send, started_recv) = crossbeam_channel::bounded(1);
        if let Err(error) = command_send.send(CubebCommand::Start {
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
                    "Cubeb system output owner disappeared during start",
                ),
                completion.into_bridge_shutdown(bridge),
            ));
        }
        match ready_recv.recv() {
            Ok(Ok(())) => {
                if accept_send.send(()).is_ok() && started_recv.recv().is_ok() {
                    Ok(Box::new(CubebRunning {
                        command_send: Some(command_send),
                        completion: Some(completion),
                        bridge: Some(bridge),
                    }))
                } else {
                    bridge.close();
                    Err(AudioOutputStartFailure::new(
                        AudioOutputError::new(
                            AudioOutputErrorKind::BackendSpecific,
                            "Cubeb system output owner failed at the accepted start boundary",
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
                        "Cubeb system output owner exited during start",
                    ),
                    completion.into_bridge_shutdown(bridge),
                ))
            }
        }
    }

    fn abort(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CubebCommand::Abort);
        }
        self.completion.take().map_or_else(
            || missing_completion("prepared abort"),
            OwnerCompletion::into_shutdown,
        )
    }
}

impl Drop for CubebPrepared {
    fn drop(&mut self) {
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CubebCommand::Abort);
        }
    }
}

struct CubebRunning {
    command_send: Option<crossbeam_channel::Sender<CubebCommand>>,
    completion: Option<OwnerCompletion>,
    bridge: Option<Arc<SystemRenderBridge>>,
}

impl fmt::Debug for CubebRunning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CubebRunning").finish_non_exhaustive()
    }
}

impl CubebRunning {
    fn state_command(
        &self,
        command: impl FnOnce(crossbeam_channel::Sender<OwnerResult>) -> CubebCommand,
    ) -> OwnerResult {
        let (response_send, response_recv) = crossbeam_channel::bounded(1);
        self.command_send
            .as_ref()
            .ok_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "Cubeb system output no longer owns its controller",
                )
            })?
            .send(command(response_send))
            .map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "Cubeb system output owner disconnected",
                )
            })?;
        response_recv.recv().map_err(|_| {
            AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "Cubeb system output owner exited during a state transition",
            )
        })?
    }
}

impl RunningAudioOutput for CubebRunning {
    fn resume(&mut self) -> OwnerResult {
        self.state_command(CubebCommand::Resume)
    }

    fn suspend(&mut self) -> OwnerResult {
        self.state_command(CubebCommand::Suspend)
    }

    fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        if let Some(bridge) = self.bridge.as_ref() {
            bridge.close();
        }
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CubebCommand::Shutdown);
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

impl Drop for CubebRunning {
    fn drop(&mut self) {
        if let Some(bridge) = self.bridge.as_ref() {
            bridge.close();
        }
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(CubebCommand::Shutdown);
        }
    }
}

fn missing_completion(operation: &str) -> AudioOutputEndpointShutdown {
    let operation = operation.to_owned();
    AudioOutputEndpointShutdown::from_future(async move {
        Err(AudioOutputError::new(
            AudioOutputErrorKind::Shutdown,
            format!("Cubeb system output lost completion ownership during {operation}"),
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use super::*;
    use crate::output::{audio_render_test_pair, AudioRenderOwner, EndpointShutdownConfirmed};
    use futures::executor;

    struct TestEndpoint {
        owner: AudioRenderOwner,
        bridge: Arc<SystemRenderBridge>,
        access: SystemRenderAccess,
        watcher: crate::output::AudioOutputEventWatcher,
        renders: Arc<AtomicUsize>,
    }

    fn endpoint(channels: usize) -> TestEndpoint {
        let format =
            AudioRenderFormat::new(48_000., channels, MAX_AUDIO_OUTPUT_CALLBACK_FRAMES).unwrap();
        let (events, watcher) = AudioOutputEventSink::bounded(8);
        let renders = Arc::new(AtomicUsize::new(0));
        let callback_renders = Arc::clone(&renders);
        let (owner, callback) = audio_render_test_pair(
            format,
            events.clone(),
            move |output| {
                callback_renders.fetch_add(1, AtomicOrdering::AcqRel);
                output.fill(0.25);
            },
            || Ok(()),
        );
        let bridge = SystemRenderBridge::new(callback, events);
        let access = SystemRenderBridge::access(&bridge);
        TestEndpoint {
            owner,
            bridge,
            access,
            watcher,
            renders,
        }
    }

    fn retire(endpoint: TestEndpoint) {
        endpoint.owner.begin_shutdown();
        endpoint.bridge.close();
        assert!(SystemRenderBridge::try_retire(endpoint.bridge).is_ok());
        endpoint
            .owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }

    #[test]
    fn planner_enforces_cubeb_abi_limits() {
        assert_eq!(cubeb_channel_count(1).unwrap(), 1);
        assert_eq!(cubeb_channel_count(8).unwrap(), 8);
        assert_eq!(
            cubeb_channel_count(9).unwrap_err().kind(),
            AudioOutputErrorKind::NotSupported
        );
        assert_eq!(exact_sample_rate(1_000.).unwrap(), 1_000);
        assert_eq!(exact_sample_rate(384_000.).unwrap(), 384_000);
        assert!(exact_sample_rate(48_000.5).is_err());
        assert!(supported_sample_rate(999.).is_err());
        assert!(supported_sample_rate(384_001.).is_err());
        assert_eq!(channel_layout(1), ChannelLayout::MONO);
        assert_eq!(channel_layout(2), ChannelLayout::STEREO);
        assert_eq!(channel_layout(4), ChannelLayout::QUAD);
        assert_eq!(channel_layout(8), ChannelLayout::UNDEFINED);
        assert_eq!(
            latency_frames(AudioContextLatencyCategory::Custom(2.), 48_000).unwrap(),
            96_000
        );
        assert!(latency_frames(AudioContextLatencyCategory::Custom(2.1), 48_000).is_err());
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn owner_platform_guard_balances_nested_mta_initialization_on_a_fresh_thread() {
        thread::spawn(|| {
            let first = CubebOwnerPlatformGuard::enter().unwrap();
            let second = CubebOwnerPlatformGuard::enter().unwrap();
            drop(second);
            drop(first);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn callback_preroll_is_silent_chunked_and_allocation_free() {
        let endpoint = endpoint(2);
        endpoint.access.suspend();
        let mut output = vec![[1.; 2]; MAX_AUDIO_OUTPUT_CALLBACK_FRAMES * 2 + 7];
        alloc_counter::deny_alloc(|| {
            assert_eq!(
                process_output::<2>(&mut output, &endpoint.access),
                output.len() as isize
            );
        });
        assert!(output.iter().flatten().all(|sample| *sample == 0.));
        assert_eq!(endpoint.renders.load(AtomicOrdering::Acquire), 0);

        endpoint.access.resume();
        alloc_counter::deny_alloc(|| {
            assert_eq!(
                process_output::<2>(&mut output, &endpoint.access),
                output.len() as isize
            );
        });
        assert!(output.iter().flatten().all(|sample| *sample == 0.25));
        assert_eq!(endpoint.renders.load(AtomicOrdering::Acquire), 3);

        endpoint.owner.begin_shutdown();
        output.fill([1.; 2]);
        assert_eq!(
            process_output::<2>(&mut output, &endpoint.access),
            output.len() as isize
        );
        assert!(output.iter().flatten().all(|sample| *sample == 0.));
        assert!(endpoint.watcher.death_reason().is_none());
        endpoint.bridge.close();
        assert!(SystemRenderBridge::try_retire(endpoint.bridge).is_ok());
        endpoint
            .owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }

    #[test]
    fn empty_callback_and_terminal_states_latch_once() {
        for state in [State::Error, State::Drained] {
            let endpoint = endpoint(2);
            handle_state(state, &endpoint.access);
            assert_eq!(
                endpoint.watcher.death_reason(),
                Some(AudioOutputDeathReason::BackendFailure)
            );
            retire(endpoint);
        }

        let endpoint = endpoint(2);
        handle_state(State::Started, &endpoint.access);
        handle_state(State::Stopped, &endpoint.access);
        assert!(endpoint.watcher.death_reason().is_none());
        let mut empty = Vec::<[f32; 2]>::new();
        assert_eq!(process_output::<2>(&mut empty, &endpoint.access), 0);
        assert_eq!(
            endpoint.watcher.death_reason(),
            Some(AudioOutputDeathReason::CallbackProtocolViolation)
        );
        retire(endpoint);
    }

    #[derive(Clone, Copy)]
    enum FakeAction {
        Ok,
        Err,
        Panic,
    }

    struct FakeStream {
        access: SystemRenderAccess,
        start: FakeAction,
        stop: FakeAction,
        dropped: Arc<AtomicBool>,
        require_closed_on_drop: bool,
    }

    impl FakeStream {
        fn act(action: FakeAction) -> Result<(), CubebError> {
            match action {
                FakeAction::Ok => Ok(()),
                FakeAction::Err => Err(CubebError::DeviceUnavailable),
                FakeAction::Panic => panic!("forced fake Cubeb panic"),
            }
        }
    }

    impl CubebStreamControl for FakeStream {
        fn start(&self) -> Result<(), CubebError> {
            Self::act(self.start)
        }

        fn stop(&self) -> Result<(), CubebError> {
            Self::act(self.stop)
        }
    }

    impl Drop for FakeStream {
        fn drop(&mut self) {
            if self.require_closed_on_drop {
                assert!(
                    self.access.is_closed(),
                    "Cubeb access must close before Stream drop"
                );
            }
            self.dropped.store(true, AtomicOrdering::Release);
        }
    }

    #[test]
    fn owner_two_phase_start_state_and_shutdown_are_ordered() {
        let endpoint = endpoint(2);
        endpoint.access.suspend();
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let (accept_send, accept_recv) = crossbeam_channel::bounded(1);
        let (started_send, started_recv) = crossbeam_channel::bounded(1);
        let (events, _) = AudioOutputEventSink::bounded(4);
        let panic_events = Arc::new(Mutex::new(None));
        let owner_events = Arc::clone(&panic_events);
        let access = endpoint.access.clone();
        let dropped = Arc::new(AtomicBool::new(false));
        let stream_dropped = Arc::clone(&dropped);
        let provisional_renders = Arc::clone(&endpoint.renders);
        let owner = thread::spawn(move || {
            run_owner_with(command_recv, &owner_events, move |build_access| {
                let mut preroll = [[1.; 2]; 4];
                assert_eq!(process_output::<2>(&mut preroll, &build_access), 4);
                assert!(preroll.iter().flatten().all(|sample| *sample == 0.));
                assert_eq!(provisional_renders.load(AtomicOrdering::Acquire), 0);
                Ok(FakeStream {
                    access: build_access,
                    start: FakeAction::Ok,
                    stop: FakeAction::Ok,
                    dropped: stream_dropped,
                    require_closed_on_drop: true,
                })
            })
        });
        command_send
            .send(CubebCommand::Start {
                access,
                events,
                ready: ready_send,
                accept: accept_recv,
                started: started_send,
            })
            .unwrap();
        ready_recv
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert_eq!(endpoint.renders.load(AtomicOrdering::Acquire), 0);
        accept_send.send(()).unwrap();
        started_recv.recv_timeout(Duration::from_secs(1)).unwrap();

        let (suspend_send, suspend_recv) = crossbeam_channel::bounded(1);
        command_send
            .send(CubebCommand::Suspend(suspend_send))
            .unwrap();
        suspend_recv
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        handle_state(State::Stopped, &endpoint.access);
        let (resume_send, resume_recv) = crossbeam_channel::bounded(1);
        command_send
            .send(CubebCommand::Resume(resume_send))
            .unwrap();
        resume_recv
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        handle_state(State::Started, &endpoint.access);
        handle_state(State::Stopped, &endpoint.access);
        assert!(endpoint.watcher.death_reason().is_none());
        command_send.send(CubebCommand::Shutdown).unwrap();
        assert!(owner.join().unwrap().is_ok());
        assert!(dropped.load(AtomicOrdering::Acquire));
        retire(endpoint);
    }

    #[test]
    fn failed_init_weak_closure_cannot_retain_callback() {
        let endpoint = endpoint(2);
        endpoint.access.suspend();
        let leaked_access = endpoint.access.clone();
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let (_accept_send, accept_recv) = crossbeam_channel::bounded(1);
        let (started_send, _started_recv) = crossbeam_channel::bounded(1);
        let (events, _) = AudioOutputEventSink::bounded(4);
        let panic_events = Mutex::new(None);
        command_send
            .send(CubebCommand::Start {
                access: endpoint.access.clone(),
                events,
                ready: ready_send,
                accept: accept_recv,
                started: started_send,
            })
            .unwrap();
        assert!(
            run_owner_with::<FakeStream, _>(command_recv, &panic_events, move |_| {
                // Models Cubeb 0.34 leaking its high-level callback box after native init Err.
                std::mem::forget(leaked_access);
                Err(AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "forced Cubeb init failure",
                ))
            })
            .is_ok()
        );
        assert!(ready_recv.recv().unwrap().is_err());
        retire(endpoint);
    }

    #[test]
    fn start_error_and_panic_close_before_stream_drop() {
        for action in [FakeAction::Err, FakeAction::Panic] {
            let endpoint = endpoint(2);
            endpoint.access.suspend();
            let (command_send, command_recv) = crossbeam_channel::bounded(1);
            let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
            let (_accept_send, accept_recv) = crossbeam_channel::bounded(1);
            let (started_send, _started_recv) = crossbeam_channel::bounded(1);
            let (events, _) = AudioOutputEventSink::bounded(4);
            let panic_events = Mutex::new(None);
            let dropped = Arc::new(AtomicBool::new(false));
            command_send
                .send(CubebCommand::Start {
                    access: endpoint.access.clone(),
                    events,
                    ready: ready_send,
                    accept: accept_recv,
                    started: started_send,
                })
                .unwrap();
            let result = run_owner_with(command_recv, &panic_events, {
                let dropped = Arc::clone(&dropped);
                move |access| {
                    Ok(FakeStream {
                        access,
                        start: action,
                        stop: FakeAction::Ok,
                        dropped,
                        require_closed_on_drop: true,
                    })
                }
            });
            assert!(dropped.load(AtomicOrdering::Acquire));
            match action {
                FakeAction::Err => {
                    assert!(result.is_ok());
                    assert!(ready_recv.recv().unwrap().is_err());
                }
                FakeAction::Panic => {
                    assert!(result.is_err());
                    assert!(ready_recv.try_recv().is_err());
                }
                FakeAction::Ok => unreachable!(),
            }
            retire(endpoint);
        }
    }

    #[test]
    fn acceptance_disconnect_and_terminal_boundary_never_open_rendering() {
        for terminal_at_boundary in [false, true] {
            let endpoint = endpoint(2);
            endpoint.access.suspend();
            let (command_send, command_recv) = crossbeam_channel::bounded(1);
            let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
            let (accept_send, accept_recv) = crossbeam_channel::bounded(1);
            let (started_send, started_recv) = crossbeam_channel::bounded(1);
            let (events, _) = AudioOutputEventSink::bounded(4);
            let panic_events = Arc::new(Mutex::new(None));
            let owner_events = Arc::clone(&panic_events);
            let dropped = Arc::new(AtomicBool::new(false));
            let stream_dropped = Arc::clone(&dropped);
            let owner = thread::spawn(move || {
                run_owner_with(command_recv, &owner_events, move |access| {
                    Ok(FakeStream {
                        access,
                        start: FakeAction::Ok,
                        stop: FakeAction::Ok,
                        dropped: stream_dropped,
                        require_closed_on_drop: true,
                    })
                })
            });
            command_send
                .send(CubebCommand::Start {
                    access: endpoint.access.clone(),
                    events,
                    ready: ready_send,
                    accept: accept_recv,
                    started: started_send,
                })
                .unwrap();
            ready_recv
                .recv_timeout(Duration::from_secs(1))
                .unwrap()
                .unwrap();
            if terminal_at_boundary {
                handle_state(State::Error, &endpoint.access);
                accept_send.send(()).unwrap();
            } else {
                drop(accept_send);
            }
            assert!(started_recv.recv_timeout(Duration::from_secs(1)).is_err());
            let result = owner.join().unwrap();
            if terminal_at_boundary {
                assert!(result.is_err());
                assert_eq!(
                    endpoint.watcher.death_reason(),
                    Some(AudioOutputDeathReason::BackendFailure)
                );
            } else {
                assert!(result.is_ok());
                assert!(endpoint.watcher.death_reason().is_none());
            }
            assert!(dropped.load(AtomicOrdering::Acquire));
            assert_eq!(endpoint.renders.load(AtomicOrdering::Acquire), 0);
            retire(endpoint);
        }
    }

    #[test]
    fn terminal_state_before_ready_rejects_start_without_rendering() {
        let endpoint = endpoint(2);
        endpoint.access.suspend();
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let (_accept_send, accept_recv) = crossbeam_channel::bounded(1);
        let (started_send, _started_recv) = crossbeam_channel::bounded(1);
        let (events, _) = AudioOutputEventSink::bounded(4);
        let panic_events = Mutex::new(None);
        let dropped = Arc::new(AtomicBool::new(false));
        command_send
            .send(CubebCommand::Start {
                access: endpoint.access.clone(),
                events,
                ready: ready_send,
                accept: accept_recv,
                started: started_send,
            })
            .unwrap();
        let result = run_owner_with(command_recv, &panic_events, {
            let dropped = Arc::clone(&dropped);
            move |access| {
                handle_state(State::Error, &access);
                Ok(FakeStream {
                    access,
                    start: FakeAction::Ok,
                    stop: FakeAction::Ok,
                    dropped,
                    require_closed_on_drop: true,
                })
            }
        });
        assert!(result.is_ok());
        assert!(ready_recv.recv().unwrap().is_err());
        assert!(dropped.load(AtomicOrdering::Acquire));
        assert_eq!(endpoint.renders.load(AtomicOrdering::Acquire), 0);
        retire(endpoint);
    }

    #[test]
    fn state_failure_is_quarantined_without_stream_drop() {
        let endpoint = endpoint(2);
        let stream = FakeStream {
            access: endpoint.access.clone(),
            start: FakeAction::Ok,
            stop: FakeAction::Err,
            dropped: Arc::new(AtomicBool::new(false)),
            require_closed_on_drop: false,
        };
        let dropped = Arc::clone(&stream.dropped);
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let (response_send, response_recv) = crossbeam_channel::bounded(1);
        command_send
            .send(CubebCommand::Suspend(response_send))
            .unwrap();
        let outcome = drive_started(
            stream,
            command_recv,
            AccessCloseGuard::new(endpoint.access.clone()),
        );
        assert!(response_recv.recv().unwrap().is_err());
        let StartedOutcome::Quarantine(stream) = outcome else {
            panic!("Cubeb stop failure was not quarantined")
        };
        assert!(!dropped.load(AtomicOrdering::Acquire));
        assert_eq!(
            endpoint.watcher.death_reason(),
            Some(AudioOutputDeathReason::DeviceUnavailable)
        );
        std::mem::forget(stream);
        retire(endpoint);
    }

    #[test]
    fn every_state_error_or_panic_quarantines_the_exact_stream() {
        enum Transition {
            SuspendPanic,
            ResumeError,
            ResumePanic,
        }
        for transition in [
            Transition::SuspendPanic,
            Transition::ResumeError,
            Transition::ResumePanic,
        ] {
            let endpoint = endpoint(2);
            let dropped = Arc::new(AtomicBool::new(false));
            let stream = FakeStream {
                access: endpoint.access.clone(),
                start: match transition {
                    Transition::ResumeError => FakeAction::Err,
                    Transition::ResumePanic => FakeAction::Panic,
                    Transition::SuspendPanic => FakeAction::Ok,
                },
                stop: match transition {
                    Transition::SuspendPanic => FakeAction::Panic,
                    Transition::ResumeError | Transition::ResumePanic => FakeAction::Ok,
                },
                dropped: Arc::clone(&dropped),
                require_closed_on_drop: false,
            };
            let (command_send, command_recv) = crossbeam_channel::bounded(2);
            let (first_send, first_recv) = crossbeam_channel::bounded(1);
            command_send
                .send(CubebCommand::Suspend(first_send))
                .unwrap();
            let resume_recv = if matches!(
                transition,
                Transition::ResumeError | Transition::ResumePanic
            ) {
                let (send, recv) = crossbeam_channel::bounded(1);
                command_send.send(CubebCommand::Resume(send)).unwrap();
                Some(recv)
            } else {
                None
            };
            let outcome = drive_started(
                stream,
                command_recv,
                AccessCloseGuard::new(endpoint.access.clone()),
            );
            match transition {
                Transition::SuspendPanic => assert!(first_recv.recv().unwrap().is_err()),
                Transition::ResumeError | Transition::ResumePanic => {
                    first_recv.recv().unwrap().unwrap();
                    assert!(resume_recv.unwrap().recv().unwrap().is_err());
                }
            }
            let StartedOutcome::Quarantine(stream) = outcome else {
                panic!("Cubeb state failure was not quarantined")
            };
            assert!(!dropped.load(AtomicOrdering::Acquire));
            assert_eq!(
                endpoint.watcher.death_reason(),
                Some(match transition {
                    Transition::ResumeError => AudioOutputDeathReason::DeviceUnavailable,
                    Transition::SuspendPanic | Transition::ResumePanic => {
                        AudioOutputDeathReason::BackendFailure
                    }
                })
            );
            std::mem::forget(stream);
            retire(endpoint);
        }
    }

    #[test]
    fn joined_owner_drops_stream_before_retiring_callback() {
        let endpoint = endpoint(2);
        endpoint.access.suspend();
        let (command_send, command_recv) = crossbeam_channel::bounded(1);
        let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
        let (accept_send, accept_recv) = crossbeam_channel::bounded(1);
        let (started_send, started_recv) = crossbeam_channel::bounded(1);
        let (events, _) = AudioOutputEventSink::bounded(4);
        let panic_events = Arc::new(Mutex::new(None));
        let owner_events = Arc::clone(&panic_events);
        let dropped = Arc::new(AtomicBool::new(false));
        let stream_dropped = Arc::clone(&dropped);
        let owner = thread::spawn(move || {
            run_owner_with(command_recv, &owner_events, move |access| {
                Ok(FakeStream {
                    access,
                    start: FakeAction::Ok,
                    stop: FakeAction::Ok,
                    dropped: stream_dropped,
                    require_closed_on_drop: true,
                })
            })
        });
        let observed = install_join_observer(owner).map_err(|_| ()).unwrap();
        drop(observed.blocking_join);
        command_send
            .send(CubebCommand::Start {
                access: endpoint.access.clone(),
                events,
                ready: ready_send,
                accept: accept_recv,
                started: started_send,
            })
            .unwrap();
        ready_recv.recv().unwrap().unwrap();
        accept_send.send(()).unwrap();
        started_recv.recv().unwrap();

        endpoint.owner.begin_shutdown();
        endpoint.bridge.close();
        command_send.send(CubebCommand::Shutdown).unwrap();
        executor::block_on(observed.completion.into_bridge_shutdown(endpoint.bridge)).unwrap();
        assert!(dropped.load(AtomicOrdering::Acquire));
        assert_eq!(Arc::strong_count(endpoint.owner.slot.as_ref().unwrap()), 1);
        endpoint
            .owner
            .try_reclaim_after_shutdown(EndpointShutdownConfirmed::new())
            .map_err(|_| ())
            .unwrap()
            .unwrap();
    }
}
