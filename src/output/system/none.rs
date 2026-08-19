use std::fmt;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use super::{
    contained_owner, install_join_observer, OwnerCompletion, OwnerResult, SystemRenderAccess,
    SystemRenderBridge,
};
use crate::output::{
    AudioOutputConfig, AudioOutputEndpointShutdown, AudioOutputError, AudioOutputErrorKind,
    AudioOutputEventSink, AudioOutputRequest, AudioOutputStartFailure, AudioRenderCallback,
    AudioRenderFormat, AudioRenderStatus, PreparedAudioOutput, RunningAudioOutput,
};

const NONE_SAMPLE_RATE: f32 = 48_000.;
const NONE_CALLBACK_FRAMES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StartBehavior {
    Normal,
    Fail,
    PanicBeforeResponse,
    PanicAfterResponse,
}

pub(super) fn prepare(
    request: &AudioOutputRequest,
) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
    prepare_inner(request, StartBehavior::Normal)
}

#[cfg(test)]
pub(super) fn prepare_with_start_failure_for_test(
    request: &AudioOutputRequest,
) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
    prepare_inner(request, StartBehavior::Fail)
}

#[cfg(test)]
pub(super) fn prepare_with_start_panic_for_test(
    request: &AudioOutputRequest,
    after_response: bool,
) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
    prepare_inner(
        request,
        if after_response {
            StartBehavior::PanicAfterResponse
        } else {
            StartBehavior::PanicBeforeResponse
        },
    )
}

fn prepare_inner(
    request: &AudioOutputRequest,
    start_behavior: StartBehavior,
) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
    let format = AudioRenderFormat::new(
        request.requested_sample_rate().unwrap_or(NONE_SAMPLE_RATE),
        request.number_of_channels(),
        NONE_CALLBACK_FRAMES,
    )?;
    let config = AudioOutputConfig::new(format, "none", 0.)?;
    request.validate_config(&config)?;

    let (command_send, command_recv) = crossbeam_channel::bounded(1);
    let (ready_send, ready_recv) = crossbeam_channel::bounded(1);
    let panic_events = Arc::new(Mutex::new(None));
    let owner_panic_events = Arc::clone(&panic_events);
    let owner = thread::Builder::new()
        .name("web-audio-system-output-none".into())
        .spawn(move || {
            let panic_ready = ready_send.clone();
            let result = contained_owner(&owner_panic_events, || {
                run_owner(
                    config,
                    command_recv,
                    ready_send,
                    &owner_panic_events,
                    start_behavior,
                )
            });
            if result.is_err() {
                let _ = panic_ready.try_send(result.clone().map(|_| unreachable!()));
            }
            result
        })
        .map_err(|error| {
            AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                format!("failed to spawn silent system output owner: {error}"),
            )
        })?;

    let observed = match install_join_observer(owner) {
        Ok(observed) => observed,
        Err(failure) => {
            let _ = command_send.try_send(NoneCommand::Abort);
            let owner_result = failure.owner.join();
            if let Err(payload) = owner_result {
                std::mem::forget(payload);
            }
            return Err(failure.error);
        }
    };

    let owner_config = match ready_recv.recv() {
        Ok(Ok(config)) => config,
        Ok(Err(error)) => {
            let _ = observed.blocking_join.recv();
            return Err(error);
        }
        Err(_) => {
            let _ = command_send.try_send(NoneCommand::Abort);
            let joined = observed.blocking_join.recv().ok();
            return Err(joined.and_then(Result::err).unwrap_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output owner exited before preparation completed",
                )
            }));
        }
    };

    Ok(Box::new(NonePrepared {
        config: owner_config,
        command_send: Some(command_send),
        completion: Some(observed.completion),
    }))
}

enum NoneCommand {
    Start {
        access: SystemRenderAccess,
        events: AudioOutputEventSink,
        response: crossbeam_channel::Sender<OwnerResult>,
    },
    Suspend(crossbeam_channel::Sender<OwnerResult>),
    Resume(crossbeam_channel::Sender<OwnerResult>),
    Shutdown,
    Abort,
}

fn run_owner(
    config: AudioOutputConfig,
    command_recv: crossbeam_channel::Receiver<NoneCommand>,
    ready_send: crossbeam_channel::Sender<Result<AudioOutputConfig, AudioOutputError>>,
    panic_events: &Mutex<Option<AudioOutputEventSink>>,
    start_behavior: StartBehavior,
) -> OwnerResult {
    ready_send.send(Ok(config.clone())).map_err(|_| {
        AudioOutputError::new(
            AudioOutputErrorKind::BackendSpecific,
            "silent system output preparer disappeared",
        )
    })?;

    let first = match command_recv.recv() {
        Ok(command) => command,
        Err(_) => return Ok(()),
    };
    let NoneCommand::Start {
        access,
        events,
        response,
    } = first
    else {
        return match first {
            NoneCommand::Abort | NoneCommand::Shutdown => Ok(()),
            NoneCommand::Suspend(response) | NoneCommand::Resume(response) => {
                let error = AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output received state work before start",
                );
                let _ = response.send(Err(error.clone()));
                Err(error)
            }
            NoneCommand::Start { .. } => unreachable!(),
        };
    };

    match panic_events.lock() {
        Ok(mut retained) => *retained = Some(events.clone()),
        Err(poisoned) => *poisoned.into_inner() = Some(events.clone()),
    }

    if start_behavior == StartBehavior::PanicBeforeResponse {
        panic!("forced silent system output panic before start response");
    }
    if start_behavior == StartBehavior::Fail {
        let error = AudioOutputError::new(
            AudioOutputErrorKind::BackendSpecific,
            "forced silent system output start failure",
        );
        let _ = response.send(Err(error));
        return Ok(());
    }

    let channels = config.format().number_of_channels();
    let mut output = vec![0.; NONE_CALLBACK_FRAMES * channels];
    let callback_period = Duration::from_secs_f64(
        f64::from(NONE_CALLBACK_FRAMES as u32) / f64::from(config.format().sample_rate()),
    );
    if response.send(Ok(())).is_err() {
        return Ok(());
    }
    if start_behavior == StartBehavior::PanicAfterResponse {
        panic!("forced silent system output panic after start response");
    }

    let result = run_started(
        &command_recv,
        &access,
        &events,
        &mut output,
        callback_period,
    );
    match panic_events.lock() {
        Ok(mut events) => *events = None,
        Err(poisoned) => *poisoned.into_inner() = None,
    }
    result
}

fn run_started(
    command_recv: &crossbeam_channel::Receiver<NoneCommand>,
    access: &SystemRenderAccess,
    events: &AudioOutputEventSink,
    output: &mut [f32],
    callback_period: Duration,
) -> OwnerResult {
    let mut suspended = false;
    let mut callback_stopped = false;
    loop {
        match command_recv.recv_timeout(callback_period) {
            Ok(NoneCommand::Suspend(response)) => {
                suspended = true;
                let _ = response.send(Ok(()));
            }
            Ok(NoneCommand::Resume(response)) => {
                suspended = false;
                let _ = response.send(Ok(()));
            }
            Ok(NoneCommand::Shutdown | NoneCommand::Abort) => return Ok(()),
            Ok(NoneCommand::Start { response, .. }) => {
                let error = AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output received a duplicate start",
                );
                let _ = response.send(Err(error.clone()));
                return Err(error);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if suspended || callback_stopped {
                    output.fill(0.);
                    continue;
                }
                callback_stopped = access.render_interleaved_f32(output) == AudioRenderStatus::Stop;
                if callback_stopped {
                    output.fill(0.);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                let _ = events
                    .report_endpoint_death(crate::output::AudioOutputDeathReason::BackendFailure);
                return Err(AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output controller disconnected",
                ));
            }
        }
    }
}

struct NonePrepared {
    config: AudioOutputConfig,
    command_send: Option<crossbeam_channel::Sender<NoneCommand>>,
    completion: Option<OwnerCompletion>,
}

impl fmt::Debug for NonePrepared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NonePrepared")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PreparedAudioOutput for NonePrepared {
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
            .expect("prepared output is single-use");
        let completion = self
            .completion
            .take()
            .expect("prepared output is single-use");
        let bridge = SystemRenderBridge::new(callback, events);
        let access = SystemRenderBridge::access(&bridge);
        let bridge_events = bridge.events();
        let (response_send, response_recv) = crossbeam_channel::bounded(1);
        if let Err(error) = command_send.send(NoneCommand::Start {
            access,
            events: bridge_events,
            response: response_send,
        }) {
            bridge.close();
            drop(error.into_inner());
            return Err(AudioOutputStartFailure::new(
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output owner disappeared during start",
                ),
                completion.into_bridge_shutdown(bridge),
            ));
        }

        match response_recv.recv() {
            Ok(Ok(())) => Ok(Box::new(NoneRunning {
                command_send: Some(command_send),
                completion: Some(completion),
                bridge: Some(bridge),
            })),
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
                        "silent system output owner exited during start",
                    ),
                    completion.into_bridge_shutdown(bridge),
                ))
            }
        }
    }

    fn abort(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(NoneCommand::Abort);
        }
        self.completion.take().map_or_else(
            || missing_completion("prepared abort"),
            OwnerCompletion::into_shutdown,
        )
    }
}

impl Drop for NonePrepared {
    fn drop(&mut self) {
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(NoneCommand::Abort);
        }
    }
}

struct NoneRunning {
    command_send: Option<crossbeam_channel::Sender<NoneCommand>>,
    completion: Option<OwnerCompletion>,
    bridge: Option<Arc<SystemRenderBridge>>,
}

impl fmt::Debug for NoneRunning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NoneRunning").finish_non_exhaustive()
    }
}

impl NoneRunning {
    fn state_command(
        &self,
        command: impl FnOnce(crossbeam_channel::Sender<OwnerResult>) -> NoneCommand,
    ) -> OwnerResult {
        let (response_send, response_recv) = crossbeam_channel::bounded(1);
        self.command_send
            .as_ref()
            .ok_or_else(|| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output no longer owns its controller",
                )
            })?
            .send(command(response_send))
            .map_err(|_| {
                AudioOutputError::new(
                    AudioOutputErrorKind::BackendSpecific,
                    "silent system output owner disconnected",
                )
            })?;
        response_recv.recv().map_err(|_| {
            AudioOutputError::new(
                AudioOutputErrorKind::BackendSpecific,
                "silent system output owner exited during a state transition",
            )
        })?
    }
}

impl RunningAudioOutput for NoneRunning {
    fn resume(&mut self) -> OwnerResult {
        self.state_command(NoneCommand::Resume)
    }

    fn suspend(&mut self) -> OwnerResult {
        self.state_command(NoneCommand::Suspend)
    }

    fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        if let Some(bridge) = self.bridge.as_ref() {
            bridge.close();
        }
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(NoneCommand::Shutdown);
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

impl Drop for NoneRunning {
    fn drop(&mut self) {
        if let Some(bridge) = self.bridge.as_ref() {
            bridge.close();
        }
        if let Some(command_send) = self.command_send.take() {
            let _ = command_send.try_send(NoneCommand::Shutdown);
        }
        // Dropping the last bridge Arc without `try_retire` destroys only its allocation. The
        // manually-held callback lease is deliberately quarantined, while the closed gate and
        // shutdown command prevent further rendering.
        self.bridge.take();
    }
}

fn missing_completion(operation: &str) -> AudioOutputEndpointShutdown {
    AudioOutputEndpointShutdown::ready(Err(AudioOutputError::new(
        AudioOutputErrorKind::Shutdown,
        format!("silent system output lost completion ownership during {operation}"),
    )))
}
