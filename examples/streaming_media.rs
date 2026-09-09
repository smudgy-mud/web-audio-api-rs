//! Hosted graph playback using the enabled Symphonia codecs and continuous resampling.
//! `cargo run --example streaming_media -- samples/sample.mp3 5 none`
//! Omit `none` for physical output. This native example is not a scripting API.

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::{AudioNode, PcmSourceNode, PCM_SOURCE_CAPACITY};
use web_audio_api::output::SystemAudioOutput;
use web_audio_api::MediaFileDecoder;

type Error = Box<dyn std::error::Error + Send + Sync>;

struct FilePlayback {
    node: PcmSourceNode,
    worker: Option<JoinHandle<Result<(), Error>>>,
    queued: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
}

struct Done(Arc<AtomicBool>);
impl Drop for Done {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl FilePlayback {
    fn new(context: &impl BaseAudioContext, path: PathBuf) -> Result<Self, Error> {
        let (node, mut writer) = PcmSourceNode::new(context)?;
        let rate = context.sample_rate() as u32;
        let queued = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let ready = Arc::clone(&queued);
        let finished = Done(Arc::clone(&done));
        let worker = thread::Builder::new()
            .name("media-file-decoder".into())
            .spawn(move || {
                let _finished = finished;
                let mut decoder = MediaFileDecoder::new(File::open(path)?, rate)?;
                let mut frames = [[0.0; 2]; 1024];
                while !writer.is_stopped() {
                    let count = decoder.read(&mut frames)?;
                    if count == 0 {
                        break;
                    }
                    let mut offset = 0;
                    while offset < count {
                        if writer.is_stopped() {
                            return Ok(());
                        }
                        let written = writer.write(&frames[offset..count]);
                        offset += written;
                        ready.fetch_add(written, Ordering::Release);
                        if written == 0 {
                            thread::park_timeout(Duration::from_millis(1));
                        }
                    }
                }
                Ok(())
            })?;
        Ok(Self {
            node,
            worker: Some(worker),
            queued,
            done,
        })
    }

    fn prefill(&mut self) -> Result<(), Error> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.queued.load(Ordering::Acquire) < PCM_SOURCE_CAPACITY {
            if self.done.load(Ordering::Acquire) {
                return self.join();
            }
            if Instant::now() >= deadline {
                return Err("media prefill timed out".into());
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    fn join(&mut self) -> Result<(), Error> {
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| "decoder worker panicked")??;
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<(), Error> {
        self.node.stop();
        if let Some(worker) = &self.worker {
            worker.thread().unpark();
        }
        self.join()
    }
}

impl Drop for FilePlayback {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: streaming_media path [seconds] [sink-id]")?;
    let seconds: f64 = args.next().as_deref().unwrap_or("10").parse()?;
    if !seconds.is_finite() || !(0.0..=3600.0).contains(&seconds) {
        return Err("invalid duration".into());
    }
    let context = AudioContext::builder(Arc::new(SystemAudioOutput::new()))
        .options(AudioContextOptions {
            sample_rate: Some(48_000.0),
            sink_id: args.next().unwrap_or_default(),
            ..Default::default()
        })
        .build()?;
    let mut playback = FilePlayback::new(&context, path.into())?;
    playback.prefill()?;
    let gain = context.create_gain();
    gain.gain().set_value(0.25);
    playback.node.connect(&gain);
    gain.connect(&context.destination());
    playback.node.start();
    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    while !playback.node.ended() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    playback.stop()?;
    println!(
        "{} frames rendered, {} starved quanta; decoder joined",
        playback.node.rendered_frames(),
        playback.node.underrun_quanta()
    );
    context.close_sync();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use web_audio_api::output::SilentAudioOutput;

    #[test]
    fn worker_decodes_common_formats_into_hosted_graph_and_joins_on_stop() {
        for file in [
            "sample.wav",
            "sample.mp3",
            "sample.flac",
            "sample.ogg",
            "sample-aac.m4a",
            "sample-alac.m4a",
            "sample.aiff",
        ] {
            let context = AudioContext::builder(Arc::new(SilentAudioOutput::new()))
                .options(AudioContextOptions {
                    sample_rate: Some(48_000.0),
                    ..Default::default()
                })
                .build()
                .unwrap();
            let mut playback =
                FilePlayback::new(&context, PathBuf::from("samples").join(file)).unwrap();
            playback.prefill().unwrap();
            let gain = context.create_gain();
            gain.gain().set_value(0.25);
            playback.node.connect(&gain);
            gain.connect(&context.destination());
            playback.node.start();
            let deadline = Instant::now() + Duration::from_secs(5);
            while playback.node.rendered_frames() < 2048 {
                assert!(Instant::now() < deadline, "{file}");
                thread::sleep(Duration::from_millis(1));
            }
            playback.stop().unwrap();
            assert!(playback.worker.is_none());
            assert!(playback.done.load(Ordering::Acquire));
            context.close_sync();
        }
    }

    #[test]
    fn context_close_cancels_backpressured_decoder_and_owner_joins() {
        let context = AudioContext::builder(Arc::new(SilentAudioOutput::new()))
            .build()
            .unwrap();
        let mut playback = FilePlayback::new(&context, "samples/sample.mp3".into()).unwrap();
        playback.prefill().unwrap();
        context.close_sync();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !playback.done.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        playback.join().unwrap();
        assert!(playback.worker.is_none());
    }
}
