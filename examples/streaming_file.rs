//! Experimental native file -> bounded PCM queue -> worklet -> GainNode proof.
//!
//! Run: cargo run --example streaming_file -- path.wav [seconds] [sink-id]
//! Test without hardware: cargo test --no-default-features --example streaming_file
//!
//! Only mono/stereo 16-bit PCM WAV at 48 kHz is accepted. This deliberately uses the
//! legacy Rust worklet registration, which is NOT admitted by the hosted graph used
//! by deno_audio. It is not a supported scripting API. See streaming_file/README.md.

#[path = "streaming_file/mod.rs"]
mod streaming_file;

use std::error::Error;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use web_audio_api::context::{AudioContext, AudioContextOptions, BaseAudioContext};
use web_audio_api::node::AudioNode;

use streaming_file::{attach, FileStream, SAMPLE_RATE};

#[cfg(test)]
#[global_allocator]
static ALLOCATOR: alloc_counter::AllocCounterSystem = alloc_counter::AllocCounterSystem;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: streaming_file path.wav [seconds] [sink-id]")?;
    let duration: f64 = args.next().as_deref().unwrap_or("10").parse()?;
    if !duration.is_finite() || !(0.0..=3600.0).contains(&duration) {
        return Err("seconds must be finite and between 0 and 3600".into());
    }
    let sink_id = args.next().unwrap_or_default();
    let (mut stream, source) = FileStream::open(path.into())?;
    stream.wait_for_prefill(Duration::from_secs(5))?;
    let context = AudioContext::new(AudioContextOptions {
        sample_rate: Some(SAMPLE_RATE as f32),
        sink_id,
        ..AudioContextOptions::default()
    });
    let node = attach(&context, source);
    let gain = context.create_gain();
    gain.gain().set_value(0.25);
    node.connect(&gain);
    gain.connect(&context.destination());
    stream.play();
    let deadline = Instant::now() + Duration::from_secs_f64(duration);
    while !stream.stats.finished.load(Ordering::Acquire) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    stream.stop()?;
    context.close_sync();
    println!(
        "{} frames decoded, {} rendered, {} empty quanta; decoder joined",
        stream.stats.decoded.load(Ordering::Acquire),
        stream.stats.rendered.load(Ordering::Acquire),
        stream.stats.underruns.load(Ordering::Acquire)
    );
    Ok(())
}
