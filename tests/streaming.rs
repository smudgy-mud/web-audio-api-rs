use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use web_audio_api::context::{
    AudioContext, AudioContextOptions, AudioControlBatchReservation, AudioNodeLifetimeReservation,
    BaseAudioContext, OfflineAudioContext,
};
use web_audio_api::node::{AudioNode, PcmSourceNode, PCM_SOURCE_CAPACITY};
use web_audio_api::output::*;
use web_audio_api::MediaFileDecoder;

#[global_allocator]
static ALLOCATOR: alloc_counter::AllocCounterSystem = alloc_counter::AllocCounterSystem;

fn wait_for(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !ready() {
        assert!(Instant::now() < deadline, "stream progress timed out");
        thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Default)]
struct Probe {
    stop: AtomicBool,
    positive_samples: AtomicU64,
    unexpected_samples: AtomicU64,
    heap_ops: AtomicU64,
    measure: AtomicBool,
}
struct Factory(Arc<Probe>);
struct Prepared {
    config: AudioOutputConfig,
    probe: Arc<Probe>,
}
struct Running {
    probe: Arc<Probe>,
    worker: Option<thread::JoinHandle<()>>,
}

impl AudioOutputFactory for Factory {
    fn prepare(
        &self,
        request: &AudioOutputRequest,
    ) -> Result<Box<dyn PreparedAudioOutput>, AudioOutputError> {
        Ok(Box::new(Prepared {
            config: AudioOutputConfig::new(
                AudioRenderFormat::new(48_000.0, 2, 128)?,
                request.sink_id(),
                0.0,
            )?,
            probe: Arc::clone(&self.0),
        }))
    }
}
impl PreparedAudioOutput for Prepared {
    fn config(&self) -> &AudioOutputConfig {
        &self.config
    }
    fn start(
        self: Box<Self>,
        mut callback: AudioRenderCallback,
        _events: AudioOutputEventSink,
    ) -> Result<Box<dyn RunningAudioOutput>, AudioOutputStartFailure> {
        let probe = Arc::clone(&self.probe);
        let worker = thread::spawn(move || {
            let mut output = [0.0; 256];
            while !probe.stop.load(Ordering::Acquire) {
                let measure = probe.measure.load(Ordering::Acquire);
                let (counts, _) =
                    alloc_counter::count_alloc(|| callback.render_interleaved_f32(&mut output));
                if measure {
                    probe
                        .heap_ops
                        .fetch_add((counts.0 + counts.1 + counts.2) as u64, Ordering::Relaxed);
                }
                for &sample in &output {
                    if sample == 0.125 {
                        probe.positive_samples.fetch_add(1, Ordering::Relaxed);
                    } else if sample != 0.0 {
                        probe.unexpected_samples.fetch_add(1, Ordering::Relaxed);
                    }
                }
                thread::sleep(Duration::from_millis(1));
            }
        });
        Ok(Box::new(Running {
            probe: self.probe,
            worker: Some(worker),
        }))
    }
    fn abort(self: Box<Self>) -> AudioOutputEndpointShutdown {
        AudioOutputEndpointShutdown::ready(Ok(()))
    }
}
impl RunningAudioOutput for Running {
    fn resume(&mut self) -> Result<(), AudioOutputError> {
        Ok(())
    }
    fn suspend(&mut self) -> Result<(), AudioOutputError> {
        Ok(())
    }
    fn shutdown(mut self: Box<Self>) -> AudioOutputEndpointShutdown {
        self.probe.stop.store(true, Ordering::Release);
        let result = self.worker.take().unwrap().join().map_err(|_| {
            AudioOutputError::new(
                AudioOutputErrorKind::Shutdown,
                "test render thread panicked",
            )
        });
        AudioOutputEndpointShutdown::ready(result)
    }
}

#[test]
fn hosted_pcm_gain_starvation_stop_and_queue_lease() {
    let (counts, data) = alloc_counter::count_alloc(|| std::hint::black_box(vec![1_u8; 64]));
    assert!(counts.0 > 0);
    drop(data);
    let probe = Arc::new(Probe::default());
    let context = AudioContext::builder(Arc::new(Factory(Arc::clone(&probe))))
        .options(AudioContextOptions {
            sample_rate: Some(48_000.0),
            ..Default::default()
        })
        .build()
        .unwrap();
    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Release);
        }
    }
    let released = Arc::new(AtomicUsize::new(0));
    let (node, mut writer) = PcmSourceNode::with_reservations(
        &context,
        Some(AudioNodeLifetimeReservation::new(Lease(Arc::clone(
            &released,
        )))),
        None,
    )
    .unwrap();
    let gain = context.create_gain();
    gain.gain().set_value(0.25);
    node.connect(&gain);
    gain.connect(&context.destination());
    // Paused queue retains its entire fixed capacity.
    assert_eq!(
        writer.write(&[[0.5; 2]; PCM_SOURCE_CAPACITY + 1]),
        PCM_SOURCE_CAPACITY
    );
    assert_eq!(writer.write(&[[0.5; 2]]), 0);
    node.start();
    wait_for(|| node.rendered_frames() == PCM_SOURCE_CAPACITY as u64);
    wait_for(|| node.underrun_quanta() > 0);
    assert!(!node.ended());
    // Bootstrap and graph construction have settled before counting the whole callback.
    probe.measure.store(true, Ordering::Release);
    assert_eq!(writer.write(&[[0.5; 2]; 257]), 257);
    wait_for(|| node.rendered_frames() == (PCM_SOURCE_CAPACITY + 257) as u64);
    node.stop();
    wait_for(|| node.ended());
    assert!(writer.is_stopped());
    assert_eq!(writer.write(&[[0.5; 2]]), 0);
    assert_eq!(probe.heap_ops.load(Ordering::Acquire), 0);
    assert_eq!(probe.unexpected_samples.load(Ordering::Acquire), 0);
    assert_eq!(
        probe.positive_samples.load(Ordering::Acquire),
        ((PCM_SOURCE_CAPACITY + 257) * 2) as u64
    );
    probe.measure.store(false, Ordering::Release);
    context.close_sync();
    drop(node);
    drop(gain);
    drop(context);
    assert_eq!(
        released.load(Ordering::Acquire),
        0,
        "writer still owns queue storage"
    );
    drop(writer);
    wait_for(|| released.load(Ordering::Acquire) == 1);
}

#[test]
fn hosted_eof_drains_final_partial_quantum_and_closed_admission_releases_guards() {
    let context = AudioContext::builder(Arc::new(SilentAudioOutput::new()))
        .build()
        .unwrap();
    let (node, mut writer) = PcmSourceNode::new(&context).unwrap();
    node.connect(&context.destination());
    assert_eq!(writer.write(&[[0.25; 2]; 129]), 129);
    drop(writer);
    node.start();
    wait_for(|| node.ended());
    assert_eq!(node.rendered_frames(), 129);
    assert_eq!(node.underrun_quanta(), 0);
    context.close_sync();
    let lifetime = Arc::new(());
    let control = Arc::new(());
    assert!(PcmSourceNode::with_reservations(
        &context,
        Some(AudioNodeLifetimeReservation::new(Arc::clone(&lifetime))),
        Some(AudioControlBatchReservation::new(Arc::clone(&control)))
    )
    .is_err());
    assert_eq!(Arc::strong_count(&lifetime), 1);
    assert_eq!(Arc::strong_count(&control), 1);
}

#[test]
fn abandoned_connected_paused_source_reclaims_without_context_close() {
    for writer_first in [false, true] {
        let context = AudioContext::builder(Arc::new(SilentAudioOutput::new()))
            .build()
            .unwrap();
        let lease = Arc::new(());
        let (node, mut writer) = PcmSourceNode::with_reservations(
            &context,
            Some(AudioNodeLifetimeReservation::new(Arc::clone(&lease))),
            None,
        )
        .unwrap();
        node.connect(&context.destination());
        assert_eq!(
            writer.write(&[[0.5; 2]; PCM_SOURCE_CAPACITY]),
            PCM_SOURCE_CAPACITY
        );
        if writer_first {
            drop(writer);
            drop(node);
        } else {
            drop(node);
            wait_for(|| writer.is_stopped());
            assert_eq!(writer.write(&[[0.5; 2]]), 0);
            assert_eq!(
                Arc::strong_count(&lease),
                2,
                "writer retains queue accounting"
            );
            drop(writer);
        }
        wait_for(|| Arc::strong_count(&lease) == 1);
        context.close_sync();
    }
}

#[test]
fn retained_connected_paused_source_can_start_after_rendering_silence() {
    let context = AudioContext::builder(Arc::new(SilentAudioOutput::new()))
        .build()
        .unwrap();
    let (node, mut writer) = PcmSourceNode::new(&context).unwrap();
    node.connect(&context.destination());
    assert_eq!(writer.write(&[[0.25; 2]; 129]), 129);
    // Let several render quanta run while the paused source's handle is retained.
    let deadline = context.current_time() + 0.05;
    wait_for(|| context.current_time() >= deadline);
    assert_eq!(node.rendered_frames(), 0);
    assert!(!node.ended());
    assert!(!writer.is_stopped());
    drop(writer);
    node.start();
    wait_for(|| node.ended());
    assert_eq!(node.rendered_frames(), 129);
    assert_eq!(node.underrun_quanta(), 0);
    context.close_sync();
}

#[test]
fn existing_common_format_fixtures_decode_incrementally() {
    for path in [
        "sample.wav",
        "sample-38000.wav",
        "sample-48000.wav",
        "sample.mp3",
        "sample.flac",
        "sample.ogg",
        "sample-aac.m4a",
        "sample-alac.m4a",
        "sample.aiff",
    ] {
        let path = format!("samples/{path}");
        let mut stream = MediaFileDecoder::new(File::open(&path).unwrap(), 48_000).unwrap();
        let source_rate = stream.source_sample_rate();
        let reference_context = OfflineAudioContext::new(2, 128, source_rate as f32);
        let reference = reference_context
            .decode_audio_data_sync(File::open(&path).unwrap())
            .unwrap();
        let expected = (reference.length() as u64 * 48_000).div_ceil(u64::from(source_rate));
        let mut scratch = [[0.0; 2]; 257]; // Deliberately cross both packet and resampler boundaries.
        let mut frames = 0;
        let mut energy = 0.0_f64;
        loop {
            let count = stream
                .read(&mut scratch)
                .unwrap_or_else(|error| panic!("{path}: {error}"));
            if count == 0 {
                break;
            }
            assert!(scratch[..count]
                .iter()
                .flatten()
                .all(|sample| sample.is_finite()));
            energy += scratch[..count]
                .iter()
                .flatten()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>();
            frames += count as u64;
            assert!(
                frames <= expected + 4096,
                "{path}: resampler failed to terminate"
            );
        }
        assert_eq!(frames, expected, "{path}");
        assert!(energy > 1.0, "{path}: silent decode");
        assert_eq!(stream.read(&mut scratch).unwrap(), 0);
        println!("{path}: {source_rate} -> 48000 Hz; {frames} frames");
    }
}

#[test]
fn unsupported_opus_is_an_error_instead_of_silent_success() {
    // Existing WebM fixture contains Opus, which the pinned decoder does not support.
    let error =
        MediaFileDecoder::new(File::open("samples/sample.webm").unwrap(), 48_000).unwrap_err();
    assert!(
        error.to_string().contains("unsupported audio codec"),
        "{error}"
    );
}

#[test]
fn native_legacy_gain_preserves_partial_frames() {
    let mut context = OfflineAudioContext::new(2, 256, 48_000.0);
    let (node, mut writer) = PcmSourceNode::new(&context).unwrap();
    let gain = context.create_gain();
    gain.gain().set_value(0.5);
    node.connect(&gain);
    gain.connect(&context.destination());
    writer.write(&[[0.5, -0.5]; 129]);
    drop(writer);
    node.start();
    let output = context.start_rendering_sync();
    assert_eq!(&output.get_channel_data(0)[..129], &[0.25; 129]);
    assert_eq!(&output.get_channel_data(1)[..129], &[-0.25; 129]);
    assert_eq!(&output.get_channel_data(0)[129..], &[0.0; 127]);
    assert!(node.ended());
}

#[test]
fn staged_admission_refusal_is_inert_and_recovers_after_resume() {
    let context = AudioContext::builder(Arc::new(SilentAudioOutput::new()))
        .initially_suspended(true)
        .build()
        .unwrap();
    let mut nodes = Vec::new();
    for _ in 0..512 {
        match PcmSourceNode::new(&context) {
            Ok(pair) => nodes.push(pair),
            Err(_) => break,
        }
    }
    assert!(!nodes.is_empty() && nodes.len() < 512);
    let lifetime = Arc::new(());
    let control = Arc::new(());
    for _ in 0..4 {
        assert!(PcmSourceNode::with_reservations(
            &context,
            Some(AudioNodeLifetimeReservation::new(Arc::clone(&lifetime))),
            Some(AudioControlBatchReservation::new(Arc::clone(&control)))
        )
        .is_err());
        assert_eq!(Arc::strong_count(&lifetime), 1);
        assert_eq!(Arc::strong_count(&control), 1);
    }
    context.resume_sync();
    let mut recovered = None;
    wait_for(|| {
        if let Ok(pair) = PcmSourceNode::new(&context) {
            recovered = Some(pair);
            true
        } else {
            false
        }
    });
    let (node, mut writer) = recovered.unwrap();
    node.connect(&context.destination());
    assert_eq!(writer.write(&[[0.5; 2]; 129]), 129);
    drop(writer);
    node.start();
    wait_for(|| node.ended());
    assert_eq!(node.rendered_frames(), 129);
    context.close_sync();
}

static NEXT_WAV: AtomicUsize = AtomicUsize::new(0);
struct Wav(std::path::PathBuf);
impl Wav {
    fn tone(rate: u32, frames: usize) -> Self {
        let path = std::env::temp_dir().join(format!(
            "smudgy-media-{}-{}.wav",
            std::process::id(),
            NEXT_WAV.fetch_add(1, Ordering::Relaxed)
        ));
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for frame in 0..frames {
            let value =
                (std::f64::consts::TAU * 1000.0 * frame as f64 / f64::from(rate)).sin() * 16000.0;
            writer.write_sample(value.round() as i16).unwrap();
        }
        writer.finalize().unwrap();
        Self(path)
    }
}
impl Drop for Wav {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
    }
}

#[test]
fn m4a_with_trailing_metadata_streams_identically() {
    fn adjust_offsets(boxes: &mut [u8], removed_start: u32, removed_size: u32) -> usize {
        let mut cursor = 0;
        let mut adjusted = 0;
        while cursor < boxes.len() {
            let size = u32::from_be_bytes(boxes[cursor..cursor + 4].try_into().unwrap()) as usize;
            assert!(size >= 8 && cursor + size <= boxes.len());
            let kind: [u8; 4] = boxes[cursor + 4..cursor + 8].try_into().unwrap();
            let contents = &mut boxes[cursor + 8..cursor + size];
            if &kind == b"stco" {
                let entries = u32::from_be_bytes(contents[4..8].try_into().unwrap()) as usize;
                assert_eq!(contents.len(), 8 + entries * 4);
                for entry in contents[8..].chunks_exact_mut(4) {
                    let offset = u32::from_be_bytes(entry.try_into().unwrap());
                    assert!(offset >= removed_start + removed_size);
                    entry.copy_from_slice(&(offset - removed_size).to_be_bytes());
                    adjusted += 1;
                }
            } else if [*b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl"].contains(&kind) {
                adjusted += adjust_offsets(contents, removed_start, removed_size);
            }
            cursor += size;
        }
        adjusted
    }
    let original = std::fs::read("samples/sample-aac.m4a").unwrap();
    let mut cursor = 0;
    while &original[cursor + 4..cursor + 8] != b"moov" {
        cursor += u32::from_be_bytes(original[cursor..cursor + 4].try_into().unwrap()) as usize;
    }
    let size = u32::from_be_bytes(original[cursor..cursor + 4].try_into().unwrap()) as usize;
    let mut metadata = original[cursor..cursor + size].to_vec();
    assert!(adjust_offsets(&mut metadata, cursor as u32, size as u32) > 0);
    let mut trailing = original[..cursor].to_vec();
    trailing.extend_from_slice(&original[cursor + size..]);
    trailing.extend_from_slice(&metadata);
    let file = Wav::tone(48_000, 1); // RAII temporary path; content probing ignores its extension.
    std::fs::write(&file.0, trailing).unwrap();
    let mut reference =
        MediaFileDecoder::new(File::open("samples/sample-aac.m4a").unwrap(), 48_000).unwrap();
    let mut stream = MediaFileDecoder::new(File::open(&file.0).unwrap(), 48_000).unwrap();
    let mut expected = [[0.0; 2]; 1024];
    let mut actual = expected;
    loop {
        let a = reference.read(&mut expected).unwrap();
        let b = stream.read(&mut actual).unwrap();
        assert_eq!(a, b);
        assert_eq!(actual[..b], expected[..a]);
        if a == 0 {
            break;
        }
    }
}

#[test]
fn resampling_preserves_tone_phase_and_duration_across_packet_boundaries() {
    for rate in [44_100, 96_000] {
        let wav = Wav::tone(rate, rate as usize / 2);
        let mut decoder = MediaFileDecoder::new(File::open(&wav.0).unwrap(), 48_000).unwrap();
        let mut samples = [[0.0; 2]; 137];
        let mut position = 0;
        let mut projection_sin = 0.0;
        let mut projection_cos = 0.0;
        let mut energy = 0.0;
        loop {
            let count = decoder.read(&mut samples).unwrap();
            if count == 0 {
                break;
            }
            for sample in &samples[..count] {
                assert_eq!(sample[0], sample[1]);
                if (480..23520).contains(&position) {
                    let phase = std::f64::consts::TAU * 1000.0 * f64::from(position) / 48_000.0;
                    let value = f64::from(sample[0]);
                    projection_sin += value * phase.sin();
                    projection_cos += value * phase.cos();
                    energy += value * value;
                }
                position += 1;
                assert!(position <= 24_000);
            }
        }
        assert_eq!(position, 24_000);
        let sine = 2.0 * projection_sin / 23040.0;
        let cosine = 2.0 * projection_cos / 23040.0;
        let amplitude = sine.hypot(cosine);
        assert!(
            (amplitude - 16000.0 / 32768.0).abs() < 0.0001,
            "{rate}: amplitude {amplitude}"
        );
        // Permit the sinc interpolator's fractional origin offset, but never a
        // missing packet, trimmed lookahead, or accumulating phase drift.
        assert!(
            cosine.atan2(sine).abs() < std::f64::consts::TAU / 48.0,
            "{rate}: phase shift"
        );
        let residual = energy / 23040.0 - amplitude * amplitude / 2.0;
        assert!(residual.abs() < 0.000001, "{rate}: residual {residual}");
    }
}

#[test]
fn short_resampler_tail_and_truncated_input_are_terminal() {
    let wav = Wav::tone(44_100, 1);
    let mut decoder = MediaFileDecoder::new(File::open(&wav.0).unwrap(), 48_000).unwrap();
    let mut frames = [[0.0; 2]; 128];
    assert_eq!(decoder.read(&mut frames).unwrap(), 2);
    assert_eq!(decoder.read(&mut frames).unwrap(), 0);
    drop(decoder);
    let wav = Wav::tone(48_000, 10_000);
    let file = File::options().write(true).open(&wav.0).unwrap();
    file.set_len(2051).unwrap();
    drop(file);
    // A truncated file may be rejected while decoding its first packet or later.
    if let Ok(mut decoder) = MediaFileDecoder::new(File::open(&wav.0).unwrap(), 48_000) {
        let mut failed = false;
        for _ in 0..100 {
            match decoder.read(&mut frames) {
                Err(_) => {
                    failed = true;
                    break;
                }
                Ok(0) => break,
                Ok(_) => {}
            }
        }
        assert!(failed, "truncation must not silently report clean EOF");
        assert!(decoder.read(&mut frames).is_err());
    }
}
