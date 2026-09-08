use super::*;
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::AtomicUsize;
use web_audio_api::context::{AudioContext, AudioContextOptions, OfflineAudioContext};
use web_audio_api::node::AudioNode;

static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

struct Fixture(PathBuf);

fn sample(frame: usize) -> i16 {
    ((frame % 1024) as i16 - 512) * 16
}

impl Fixture {
    fn wav(frames: u32, channels: u16, written_frames: u32) -> Self {
        let path = std::env::temp_dir().join(format!(
            "web-audio-stream-proof-{}-{}.wav",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::create(&path).unwrap();
        let data_bytes = frames * u32::from(channels) * 2;
        file.write_all(b"RIFF").unwrap();
        file.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
        file.write_all(b"WAVEfmt ").unwrap();
        file.write_all(&16_u32.to_le_bytes()).unwrap();
        file.write_all(&1_u16.to_le_bytes()).unwrap();
        file.write_all(&channels.to_le_bytes()).unwrap();
        file.write_all(&SAMPLE_RATE.to_le_bytes()).unwrap();
        file.write_all(&(SAMPLE_RATE * u32::from(channels) * 2).to_le_bytes())
            .unwrap();
        file.write_all(&(channels * 2).to_le_bytes()).unwrap();
        file.write_all(&16_u16.to_le_bytes()).unwrap();
        file.write_all(b"data").unwrap();
        file.write_all(&data_bytes.to_le_bytes()).unwrap();
        {
            let mut writer = io::BufWriter::new(&mut file);
            for frame in 0..written_frames {
                let left = sample(frame as usize);
                writer.write_all(&left.to_le_bytes()).unwrap();
                if channels == 2 {
                    writer.write_all(&(-left).to_le_bytes()).unwrap();
                }
            }
            writer.flush().unwrap();
        }
        // Extend on disk without assembling either encoded data or PCM in memory.
        file.set_len(44 + u64::from(data_bytes)).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
    }
}

fn wait_for(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "decoder did not make expected progress"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn assert_silent_without_allocations(source: &mut Source, active: bool) {
    let mut left = [1.0; QUANTUM];
    let mut right = [1.0; QUANTUM];
    let (counts, result) = alloc_counter::count_alloc(|| source.fill(&mut [&mut left, &mut right]));
    assert_eq!(counts, (0, 0, 0));
    assert_eq!(result, active);
    assert_eq!(left, [0.0; QUANTUM]);
    assert_eq!(right, [0.0; QUANTUM]);
}

#[test]
fn allocation_instrumentation_is_active() {
    let (counts, data) = alloc_counter::count_alloc(|| std::hint::black_box(vec![1_u8; 1024]));
    assert!(
        counts.0 > 0,
        "allocator instrumentation must actually be installed"
    );
    drop(data);
}

#[test]
fn ten_minute_file_is_only_partly_read_and_plays_through_gain() {
    let fixture = Fixture::wav(SAMPLE_RATE * 600, 2, 1024);
    assert_eq!(std::fs::metadata(&fixture.0).unwrap().len(), 115_200_044);
    let (mut stream, source) = FileStream::open(fixture.0.clone()).unwrap();
    stream.wait_for_prefill(Duration::from_secs(5)).unwrap();
    wait_for(|| {
        stream.stats.decoded.load(Ordering::Acquire) == ((QUEUE_BLOCKS + 1) * QUANTUM) as u64
    });
    let read_before = stream.stats.bytes_read.load(Ordering::Acquire);
    assert!(read_before <= (44 + READ_BUFFER_BYTES + (QUEUE_BLOCKS + 1) * QUANTUM * 4) as u64);
    assert!(!stream.stats.worker_done.load(Ordering::Acquire));

    let mut context = OfflineAudioContext::new(2, 1024, SAMPLE_RATE as f32);
    let node = attach(&context, source);
    let gain = context.create_gain();
    gain.gain().set_value(0.25);
    node.connect(&gain);
    gain.connect(&context.destination());
    stream.play();
    let output = context.start_rendering_sync();
    stream.stop().unwrap();
    for frame in 0..1024 {
        let expected = f32::from(sample(frame)) / 32768.0 * 0.25;
        assert_eq!(output.get_channel_data(0)[frame], expected);
        assert_eq!(output.get_channel_data(1)[frame], -expected);
    }
    assert_eq!(stream.stats.underruns.load(Ordering::Acquire), 0);
    assert_eq!(stream.stats.callback_allocations.load(Ordering::Acquire), 0);
    let decoded = stream.stats.decoded.load(Ordering::Acquire);
    assert!(decoded <= (1024 + (QUEUE_BLOCKS + 1) * QUANTUM) as u64);
    assert!(
        stream.stats.bytes_read.load(Ordering::Acquire)
            <= 44 + decoded * 4 + READ_BUFFER_BYTES as u64
    );
    assert!(
        stream.worker.is_none(),
        "stop must join, not just signal cancellation"
    );
    println!(
        "long file: 115200044 encoded bytes; {read_before} bytes read before play; \
         {decoded} frames decoded for 1024 rendered; source callback heap operations: 0"
    );
}

#[test]
fn online_silent_sink_reaches_eof_and_close_cancels_an_active_source() {
    for play_to_end in [true, false] {
        let frames = if play_to_end { 4097 } else { 48_000 };
        let fixture = Fixture::wav(frames, 2, 0);
        let (mut stream, source) = FileStream::open(fixture.0.clone()).unwrap();
        stream.wait_for_prefill(Duration::from_secs(5)).unwrap();
        let context = AudioContext::new(AudioContextOptions {
            sample_rate: Some(SAMPLE_RATE as f32),
            sink_id: "none".into(),
            ..AudioContextOptions::default()
        });
        let node = attach(&context, source);
        let gain = context.create_gain();
        gain.gain().set_value(0.25);
        node.connect(&gain);
        gain.connect(&context.destination());
        stream.play();
        if play_to_end {
            wait_for(|| stream.stats.finished.load(Ordering::Acquire));
            assert_eq!(
                stream.stats.rendered.load(Ordering::Acquire),
                u64::from(frames)
            );
            stream.join().unwrap();
        } else {
            wait_for(|| stream.stats.rendered.load(Ordering::Acquire) > 0);
        }
        context.close_sync();
        drop(node);
        drop(gain);
        drop(context);
        wait_for(|| stream.stats.cancelled.load(Ordering::Acquire));
        stream.stop().unwrap();
        assert!(stream.worker.is_none());
        assert_eq!(stream.stats.callback_allocations.load(Ordering::Acquire), 0);
    }
}

#[test]
fn gain_preserves_order_across_refills_and_the_partial_final_block() {
    for channels in [1, 2] {
        let frames = (40 * QUANTUM + 17) as u32;
        let fixture = Fixture::wav(frames, channels, frames);
        let (mut stream, source) = FileStream::open(fixture.0.clone()).unwrap();
        let stats = Arc::clone(&stream.stats);
        stream.wait_for_prefill(Duration::from_secs(5)).unwrap();
        let blocks = (frames as usize).div_ceil(QUANTUM);
        let mut context = OfflineAudioContext::new(2, (blocks + 1) * QUANTUM, SAMPLE_RATE as f32);
        let node = attach(&context, source);
        let gain = context.create_gain();
        gain.gain().set_value(0.5);
        node.connect(&gain);
        gain.connect(&context.destination());
        // Offline rendering outruns disk by design. Wait in control-side suspend
        // callbacks, never inside the source render callback. Starvation is tested separately.
        for block in 0..=blocks {
            let stats = Arc::clone(&stats);
            context.suspend_sync(
                (block as f64 - 0.25).max(0.0) * QUANTUM as f64 / f64::from(SAMPLE_RATE),
                move |_| {
                    wait_for(|| {
                        stats.queued.load(Ordering::Acquire) >= (block + 1) as u64
                            || stats.worker_done.load(Ordering::Acquire)
                    });
                },
            );
        }
        stream.play();
        let output = context.start_rendering_sync();
        stream.join().unwrap();
        for frame in 0..frames as usize {
            let expected = f32::from(sample(frame)) / 32768.0 * 0.5;
            assert_eq!(
                output.get_channel_data(0)[frame],
                expected,
                "left frame {frame}"
            );
            let right = if channels == 1 { expected } else { -expected };
            assert_eq!(
                output.get_channel_data(1)[frame],
                right,
                "right frame {frame}"
            );
        }
        for channel in 0..2 {
            assert!(output.get_channel_data(channel)[frames as usize..]
                .iter()
                .all(|&x| x == 0.0));
        }
        assert_eq!(stats.rendered.load(Ordering::Acquire), u64::from(frames));
        assert!(stats.finished.load(Ordering::Acquire));
        assert!(!stats.worker_failed.load(Ordering::Acquire));
        assert_eq!(stats.underruns.load(Ordering::Acquire), 0);
        assert_eq!(stats.callback_allocations.load(Ordering::Acquire), 0);
    }
}

#[test]
fn starvation_is_silence_then_recovery_then_eof_without_allocations() {
    let (mut producer, consumer) = RingBuffer::new(QUEUE_BLOCKS);
    let stats = Arc::new(Stats::default());
    let mut source = Source {
        consumer,
        stats: Arc::clone(&stats),
    };
    assert_silent_without_allocations(&mut source, true); // Not started.
    stats.playing.store(true, Ordering::Release);
    assert_silent_without_allocations(&mut source, true); // Starved, still alive.
    assert_eq!(stats.underruns.load(Ordering::Acquire), 1);
    assert_eq!(stats.rendered.load(Ordering::Acquire), 0);
    assert!(producer
        .push(Block {
            samples: [[0.25; QUANTUM]; 2],
            frames: QUANTUM
        })
        .is_ok());
    // EOF publication must not discard an already queued final block.
    stats.worker_done.store(true, Ordering::Release);
    let mut left = [0.0; QUANTUM];
    let mut right = [0.0; QUANTUM];
    let (counts, active) = alloc_counter::count_alloc(|| source.fill(&mut [&mut left, &mut right]));
    assert_eq!(counts, (0, 0, 0));
    assert!(active);
    assert_eq!(left, [0.25; QUANTUM]);
    assert_eq!(right, left);
    assert!(!stats.finished.load(Ordering::Acquire));
    assert_silent_without_allocations(&mut source, false);
    assert!(stats.finished.load(Ordering::Acquire));
}

#[test]
fn stop_joins_a_backpressured_worker_and_silences_queued_audio() {
    let fixture = Fixture::wav(48_000, 2, 0);
    let (mut stream, mut source) = FileStream::open(fixture.0.clone()).unwrap();
    wait_for(|| {
        stream.stats.decoded.load(Ordering::Acquire) == ((QUEUE_BLOCKS + 1) * QUANTUM) as u64
    });
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let stopper = thread::spawn(move || {
        stream.stop().unwrap();
        stream.stop().unwrap(); // Idempotent.
        assert!(stream.worker.is_none());
        done_tx.send(()).unwrap();
    });
    done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("stop hung on a full queue");
    stopper.join().unwrap();
    assert_silent_without_allocations(&mut source, false);
    assert!(source.stats.worker_done.load(Ordering::Acquire));
}

#[test]
fn dropping_owner_joins_and_dropping_source_cancels_without_joining() {
    let fixture = Fixture::wav(48_000, 2, 0);
    let (stream, mut source) = FileStream::open(fixture.0.clone()).unwrap();
    stream.wait_for_prefill(Duration::from_secs(5)).unwrap();
    drop(stream);
    assert!(source.stats.worker_done.load(Ordering::Acquire));
    assert_silent_without_allocations(&mut source, false);

    let (mut stream, source) = FileStream::open(fixture.0.clone()).unwrap();
    stream.wait_for_prefill(Duration::from_secs(5)).unwrap();
    drop(source);
    wait_for(|| stream.stats.worker_done.load(Ordering::Acquire));
    stream.join().unwrap();
    assert!(stream.worker.is_none());
}

#[test]
fn dropping_unrendered_graph_releases_source_then_owner_joins() {
    let fixture = Fixture::wav(48_000, 2, 0);
    let (mut stream, source) = FileStream::open(fixture.0.clone()).unwrap();
    stream.wait_for_prefill(Duration::from_secs(5)).unwrap();
    let context = OfflineAudioContext::new(2, 128, SAMPLE_RATE as f32);
    let node = attach(&context, source);
    drop(node);
    drop(context);
    wait_for(|| stream.stats.cancelled.load(Ordering::Acquire));
    stream.stop().unwrap();
    assert!(stream.worker.is_none());
}

#[test]
fn malformed_missing_unsupported_and_truncated_files_return_errors_and_join() {
    let fixture = Fixture::wav(1024, 2, 0);
    let mut file = File::options().write(true).open(&fixture.0).unwrap();
    file.seek(SeekFrom::Start(24)).unwrap();
    file.write_all(&44_100_u32.to_le_bytes()).unwrap();
    assert!(FileStream::open(fixture.0.clone()).is_err());
    file.set_len(0).unwrap();
    assert!(FileStream::open(fixture.0.clone()).is_err());
    assert!(FileStream::open(fixture.0.with_extension("missing")).is_err());
    drop(file);

    let fixture = Fixture::wav(1024, 2, 0);
    let file = File::options().write(true).open(&fixture.0).unwrap();
    // Header promises 1024 frames, but the second decode block is truncated.
    file.set_len(44 + (QUANTUM * 4 + 3) as u64).unwrap();
    drop(file);
    let (mut stream, mut source) = FileStream::open(fixture.0.clone()).unwrap();
    wait_for(|| stream.stats.worker_done.load(Ordering::Acquire));
    assert!(stream.join().is_err());
    assert!(stream.worker.is_none());
    assert!(stream.stats.worker_failed.load(Ordering::Acquire));
    stream.play();
    let mut left = [0.0; QUANTUM];
    let mut right = [0.0; QUANTUM];
    assert!(source.fill(&mut [&mut left, &mut right])); // Last complete block remains.
    assert_silent_without_allocations(&mut source, false);
}
