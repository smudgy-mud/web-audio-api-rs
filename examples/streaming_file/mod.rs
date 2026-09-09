use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rtrb::{Consumer, Producer, RingBuffer};
use web_audio_api::context::BaseAudioContext;
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};

pub const SAMPLE_RATE: u32 = 48_000;
const QUANTUM: usize = 128;
const QUEUE_BLOCKS: usize = 16;
const READ_BUFFER_BYTES: usize = 8 * 1024;
type Failure = Box<dyn std::error::Error + Send + Sync>;

// No owned heap storage travels through the render callback.
struct Block {
    samples: [[f32; QUANTUM]; 2],
    frames: usize,
}

#[derive(Default)]
pub struct Stats {
    playing: AtomicBool,
    cancelled: AtomicBool,
    worker_done: AtomicBool,
    worker_failed: AtomicBool,
    pub finished: AtomicBool,
    pub decoded: AtomicU64,
    queued: AtomicU64,
    pub rendered: AtomicU64,
    pub underruns: AtomicU64,
    bytes_read: AtomicU64,
    #[cfg(test)]
    callback_allocations: AtomicU64,
}

struct CountedFile {
    file: File,
    stats: Arc<Stats>,
}

impl Read for CountedFile {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = self.file.read(bytes)?;
        self.stats
            .bytes_read
            .fetch_add(count as u64, Ordering::Relaxed);
        Ok(count)
    }
}

/// The control-side owner. Never transfer this handle to the audio callback.
/// Stop and Drop cancel and JOIN the worker, including while its queue is full.
pub struct FileStream {
    worker: Option<JoinHandle<Result<(), Failure>>>,
    pub stats: Arc<Stats>,
}

impl FileStream {
    pub fn open(path: PathBuf) -> Result<(Self, Source), Failure> {
        let (producer, consumer) = RingBuffer::new(QUEUE_BLOCKS);
        let stats = Arc::new(Stats::default());
        let shared = Arc::clone(&stats);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("wav-stream-proof".into())
            .spawn(move || {
                // Opening, header parsing, reads, decoding, and errors stay off render.
                let result = (|| {
                    let file = File::open(path)?;
                    if !file.metadata()?.is_file() {
                        return Err("source must be a regular file".into());
                    }
                    let reader = hound::WavReader::new(BufReader::with_capacity(
                        READ_BUFFER_BYTES,
                        CountedFile {
                            file,
                            stats: Arc::clone(&shared),
                        },
                    ))?;
                    let spec = reader.spec();
                    if spec.sample_rate != SAMPLE_RATE
                        || !(1..=2).contains(&spec.channels)
                        || spec.bits_per_sample != 16
                        || spec.sample_format != hound::SampleFormat::Int
                    {
                        return Err(
                            "prototype requires mono/stereo 16-bit PCM WAV at 48000 Hz".into()
                        );
                    }
                    // Only success crosses the startup channel. The join result owns errors.
                    ready_tx
                        .send(())
                        .map_err(|_| "stream owner dropped during open")?;
                    decode(reader, producer, &shared)
                })();
                shared
                    .worker_failed
                    .store(result.is_err(), Ordering::Release);
                shared.worker_done.store(true, Ordering::Release);
                result
            })?;
        let mut owner = Self {
            worker: Some(worker),
            stats: Arc::clone(&stats),
        };
        if ready_rx.recv().is_err() {
            owner.join()?;
            return Err("decoder exited before opening the stream".into());
        }
        Ok((owner, Source { consumer, stats }))
    }

    /// This is a control-side convenience for the proof, never an audio callback wait.
    pub fn wait_for_prefill(&self, timeout: Duration) -> Result<(), Failure> {
        let deadline = Instant::now() + timeout;
        while self.stats.queued.load(Ordering::Acquire) < QUEUE_BLOCKS as u64
            && !self.stats.worker_done.load(Ordering::Acquire)
        {
            if Instant::now() >= deadline {
                return Err("decoder prefill timed out".into());
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    pub fn play(&self) {
        self.stats.playing.store(true, Ordering::Release);
    }

    pub fn stop(&mut self) -> Result<(), Failure> {
        self.stats.cancelled.store(true, Ordering::Release);
        if let Some(worker) = &self.worker {
            worker.thread().unpark();
        }
        self.join()
    }

    fn join(&mut self) -> Result<(), Failure> {
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| "decoder worker panicked")??;
        }
        Ok(())
    }
}

impl Drop for FileStream {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn decode(
    mut reader: hound::WavReader<BufReader<CountedFile>>,
    mut producer: Producer<Block>,
    stats: &Stats,
) -> Result<(), Failure> {
    let channels = reader.spec().channels;
    let mut samples = reader.samples::<i16>();
    loop {
        if stats.cancelled.load(Ordering::Acquire) || producer.is_abandoned() {
            return Ok(());
        }
        let mut block = Block {
            samples: [[0.0; QUANTUM]; 2],
            frames: 0,
        };
        for frame in 0..QUANTUM {
            let Some(left) = samples.next() else { break };
            let left = f32::from(left?) / 32768.0;
            let right = if channels == 2 {
                f32::from(samples.next().ok_or("incomplete stereo frame")??) / 32768.0
            } else {
                left
            };
            block.samples[0][frame] = left;
            block.samples[1][frame] = right;
            block.frames += 1;
        }
        if block.frames == 0 {
            return Ok(());
        }
        stats
            .decoded
            .fetch_add(block.frames as u64, Ordering::Release);
        loop {
            if stats.cancelled.load(Ordering::Acquire) || producer.is_abandoned() {
                return Ok(());
            }
            match producer.push(block) {
                Ok(()) => {
                    stats.queued.fetch_add(1, Ordering::Release);
                    break;
                }
                Err(rtrb::PushError::Full(returned)) => {
                    block = returned;
                    // Render never signals a condvar or unparks a thread. Only the
                    // worker polls, and stop can wake it immediately.
                    thread::park_timeout(Duration::from_millis(1));
                }
            }
        }
    }
}

pub struct Source {
    consumer: Consumer<Block>,
    stats: Arc<Stats>,
}

impl Source {
    fn fill(&mut self, output: &mut [&mut [f32]]) -> bool {
        for channel in output.iter_mut() {
            channel.fill(0.0);
        }
        if self.stats.cancelled.load(Ordering::Acquire) {
            self.stats.finished.store(true, Ordering::Release);
            return false;
        }
        if !self.stats.playing.load(Ordering::Acquire) {
            return true;
        }
        let block = match self.consumer.pop() {
            Ok(block) => Some(block),
            Err(_) if self.stats.worker_done.load(Ordering::Acquire) => {
                // The first empty observation can precede the final worker push.
                // Acquire terminal publication before checking the queue again.
                self.consumer.pop().ok()
            }
            Err(_) => {
                self.stats.underruns.fetch_add(1, Ordering::Relaxed);
                return true; // Silence advances the graph, not the source cursor.
            }
        };
        let Some(block) = block else {
            self.stats.finished.store(true, Ordering::Release);
            return false;
        };
        for (out, input) in output.iter_mut().zip(&block.samples) {
            out.copy_from_slice(input);
        }
        self.stats
            .rendered
            .fetch_add(block.frames as u64, Ordering::Release);
        true
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        // No joining here. The owner retains the worker handle and joins off render.
        self.stats.cancelled.store(true, Ordering::Release);
    }
}

impl AudioWorkletProcessor for Source {
    type ProcessorOptions = Self;

    fn constructor(source: Self) -> Self {
        source
    }

    fn process<'a, 'b>(
        &mut self,
        _inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        _params: AudioParamValues<'b>,
        _scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        #[cfg(test)]
        {
            let (counts, active) = alloc_counter::count_alloc(|| self.fill(outputs[0]));
            self.stats
                .callback_allocations
                .fetch_add((counts.0 + counts.1 + counts.2) as u64, Ordering::Relaxed);
            active
        }
        #[cfg(not(test))]
        self.fill(outputs[0])
    }
}

pub fn attach(context: &impl BaseAudioContext, source: Source) -> AudioWorkletNode {
    assert_eq!(context.sample_rate(), SAMPLE_RATE as f32);
    AudioWorkletNode::new::<Source>(
        context,
        AudioWorkletNodeOptions {
            number_of_inputs: 0,
            number_of_outputs: 1,
            output_channel_count: vec![2],
            parameter_data: Default::default(),
            processor_options: source,
            audio_node_options: Default::default(),
        },
    )
}

#[cfg(test)]
mod tests;
