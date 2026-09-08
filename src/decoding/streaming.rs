//! Worker-side file decoder with a continuous resampler and bounded PCM residency.

use super::*;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const INPUT_FRAMES: usize = 1024;
const MAX_PACKET_FRAMES: usize = 65_536;
const MAX_PACKET_BYTES: usize = 1024 * 1024;

/// Incrementally decodes a seekable local file into stereo frames at a target rate.
///
/// All enabled Symphonia codecs are available subject to mono/stereo, 8-192 kHz,
/// and per-packet limits. Mono is duplicated. Files are probed by content, allowing
/// M4A containers with metadata at the end to seek without buffering the whole file.
///
/// This API performs blocking I/O and allocations and must run on a decoder worker.
/// PCM storage is bounded by one decoded packet, fixed resampler buffers, and the
/// caller's output slice. It is NOT a complete untrusted-media memory budget:
/// demuxer indexes, codec-private storage, encoded packets before validation, and
/// metadata require separate host policy. No decoder thread is spawned here.
pub struct MediaFileDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    plan: CodecDimensionPlan,
    packet: Option<AudioBuffer>,
    packet_offset: usize,
    source_rate: u32,
    target_rate: u32,
    source_channels: usize,
    input: [Vec<f32>; 2],
    output: Vec<Vec<f32>>,
    resampler: Option<SincFixedIn<f32>>,
    output_offset: usize,
    output_len: usize,
    input_frames: u64,
    emitted_frames: u64,
    eof: bool,
    failed: bool,
}

impl std::fmt::Debug for MediaFileDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaFileDecoder")
            .field("source_rate", &self.source_rate)
            .field("target_rate", &self.target_rate)
            .field("input_frames", &self.input_frames)
            .field("emitted_frames", &self.emitted_frames)
            .finish_non_exhaustive()
    }
}

impl MediaFileDecoder {
    /// Probes an already authorized, opened regular file on the calling worker.
    ///
    /// # Errors
    /// Returns errors for I/O, unsupported formats/dimensions, or corrupt input.
    pub fn new(file: std::fs::File, target_rate: u32) -> Result<Self, BoxError> {
        if !(8000..=192_000).contains(&target_rate) || !file.metadata()?.is_file() {
            return Err("stream requires a regular file and an 8-192 kHz target rate".into());
        }
        let stream =
            symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default());
        let format = symphonia::default::get_probe().probe(
            &Hint::new(),
            stream,
            FormatOptions::default(),
            MetadataOptions::default()
                .limit_tag_bytes(symphonia::core::common::Limit::Maximum(256 * 1024))
                .limit_visual_bytes(symphonia::core::common::Limit::Maximum(0)),
        )?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or("no audio track")?;
        let track_id = track.id;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or("no audio parameters")?;
        let plan = CodecDimensionPlan::try_new(params)?;
        if params
            .channels
            .as_ref()
            .is_some_and(|channels| !(1..=2).contains(&channels.count()))
            || params
                .sample_rate
                .is_some_and(|rate| !(8000..=192_000).contains(&rate))
        {
            return Err("stream requires mono/stereo at 8-192 kHz".into());
        }
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default().verify(true))?;
        let mut this = Self {
            format,
            decoder,
            track_id,
            plan,
            packet: None,
            packet_offset: 0,
            source_rate: 0,
            target_rate,
            source_channels: 0,
            input: [vec![0.0; INPUT_FRAMES], vec![0.0; INPUT_FRAMES]],
            output: Vec::new(),
            resampler: None,
            output_offset: 0,
            output_len: 0,
            input_frames: 0,
            emitted_frames: 0,
            eof: false,
            failed: false,
        };
        this.load_packet()?;
        if this.eof {
            return Err("audio track is empty".into());
        }
        if this.source_rate == target_rate {
            this.output = vec![vec![0.0; INPUT_FRAMES]; 2];
        } else {
            let resampler = SincFixedIn::new(
                f64::from(target_rate) / f64::from(this.source_rate),
                1.0,
                SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 128,
                    window: WindowFunction::BlackmanHarris2,
                },
                INPUT_FRAMES,
                2,
            )?;
            // SincFixedIn withholds output for lookahead; its returned samples
            // already begin at the source origin (within its fractional sample
            // offset). output_delay() describes buffering latency, not a prefix
            // of silent samples to discard. Trimming it would lose real audio.
            this.output = resampler.output_buffer_allocate(true);
            this.resampler = Some(resampler);
        }
        Ok(this)
    }

    /// Decoded file sample rate before conversion.
    #[must_use]
    pub const fn source_sample_rate(&self) -> u32 {
        self.source_rate
    }

    /// Decoded source channel count (one or two).
    #[must_use]
    pub const fn source_channels(&self) -> usize {
        self.source_channels
    }

    /// Reads up to `output.len()` resampled stereo frames. Zero indicates EOF when
    /// the supplied output slice is nonempty.
    /// Resampler history continues across packets and the tail
    /// is flushed to ceil(source_frames * target_rate / source_rate).
    ///
    /// # Errors
    /// Decode errors, midstream dimension changes, and oversized packets are fatal.
    /// After an error the decoder remains failed; callers must discard its output.
    pub fn read(&mut self, output: &mut [[f32; 2]]) -> Result<usize, BoxError> {
        if self.failed {
            return Err("stream decoder has failed".into());
        }
        let result = self.read_inner(output);
        self.failed = result.is_err();
        result
    }

    fn read_inner(&mut self, output: &mut [[f32; 2]]) -> Result<usize, BoxError> {
        let mut count = 0;
        while count < output.len() {
            if self.output_offset == self.output_len && !self.refill()? {
                break;
            }
            let available = (self.output_len - self.output_offset).min(output.len() - count);
            for frame in &mut output[count..count + available] {
                *frame = [
                    self.output[0][self.output_offset],
                    self.output[1][self.output_offset],
                ];
                self.output_offset += 1;
            }
            count += available;
            self.emitted_frames = self
                .emitted_frames
                .checked_add(available as u64)
                .ok_or("stream duration overflow")?;
        }
        Ok(count)
    }

    fn refill(&mut self) -> Result<bool, BoxError> {
        loop {
            let final_frames = || {
                (u128::from(self.input_frames) * u128::from(self.target_rate))
                    .div_ceil(u128::from(self.source_rate))
            };
            if self.eof && u128::from(self.emitted_frames) >= final_frames() {
                return Ok(false);
            }
            let mut filled = 0;
            while filled < INPUT_FRAMES {
                let exhausted = self
                    .packet
                    .as_ref()
                    .is_none_or(|p| self.packet_offset == p.length());
                if exhausted {
                    self.packet = None; // Release the previous packet before decoding another.
                    if self.eof {
                        break;
                    }
                    self.load_packet()?;
                    if self.eof {
                        break;
                    }
                }
                let packet = self.packet.as_ref().ok_or("missing decoded packet")?;
                let take = (packet.length() - self.packet_offset).min(INPUT_FRAMES - filled);
                for channel in 0..2 {
                    let source_channel = channel.min(self.source_channels - 1);
                    self.input[channel][filled..filled + take].copy_from_slice(
                        &packet.get_channel_data(source_channel)
                            [self.packet_offset..self.packet_offset + take],
                    );
                }
                self.packet_offset += take;
                filled += take;
                self.input_frames = self
                    .input_frames
                    .checked_add(take as u64)
                    .ok_or("stream duration overflow")?;
            }
            for channel in &mut self.input {
                channel[filled..].fill(0.0);
            }
            let produced = if let Some(resampler) = &mut self.resampler {
                resampler
                    .process_into_buffer(&self.input, &mut self.output, None)?
                    .1
            } else {
                for (output, input) in self.output.iter_mut().zip(&self.input) {
                    output.copy_from_slice(input);
                }
                filled
            };
            self.output_offset = 0;
            self.output_len = produced;
            if self.eof {
                let final_frames = (u128::from(self.input_frames) * u128::from(self.target_rate))
                    .div_ceil(u128::from(self.source_rate));
                let remaining = final_frames.saturating_sub(u128::from(self.emitted_frames));
                self.output_len = self.output_len.min(
                    self.output_offset
                        + usize::try_from(remaining).unwrap_or(usize::MAX - self.output_offset),
                );
            }
            if self.output_len > self.output_offset {
                return Ok(true);
            }
            if self.eof && self.resampler.is_none() {
                return Ok(false);
            }
        }
    }

    fn load_packet(&mut self) -> Result<(), BoxError> {
        loop {
            let Some(packet) = self.format.next_packet()? else {
                if self.decoder.finalize().verify_ok == Some(false) {
                    return Err("decoded audio checksum failed".into());
                }
                self.eof = true;
                return Ok(());
            };
            if packet.track_id != self.track_id {
                continue;
            }
            if packet.data.len() > MAX_PACKET_BYTES
                || self.plan.packet_frame_bound(&packet)? > MAX_PACKET_FRAMES
            {
                return Err("audio packet exceeds streaming limit".into());
            }
            let input = self.decoder.decode(&packet)?;
            let rate = input.spec().rate();
            let channels = input.spec().channels().count();
            if !(8000..=192_000).contains(&rate)
                || !(1..=2).contains(&channels)
                || input.frames() > MAX_PACKET_FRAMES
            {
                return Err("stream requires mono/stereo at 8-192 kHz within packet limits".into());
            }
            if self.source_rate != 0
                && (self.source_rate != rate || self.source_channels != channels)
            {
                return Err("audio dimensions changed midstream".into());
            }
            self.source_rate = rate;
            self.source_channels = channels;
            if input.frames() == 0 {
                continue;
            }
            self.packet = Some(input.into());
            self.packet_offset = 0;
            return Ok(());
        }
    }
}
