use std::alloc::Layout;
use std::error::Error;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::sync::Arc;

use crate::buffer::{AudioBuffer, ChannelData};
use crate::context::BaseAudioContext;

use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::well_known::*;
use symphonia::core::codecs::audio::{
    AudioCodecId, AudioCodecParameters, AudioDecoder, AudioDecoderOptions, FinalizeResult,
};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::BufReader;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::packet::Packet;
use symphonia_common::apple::audio::alac::MagicCookie;
use symphonia_common::mpeg::audio::AudioSpecificConfig;
use symphonia_common::xiph::audio::flac::StreamInfo;

type BoxError = Box<dyn Error + Send + Sync>;

#[cfg(test)]
thread_local! {
    static WRAPPER_PCM_ALLOCATION_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[inline]
fn record_wrapper_pcm_allocation() {
    #[cfg(test)]
    WRAPPER_PCM_ALLOCATION_COUNT.set(WRAPPER_PCM_ALLOCATION_COUNT.get() + 1);
}

#[cfg(test)]
fn wrapper_pcm_allocation_count() -> usize {
    WRAPPER_PCM_ALLOCATION_COUNT.get()
}

/// Error type returned by a [`DecodeBudget`] implementation.
pub type DecodeBudgetError = Box<dyn Error + Send + Sync>;

/// A category of result dimensions or wrapper-owned PCM requested by budgeted decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeBudgetKind {
    /// Number of channels in the decoded result, counted once across all storage copies.
    DecodedChannels,
    /// Number of frames per channel in the decoded result, counted once across all storage
    /// copies.
    DecodedFrames,
    /// Logical samples in the decoded result, counted once despite temporary copies.
    DecodedSamples,
    /// Canonical planar `f32` storage before resampling.
    CanonicalPcmBytes,
    /// Canonical planar `f32` storage allocated by resampling.
    ResamplePcmBytes,
}

/// A fallible reservation request made during trusted-media decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DecodeBudgetRequest {
    kind: DecodeBudgetKind,
    amount: u64,
}

impl DecodeBudgetRequest {
    const fn new(kind: DecodeBudgetKind, amount: u64) -> Self {
        Self { kind, amount }
    }

    /// The resource being reserved.
    pub const fn kind(self) -> DecodeBudgetKind {
        self.kind
    }

    /// Bytes, channels, frames, or samples, according to [`Self::kind`].
    pub const fn amount(self) -> u64 {
        self.amount
    }
}

/// An owned reservation returned by [`DecodeBudget::try_reserve`].
///
/// Dropping the guard must release its remaining reservation without blocking
/// or panicking. The decode path may reduce a conservative reservation to an
/// exact amount after a packet has decoded. `shrink_to` must likewise be
/// nonblocking, must not panic, and must never increase the reservation.
pub trait DecodeBudgetReservation: Send + Sync + 'static {
    /// Reduces the reservation to `amount`, releasing the difference.
    fn shrink_to(&mut self, amount: u64);
}

/// A nonblocking trusted-media budget used by synchronous audio decoding.
///
/// Every successful [`Self::try_reserve`] call returns an owned guard. The
/// guard makes unwind and cancellation cleanup automatic. Implementations may
/// charge related kinds to aggregate ceilings while retaining the kind for
/// diagnostics. Requests cover decoded-result dimensions and PCM storage owned
/// by this crate. They do not cover the decoder library's private allocations,
/// encoded packets, demuxer state, allocator overhead, or copy-on-write duplication.
pub trait DecodeBudget: Send + Sync + 'static {
    /// Attempts a reservation without waiting or blocking.
    ///
    /// The returned error is preserved as the source of
    /// [`BudgetedDecodeError::BudgetRefused`].
    fn try_reserve(
        &self,
        request: DecodeBudgetRequest,
    ) -> Result<Box<dyn DecodeBudgetReservation>, DecodeBudgetError>;
}

/// Additive synchronous budgeted decoding for every [`BaseAudioContext`].
///
/// This extension trait intentionally adds no asynchronous overload: embedders
/// should acquire their decode-job permit before spawning blocking work, then
/// call this method on that worker.
pub trait BaseAudioContextDecodeBudgetExt: BaseAudioContext {
    /// Decodes trusted audio while charging result dimensions, wrapper-owned canonical chunks
    /// and assembly, and source-plus-target resampling residency.
    ///
    /// This does not charge private decoder or demuxer allocations, encoded packets, allocator
    /// overhead, spare capacity or reallocation transients, or copy-on-write duplication after
    /// the returned buffer is cloned and mutated.
    fn decode_audio_data_sync_with_budget<R: Read + Send + Sync + 'static>(
        &self,
        input: R,
        budget: Arc<dyn DecodeBudget>,
    ) -> Result<AudioBuffer, BudgetedDecodeError> {
        decode_media_data_with_budget(input, self.sample_rate(), budget)
    }
}

impl<T: BaseAudioContext + ?Sized> BaseAudioContextDecodeBudgetExt for T {}

/// An error from the synchronous budgeted decode path.
#[derive(Debug)]
#[non_exhaustive]
pub enum BudgetedDecodeError {
    /// The supplied budget rejected a reservation.
    BudgetRefused {
        /// The rejected resource request.
        request: DecodeBudgetRequest,
        /// The error returned by the budget implementation.
        source: BoxError,
    },
    /// A byte, sample, frame, or resampling calculation overflowed.
    ArithmeticOverflow {
        /// The calculation that could not be represented safely.
        operation: &'static str,
    },
    /// A requested or decoded sample rate cannot be represented by an [`AudioBuffer`].
    InvalidSampleRate {
        /// The invalid sample rate in hertz.
        sample_rate: f32,
    },
    /// Probing, decoder construction, or decoding failed.
    Decode {
        /// The underlying media or codec error.
        source: BoxError,
    },
    /// A decoder returned output beyond its pre-decoding channel or frame bound.
    DecoderInvariant {
        /// The violated codec-dimension invariant.
        description: &'static str,
    },
}

impl std::fmt::Display for BudgetedDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetRefused { request, source } => {
                write!(f, "decode budget refused {request:?}: {source}")
            }
            Self::ArithmeticOverflow { operation } => {
                write!(f, "decode resource calculation overflowed: {operation}")
            }
            Self::InvalidSampleRate { sample_rate } => {
                write!(f, "invalid decoded-audio sample rate: {sample_rate}")
            }
            Self::Decode { source } => source.fmt(f),
            Self::DecoderInvariant { description } => {
                write!(
                    f,
                    "decoder exceeded its pre-decoding dimension bound: {description}"
                )
            }
        }
    }
}

impl Error for BudgetedDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BudgetRefused { source, .. } | Self::Decode { source } => Some(source.as_ref()),
            Self::ArithmeticOverflow { .. }
            | Self::InvalidSampleRate { .. }
            | Self::DecoderInvariant { .. } => None,
        }
    }
}

impl BudgetedDecodeError {
    fn decode(source: impl Error + Send + Sync + 'static) -> Self {
        Self::Decode {
            source: Box::new(source),
        }
    }

    fn boxed_decode(source: BoxError) -> Self {
        Self::Decode { source }
    }

    const fn overflow(operation: &'static str) -> Self {
        Self::ArithmeticOverflow { operation }
    }

    const fn invariant(description: &'static str) -> Self {
        Self::DecoderInvariant { description }
    }
}

struct DecodeLease {
    request: DecodeBudgetRequest,
    reservation: Box<dyn DecodeBudgetReservation>,
}

impl std::fmt::Debug for DecodeLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeLease")
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

impl DecodeLease {
    fn shrink_by(&mut self, amount: u64) {
        debug_assert!(amount <= self.request.amount);
        if amount == 0 {
            return;
        }

        let remaining = self.request.amount - amount;
        self.reservation.shrink_to(remaining);
        self.request.amount = remaining;
    }
}

#[derive(Debug, Default)]
struct LeaseSet {
    leases: Vec<DecodeLease>,
}

impl LeaseSet {
    fn try_reserve(
        &mut self,
        budget: &Arc<dyn DecodeBudget>,
        request: DecodeBudgetRequest,
    ) -> Result<(), BudgetedDecodeError> {
        if request.amount == 0 {
            return Ok(());
        }

        let next_len = self
            .leases
            .len()
            .checked_add(1)
            .ok_or_else(|| BudgetedDecodeError::overflow("reservation guard count"))?;
        checked_array_layout::<DecodeLease>(next_len, "reservation guard layout")?;
        self.leases
            .try_reserve_exact(1)
            .map_err(BudgetedDecodeError::decode)?;

        let reservation = budget
            .try_reserve(request)
            .map_err(|source| BudgetedDecodeError::BudgetRefused { request, source })?;

        self.leases.push(DecodeLease {
            request,
            reservation,
        });
        Ok(())
    }

    fn append(&mut self, mut other: Self) -> Result<(), BudgetedDecodeError> {
        let next_len = self
            .leases
            .len()
            .checked_add(other.leases.len())
            .ok_or_else(|| BudgetedDecodeError::overflow("reservation guard count"))?;
        checked_array_layout::<DecodeLease>(next_len, "reservation guard layout")?;
        self.leases
            .try_reserve_exact(other.leases.len())
            .map_err(BudgetedDecodeError::decode)?;
        self.leases.append(&mut other.leases);
        Ok(())
    }

    fn total(&self, kind: DecodeBudgetKind) -> Result<u64, BudgetedDecodeError> {
        self.leases
            .iter()
            .filter(|lease| lease.request.kind == kind)
            .try_fold(0u64, |total, lease| {
                total
                    .checked_add(lease.request.amount)
                    .ok_or_else(|| BudgetedDecodeError::overflow("reservation total"))
            })
    }

    fn shrink_kind_to(
        &mut self,
        kind: DecodeBudgetKind,
        target: u64,
    ) -> Result<(), BudgetedDecodeError> {
        let total = self.total(kind)?;
        if target > total {
            return Err(BudgetedDecodeError::invariant(
                "reservation shrink target exceeds held amount",
            ));
        }

        let mut release = total - target;
        for lease in self
            .leases
            .iter_mut()
            .rev()
            .filter(|lease| lease.request.kind == kind)
        {
            let amount = release.min(lease.request.amount);
            lease.shrink_by(amount);
            release -= amount;
            if release == 0 {
                break;
            }
        }
        debug_assert_eq!(release, 0);
        Ok(())
    }

    fn release_kind(&mut self, kind: DecodeBudgetKind) -> Result<(), BudgetedDecodeError> {
        self.shrink_kind_to(kind, 0)
    }
}

fn checked_product(operation: &'static str, factors: &[usize]) -> Result<u64, BudgetedDecodeError> {
    factors
        .iter()
        .try_fold(1u128, |product, &factor| {
            product
                .checked_mul(factor as u128)
                .ok_or_else(|| BudgetedDecodeError::overflow(operation))
        })
        .and_then(|product| {
            u64::try_from(product).map_err(|_| BudgetedDecodeError::overflow(operation))
        })
}

fn checked_array_layout<T>(
    length: usize,
    operation: &'static str,
) -> Result<Layout, BudgetedDecodeError> {
    Layout::array::<T>(length).map_err(|_| BudgetedDecodeError::overflow(operation))
}

fn checked_frames(value: u64, operation: &'static str) -> Result<usize, BudgetedDecodeError> {
    usize::try_from(value).map_err(|_| BudgetedDecodeError::overflow(operation))
}

/// Decodes an [`AudioBuffer`] while reserving result dimensions and wrapper-owned PCM.
///
/// This is the synchronous integration seam for embedders that need finite
/// decoded-result limits for trusted media. It intentionally does not account for Symphonia's
/// private decoder workspace or capacity/reallocation transients, encoded packets, demuxer
/// metadata, codec setup tables, allocator metadata, or allocator rounding. Existing unbudgeted
/// decoding APIs retain their current behavior. The returned buffer owns its exact channel,
/// frame, logical-sample, and wrapper-owned canonical byte reservations, and clones share that
/// lease until the last clone is dropped. Copy-on-write duplication after a clone is mutated is
/// not separately charged by this phase; callers requiring that accounting must mediate mutation.
pub(crate) fn decode_media_data_with_budget<R: std::io::Read + Send + Sync + 'static>(
    input: R,
    target_sample_rate: f32,
    budget: Arc<dyn DecodeBudget>,
) -> Result<AudioBuffer, BudgetedDecodeError> {
    if !crate::is_valid_sample_rate(target_sample_rate) {
        return Err(BudgetedDecodeError::InvalidSampleRate {
            sample_rate: target_sample_rate,
        });
    }

    let mut decoder = BudgetedMediaDecoder::try_new(input, Arc::clone(&budget))?;
    let mut chunks = Vec::new();
    let mut sample_rate = None;
    let mut channel_count = None;
    let mut total_frames = 0usize;

    while let Some(chunk) = decoder.next_chunk()? {
        match sample_rate {
            Some(rate) if rate != chunk.sample_rate => {
                return Err(BudgetedDecodeError::decode(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "decoded audio sample rate changed midstream",
                )));
            }
            Some(_) => {}
            None => sample_rate = Some(chunk.sample_rate),
        }

        match channel_count {
            Some(count) if count != chunk.channels.len() => {
                return Err(BudgetedDecodeError::decode(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "decoded audio channel count changed midstream",
                )));
            }
            Some(_) => {}
            None => channel_count = Some(chunk.channels.len()),
        }

        total_frames = total_frames
            .checked_add(chunk.frames)
            .ok_or_else(|| BudgetedDecodeError::overflow("decoded frame total"))?;
        let next_chunk_count = chunks
            .len()
            .checked_add(1)
            .ok_or_else(|| BudgetedDecodeError::overflow("decoded chunk count"))?;
        checked_array_layout::<PlanarChunk>(next_chunk_count, "decoded chunk list layout")?;
        chunks
            .try_reserve_exact(1)
            .map_err(BudgetedDecodeError::decode)?;
        chunks.push(chunk);
    }

    let mut final_leases = decoder.take_result_leases();
    // Drop the decoder before wrapper-owned final assembly and resampling.
    drop(decoder);

    if chunks.is_empty() {
        final_leases.shrink_kind_to(DecodeBudgetKind::DecodedChannels, 1)?;
        checked_array_layout::<Vec<f32>>(1, "empty result channel layout")?;
        checked_array_layout::<ChannelData>(1, "empty result channel-data layout")?;
        let mut buffer = AudioBuffer::from(vec![vec![]], target_sample_rate);
        buffer.set_accounting_lease(Arc::new(final_leases));
        return Ok(buffer);
    }

    let sample_rate = sample_rate.expect("a decoded chunk has a sample rate");
    let channel_count = channel_count.expect("a decoded chunk has channels");
    let source_bytes = checked_product(
        "canonical source bytes",
        &[channel_count, total_frames, size_of::<f32>()],
    )?;
    checked_array_layout::<f32>(total_frames, "canonical source plane layout")?;
    checked_array_layout::<Vec<f32>>(channel_count, "canonical source outer layout")?;
    checked_array_layout::<ChannelData>(channel_count, "canonical channel-data layout")?;

    final_leases.try_reserve(
        &budget,
        DecodeBudgetRequest::new(DecodeBudgetKind::CanonicalPcmBytes, source_bytes),
    )?;

    record_wrapper_pcm_allocation();
    let mut channels = (0..channel_count)
        .map(|_| vec![0.0; total_frames])
        .collect::<Vec<_>>();
    let mut offset = 0usize;

    for mut chunk in chunks {
        let end = offset
            .checked_add(chunk.frames)
            .ok_or_else(|| BudgetedDecodeError::overflow("chunk assembly offset"))?;
        for (destination, source) in channels.iter_mut().zip(&chunk.channels) {
            destination[offset..end].copy_from_slice(source);
        }
        offset = end;

        // Drop the chunk PCM before releasing its reservation. The final source allocation now
        // represents these bytes; logical samples remain held.
        chunk.channels.clear();
        chunk
            .leases
            .release_kind(DecodeBudgetKind::CanonicalPcmBytes)?;
        final_leases.append(chunk.leases)?;
    }

    let mut buffer = AudioBuffer::from(channels, sample_rate);

    if float_eq::float_eq!(sample_rate, target_sample_rate, abs <= 0.1) {
        buffer.resample(target_sample_rate);
    } else {
        let target_frames = buffer
            .checked_resample_length(target_sample_rate)
            .ok_or_else(|| BudgetedDecodeError::overflow("resampled frame count"))?;
        let source_samples =
            checked_product("source logical samples", &[channel_count, total_frames])?;
        let target_samples =
            checked_product("resampled logical samples", &[channel_count, target_frames])?;

        checked_array_layout::<f32>(target_frames, "resampled channel plane layout")?;
        checked_array_layout::<Vec<f32>>(channel_count, "resampled channel outer layout")?;

        if target_frames > total_frames {
            final_leases.try_reserve(
                &budget,
                DecodeBudgetRequest::new(
                    DecodeBudgetKind::DecodedFrames,
                    u64::try_from(target_frames - total_frames)
                        .map_err(|_| BudgetedDecodeError::overflow("resampled frame delta"))?,
                ),
            )?;
        }

        if target_samples > source_samples {
            final_leases.try_reserve(
                &budget,
                DecodeBudgetRequest::new(
                    DecodeBudgetKind::DecodedSamples,
                    target_samples - source_samples,
                ),
            )?;
        }

        let target_bytes = checked_product(
            "resampled PCM bytes",
            &[channel_count, target_frames, size_of::<f32>()],
        )?;
        final_leases.try_reserve(
            &budget,
            DecodeBudgetRequest::new(DecodeBudgetKind::ResamplePcmBytes, target_bytes),
        )?;

        buffer.resample_with_length(target_sample_rate, target_frames);
        final_leases.release_kind(DecodeBudgetKind::CanonicalPcmBytes)?;
        final_leases.shrink_kind_to(
            DecodeBudgetKind::DecodedFrames,
            u64::try_from(target_frames)
                .map_err(|_| BudgetedDecodeError::overflow("resampled frame count"))?,
        )?;
        final_leases.shrink_kind_to(DecodeBudgetKind::DecodedSamples, target_samples)?;
    }

    buffer.set_accounting_lease(Arc::new(final_leases));
    Ok(buffer)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PacketFrameBound {
    Fixed(usize),
    Adpcm {
        frames_per_block: usize,
        max_frames_per_packet: usize,
    },
    Pcm {
        bytes_per_coded_frame: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct CodecDimensionPlan {
    channels_bound: usize,
    packet_bound: PacketFrameBound,
}

impl CodecDimensionPlan {
    fn try_new(params: &AudioCodecParameters) -> Result<Self, BudgetedDecodeError> {
        let codec = params.codec;

        let plan = match codec {
            CODEC_ID_AAC => {
                let channels = if let Some(extra_data) = &params.extra_data {
                    let config = AudioSpecificConfig::read(extra_data)
                        .map_err(BudgetedDecodeError::decode)?;
                    if config.samples != 1024 || config.sbr_present {
                        return Err(BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                            "aac: aac too complex",
                        )));
                    }
                    config
                        .channels
                        .map(|channels| channels.count())
                        .ok_or_else(|| {
                            BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                                "aac: channels or channel layout is required",
                            ))
                        })?
                } else {
                    required_channels(params, "aac: channels or channel layout is required")?
                };

                if channels > 2 {
                    return Err(BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                        "aac: aac too complex",
                    )));
                }
                Self::fixed(channels, 1024)
            }
            CODEC_ID_ADPCM_MS | CODEC_ID_ADPCM_IMA_WAV | CODEC_ID_ADPCM_IMA_QT => {
                let channels =
                    required_channels(params, "adpcm: channels or channel layout is required")?;
                if channels > 2 {
                    return Err(BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                        "adpcm: up to two channels are supported",
                    )));
                }
                let max_frames_per_packet = params.max_frames_per_packet.ok_or_else(|| {
                    BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                        "adpcm: maximum frames per packet is required",
                    ))
                })?;
                let max_frames_per_packet =
                    checked_frames(max_frames_per_packet, "ADPCM maximum frames per packet")?;
                let frames_per_block = params
                    .frames_per_block
                    .filter(|&value| value != 0)
                    .ok_or_else(|| {
                        BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                            "adpcm: valid frames per block is required",
                        ))
                    })?;
                let frames_per_block = checked_frames(frames_per_block, "ADPCM frames per block")?;
                Self {
                    channels_bound: channels,
                    packet_bound: PacketFrameBound::Adpcm {
                        frames_per_block,
                        max_frames_per_packet,
                    },
                }
            }
            CODEC_ID_ALAC => {
                let extra_data = params.extra_data.as_deref().ok_or_else(|| {
                    BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                        "alac: missing extra data",
                    ))
                })?;
                let cookie = MagicCookie::read(extra_data).map_err(BudgetedDecodeError::decode)?;
                if cookie.frame_length > 4096 * 16 {
                    return Err(BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                        "common (alac): frame length too large",
                    )));
                }
                let frames =
                    checked_frames(u64::from(cookie.frame_length), "ALAC packet frame bound")?;
                Self::fixed(usize::from(cookie.num_channels), frames)
            }
            CODEC_ID_FLAC => {
                let extra_data = params.extra_data.as_deref().ok_or_else(|| {
                    BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                        "flac: missing extra data",
                    ))
                })?;
                let info = StreamInfo::read(&mut BufReader::new(extra_data))
                    .map_err(BudgetedDecodeError::decode)?;
                let frames = usize::from(info.block_len_max);
                Self::fixed(info.channels.count(), frames)
            }
            // The pinned MPEG audio decoder supports at most stereo and emits no more than 1152
            // frames per packet (MP1 emits 384, deliberately covered by this shared bound).
            CODEC_ID_MP1 | CODEC_ID_MP2 | CODEC_ID_MP3 => Self::fixed(2, 1152),
            CODEC_ID_PCM_S32LE | CODEC_ID_PCM_S32BE | CODEC_ID_PCM_S24LE | CODEC_ID_PCM_S24BE
            | CODEC_ID_PCM_S16LE | CODEC_ID_PCM_S16BE | CODEC_ID_PCM_S8 | CODEC_ID_PCM_U32LE
            | CODEC_ID_PCM_U32BE | CODEC_ID_PCM_U24LE | CODEC_ID_PCM_U24BE | CODEC_ID_PCM_U16LE
            | CODEC_ID_PCM_U16BE | CODEC_ID_PCM_U8 | CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE
            | CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE | CODEC_ID_PCM_ALAW | CODEC_ID_PCM_MULAW => {
                let channels =
                    required_channels(params, "pcm: channels or channel layout is required")?;
                let coded_sample_bytes = pcm_coded_sample_bytes(codec).ok_or_else(|| {
                    BudgetedDecodeError::invariant("enabled PCM codec has no coded-width mapping")
                })?;
                let bytes_per_coded_frame = channels
                    .checked_mul(coded_sample_bytes)
                    .ok_or_else(|| BudgetedDecodeError::overflow("PCM coded frame width"))?;
                Self {
                    channels_bound: channels,
                    packet_bound: PacketFrameBound::Pcm {
                        bytes_per_coded_frame,
                    },
                }
            }
            // Symphonia 0.6.1 accepts at most eight mapped Vorbis channels and a 2^13 long block;
            // each packet emits at most half the long block.
            CODEC_ID_VORBIS => Self::fixed(8, 4096),
            _ => {
                return Err(BudgetedDecodeError::boxed_decode(Box::new(
                    UnsupportedAudioCodecError { codec },
                )))
            }
        };

        if plan.channels_bound == 0 {
            return Err(BudgetedDecodeError::decode(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decoded channel count is zero",
            )));
        }
        if plan.channels_bound > crate::MAX_CHANNELS {
            return Err(BudgetedDecodeError::decode(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decoded channel count exceeds the AudioBuffer limit",
            )));
        }
        if matches!(plan.packet_bound, PacketFrameBound::Fixed(0)) {
            return Err(BudgetedDecodeError::decode(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decoded frame bound is zero",
            )));
        }
        Ok(plan)
    }

    const fn fixed(channels_bound: usize, packet_frames: usize) -> Self {
        Self {
            channels_bound,
            packet_bound: PacketFrameBound::Fixed(packet_frames),
        }
    }

    fn packet_frame_bound(&self, packet: &Packet) -> Result<usize, BudgetedDecodeError> {
        match self.packet_bound {
            PacketFrameBound::Fixed(frames) => Ok(frames),
            PacketFrameBound::Adpcm {
                frames_per_block,
                max_frames_per_packet,
            } => {
                let block_duration =
                    checked_frames(packet.block_dur().get(), "ADPCM packet block duration")?;
                if block_duration > max_frames_per_packet {
                    return Err(BudgetedDecodeError::decode(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "ADPCM packet exceeds maximum frames per packet",
                    )));
                }
                Ok((block_duration / frames_per_block) * frames_per_block)
            }
            PacketFrameBound::Pcm {
                bytes_per_coded_frame,
            } => Ok(packet.data.len() / bytes_per_coded_frame),
        }
    }
}

fn required_channels(
    params: &AudioCodecParameters,
    message: &'static str,
) -> Result<usize, BudgetedDecodeError> {
    params
        .channels
        .as_ref()
        .map(|channels| channels.count())
        .ok_or_else(|| BudgetedDecodeError::decode(SymphoniaError::Unsupported(message)))
}

fn pcm_coded_sample_bytes(codec: AudioCodecId) -> Option<usize> {
    match codec {
        CODEC_ID_PCM_S8 | CODEC_ID_PCM_U8 => Some(1),
        CODEC_ID_PCM_S16LE | CODEC_ID_PCM_S16BE | CODEC_ID_PCM_U16LE | CODEC_ID_PCM_U16BE => {
            Some(2)
        }
        CODEC_ID_PCM_S24LE | CODEC_ID_PCM_S24BE | CODEC_ID_PCM_U24LE | CODEC_ID_PCM_U24BE => {
            Some(3)
        }
        CODEC_ID_PCM_S32LE | CODEC_ID_PCM_S32BE | CODEC_ID_PCM_U32LE | CODEC_ID_PCM_U32BE
        | CODEC_ID_PCM_F32LE | CODEC_ID_PCM_F32BE => Some(4),
        CODEC_ID_PCM_F64LE | CODEC_ID_PCM_F64BE => Some(8),
        CODEC_ID_PCM_ALAW | CODEC_ID_PCM_MULAW => Some(1),
        _ => None,
    }
}

#[derive(Debug)]
struct PlanarChunk {
    channels: Vec<Box<[f32]>>,
    frames: usize,
    sample_rate: f32,
    leases: LeaseSet,
}

struct BudgetedMediaDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_index: usize,
    packet_count: usize,
    plan: CodecDimensionPlan,
    result_leases: LeaseSet,
    result_channels: Option<usize>,
    budget: Arc<dyn DecodeBudget>,
}

impl std::fmt::Debug for BudgetedMediaDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetedMediaDecoder")
            .field("track_index", &self.track_index)
            .field("packet_count", &self.packet_count)
            .field("plan", &self.plan)
            .field("result_channels", &self.result_channels)
            .finish_non_exhaustive()
    }
}

impl BudgetedMediaDecoder {
    fn try_new<R: std::io::Read + Send + Sync + 'static>(
        input: R,
        budget: Arc<dyn DecodeBudget>,
    ) -> Result<Self, BudgetedDecodeError> {
        let input = Box::new(MediaInput::new(input));
        let stream = symphonia::core::io::MediaSourceStream::new(input, Default::default());
        let format = symphonia::default::get_probe()
            .probe(
                &Hint::new(),
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(BudgetedDecodeError::decode)?;

        let track = format.default_track(TrackType::Audio).ok_or_else(|| {
            BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                "no default media track available",
            ))
        })?;
        let track_index = format
            .tracks()
            .iter()
            .position(|candidate| candidate.id == track.id)
            .expect("the default track belongs to the format reader");
        let codec_params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .ok_or_else(|| {
                BudgetedDecodeError::decode(SymphoniaError::Unsupported(
                    "default media track is not an audio track",
                ))
            })?;
        let plan = CodecDimensionPlan::try_new(codec_params)?;

        // Reserve the result's independent channel dimension before constructing a decoder.
        // The first successful packet shrinks this conservative codec bound to the exact count.
        let mut result_leases = LeaseSet::default();
        result_leases.try_reserve(
            &budget,
            DecodeBudgetRequest::new(
                DecodeBudgetKind::DecodedChannels,
                u64::try_from(plan.channels_bound)
                    .map_err(|_| BudgetedDecodeError::overflow("decoded channel count"))?,
            ),
        )?;

        let decoder_opts = AudioDecoderOptions::default().verify(true);
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(codec_params, &decoder_opts)
            .map_err(|err| {
                if matches!(
                    err,
                    SymphoniaError::Unsupported("core (codec): unsupported audio codec")
                ) {
                    BudgetedDecodeError::boxed_decode(Box::new(UnsupportedAudioCodecError {
                        codec: codec_params.codec,
                    }))
                } else {
                    BudgetedDecodeError::decode(err)
                }
            })?;

        Ok(Self {
            format,
            decoder,
            track_index,
            packet_count: 0,
            plan,
            result_leases,
            result_channels: None,
            budget,
        })
    }

    fn take_result_leases(&mut self) -> LeaseSet {
        std::mem::take(&mut self.result_leases)
    }

    fn next_chunk(&mut self) -> Result<Option<PlanarChunk>, BudgetedDecodeError> {
        let track_id = self
            .format
            .tracks()
            .get(self.track_index)
            .ok_or_else(|| BudgetedDecodeError::invariant("selected track disappeared"))?
            .id;

        loop {
            let packet = match self.format.next_packet() {
                Ok(None) => {
                    self.finalize();
                    return Ok(None);
                }
                Err(SymphoniaError::IoError(err))
                    if err.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.finalize();
                    return Ok(None);
                }
                Err(err) => return Err(BudgetedDecodeError::decode(err)),
                Ok(Some(packet)) => {
                    self.packet_count += 1;
                    packet
                }
            };

            if packet.track_id != track_id {
                continue;
            }

            let frame_bound = self.plan.packet_frame_bound(&packet)?;
            let sample_bound = checked_product(
                "decoded packet sample bound",
                &[self.plan.channels_bound, frame_bound],
            )?;
            let mut leases = LeaseSet::default();
            leases.try_reserve(
                &self.budget,
                DecodeBudgetRequest::new(
                    DecodeBudgetKind::DecodedFrames,
                    u64::try_from(frame_bound)
                        .map_err(|_| BudgetedDecodeError::overflow("decoded packet frame bound"))?,
                ),
            )?;
            leases.try_reserve(
                &self.budget,
                DecodeBudgetRequest::new(DecodeBudgetKind::DecodedSamples, sample_bound),
            )?;

            match self.decoder.decode(&packet) {
                Ok(input) => {
                    let channels = input.spec().channels().count();
                    let frames = input.frames();
                    let sample_rate = input.spec().rate() as f32;
                    if channels == 0 {
                        return Err(BudgetedDecodeError::invariant(
                            "packet produced zero channels",
                        ));
                    }
                    if channels > self.plan.channels_bound {
                        return Err(BudgetedDecodeError::invariant(
                            "packet produced more channels than the codec bound",
                        ));
                    }
                    if frames > frame_bound {
                        return Err(BudgetedDecodeError::invariant(
                            "packet produced more frames than the codec bound",
                        ));
                    }
                    if !crate::is_valid_sample_rate(sample_rate) {
                        return Err(BudgetedDecodeError::InvalidSampleRate { sample_rate });
                    }
                    let samples = checked_product("decoded packet samples", &[channels, frames])?;
                    if samples > sample_bound {
                        return Err(BudgetedDecodeError::invariant(
                            "packet produced more samples than the codec bound",
                        ));
                    }
                    match self.result_channels {
                        Some(expected) if channels != expected => {
                            return Err(BudgetedDecodeError::invariant(
                                "packet changed the decoded result channel count",
                            ));
                        }
                        Some(_) => {}
                        None => {
                            self.result_leases.shrink_kind_to(
                                DecodeBudgetKind::DecodedChannels,
                                u64::try_from(channels).map_err(|_| {
                                    BudgetedDecodeError::overflow("decoded channel count")
                                })?,
                            )?;
                            self.result_channels = Some(channels);
                        }
                    }
                    leases.shrink_kind_to(
                        DecodeBudgetKind::DecodedFrames,
                        u64::try_from(frames)
                            .map_err(|_| BudgetedDecodeError::overflow("decoded packet frames"))?,
                    )?;
                    leases.shrink_kind_to(DecodeBudgetKind::DecodedSamples, samples)?;

                    let canonical_bytes = checked_product(
                        "decoded planar chunk bytes",
                        &[channels, frames, size_of::<f32>()],
                    )?;
                    checked_array_layout::<f32>(frames, "decoded planar chunk plane layout")?;
                    checked_array_layout::<Box<[f32]>>(
                        channels,
                        "decoded planar chunk outer layout",
                    )?;
                    leases.try_reserve(
                        &self.budget,
                        DecodeBudgetRequest::new(
                            DecodeBudgetKind::CanonicalPcmBytes,
                            canonical_bytes,
                        ),
                    )?;

                    record_wrapper_pcm_allocation();
                    let mut output = (0..channels)
                        .map(|_| vec![0.0; frames].into_boxed_slice())
                        .collect::<Vec<_>>();
                    input.copy_to_slice_planar::<f32, _>(&mut output);

                    return Ok(Some(PlanarChunk {
                        channels: output,
                        frames,
                        sample_rate,
                        leases,
                    }));
                }
                Err(SymphoniaError::DecodeError(err)) => {
                    log::warn!("Failed to decode packet #{}: {err}", self.packet_count);
                }
                Err(SymphoniaError::IoError(err)) => {
                    log::warn!(
                        "I/O error while decoding packet #{}: {err}",
                        self.packet_count
                    );
                }
                Err(err) => return Err(BudgetedDecodeError::decode(err)),
            }
        }
    }

    fn finalize(&mut self) {
        log::debug!(
            "Budgeted decoding finished after {} packet(s)",
            self.packet_count
        );
        let FinalizeResult { verify_ok } = self.decoder.finalize();
        if verify_ok == Some(false) {
            log::warn!("Verification of decoded data failed");
        }
    }
}

pub(crate) fn decode_media_data<R: std::io::Read + Send + Sync + 'static>(
    input: R,
    target_sample_rate: f32,
) -> Result<AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
    let mut sample_rate = None;
    let mut buffer: Option<AudioBuffer> = None;

    for chunk in MediaDecoder::try_new(input)? {
        let chunk = chunk?;

        match sample_rate {
            Some(rate) if rate != chunk.sample_rate() => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "decoded audio sample rate changed midstream",
                )));
            }
            Some(_) => {}
            None => sample_rate = Some(chunk.sample_rate()),
        }

        match buffer {
            Some(ref mut buffer) if buffer.number_of_channels() != chunk.number_of_channels() => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "decoded audio channel count changed midstream",
                )));
            }
            Some(ref mut buffer) => buffer.extend(&chunk),
            None => buffer = Some(chunk),
        }
    }

    let mut buffer = buffer.unwrap_or_else(|| AudioBuffer::from(vec![vec![]], target_sample_rate));

    // Resample to desired rate (no-op if already matching).
    buffer.resample(target_sample_rate);

    Ok(buffer)
}

/// Wrapper for `Read` implementers to be used in Symphonia decoding
///
/// Symphonia requires its input to impl `Seek` - but allows non-seekable sources. Hence we
/// implement Seek but return false for `is_seekable()`.
struct MediaInput<R> {
    input: R,
}

impl<R: Read> MediaInput<R> {
    pub fn new(input: R) -> Self {
        Self { input }
    }
}

impl<R: Read> Read for MediaInput<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.input.read(buf)
    }
}

impl<R> Seek for MediaInput<R> {
    fn seek(&mut self, _pos: SeekFrom) -> std::io::Result<u64> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "MediaInput does not support seeking",
        ))
    }
}

impl<R: Read + Send + Sync> symphonia::core::io::MediaSource for MediaInput<R> {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

/// Media stream decoder (OGG, WAV, FLAC, ..)
///
/// The current implementation supports Symphonia's audio formats and codecs.
pub(crate) struct MediaDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_index: usize,
    packet_count: usize,
}

impl MediaDecoder {
    /// Try to construct a new instance from a `Read` implementer
    ///
    /// # Errors
    ///
    /// This method returns an Error in various cases (IO, mime sniffing, decoding).
    pub fn try_new<R: std::io::Read + Send + Sync + 'static>(
        input: R,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Symphonia lib needs a Box<dyn MediaSource> - use our own MediaInput
        let input = Box::new(MediaInput::new(input));

        // Create the media source stream using the boxed media source from above.
        let stream = symphonia::core::io::MediaSourceStream::new(input, Default::default());

        // Create a hint to help the format registry guess what format reader is appropriate. In this
        // function we'll leave it empty.
        let hint = Hint::new();

        // TODO: Allow to customize some options.
        let format_opts: FormatOptions = Default::default();
        let metadata_opts: MetadataOptions = Default::default();
        // Opt-in to verify the decoded data against the checksums in the container.
        let decoder_opts = AudioDecoderOptions::default().verify(true);

        // Probe the media source stream for a format.
        let format =
            symphonia::default::get_probe().probe(&hint, stream, format_opts, metadata_opts)?;

        // Get the default audio track.
        let track = format
            .default_track(TrackType::Audio)
            .ok_or(SymphoniaError::Unsupported(
                "no default media track available",
            ))?;
        let track_index = format
            .tracks()
            .iter()
            .position(|t| t.id == track.id)
            .unwrap();

        let codec_params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .ok_or(SymphoniaError::Unsupported(
                "default media track is not an audio track",
            ))?;

        // Create a (stateful) decoder for the track.
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(codec_params, &decoder_opts)
            .map_err(|err| {
                if matches!(
                    err,
                    SymphoniaError::Unsupported("core (codec): unsupported audio codec")
                ) {
                    Box::new(UnsupportedAudioCodecError {
                        codec: codec_params.codec,
                    }) as Box<dyn std::error::Error + Send + Sync>
                } else {
                    Box::new(err) as Box<dyn std::error::Error + Send + Sync>
                }
            })?;

        Ok(Self {
            format,
            decoder,
            track_index,
            packet_count: 0,
        })
    }
}

#[derive(Debug)]
struct UnsupportedAudioCodecError {
    codec: AudioCodecId,
}

impl std::fmt::Display for UnsupportedAudioCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unsupported audio codec: {}", self.codec)
    }
}

impl std::error::Error for UnsupportedAudioCodecError {}

impl Iterator for MediaDecoder {
    type Item = Result<AudioBuffer, Box<dyn Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            format,
            decoder,
            track_index,
            packet_count,
        } = self;

        // Get the track.
        let track = format.tracks().get(*track_index)?;
        let track_id = track.id;

        loop {
            // Get the next packet from the format reader.
            let packet = match format.next_packet() {
                Ok(None) => {
                    log::debug!("Decoding finished after {packet_count} packet(s)");
                    let FinalizeResult { verify_ok } = decoder.finalize();
                    if verify_ok == Some(false) {
                        log::warn!("Verification of decoded data failed");
                    }
                    return None;
                }
                Err(err) => {
                    if let SymphoniaError::IoError(err) = &err {
                        if err.kind() == std::io::ErrorKind::UnexpectedEof {
                            log::debug!(
                                "Decoding finished after {packet_count} packet(s) at unexpected EOF"
                            );
                            let FinalizeResult { verify_ok } = decoder.finalize();
                            if verify_ok == Some(false) {
                                log::warn!("Verification of decoded data failed");
                            }
                            return None;
                        }
                    }

                    log::warn!(
                        "Failed to fetch next packet following packet #{packet_count}: {err}"
                    );
                    return Some(Err(Box::new(err)));
                }
                Ok(Some(packet)) => {
                    *packet_count += 1;
                    packet
                }
            };

            // If the packet does not belong to the selected track, skip it.
            let packet_track_id = packet.track_id;
            if packet_track_id != track_id {
                log::debug!(
                    "Skipping packet from other track {packet_track_id} while decoding track {track_id}"
                );
                continue;
            }

            // Decode the packet into audio samples.
            match decoder.decode(&packet) {
                Ok(input) => {
                    let output = input.into();
                    return Some(Ok(output));
                }
                Err(SymphoniaError::DecodeError(err)) => {
                    // Recoverable error, continue with the next packet.
                    log::warn!("Failed to decode packet #{packet_count}: {err}");
                }
                Err(SymphoniaError::IoError(err)) => {
                    // Recoverable error, continue with the next packet.
                    log::warn!("I/O error while decoding packet #{packet_count}: {err}");
                }
                Err(err) => {
                    // All other errors are considered fatal and decoding must be aborted.
                    return Some(Err(Box::new(err)));
                }
            };
        }
    }
}

impl From<GenericAudioBufferRef<'_>> for AudioBuffer {
    fn from(input: GenericAudioBufferRef<'_>) -> Self {
        let sample_rate = input.spec().rate() as f32;

        let mut data = Vec::new();
        input.copy_to_vecs_planar::<f32>(&mut data);

        let channels = data.into_iter().map(ChannelData::from).collect();
        AudioBuffer::from_channels(channels, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use std::io::Cursor;
    use std::sync::Mutex;

    use symphonia::core::audio::layouts::CHANNEL_LAYOUT_STEREO;
    use symphonia::core::units::{Duration, Timestamp};

    const BUDGET_KIND_COUNT: usize = 5;

    const fn kind_index(kind: DecodeBudgetKind) -> usize {
        match kind {
            DecodeBudgetKind::DecodedChannels => 0,
            DecodeBudgetKind::DecodedFrames => 1,
            DecodeBudgetKind::DecodedSamples => 2,
            DecodeBudgetKind::CanonicalPcmBytes => 3,
            DecodeBudgetKind::ResamplePcmBytes => 4,
        }
    }

    #[derive(Debug)]
    struct TestBudgetError;

    impl fmt::Display for TestBudgetError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("test budget limit")
        }
    }

    impl Error for TestBudgetError {}

    #[derive(Debug)]
    struct TestBudgetState {
        held: [u64; BUDGET_KIND_COUNT],
        peak: [u64; BUDGET_KIND_COUNT],
        limit: [u64; BUDGET_KIND_COUNT],
        requests: Vec<DecodeBudgetRequest>,
        panic_on: Option<DecodeBudgetKind>,
    }

    #[derive(Clone, Debug)]
    struct TestBudget {
        state: Arc<Mutex<TestBudgetState>>,
    }

    impl Default for TestBudget {
        fn default() -> Self {
            Self {
                state: Arc::new(Mutex::new(TestBudgetState {
                    held: [0; BUDGET_KIND_COUNT],
                    peak: [0; BUDGET_KIND_COUNT],
                    limit: [u64::MAX; BUDGET_KIND_COUNT],
                    requests: Vec::new(),
                    panic_on: None,
                })),
            }
        }
    }

    impl TestBudget {
        fn with_limit(kind: DecodeBudgetKind, limit: u64) -> Self {
            let budget = Self::default();
            budget.lock().limit[kind_index(kind)] = limit;
            budget
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, TestBudgetState> {
            self.state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
        }

        fn held(&self, kind: DecodeBudgetKind) -> u64 {
            self.lock().held[kind_index(kind)]
        }

        fn peak(&self, kind: DecodeBudgetKind) -> u64 {
            self.lock().peak[kind_index(kind)]
        }

        fn assert_clear(&self) {
            assert_eq!(self.lock().held, [0; BUDGET_KIND_COUNT]);
        }
    }

    struct TestReservation {
        state: Arc<Mutex<TestBudgetState>>,
        kind: DecodeBudgetKind,
        amount: u64,
    }

    impl DecodeBudgetReservation for TestReservation {
        fn shrink_to(&mut self, amount: u64) {
            assert!(amount <= self.amount);
            let released = self.amount - amount;
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.held[kind_index(self.kind)] -= released;
            self.amount = amount;
        }
    }

    impl Drop for TestReservation {
        fn drop(&mut self) {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.held[kind_index(self.kind)] -= self.amount;
            self.amount = 0;
        }
    }

    impl DecodeBudget for TestBudget {
        fn try_reserve(
            &self,
            request: DecodeBudgetRequest,
        ) -> Result<Box<dyn DecodeBudgetReservation>, DecodeBudgetError> {
            let mut state = self.lock();
            state.requests.push(request);
            if state.panic_on == Some(request.kind()) {
                panic!("injected budget panic");
            }

            let index = kind_index(request.kind());
            let next = state.held[index]
                .checked_add(request.amount())
                .ok_or_else(|| Box::new(TestBudgetError) as DecodeBudgetError)?;
            if next > state.limit[index] {
                return Err(Box::new(TestBudgetError));
            }
            state.held[index] = next;
            state.peak[index] = state.peak[index].max(next);
            drop(state);

            Ok(Box::new(TestReservation {
                state: Arc::clone(&self.state),
                kind: request.kind(),
                amount: request.amount(),
            }))
        }
    }

    fn budget_arc(budget: &TestBudget) -> Arc<dyn DecodeBudget> {
        Arc::new(budget.clone())
    }

    fn pcm16_wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_len = u32::try_from(std::mem::size_of_val(samples)).unwrap();
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for &sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }

    fn params(codec: AudioCodecId) -> AudioCodecParameters {
        let mut params = AudioCodecParameters::new();
        params
            .for_codec(codec)
            .with_channels(CHANNEL_LAYOUT_STEREO)
            .with_sample_rate(44_100);
        params
    }

    #[test]
    fn test_media_decoder() {
        let input = Cursor::new(vec![0; 32]);
        let media = MediaDecoder::try_new(input);

        assert!(media.is_err()); // the input was not a valid MIME type
    }

    #[test]
    fn test_unsupported_audio_codec_error_includes_codec_id() {
        let input = std::fs::File::open("samples/sample.webm").unwrap();
        let media = MediaDecoder::try_new(input);

        let err = match media {
            Ok(_) => panic!("expected unsupported codec error"),
            Err(err) => err.to_string(),
        };
        assert_eq!(err, "unsupported audio codec: 0x1001");
    }

    #[test]
    fn codec_dimension_bounds_cover_enabled_families() {
        let aac = CodecDimensionPlan::try_new(&params(CODEC_ID_AAC)).unwrap();
        assert_eq!(aac.channels_bound, 2);
        assert_eq!(aac.packet_bound, PacketFrameBound::Fixed(1024));

        for codec in [CODEC_ID_MP1, CODEC_ID_MP2] {
            let plan = CodecDimensionPlan::try_new(&params(codec)).unwrap();
            assert_eq!(plan.channels_bound, 2);
            assert_eq!(plan.packet_bound, PacketFrameBound::Fixed(1152));
        }

        let mp3 = CodecDimensionPlan::try_new(&params(CODEC_ID_MP3)).unwrap();
        assert_eq!(mp3.channels_bound, 2);
        assert_eq!(mp3.packet_bound, PacketFrameBound::Fixed(1152));

        let vorbis = CodecDimensionPlan::try_new(&params(CODEC_ID_VORBIS)).unwrap();
        assert_eq!(vorbis.channels_bound, 8);
        assert_eq!(vorbis.packet_bound, PacketFrameBound::Fixed(4096));

        for codec in [
            CODEC_ID_ADPCM_MS,
            CODEC_ID_ADPCM_IMA_WAV,
            CODEC_ID_ADPCM_IMA_QT,
        ] {
            let mut params = params(codec);
            params
                .with_max_frames_per_packet(505)
                .with_frames_per_block(101);
            let plan = CodecDimensionPlan::try_new(&params).unwrap();
            assert_eq!(plan.channels_bound, 2);
            assert!(matches!(
                plan.packet_bound,
                PacketFrameBound::Adpcm {
                    frames_per_block: 101,
                    max_frames_per_packet: 505,
                }
            ));
        }
    }

    #[test]
    fn pcm_coded_width_table_covers_every_enabled_pcm_codec() {
        let cases = [
            (CODEC_ID_PCM_S8, 1),
            (CODEC_ID_PCM_U8, 1),
            (CODEC_ID_PCM_S16LE, 2),
            (CODEC_ID_PCM_S16BE, 2),
            (CODEC_ID_PCM_U16LE, 2),
            (CODEC_ID_PCM_U16BE, 2),
            (CODEC_ID_PCM_S24LE, 3),
            (CODEC_ID_PCM_S24BE, 3),
            (CODEC_ID_PCM_U24LE, 3),
            (CODEC_ID_PCM_U24BE, 3),
            (CODEC_ID_PCM_S32LE, 4),
            (CODEC_ID_PCM_S32BE, 4),
            (CODEC_ID_PCM_U32LE, 4),
            (CODEC_ID_PCM_U32BE, 4),
            (CODEC_ID_PCM_F32LE, 4),
            (CODEC_ID_PCM_F32BE, 4),
            (CODEC_ID_PCM_F64LE, 8),
            (CODEC_ID_PCM_F64BE, 8),
            (CODEC_ID_PCM_ALAW, 1),
            (CODEC_ID_PCM_MULAW, 1),
        ];
        for (codec, width) in cases {
            assert_eq!(pcm_coded_sample_bytes(codec), Some(width));
        }
    }

    #[test]
    fn budgeted_decode_accepts_all_available_codec_fixtures() {
        for path in [
            "samples/sample-aac.m4a",
            "samples/sample-alac.m4a",
            "samples/sample.flac",
            "samples/sample.mp3",
            "samples/sample.ogg",
            "samples/sample.wav",
        ] {
            let budget = TestBudget::default();
            let input = std::fs::File::open(path).unwrap();
            let buffer = decode_media_data_with_budget(input, 44_100.0, budget_arc(&budget))
                .unwrap_or_else(|error| panic!("failed to decode {path}: {error}"));
            assert_ne!(buffer.length(), 0, "empty decoded fixture: {path}");
            assert_eq!(
                budget.held(DecodeBudgetKind::DecodedChannels),
                buffer.number_of_channels() as u64,
                "{path}"
            );
            assert_eq!(
                budget.held(DecodeBudgetKind::DecodedFrames),
                buffer.length() as u64,
                "{path}"
            );
            assert_eq!(
                budget.held(DecodeBudgetKind::DecodedSamples),
                (buffer.number_of_channels() * buffer.length()) as u64,
                "{path}"
            );
            drop(buffer);
            budget.assert_clear();
        }
    }

    #[test]
    fn every_result_dimension_can_refuse_before_the_corresponding_allocation() {
        for kind in [
            DecodeBudgetKind::DecodedChannels,
            DecodeBudgetKind::DecodedFrames,
            DecodeBudgetKind::DecodedSamples,
        ] {
            let budget = TestBudget::with_limit(kind, 0);
            let error = decode_media_data_with_budget(
                Cursor::new(pcm16_wav(3_000, &[1, 2])),
                3_000.0,
                budget_arc(&budget),
            )
            .unwrap_err();

            match error {
                BudgetedDecodeError::BudgetRefused { request, source } => {
                    assert_eq!(request.kind(), kind);
                    assert!(request.amount() > 0);
                    assert!(source.downcast_ref::<TestBudgetError>().is_some());
                }
                other => panic!("unexpected error: {other}"),
            }
            budget.assert_clear();
        }
    }

    #[test]
    fn invalid_target_sample_rate_is_rejected_before_probe_or_reservation() {
        for sample_rate in [f32::NAN, 0.0, 2_999.0, 768_001.0] {
            let budget = TestBudget::default();
            let error = decode_media_data_with_budget(
                Cursor::new(Vec::<u8>::new()),
                sample_rate,
                budget_arc(&budget),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                BudgetedDecodeError::InvalidSampleRate { sample_rate: value }
                    if value.to_bits() == sample_rate.to_bits()
            ));
            assert!(budget.lock().requests.is_empty());
            budget.assert_clear();
        }
    }

    #[test]
    fn chunk_refusal_precedes_canonical_output_allocation_and_cleans_up() {
        let budget = TestBudget::with_limit(DecodeBudgetKind::CanonicalPcmBytes, 0);
        let allocations_before = wrapper_pcm_allocation_count();
        let error = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(3_000, &[1, 2])),
            3_000.0,
            budget_arc(&budget),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            BudgetedDecodeError::BudgetRefused { request, .. }
                if request.kind() == DecodeBudgetKind::CanonicalPcmBytes
        ));
        // Symphonia has already decoded the trusted packet, but the wrapper's planar output
        // allocation is sequenced strictly after this rejected request.
        assert_eq!(wrapper_pcm_allocation_count(), allocations_before);
        budget.assert_clear();
    }

    #[test]
    fn final_assembly_is_reserved_while_exact_chunks_are_live() {
        let wav = pcm16_wav(3_000, &[1, 2, 3, 4]);
        let budget = TestBudget::default();
        let buffer =
            decode_media_data_with_budget(Cursor::new(wav.clone()), 3_000.0, budget_arc(&budget))
                .unwrap();
        let bytes = u64::try_from(buffer.length() * buffer.number_of_channels() * 4).unwrap();
        assert_eq!(budget.peak(DecodeBudgetKind::CanonicalPcmBytes), bytes * 2);
        drop(buffer);
        budget.assert_clear();

        let budget = TestBudget::with_limit(DecodeBudgetKind::CanonicalPcmBytes, bytes);
        let allocations_before = wrapper_pcm_allocation_count();
        let error = decode_media_data_with_budget(Cursor::new(wav), 3_000.0, budget_arc(&budget))
            .unwrap_err();
        assert!(matches!(
            error,
            BudgetedDecodeError::BudgetRefused { request, .. }
                if request.kind() == DecodeBudgetKind::CanonicalPcmBytes
                    && request.amount() == bytes
        ));
        // The single packet's chunk was allocated, but the rejected final-planar request did not
        // reach the second wrapper-owned allocation site.
        assert_eq!(wrapper_pcm_allocation_count(), allocations_before + 1);
        budget.assert_clear();
    }

    #[test]
    fn unknown_duration_stream_resamples_3k_to_768k_with_exact_final_leases() {
        let budget = TestBudget::default();
        // `MediaInput` deliberately reports no byte length or seekability, even for this cursor.
        let buffer = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(3_000, &[1, 2])),
            768_000.0,
            budget_arc(&budget),
        )
        .unwrap();

        assert_eq!(buffer.length(), 512);
        assert_eq!(budget.held(DecodeBudgetKind::DecodedChannels), 1);
        assert_eq!(budget.held(DecodeBudgetKind::DecodedFrames), 512);
        assert_eq!(budget.held(DecodeBudgetKind::CanonicalPcmBytes), 0);
        assert_eq!(budget.held(DecodeBudgetKind::DecodedSamples), 512);
        assert_eq!(budget.held(DecodeBudgetKind::ResamplePcmBytes), 512 * 4);
        drop(buffer);
        budget.assert_clear();
    }

    #[test]
    fn downsample_shrinks_exact_frame_and_sample_dimensions() {
        let budget = TestBudget::default();
        let buffer = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(768_000, &[1, 2])),
            3_000.0,
            budget_arc(&budget),
        )
        .unwrap();

        assert_eq!(buffer.length(), 1);
        assert_eq!(budget.peak(DecodeBudgetKind::DecodedFrames), 2);
        assert_eq!(budget.peak(DecodeBudgetKind::DecodedSamples), 2);
        assert_eq!(budget.held(DecodeBudgetKind::DecodedChannels), 1);
        assert_eq!(budget.held(DecodeBudgetKind::DecodedFrames), 1);
        assert_eq!(budget.held(DecodeBudgetKind::DecodedSamples), 1);
        assert_eq!(budget.held(DecodeBudgetKind::CanonicalPcmBytes), 0);
        assert_eq!(budget.held(DecodeBudgetKind::ResamplePcmBytes), 4);
        drop(buffer);
        budget.assert_clear();
    }

    #[test]
    fn resample_refusal_precedes_target_allocation_and_cleans_up() {
        let target_bytes = 512 * 4;
        let budget = TestBudget::with_limit(DecodeBudgetKind::ResamplePcmBytes, target_bytes - 1);
        let allocations_before = crate::buffer::resample_pcm_allocation_count();
        let error = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(3_000, &[1, 2])),
            768_000.0,
            budget_arc(&budget),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            BudgetedDecodeError::BudgetRefused { request, .. }
                if request.kind() == DecodeBudgetKind::ResamplePcmBytes
                    && request.amount() == target_bytes
        ));
        assert_eq!(
            crate::buffer::resample_pcm_allocation_count(),
            allocations_before
        );
        budget.assert_clear();
    }

    #[test]
    fn final_reservation_is_shared_across_clones() {
        let budget = TestBudget::default();
        let buffer = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(3_000, &[1, 2])),
            3_000.0,
            budget_arc(&budget),
        )
        .unwrap();
        let clone = buffer.clone();
        let held = budget.lock().held;

        drop(buffer);
        assert_eq!(budget.lock().held, held);
        drop(clone);
        budget.assert_clear();
    }

    #[test]
    fn clone_cow_mutation_keeps_the_result_lease_until_both_buffers_drop() {
        let budget = TestBudget::default();
        let buffer = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(3_000, &[1, 2])),
            3_000.0,
            budget_arc(&budget),
        )
        .unwrap();
        let mut clone = buffer.clone();
        let held = budget.lock().held;

        clone.get_channel_data_mut(0)[0] = 0.5;
        assert_eq!(budget.lock().held, held);
        drop(buffer);
        assert_eq!(budget.lock().held, held);
        drop(clone);
        budget.assert_clear();
    }

    #[test]
    fn split_result_conservatively_shares_the_full_lease_until_both_halves_drop() {
        let budget = TestBudget::default();
        let mut buffer = decode_media_data_with_budget(
            Cursor::new(pcm16_wav(3_000, &[1, 2])),
            3_000.0,
            budget_arc(&budget),
        )
        .unwrap();
        let split = buffer.split_off(1);
        let held = budget.lock().held;

        drop(buffer);
        assert_eq!(budget.lock().held, held);
        drop(split);
        budget.assert_clear();
    }

    #[test]
    fn packet_bounds_are_coupled_to_pinned_adpcm_and_mpeg_decoder_contracts() {
        // Symphonia 0.6.1 ADPCM uses Packet::block_dur rounded down to whole codec blocks.
        let mut adpcm_params = params(CODEC_ID_ADPCM_IMA_WAV);
        adpcm_params
            .with_max_frames_per_packet(505)
            .with_frames_per_block(101);
        let adpcm = CodecDimensionPlan::try_new(&adpcm_params).unwrap();
        let packet = Packet::new(0, Timestamp::ZERO, Duration::new(504), Vec::new());
        assert_eq!(adpcm.packet_frame_bound(&packet).unwrap(), 404);

        let oversized = Packet::new(0, Timestamp::ZERO, Duration::new(506), Vec::new());
        assert!(adpcm.packet_frame_bound(&oversized).is_err());

        // The pinned MP1/MP2 implementation's largest decoded packet is 1152 frames; MP1's
        // actual 384-frame packets are intentionally covered by that shared conservative bound.
        for codec in [CODEC_ID_MP1, CODEC_ID_MP2] {
            let mpeg = CodecDimensionPlan::try_new(&params(codec)).unwrap();
            assert_eq!(mpeg.packet_frame_bound(&packet).unwrap(), 1152);
            assert_eq!(mpeg.channels_bound, 2);
        }
    }

    #[test]
    fn unwind_drops_owned_reservations() {
        let budget = TestBudget::default();
        budget.lock().panic_on = Some(DecodeBudgetKind::CanonicalPcmBytes);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = decode_media_data_with_budget(
                Cursor::new(pcm16_wav(3_000, &[1, 2])),
                3_000.0,
                budget_arc(&budget),
            );
        }));
        assert!(result.is_err());
        budget.assert_clear();
    }

    #[test]
    fn resource_arithmetic_overflow_is_typed() {
        let error =
            checked_product("test product", &[usize::MAX, usize::MAX, usize::MAX]).unwrap_err();
        assert!(matches!(
            error,
            BudgetedDecodeError::ArithmeticOverflow {
                operation: "test product"
            }
        ));

        assert!(matches!(
            checked_array_layout::<u64>(usize::MAX, "test array layout"),
            Err(BudgetedDecodeError::ArithmeticOverflow {
                operation: "test array layout"
            })
        ));
    }
}
