# Native file streaming

`PcmSourceNode` is native PCM ingress for legacy and hosted contexts.
`MediaFileDecoder` incrementally decodes an opened file on a worker and converts
it to stereo at the graph's sample rate. Both are experimental Rust host APIs.
They add no script globals, file permissions, or automatic decoder threads.

```text
authorized File -> decoder worker / continuous resampler
                         -> fixed PCM queue -> PcmSourceNode -> GainNode -> destination
```

## Formats

The initial format set uses the already enabled Symphonia 0.6.1 dependencies:

| File family | Tested codec/container |
| --- | --- |
| WAV | PCM |
| MP3 | MPEG Layer III |
| FLAC | FLAC |
| Ogg | Vorbis |
| M4A | AAC-LC |
| M4A | ALAC |
| AIFF | PCM |

All seven have file-decoder and decoder-worker-to-hosted-graph tests. Common-format
fixtures are also decoded to completion with exact frame-count checks. WAV at
38, 44.1, and 48 kHz is covered; tone tests cover 44.1 -> 48 and 96 -> 48 kHz
conversion, phase continuity, amplitude, and EOF duration. A M4A fixture with its
`moov` metadata relocated after its media payload verifies seekable-file handling.
Format probing uses content, not the filename extension.

The decoder accepts mono/stereo input and source/target rates from 8 to 192 kHz.
Mono is duplicated into stereo. Opus, HE-AAC, WMA, multichannel downmixing, DRM,
HTTP, arbitrary byte streams, seeking, and looping are outside this slice.
An existing Opus/WebM fixture verifies an explicit unsupported-codec error.

## Host contract

1. Authorize the source and reserve the decoder job and working storage before
   opening/spawning. Call `PcmSourceNode::with_reservations` with accounting for
   **one node, one graph-control command, and the queue/state storage**.
2. Retain the returned node in the host, and move its unique writer to the worker.
   The queue stores exactly 2048 stereo frames (16 KiB PCM). Writer calls return
   the number of frames accepted. A full queue must apply backpressure to decoding.
3. Run `MediaFileDecoder::new` and `read` on that worker. It retains one decoded
   packet and fixed resampling buffers rather than assembling a whole sound.
   A continuous sinc resampler carries history across packets and flushes its
   tail to the source duration rounded up at the target rate.
4. Prefill, connect the node, and call `start`. The source consumes at the next
   quantum. Starvation inserts silence without advancing its source-frame count;
   playback resumes with the next queued frame. Start and stop are native atomic
   flags, not scheduled Web Audio control commands.
5. Dropping the writer publishes EOF; the source drains queued frames before
   reporting `ended()`. Decode errors must be reported separately by the worker's
   result. The example surfaces errors when joining; no JavaScript event contract
   is defined here.
6. `stop` permanently silences the source and cancels its writer. Physical graph
   destruction also cancels the writer. The host must retain and join the worker
   off render. A stop cannot interrupt an in-flight OS file read.

The lifetime guard stays held until both graph reclamation and release of all
queue/state handles. Keeping a writer after context close cannot release the
queue's accounting early. Decoder working storage and jobs need independent
guards; they can outlive source-node retirement.

The source's render processor performs at most 128 frame pops per quantum, uses
stack scratch space, and performs no I/O, lock acquisition, thread joins, or
explicit heap allocation. Its queue and processor destruction follow the hosted
graph's existing off-render reclamation. A test with an installed counting
allocator measures zero heap operations over settled whole hosted callbacks
during playback, starvation, recovery, and stop; graph bootstrap/mutations are
outside that particular measurement.

## Decoder limits and remaining integration

The decoder rejects packets above 1 MiB encoded or 65,536 decoded frames, rejects
midstream channel/rate changes, and latches fatal errors. Tag size is limited to
256 KiB per tag and artwork is disabled. These checks are **not** a complete
untrusted-media budget: encoded packets are checked after demuxing, and container
indexes, aggregate metadata, and codec-private allocations are not accounted by
the PCM queue. Per-packet limits do not imply a process RSS limit.

Before exposure to untrusted scripts, the embedder still needs source resolution
and permission checks, aggregate job/working-memory admission, cancellation-safe
worker ownership during isolate teardown, and completion/error reconciliation.
These Rust APIs do not establish those application guarantees. No physical-output
quality certification is implied by silent-output tests.

## Exercise

```sh
cargo test --no-default-features --test streaming
cargo test --no-default-features --example streaming_media
cargo run --no-default-features --example streaming_media -- samples/sample.mp3 5 none
```

Omit `none` and enable the default backend to use physical output. The example
uses a gain of 0.25 and joins its decoder after EOF, early stop, or context close.
Its blocking owner Drop is an example ownership pattern, not an isolate teardown
implementation. The earlier `streaming_file` example remains a standalone WAV
transport experiment using legacy worklet registration.
