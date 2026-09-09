# Bounded file streaming experiment

The subsequent hosted implementation and common-format decoder are documented in
[STREAMING.md](../../STREAMING.md) and demonstrated by `streaming_media`. This
earlier WAV-only experiment remains as transport/worker-lifecycle evidence.

This native Rust experiment tests file acquisition, incremental decoding, queue
backpressure, gain-node routing, and decoder ownership. It is **not a supported
Smudgy or deno_audio API**. It uses the fork's legacy Rust AudioWorklet registration;
that constructor is not admitted by Smudgy's hosted graph.

```text
File -> worker / WAV decoder -> fixed PCM queue -> native worklet -> GainNode -> output
```

## Run

From the repository root:

```sh
cargo test --no-default-features --example streaming_file
cargo run --example streaming_file -- path.wav 10
cargo run --no-default-features --example streaming_file -- path.wav 10 none
```

The last command uses the silent output device. The duration is an early-stop
limit in seconds; natural end also stops playback. The example accepts only
mono/stereo 16-bit integer PCM WAV at 48 kHz. Playback uses gain 0.25. All file
opening, header parsing, reads, and decoding occur on the decoder worker.

## What the proof establishes

- Sixteen 128-frame stereo blocks hold **16 KiB of PCM**, about 42.7 ms at 48 kHz.
  There can also be one worker-owned block and one block being consumed, plus an
  8 KiB encoded-file read buffer. Block metadata, graph storage, thread stacks,
  allocator overhead, and OS filesystem caching are separate from this bound.
- Backpressure stops further decoding when the queue is full. A ten-minute,
  115,200,044-byte WAV fixture is only partly read before and during a short
  playback through a gain node. Tests assert both decoded-frame and file-read
  bounds; they do not infer memory use merely from file size or an RSS snapshot.
- An empty queue produces a silent quantum without blocking or advancing the
  source cursor. Playback recovers when data arrives. This policy inserts silence
  and extends playback; it does not drop late samples to maintain wall-clock sync.
- Complete blocks remain playable after the worker publishes EOF or an error.
  The final partial block is zero padded, and EOF is observed after queued data.
- Stop signals cancellation and joins the worker on the control side. It also
  silences queued audio. Dropping the owner joins too. The processor has no worker
  handle; dropping it only signals cancellation. Context-close tests subsequently
  join through the retained owner.
- Tests verify exact gain output across queue refills, mono duplication, stereo
  ordering, partial EOF, starvation/recovery, invalid/truncated input, stop with a
  full queue, and graph destruction before and during rendering.
- An installed counting allocator verifies **zero allocations, reallocations,
  and deallocations in the source's sample-fill callback**, including starvation
  and terminal cases. A separate test confirms the allocator is active. This is
  not a claim about the legacy worklet wrapper, graph mutations, or destruction.

Offline graph tests wait for the decoder in scheduled control-side suspensions;
they never wait in the source callback. Their short rendered output is collected
for exact sample assertions. Online tests use the silent sink and actual render
and decode threads. No physical-output quality or deadline guarantee is claimed.

## Required before product integration

1. Add a native source with hosted graph admission, connection/lifetime accounting,
   bounded control transport, and off-render destruction. Do not expose the legacy
   worklet constructor as a shortcut.
2. Move decoder ownership into the host's lifecycle service. Isolate teardown,
   context close, failed installation, and source retirement must retain the join
   obligation. A synchronous owner Drop is convenient for this experiment, but
   production teardown needs cancellation-independent cleanup receipts. Local file
   reads can still stall in the OS; cancellation here interrupts queue backpressure,
   not an in-flight filesystem operation.
3. Admit jobs and bytes before spawning/allocating. Establish bounds for decoder
   packets, metadata, channels, and resampling with the chosen production decoder.
   This restricted WAV proof does not establish bounds for hostile media or other
   codecs. Add a prebuffer policy suitable for real storage latency.
4. Integrate the existing package/file resolver and permission checks before opening
   sources, then bind start/stop/end/error to deno_audio and Smudgy. The command-line
   example is trusted native code with no script permission boundary.

HTTP, caller-provided byte streams, seeking, looping, resampling, scheduled starts,
and public scripting API design are deferred. This experiment changes no library
surface or application dependency pin.
