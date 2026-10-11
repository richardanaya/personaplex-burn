# personaplex-burn

NVIDIA [PersonaPlex](https://github.com/NVIDIA/personaplex) (full-duplex speech-to-speech, Moshi architecture) on [Burn](https://github.com/tracel-ai/burn) 0.22. Inference runs on Burn's **wgpu** backend: Vulkan, Metal, DirectX 12, OpenGL, or WebGPU, whichever the machine offers. The CPU backend exists for parity tests only.

The weights are `nvidia/personaplex-7b-v1` on Hugging Face, under the NVIDIA Open Model License. They are gated.

## Session API

[`Session`](src/session.rs) is the interface. It does not open a microphone, a socket, or a file. Adapters do that.

```rust,ignore
use personaplex_burn::{
    inference_device, DtypePolicy, Event, PersonaPlex, Session, SessionConfig,
    VoicePrompt, WeightSource,
};

let model = PersonaPlex::load(
    WeightSource::huggingface(), // or WeightSource::local("checkpoints/")
    inference_device(),          // Device::wgpu
    DtypePolicy::Auto,           // f16 when the adapter has shader-f16, else f32
)?;
let mut session = Session::start(
    model,
    SessionConfig::new("You are a wise teacher.")
        .with_voice(VoicePrompt::from_path("voice.wav")),
)?;

// Any rate, any channel count. Downmix + linear resample to 24 kHz mono.
// Frames are 80 ms (1920 samples) and stay inside the session.
let events = session.push_audio(&mic_samples, sample_rate, channels)?;
for event in events {
    match event {
        Event::AudioOut { pcm, sample_rate } => play(&pcm, sample_rate),
        Event::Text { text, token, .. } => show_inner_monologue(&text, token),
        Event::Status(status) => log(status),
    }
}

session.interrupt();          // playback should stop; the KV cache stays
session.set_muted(true);      // further input is silence
session.reset()?;             // clear streaming state and re-apply prompts
let tail = session.finish()?; // zero-pad the last partial frame
```

`push_audio_async` is the same call as a ready future, so an async adapter can `.await` it without this crate depending on a runtime.

A whole file uses the same session:

```rust,ignore
let batch = personaplex_burn::process_batch(&model, cfg, &samples, rate, channels)?;
// batch.pcm is model-rate mono, trimmed to the input duration
// batch.text is the inner monologue
```

`TextKind::UserTranscript` is part of the event enum and is never produced. This checkpoint has no separate user-transcript head.

### What a turn contains

PersonaPlex steps one Mimi frame at a time (12.5 Hz). Each step can emit:

- agent audio, decoded from codebooks 1..8
- one inner-monologue text token (`EPAD` / `PAD` for the special ids 0 and 3, otherwise the sentencepiece piece)
- a frame-timing status, when `emit_timing` is set

Voice conditioning is a clip (resampled, downmixed, normalized to -24 LUFS) or a safetensors cache with `embeddings` and an optional token `cache`. PyTorch `.pt` voice files are rejected; convert them first. The persona is wrapped as `<system> … <system>` unless it already starts and ends with that tag.

## Adapters

| Binary | Crate | What it does |
| --- | --- | --- |
| `personaplex-batch` | `adapters/batch` | WAV in, WAV and text out |
| `personaplex-talk` | `adapters/talk` | cpal mic/speaker and a ratatui TUI. `q` quit, `m` mute, `r` reset |
| `personaplex-server` | `adapters/server` | WebSocket plus the page in `realtime_web/` |
| `personaplex-realtime` | library feature `realtime-server` | OpenAI Realtime / xAI Voice protocol. See [docs/realtime.md](docs/realtime.md) |
| `realtime_web` | `examples/realtime_web` | Same protocol, plus a browser page on that port. In addition to the TUI |

The first three call `inference_device()` (`Device::wgpu`). cpal, ratatui, and the simple websocket server are not dependencies of the library. `personaplex-realtime` is behind `--features realtime-server` (tokio is not in the default build) and also uses `Device::wgpu`.

```bash
cargo run -p personaplex-batch -- --weights ./checkpoints --input in.wav --output out.wav --text out.txt --persona "You are a wise teacher." --voice voice.wav
cargo run -p personaplex-talk -- --weights ./checkpoints --persona "You are a wise teacher." --voice voice.wav
cargo run -p personaplex-server -- --weights ./checkpoints --bind 127.0.0.1:8787 --web realtime_web
cargo run --features realtime-server --bin personaplex-realtime -- --port 8080 --weights ./checkpoints --voices ./voices
cargo run --example realtime_web --features realtime-server -- --port 8080 --weights ./checkpoints --voices ./voices
```

`personaplex-talk` is the cpal + ratatui TUI. `realtime_web` does not replace it. That example is one process: the page at `http://127.0.0.1:8080/` and the WebSocket at `ws://127.0.0.1:8080/v1/realtime`. The HTML, CSS, and JS are embedded in the binary (`include_str!`); there is no npm build. Open that URL after the weights load. The page captures the microphone, so it has to be served from localhost or from HTTPS. A plain `http://` page on another host will not get `getUserMedia`.

Weights are the gated checkpoint. Accept the license, export `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`), or pass `--weights` pointing at a local directory with `model.safetensors`, the tokenizer safetensors, and `tokenizer_spm_32k_3.model`. `--voices` is a directory of `{name}.wav` or `{name}.safetensors` presets named by the voice field on the page.

```bash
export HF_TOKEN=hf_...
cargo run --example realtime_web --features realtime-server -- --port 8080 --voices ./voices
```

The other web page (`http://127.0.0.1:8787/`, from `personaplex-server`) speaks the small JSON protocol below, not the OpenAI realtime events. The 7B model stays on the server. `Device::webgpu` is the browser WebGPU API and is not what this crate uses for inference: a 7B model does not fit in a tab, and that device constructor is for wasm.

### WebSocket

Text frames are JSON. Binary frames are little-endian `f32` PCM.

Client to server:

- `{"op":"start","persona":"...","sample_rate":16000,"channels":1}`
- `{"op":"format","sample_rate":16000,"channels":1}`
- `{"op":"interrupt"}` `{"op":"reset"}` `{"op":"mute","muted":true}`
- binary: interleaved f32 PCM at `sample_rate`

Server to client:

- `{"type":"audio_format","sample_rate":24000}` before audio
- `{"type":"text","text":" hello","token":12}`
- `{"type":"status","name":"listening"}` or `{"type":"status","name":"frame","index":0,"elapsed_ms":40.0}`
- binary: mono f32 PCM at the model rate

## Weights

Accept the license at <https://huggingface.co/nvidia/personaplex-7b-v1>, then:

```bash
export HF_TOKEN=hf_...
```

`HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) is required. With no token, `PersonaPlex::load` returns immediately and does not download the 16 GB checkpoint. The error includes the license URL.

A local directory works with no token. It needs:

- `model.safetensors`
- `tokenizer-e351c8d8-checkpoint125.safetensors`
- `tokenizer_spm_32k_3.model`

The checkpoint is bf16. Each tensor is converted to f16 or f32 when it is uploaded (`DtypePolicy::Auto` asks `device.supports_dtype(F16)`), and the file pages for that tensor are released. The whole checkpoint is not widened to f32 in host memory. Burn's wgpu backend does not run bf16 kernels. If the real checkpoint was trained with `dep_q=8`, those depth-transformer weights are tiled out to 16 the way the NVIDIA loader does.

## WebGPU limits and VRAM

`inference_device` is `Device::wgpu`, not `Device::flex` and not `Device::webgpu`.

WebGPU's guaranteed limits are `maxStorageBufferBindingSize` = 128 MiB and `maxBufferSize` = 256 MiB. CubeCL sizes its storage buffers from the tighter of those. A 7B PersonaPlex matrix exceeds 128 MiB in f16 (the text embedding is about 250 MiB; each temporal gating `linear_in` is about 168 MiB). Matrices are split along the output rows so each buffer stays at or under 64 MiB (`MAX_WEIGHT_BUFFER_BYTES`). That also fits f32 weights when the adapter has no shader-f16. A single row of a 4096-wide matrix is 16 KB in f32, so the split always has a legal chunk size.

Streaming attention steps one frame at a time. The temporal KV cache is about 1.6 GB total in f16 (32 layers, K and V, 32 heads, 3000 positions, head dim 128), and each layer buffer is about 25 MB, under the binding limit. A non-streaming score matrix for the full 3000-token context would be about 1 GB and is not the session path.

Rough device memory for the release model:

| | f16 weights (shader-f16) | f32 weights |
| --- | --- | --- |
| LM + Mimi parameters | ~14 GB | ~28 GB |
| Temporal KV (context 3000) | ~1.6 GB | ~3 GB |
| Activations, Mimi cache, depth KV | under 1 GB | under 1 GB |

Plan on **24 GB** when shader-f16 is available, and more like **40 GB** when the run falls back to f32. The paper's timing was measured on an A100 80 GB; this port does not use CUDA graphs or CPU offload.

Each frame has an 80 ms budget (temporal step plus 16 depth steps). wgpu will not match a CUDA PyTorch baseline. The Flex CPU backend will not hold that budget. Neither number was measured here.

Override the adapter with `CUBECL_WGPU_DEFAULT_DEVICE` (for example `DiscreteGpu(0)`).

## Architecture

The reference is the Python package in [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) (`moshi/`). Behavior follows that code, including the LMGen delay cache.

| Reference | Rust |
| --- | --- |
| `moshi.models.loaders` / `LMModel` | `lm`, `config::ModelConfig::personaplex_7b` |
| `LMGen` (`prepare_step_input`, `process_transformer_output`, depformer step) | `lm::LmState` |
| `modules.transformer` (RoPE, SDPA, gating, ring KV) | `transformer`, `norm` |
| `modules.seanet`, `modules.conv`, `modules.resample` | `seanet`, `conv`, `mimi` |
| `quantization` split RVQ | `quantizer` |
| `utils.sampling.sample_token` | `sampling` (greedy and recorded-noise paths). Seeded sampling uses SplitMix64, not PyTorch's RNG |
| Mimi codec + sentencepiece | `mimi`, `tokenizer` |
| voice prompt, system tags, offline trim | `session`, `batch`, `audio` |
| safetensors + hf-hub | `weights` |

Mimi: SEANet encoder, 8-layer transformer, stride-2 downsample, 8 active codebooks (1 semantic + 7 acoustic; 32 are stored), stride-2 upsample, decoder transformer, SEANet decoder. Sample rate 24 kHz, frame rate 12.5 Hz.

LM: dim 4096, 32 layers, 32 heads, feed-forward 16896, RMSNorm in f32, SiLU gating, RoPE, context 3000. Depth transformer: dim 1024, 6 layers, 16 codebooks, a fresh KV cache every frame. Delays are `[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`. Streams: text, 8 agent codebooks, 8 user codebooks.

Attention and norms reduce in f32 and cast back to the weight dtype. The delay cache and the sampler stay on the CPU so token control flow does not depend on integer kernels.

## Tests

```bash
cargo test -p personaplex-burn --no-default-features --features flex
```

`scripts/gen_fixtures.py` builds tiny randomly initialized copies of the same modules with the NVIDIA package and writes `fixtures/`. `cargo test` does not need the gated weights or a network. The Flex feature (`Device::flex()`) is the pure-Rust CPU backend. Burn 0.22 deprecates NdArray; Flex replaces it. It is not a default feature.

Covered against those fixtures on Flex: Mimi latent, codes, decode, and streaming halves; temporal and depth logits; streaming temporal steps; LMGen greedy steps (including the two leading null steps); top-k sampling with the recorded exponential noise; linear resample and the LUFS gain formula; row-chunked matmul; a WAV batch through `Session`; interrupt, mute, reset; `push_audio_async` matching `push_audio`; a missing `HF_TOKEN` returning the license error without downloading.

Two extra tests are `#[ignore]`d so CI does not need a graphics adapter. On a machine with wgpu (a GPU, or lavapipe) they run with:

```bash
cargo test -p personaplex-burn --lib wgpu_ -- --ignored --nocapture
```

`wgpu_device_can_allocate` checks f32 and, when `shader-f16` is present, f16 matmul on `Device::wgpu`. `wgpu_tiny_mimi_and_lm_match_fixtures` runs the tiny Mimi and LM fixtures on that device at the same tolerances as Flex.

Loudness measurement uses the `ebur128` crate and is compared with the pyloudnorm value stored in the fixture. The gain applied once loudness is known matches the reference. The two meters are not the same implementation.

## What was run on wgpu

This VM has no GPU and no Vulkan ICD of its own. With Mesa's lavapipe software Vulkan driver (`VK_DRIVER_FILES` pointing at `lvp_icd.json`), `Device::wgpu` came up, reported `shader-f16`, and passed both ignored tests: an f16 matmul, and the tiny Mimi plus LM fixtures (latent, codes, decode, temporal logits, depth logits) at the Flex tolerances. That is a software rasterizer, not a discrete GPU, and it is not what CI runs.

## What this environment did not run

- The gated 7B weights. There is no `HF_TOKEN` here, and the loader refuses to start the download without one. A 7B f16 checkpoint plus the KV cache does not fit this VM's 16 GB of RAM either.
- A physical GPU. Adapter selection, VRAM residency, and the 80 ms frame budget were not measured.
- Browser WebGPU. The page talks to the server. `Device::webgpu` is not the inference device.

## Gaps

- No CUDA graphs, no CPU weight offload. wgpu is not the CUDA path the paper timed.
- User audio is resampled with the same linear interpolator as the voice prompt. The reference uses `sphn` for user audio.
- Integrated loudness is BS.1770 via `ebur128`, not pyloudnorm. The gain formula matches once the LUFS number is known.
- `.pt` voice embeddings are not loaded.
- SentencePiece encode/decode is implemented here (unigram Viterbi and BPE) and was not checked against a real `tokenizer_spm_32k_3.model`, because that file is gated.
- There is no user-transcript stream in this checkpoint.
- Real-time factor on wgpu was not measured.

## License

The code in this repository is MIT. The PersonaPlex weights are under the NVIDIA Open Model License, accepted on the model page before download.
