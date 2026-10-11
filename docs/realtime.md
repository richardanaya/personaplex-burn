# Realtime protocol

PersonaPlex can sit behind the same WebSocket events that OpenAI Realtime and xAI Grok Voice use (`wss://api.x.ai/v1/realtime`). The server is `personaplex-realtime`, built with the `realtime-server` feature. It is not a default feature, so the library crate does not link tokio unless you ask for it.

```bash
cargo run --features realtime-server --bin personaplex-realtime -- --port 8080 --weights ./checkpoints --voices ./voices
```

Clients connect to `ws://127.0.0.1:8080/v1/realtime`. A `?model=` query is echoed in `session.created` and does not select a different checkpoint. The `xai-client-secret.*` subprotocol is accepted and the token is ignored.

The same server code also serves a browser client. `examples/realtime_web` embeds one HTML file and listens on a single port: the page at `/`, the socket at `/v1/realtime`. It is in addition to the cpal + ratatui TUI (`personaplex-talk`).

```bash
export HF_TOKEN=hf_...
cargo run --example realtime_web --features realtime-server -- --port 8080 --weights ./checkpoints --voices ./voices
```

Open <http://127.0.0.1:8080/>. `--weights` can replace the download when the checkpoint is already on disk. The page uses `getUserMedia`, which browsers allow on localhost or on HTTPS only. There is no npm step; the page is plain HTML, CSS, and JS compiled in with `include_str!`.

The page streams PCM16 at 24 kHz (`input_audio_buffer.append`), plays `response.output_audio.delta` (and the older `response.audio.delta` name) through a jitter buffer, and clears that buffer on `input_audio_buffer.speech_started`, `response.cancelled`, and `conversation.item.truncated`. Persona and voice controls send `session.update`. This checkpoint still does not emit a user transcript; the page shows that turn as heard, and will show text if a transcription event arrives. Model text is the inner-monologue deltas.

This is feasible for audio, persona text, and a synthesized turn lifecycle. It is not a drop-in replacement for a tool-calling voice agent.

## Why the shapes do not match the model

PersonaPlex is full duplex. Audio goes in on every 80 ms frame, and audio and an inner-monologue token can come out on that same frame. There is no end of turn, no VAD, and no response object. The user can talk over the model; the KV cache stays.

OpenAI and xAI are turn based. The server (or the client) decides when a user turn ends, then opens one assistant response, streams it, and closes it. Barge-in cancels that response.

The bridge keeps the model on its own clock: every appended buffer is resampled and pushed into `Session::push_audio` immediately, including silence. The events around that are synthesized:

- Input RMS against `turn_detection.threshold` is the VAD. The threshold is a linear RMS on float PCM in [-1, 1], not an OpenAI speech probability. `0.3` (what pi-spacexai sends) needs a fairly strong signal. Default when omitted is `0.5`.
- A response opens when the model emits audible audio or an inner-monologue piece, and closes after `silence_duration_ms` of model silence (default 500).
- If user energy rises while a response is open, the server emits `input_audio_buffer.speech_started`, `conversation.item.truncated`, and `response.cancelled`, and calls `Session::interrupt`. New model audio is held until `speech_stopped`, then a new response can open. That matches xAI's "interruptions are automatic" and the events pi-spacexai already handles. pi also sends `conversation.item.truncate` and `response.cancel` itself; a cancel with nothing in flight returns `error` with message `no active response`, which that client ignores. A truncate for the item just cancelled is answered again with `conversation.item.truncated`.

`turn_detection: null` and `turn_detection.type: "none"` are manual. No speech events. Audio still advances the model, but deltas wait for `response.create`. `input_audio_buffer.commit` is accepted only in this mode. `semantic_vad` is treated as energy VAD. `idle_timeout_ms` is stored and does not fire a check-in.

## Event mapping

Wire names are the current xAI / OpenAI GA names. pi-spacexai (the `realtime-voice.ts` on `main`) accepts those and the older `response.audio.delta` / `response.audio_transcript.delta` aliases. This server emits the GA name once, so a client that listens for both does not play the audio twice.

| PersonaPlex | Server event |
| --- | --- |
| connection open | `session.created`, `conversation.created` |
| `session.update` `instructions` | persona prompt, applied with `Session::start` (resets streaming state) |
| `session.update` `voice`, or `audio.output.voice` | voice-conditioning preset: `{voices}/{name}.wav`, `{name}.safetensors`, or a filesystem path. Unknown names leave conditioning off and still return `session.updated` |
| `audio.input/output.format` `audio/pcm` at any rate | resample to and from 24 kHz. PCM16 little-endian, base64 in JSON |
| `audio/pcmu`, `audio/pcma`, legacy `g711_ulaw` / `g711_alaw` | G.711 at 8 kHz, then the same resample |
| `audio/opus` | `error` `unsupported_format` |
| `transport: "binary"` | raw codec bytes on binary WebSocket frames. Lifecycle stays JSON |
| user energy rises (`server_vad`) | `input_audio_buffer.speech_started` |
| user energy stays under the threshold for `silence_duration_ms` | `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`, `conversation.item.added` |
| first audible frame or inner-monologue piece of a turn | `response.created`, `response.output_item.added`, `response.content_part.added` |
| decoded PCM | `response.output_audio.delta` (base64 PCM16 at the output rate, default 24 kHz) |
| inner-monologue piece, skipping `EPAD` / `PAD` | `response.output_audio_transcript.delta` |
| model silent for `silence_duration_ms` | `response.output_audio.done`, `response.output_audio_transcript.done`, `response.content_part.done`, `response.output_item.done`, `response.done` |
| barge-in | `conversation.item.truncated` (transcript so far, `audio_end_ms` from samples already sent), `response.cancelled` |
| client `response.cancel` | same cancel, or `no active response` |
| client `conversation.item.truncate` | `conversation.item.truncated` |
| user transcript | not emitted. This checkpoint has no user-transcript head. `audio.input.transcription` is echoed and ignored, including `grok-transcribe` |

## OpenAI and xAI

Both use the same socket shape. Differences that matter here:

| | OpenAI Realtime (GA) | xAI Grok Voice |
| --- | --- | --- |
| URL | `wss://api.openai.com/v1/realtime` | `wss://api.x.ai/v1/realtime?model=grok-voice-latest` |
| Session type | `session.type` must be `"realtime"` on update | omitted; voice and instructions are top-level |
| Voice field | `audio.output.voice` after GA; a voice cannot change after the first audio | top-level `voice`, and it can change mid-session |
| Turn detection off | `null` or older `"none"` | `null` means manual text turns |
| Audio events | GA names: `response.output_audio.delta`, `response.output_audio_transcript.delta` | same GA names in the current voice reference; older clients still say `response.audio.delta` |
| Text events | `response.output_text.delta` | that name and `response.text.delta` |
| Tools | functions, and the model calls them | functions plus `web_search`, `x_search`, `mcp` |
| Extra | semantic VAD, noise reduction | `idle_timeout_ms`, session resumption, opus, binary transport, input transcription `grok-transcribe` |

`session.update` here follows xAI's top-level `voice` and `instructions`, and also reads OpenAI's `audio.output.voice`. Changing either restarts the PersonaPlex session so the new prompt and voice conditioning are applied. Tools are echoed in `session.updated` and never called.

## Unsupported, and what the server does

| Client feature | Policy |
| --- | --- |
| Tools and function calls (`send_task`, `web_search`, `mcp`, …) | Accepted on `session.update` so the socket stays up. The model never emits `response.function_call_arguments.done`. A later `function_call` or `function_call_output` gets `error` "function calls are not supported". |
| Multiple responses | A second `response.create` while one is open gets `error` "already an active response". |
| Text-only input (`conversation.item.create` with `input_text`) | `error` "text-only input is not supported". The persona is the only text prompt. |
| User transcription | No event. Do not wait on `conversation.item.input_audio_transcription.completed`. |
| `audio/opus` | `error` on `session.update`. |
| `conversation.item.delete` | `error`. |
| Session resumption, idle check-in, reasoning effort, output speed, pronunciation `replace` | Ignored. |
| Auth | No token required. `Authorization` and `xai-client-secret.*` are not checked. |

`session.update` itself does not emit `error` for tools or for a missing voice file. pi-spacexai treats any `error` before `session.updated` as fatal and closes the socket.

## pi-spacexai

`realtime-voice.ts` on `main` connects to `wss://api.x.ai/v1/realtime?model=...` with subprotocol `xai-client-secret.${token}`, then sends `session.update` (server VAD at threshold 0.3 and 500 ms, PCM 24 kHz, `grok-transcribe`, and four functions). Mic frames are 100 ms of PCM16, base64, in `input_audio_buffer.append`. It plays `response.output_audio.delta` or `response.audio.delta`, captions `response.output_audio_transcript.delta` or `response.audio_transcript.delta`, and on `input_audio_buffer.speech_started` it sends `conversation.item.truncate` plus `response.cancel`.

To point that client at this server:

1. Change `REALTIME_URL` to `ws://127.0.0.1:8080/v1/realtime`. The model query can stay; it is only a label.
2. Skip the xAI ephemeral-token request, or keep it. This server accepts the subprotocol and also a socket with no token. The stock client still mints a token before it opens the socket, so that call has to be stubbed or it never connects.
3. Put voice presets on disk (`--voices`) if you want conditioning. `leo` and `eve` are not built in. An unknown name connects and runs with no voice prompt.
4. Expect assistant captions to be inner-monologue pieces, not a separate transcript stream.
5. Do not expect `send_task`, `search_conversations`, `end_the_call`, or `read_background_updates` to run. The model has no tool head, so the coding-harness bridge will not fire.

Audio duplex, mute-by-interrupt, and the session lifecycle do line up with that client. The tool loop does not.
