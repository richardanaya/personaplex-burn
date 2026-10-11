//! Turn synthesis on top of a full-duplex engine.
//!
//! PersonaPlex has no VAD and no response object. This bridge invents
//! `speech_started` / `speech_stopped`, `response.created` / `response.done`,
//! and barge-in `response.cancelled` + `conversation.item.truncated` from
//! input energy and from when the engine actually emits audio or text.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};

use super::codec::{self, rms};
use crate::audio;

/// One chunk of model activity at [`RealtimeEngine::sample_rate`].
#[derive(Debug, Clone, PartialEq)]
pub enum Activity {
    /// Mono PCM. Near-silent buffers do not open or extend a response.
    Audio(Vec<f32>),
    /// Inner-monologue piece. Becomes an audio-transcript delta.
    Transcript(String),
}

/// What the bridge asks the model to do. The WebSocket layer never sees this.
pub trait RealtimeEngine: Send {
    /// Rate of [`Activity::Audio`] and the rate `push_audio` should be given.
    fn sample_rate(&self) -> u32;

    /// Apply persona text and a voice-preset name. Missing presets are not errors.
    fn apply_session(&mut self, instructions: &str, voice: &str) -> Result<(), String>;

    /// Advance the model by this much user audio.
    fn push_audio(&mut self, mono: &[f32], sample_rate: u32) -> Result<Vec<Activity>, String>;

    /// Drop playback. The engine keeps its streaming state.
    fn interrupt(&mut self) -> Result<(), String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Turn {
    ServerVad,
    Manual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    Pcm,
    Pcmu,
    Pcma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    Json,
    Binary,
}

#[derive(Debug, Clone)]
struct Format {
    codec: Codec,
    rate: u32,
    transport: Transport,
}

impl Default for Format {
    fn default() -> Self {
        Self {
            codec: Codec::Pcm,
            rate: 24_000,
            transport: Transport::Json,
        }
    }
}

struct OpenResponse {
    id: String,
    item_id: String,
    transcript: String,
    played_samples: usize,
    quiet_ms: f32,
}

/// Protocol state for one WebSocket connection.
pub struct Bridge<E: RealtimeEngine> {
    engine: E,
    seq: u64,
    instructions: String,
    voice: String,
    model: String,
    turn: Turn,
    threshold: f32,
    silence_ms: f32,
    input: Format,
    output: Format,
    tools: Value,
    transcription: Value,
    speaking: bool,
    trailing_silence_ms: f32,
    response: Option<OpenResponse>,
    /// User speech started during a response; hold new audio until they stop.
    held: bool,
    stash: Vec<Activity>,
    /// Manual mode: `response.create` opened a response that should receive audio.
    armed: bool,
    applied_key: Option<(String, String)>,
    last_item_id: String,
    last_transcript: String,
    pending_binary: Vec<Vec<u8>>,
}

impl<E: RealtimeEngine> Bridge<E> {
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            seq: 0,
            instructions: String::new(),
            voice: String::new(),
            model: "personaplex".to_string(),
            turn: Turn::ServerVad,
            threshold: 0.5,
            silence_ms: 500.0,
            input: Format::default(),
            output: Format::default(),
            tools: json!([]),
            transcription: Value::Null,
            speaking: false,
            trailing_silence_ms: 0.0,
            response: None,
            held: false,
            stash: Vec::new(),
            armed: false,
            applied_key: None,
            last_item_id: String::new(),
            last_transcript: String::new(),
            pending_binary: Vec::new(),
        }
    }

    /// Raw codec bytes produced while `audio.output.transport` is `binary`.
    pub fn take_binary(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.pending_binary)
    }

    pub fn set_advertised_model(&mut self, model: impl Into<String>) {
        let model = model.into();
        if !model.is_empty() {
            self.model = model;
        }
    }

    /// Persona and voice used until the client sends `session.update`.
    pub fn with_defaults(
        mut self,
        instructions: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        self.instructions = instructions.into();
        self.voice = voice.into();
        self
    }

    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// `session.created` and `conversation.created`, in that order.
    pub fn hello(&mut self) -> Vec<Value> {
        vec![
            self.event(json!({
                "type": "session.created",
                "session": self.session_body(),
            })),
            self.event(json!({
                "type": "conversation.created",
                "conversation": { "id": "conv_1" },
            })),
        ]
    }

    pub fn handle_text(&mut self, text: &str) -> Vec<Value> {
        let value: Value = match serde_json::from_str(text) {
            Ok(value) => value,
            Err(err) => return vec![self.error("invalid_json", &err.to_string())],
        };
        let kind = value.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "session.update" => self.session_update(&value),
            "input_audio_buffer.append" => {
                let audio = value.get("audio").and_then(|a| a.as_str()).unwrap_or("");
                match STANDARD.decode(audio) {
                    Ok(bytes) => self.append_encoded(&bytes),
                    Err(err) => vec![self.error("invalid_audio", &err.to_string())],
                }
            }
            "input_audio_buffer.commit" => self.commit(),
            "input_audio_buffer.clear" => {
                vec![self.event(json!({"type": "input_audio_buffer.cleared"}))]
            }
            "conversation.item.create" => self.item_create(&value),
            "conversation.item.truncate" => self.truncate_from_client(&value),
            "conversation.item.delete" => {
                vec![self.error("unsupported", "conversation.item.delete is not supported")]
            }
            "response.create" => self.response_create(),
            "response.cancel" => self.response_cancel(),
            "" => vec![self.error("invalid_request", "event type is missing")],
            other => vec![self.error("unsupported", &format!("unsupported event {other}"))],
        }
    }

    /// Binary frames are raw codec bytes at the configured input format.
    pub fn handle_binary(&mut self, bytes: &[u8]) -> Vec<Value> {
        self.append_encoded(bytes)
    }

    fn session_update(&mut self, value: &Value) -> Vec<Value> {
        let session = value.get("session").cloned().unwrap_or(Value::Null);
        if let Some(instructions) = session.get("instructions").and_then(|v| v.as_str()) {
            self.instructions = instructions.to_string();
        }
        if let Some(voice) = session.get("voice").and_then(|v| v.as_str()) {
            self.voice = voice.to_string();
        } else if let Some(voice) = session
            .pointer("/audio/output/voice")
            .and_then(|v| v.as_str())
        {
            self.voice = voice.to_string();
        }
        if let Some(turn) = session.get("turn_detection") {
            if turn.is_null() {
                self.turn = Turn::Manual;
            } else if let Some(kind) = turn.get("type").and_then(|v| v.as_str()) {
                self.turn = if kind == "none" {
                    Turn::Manual
                } else {
                    Turn::ServerVad
                };
            }
            if let Some(threshold) = turn.get("threshold").and_then(|v| v.as_f64()) {
                self.threshold = threshold as f32;
            }
            if let Some(silence) = turn.get("silence_duration_ms").and_then(|v| v.as_f64()) {
                self.silence_ms = silence as f32;
            }
        }
        if let Some(format) = session.pointer("/audio/input/format") {
            if let Some(err) = self.read_format(format, true) {
                return vec![self.error("unsupported_format", &err)];
            }
        }
        if let Some(format) = session.pointer("/audio/output/format") {
            if let Some(err) = self.read_format(format, false) {
                return vec![self.error("unsupported_format", &err)];
            }
        }
        if let Some(err) = self.read_legacy_format(&session, "input_audio_format", true) {
            return vec![self.error("unsupported_format", &err)];
        }
        if let Some(err) = self.read_legacy_format(&session, "output_audio_format", false) {
            return vec![self.error("unsupported_format", &err)];
        }
        if let Some(transport) = session
            .pointer("/audio/input/transport")
            .and_then(|v| v.as_str())
        {
            self.input.transport = transport_of(transport);
        }
        if let Some(transport) = session
            .pointer("/audio/output/transport")
            .and_then(|v| v.as_str())
        {
            self.output.transport = transport_of(transport);
        }
        if let Some(tools) = session.get("tools") {
            self.tools = tools.clone();
        }
        if let Some(transcription) = session.pointer("/audio/input/transcription") {
            self.transcription = transcription.clone();
        }
        if let Err(err) = self.ensure_engine() {
            return vec![self.error("session", &err)];
        }
        vec![self.event(json!({
            "type": "session.updated",
            "session": self.session_body(),
        }))]
    }

    fn read_format(&mut self, format: &Value, input: bool) -> Option<String> {
        let kind = format
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("audio/pcm");
        let codec = match kind {
            "audio/pcm" | "pcm16" => Codec::Pcm,
            "audio/pcmu" | "g711_ulaw" => Codec::Pcmu,
            "audio/pcma" | "g711_alaw" => Codec::Pcma,
            "audio/opus" => {
                return Some("audio/opus is not supported".to_string());
            }
            other => return Some(format!("unsupported audio format {other}")),
        };
        let mut slot = if input {
            self.input.clone()
        } else {
            self.output.clone()
        };
        slot.codec = codec;
        if codec == Codec::Pcm {
            if let Some(rate) = format.get("rate").and_then(|v| v.as_u64()) {
                if rate == 0 {
                    return Some("sample rate must be positive".to_string());
                }
                slot.rate = rate as u32;
            }
        } else {
            slot.rate = 8_000;
        }
        if input {
            self.input = slot;
        } else {
            self.output = slot;
        }
        None
    }

    fn read_legacy_format(&mut self, session: &Value, key: &str, input: bool) -> Option<String> {
        let name = session.get(key).and_then(|v| v.as_str())?;
        let value = json!({"type": name});
        self.read_format(&value, input)
    }

    fn commit(&mut self) -> Vec<Value> {
        if self.turn != Turn::Manual {
            return vec![self.error(
                "invalid_request",
                "input_audio_buffer.commit is only valid when turn_detection is null",
            )];
        }
        vec![self.event(json!({"type": "input_audio_buffer.committed"}))]
    }

    fn item_create(&mut self, value: &Value) -> Vec<Value> {
        let item = value.get("item").cloned().unwrap_or(Value::Null);
        let kind = item
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");
        if kind == "function_call" || kind == "function_call_output" {
            return vec![self.error(
                "unsupported",
                "function calls are not supported; PersonaPlex has no tool head",
            )];
        }
        let text = item
            .pointer("/content")
            .and_then(|c| c.as_array())
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|part| part.get("type").and_then(|t| t.as_str()) == Some("input_text"))
            });
        if text || kind == "message" {
            return vec![self.error(
                "unsupported",
                "text-only input is not supported; send input_audio_buffer.append",
            )];
        }
        vec![self.error("unsupported", "conversation.item.create is not supported")]
    }

    fn response_create(&mut self) -> Vec<Value> {
        if self.response.is_some() {
            return vec![self.error("invalid_request", "already an active response")];
        }
        self.armed = true;
        let mut events = self.open_response();
        let stash = std::mem::take(&mut self.stash);
        events.extend(self.absorb(stash, 0.0));
        events
    }

    fn response_cancel(&mut self) -> Vec<Value> {
        if self.response.is_none() {
            return vec![self.error("invalid_request", "no active response")];
        }
        let _ = self.engine.interrupt();
        self.finish(true)
    }

    fn truncate_from_client(&mut self, value: &Value) -> Vec<Value> {
        let item_id = value
            .get("item_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content_index = value
            .get("content_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let audio_end_ms = value
            .get("audio_end_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let live = self
            .response
            .as_ref()
            .filter(|response| response.item_id == item_id)
            .map(|response| response.transcript.clone());
        let known = live.is_some() || (!item_id.is_empty() && item_id == self.last_item_id);
        if !item_id.is_empty() && !known {
            return vec![self.error("invalid_request", "unknown item_id")];
        }
        let transcript = live.unwrap_or_else(|| self.last_transcript.clone());
        vec![self.event(json!({
            "type": "conversation.item.truncated",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
            "transcript": transcript,
        }))]
    }

    fn append_encoded(&mut self, bytes: &[u8]) -> Vec<Value> {
        if let Err(err) = self.ensure_engine() {
            return vec![self.error("session", &err)];
        }
        let samples = match self.input.codec {
            Codec::Pcm => codec::pcm16_to_f32(bytes),
            Codec::Pcmu => codec::mulaw_to_f32(bytes),
            Codec::Pcma => codec::alaw_to_f32(bytes),
        };
        if samples.is_empty() {
            return Vec::new();
        }
        let input_ms = samples.len() as f32 / self.input.rate as f32 * 1000.0;
        let mut events = self.vad(&samples, input_ms);
        let model_rate = self.engine.sample_rate().max(1);
        let mono = audio::linear_resample(&samples, self.input.rate, model_rate);
        let activity = match self.engine.push_audio(&mono, model_rate) {
            Ok(activity) => activity,
            Err(err) => return vec![self.error("session", &err)],
        };
        events.extend(self.absorb(activity, input_ms));
        events
    }

    fn vad(&mut self, samples: &[f32], input_ms: f32) -> Vec<Value> {
        if self.turn != Turn::ServerVad {
            return Vec::new();
        }
        let hot = rms(samples) >= self.threshold;
        if hot {
            self.trailing_silence_ms = 0.0;
            if self.speaking {
                return Vec::new();
            }
            self.speaking = true;
            if self.response.is_some() {
                self.held = true;
                let _ = self.engine.interrupt();
                let mut events = Vec::new();
                events.push(self.speech_started());
                events.extend(self.finish(true));
                return events;
            }
            return vec![self.speech_started()];
        }
        if !self.speaking {
            return Vec::new();
        }
        self.trailing_silence_ms += input_ms;
        if self.trailing_silence_ms < self.silence_ms {
            return Vec::new();
        }
        self.speaking = false;
        self.trailing_silence_ms = 0.0;
        let mut events = vec![
            self.event(json!({"type": "input_audio_buffer.speech_stopped"})),
            self.event(json!({
                "type": "input_audio_buffer.committed",
                "item_id": format!("item_user_{}", self.seq),
            })),
            self.event(json!({
                "type": "conversation.item.added",
                "item": {
                    "id": format!("item_user_{}", self.seq),
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                },
            })),
        ];
        if self.held {
            self.held = false;
            let stash = std::mem::take(&mut self.stash);
            events.extend(self.absorb(stash, 0.0));
        }
        events
    }

    fn speech_started(&mut self) -> Value {
        self.event(json!({
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": 0,
        }))
    }

    fn absorb(&mut self, items: Vec<Activity>, input_ms: f32) -> Vec<Value> {
        if self.held {
            self.stash.extend(items);
            return Vec::new();
        }
        if self.turn == Turn::Manual && self.response.is_none() {
            self.stash.extend(items);
            return Vec::new();
        }
        let mut events = Vec::new();
        let mut produced = false;
        for item in items {
            match item {
                Activity::Audio(pcm) if audible(&pcm) => {
                    produced = true;
                    if self.response.is_none() {
                        events.extend(self.open_response());
                    }
                    if let Some(delta) = self.audio_delta(&pcm) {
                        events.push(delta);
                    }
                }
                Activity::Transcript(text) if !text.is_empty() => {
                    produced = true;
                    if self.response.is_none() {
                        events.extend(self.open_response());
                    }
                    let ids = self.response.as_mut().map(|response| {
                        response.transcript.push_str(&text);
                        response.quiet_ms = 0.0;
                        (response.id.clone(), response.item_id.clone())
                    });
                    if let Some((id, item_id)) = ids {
                        events.push(self.event(json!({
                            "type": "response.output_audio_transcript.delta",
                            "response_id": id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        })));
                    }
                }
                Activity::Audio(_) | Activity::Transcript(_) => {}
            }
        }
        if let Some(response) = self.response.as_mut() {
            if produced {
                response.quiet_ms = 0.0;
            } else {
                response.quiet_ms += input_ms;
            }
        }
        let quiet = self
            .response
            .as_ref()
            .is_some_and(|response| response.quiet_ms >= self.silence_ms);
        if quiet {
            events.extend(self.finish(false));
        }
        events
    }

    fn open_response(&mut self) -> Vec<Value> {
        let id = self.ident("resp");
        let item_id = self.ident("item");
        self.response = Some(OpenResponse {
            id: id.clone(),
            item_id: item_id.clone(),
            transcript: String::new(),
            played_samples: 0,
            quiet_ms: 0.0,
        });
        self.armed = true;
        vec![
            self.event(json!({
                "type": "response.created",
                "response": { "id": id, "status": "in_progress", "output": [] },
            })),
            self.event(json!({
                "type": "response.output_item.added",
                "response_id": id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            })),
            self.event(json!({
                "type": "response.content_part.added",
                "response_id": id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": { "type": "audio", "transcript": "" },
            })),
        ]
    }

    fn audio_delta(&mut self, pcm: &[f32]) -> Option<Value> {
        let rate = self.engine.sample_rate().max(1);
        let out_rate = self.output.rate.max(1);
        let codec = self.output.codec;
        let transport = self.output.transport;
        let pcm = audio::linear_resample(pcm, rate, out_rate);
        let sample_count = pcm.len();
        let bytes = match codec {
            Codec::Pcm => codec::f32_to_pcm16(&pcm),
            Codec::Pcmu => codec::f32_to_mulaw(&pcm),
            Codec::Pcma => codec::f32_to_alaw(&pcm),
        };
        let (id, item_id) = {
            let response = self.response.as_mut()?;
            response.played_samples += sample_count;
            response.quiet_ms = 0.0;
            (response.id.clone(), response.item_id.clone())
        };
        if transport == Transport::Binary {
            self.pending_binary.push(bytes);
            return None;
        }
        Some(self.event(json!({
            "type": "response.output_audio.delta",
            "response_id": id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": STANDARD.encode(bytes),
        })))
    }

    fn finish(&mut self, cancelled: bool) -> Vec<Value> {
        let Some(response) = self.response.take() else {
            return Vec::new();
        };
        self.armed = false;
        self.stash.clear();
        self.last_item_id = response.item_id.clone();
        self.last_transcript = response.transcript.clone();
        let played_ms = if self.output.rate == 0 {
            0
        } else {
            (response.played_samples as f64 / f64::from(self.output.rate) * 1000.0).round() as u64
        };
        if cancelled {
            return vec![
                self.event(json!({
                    "type": "conversation.item.truncated",
                    "item_id": response.item_id,
                    "content_index": 0,
                    "audio_end_ms": played_ms,
                    "transcript": response.transcript,
                })),
                self.event(json!({
                    "type": "response.cancelled",
                    "response": { "id": response.id, "status": "cancelled" },
                })),
            ];
        }
        let id = response.id;
        let item_id = response.item_id;
        let transcript = response.transcript;
        vec![
            self.event(json!({
                "type": "response.output_audio.done",
                "response_id": id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            })),
            self.event(json!({
                "type": "response.output_audio_transcript.done",
                "response_id": id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "transcript": transcript,
            })),
            self.event(json!({
                "type": "response.content_part.done",
                "response_id": id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            })),
            self.event(json!({
                "type": "response.output_item.done",
                "response_id": id,
                "output_index": 0,
                "item": { "id": item_id, "status": "completed" },
            })),
            self.event(json!({
                "type": "response.done",
                "response": { "id": id, "status": "completed" },
            })),
        ]
    }

    fn ensure_engine(&mut self) -> Result<(), String> {
        let key = (self.instructions.clone(), self.voice.clone());
        if self.applied_key.as_ref() == Some(&key) {
            return Ok(());
        }
        self.engine.apply_session(&self.instructions, &self.voice)?;
        self.applied_key = Some(key);
        Ok(())
    }

    fn session_body(&self) -> Value {
        let turn = match self.turn {
            Turn::ServerVad => json!({
                "type": "server_vad",
                "threshold": self.threshold,
                "silence_duration_ms": self.silence_ms as u64,
            }),
            Turn::Manual => Value::Null,
        };
        json!({
            "type": "realtime",
            "model": self.model,
            "instructions": self.instructions,
            "voice": self.voice,
            "turn_detection": turn,
            "tools": self.tools,
            "audio": {
                "input": {
                    "format": format_body(&self.input),
                    "transcription": self.transcription,
                },
                "output": {
                    "format": format_body(&self.output),
                    "voice": self.voice,
                },
            },
        })
    }

    fn event(&mut self, mut value: Value) -> Value {
        if let Some(object) = value.as_object_mut() {
            object.insert("event_id".to_string(), json!(self.ident("evt")));
        }
        value
    }

    fn error(&mut self, code: &str, message: &str) -> Value {
        self.event(json!({
            "type": "error",
            "error": { "type": "invalid_request_error", "code": code, "message": message },
        }))
    }

    fn ident(&mut self, prefix: &str) -> String {
        self.seq += 1;
        format!("{prefix}_{}", self.seq)
    }
}

fn format_body(format: &Format) -> Value {
    let kind = match format.codec {
        Codec::Pcm => "audio/pcm",
        Codec::Pcmu => "audio/pcmu",
        Codec::Pcma => "audio/pcma",
    };
    json!({
        "type": kind,
        "rate": format.rate,
        "transport": match format.transport {
            Transport::Json => "json",
            Transport::Binary => "binary",
        },
    })
}

fn transport_of(name: &str) -> Transport {
    if name == "binary" {
        Transport::Binary
    } else {
        Transport::Json
    }
}

fn audible(samples: &[f32]) -> bool {
    samples.iter().any(|sample| sample.abs() >= 1.0e-3)
}
