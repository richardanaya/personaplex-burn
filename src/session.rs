//! Streaming session. This is the primary public interface.
//!
//! Audio frames go in. Audio, text, and status events come out. Persona and
//! voice are configuration, applied once when the session starts and again on
//! [`Session::reset`]. [`Session::interrupt`] does not clear the KV cache: full
//! duplex keeps running, and adapters drop their playback queues.
//!
//! The same calls work for a microphone, a websocket, or a whole file. Nothing
//! here opens a device, a socket, or a UI. [`push_audio_async`] is a ready
//! future so an async adapter can await it without this crate depending on a
//! runtime.

use std::future::Ready;
use std::time::Instant;

use burn::tensor::{Int, Tensor};
use serde::Serialize;

use crate::audio::{self, FrameBuffer};
use crate::config::{SessionConfig, VoicePrompt};
use crate::error::{Error, Result};
use crate::lm::LmState;
use crate::streaming::StreamingBag;
use crate::util::{self, tensor_to_f32_vec};
use crate::weights::{PersonaPlex, Weights};
use burn::tensor::DType;

/// Which text stream a [`Event::Text`] came from.
///
/// This checkpoint has no separate user-transcript head. [`TextKind::UserTranscript`]
/// is reserved and is not produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TextKind {
    /// Text tokens from the temporal transformer (the model's inner monologue).
    InnerMonologue,
    /// Not emitted. PersonaPlex does not ship a user-transcript head.
    UserTranscript,
}

/// Lifecycle and timing signals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Status {
    /// Weights are loaded and prompts have been applied.
    Ready,
    /// System prompts (voice, silence, persona) are being stepped.
    Prompting,
    /// Waiting on user audio.
    Listening,
    /// An output frame contained samples.
    Speaking,
    /// Caller asked to cut off playback. The KV cache is unchanged.
    Interrupted,
    /// Streaming state was cleared and prompts were applied again.
    Reset,
    /// Input frames are replaced with zeros until unmute.
    Muted,
    /// Wall time spent inside one model frame.
    Frame {
        /// Frame index since the last reset, starting at 0.
        index: u64,
        /// Milliseconds spent encoding, stepping, and decoding this frame.
        elapsed_ms: f32,
    },
}

/// One item from the session.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Event {
    /// PCM at the model rate, mono, as soon as a frame is decoded.
    AudioOut { pcm: Vec<f32>, sample_rate: u32 },
    /// One decoded text token.
    Text {
        kind: TextKind,
        text: String,
        token: i32,
    },
    /// Status change or frame timing.
    Status(Status),
}

/// Full-duplex streaming state over one [`PersonaPlex`] model.
pub struct Session {
    model: PersonaPlex,
    lm: LmState,
    mimi: StreamingBag,
    frames: FrameBuffer,
    cfg: SessionConfig,
    muted: bool,
    frame_index: u64,
    prompt_log: Vec<Event>,
    silence: Vec<i32>,
    sine: Vec<i32>,
}

impl Session {
    /// Apply persona and voice prompts, then accept audio.
    pub fn start(model: PersonaPlex, cfg: SessionConfig) -> Result<Self> {
        let lm = LmState::new(&model.config().lm, cfg.sampling)?;
        let width = model.config().lm.audio_tokens_per_stream;
        let mut session = Self {
            silence: fit_tokens(&model.config().lm.silence_tokens, width),
            sine: fit_tokens(&model.config().lm.sine_tokens, width),
            model,
            lm,
            mimi: StreamingBag::new(),
            frames: FrameBuffer::new(0),
            cfg,
            muted: false,
            frame_index: 0,
            prompt_log: Vec::new(),
        };
        session.frames = FrameBuffer::new(session.model.config().frame_size());
        session.prompt_log = session.run_prompts()?;
        Ok(session)
    }

    /// Events produced while applying prompts inside [`Self::start`] or [`Self::reset`].
    pub fn prompt_events(&self) -> &[Event] {
        &self.prompt_log
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.config().sample_rate()
    }

    pub fn frame_size(&self) -> usize {
        self.model.config().frame_size()
    }

    /// Push interleaved PCM at any rate and channel count.
    ///
    /// Samples are downmixed and linearly resampled to the model rate. They are
    /// not loudness-normalized; only the voice prompt is.
    pub fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: usize,
    ) -> Result<Vec<Event>> {
        let mono = audio::downmix(samples, channels)?;
        let mono = audio::linear_resample(&mono, sample_rate, self.sample_rate());
        let mono = if self.muted {
            vec![0.0; mono.len()]
        } else {
            mono
        };
        let frames = self.frames.push(&mono);
        let mut events = Vec::new();
        for frame in frames {
            events.extend(self.step_frame(&frame)?);
        }
        Ok(events)
    }

    /// Same work as [`Self::push_audio`], as a ready future. No async runtime required.
    pub fn push_audio_async(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: usize,
    ) -> Ready<Result<Vec<Event>>> {
        std::future::ready(self.push_audio(samples, sample_rate, channels))
    }

    /// Zero-pad the input tail into one last frame.
    pub fn finish(&mut self) -> Result<Vec<Event>> {
        match self.frames.finish() {
            Some(frame) => self.step_frame(&frame),
            None => Ok(Vec::new()),
        }
    }

    /// Signal that playback should stop. The model keeps its cache.
    pub fn interrupt(&mut self) -> Event {
        Event::Status(Status::Interrupted)
    }

    /// When muted, further input frames are silence.
    pub fn set_muted(&mut self, muted: bool) -> Event {
        self.muted = muted;
        Event::Status(if muted {
            Status::Muted
        } else {
            Status::Listening
        })
    }

    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Drop streaming state, re-seed the sampler, and re-apply prompts.
    pub fn reset(&mut self) -> Result<Vec<Event>> {
        self.mimi.reset();
        self.frames.clear();
        self.lm.reset(self.cfg.sampling);
        self.frame_index = 0;
        self.muted = false;
        let mut events = vec![Event::Status(Status::Reset)];
        let prompts = self.run_prompts()?;
        self.prompt_log = prompts.clone();
        events.extend(prompts);
        Ok(events)
    }

    fn step_frame(&mut self, frame: &[f32]) -> Result<Vec<Event>> {
        let started = Instant::now();
        let codes = self.encode_frame(frame)?;
        let mut events = Vec::new();
        let time = codes.time.max(1);
        if codes.values.is_empty() {
            return Ok(events);
        }
        for step in 0..time {
            let user = column(&codes.values, codes.codebooks, time, step);
            let tokens = self.lm.step(self.model.lm(), Some(&user), None, None);
            if let Some(tokens) = tokens {
                events.extend(self.emit_tokens(&tokens)?);
            }
        }
        if self.cfg.emit_timing {
            events.push(Event::Status(Status::Frame {
                index: self.frame_index,
                elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
            }));
        }
        self.frame_index += 1;
        Ok(events)
    }

    fn emit_tokens(&mut self, tokens: &[i32]) -> Result<Vec<Event>> {
        let width = self.model.config().lm.audio_tokens_per_stream;
        if tokens.len() < 1 + width {
            return Err(Error::session("model step returned too few codebooks"));
        }
        let agent = tokens[1..1 + width].to_vec();
        let codes = util::i32_tensor(agent, [1, width, 1], self.model.device());
        let pcm = self.model.mimi().decode(codes, Some(&mut self.mimi));
        let samples = flatten_f32(pcm);
        let mut events = vec![Event::AudioOut {
            pcm: samples,
            sample_rate: self.sample_rate(),
        }];
        if !events[0].is_silent_audio() {
            events.push(Event::Status(Status::Speaking));
        }
        let token = tokens[0];
        events.push(Event::Text {
            kind: TextKind::InnerMonologue,
            text: display_token(self.model.tokenizer(), token),
            token,
        });
        Ok(events)
    }

    fn encode_frame(&mut self, frame: &[f32]) -> Result<Codes> {
        let tensor = util::f32_tensor(frame.to_vec(), [1, 1, frame.len()], self.model.device());
        let encoded = self.model.mimi().encode(tensor, Some(&mut self.mimi));
        Ok(Codes::from_tensor(encoded))
    }

    fn run_prompts(&mut self) -> Result<Vec<Event>> {
        let mut events = vec![
            Event::Status(Status::Ready),
            Event::Status(Status::Prompting),
        ];
        self.feed_voice()?;
        self.feed_silence();
        self.feed_text();
        self.feed_silence();
        self.mimi.reset();
        events.push(Event::Status(Status::Listening));
        Ok(events)
    }

    fn feed_voice(&mut self) -> Result<()> {
        match self.cfg.voice.clone() {
            VoicePrompt::None => Ok(()),
            VoicePrompt::Pcm {
                samples,
                sample_rate,
                channels,
            } => self.feed_voice_pcm(&samples, sample_rate, channels),
            VoicePrompt::Wav(path) => {
                let (samples, sample_rate, channels) = audio::read_wav(&path)?;
                self.feed_voice_pcm(&samples, sample_rate, channels)
            }
            VoicePrompt::Embeddings(path) => self.feed_voice_embeddings(&path),
        }
    }

    fn feed_voice_pcm(&mut self, samples: &[f32], sample_rate: u32, channels: usize) -> Result<()> {
        let mono = audio::downmix(samples, channels)?;
        let mono = audio::linear_resample(&mono, sample_rate, self.sample_rate());
        let (mono, _) = audio::normalize_lufs(&mono, self.sample_rate(), -24.0)?;
        let frame_size = self.frame_size();
        let mut buffer = FrameBuffer::new(frame_size);
        let mut frames = buffer.push(&mono);
        if let Some(tail) = buffer.finish() {
            frames.push(tail);
        }
        for frame in frames {
            let codes = self.encode_frame(&frame)?;
            let time = codes.time.max(1);
            for step in 0..codes.time {
                let moshi = column(&codes.values, codes.codebooks, time, step);
                let _ = self.lm.step(
                    self.model.lm(),
                    Some(&self.sine),
                    Some(&moshi),
                    Some(self.model.config().lm.text_padding_id()),
                );
            }
        }
        Ok(())
    }

    fn feed_voice_embeddings(&mut self, path: &std::path::Path) -> Result<()> {
        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("pt") {
            return Err(Error::checkpoint(
                "PyTorch .pt voice caches are not loaded. Save a safetensors file with `embeddings` [frames, dim] and optional `cache` [codebooks, cache_len].",
            ));
        }
        let weights = Weights::from_file(path, DType::F32)?;
        let (shape, data) = weights.copy_f32("embeddings")?;
        let (frames, dim) = match shape.as_slice() {
            [frames, dim] => (*frames, *dim),
            [frames, 1, dim] => (*frames, *dim),
            [frames, 1, 1, dim] => (*frames, *dim),
            other => {
                return Err(Error::checkpoint(format!(
                    "voice embeddings shape {other:?} is not [frames, dim]"
                )))
            }
        };
        for frame in 0..frames {
            let start = frame * dim;
            let embed = util::f32_tensor(
                data[start..start + dim].to_vec(),
                [1, 1, dim],
                self.model.device(),
            );
            let _ = self.lm.step_embeddings(self.model.lm(), embed);
        }
        if weights.contains("cache") {
            let (shape, cache) = weights.copy_f32("cache")?;
            let (codebooks, len, values) = match shape.as_slice() {
                [k, ct] => (*k, *ct, cache.iter().map(|v| *v as i32).collect::<Vec<_>>()),
                [1, k, ct] => (*k, *ct, cache.iter().map(|v| *v as i32).collect::<Vec<_>>()),
                other => {
                    return Err(Error::checkpoint(format!(
                        "voice cache shape {other:?} is not [codebooks, cache_len]"
                    )))
                }
            };
            self.lm.install_cache(&values, codebooks, len)?;
        }
        Ok(())
    }

    fn feed_silence(&mut self) {
        let pad = self.model.config().lm.text_padding_id();
        for _ in 0..self.cfg.silence_frames {
            let _ = self.lm.step(
                self.model.lm(),
                Some(&self.sine),
                Some(&self.silence),
                Some(pad),
            );
        }
    }

    fn feed_text(&mut self) {
        let persona = wrap_system(&self.cfg.persona);
        if persona.is_empty() {
            return;
        }
        let tokens = self.model.tokenizer().encode(&persona);
        for token in tokens {
            let _ = self.lm.step(
                self.model.lm(),
                Some(&self.sine),
                Some(&self.silence),
                Some(token),
            );
        }
    }
}

impl Event {
    fn is_silent_audio(&self) -> bool {
        match self {
            Self::AudioOut { pcm, .. } => pcm.iter().all(|sample| sample.abs() < 1.0e-6),
            _ => true,
        }
    }
}

struct Codes {
    values: Vec<i32>,
    codebooks: usize,
    time: usize,
}

impl Codes {
    fn from_tensor(tensor: Tensor<3, Int>) -> Self {
        let [_, codebooks, time] = tensor.dims();
        let values = tensor
            .reshape([codebooks * time])
            .try_into_vec_as::<i32>()
            .unwrap_or_default();
        Self {
            values,
            codebooks,
            time,
        }
    }
}

fn column(values: &[i32], codebooks: usize, time: usize, step: usize) -> Vec<i32> {
    (0..codebooks)
        .map(|codebook| values[codebook * time + step])
        .collect()
}

fn flatten_f32(tensor: Tensor<3>) -> Vec<f32> {
    let n: usize = tensor.dims().iter().product();
    tensor_to_f32_vec(tensor.reshape([n]))
}

fn fit_tokens(tokens: &[i32], width: usize) -> Vec<i32> {
    let mut out = vec![0; width];
    for (index, token) in tokens.iter().take(width).enumerate() {
        out[index] = *token;
    }
    out
}

fn display_token(tokenizer: &crate::tokenizer::Tokenizer, token: i32) -> String {
    match token {
        0 => "EPAD".to_string(),
        3 => "PAD".to_string(),
        other => tokenizer.id_to_piece(other).replace('▁', " "),
    }
}

pub(crate) fn wrap_system(persona: &str) -> String {
    let cleaned = persona.trim();
    if cleaned.is_empty() {
        return String::new();
    }
    if cleaned.starts_with("<system>") && cleaned.ends_with("<system>") {
        cleaned.to_string()
    } else {
        format!("<system> {cleaned} <system>")
    }
}
