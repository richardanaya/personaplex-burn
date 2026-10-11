//! Architecture and session configuration.
//!
//! [`ModelConfig::personaplex_7b`] matches `moshi.models.loaders` in the
//! NVIDIA reference. Smaller configs are for parity fixtures.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Where a conditioning voice comes from.
///
/// This is the reference's voice prompt: a clip encoded by Mimi, or a cached
/// embedding tensor produced the same way (`embeddings` plus the LM token cache).
#[derive(Debug, Clone)]
pub enum VoicePrompt {
    /// Interleaved PCM. Any rate and channel count; the session downmixes and
    /// resamples to the model rate and normalizes to -24 LUFS.
    Pcm {
        /// Interleaved samples.
        samples: Vec<f32>,
        /// Source sample rate.
        sample_rate: u32,
        /// Channel count. `1` is mono.
        channels: usize,
    },
    /// A WAV file (any rate / channel layout hound can read).
    Wav(std::path::PathBuf),
    /// Safetensors with `embeddings` `[frames, dim]` and optional `cache`
    /// `[codebooks, cache_len]`, or a reference `.pt` dict with those keys.
    Embeddings(std::path::PathBuf),
    /// No voice prompt. The released model is trained with one.
    None,
}

impl VoicePrompt {
    /// Load a voice clip from a path. `.wav` is PCM; anything else is treated
    /// as a cached embedding file.
    pub fn from_path(path: impl Into<std::path::PathBuf>) -> Self {
        let path = path.into();
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
        {
            Self::Wav(path)
        } else {
            Self::Embeddings(path)
        }
    }
}

/// Sampling for the temporal (text) and depth (audio) transformers.
///
/// Defaults match `LMGen` in the reference (`temp` 0.8 / `temp_text` 0.7,
/// `top_k` 250 / `top_k_text` 25).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Audio depth-transformer temperature. `0` is greedy.
    pub temperature: f32,
    /// Text (temporal) temperature. `0` is greedy.
    pub temperature_text: f32,
    /// Audio top-k. `0` disables the cutoff.
    pub top_k: usize,
    /// Text top-k. `0` disables the cutoff.
    pub top_k_text: usize,
    /// Seed for the portable SplitMix64 sampler.
    pub seed: u64,
    /// When false, both heads are argmax and `seed` is unused.
    pub sample: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            temperature_text: 0.7,
            top_k: 250,
            top_k_text: 25,
            seed: 42,
            sample: true,
        }
    }
}

impl SamplingConfig {
    /// Argmax on both heads.
    pub fn greedy() -> Self {
        Self {
            sample: false,
            temperature: 0.0,
            temperature_text: 0.0,
            top_k: 0,
            top_k_text: 0,
            seed: 0,
        }
    }
}

/// Persona text plus voice, sampling, and the silence spacer used by the reference.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Role prompt. Wrapped as `<system> … <system>` when those tags are absent.
    pub persona: String,
    /// Voice clip or cached embedding.
    pub voice: VoicePrompt,
    /// Token sampling.
    pub sampling: SamplingConfig,
    /// Silence frames after the voice prompt and again after the text prompt.
    /// The reference uses `0.5 * frame_rate` (6 frames at 12.5 Hz).
    pub silence_frames: usize,
    /// Emit per-frame timing status events.
    pub emit_timing: bool,
}

impl SessionConfig {
    /// Persona text, default sampling, half a second of silence spacers, no voice yet.
    pub fn new(persona: impl Into<String>) -> Self {
        Self {
            persona: persona.into(),
            voice: VoicePrompt::None,
            sampling: SamplingConfig::default(),
            silence_frames: 6,
            emit_timing: true,
        }
    }

    /// Attach a voice prompt.
    pub fn with_voice(mut self, voice: VoicePrompt) -> Self {
        self.voice = voice;
        self
    }

    /// Replace sampling settings.
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Override the post-prompt silence spacer.
    pub fn with_silence_frames(mut self, frames: usize) -> Self {
        self.silence_frames = frames;
        self
    }
}

/// Mimi SEANet + transformer + split residual vector quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimiConfig {
    pub sample_rate: u32,
    pub frame_rate: f64,
    pub channels: usize,
    pub dimension: usize,
    pub n_filters: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub n_residual_layers: usize,
    pub dilation_base: usize,
    pub compress: usize,
    pub causal: bool,
    pub transformer_layers: usize,
    pub transformer_heads: usize,
    pub transformer_dim: usize,
    pub transformer_ff: usize,
    pub transformer_context: usize,
    pub layer_scale: Option<f32>,
    pub max_period: f32,
    pub quantizer_dim: usize,
    pub quantizer_bins: usize,
    /// Total codebooks stored in the checkpoint (32 for the release Mimi).
    pub quantizer_n_q: usize,
    /// Semantic codebooks. The release model uses 1.
    pub n_q_semantic: usize,
    /// Codebooks actually encoded and decoded. The release server uses 8.
    pub active_codebooks: usize,
}

impl MimiConfig {
    pub fn encoder_frame_rate(&self) -> f64 {
        let hop: usize = self.ratios.iter().product();
        self.sample_rate as f64 / hop as f64
    }

    pub fn resample_stride(&self) -> usize {
        let ratio = self.encoder_frame_rate() / self.frame_rate;
        ratio.round() as usize
    }
}

/// Temporal transformer + depth transformer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmConfig {
    pub delays: Vec<usize>,
    pub n_q: usize,
    pub dep_q: usize,
    pub card: usize,
    pub text_card: usize,
    pub dim: usize,
    pub num_heads: usize,
    pub hidden_scale: f64,
    pub num_layers: usize,
    pub context: usize,
    pub max_period: f32,
    pub depformer_dim: usize,
    pub depformer_num_heads: usize,
    pub depformer_num_layers: usize,
    pub depformer_dim_feedforward: usize,
    pub depformer_weights_per_step: bool,
    pub depformer_multi_linear: bool,
    pub existing_text_padding_id: i32,
    /// Codebooks on the user stream and on the agent stream (8 for PersonaPlex).
    pub audio_tokens_per_stream: usize,
    /// Constant Mimi tokens used as acoustic silence while prompting.
    pub silence_tokens: Vec<i32>,
    /// Constant Mimi tokens for the 440 Hz sine placeholder on the user stream.
    pub sine_tokens: Vec<i32>,
}

impl LmConfig {
    pub fn dim_feedforward(&self) -> usize {
        (self.hidden_scale * self.dim as f64) as usize
    }

    pub fn num_codebooks(&self) -> usize {
        self.n_q + 1
    }

    pub fn text_padding_id(&self) -> i32 {
        self.existing_text_padding_id
    }

    pub fn initial_audio_id(&self) -> i32 {
        self.card as i32
    }

    pub fn initial_text_id(&self) -> i32 {
        self.text_card as i32
    }

    pub fn validate(&self) -> Result<()> {
        if self.delays.len() != self.num_codebooks() {
            return Err(Error::checkpoint(format!(
                "delays len {} != codebooks {}",
                self.delays.len(),
                self.num_codebooks()
            )));
        }
        if self.audio_tokens_per_stream * 2 != self.n_q {
            return Err(Error::checkpoint(
                "PersonaPlex expects n_q == 2 * audio_tokens_per_stream",
            ));
        }
        if !self.dim.is_multiple_of(self.num_heads)
            || !self.depformer_dim.is_multiple_of(self.depformer_num_heads)
        {
            return Err(Error::checkpoint(
                "model dim must be divisible by the head count",
            ));
        }
        Ok(())
    }
}

/// Full PersonaPlex checkpoint description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub mimi: MimiConfig,
    pub lm: LmConfig,
}

impl ModelConfig {
    /// Release configuration of `nvidia/personaplex-7b-v1`.
    pub fn personaplex_7b() -> Self {
        let delays = vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];
        Self {
            mimi: MimiConfig {
                sample_rate: 24_000,
                frame_rate: 12.5,
                channels: 1,
                dimension: 512,
                n_filters: 64,
                ratios: vec![8, 6, 5, 4],
                kernel_size: 7,
                residual_kernel_size: 3,
                last_kernel_size: 3,
                n_residual_layers: 1,
                dilation_base: 2,
                compress: 2,
                causal: true,
                transformer_layers: 8,
                transformer_heads: 8,
                transformer_dim: 512,
                transformer_ff: 2048,
                transformer_context: 250,
                layer_scale: Some(0.01),
                max_period: 10_000.0,
                quantizer_dim: 256,
                quantizer_bins: 2048,
                quantizer_n_q: 32,
                n_q_semantic: 1,
                active_codebooks: 8,
            },
            lm: LmConfig {
                delays,
                n_q: 16,
                dep_q: 16,
                card: 2048,
                text_card: 32_000,
                dim: 4096,
                num_heads: 32,
                hidden_scale: 4.125,
                num_layers: 32,
                context: 3000,
                max_period: 10_000.0,
                depformer_dim: 1024,
                depformer_num_heads: 16,
                depformer_num_layers: 6,
                depformer_dim_feedforward: (4.125 * 1024.0) as usize,
                depformer_weights_per_step: true,
                depformer_multi_linear: true,
                existing_text_padding_id: 3,
                audio_tokens_per_stream: 8,
                silence_tokens: vec![948, 243, 1178, 546, 1736, 1030, 1978, 2008],
                sine_tokens: vec![430, 1268, 381, 1611, 1095, 1495, 56, 472],
            },
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.mimi.sample_rate
    }

    pub fn frame_rate(&self) -> f64 {
        self.mimi.frame_rate
    }

    /// Samples per streamed frame. 1920 at 24 kHz / 12.5 Hz, which is 80 ms.
    pub fn frame_size(&self) -> usize {
        (self.mimi.sample_rate as f64 / self.mimi.frame_rate).round() as usize
    }

    pub fn validate(&self) -> Result<()> {
        self.lm.validate()?;
        if self.mimi.active_codebooks != self.lm.audio_tokens_per_stream {
            return Err(Error::checkpoint(
                "Mimi active codebooks must match the per-stream audio width",
            ));
        }
        Ok(())
    }
}

/// Hidden size of a SiLU gating FFN, matching `moshi.modules.gating`.
pub fn gating_hidden(dim: usize, dim_feedforward: usize) -> usize {
    if dim_feedforward == 4 * dim {
        (21 * dim) / 8
    } else {
        (2 * dim_feedforward) / 3
    }
}
