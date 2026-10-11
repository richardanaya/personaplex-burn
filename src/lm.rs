//! Temporal transformer, depth transformer, and the LMGen delay cache.
//!
//! [`LmState::step`] follows `LMGen.prepare_step_input` and
//! `process_transformer_output` from the NVIDIA reference, including the
//! offset-0 initialization and the rule that depth-transformer outputs are
//! sampled tokens while the next depth step may be teacher-forced.

use burn::module::Param;
use burn::tensor::{Device, Int, Tensor};

use crate::config::{LmConfig, SamplingConfig};
use crate::error::{Error, Result};
use crate::norm::{Norm, RmsNorm};
use crate::sampling::Sampler;
use crate::streaming::StreamingBag;
use crate::transformer::{Transformer, TransformerSpec};
use crate::util::{self, RowChunks};
use crate::weights::Weights;

pub struct LmModel {
    emb: Vec<RowChunks>,
    text_emb: RowChunks,
    text_linear: RowChunks,
    transformer: Transformer,
    out_norm: Norm,
    depformer_in: Vec<RowChunks>,
    depformer_emb: Vec<RowChunks>,
    depformer_text_emb: RowChunks,
    depformer: Transformer,
    linears: Vec<RowChunks>,
    device: Device,
    n_q: usize,
    dep_q: usize,
}

struct Prepared {
    input: Vec<i32>,
    provided: Vec<bool>,
    target: Vec<i32>,
    model_input: usize,
    target_pos: usize,
}

/// Delay cache, depth-transformer sampler, and the temporal KV bag.
pub struct LmState {
    cache: Vec<i32>,
    provided: Vec<bool>,
    offset: usize,
    ct: usize,
    codebooks: usize,
    max_delay: usize,
    delays: Vec<usize>,
    initial: Vec<i32>,
    dep_q: usize,
    audio_per_stream: usize,
    text_pad: i32,
    stream: StreamingBag,
    sampler: Sampler,
}

impl LmModel {
    pub fn from_weights(cfg: &LmConfig, weights: &mut Weights, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let mut emb = Vec::with_capacity(cfg.n_q);
        for index in 0..cfg.n_q {
            emb.push(weights.matrix(&format!("emb.{index}.weight"), device)?);
        }
        let mut depformer_in = Vec::with_capacity(cfg.dep_q);
        for index in 0..cfg.dep_q {
            depformer_in.push(weights.matrix(&format!("depformer_in.{index}.weight"), device)?);
        }
        let mut depformer_emb = Vec::with_capacity(cfg.dep_q.saturating_sub(1));
        for index in 0..cfg.dep_q.saturating_sub(1) {
            depformer_emb.push(weights.matrix(&format!("depformer_emb.{index}.weight"), device)?);
        }
        let mut linears = Vec::with_capacity(cfg.dep_q);
        for index in 0..cfg.dep_q {
            linears.push(weights.matrix(&format!("linears.{index}.weight"), device)?);
        }
        let temporal = TransformerSpec {
            layers: cfg.num_layers,
            dim: cfg.dim,
            heads: cfg.num_heads,
            dim_feedforward: cfg.dim_feedforward(),
            context: Some(cfg.context),
            rope: true,
            max_period: cfg.max_period,
            gating: true,
            rms: true,
            weights_per_step: 0,
            layer_scale: false,
        };
        let depth = TransformerSpec {
            layers: cfg.depformer_num_layers,
            dim: cfg.depformer_dim,
            heads: cfg.depformer_num_heads,
            dim_feedforward: cfg.depformer_dim_feedforward,
            context: None,
            rope: false,
            max_period: cfg.max_period,
            gating: true,
            rms: true,
            weights_per_step: if cfg.depformer_weights_per_step {
                cfg.dep_q
            } else {
                0
            },
            layer_scale: false,
        };
        Ok(Self {
            emb,
            text_emb: weights.matrix("text_emb.weight", device)?,
            text_linear: weights.matrix("text_linear.weight", device)?,
            transformer: Transformer::load(weights, "transformer", &temporal, device)?,
            out_norm: load_rms(weights, "out_norm", device)?,
            depformer_in,
            depformer_emb,
            depformer_text_emb: weights.matrix("depformer_text_emb.weight", device)?,
            depformer: Transformer::load(weights, "depformer", &depth, device)?,
            linears,
            device: device.clone(),
            n_q: cfg.n_q,
            dep_q: cfg.dep_q,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward_codes(
        &self,
        sequence: Tensor<3, Int>,
        stream: Option<&mut StreamingBag>,
    ) -> (Tensor<3>, Tensor<3>) {
        self.forward_embeddings(self.embed_codes(sequence), stream)
    }

    pub fn forward_embeddings(
        &self,
        input: Tensor<3>,
        stream: Option<&mut StreamingBag>,
    ) -> (Tensor<3>, Tensor<3>) {
        let hidden = self.transformer.forward(input, stream);
        let hidden = self.out_norm.forward(hidden);
        let text = self.text_linear.matmul(hidden.clone());
        (hidden, text)
    }

    /// Non-streaming depth transformer. Logits are `[batch, dep_q, time, card]`.
    pub fn depformer_logits_training(
        &self,
        sequence: Tensor<3, Int>,
        transformer_out: Tensor<3>,
    ) -> Tensor<4> {
        let [b, _, t] = sequence.dims();
        let mut steps = Vec::with_capacity(self.dep_q);
        for cb in 0..self.dep_q {
            let projected = self.depformer_in[cb].matmul(transformer_out.clone());
            let token = if cb == 0 {
                self.depformer_text_emb
                    .embed(sequence.clone().narrow(1, 0, 1).squeeze_dim::<2>(1), -1)
            } else {
                self.depformer_emb[cb - 1]
                    .embed(sequence.clone().narrow(1, cb, 1).squeeze_dim::<2>(1), -1)
            };
            steps.push((projected + token).unsqueeze_dim::<4>(2));
        }
        let stacked = Tensor::cat(steps, 2);
        let dim = stacked.dims()[3];
        let flat = stacked.reshape([b * t, self.dep_q, dim]);
        let output = self.depformer.forward(flat, None);
        let mut logits = Vec::with_capacity(self.dep_q);
        for cb in 0..self.dep_q {
            let slice = output.clone().narrow(1, cb, 1);
            let row = self.linears[cb].matmul(slice);
            let card = row.dims()[2];
            logits.push(row.reshape([b, t, card]).unsqueeze_dim::<4>(1));
        }
        Tensor::cat(logits, 1)
    }

    fn embed_codes(&self, sequence: Tensor<3, Int>) -> Tensor<3> {
        let mut acc: Option<Tensor<3>> = None;
        for cb in 0..self.n_q {
            let ids = sequence.clone().narrow(1, cb + 1, 1).squeeze_dim::<2>(1);
            let emb = self.emb[cb].embed(ids, -1);
            acc = Some(match acc {
                None => emb,
                Some(prev) => prev + emb,
            });
        }
        let text = self
            .text_emb
            .embed(sequence.narrow(1, 0, 1).squeeze_dim::<2>(1), -1);
        match acc {
            None => text,
            Some(audio) => audio + text,
        }
    }

    fn depformer_step_logits(
        &self,
        cb: usize,
        prev: i32,
        transformer_out: Tensor<3>,
        bag: &mut StreamingBag,
    ) -> Tensor<3> {
        let projected = self.depformer_in[cb].matmul(transformer_out);
        let ids = util::i32_tensor(vec![prev], [1, 1], &projected.device());
        let emb = if cb == 0 {
            self.depformer_text_emb.embed(ids, -1)
        } else {
            self.depformer_emb[cb - 1].embed(ids, -1)
        };
        let hidden = self.depformer.forward(projected + emb, Some(bag));
        self.linears[cb].matmul(hidden)
    }
}

fn load_rms(weights: &mut Weights, prefix: &str, device: &Device) -> Result<Norm> {
    let alpha = weights.tensor3(&format!("{prefix}.alpha"), device)?;
    let dim = alpha.dims()[2];
    Ok(Norm::Rms(RmsNorm {
        alpha: Param::from_tensor(alpha.reshape([dim])),
        eps: 1e-8,
    }))
}

impl LmState {
    pub fn new(cfg: &LmConfig, sampling: SamplingConfig) -> Result<Self> {
        cfg.validate()?;
        let max_delay = cfg.delays.iter().copied().max().unwrap_or(0);
        let ct = max_delay + 3;
        let codebooks = cfg.num_codebooks();
        let mut initial = vec![cfg.initial_audio_id(); codebooks];
        initial[0] = cfg.initial_text_id();
        Ok(Self {
            cache: vec![-2; codebooks * ct],
            provided: vec![false; codebooks * ct],
            offset: 0,
            ct,
            codebooks,
            max_delay,
            delays: cfg.delays.clone(),
            initial,
            dep_q: cfg.dep_q,
            audio_per_stream: cfg.audio_tokens_per_stream,
            text_pad: cfg.text_padding_id(),
            stream: StreamingBag::new(),
            sampler: Sampler::new(sampling),
        })
    }

    pub fn reset(&mut self, sampling: SamplingConfig) {
        self.cache.fill(-2);
        self.provided.fill(false);
        self.offset = 0;
        self.stream.reset();
        self.sampler = Sampler::new(sampling);
    }

    /// Replace the delay cache after replaying stored voice embeddings.
    /// `cache` is row-major `[codebooks, cache_len]` or `[1, codebooks, cache_len]`.
    pub fn install_cache(
        &mut self,
        cache: &[i32],
        codebooks: usize,
        cache_len: usize,
    ) -> Result<()> {
        if codebooks != self.codebooks
            || cache_len != self.ct
            || cache.len() != codebooks * cache_len
        {
            return Err(Error::session(format!(
                "voice cache shape {codebooks}x{cache_len} does not match {}x{}",
                self.codebooks, self.ct
            )));
        }
        self.cache.copy_from_slice(cache);
        Ok(())
    }

    pub fn step(
        &mut self,
        model: &LmModel,
        user: Option<&[i32]>,
        moshi: Option<&[i32]>,
        text: Option<i32>,
    ) -> Option<Vec<i32>> {
        let prepared = self.prepare(user, moshi, text)?;
        let input = util::i32_tensor(
            prepared.input.clone(),
            [1, self.codebooks, 1],
            model.device(),
        );
        let (hidden, text_logits) = model.forward_codes(input, Some(&mut self.stream));
        self.consume(model, prepared, hidden, text_logits)
    }

    pub fn step_embeddings(&mut self, model: &LmModel, embeddings: Tensor<3>) -> Option<Vec<i32>> {
        let (user, moshi) = self.dummy_prompt();
        let prepared = loop {
            if let Some(prepared) = self.prepare(Some(&user), Some(&moshi), Some(self.text_pad)) {
                break prepared;
            }
        };
        let (hidden, text_logits) = model.forward_embeddings(embeddings, Some(&mut self.stream));
        self.consume(model, prepared, hidden, text_logits)
    }

    fn dummy_prompt(&self) -> (Vec<i32>, Vec<i32>) {
        let needed = self.codebooks - self.audio_per_stream - 1;
        let user = self.initial[1..1 + needed].to_vec();
        let moshi = self.initial[1 + needed..].to_vec();
        (user, moshi)
    }

    fn prepare(
        &mut self,
        user: Option<&[i32]>,
        moshi: Option<&[i32]>,
        text: Option<i32>,
    ) -> Option<Prepared> {
        let needed = self.codebooks - self.audio_per_stream - 1;
        if let Some(user) = user {
            for (q, token) in user.iter().take(needed).enumerate() {
                let k = self.audio_per_stream + 1 + q;
                let pos = (self.offset + self.delays[k]) % self.ct;
                self.set(k, pos, *token, true);
            }
        }
        if let Some(moshi) = moshi {
            for (q, token) in moshi.iter().take(needed).enumerate() {
                let k = 1 + q;
                let pos = (self.offset + self.delays[k]) % self.ct;
                self.set(k, pos, *token, true);
            }
        }
        if let Some(token) = text {
            let pos = (self.offset + self.delays[0]) % self.ct;
            self.set(0, pos, token, true);
        }
        for k in 0..self.codebooks {
            if self.offset <= self.delays[k] {
                let pos = self.offset % self.ct;
                self.set(k, pos, self.initial[k], true);
            }
        }
        if self.offset == 0 {
            for k in 0..self.codebooks {
                self.cache[k * self.ct] = self.initial[k];
            }
            self.offset += 1;
            return None;
        }
        let model_input = (self.offset + self.ct - 1) % self.ct;
        let target_pos = self.offset % self.ct;
        let input = (0..self.codebooks)
            .map(|k| self.get(k, model_input))
            .collect();
        let provided = (0..self.codebooks)
            .map(|k| self.flag(k, target_pos))
            .collect();
        let target = (0..self.codebooks)
            .map(|k| self.get(k, target_pos))
            .collect();
        Some(Prepared {
            input,
            provided,
            target,
            model_input,
            target_pos,
        })
    }

    fn consume(
        &mut self,
        model: &LmModel,
        prepared: Prepared,
        transformer_out: Tensor<3>,
        text_logits: Tensor<3>,
    ) -> Option<Vec<i32>> {
        let text_row = row_f32(text_logits);
        let sampled_text = self.sampler.text(&text_row);
        let next_text = if prepared.provided[0] {
            prepared.target[0]
        } else {
            sampled_text
        };
        let audio_target = &prepared.target[1..];
        let audio_provided = &prepared.provided[1..];
        let sampled_audio = self.sample_depth(
            model,
            next_text,
            transformer_out,
            audio_target,
            audio_provided,
        );

        for k in 0..self.codebooks {
            self.provided[k * self.ct + prepared.model_input] = false;
        }
        if !self.flag(0, prepared.target_pos) {
            self.cache[prepared.target_pos] = sampled_text;
        }
        for (cb, token) in sampled_audio.iter().enumerate() {
            let k = 1 + cb;
            if !self.flag(k, prepared.target_pos) {
                self.cache[k * self.ct + prepared.target_pos] = *token;
            }
        }
        if self.offset <= self.max_delay {
            self.offset += 1;
            return None;
        }
        let mut out = Vec::with_capacity(self.dep_q + 1);
        for k in 0..=self.dep_q {
            let index = (self.offset - self.max_delay + self.delays[k]) % self.ct;
            out.push(self.get(k, index));
        }
        self.offset += 1;
        Some(out)
    }

    fn sample_depth(
        &mut self,
        model: &LmModel,
        next_text: i32,
        transformer_out: Tensor<3>,
        audio_target: &[i32],
        audio_provided: &[bool],
    ) -> Vec<i32> {
        let mut prev = next_text;
        let mut bag = StreamingBag::new();
        let mut sampled = Vec::with_capacity(model.dep_q);
        for cb in 0..model.dep_q {
            let logits = model.depformer_step_logits(cb, prev, transformer_out.clone(), &mut bag);
            let next = self.sampler.audio(&row_f32(logits));
            sampled.push(next);
            let forced = audio_provided.get(cb).copied().unwrap_or(false);
            prev = if forced {
                audio_target.get(cb).copied().unwrap_or(next)
            } else {
                next
            };
        }
        sampled
    }

    fn get(&self, k: usize, t: usize) -> i32 {
        self.cache[k * self.ct + t]
    }

    fn flag(&self, k: usize, t: usize) -> bool {
        self.provided[k * self.ct + t]
    }

    fn set(&mut self, k: usize, t: usize, value: i32, provided: bool) {
        let index = k * self.ct + t;
        self.cache[index] = value;
        self.provided[index] = provided;
    }
}

fn row_f32(logits: Tensor<3>) -> Vec<f32> {
    let [b, t, c] = logits.dims();
    let flat = logits.reshape([b * t * c]);
    let values = util::tensor_to_f32_vec(flat);
    if t <= 1 {
        values
    } else {
        values[values.len() - c..].to_vec()
    }
}
