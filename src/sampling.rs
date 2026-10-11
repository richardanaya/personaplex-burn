//! Token sampling.
//!
//! Greedy sampling is argmax. Stochastic sampling follows the reference
//! top-k exponential-argmax trick. The noise source is SplitMix64 so a seed
//! reproduces the same tokens on every platform. It is not PyTorch's RNG.

use crate::config::SamplingConfig;

/// Portable SplitMix64 used by the seeded sampler and `scripts/gen_fixtures.py`.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    pub fn next_uniform(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) + 1;
        u as f64 / 9_007_199_254_740_992.0
    }

    pub fn next_exponential(&mut self) -> f32 {
        (-self.next_uniform().ln()) as f32
    }
}

/// Sampler held by a session. `reset` restores the configured seed.
#[derive(Clone, Debug)]
pub struct Sampler {
    cfg: SamplingConfig,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Self {
        let rng = SplitMix64::new(cfg.seed);
        Self { cfg, rng }
    }

    pub fn reset(&mut self) {
        self.rng = SplitMix64::new(self.cfg.seed);
    }

    pub fn text(&mut self, logits: &[f32]) -> i32 {
        sample_logits(
            logits,
            self.cfg.sample,
            self.cfg.temperature_text,
            self.cfg.top_k_text,
            &mut self.rng,
        )
    }

    pub fn audio(&mut self, logits: &[f32]) -> i32 {
        sample_logits(
            logits,
            self.cfg.sample,
            self.cfg.temperature,
            self.cfg.top_k,
            &mut self.rng,
        )
    }
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in values.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    best_i
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in &mut exps {
            *v /= sum;
        }
    }
    exps
}

/// Top-k then `argmax(prob / exponential_noise)`.
///
/// `noise`, when set, is used instead of the RNG and must have one entry per
/// kept class (the top-k width, or the full vocabulary when `top_k` is 0).
pub fn sample_logits_with_noise(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    noise: &[f32],
) -> i32 {
    let probs = softmax(&logits.iter().map(|v| v / temperature).collect::<Vec<_>>());
    let k = if top_k == 0 {
        probs.len()
    } else {
        top_k.min(probs.len())
    };
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);
    let mut best_i = order[0];
    let mut best = f32::NEG_INFINITY;
    for (rank, &idx) in order.iter().enumerate() {
        let n = noise.get(rank).copied().unwrap_or(1.0);
        let score = probs[idx] / n.max(1e-20);
        if score > best {
            best = score;
            best_i = idx;
        }
    }
    best_i as i32
}

fn sample_logits(
    logits: &[f32],
    sample: bool,
    temperature: f32,
    top_k: usize,
    rng: &mut SplitMix64,
) -> i32 {
    if logits.is_empty() {
        return 0;
    }
    if !sample || temperature <= 0.0 {
        return argmax(logits) as i32;
    }
    let k = if top_k == 0 {
        logits.len()
    } else {
        top_k.min(logits.len())
    };
    let noise: Vec<f32> = (0..k).map(|_| rng.next_exponential()).collect();
    sample_logits_with_noise(logits, temperature, top_k, &noise)
}

/// Draw `n` SplitMix64 uniforms, matching the fixture script.
pub fn seed_uniforms(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_uniform()).collect()
}
