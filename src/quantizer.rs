//! Split residual vector quantizer.
//!
//! Semantic and acoustic codebooks each encode the same latent. Inside one
//! group the quantizers are residual. Centroids are
//! `embedding_sum / clamp(cluster_usage, 1e-5)`. Nearest centroid breaks ties
//! toward the lowest index.

use burn::module::Param;
use burn::tensor::module::embedding;
use burn::tensor::{Device, Int, Tensor};

use crate::conv::CausalConv;
use crate::error::Result;
use crate::weights::Weights;

struct Codebook {
    embedding_sum: Param<Tensor<2>>,
    cluster_usage: Param<Tensor<1>>,
    epsilon: f32,
}

impl Codebook {
    fn embedding(&self) -> Tensor<2> {
        let usage = self
            .cluster_usage
            .val()
            .clamp_min(self.epsilon)
            .reshape([self.cluster_usage.val().dims()[0], 1]);
        self.embedding_sum.val() / usage
    }

    fn encode(&self, x: Tensor<3>) -> Tensor<2, Int> {
        let embed = self.embedding();
        let bins = embed.dims()[0];
        let [b, t, d] = x.dims();
        let x2 = x.clone().powf_scalar(2.0).sum_dim(2);
        let e2 = embed
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .reshape([1, 1, bins]);
        let dots = x
            .reshape([b * t, d])
            .matmul(embed.transpose())
            .reshape([b, t, bins]);
        let dist = x2 + e2 - dots * 2.0;
        dist.argmin(2).squeeze_dim::<2>(2)
    }

    fn decode(&self, codes: Tensor<2, Int>) -> Tensor<3> {
        let bins = self.embedding_sum.val().dims()[0] as i32;
        let codes = codes.clamp(0, (bins - 1).max(0));
        embedding(self.embedding(), codes)
    }
}

struct Rvq {
    input_proj: CausalConv,
    output_proj: CausalConv,
    layers: Vec<Codebook>,
}

impl Rvq {
    fn encode(&self, x: Tensor<3>, n_q: usize) -> Tensor<3, Int> {
        let [b, _, t] = x.dims();
        let n_q = n_q.min(self.layers.len());
        if t == 0 || n_q == 0 {
            return Tensor::<3, Int>::zeros([b, n_q, 0], &x.device());
        }
        let projected = self.input_proj.forward(x, None);
        let mut residual = projected.swap_dims(1, 2);
        let mut codes = Vec::with_capacity(n_q);
        for layer in self.layers.iter().take(n_q) {
            let ids = layer.encode(residual.clone());
            let quantized = layer.decode(ids.clone());
            residual = residual - quantized;
            codes.push(ids.unsqueeze_dim::<3>(1));
        }
        Tensor::cat(codes, 1)
    }

    fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let [b, k, t] = codes.dims();
        if t == 0 || k == 0 {
            let out_ch = self.output_proj.out_channels_public();
            return Tensor::<3>::zeros([b, out_ch, 0], &codes.device());
        }
        let mut acc: Option<Tensor<3>> = None;
        let limit = k.min(self.layers.len());
        for (index, layer) in self.layers.iter().take(limit).enumerate() {
            let ids = codes.clone().narrow(1, index, 1).squeeze_dim::<2>(1);
            let quantized = layer.decode(ids);
            acc = Some(match acc {
                None => quantized,
                Some(prev) => prev + quantized,
            });
        }
        let acc = acc
            .unwrap_or_else(|| Tensor::<3>::zeros([b, t, 1], &codes.device()))
            .swap_dims(1, 2);
        self.output_proj.forward(acc, None)
    }
}

/// Semantic codebooks plus acoustic codebooks.
pub struct SplitRvq {
    first: Rvq,
    rest: Rvq,
    pub n_q_semantic: usize,
    pub active: usize,
}

impl SplitRvq {
    pub fn from_weights(
        weights: &mut Weights,
        n_q_semantic: usize,
        n_q_acoustic: usize,
        active: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            first: load_rvq(weights, "quantizer.rvq_first", n_q_semantic, device)?,
            rest: load_rvq(weights, "quantizer.rvq_rest", n_q_acoustic, device)?,
            n_q_semantic,
            active,
        })
    }

    fn counts(&self) -> (usize, usize) {
        let semantic = self.n_q_semantic.min(self.active);
        let acoustic = self.active.saturating_sub(semantic);
        (semantic, acoustic)
    }

    pub fn encode(&self, x: Tensor<3>) -> Tensor<3, Int> {
        let (semantic, acoustic) = self.counts();
        let first = self.first.encode(x.clone(), semantic);
        let rest = self.rest.encode(x, acoustic);
        if rest.dims()[1] == 0 {
            first
        } else if first.dims()[1] == 0 {
            rest
        } else {
            Tensor::cat(vec![first, rest], 1)
        }
    }

    pub fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let k = codes.dims()[1];
        let semantic = self.n_q_semantic.min(k);
        let mut acc = None;
        if semantic > 0 {
            acc = Some(self.first.decode(codes.clone().narrow(1, 0, semantic)));
        }
        if k > semantic {
            let rest = self
                .rest
                .decode(codes.clone().narrow(1, semantic, k - semantic));
            acc = Some(match acc {
                None => rest,
                Some(first) => first + rest,
            });
        }
        acc.unwrap_or_else(|| {
            let [b, _, t] = codes.dims();
            Tensor::<3>::zeros([b, 1, t], &codes.device())
        })
    }
}

fn load_rvq(weights: &mut Weights, prefix: &str, n_q: usize, device: &Device) -> Result<Rvq> {
    let input_proj =
        crate::conv::load_pointwise(weights, &format!("{prefix}.input_proj.weight"), device)?;
    let output_proj =
        crate::conv::load_pointwise(weights, &format!("{prefix}.output_proj.weight"), device)?;
    let mut layers = Vec::with_capacity(n_q);
    for index in 0..n_q {
        let stem = format!("{prefix}.vq.layers.{index}._codebook");
        layers.push(Codebook {
            embedding_sum: weights.param2(&format!("{stem}.embedding_sum"), device)?,
            cluster_usage: weights.param1(&format!("{stem}.cluster_usage"), device)?,
            epsilon: 1e-5,
        });
    }
    Ok(Rvq {
        input_proj,
        output_proj,
        layers,
    })
}

impl CausalConv {
    pub fn out_channels_public(&self) -> usize {
        self.weight.val().dims()[0]
    }
}
