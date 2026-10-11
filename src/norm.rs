//! Normalizations and activations used by Mimi and the language model.

use burn::tensor::activation::{self, gelu, silu};
use burn::tensor::{DType, Tensor};

use crate::util::RowChunks;

/// LayerNorm over the last dimension, computed in f32. Epsilon matches PyTorch (`1e-5`).
pub struct LayerNorm {
    pub weight: burn::module::Param<Tensor<1>>,
    pub bias: burn::module::Param<Tensor<1>>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let dim = x.dims()[2];
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let y = centered / (var + self.eps).sqrt();
        let w = self.weight.val().reshape([1, 1, dim]);
        let b = self.bias.val().reshape([1, 1, dim]);
        (y * w + b).cast(dtype)
    }
}

/// RMSNorm. `rms_norm_f32` always reduces in f32 with epsilon `1e-8`.
pub struct RmsNorm {
    pub alpha: burn::module::Param<Tensor<1>>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let dim = x.dims()[2];
        let var = x.clone().powf_scalar(2.0).mean_dim(2) + self.eps;
        let alpha = self.alpha.val().reshape([1, 1, dim]);
        (x * (alpha * var.sqrt().recip())).cast(dtype)
    }
}

pub enum Norm {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

impl Norm {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Layer(n) => n.forward(x),
            Self::Rms(n) => n.forward(x),
        }
    }
}

/// SiLU gate: `silu(a) * b` after a fused input projection.
pub struct Gating {
    pub linear_in: RowChunks,
    pub linear_out: RowChunks,
    pub hidden: usize,
}

impl Gating {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self.linear_in.matmul(x);
        let [b, t, _] = y.dims();
        let h = self.hidden;
        let y = y.reshape([b, t, 2, h]);
        let left = y.clone().narrow(2, 0, 1).squeeze_dim::<3>(2);
        let right = y.narrow(2, 1, 1).squeeze_dim::<3>(2);
        self.linear_out.matmul(silu(left) * right)
    }
}

/// ELU with alpha 1, matching `nn.ELU`.
pub fn elu(x: Tensor<3>) -> Tensor<3> {
    activation::elu(x, 1.0)
}

pub fn gelu_exact(x: Tensor<3>) -> Tensor<3> {
    gelu(x)
}

/// Learned residual multiplier, channel-last.
pub fn layer_scale(x: Tensor<3>, scale: Tensor<1>) -> Tensor<3> {
    let c = scale.dims()[0];
    x * scale.reshape([1, 1, c])
}
