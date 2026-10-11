//! Causal streaming conv1d and conv-transpose1d.
//!
//! Padding matches `moshi.modules.conv` and `moshi.modules.streaming`:
//! causal left pad of `effective_kernel - stride`, extra right pad so the
//! last window is full, and a streaming overlap cache.

use burn::module::Param;
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, PadMode};
use burn::tensor::{Device, Tensor};

use crate::error::{Error, Result};
use crate::streaming::{self, StreamingBag};
use crate::weights::Weights;

pub struct CausalConv {
    pub id: u32,
    pub weight: Param<Tensor<3>>,
    pub bias: Option<Param<Tensor<1>>>,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub kernel: usize,
    pub causal: bool,
    pub pad_mode: PadMode,
}

pub struct CausalConvTr {
    pub id: u32,
    pub weight: Param<Tensor<3>>,
    pub bias: Option<Param<Tensor<1>>>,
    pub stride: usize,
    pub groups: usize,
    pub kernel: usize,
    pub causal: bool,
    pub trim_right_ratio: f32,
}

impl CausalConv {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &mut Weights,
        weight_key: &str,
        bias_key: &str,
        stride: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
        pad_mode: PadMode,
        device: &Device,
    ) -> Result<Self> {
        let weight = weights.param3(weight_key, device)?;
        let bias = if weights.contains(bias_key) {
            Some(weights.param1(bias_key, device)?)
        } else {
            None
        };
        let kernel = weight.val().dims()[2];
        Ok(Self {
            id: streaming::next_id(),
            weight,
            bias,
            stride,
            dilation,
            groups,
            kernel,
            causal,
            pad_mode,
        })
    }

    fn out_channels(&self) -> usize {
        self.weight.val().dims()[0]
    }

    fn effective_kernel(&self) -> usize {
        (self.kernel - 1) * self.dilation + 1
    }

    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        conv1d(
            x,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new([self.stride], [0], [self.dilation], self.groups),
        )
    }

    pub fn forward(&self, x: Tensor<3>, stream: Option<&mut StreamingBag>) -> Tensor<3> {
        let [b, _, t] = x.dims();
        if t == 0 && stream.is_none() {
            return Tensor::<3>::zeros([b, self.out_channels(), 0], (&x.device(), x.dtype()));
        }
        let eff = self.effective_kernel();
        let pad_total = eff.saturating_sub(self.stride);
        if let Some(bag) = stream {
            if !self.causal {
                return self.apply(x);
            }
            let state = bag.conv(self.id, pad_total);
            let mut x = x;
            if state.padding_to_add > 0 && x.dims()[2] > 0 {
                let pad = state.padding_to_add;
                state.padding_to_add = 0;
                x = pad_time(x, pad, 0, self.pad_mode);
            }
            return self.raw_streaming(x, state);
        }
        if self.causal {
            let extra = extra_padding(t, eff, self.stride, pad_total);
            let x = pad_time(x, pad_total, extra, self.pad_mode);
            self.apply(x)
        } else {
            self.apply(x)
        }
    }

    fn raw_streaming(
        &self,
        input: Tensor<3>,
        state: &mut crate::streaming::ConvStream,
    ) -> Tensor<3> {
        let mut input = input;
        if let Some(previous) = state.previous.take() {
            input = Tensor::cat(vec![previous, input], 2);
        }
        let t = input.dims()[2];
        let kernel = self.effective_kernel();
        let num_frames = if t < kernel {
            0
        } else {
            (t - kernel) / self.stride + 1
        };
        let offset = num_frames * self.stride;
        state.previous = if offset < t {
            Some(input.clone().narrow(2, offset, t - offset))
        } else {
            None
        };
        if num_frames == 0 {
            let b = input.dims()[0];
            return Tensor::<3>::zeros(
                [b, self.out_channels(), 0],
                (&input.device(), input.dtype()),
            );
        }
        let input_length = (num_frames - 1) * self.stride + kernel;
        self.apply(input.narrow(2, 0, input_length))
    }
}

impl CausalConvTr {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &mut Weights,
        weight_key: &str,
        bias_key: &str,
        stride: usize,
        groups: usize,
        causal: bool,
        trim_right_ratio: f32,
        device: &Device,
    ) -> Result<Self> {
        let weight = weights.param3(weight_key, device)?;
        let bias = if weights.contains(bias_key) {
            Some(weights.param1(bias_key, device)?)
        } else {
            None
        };
        let kernel = weight.val().dims()[2];
        if !(0.0..=1.0).contains(&trim_right_ratio) {
            return Err(Error::checkpoint(
                "trim_right_ratio must be between 0 and 1",
            ));
        }
        Ok(Self {
            id: streaming::next_id(),
            weight,
            bias,
            stride,
            groups,
            kernel,
            causal,
            trim_right_ratio,
        })
    }

    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        conv_transpose1d(
            x,
            self.weight.val(),
            self.bias.as_ref().map(|bias| bias.val()),
            ConvTransposeOptions::new([self.stride], [0], [0], [1], self.groups.max(1)),
        )
    }

    pub fn forward(&self, x: Tensor<3>, stream: Option<&mut StreamingBag>) -> Tensor<3> {
        let t = x.dims()[2];
        if t == 0 {
            let b = x.dims()[0];
            let out_ch = self.weight.val().dims()[1] * self.groups.max(1);
            return Tensor::<3>::zeros([b, out_ch, 0], (&x.device(), x.dtype()));
        }
        if let Some(bag) = stream {
            if !self.causal {
                return self.apply(x);
            }
            let state = bag.convtr(self.id);
            return self.raw_streaming(x, state);
        }
        let y = self.apply(x);
        let padding_total = self.kernel.saturating_sub(self.stride);
        let (left, right) = if self.causal {
            let padding_right = (padding_total as f32 * self.trim_right_ratio).ceil() as usize;
            (padding_total - padding_right, padding_right)
        } else {
            let padding_right = padding_total / 2;
            (padding_total - padding_right, padding_right)
        };
        let len = y.dims()[2];
        if left + right > len {
            return y;
        }
        y.narrow(2, left, len - left - right)
    }

    fn raw_streaming(&self, x: Tensor<3>, state: &mut crate::streaming::ConvTrStream) -> Tensor<3> {
        let mut out = self.apply(x);
        let mut ot = out.dims()[2];
        if let Some(partial) = state.partial.take() {
            let pt = partial.dims()[2];
            if pt > 0 && pt <= ot {
                let head = out.clone().narrow(2, 0, pt);
                let rest = out.clone().narrow(2, pt, ot - pt);
                let head = if let Some(bias) = &self.bias {
                    let channels = bias.val().dims()[0];
                    let bias = bias.val().reshape([1, channels, 1]);
                    head + (partial - bias)
                } else {
                    head + partial
                };
                out = Tensor::cat(vec![head, rest], 2);
                ot = out.dims()[2];
            }
        }
        let invalid = self.kernel.saturating_sub(self.stride);
        if invalid == 0 || ot < invalid {
            state.partial = None;
            return out;
        }
        state.partial = Some(out.clone().narrow(2, ot - invalid, invalid));
        out.narrow(2, 0, ot - invalid)
    }
}

fn pad_time(x: Tensor<3>, left: usize, right: usize, mode: PadMode) -> Tensor<3> {
    if left == 0 && right == 0 {
        return x;
    }
    x.pad([(left, right)], mode)
}

fn extra_padding(length: usize, kernel: usize, stride: usize, padding_total: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    let n_frames = (length as f64 - kernel as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal = (n_frames.ceil() as isize - 1).max(0) as usize * stride
        + kernel.saturating_sub(padding_total);
    ideal.saturating_sub(length)
}

/// Pointwise conv used by the quantizer projections. Kernel size is 1.
pub fn load_pointwise(weights: &mut Weights, key: &str, device: &Device) -> Result<CausalConv> {
    CausalConv::load(
        weights,
        key,
        "",
        1,
        1,
        1,
        true,
        PadMode::Constant(0.0),
        device,
    )
}
