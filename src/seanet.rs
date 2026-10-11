//! SEANet encoder and decoder.
//!
//! Module order matches the reference `nn.Sequential`, including parameter-free
//! ELU slots, so safetensors indices line up (`model.0`, `model.1`, `model.3`, ...).

use burn::tensor::ops::PadMode;
use burn::tensor::{Device, Tensor};

use crate::config::MimiConfig;
use crate::conv::{CausalConv, CausalConvTr};
use crate::error::Result;
use crate::norm::elu;
use crate::streaming::{reborrow, StreamingBag};
use crate::weights::Weights;

enum SeaOp {
    Conv(Box<CausalConv>),
    ConvTr(Box<CausalConvTr>),
    Elu,
    Resnet(Box<ResnetBlock>),
}

struct ResnetBlock {
    conv1: CausalConv,
    conv2: CausalConv,
}

impl ResnetBlock {
    fn forward(&self, x: Tensor<3>, mut stream: Option<&mut StreamingBag>) -> Tensor<3> {
        let y = elu(x.clone());
        let y = self.conv1.forward(y, reborrow(&mut stream));
        let y = elu(y);
        let y = self.conv2.forward(y, reborrow(&mut stream));
        x + y
    }
}

pub struct Seanet {
    ops: Vec<SeaOp>,
}

impl Seanet {
    pub fn encoder(cfg: &MimiConfig, weights: &mut Weights, device: &Device) -> Result<Self> {
        let mut ops = Vec::new();
        let mut idx = 0usize;
        ops.push(SeaOp::Conv(Box::new(load_conv(
            weights,
            &format!("encoder.model.{idx}.conv.conv"),
            cfg.n_filters,
            1,
            cfg.kernel_size,
            device,
        )?)));
        idx += 1;
        let mut mult = 1usize;
        let ratios: Vec<usize> = cfg.ratios.iter().rev().copied().collect();
        for ratio in ratios {
            for _ in 0..cfg.n_residual_layers {
                let dim = mult * cfg.n_filters;
                ops.push(SeaOp::Resnet(Box::new(load_resnet(
                    weights,
                    &format!("encoder.model.{idx}"),
                    dim,
                    cfg.compress,
                    cfg.residual_kernel_size,
                    device,
                )?)));
                idx += 1;
            }
            ops.push(SeaOp::Elu);
            idx += 1;
            ops.push(SeaOp::Conv(Box::new(load_conv(
                weights,
                &format!("encoder.model.{idx}.conv.conv"),
                mult * cfg.n_filters * 2,
                ratio,
                ratio * 2,
                device,
            )?)));
            idx += 1;
            mult *= 2;
        }
        ops.push(SeaOp::Elu);
        idx += 1;
        ops.push(SeaOp::Conv(Box::new(load_conv(
            weights,
            &format!("encoder.model.{idx}.conv.conv"),
            cfg.dimension,
            1,
            cfg.last_kernel_size,
            device,
        )?)));
        Ok(Self { ops })
    }

    pub fn decoder(cfg: &MimiConfig, weights: &mut Weights, device: &Device) -> Result<Self> {
        let mut ops = Vec::new();
        let mut idx = 0usize;
        let mut mult = 1usize << cfg.ratios.len();
        ops.push(SeaOp::Conv(Box::new(load_conv(
            weights,
            &format!("decoder.model.{idx}.conv.conv"),
            mult * cfg.n_filters,
            1,
            cfg.kernel_size,
            device,
        )?)));
        idx += 1;
        for ratio in &cfg.ratios {
            ops.push(SeaOp::Elu);
            idx += 1;
            ops.push(SeaOp::ConvTr(Box::new(load_convtr(
                weights,
                &format!("decoder.model.{idx}.convtr.convtr"),
                *ratio,
                *ratio * 2,
                device,
            )?)));
            idx += 1;
            for _ in 0..cfg.n_residual_layers {
                let dim = mult * cfg.n_filters / 2;
                ops.push(SeaOp::Resnet(Box::new(load_resnet(
                    weights,
                    &format!("decoder.model.{idx}"),
                    dim,
                    cfg.compress,
                    cfg.residual_kernel_size,
                    device,
                )?)));
                idx += 1;
            }
            mult /= 2;
        }
        ops.push(SeaOp::Elu);
        idx += 1;
        ops.push(SeaOp::Conv(Box::new(load_conv(
            weights,
            &format!("decoder.model.{idx}.conv.conv"),
            cfg.channels,
            1,
            cfg.last_kernel_size,
            device,
        )?)));
        Ok(Self { ops })
    }

    pub fn forward(&self, x: Tensor<3>, mut stream: Option<&mut StreamingBag>) -> Tensor<3> {
        let mut y = x;
        for op in &self.ops {
            y = match op {
                SeaOp::Conv(conv) => conv.forward(y, reborrow(&mut stream)),
                SeaOp::ConvTr(conv) => conv.forward(y, reborrow(&mut stream)),
                SeaOp::Elu => elu(y),
                SeaOp::Resnet(block) => block.forward(y, reborrow(&mut stream)),
            };
        }
        y
    }
}

fn load_conv(
    weights: &mut Weights,
    prefix: &str,
    _out_channels: usize,
    stride: usize,
    _kernel: usize,
    device: &Device,
) -> Result<CausalConv> {
    CausalConv::load(
        weights,
        &format!("{prefix}.weight"),
        &format!("{prefix}.bias"),
        stride,
        1,
        1,
        true,
        PadMode::Constant(0.0),
        device,
    )
}

fn load_convtr(
    weights: &mut Weights,
    prefix: &str,
    stride: usize,
    _kernel: usize,
    device: &Device,
) -> Result<CausalConvTr> {
    CausalConvTr::load(
        weights,
        &format!("{prefix}.weight"),
        &format!("{prefix}.bias"),
        stride,
        1,
        true,
        1.0,
        device,
    )
}

fn load_resnet(
    weights: &mut Weights,
    prefix: &str,
    _dim: usize,
    _compress: usize,
    _kernel: usize,
    device: &Device,
) -> Result<ResnetBlock> {
    let conv1 = CausalConv::load(
        weights,
        &format!("{prefix}.block.1.conv.conv.weight"),
        &format!("{prefix}.block.1.conv.conv.bias"),
        1,
        1,
        1,
        true,
        PadMode::Constant(0.0),
        device,
    )?;
    let conv2 = CausalConv::load(
        weights,
        &format!("{prefix}.block.3.conv.conv.weight"),
        &format!("{prefix}.block.3.conv.conv.bias"),
        1,
        1,
        1,
        true,
        PadMode::Constant(0.0),
        device,
    )?;
    Ok(ResnetBlock { conv1, conv2 })
}
