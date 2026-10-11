//! Mimi encoder / decoder: SEANet, transformer, conv resample, split RVQ.
//!
//! `encode` returns `[batch, codebooks, time]`. `decode` does not trim; the
//! session trims a batch to the input duration the way `offline.py` does.

use burn::tensor::ops::PadMode;
use burn::tensor::{DType, Device, Int, Tensor};

use crate::config::MimiConfig;
use crate::conv::{CausalConv, CausalConvTr};
use crate::error::Result;
use crate::quantizer::SplitRvq;
use crate::seanet::Seanet;
use crate::streaming::{reborrow, StreamingBag};
use crate::transformer::{Transformer, TransformerSpec};
use crate::weights::Weights;

pub struct Mimi {
    encoder: Seanet,
    decoder: Seanet,
    encoder_tr: Transformer,
    decoder_tr: Transformer,
    quantizer: SplitRvq,
    downsample: CausalConv,
    upsample: CausalConvTr,
    dtype: DType,
}

impl Mimi {
    pub fn from_weights(cfg: &MimiConfig, weights: &mut Weights, device: &Device) -> Result<Self> {
        let spec = TransformerSpec {
            layers: cfg.transformer_layers,
            dim: cfg.transformer_dim,
            heads: cfg.transformer_heads,
            dim_feedforward: cfg.transformer_ff,
            context: Some(cfg.transformer_context),
            rope: true,
            max_period: cfg.max_period,
            gating: false,
            rms: false,
            weights_per_step: 0,
            layer_scale: cfg.layer_scale.is_some(),
        };
        let stride = cfg.resample_stride().max(1);
        let acoustic = cfg.quantizer_n_q.saturating_sub(cfg.n_q_semantic);
        Ok(Self {
            encoder: Seanet::encoder(cfg, weights, device)?,
            decoder: Seanet::decoder(cfg, weights, device)?,
            encoder_tr: Transformer::load(
                weights,
                "encoder_transformer.transformer",
                &spec,
                device,
            )?,
            decoder_tr: Transformer::load(
                weights,
                "decoder_transformer.transformer",
                &spec,
                device,
            )?,
            quantizer: SplitRvq::from_weights(
                weights,
                cfg.n_q_semantic,
                acoustic,
                cfg.active_codebooks,
                device,
            )?,
            downsample: CausalConv::load(
                weights,
                "downsample.conv.conv.conv.weight",
                "downsample.conv.conv.conv.bias",
                stride,
                1,
                1,
                true,
                PadMode::Edge,
                device,
            )?,
            upsample: CausalConvTr::load(
                weights,
                "upsample.convtr.convtr.convtr.weight",
                "upsample.convtr.convtr.convtr.bias",
                stride,
                cfg.dimension,
                true,
                1.0,
                device,
            )?,
            dtype: weights.dtype(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn encode(&self, x: Tensor<3>, mut stream: Option<&mut StreamingBag>) -> Tensor<3, Int> {
        let latent = self.latent(x, reborrow(&mut stream));
        self.quantizer.encode(latent)
    }

    pub fn encode_latent(&self, x: Tensor<3>, stream: Option<&mut StreamingBag>) -> Tensor<3> {
        self.latent(x, stream)
    }

    pub fn decode(
        &self,
        codes: Tensor<3, Int>,
        mut stream: Option<&mut StreamingBag>,
    ) -> Tensor<3> {
        let mut emb = self.quantizer.decode(codes);
        emb = self.upsample.forward(emb, reborrow(&mut stream));
        emb = conv_transformer(&self.decoder_tr, emb, reborrow(&mut stream));
        self.decoder.forward(emb, reborrow(&mut stream))
    }

    fn latent(&self, x: Tensor<3>, mut stream: Option<&mut StreamingBag>) -> Tensor<3> {
        let x = x.cast(self.dtype);
        let emb = self.encoder.forward(x, reborrow(&mut stream));
        let emb = conv_transformer(&self.encoder_tr, emb, reborrow(&mut stream));
        self.downsample.forward(emb, reborrow(&mut stream))
    }
}

fn conv_transformer(
    transformer: &Transformer,
    x: Tensor<3>,
    stream: Option<&mut StreamingBag>,
) -> Tensor<3> {
    let x = x.swap_dims(1, 2);
    transformer.forward(x, stream).swap_dims(1, 2)
}
