//! Streaming state for causal convolutions and attention.
//!
//! The model weights stay immutable. A [`StreamingBag`] holds the per-step
//! caches and is owned by the session, so the same weights can serve batch
//! and streaming callers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use burn::tensor::{Device, Tensor};

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// Unique id for a module that keeps streaming state.
pub fn next_id() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Left-pad remainder and overlap cache for a causal convolution.
pub struct ConvStream {
    pub padding_to_add: usize,
    pub previous: Option<Tensor<3>>,
}

/// Overlap tail of a causal transposed convolution.
pub struct ConvTrStream {
    pub partial: Option<Tensor<3>>,
}

/// Ring KV cache plus the query position.
pub struct AttnStream {
    pub k: Tensor<4>,
    pub v: Tensor<4>,
    pub end_offset: usize,
    pub offset: usize,
}

/// Per-layer offset used by depth-transformer weights.
pub struct LayerStream {
    pub offset: usize,
}

/// All streaming caches for one encode or decode pass.
#[derive(Default)]
pub struct StreamingBag {
    conv: HashMap<u32, ConvStream>,
    convtr: HashMap<u32, ConvTrStream>,
    attn: HashMap<u32, AttnStream>,
    layer: HashMap<u32, LayerStream>,
}

impl StreamingBag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.conv.clear();
        self.convtr.clear();
        self.attn.clear();
        self.layer.clear();
    }

    pub fn conv(&mut self, id: u32, padding_total: usize) -> &mut ConvStream {
        self.conv.entry(id).or_insert(ConvStream {
            padding_to_add: padding_total,
            previous: None,
        })
    }

    pub fn convtr(&mut self, id: u32) -> &mut ConvTrStream {
        self.convtr
            .entry(id)
            .or_insert(ConvTrStream { partial: None })
    }

    pub fn layer(&mut self, id: u32) -> &mut LayerStream {
        self.layer.entry(id).or_insert(LayerStream { offset: 0 })
    }

    pub fn attn_mut(&mut self, id: u32) -> Option<&mut AttnStream> {
        self.attn.get_mut(&id)
    }

    pub fn insert_attn(&mut self, id: u32, state: AttnStream) {
        self.attn.insert(id, state);
    }
}

/// Reborrow a streaming bag across a loop of layers.
pub fn reborrow<'a>(stream: &'a mut Option<&mut StreamingBag>) -> Option<&'a mut StreamingBag> {
    stream.as_mut().map(|s| &mut **s)
}

pub fn zeros(shape: [usize; 3], device: &Device) -> Tensor<3> {
    Tensor::<3>::zeros(shape, device)
}
