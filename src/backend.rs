//! Where inference runs.
//!
//! The library, the TUI, the realtime server, and the web example all call
//! [`inference_device`], which is Burn's wgpu backend (`Device::wgpu`). That
//! runtime picks Vulkan, Metal, DirectX 12, OpenGL, or WebGPU from the machine.
//! Flex (`Device::flex`) is the pure-Rust CPU backend and is only for parity
//! tests. It is not a default feature.
//!
//! # WebGPU buffer limits
//!
//! The WebGPU spec guarantees `maxStorageBufferBindingSize` of 128 MiB and
//! `maxBufferSize` of 256 MiB. CubeCL sizes allocations from the tighter of
//! those two. A 7B PersonaPlex checkpoint has individual matrices larger than
//! 128 MiB in f16 (the text embedding is about 250 MiB; each temporal gating
//! `linear_in` is about 168 MiB). [`crate::util::RowChunks`] splits those
//! matrices so every buffer stays at or under [`MAX_WEIGHT_BUFFER_BYTES`]
//! (64 MiB), which also fits f32 weights when the device has no shader-f16.
//!
//! # dtype
//!
//! WGPU advertises f32 everywhere and f16 only when the adapter enables
//! `shader-f16`. [`DtypePolicy::Auto`] keeps f16 weights when
//! `device.supports_dtype(F16)` is true and otherwise loads f32. Attention
//! and norms reduce in f32 either way. bf16 checkpoints are converted on load;
//! Burn's wgpu backend does not run bf16 kernels.

use burn::tensor::{DType, Device};

/// Largest weight buffer uploaded to the device.
///
/// Half of WebGPU's guaranteed 128 MiB storage-buffer binding, so allocator
/// alignment cannot push a chunk over the limit.
pub const MAX_WEIGHT_BUFFER_BYTES: usize = 64 << 20;

/// Resolved floating type stored on device for weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DtypePolicy {
    /// Force f32. Parity tests use this.
    F32,
    /// f16 when the device can do f16 arithmetic, otherwise f32.
    F16,
    /// Same selection as [`Self::F16`]. This is the inference default.
    #[default]
    Auto,
}

impl DtypePolicy {
    /// dtype to store weights in, given what this device can actually compute.
    pub fn resolve(self, device: &Device) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 | Self::Auto => {
                if device.supports_dtype(DType::F16) {
                    DType::F16
                } else {
                    DType::F32
                }
            }
        }
    }
}

/// Device used by every inference entry point (session callers, TUI, server).
///
/// Burn selects the graphics API. Override the adapter with
/// `CUBECL_WGPU_DEFAULT_DEVICE` (for example `DiscreteGpu(0)`).
#[cfg(feature = "wgpu")]
pub fn inference_device() -> Device {
    Device::wgpu(Default::default())
}

/// Short description of the inference backend for logs and the README's claims.
pub fn backend_name() -> &'static str {
    if cfg!(feature = "wgpu") {
        "wgpu (Vulkan, Metal, DirectX 12, OpenGL, or WebGPU)"
    } else {
        "flex (CPU, tests only)"
    }
}
