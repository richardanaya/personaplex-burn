//! NVIDIA PersonaPlex on Burn.
//!
//! The primary interface is [`session::Session`]: audio frames in, audio, text,
//! and status events out. Transport (microphone, WebSocket, files) lives in
//! adapters, not in this crate.

pub mod audio;
pub mod backend;
pub mod batch;
pub mod config;
pub mod conv;
pub mod error;
pub mod lm;
pub mod mimi;
pub mod norm;
pub mod quantizer;
pub mod sampling;
pub mod seanet;
pub mod session;
pub mod streaming;
pub mod tokenizer;
pub mod transformer;
pub mod util;
pub mod weights;

pub use backend::{backend_name, DtypePolicy, MAX_WEIGHT_BUFFER_BYTES};
pub use batch::{process_batch, BatchOutput};
pub use config::{ModelConfig, SamplingConfig, SessionConfig, VoicePrompt};
pub use error::{Error, Result};
pub use session::{Event, Session, Status, TextKind};
pub use weights::{PersonaPlex, WeightSource};

#[cfg(feature = "wgpu")]
pub use backend::inference_device;

#[cfg(test)]
mod tests;
