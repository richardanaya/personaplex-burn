//! OpenAI Realtime and xAI Voice WebSocket protocol, over [`crate::session::Session`].
//!
//! The library stays transport-free. This module is compiled only with the
//! `realtime-server` feature. See `docs/realtime.md` for the mapping.

mod bridge;
mod codec;
mod server;

pub use bridge::{Activity, Bridge, RealtimeEngine};
pub use server::serve_connection;

#[cfg(test)]
mod tests;
