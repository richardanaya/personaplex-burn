//! Errors for loading, streaming, and I/O.

/// How to unblock a gated Hugging Face download of `nvidia/personaplex-7b-v1`.
pub const GATED_HELP: &str = "\
PersonaPlex weights are gated under the NVIDIA Open Model License. \
Accept the license at https://huggingface.co/nvidia/personaplex-7b-v1 \
then set HF_TOKEN to an access token from that account. \
You can also load a local directory that already contains \
model.safetensors, tokenizer-e351c8d8-checkpoint125.safetensors, \
and tokenizer_spm_32k_3.model.";

/// Library error. Weight-gate failures include [`GATED_HELP`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The checkpoint is gated, missing a token, or the token was rejected.
    #[error("{message}\n{GATED_HELP}")]
    Gated {
        /// Hub or filesystem detail.
        message: String,
    },
    /// A weight tensor or config field did not match the architecture.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    /// Audio or text given to a session was the wrong shape or rate.
    #[error("session: {0}")]
    Session(String),
    /// Tokenizer model could not be read.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// Filesystem failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// WAV encode/decode failure.
    #[error("wav: {0}")]
    Wav(String),
    /// JSON fixture or sidecar failure.
    #[error("json: {0}")]
    Json(String),
    /// Hugging Face download failed for a reason other than gating.
    #[error("huggingface: {0}\nIf this is a 401/403, {GATED_HELP}")]
    Hub(String),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn checkpoint(msg: impl Into<String>) -> Self {
        Self::Checkpoint(msg.into())
    }

    pub fn session(msg: impl Into<String>) -> Self {
        Self::Session(msg.into())
    }

    pub fn tokenizer(msg: impl Into<String>) -> Self {
        Self::Tokenizer(msg.into())
    }

    pub fn wav(msg: impl Into<String>) -> Self {
        Self::Wav(msg.into())
    }

    pub fn gated(msg: impl Into<String>) -> Self {
        Self::Gated {
            message: msg.into(),
        }
    }

    /// True when a hub error looks like a missing license acceptance or token.
    pub fn is_gated(err: &dyn std::fmt::Display) -> bool {
        let text = err.to_string().to_ascii_lowercase();
        text.contains("401")
            || text.contains("403")
            || text.contains("gated")
            || text.contains("unauthorized")
            || text.contains("access denied")
            || text.contains("restricted")
    }
}
