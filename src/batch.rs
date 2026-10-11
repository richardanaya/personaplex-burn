//! Whole-file inference on top of [`crate::session::Session`].
//!
//! This is the batch shape of the same API the microphone and websocket
//! adapters call one frame at a time. It does not open files itself; the
//! `adapters/batch` binary does that.

use serde::Serialize;

use crate::audio::{self};
use crate::config::SessionConfig;
use crate::error::Result;
use crate::session::{Event, Session};
use crate::weights::PersonaPlex;

/// Model-rate audio and the decoded inner-monologue text for one clip.
#[derive(Debug, Clone, Serialize)]
pub struct BatchOutput {
    /// Mono PCM at [`Self::sample_rate`], trimmed or padded to the input duration.
    pub pcm: Vec<f32>,
    /// Model sample rate. 24 kHz for the release checkpoint.
    pub sample_rate: u32,
    /// Inner-monologue pieces concatenated, including `EPAD` / `PAD` markers.
    pub text: String,
    /// Every event the session emitted, including prompt status.
    pub events: Vec<Event>,
}

/// Run one clip through a fresh session.
///
/// `samples` are interleaved. Any rate and channel count is accepted. Output
/// audio is at the model rate and matches the resampled input length.
pub fn process_batch(
    model: &PersonaPlex,
    cfg: SessionConfig,
    samples: &[f32],
    sample_rate: u32,
    channels: usize,
) -> Result<BatchOutput> {
    let mono = audio::downmix(samples, channels)?;
    let target = audio::linear_resample(&mono, sample_rate, model.config().sample_rate()).len();
    let mut session = Session::start(model.clone(), cfg)?;
    let mut events = session.prompt_events().to_vec();
    events.extend(session.push_audio(samples, sample_rate, channels)?);
    events.extend(session.finish()?);
    let mut pcm = Vec::new();
    let mut text = String::new();
    for event in &events {
        match event {
            Event::AudioOut { pcm: frame, .. } => pcm.extend_from_slice(frame),
            Event::Text { text: piece, .. } => {
                if !text.is_empty() && !piece.starts_with(' ') {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            Event::Status(_) => {}
        }
    }
    if pcm.len() > target {
        pcm.truncate(target);
    } else {
        pcm.resize(target, 0.0);
    }
    let rate = session.sample_rate();
    Ok(BatchOutput {
        pcm,
        sample_rate: rate,
        text,
        events,
    })
}
