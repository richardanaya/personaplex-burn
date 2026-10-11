//! PersonaPlex behind [`RealtimeEngine`](super::RealtimeEngine).
//!
//! The websocket binary and `examples/realtime_web` both use this. Session
//! setup stays here so neither one grows a second copy of the protocol.

use std::path::PathBuf;

use super::bridge::{Activity, RealtimeEngine};
use crate::{Event, PersonaPlex, Session, SessionConfig, TextKind, VoicePrompt};

/// One loaded checkpoint and the live [`Session`], if a client has configured one.
pub struct PersonaEngine {
    model: PersonaPlex,
    session: Option<Session>,
    voices: Option<PathBuf>,
    fallback_persona: String,
}

impl PersonaEngine {
    pub fn new(model: PersonaPlex, voices: Option<PathBuf>, persona: String) -> Self {
        Self {
            model,
            session: None,
            voices,
            fallback_persona: persona,
        }
    }
}

impl RealtimeEngine for PersonaEngine {
    fn sample_rate(&self) -> u32 {
        self.model.config().sample_rate()
    }

    fn apply_session(&mut self, instructions: &str, voice: &str) -> Result<(), String> {
        let instructions = if instructions.is_empty() {
            self.fallback_persona.as_str()
        } else {
            instructions
        };
        let prompt = resolve_voice(self.voices.as_ref(), voice);
        let mut cfg = SessionConfig::new(instructions).with_voice(prompt);
        cfg.emit_timing = false;
        self.session =
            Some(Session::start(self.model.clone(), cfg).map_err(|err| err.to_string())?);
        Ok(())
    }

    fn push_audio(&mut self, mono: &[f32], sample_rate: u32) -> Result<Vec<Activity>, String> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| "session is not configured".to_string())?;
        let events = session
            .push_audio(mono, sample_rate, 1)
            .map_err(|err| err.to_string())?;
        Ok(events.into_iter().filter_map(activity_from).collect())
    }

    fn interrupt(&mut self) -> Result<(), String> {
        if let Some(session) = self.session.as_mut() {
            session.interrupt();
        }
        Ok(())
    }
}

fn activity_from(event: Event) -> Option<Activity> {
    match event {
        Event::AudioOut { pcm, .. } => Some(Activity::Audio(pcm)),
        Event::Text { kind, text, .. } => {
            if kind != TextKind::InnerMonologue
                || text.is_empty()
                || text == "EPAD"
                || text == "PAD"
            {
                None
            } else {
                Some(Activity::Transcript(text))
            }
        }
        Event::Status(_) => None,
    }
}

fn resolve_voice(dir: Option<&PathBuf>, name: &str) -> VoicePrompt {
    if name.is_empty() || name.eq_ignore_ascii_case("none") {
        return VoicePrompt::None;
    }
    let direct = PathBuf::from(name);
    if direct.exists() {
        return VoicePrompt::from_path(direct);
    }
    let Some(dir) = dir else {
        eprintln!("voice preset `{name}` has no file; continuing without voice conditioning");
        return VoicePrompt::None;
    };
    let wav = dir.join(format!("{name}.wav"));
    if wav.exists() {
        return VoicePrompt::Wav(wav);
    }
    let embeddings = dir.join(format!("{name}.safetensors"));
    if embeddings.exists() {
        return VoicePrompt::Embeddings(embeddings);
    }
    eprintln!(
        "voice preset `{name}` was not in {}; continuing without voice conditioning",
        dir.display()
    );
    VoicePrompt::None
}
