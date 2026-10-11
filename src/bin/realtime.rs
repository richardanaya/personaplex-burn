//! OpenAI Realtime / xAI Voice compatible WebSocket server.
//!
//! Point a client at `ws://127.0.0.1:8080/v1/realtime`. The model is PersonaPlex
//! on `Device::wgpu`. See `docs/realtime.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use personaplex_burn::realtime::{serve_connection, Activity, Bridge, RealtimeEngine};
use personaplex_burn::{
    inference_device, DtypePolicy, Event, PersonaPlex, Session, SessionConfig, TextKind,
    VoicePrompt, WeightSource,
};
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(
    name = "personaplex-realtime",
    about = "PersonaPlex behind the OpenAI Realtime and xAI Voice WebSocket protocol"
)]
struct Args {
    /// Checkpoint directory. Without this, the gated Hugging Face repo is used.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Bind address.
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    /// TCP port. Clients connect to `ws://<bind>:<port>/v1/realtime`.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Persona used until a client sends `session.update` instructions.
    #[arg(long, default_value = "You are a wise teacher.")]
    persona: String,
    /// Directory of `{voice}.wav` or `{voice}.safetensors` presets.
    #[arg(long)]
    voices: Option<PathBuf>,
}

struct PersonaEngine {
    model: PersonaPlex,
    session: Option<Session>,
    voices: Option<PathBuf>,
    fallback_persona: String,
}

impl PersonaEngine {
    fn new(model: PersonaPlex, voices: Option<PathBuf>, persona: String) -> Self {
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

fn main() -> ExitCode {
    let args = Args::parse();
    let source = match args.weights {
        Some(path) => WeightSource::local(path),
        None => WeightSource::huggingface(),
    };
    let device = inference_device();
    eprintln!("loading on {device:?}");
    let model = match PersonaPlex::load(source, device, DtypePolicy::Auto) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(err) = runtime.block_on(listen(
        model,
        args.bind,
        args.port,
        args.persona,
        args.voices,
    )) {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn listen(
    model: PersonaPlex,
    bind: String,
    port: u16,
    persona: String,
    voices: Option<PathBuf>,
) -> personaplex_burn::Result<()> {
    let addr = format!("{bind}:{port}");
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|err| personaplex_burn::Error::session(format!("bind {addr}: {err}")))?;
    eprintln!("ws://{addr}/v1/realtime");
    loop {
        let (stream, _) = listener
            .accept()
            .await
            .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
        let engine = PersonaEngine::new(model.clone(), voices.clone(), persona.clone());
        let bridge = Bridge::new(engine).with_defaults(persona.clone(), "");
        tokio::spawn(async move {
            if let Err(err) = serve_connection(stream, bridge).await {
                eprintln!("connection: {err}");
            }
        });
    }
}
