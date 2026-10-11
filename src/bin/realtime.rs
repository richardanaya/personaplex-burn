//! OpenAI Realtime / xAI Voice compatible WebSocket server.
//!
//! Point a client at `ws://127.0.0.1:8080/v1/realtime`. The model is PersonaPlex
//! on `Device::wgpu`. See `docs/realtime.md`. The browser page is the
//! `realtime_web` example, which uses this same server code.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use personaplex_burn::realtime::{serve_connection, Bridge, PersonaEngine};
use personaplex_burn::{inference_device, DtypePolicy, PersonaPlex, WeightSource};
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
