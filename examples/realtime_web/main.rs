//! Browser page and the OpenAI / xAI realtime WebSocket, one process, one port.
//!
//! ```text
//! cargo run --example realtime_web --features realtime-server -- --port 8080 --weights ./checkpoints
//! ```
//!
//! Open `http://127.0.0.1:8080/`. The socket is `/v1/realtime`. The page is
//! `index.html` in this directory, embedded with `include_str!`. Protocol
//! handling is [`personaplex_burn::realtime::serve_connection`], reached from
//! [`serve_with_page`](personaplex_burn::realtime::serve_with_page).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use personaplex_burn::realtime::{serve_with_page, Bridge, PersonaEngine};
use personaplex_burn::{inference_device, DtypePolicy, PersonaPlex, WeightSource};
use tokio::net::TcpListener;

const PAGE: &str = include_str!("index.html");

#[derive(Parser)]
#[command(
    name = "realtime_web",
    about = "PersonaPlex realtime WebSocket plus a browser page on the same port"
)]
struct Args {
    /// Checkpoint directory. Without this, the gated Hugging Face repo is used.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Bind address. Use 127.0.0.1 so the page is a secure context for the mic.
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    /// TCP port. Page at `http://<bind>:<port>/`, socket at `/v1/realtime`.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Persona used when the page sends an empty prompt.
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
    eprintln!("http://{addr}/");
    eprintln!("ws://{addr}/v1/realtime");
    eprintln!("the microphone works on localhost, or on HTTPS");
    loop {
        let (stream, _) = listener
            .accept()
            .await
            .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
        let engine = PersonaEngine::new(model.clone(), voices.clone(), persona.clone());
        let bridge = Bridge::new(engine).with_defaults(persona.clone(), "");
        tokio::spawn(async move {
            if let Err(err) = serve_with_page(stream, PAGE, bridge).await {
                eprintln!("connection: {err}");
            }
        });
    }
}
