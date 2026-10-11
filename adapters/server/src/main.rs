//! One WebSocket session per connection, plus the static page at `/`.
//!
//! Text frames are JSON control (`start`, `interrupt`, `reset`, `mute`, `format`).
//! Binary frames are little-endian f32 PCM at the rate from the last `format`
//! or `start` message. The server answers with JSON events and binary f32 PCM
//! at the model rate. Inference uses Burn's wgpu device. The page does not run
//! the 7B model in the browser.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use personaplex_burn::{
    backend_name, inference_device, DtypePolicy, Event, PersonaPlex, Session, SessionConfig,
    Status, VoicePrompt, WeightSource,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

#[derive(Parser)]
#[command(
    name = "personaplex-server",
    about = "PersonaPlex realtime WebSocket server"
)]
struct Args {
    /// Checkpoint directory. Without this, the gated Hugging Face repo is used.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Bind address.
    #[arg(long, default_value = "127.0.0.1:8787")]
    bind: String,
    /// Default role prompt. A `start` message can replace it per connection.
    #[arg(long, default_value = "You are a wise teacher.")]
    persona: String,
    /// Voice clip (`.wav`) or a safetensors embedding cache, shared by every connection.
    #[arg(long)]
    voice: Option<PathBuf>,
    /// Directory containing `index.html`.
    #[arg(long, default_value = "realtime_web")]
    web: PathBuf,
}

struct State {
    model: PersonaPlex,
    persona: String,
    voice: VoicePrompt,
    html: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match runtime(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn runtime(args: Args) -> personaplex_burn::Result<()> {
    let html_path = args.web.join("index.html");
    let html = std::fs::read_to_string(&html_path).map_err(|err| {
        personaplex_burn::Error::session(format!("read {}: {err}", html_path.display()))
    })?;
    let device = inference_device();
    eprintln!("loading on {} ({device:?})", backend_name());
    let model = PersonaPlex::load(
        match args.weights {
            Some(path) => WeightSource::local(path),
            None => WeightSource::huggingface(),
        },
        device,
        DtypePolicy::Auto,
    )?;
    let voice = match args.voice {
        Some(path) => VoicePrompt::from_path(path),
        None => VoicePrompt::None,
    };
    let state = Arc::new(State {
        model,
        persona: args.persona,
        voice,
        html,
    });
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    rt.block_on(serve(args.bind, state))
}

async fn serve(bind: String, state: Arc<State>) -> personaplex_burn::Result<()> {
    let listener = TcpListener::bind(&bind)
        .await
        .map_err(|err| personaplex_burn::Error::session(format!("bind {bind}: {err}")))?;
    eprintln!("http://{bind}/   websocket ws://{bind}/");
    loop {
        let (stream, _) = listener
            .accept()
            .await
            .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(err) = dispatch(stream, state).await {
                eprintln!("connection: {err}");
            }
        });
    }
}

async fn dispatch(stream: TcpStream, state: Arc<State>) -> personaplex_burn::Result<()> {
    if websocket_upgrade(&stream).await {
        let socket = tokio_tungstenite::accept_async(stream)
            .await
            .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
        connection(socket, state).await
    } else {
        serve_page(stream, &state.html).await
    }
}

async fn websocket_upgrade(stream: &TcpStream) -> bool {
    let mut buf = [0u8; 2048];
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        let n = stream.peek(&mut buf).await.unwrap_or(0);
        if n > 0 && (has_header_end(&buf[..n]) || Instant::now() > deadline) {
            let head = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
            return head.contains("upgrade: websocket");
        }
        if Instant::now() > deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn has_header_end(bytes: &[u8]) -> bool {
    bytes.windows(4).any(|window| window == b"\r\n\r\n")
}

async fn serve_page(mut stream: TcpStream, html: &str) -> personaplex_burn::Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream
        .read(&mut buf)
        .await
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = req.split_whitespace().nth(1).unwrap_or("/");
    let (status, body, ctype) = if path == "/" || path == "/index.html" {
        ("200 OK", html, "text/html; charset=utf-8")
    } else {
        ("404 Not Found", "not found", "text/plain; charset=utf-8")
    };
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(resp.as_bytes())
        .await
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    Ok(())
}

async fn connection(
    mut socket: WebSocketStream<TcpStream>,
    state: Arc<State>,
) -> personaplex_burn::Result<()> {
    let mut session: Option<Session> = None;
    let mut rate = 16_000u32;
    let mut channels = 1usize;
    while let Some(message) = socket.next().await {
        let message = message.map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
        match message {
            Message::Text(text) => {
                handle_text(
                    &mut socket,
                    &state,
                    &mut session,
                    &mut rate,
                    &mut channels,
                    text.as_str(),
                )
                .await?;
            }
            Message::Binary(bytes) => {
                let Some(session) = session.as_mut() else {
                    send_json(
                        &mut socket,
                        r#"{"type":"error","message":"send start first"}"#,
                    )
                    .await?;
                    continue;
                };
                if !bytes.len().is_multiple_of(4) {
                    send_json(
                        &mut socket,
                        r#"{"type":"error","message":"binary frames are f32le"}"#,
                    )
                    .await?;
                    continue;
                }
                let samples: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
                let events =
                    tokio::task::block_in_place(|| session.push_audio(&samples, rate, channels))?;
                send_events(&mut socket, &events).await?;
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    Ok(())
}

async fn handle_text(
    socket: &mut WebSocketStream<TcpStream>,
    state: &State,
    session: &mut Option<Session>,
    rate: &mut u32,
    channels: &mut usize,
    text: &str,
) -> personaplex_burn::Result<()> {
    let value: serde_json::Value = serde_json::from_str(text).unwrap_or(serde_json::Value::Null);
    let op = value.get("op").and_then(|op| op.as_str()).unwrap_or("");
    if let Some(sample_rate) = value.get("sample_rate").and_then(|v| v.as_u64()) {
        *rate = sample_rate as u32;
    }
    if let Some(count) = value.get("channels").and_then(|v| v.as_u64()) {
        *channels = count.max(1) as usize;
    }
    match op {
        "start" => {
            let persona = value
                .get("persona")
                .and_then(|v| v.as_str())
                .unwrap_or(&state.persona);
            let mut cfg = SessionConfig::new(persona).with_voice(state.voice.clone());
            if let Some(seed) = value.get("seed").and_then(|v| v.as_u64()) {
                cfg.sampling.seed = seed;
            }
            let started = tokio::task::block_in_place(|| Session::start(state.model.clone(), cfg))?;
            let events = started.prompt_events().to_vec();
            *session = Some(started);
            send_json(
                socket,
                &format!(
                    r#"{{"type":"audio_format","sample_rate":{}}}"#,
                    state.model.config().sample_rate()
                ),
            )
            .await?;
            send_events(socket, &events).await?;
        }
        "interrupt" => {
            if let Some(session) = session.as_mut() {
                let event = session.interrupt();
                send_events(socket, &[event]).await?;
            }
        }
        "reset" => {
            if let Some(session) = session.as_mut() {
                let events = tokio::task::block_in_place(|| session.reset())?;
                send_events(socket, &events).await?;
            }
        }
        "mute" => {
            if let Some(session) = session.as_mut() {
                let muted = value
                    .get("muted")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(!session.is_muted());
                let event = session.set_muted(muted);
                send_events(socket, &[event]).await?;
            }
        }
        "format" => {
            send_json(
                socket,
                &format!(r#"{{"type":"format","sample_rate":{rate},"channels":{channels}}}"#),
            )
            .await?;
        }
        _ => {
            send_json(socket, r#"{"type":"error","message":"unknown op"}"#).await?;
        }
    }
    Ok(())
}

async fn send_events(
    socket: &mut WebSocketStream<TcpStream>,
    events: &[Event],
) -> personaplex_burn::Result<()> {
    for event in events {
        match event {
            Event::AudioOut {
                pcm,
                sample_rate: _,
            } => {
                let mut bytes = Vec::with_capacity(pcm.len() * 4);
                for sample in pcm {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                socket
                    .send(Message::Binary(bytes.into()))
                    .await
                    .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
            }
            Event::Text { text, token, .. } => {
                let payload = serde_json::json!({
                    "type": "text",
                    "text": text,
                    "token": token,
                });
                send_json(socket, &payload.to_string()).await?;
            }
            Event::Status(status) => {
                let payload = status_json(status);
                send_json(socket, &payload).await?;
            }
        }
    }
    Ok(())
}

fn status_json(status: &Status) -> String {
    match status {
        Status::Frame { index, elapsed_ms } => serde_json::json!({
            "type": "status",
            "name": "frame",
            "index": index,
            "elapsed_ms": elapsed_ms,
        })
        .to_string(),
        other => serde_json::json!({
            "type": "status",
            "name": format!("{other:?}").to_ascii_lowercase(),
        })
        .to_string(),
    }
}

async fn send_json(
    socket: &mut WebSocketStream<TcpStream>,
    payload: &str,
) -> personaplex_burn::Result<()> {
    socket
        .send(Message::Text(payload.to_string().into()))
        .await
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))
}
