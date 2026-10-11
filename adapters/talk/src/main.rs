//! Microphone and speaker adapter. Audio frames go into [`Session::push_audio`](personaplex_burn::Session::push_audio);
//! decoded frames go to the speaker. The terminal shows both turns, the inner
//! monologue, levels, and frame time.
//!
//! Keys: `q` quit, `m` mute, `r` reset. Inference uses Burn's wgpu device.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use crossterm::event::{self, Event, KeyCode};
use personaplex_burn::audio;
use personaplex_burn::{
    backend_name, inference_device, DtypePolicy, Event as ModelEvent, PersonaPlex, Session,
    SessionConfig, Status, VoicePrompt, WeightSource,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Wrap};
use ratatui::DefaultTerminal;

#[derive(Parser)]
#[command(
    name = "personaplex-talk",
    about = "Talk to PersonaPlex from the microphone"
)]
struct Args {
    /// Checkpoint directory. Without this, the gated Hugging Face repo is used.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Role prompt.
    #[arg(long, default_value = "You are a wise teacher.")]
    persona: String,
    /// Voice clip (`.wav`) or a safetensors embedding cache.
    #[arg(long)]
    voice: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> personaplex_burn::Result<()> {
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
    let mut cfg = SessionConfig::new(args.persona);
    if let Some(voice) = args.voice {
        cfg = cfg.with_voice(VoicePrompt::from_path(voice));
    }
    let mut session = Session::start(model, cfg)?;
    let host = cpal::default_host();
    let input = host
        .default_input_device()
        .ok_or_else(|| personaplex_burn::Error::session("no input audio device"))?;
    let output = host
        .default_output_device()
        .ok_or_else(|| personaplex_burn::Error::session("no output audio device"))?;
    let input_config = input
        .default_input_config()
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    let output_config = output
        .default_output_config()
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    let in_rate = input_config.sample_rate().0;
    let in_channels = input_config.channels() as usize;
    let out_rate = output_config.sample_rate().0;
    let out_channels = output_config.channels() as usize;
    let captured = Arc::new(Mutex::new(Vec::<f32>::new()));
    let playback = Arc::new(Mutex::new(Vec::<f32>::new()));
    let in_level = Arc::new(AtomicU32::new(0));
    let out_level = Arc::new(AtomicU32::new(0));
    let alive = Arc::new(AtomicBool::new(true));
    let input_stream = build_input(
        &input,
        &input_config.config(),
        input_config.sample_format(),
        Arc::clone(&captured),
        Arc::clone(&in_level),
    )?;
    let output_stream = build_output(
        &output,
        &output_config.config(),
        output_config.sample_format(),
        out_channels,
        Arc::clone(&playback),
        Arc::clone(&out_level),
    )?;
    input_stream
        .play()
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    output_stream
        .play()
        .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;

    let mut terminal = ratatui::init();
    let mut text = String::new();
    let mut status = "listening".to_string();
    let mut frame_ms = 0.0f32;
    let ui = Ui {
        in_level,
        out_level,
        in_rate,
        out_rate,
        in_channels,
    };
    let mut ctx = LoopCtx {
        session: &mut session,
        captured: &captured,
        playback: &playback,
        alive: &alive,
        in_rate,
        in_channels,
        out_rate,
        text: &mut text,
        status: &mut status,
        frame_ms: &mut frame_ms,
        ui: &ui,
    };
    let result = event_loop(&mut terminal, &mut ctx);
    ratatui::restore();
    result
}

struct LoopCtx<'a> {
    session: &'a mut Session,
    captured: &'a Mutex<Vec<f32>>,
    playback: &'a Mutex<Vec<f32>>,
    alive: &'a AtomicBool,
    in_rate: u32,
    in_channels: usize,
    out_rate: u32,
    text: &'a mut String,
    status: &'a mut String,
    frame_ms: &'a mut f32,
    ui: &'a Ui,
}

struct Ui {
    in_level: Arc<AtomicU32>,
    out_level: Arc<AtomicU32>,
    in_rate: u32,
    out_rate: u32,
    in_channels: usize,
}

fn event_loop(
    terminal: &mut DefaultTerminal,
    ctx: &mut LoopCtx<'_>,
) -> personaplex_burn::Result<()> {
    while ctx.alive.load(Ordering::Relaxed) {
        if event::poll(Duration::from_millis(20))
            .map_err(|err| personaplex_burn::Error::session(err.to_string()))?
        {
            let key =
                event::read().map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
            if let Event::Key(key) = key {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        ctx.alive.store(false, Ordering::Relaxed);
                    }
                    KeyCode::Char('m') => {
                        let event = ctx.session.set_muted(!ctx.session.is_muted());
                        apply_event(
                            event,
                            ctx.text,
                            ctx.status,
                            ctx.frame_ms,
                            ctx.playback,
                            ctx.out_rate,
                        );
                    }
                    KeyCode::Char('r') => {
                        ctx.playback.lock().expect("playback").clear();
                        for event in ctx.session.reset()? {
                            apply_event(
                                event,
                                ctx.text,
                                ctx.status,
                                ctx.frame_ms,
                                ctx.playback,
                                ctx.out_rate,
                            );
                        }
                        ctx.text.clear();
                    }
                    _ => {}
                }
            }
        }
        let samples = {
            let mut buf = ctx.captured.lock().expect("capture");
            std::mem::take(&mut *buf)
        };
        if !samples.is_empty() {
            for event in ctx
                .session
                .push_audio(&samples, ctx.in_rate, ctx.in_channels)?
            {
                apply_event(
                    event,
                    ctx.text,
                    ctx.status,
                    ctx.frame_ms,
                    ctx.playback,
                    ctx.out_rate,
                );
            }
        }
        let muted = ctx.session.is_muted();
        terminal
            .draw(|frame| draw(frame, ctx.ui, ctx.text, ctx.status, ctx.frame_ms, muted))
            .map_err(|err| personaplex_burn::Error::session(err.to_string()))?;
    }
    Ok(())
}

fn apply_event(
    event: ModelEvent,
    text: &mut String,
    status: &mut String,
    frame_ms: &mut f32,
    playback: &Mutex<Vec<f32>>,
    out_rate: u32,
) {
    match event {
        ModelEvent::AudioOut { pcm, sample_rate } => {
            let pcm = audio::linear_resample(&pcm, sample_rate, out_rate);
            playback.lock().expect("playback").extend(pcm);
        }
        ModelEvent::Text { text: piece, .. } => {
            if !text.is_empty() && !piece.starts_with(' ') {
                text.push(' ');
            }
            text.push_str(&piece);
            if text.len() > 4000 {
                text.drain(..text.len() - 4000);
            }
        }
        ModelEvent::Status(state) => match state {
            Status::Frame { elapsed_ms, .. } => *frame_ms = elapsed_ms,
            other => *status = format!("{other:?}").to_ascii_lowercase(),
        },
    }
}

fn draw(
    frame: &mut ratatui::Frame<'_>,
    ui: &Ui,
    text: &str,
    status: &str,
    frame_ms: &f32,
    muted: bool,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(4),
        ])
        .split(frame.area());
    let header = Paragraph::new(Line::from(vec![
        Span::raw(format!("backend {}  ", backend_name())),
        Span::styled(status, Style::default().fg(Color::Cyan)),
        Span::raw(format!(
            "  in {} Hz x {}  out {} Hz  frame {frame_ms:.1} ms  {}",
            ui.in_rate,
            ui.in_channels,
            ui.out_rate,
            if muted { "MUTED" } else { "live" }
        )),
    ]))
    .block(Block::default().borders(Borders::ALL).title("personaplex"));
    frame.render_widget(header, rows[0]);
    frame.render_widget(level_gauge("mic", &ui.in_level, Color::Green), rows[1]);
    frame.render_widget(level_gauge("model", &ui.out_level, Color::Magenta), rows[2]);
    let body = Paragraph::new(text).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("inner monologue  q quit  m mute  r reset"),
    );
    frame.render_widget(body, rows[3]);
}

fn level_gauge(title: &'static str, bits: &AtomicU32, color: Color) -> Gauge<'static> {
    let rms = f32::from_bits(bits.load(Ordering::Relaxed)).clamp(0.0, 1.0);
    Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .gauge_style(Style::default().fg(color))
        .ratio(f64::from(rms))
}

fn build_input(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    captured: Arc<Mutex<Vec<f32>>>,
    level: Arc<AtomicU32>,
) -> personaplex_burn::Result<Stream> {
    let stream = match format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| push_input(data, &captured, &level),
            log_input,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                let samples: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                push_input(&samples, &captured, &level);
            },
            log_input,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _| {
                let samples: Vec<f32> = data
                    .iter()
                    .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                    .collect();
                push_input(&samples, &captured, &level);
            },
            log_input,
            None,
        ),
        other => {
            return Err(personaplex_burn::Error::session(format!(
                "unsupported input sample format {other:?}"
            )))
        }
    };
    stream.map_err(|err| personaplex_burn::Error::session(err.to_string()))
}

fn push_input(data: &[f32], captured: &Mutex<Vec<f32>>, level: &AtomicU32) {
    level.store(rms(data).to_bits(), Ordering::Relaxed);
    captured.lock().expect("capture").extend_from_slice(data);
}

fn build_output(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    channels: usize,
    playback: Arc<Mutex<Vec<f32>>>,
    level: Arc<AtomicU32>,
) -> personaplex_burn::Result<Stream> {
    let stream = match format {
        SampleFormat::F32 => device.build_output_stream(
            config,
            move |data: &mut [f32], _| fill_output(data, channels, &playback, &level),
            log_output,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            config,
            move |data: &mut [i16], _| {
                let mut floats = vec![0.0f32; data.len()];
                fill_output(&mut floats, channels, &playback, &level);
                for (dst, src) in data.iter_mut().zip(floats) {
                    *dst = (src.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                }
            },
            log_output,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            config,
            move |data: &mut [u16], _| {
                let mut floats = vec![0.0f32; data.len()];
                fill_output(&mut floats, channels, &playback, &level);
                for (dst, src) in data.iter_mut().zip(floats) {
                    *dst = ((src.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
                }
            },
            log_output,
            None,
        ),
        other => {
            return Err(personaplex_burn::Error::session(format!(
                "unsupported output sample format {other:?}"
            )))
        }
    };
    stream.map_err(|err| personaplex_burn::Error::session(err.to_string()))
}

fn fill_output(data: &mut [f32], channels: usize, playback: &Mutex<Vec<f32>>, level: &AtomicU32) {
    let mut queue = playback.lock().expect("playback");
    let frames = data.len() / channels.max(1);
    let mut mono = Vec::with_capacity(frames);
    for _ in 0..frames {
        mono.push(if queue.is_empty() {
            0.0
        } else {
            queue.remove(0)
        });
    }
    level.store(rms(&mono).to_bits(), Ordering::Relaxed);
    for (frame, sample) in mono.into_iter().enumerate() {
        for channel in 0..channels {
            if let Some(slot) = data.get_mut(frame * channels + channel) {
                *slot = sample;
            }
        }
    }
}

fn log_input(error: cpal::StreamError) {
    eprintln!("input: {error}");
}

fn log_output(error: cpal::StreamError) {
    eprintln!("output: {error}");
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy: f32 = samples.iter().map(|sample| sample * sample).sum();
    (energy / samples.len() as f32).sqrt().clamp(0.0, 1.0)
}
