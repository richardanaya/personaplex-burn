//! Whole-file adapter. Reads a WAV, runs one [`Session`](personaplex_burn::Session),
//! and writes model-rate audio plus the inner-monologue text.
//!
//! Inference uses [`inference_device`](personaplex_burn::inference_device), Burn's
//! wgpu backend.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use personaplex_burn::audio::{self};
use personaplex_burn::{
    backend_name, inference_device, process_batch, DtypePolicy, PersonaPlex, SessionConfig,
    VoicePrompt, WeightSource,
};

#[derive(Parser)]
#[command(
    name = "personaplex-batch",
    about = "Run a WAV through PersonaPlex and write audio plus inner-monologue text"
)]
struct Args {
    /// Checkpoint directory (model.safetensors, the Mimi safetensors, and the sentencepiece model).
    /// Without this flag the gated Hugging Face repo is used and `HF_TOKEN` must be set.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Input WAV. Any rate and channel count.
    #[arg(long)]
    input: PathBuf,
    /// Output WAV at the model rate (24 kHz for the release checkpoint).
    #[arg(long)]
    output: PathBuf,
    /// Optional path for the decoded inner-monologue text.
    #[arg(long)]
    text: Option<PathBuf>,
    /// Role prompt. Wrapped as `<system> … <system>` when those tags are absent.
    #[arg(long, default_value = "")]
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
    eprintln!("backend: {} ({device:?})", backend_name());
    let model = PersonaPlex::load(source(args.weights), device, DtypePolicy::Auto)?;
    let mut cfg = SessionConfig::new(args.persona);
    if let Some(voice) = args.voice {
        cfg = cfg.with_voice(VoicePrompt::from_path(voice));
    }
    let (samples, rate, channels) = audio::read_wav(&args.input)?;
    let batch = process_batch(&model, cfg, &samples, rate, channels)?;
    audio::write_wav(&args.output, &batch.pcm, batch.sample_rate)?;
    if let Some(path) = &args.text {
        std::fs::write(path, &batch.text)?;
    } else {
        println!("{}", batch.text);
    }
    eprintln!(
        "wrote {} samples at {} Hz",
        batch.pcm.len(),
        batch.sample_rate
    );
    Ok(())
}

fn source(weights: Option<PathBuf>) -> WeightSource {
    match weights {
        Some(path) => WeightSource::local(path),
        None => WeightSource::huggingface(),
    }
}
