//! One-shot load of the local 7B checkpoint and a single silent frame.
fn main() {
    let started = std::time::Instant::now();
    eprintln!("loading checkpoints/");
    let model = personaplex_burn::PersonaPlex::load(
        personaplex_burn::WeightSource::local("checkpoints"),
        personaplex_burn::inference_device(),
        personaplex_burn::DtypePolicy::Auto,
    )
    .expect("load");
    eprintln!(
        "loaded dtype={:?} in {:.1}s",
        model.weight_dtype(),
        started.elapsed().as_secs_f32()
    );
    let mut cfg = personaplex_burn::SessionConfig::new("You are a wise teacher.");
    cfg.silence_frames = 0;
    let mut session = personaplex_burn::Session::start(model, cfg).expect("session");
    eprintln!("session ready in {:.1}s", started.elapsed().as_secs_f32());
    for event in session.prompt_events() {
        match event {
            personaplex_burn::Event::Text { text, token, .. } => {
                eprintln!("prompt text token={token} {text:?}");
            }
            personaplex_burn::Event::Status(status) => eprintln!("prompt status {status:?}"),
            personaplex_burn::Event::AudioOut { pcm, .. } => {
                eprintln!("prompt audio {} samples", pcm.len());
            }
        }
    }
    let frame = vec![0.0f32; session.frame_size()];
    let rate = session.sample_rate();
    let events = session.push_audio(&frame, rate, 1).expect("frame");
    eprintln!(
        "one frame ({} samples @ {rate} Hz) in {:.1}s",
        frame.len(),
        started.elapsed().as_secs_f32()
    );
    for event in events {
        match event {
            personaplex_burn::Event::Text { text, token, .. } => {
                eprintln!("text token={token} {text:?}");
            }
            personaplex_burn::Event::Status(status) => eprintln!("status {status:?}"),
            personaplex_burn::Event::AudioOut { pcm, .. } => {
                let peak = pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()));
                eprintln!("audio {} samples peak={peak:.4}", pcm.len());
            }
        }
    }
}
