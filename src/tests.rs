//! Parity against `fixtures/` plus the transport-free batch session.
//!
//! Numeric tests run on Burn Flex (`--features flex`). They do not need a GPU,
//! a Hugging Face token, or the gated weights.

use std::path::PathBuf;

#[cfg(feature = "flex")]
use crate::audio::{self, apply_lufs_gain};
use crate::backend::{DtypePolicy, MAX_WEIGHT_BUFFER_BYTES};
use crate::config::{LmConfig, MimiConfig, SamplingConfig};
#[cfg(feature = "flex")]
use crate::config::{ModelConfig, SessionConfig};
use crate::error::{Error, GATED_HELP};
use crate::sampling::{sample_logits_with_noise, seed_uniforms};
use crate::util;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn assert_close(got: &[f32], expect: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        expect.len(),
        "len {} vs {}",
        got.len(),
        expect.len()
    );
    let mut max = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(expect).enumerate() {
        let err = (a - b).abs();
        if err > max {
            max = err;
            at = i;
        }
    }
    assert!(
        max <= tol,
        "max abs {max} at {at} (got {}, expect {}), tol {tol}",
        got[at],
        expect[at]
    );
}

#[test]
fn weight_chunks_respect_webgpu_buffer_limit() {
    // Text embedding of the 7B model, f16: (32001, 4096).
    let text_rows = util::rows_per_buffer(4096, 2);
    assert!(text_rows * 4096 * 2 <= MAX_WEIGHT_BUFFER_BYTES);
    assert!(32001_usize.div_ceil(text_rows) > 1);
    // Temporal gating linear_in, f16: (21504, 4096) ≈ 168 MiB unsplit.
    let gate_rows = util::rows_per_buffer(4096, 2);
    assert!(gate_rows * 4096 * 2 <= MAX_WEIGHT_BUFFER_BYTES);
    assert!(21504_usize.div_ceil(gate_rows) > 1);
    // Same matrices in f32, used when the adapter has no shader-f16.
    let text_f32 = util::rows_per_buffer(4096, 4);
    assert!(text_f32 * 4096 * 4 <= MAX_WEIGHT_BUFFER_BYTES);
    assert!(text_f32 * 4096 * 4 <= 128 << 20);
    let policy = DtypePolicy::Auto;
    assert_eq!(policy, DtypePolicy::default());
}

#[test]
fn gated_help_names_the_license() {
    assert!(GATED_HELP.contains("https://huggingface.co/nvidia/personaplex-7b-v1"));
    assert!(GATED_HELP.contains("HF_TOKEN"));
    assert!(Error::is_gated(&"401 Unauthorized"));
    assert!(Error::is_gated(&"gated repo"));
    assert!(!Error::is_gated(&"connection reset"));
}

#[test]
fn system_wrap_matches_offline_py() {
    assert_eq!(
        crate::session::wrap_system("hello"),
        "<system> hello <system>"
    );
    assert_eq!(
        crate::session::wrap_system("<system> hello <system>"),
        "<system> hello <system>"
    );
    assert!(crate::session::wrap_system("  ").is_empty());
}

#[test]
#[allow(clippy::excessive_precision)]
fn seeded_sampler_matches_splitmix_reference() {
    let uniforms = seed_uniforms(42, 4);
    let expect = [
        0.7415648787718234,
        0.15991039287692022,
        0.2786011302551388,
        0.34419071652363764,
    ];
    for (got, exp) in uniforms.iter().zip(expect) {
        assert!((got - exp).abs() < 1e-12, "{got} vs {exp}");
    }
    let logits = [0.2, 1.5, -0.4, 0.7, 2.2, -1.0, 0.3, 1.1, 0.0, -0.2];
    let noise = [
        2.1744072437286377,
        0.4852575957775116,
        0.7569032907485962,
        1.0019547939300537,
    ];
    // sample.json stores the second draw's noise, which selects token 1
    // (`sampled_repeat`). The first draw's token is 4 with different noise.
    assert_eq!(sample_logits_with_noise(&logits, 0.7, 4, &noise), 1);
    let mut sampler = crate::sampling::Sampler::new(SamplingConfig::greedy());
    assert_eq!(sampler.text(&logits), 4);
}

/// Allocates a tensor and runs a matmul on [`crate::inference_device`].
/// Ignored in CI: this VM has no GPU. A software Vulkan adapter such as
/// lavapipe can still execute it. Run with `--ignored`.
#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a GPU or a software wgpu adapter"]
fn wgpu_device_can_allocate() {
    use burn::tensor::{DType, Tensor, TensorData};

    let device = crate::inference_device();
    let f16 = device.supports_dtype(DType::F16);
    eprintln!("wgpu shader-f16: {f16}");
    let values = vec![1.0f32, 2.0, 3.0, 4.0];
    let data = TensorData::new(values.clone(), [2, 2]);
    let left = Tensor::<2>::from_data(data.clone(), &device);
    let right = Tensor::<2>::from_data(data, &device);
    let product = left.matmul(right);
    let back = product.try_into_vec_as::<f32>().expect("wgpu readback");
    assert_eq!(back, vec![7.0, 10.0, 15.0, 22.0]);
    if f16 {
        let left = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, 2]), &device)
            .cast(DType::F16);
        let right =
            Tensor::<2>::from_data(TensorData::new(values, [2, 2]), &device).cast(DType::F16);
        let product = left.matmul(right).cast(DType::F32);
        let back = product.try_into_vec_as::<f32>().expect("f16 readback");
        assert_eq!(back, vec![7.0, 10.0, 15.0, 22.0]);
    }
}

/// Tiny Mimi and LM fixtures on the wgpu device. Same numeric checks as the
/// Flex parity tests, ignored so CI does not need a graphics adapter.
#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a GPU or a software wgpu adapter"]
fn wgpu_tiny_mimi_and_lm_match_fixtures() {
    use crate::lm::LmModel;
    use crate::mimi::Mimi;
    use crate::weights::Weights;
    use burn::tensor::DType;

    let device = crate::inference_device();
    let mimi_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("mimi.json")).expect("mimi fixture"))
            .expect("mimi json");
    let mut mimi_weights =
        Weights::from_file(fixture("mimi.safetensors"), DType::F32).expect("mimi weights");
    let mimi = Mimi::from_weights(&tiny_mimi_cfg(), &mut mimi_weights, &device).expect("mimi");
    let input = json_f32s(&mimi_json["input"]);
    let wav = util::f32_tensor(input, [1, 1, 32], &device);
    let latent = flat_f32(mimi.encode_latent(wav.clone(), None));
    assert_close(&latent, &json_f32s(&mimi_json["latent"]), 1e-3);
    let codes = mimi.encode(wav, None);
    let code_len: usize = codes.dims().iter().product();
    let codes_i32 = codes
        .reshape([code_len])
        .try_into_vec_as::<i32>()
        .expect("codes");
    assert_eq!(codes_i32, json_i32s(&mimi_json["codes"]));
    let decoded = flat_f32(mimi.decode(util::i32_tensor(codes_i32, [1, 8, 4], &device), None));
    assert_close(&decoded, &json_f32s(&mimi_json["decoded"]), 1e-3);
    eprintln!("wgpu tiny mimi matched the fixture");

    let lm_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("lm.json")).expect("lm fixture"))
            .expect("lm json");
    let mut lm_weights =
        Weights::from_file(fixture("lm.safetensors"), DType::F32).expect("lm weights");
    let lm = LmModel::from_weights(&tiny_lm_cfg(), &mut lm_weights, &device).expect("lm");
    let sequence = util::i32_tensor(json_i32s(&lm_json["codes"]), [1, 17, 4], &device);
    let (hidden, text) = lm.forward_codes(sequence.clone(), None);
    assert_close(
        &flat_f32(hidden.clone()),
        &json_f32s(&lm_json["transformer_out"]),
        2e-3,
    );
    assert_close(&flat_f32(text), &json_f32s(&lm_json["text_logits"]), 2e-3);
    let dep = lm.depformer_logits_training(sequence, hidden);
    assert_close(&flat_f32(dep), &json_f32s(&lm_json["dep_logits"]), 2e-3);
    eprintln!("wgpu tiny lm matched the fixture");
}

#[cfg(feature = "wgpu")]
fn flat_f32<const D: usize>(tensor: burn::tensor::Tensor<D>) -> Vec<f32> {
    let n: usize = tensor.dims().iter().product();
    util::tensor_to_f32_vec(tensor.reshape([n]))
}

#[cfg(feature = "wgpu")]
fn json_f32s(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .flat_map(|item| {
            if item.is_array() {
                json_f32s(item)
            } else {
                vec![item.as_f64().expect("float") as f32]
            }
        })
        .collect()
}

#[cfg(feature = "wgpu")]
fn json_i32s(value: &serde_json::Value) -> Vec<i32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .flat_map(|item| {
            if item.is_array() {
                json_i32s(item)
            } else {
                vec![item.as_i64().expect("int") as i32]
            }
        })
        .collect()
}

#[cfg(feature = "wgpu")]
fn tiny_mimi_cfg() -> MimiConfig {
    MimiConfig {
        sample_rate: 16,
        frame_rate: 2.0,
        channels: 1,
        dimension: 8,
        n_filters: 4,
        ratios: vec![2, 2],
        kernel_size: 3,
        residual_kernel_size: 3,
        last_kernel_size: 3,
        n_residual_layers: 1,
        dilation_base: 2,
        compress: 2,
        causal: true,
        transformer_layers: 1,
        transformer_heads: 2,
        transformer_dim: 8,
        transformer_ff: 16,
        transformer_context: 16,
        layer_scale: Some(0.01),
        max_period: 10_000.0,
        quantizer_dim: 4,
        quantizer_bins: 8,
        quantizer_n_q: 8,
        n_q_semantic: 1,
        active_codebooks: 8,
    }
}

#[cfg(feature = "wgpu")]
fn tiny_lm_cfg() -> LmConfig {
    LmConfig {
        delays: vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        n_q: 16,
        dep_q: 16,
        card: 32,
        text_card: 48,
        dim: 32,
        num_heads: 4,
        hidden_scale: 4.0,
        num_layers: 1,
        context: 64,
        max_period: 10_000.0,
        depformer_dim: 16,
        depformer_num_heads: 4,
        depformer_num_layers: 1,
        depformer_dim_feedforward: 32,
        depformer_weights_per_step: true,
        depformer_multi_linear: true,
        existing_text_padding_id: 3,
        audio_tokens_per_stream: 8,
        silence_tokens: vec![0; 8],
        sine_tokens: vec![0; 8],
    }
}

#[cfg(feature = "flex")]
mod flex_parity {
    use super::*;
    use crate::batch::process_batch;
    use crate::lm::{LmModel, LmState};
    use crate::mimi::Mimi;
    use crate::session::{Event, Session, Status};
    use crate::streaming::StreamingBag;
    use crate::tokenizer::Tokenizer;
    use crate::weights::{PersonaPlex, Weights};
    use burn::tensor::{DType, Device, Int, Tensor};

    fn device() -> Device {
        Device::flex()
    }

    fn f32s(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|item| {
                if item.is_array() {
                    f32s(item)
                } else {
                    vec![item.as_f64().unwrap() as f32]
                }
            })
            .collect()
    }

    fn i32s(value: &serde_json::Value) -> Vec<i32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|item| {
                if item.is_array() {
                    i32s(item)
                } else {
                    vec![item.as_i64().unwrap() as i32]
                }
            })
            .collect()
    }

    fn tensor_f32<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
        let n: usize = tensor.dims().iter().product();
        util::tensor_to_f32_vec(tensor.reshape([n]))
    }

    fn tensor_i32<const D: usize>(tensor: Tensor<D, Int>) -> Vec<i32> {
        let n: usize = tensor.dims().iter().product();
        tensor
            .reshape([n])
            .try_into_vec_as::<i32>()
            .expect("int tensor")
    }

    fn mimi_cfg() -> MimiConfig {
        MimiConfig {
            sample_rate: 16,
            frame_rate: 2.0,
            channels: 1,
            dimension: 8,
            n_filters: 4,
            ratios: vec![2, 2],
            kernel_size: 3,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            n_residual_layers: 1,
            dilation_base: 2,
            compress: 2,
            causal: true,
            transformer_layers: 1,
            transformer_heads: 2,
            transformer_dim: 8,
            transformer_ff: 16,
            transformer_context: 16,
            layer_scale: Some(0.01),
            max_period: 10_000.0,
            quantizer_dim: 4,
            quantizer_bins: 8,
            quantizer_n_q: 8,
            n_q_semantic: 1,
            active_codebooks: 8,
        }
    }

    fn lm_cfg() -> LmConfig {
        LmConfig {
            delays: vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            n_q: 16,
            dep_q: 16,
            card: 32,
            text_card: 48,
            dim: 32,
            num_heads: 4,
            hidden_scale: 4.0,
            num_layers: 1,
            context: 64,
            max_period: 10_000.0,
            depformer_dim: 16,
            depformer_num_heads: 4,
            depformer_num_layers: 1,
            depformer_dim_feedforward: 32,
            depformer_weights_per_step: true,
            depformer_multi_linear: true,
            existing_text_padding_id: 3,
            audio_tokens_per_stream: 8,
            silence_tokens: vec![0; 8],
            sine_tokens: vec![0; 8],
        }
    }

    fn load_mimi() -> (Mimi, serde_json::Value, Device) {
        let device = device();
        let mut weights = Weights::from_file(fixture("mimi.safetensors"), DType::F32).unwrap();
        let mimi = Mimi::from_weights(&mimi_cfg(), &mut weights, &device).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture("mimi.json")).unwrap()).unwrap();
        (mimi, json, device)
    }

    #[test]
    fn mimi_matches_reference_latent_codes_and_audio() {
        let (mimi, json, device) = load_mimi();
        let input = f32s(&json["input"]);
        let wav = util::f32_tensor(input, [1, 1, 32], &device);
        let latent = tensor_f32(mimi.encode_latent(wav.clone(), None));
        assert_close(&latent, &f32s(&json["latent"]), 1e-3);
        let codes = tensor_i32(mimi.encode(wav, None));
        assert_eq!(codes, i32s(&json["codes"]));
        let decoded =
            tensor_f32(mimi.decode(util::i32_tensor(codes.clone(), [1, 8, 4], &device), None));
        assert_close(&decoded, &f32s(&json["decoded"]), 1e-3);
    }

    #[test]
    fn mimi_streaming_halves_match_reference() {
        let (mimi, json, device) = load_mimi();
        let input = f32s(&json["input"]);
        let mut bag = StreamingBag::new();
        let left = util::f32_tensor(input[..16].to_vec(), [1, 1, 16], &device);
        let right = util::f32_tensor(input[16..].to_vec(), [1, 1, 16], &device);
        let lat_l = mimi.encode_latent(left.clone(), Some(&mut bag));
        let lat_r = mimi.encode_latent(right.clone(), Some(&mut bag));
        let latent = tensor_f32(Tensor::cat(vec![lat_l, lat_r], 2));
        assert_close(&latent, &f32s(&json["streaming_latent"]), 1e-3);
        let mut bag = StreamingBag::new();
        let codes_l = mimi.encode(left, Some(&mut bag));
        let codes_r = mimi.encode(right, Some(&mut bag));
        let codes = tensor_i32(Tensor::cat(vec![codes_l, codes_r], 2));
        assert_eq!(codes, i32s(&json["streaming_codes"]));
    }

    fn load_lm() -> (LmModel, serde_json::Value, Device) {
        let device = device();
        let mut weights = Weights::from_file(fixture("lm.safetensors"), DType::F32).unwrap();
        let lm = LmModel::from_weights(&lm_cfg(), &mut weights, &device).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture("lm.json")).unwrap()).unwrap();
        (lm, json, device)
    }

    #[test]
    fn temporal_and_depth_match_reference() {
        let (lm, json, device) = load_lm();
        let codes = i32s(&json["codes"]);
        let sequence = util::i32_tensor(codes, [1, 17, 4], &device);
        let (hidden, text) = lm.forward_codes(sequence.clone(), None);
        assert_close(
            &tensor_f32(hidden.clone()),
            &f32s(&json["transformer_out"]),
            2e-3,
        );
        assert_close(&tensor_f32(text), &f32s(&json["text_logits"]), 2e-3);
        let dep = lm.depformer_logits_training(sequence, hidden);
        assert_close(&tensor_f32(dep), &f32s(&json["dep_logits"]), 2e-3);
    }

    #[test]
    fn streaming_temporal_matches_full_pass() {
        let (lm, json, device) = load_lm();
        let codes = i32s(&json["codes"]);
        let mut bag = StreamingBag::new();
        let mut hidden = Vec::new();
        let mut text = Vec::new();
        for step in 0..4 {
            let mut column = Vec::with_capacity(17);
            for codebook in 0..17 {
                column.push(codes[codebook * 4 + step]);
            }
            let sequence = util::i32_tensor(column, [1, 17, 1], &device);
            let (h, t) = lm.forward_codes(sequence, Some(&mut bag));
            hidden.extend(tensor_f32(h));
            text.extend(tensor_f32(t));
        }
        assert_close(&hidden, &f32s(&json["stream_transformer_out"]), 2e-3);
        assert_close(&text, &f32s(&json["stream_text_logits"]), 2e-3);
    }

    #[test]
    fn lmgen_greedy_steps_match_reference() {
        let (lm, json, _device) = load_lm();
        let mut state = LmState::new(&lm_cfg(), SamplingConfig::greedy()).unwrap();
        let users = json["lmgen_user"].as_array().unwrap();
        let outs = json["lmgen_out"].as_array().unwrap();
        assert_eq!(users.len(), outs.len());
        for (user, expect) in users.iter().zip(outs) {
            let user = i32s(user);
            let got = state.step(&lm, Some(&user), None, None);
            if expect.is_null() {
                assert!(got.is_none(), "expected no tokens, got {got:?}");
            } else {
                assert_eq!(got.unwrap(), i32s(expect));
            }
        }
    }

    #[test]
    fn row_chunks_match_an_unsplit_matmul() {
        let device = device();
        let weight = util::f32_tensor((0..32).map(|v| v as f32 * 0.01).collect(), [8, 4], &device);
        let input = util::f32_tensor(
            (0..12).map(|v| v as f32 * 0.1).collect(),
            [1, 3, 4],
            &device,
        );
        let full = util::linear(input.clone(), weight.clone());
        let chunks = util::RowChunks::from_tensor(weight, 16);
        assert!(chunks.chunk_count() > 1);
        let split = chunks.matmul(input);
        assert_close(&tensor_f32(split), &tensor_f32(full), 1e-5);
    }

    #[test]
    fn batch_session_turns_wav_samples_into_audio_and_text() {
        let device = device();
        let mut mimi_w = Weights::from_file(fixture("mimi.safetensors"), DType::F32).unwrap();
        let mut lm_w = Weights::from_file(fixture("lm.safetensors"), DType::F32).unwrap();
        let mimi = Mimi::from_weights(&mimi_cfg(), &mut mimi_w, &device).unwrap();
        let lm = LmModel::from_weights(&lm_cfg(), &mut lm_w, &device).unwrap();
        let cfg_model = ModelConfig {
            mimi: mimi_cfg(),
            lm: lm_cfg(),
        };
        let model =
            PersonaPlex::from_parts(cfg_model, mimi, lm, Tokenizer::stub(48), device, DType::F32);
        let session_cfg = SessionConfig::new("")
            .with_sampling(SamplingConfig::greedy())
            .with_silence_frames(0);
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let dir = std::env::temp_dir().join("personaplex-burn-batch-test");
        let _ = std::fs::create_dir_all(&dir);
        let wav_in = dir.join("in.wav");
        let wav_out = dir.join("out.wav");
        audio::write_wav(&wav_in, &input, 16).unwrap();
        let (samples, rate, channels) = audio::read_wav(&wav_in).unwrap();
        let batch = process_batch(&model, session_cfg, &samples, rate, channels).unwrap();
        audio::write_wav(&wav_out, &batch.pcm, batch.sample_rate).unwrap();
        let (written, written_rate, written_ch) = audio::read_wav(&wav_out).unwrap();
        assert_eq!(written_rate, 16);
        assert_eq!(written_ch, 1);
        assert_eq!(written.len(), 32);
        assert_eq!(batch.pcm.len(), 32);
        assert_eq!(batch.sample_rate, 16);
        assert!(batch
            .events
            .iter()
            .any(|event| matches!(event, Event::AudioOut { .. })));
        assert!(batch
            .events
            .iter()
            .any(|event| matches!(event, Event::Status(Status::Listening))));
        let mut session = Session::start(
            model.clone(),
            SessionConfig::new("Hi")
                .with_sampling(SamplingConfig::greedy())
                .with_silence_frames(0),
        )
        .unwrap();
        assert!(matches!(
            session.interrupt(),
            Event::Status(Status::Interrupted)
        ));
        assert!(matches!(
            session.set_muted(true),
            Event::Status(Status::Muted)
        ));
        let reset = session.reset().unwrap();
        assert!(reset
            .iter()
            .any(|event| matches!(event, Event::Status(Status::Reset))));
        let mut quiet = SessionConfig::new("")
            .with_sampling(SamplingConfig::greedy())
            .with_silence_frames(0);
        quiet.emit_timing = false;
        let sync = {
            let mut a = Session::start(model.clone(), quiet.clone()).unwrap();
            a.push_audio(&input, 16, 1).unwrap()
        };
        let async_events = {
            let mut b = Session::start(model, quiet).unwrap();
            poll_ready(b.push_audio_async(&input, 16, 1)).unwrap()
        };
        assert_eq!(sync, async_events);
    }

    #[test]
    fn voice_resample_matches_numpy_and_gain_formula() {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture("voice.json")).unwrap()).unwrap();
        let stereo = json["stereo"].as_array().unwrap();
        let channels = stereo.len();
        let frames = stereo[0].as_array().unwrap().len();
        let mut interleaved = vec![0.0f32; channels * frames];
        for (channel, row) in stereo.iter().enumerate() {
            for (index, sample) in row.as_array().unwrap().iter().enumerate() {
                interleaved[index * channels + channel] = sample.as_f64().unwrap() as f32;
            }
        }
        let mono = audio::downmix(&interleaved, channels).unwrap();
        assert_close(&mono, &f32s(&json["mono"]), 1e-5);
        let resampled = audio::linear_resample(&mono, 16_000, 24_000);
        assert_close(&resampled, &f32s(&json["resampled"]), 1e-4);
        let recorded = json["loudness"].as_f64().unwrap() as f32;
        let gained = apply_lufs_gain(&resampled, recorded, -24.0);
        assert_close(&gained, &f32s(&json["normalized"]), 1e-3);
        let (_out, measured) = audio::normalize_lufs(&resampled, 24_000, -24.0).unwrap();
        assert!(measured.is_finite(), "ebur128 loudness {measured}");
        // pyloudnorm and ebur128 are different meters. The gain formula above
        // is what both apply once loudness is known. A gap larger than about
        // one LU is recorded for the README rather than forced to match.
        let gap = (measured - recorded).abs();
        eprintln!("voice loudness ebur128={measured} pyloudnorm={recorded} gap={gap}");
        assert!(gap < 3.0, "loudness meters diverged by {gap} LU");
    }

    #[test]
    fn missing_token_skips_gated_download() {
        if std::env::var_os("HF_TOKEN").is_some()
            || std::env::var_os("PERSONAPLEX_WEIGHTS").is_some()
        {
            eprintln!("weights are configured; not downloading the 7B checkpoint in this test");
            return;
        }
        let err = match crate::weights::PersonaPlex::load(
            crate::weights::WeightSource::huggingface(),
            device(),
            DtypePolicy::F32,
        ) {
            Ok(_) => panic!("gated load returned a model without a token"),
            Err(err) => err,
        };
        let text = err.to_string();
        assert!(
            text.contains("https://huggingface.co/nvidia/personaplex-7b-v1"),
            "{text}"
        );
    }

    fn poll_ready<T>(fut: impl std::future::Future<Output = T>) -> T {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        fn noop(_: *const ()) {}
        fn clone(data: *const ()) -> RawWaker {
            RawWaker::new(data, &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("push_audio_async was not ready"),
        }
    }
}
