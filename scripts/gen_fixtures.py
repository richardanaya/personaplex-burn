#!/usr/bin/env python3
"""Generate tiny-model parity fixtures from the NVIDIA PersonaPlex / Moshi reference.

The script imports the reference package (``moshi``) and runs randomly initialized
models with the same architecture as PersonaPlex, scaled down. Outputs are
safetensors weights plus JSON arrays that ``cargo test`` compares against.

Set ``PERSONAPLEX_ROOT`` to a checkout of https://github.com/NVIDIA/personaplex
(the directory that contains the ``moshi/`` Python package). When unset, the
script tries ``/tmp/personaplex`` and ``../personaplex``.

No gated weights or network access are required.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


def _import_moshi():
    root = os.environ.get("PERSONAPLEX_ROOT", "")
    candidates = []
    if root:
        candidates.append(Path(root))
    candidates += [Path("/tmp/personaplex"), Path(__file__).resolve().parents[1] / ".." / "personaplex"]
    for cand in candidates:
        pkg = cand / "moshi"
        if (pkg / "moshi").is_dir():
            # Stub sphn: lm.py imports it at module scope, fixtures do not call it.
            if "sphn" not in sys.modules:
                sys.modules["sphn"] = types.ModuleType("sphn")
            sys.path.insert(0, str(pkg))
            import moshi  # noqa: F401

            return cand
    raise SystemExit(
        "Cannot import the PersonaPlex reference package. Clone "
        "https://github.com/NVIDIA/personaplex and set PERSONAPLEX_ROOT."
    )


def _dump_safetensors(path: Path, state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    cpu = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(cpu, str(path))


def _tolist(t: torch.Tensor):
    return t.detach().cpu().to(torch.float32).numpy().tolist()


def _intlist(t: torch.Tensor):
    return t.detach().cpu().to(torch.int64).numpy().tolist()


def build_mimi():
    from moshi.models.compression import MimiModel
    from moshi.modules.seanet import SEANetDecoder, SEANetEncoder
    from moshi.modules.transformer import ProjectedTransformer
    from moshi.quantization import SplitResidualVectorQuantizer

    seanet_kwargs = dict(
        channels=1,
        dimension=8,
        causal=True,
        n_filters=4,
        n_residual_layers=1,
        activation="ELU",
        compress=2,
        dilation_base=2,
        disable_norm_outer_blocks=0,
        kernel_size=3,
        residual_kernel_size=3,
        last_kernel_size=3,
        norm="none",
        pad_mode="constant",
        ratios=[2, 2],
        true_skip=True,
    )
    torch.manual_seed(7)
    enc = SEANetEncoder(**seanet_kwargs)
    dec = SEANetDecoder(**seanet_kwargs)
    tr_kwargs = dict(
        d_model=8,
        num_heads=2,
        num_layers=1,
        causal=True,
        layer_scale=0.01,
        context=16,
        conv_layout=True,
        max_period=10000,
        gating="none",
        norm="layer_norm",
        positional_embedding="rope",
        dim_feedforward=16,
        input_dimension=8,
        output_dimensions=[8],
    )
    etr = ProjectedTransformer(device="cpu", **tr_kwargs)
    dtr = ProjectedTransformer(device="cpu", **tr_kwargs)
    quant = SplitResidualVectorQuantizer(
        dimension=4, n_q=8, bins=8, input_dimension=8, output_dimension=8
    )
    # Non-trivial codebook centroids (embedding = embedding_sum / cluster_usage).
    with torch.no_grad():
        for module in (quant.rvq_first, quant.rvq_rest):
            for layer in module.vq.layers:
                cb = layer._codebook
                cb.embedding_sum.copy_(torch.randn_like(cb.embedding_sum))
                cb.cluster_usage.copy_(torch.rand_like(cb.cluster_usage) + 0.25)
                if hasattr(cb, "_embedding"):
                    cb._embedding = None

    sample_rate = 16
    hop = enc.hop_length
    encoder_frame_rate = sample_rate / hop
    frame_rate = encoder_frame_rate / 2  # stride-2 conv resample
    model = MimiModel(
        enc,
        dec,
        quant,
        channels=1,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        encoder_frame_rate=encoder_frame_rate,
        causal=True,
        resample_method="conv",
        encoder_transformer=etr,
        decoder_transformer=dtr,
    ).eval()
    return model, {
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "encoder_frame_rate": encoder_frame_rate,
        "dimension": 8,
        "n_filters": 4,
        "ratios": [2, 2],
        "kernel_size": 3,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        "n_residual_layers": 1,
        "dilation_base": 2,
        "compress": 2,
        "channels": 1,
        "causal": True,
        "seanet_activation": "elu",
        "transformer_layers": 1,
        "transformer_heads": 2,
        "transformer_dim": 8,
        "transformer_ff": 16,
        "transformer_context": 16,
        "layer_scale": 0.01,
        "quantizer_dim": 4,
        "quantizer_bins": 8,
        "quantizer_n_q": 8,
        "n_q_semantic": 1,
        "max_period": 10000,
    }


def run_mimi(model, cfg):
    torch.manual_seed(11)
    # Length is a multiple of hop (4) so the causal conv path is well defined.
    wav = torch.randn(1, 1, 32)
    with torch.no_grad():
        latent = model._encode_to_unquantized_latent(wav)
        codes = model.encode(wav)
        decoded = model.decode(codes)
    # Streaming: one shot should track the causal encoder. Feed the waveform in
    # two halves while streaming is enabled and record codes.
    model.streaming_forever(1)
    model.reset_streaming()
    halves = []
    latents = []
    with torch.no_grad():
        for part in wav.split(16, dim=-1):
            latents.append(model._encode_to_unquantized_latent(part))
            halves.append(model.quantizer.encode(latents[-1]))
    stream_codes = torch.cat(halves, dim=-1)
    stream_latent = torch.cat(latents, dim=-1)
    model._stop_streaming()
    return {
        "input": _tolist(wav[0, 0]),
        "latent": _tolist(latent[0]),
        "codes": _intlist(codes[0]),
        "decoded": _tolist(decoded[0, 0]),
        "streaming_latent": _tolist(stream_latent[0]),
        "streaming_codes": _intlist(stream_codes[0]),
        "config": cfg,
    }


def build_lm():
    from moshi.models.lm import LMModel

    torch.manual_seed(3)
    delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    model = LMModel(
        delays=delays,
        n_q=16,
        dep_q=16,
        card=32,
        text_card=48,
        dim=32,
        num_heads=4,
        hidden_scale=4.0,
        norm="rms_norm_f32",
        causal=True,
        num_layers=1,
        gating="silu",
        positional_embedding="rope",
        max_period=10000,
        context=64,
        depformer_dim=16,
        depformer_num_heads=4,
        depformer_num_layers=1,
        depformer_causal=True,
        depformer_gating="silu",
        depformer_pos_emb="none",
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_dim_feedforward=32,
        depformer_context=16,
        existing_text_padding_id=3,
        device="cpu",
        dtype=torch.float32,
    ).eval()
    cfg = {
        "delays": delays,
        "n_q": 16,
        "dep_q": 16,
        "card": 32,
        "text_card": 48,
        "dim": 32,
        "num_heads": 4,
        "hidden_scale": 4.0,
        "norm": "rms_norm_f32",
        "num_layers": 1,
        "gating": "silu",
        "positional_embedding": "rope",
        "max_period": 10000,
        "context": 64,
        "depformer_dim": 16,
        "depformer_num_heads": 4,
        "depformer_num_layers": 1,
        "depformer_gating": "silu",
        "depformer_pos_emb": "none",
        "depformer_multi_linear": True,
        "depformer_weights_per_step": True,
        "depformer_dim_feedforward": 32,
        "existing_text_padding_id": 3,
        "audio_tokens_per_stream": 8,
    }
    return model, cfg


def run_lm(model, cfg):
    from moshi.models.lm import LMGen

    torch.manual_seed(5)
    b, k, t = 1, model.num_codebooks, 4
    codes = torch.randint(0, cfg["card"], (b, k, t))
    codes[:, 0] = torch.randint(0, cfg["text_card"], (b, t))
    with torch.no_grad():
        transformer_out, text_logits = model.forward_codes(codes)
        dep_logits = model.forward_depformer_training(codes, transformer_out)

    # Streaming temporal steps, one frame at a time, must match the full pass
    # when the cache has not wrapped.
    model.streaming_forever(1)
    model.reset_streaming()
    step_outs = []
    step_text = []
    with torch.no_grad():
        for i in range(t):
            out, logits = model.forward_codes(codes[:, :, i : i + 1])
            step_outs.append(out[0, 0])
            step_text.append(logits[0, 0, 0])
    model._stop_streaming()

    # Greedy LMGen steps. User stream is 8 codebooks, agent prompt is empty
    # (sampled). use_sampling False -> argmax, no RNG.
    device = "cpu"
    lm_gen = LMGen(
        model,
        device=device,
        use_sampling=False,
        temp=0.8,
        temp_text=0.7,
        top_k=4,
        top_k_text=4,
        audio_silence_frame_cnt=1,
        sample_rate=16,
        frame_rate=4,
    )
    lm_gen.streaming_forever(1)
    lm_gen.reset_streaming()
    user_steps = []
    gen_steps = []
    g = torch.Generator().manual_seed(9)
    for _ in range(6):
        user = torch.randint(0, cfg["card"], (1, 8, 1), generator=g)
        user_steps.append(_intlist(user[0, :, 0]))
        out = lm_gen.step(input_tokens=user)
        if out is None:
            gen_steps.append(None)
        else:
            gen_steps.append(_intlist(out[0, :, 0]))
    lm_gen._stop_streaming()

    return {
        "config": cfg,
        "codes": _intlist(codes[0]),
        "transformer_out": _tolist(transformer_out[0]),
        "text_logits": _tolist(text_logits[0, 0]),
        "dep_logits": _tolist(dep_logits[0]),
        "stream_transformer_out": [_tolist(x) for x in step_outs],
        "stream_text_logits": [_tolist(x) for x in step_text],
        "lmgen_user": user_steps,
        "lmgen_out": gen_steps,
    }


def run_sampling():
    """Match moshi.utils.sampling.sample_token, including its exponential noise."""
    from moshi.utils.sampling import sample_token

    torch.manual_seed(123)
    logits = torch.tensor(
        [[0.2, 1.5, -0.4, 0.7, 2.2, -1.0, 0.3, 1.1, 0.0, -0.2]], dtype=torch.float32
    )
    greedy = sample_token(logits, use_sampling=False, temp=1.0, top_k=0)

    recorded = {}
    real_exp = torch.Tensor.exponential_

    def wrapped(self, lambd=1, generator=None):
        out = real_exp(self, lambd, generator=generator)
        recorded["noise"] = out.detach().clone()
        return out

    torch.Tensor.exponential_ = wrapped
    try:
        torch.manual_seed(99)
        sampled = sample_token(logits, use_sampling=True, temp=0.7, top_k=4)
    finally:
        torch.Tensor.exponential_ = real_exp

    # Second draw with an explicit generator, noise captured the same way.
    g = torch.Generator().manual_seed(99)
    # Rebuild logits so top-k sees a fresh tensor (softmax is deterministic).
    logits2 = logits.clone()
    recorded.clear()
    torch.Tensor.exponential_ = wrapped
    try:
        sampled_gen = sample_token(logits2, use_sampling=True, temp=0.7, top_k=4)
    finally:
        torch.Tensor.exponential_ = real_exp
    _ = g  # seed is via torch.manual_seed above; generator path is covered by the module

    # Portable SplitMix64 so Rust's seeded sampler matches this script bit for bit.
    # The exponential-argmax step is the same one sample_token uses; only the
    # uniform source differs from PyTorch's generator (which is not stable).
    def splitmix_uniforms(seed: int, n: int) -> list[float]:
        mask = (1 << 64) - 1
        state = seed & mask

        def next_u64() -> int:
            nonlocal state
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            return (z ^ (z >> 31)) & mask

        out = []
        for _ in range(n):
            # (0, 1]: never zero, so log is finite.
            u = (next_u64() >> 11) + 1
            out.append(u / 9007199254740992.0)
        return out

    uniforms = splitmix_uniforms(42, 4)
    noise_seed = [-float(np.log(u)) for u in uniforms]
    probs = torch.softmax(logits / 0.7, dim=-1)
    topv, topi = torch.topk(probs, 4, dim=-1)
    scores = topv / torch.tensor(noise_seed, dtype=torch.float32).view(1, -1)
    seeded = int(topi.gather(-1, scores.argmax(-1, keepdim=True))[0, 0].item())

    return {
        "logits": _tolist(logits[0]),
        "greedy": int(greedy[0].item()),
        "temp": 0.7,
        "top_k": 4,
        "noise": _tolist(recorded["noise"][0]),
        "sampled": int(sampled[0].item()),
        "sampled_repeat": int(sampled_gen[0].item()),
        "seed": 42,
        "seed_uniforms": uniforms,
        "seeded_token": seeded,
    }


def _linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32)
    duration = x.shape[0] / float(src_sr)
    dst_len = int(round(duration * dst_sr))
    if dst_len <= 1 or x.shape[0] <= 1:
        return np.resize(x, dst_len).astype(np.float32)
    src_pos = np.linspace(0.0, 1.0, x.shape[0], endpoint=False)
    dst_pos = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    return np.interp(dst_pos, src_pos, x).astype(np.float32)


def _lufs_normalize(mono: np.ndarray, sr: int, target: float) -> tuple[np.ndarray, float]:
    try:
        import pyloudnorm as pyln
    except ImportError:
        # Energy fallback so the fixture still records a gain. The Rust side
        # uses BS.1770 via ebur128 and compares with a loose tolerance when
        # pyloudnorm produced the reference; this branch is only a last resort.
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
        loudness = 20.0 * np.log10(rms + 1e-12) - 0.691
        gain = 10.0 ** ((target - loudness) / 20.0)
        return (mono * gain).astype(np.float32), float(loudness)
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(mono))
    out = pyln.normalize.loudness(mono, loudness, target).astype(np.float32)
    return out, loudness


def run_voice():
    rng = np.random.default_rng(21)
    sr = 16000
    channels = 2
    # pyloudnorm's integrated loudness needs more than one 400 ms block.
    n = sr  # 1 second
    stereo = rng.standard_normal((channels, n)).astype(np.float32) * 0.1
    mono = stereo.mean(axis=0)
    resampled = _linear_resample(mono, sr, 24000)
    normalized, loudness = _lufs_normalize(resampled, 24000, -24.0)
    frame = 1920
    padded = normalized
    if padded.shape[0] % frame != 0:
        pad = frame - (padded.shape[0] % frame)
        padded = np.concatenate([padded, np.zeros(pad, dtype=np.float32)])
    frames = padded.reshape(-1, frame)
    return {
        "src_sr": sr,
        "dst_sr": 24000,
        "channels": channels,
        "frame_size": frame,
        "target_lufs": -24.0,
        "loudness": loudness,
        "stereo": stereo.tolist(),
        "mono": mono.tolist(),
        "resampled": resampled.tolist(),
        "normalized": normalized.tolist(),
        "frames": frames.tolist(),
        "lufs_backend": "pyloudnorm" if "pyloudnorm" in sys.modules else "rms-fallback",
    }


def main():
    root = _import_moshi()
    print(f"using reference at {root}")
    out = Path(__file__).resolve().parents[1] / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    mimi, mcfg = build_mimi()
    mimi_io = run_mimi(mimi, mcfg)
    _dump_safetensors(out / "mimi.safetensors", mimi.state_dict())
    (out / "mimi.json").write_text(json.dumps(mimi_io))
    print("mimi", "codes", np.array(mimi_io["codes"]).shape, "decoded", len(mimi_io["decoded"]))

    lm, lcfg = build_lm()
    lm_io = run_lm(lm, lcfg)
    _dump_safetensors(out / "lm.safetensors", lm.state_dict())
    (out / "lm.json").write_text(json.dumps(lm_io))
    print("lmgen", lm_io["lmgen_out"])

    sample = run_sampling()
    (out / "sample.json").write_text(json.dumps(sample))
    print("sample", sample["greedy"], sample["sampled"], "noise", len(sample["noise"]))

    voice = run_voice()
    (out / "voice.json").write_text(json.dumps(voice))
    print("voice frames", len(voice["frames"]), "lufs", voice["loudness"], voice["lufs_backend"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
