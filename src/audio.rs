//! PCM helpers shared by the session and the batch adapter.
//!
//! Resampling matches `numpy.interp` on `linspace(..., endpoint=False)`
//! positions. Loudness uses BS.1770 integrated loudness via `ebur128`.
//! The reference voice prompt targets -24 LUFS.

use crate::error::{Error, Result};

/// Interleaved PCM to mono by averaging channels.
pub fn downmix(interleaved: &[f32], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(Error::session("channel count must be at least 1"));
    }
    if channels == 1 {
        return Ok(interleaved.to_vec());
    }
    if !interleaved.len().is_multiple_of(channels) {
        return Err(Error::session(
            "interleaved audio length is not divisible by the channel count",
        ));
    }
    let frames = interleaved.len() / channels;
    let mut mono = vec![0.0f32; frames];
    for (i, sample) in interleaved.iter().enumerate() {
        mono[i / channels] += *sample;
    }
    let scale = channels as f32;
    for sample in &mut mono {
        *sample /= scale;
    }
    Ok(mono)
}

/// Linear resample. `src_sr == dst_sr` returns a copy.
pub fn linear_resample(input: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || input.is_empty() {
        return input.to_vec();
    }
    let duration = input.len() as f64 / f64::from(src_sr);
    let dst_len = (duration * f64::from(dst_sr)).round() as usize;
    if dst_len == 0 {
        return Vec::new();
    }
    if dst_len == 1 || input.len() == 1 {
        return vec![input[0]; dst_len];
    }
    let n_src = input.len();
    let max_src = (n_src - 1) as f64 / n_src as f64;
    let mut out = Vec::with_capacity(dst_len);
    for j in 0..dst_len {
        let dst_pos = j as f64 / dst_len as f64;
        if dst_pos >= max_src {
            out.push(input[n_src - 1]);
            continue;
        }
        let src_pos = dst_pos * n_src as f64;
        let i = src_pos.floor() as usize;
        let t = (src_pos - i as f64) as f32;
        let a = input[i];
        let b = input[i + 1];
        out.push(a + (b - a) * t);
    }
    out
}

/// Normalize mono audio to `target` LUFS. Returns the scaled signal and the measured LUFS.
pub fn normalize_lufs(mono: &[f32], sample_rate: u32, target: f32) -> Result<(Vec<f32>, f32)> {
    if mono.is_empty() {
        return Ok((Vec::new(), f32::NEG_INFINITY));
    }
    let mut meter = ebur128::EbuR128::new(1, sample_rate, ebur128::Mode::I)
        .map_err(|err| Error::session(format!("loudness meter: {err}")))?;
    meter
        .add_frames_f32(mono)
        .map_err(|err| Error::session(format!("loudness meter: {err}")))?;
    let loudness = meter
        .loudness_global()
        .map_err(|err| Error::session(format!("loudness meter: {err}")))? as f32;
    if !loudness.is_finite() {
        return Ok((mono.to_vec(), loudness));
    }
    let gain = 10f32.powf((target - loudness) / 20.0);
    Ok((mono.iter().map(|sample| sample * gain).collect(), loudness))
}

/// Gain that takes `loudness` to `target`, applied to `mono`.
pub fn apply_lufs_gain(mono: &[f32], loudness: f32, target: f32) -> Vec<f32> {
    let gain = 10f32.powf((target - loudness) / 20.0);
    mono.iter().map(|sample| sample * gain).collect()
}

/// Accumulates samples until `frame_size` is available.
#[derive(Clone, Debug)]
pub struct FrameBuffer {
    pending: Vec<f32>,
    frame_size: usize,
}

impl FrameBuffer {
    pub fn new(frame_size: usize) -> Self {
        Self {
            pending: Vec::new(),
            frame_size,
        }
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.pending.extend_from_slice(samples);
        let mut frames = Vec::new();
        while self.pending.len() >= self.frame_size {
            let frame: Vec<f32> = self.pending.drain(..self.frame_size).collect();
            frames.push(frame);
        }
        frames
    }

    /// Zero-pad the tail into one last frame. `None` when nothing is pending.
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        if self.pending.is_empty() {
            return None;
        }
        self.pending.resize(self.frame_size, 0.0);
        Some(std::mem::take(&mut self.pending))
    }
}

/// Read an interleaved WAV as f32 samples.
pub fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32, usize)> {
    let mut reader = hound::WavReader::open(path).map_err(|err| Error::wav(err.to_string()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|err| Error::wav(err.to_string()))?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|err| Error::wav(err.to_string()))?
        }
    };
    Ok((samples, spec.sample_rate, channels))
}

/// Write mono f32 WAV.
pub fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|err| Error::wav(err.to_string()))?;
    for sample in samples {
        writer
            .write_sample(*sample)
            .map_err(|err| Error::wav(err.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|err| Error::wav(err.to_string()))?;
    Ok(())
}
