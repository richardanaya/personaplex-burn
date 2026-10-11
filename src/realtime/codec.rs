//! PCM16 and G.711 for the realtime wire format.
//!
//! PCM is little-endian linear16. G.711 μ-law and A-law are the 8 kHz
//! telephony codecs (`audio/pcmu`, `audio/pcma`).

/// Decode little-endian PCM16 to float in about [-1, 1]. A trailing odd byte is dropped.
pub fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair) as f32 / 32767.0)
        .collect()
}

/// Encode float PCM to little-endian PCM16.
pub fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&scaled.to_le_bytes());
    }
    bytes
}

pub fn mulaw_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|byte| mulaw_decode(*byte) as f32 / 32767.0)
        .collect()
}

pub fn alaw_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|byte| alaw_decode(*byte) as f32 / 32767.0)
        .collect()
}

pub fn f32_to_mulaw(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .map(|sample| mulaw_encode((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16))
        .collect()
}

pub fn f32_to_alaw(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .map(|sample| alaw_encode((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16))
        .collect()
}

fn mulaw_decode(byte: u8) -> i16 {
    let byte = !byte;
    let sign = byte & 0x80;
    let exponent = (byte >> 4) & 0x07;
    let mantissa = byte & 0x0f;
    let mut sample = (((i32::from(mantissa) << 3) + 0x84) << exponent) - 0x84;
    if sign != 0 {
        sample = -sample;
    }
    sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn mulaw_encode(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;
    let mut sample = i32::from(sample);
    let sign = if sample < 0 { 0x80 } else { 0 };
    if sample < 0 {
        sample = -sample;
    }
    if sample > CLIP {
        sample = CLIP;
    }
    sample += BIAS;
    let mut exponent = 7i32;
    let mut mask = 0x4000i32;
    while exponent > 0 && (sample & mask) == 0 {
        exponent -= 1;
        mask >>= 1;
    }
    let mantissa = (sample >> (exponent + 3)) & 0x0f;
    !(sign | ((exponent as u8) << 4) | mantissa as u8)
}

fn alaw_decode(byte: u8) -> i16 {
    let byte = byte ^ 0x55;
    let sign = byte & 0x80;
    let exponent = (byte >> 4) & 0x07;
    let mantissa = byte & 0x0f;
    let mut sample = if exponent == 0 {
        (i32::from(mantissa) << 4) + 8
    } else {
        ((i32::from(mantissa) << 4) + 0x108) << (exponent - 1)
    };
    if sign != 0 {
        sample = -sample;
    }
    sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn alaw_encode(sample: i16) -> u8 {
    let mut sample = i32::from(sample);
    let sign = if sample >= 0 { 0x80u8 } else { 0 };
    if sample < 0 {
        sample = -sample;
    }
    if sample > 32635 {
        sample = 32635;
    }
    let exponent = if sample < 256 {
        0
    } else {
        let mut exp = 7i32;
        let mut mask = 0x4000i32;
        while exp > 1 && (sample & mask) == 0 {
            exp -= 1;
            mask >>= 1;
        }
        exp
    };
    let mantissa = if exponent == 0 {
        (sample >> 4) & 0x0f
    } else {
        (sample >> (exponent + 3)) & 0x0f
    };
    (sign | ((exponent as u8) << 4) | mantissa as u8) ^ 0x55
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy: f32 = samples.iter().map(|sample| sample * sample).sum();
    (energy / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulaw_roundtrip_keeps_a_loud_signal_loud() {
        let pcm: Vec<f32> = (0..160)
            .map(|i| if i % 2 == 0 { 0.8 } else { -0.8 })
            .collect();
        let back = mulaw_to_f32(&f32_to_mulaw(&pcm));
        assert!(rms(&back) > 0.3, "rms {}", rms(&back));
        assert!(mulaw_decode(mulaw_encode(0)).abs() < 16);
    }

    #[test]
    fn alaw_roundtrip_keeps_a_loud_signal_loud() {
        let pcm: Vec<f32> = (0..160)
            .map(|i| if i % 2 == 0 { 0.8 } else { -0.8 })
            .collect();
        let back = alaw_to_f32(&f32_to_alaw(&pcm));
        assert!(rms(&back) > 0.3, "rms {}", rms(&back));
    }

    #[test]
    fn pcm16_roundtrip() {
        let pcm = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let back = pcm16_to_f32(&f32_to_pcm16(&pcm));
        for (got, expect) in back.iter().zip(&pcm) {
            assert!((got - expect).abs() < 1.0e-4, "{got} vs {expect}");
        }
    }
}
