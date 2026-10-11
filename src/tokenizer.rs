//! SentencePiece tokenizer.
//!
//! The release checkpoint ships `tokenizer_spm_32k_3.model`. This reader
//! understands the protobuf container (pieces, scores, model type) and encodes
//! with unigram Viterbi or BPE. It does not depend on the sentencepiece C++
//! library. Parity tests use [`Tokenizer::stub`] because that model file is gated.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelType {
    Unigram,
    Bpe,
}

#[derive(Clone)]
struct Piece {
    surface: String,
    score: f32,
}

/// Text tokenizer owned by a [`crate::PersonaPlex`] model.
#[derive(Clone)]
pub struct Tokenizer {
    pieces: Vec<Piece>,
    by_first: HashMap<char, Vec<usize>>,
    by_surface: HashMap<String, i32>,
    unk: i32,
    model: ModelType,
    stub: bool,
    card: usize,
}

impl Tokenizer {
    /// Char-to-id map that avoids padding ids 0 and 3. Used by tiny-model tests.
    pub fn stub(card: usize) -> Self {
        let card = card.max(8);
        let mut pieces = Vec::with_capacity(card);
        for id in 0..card {
            let surface = match id {
                0 => "EPAD".to_string(),
                3 => "PAD".to_string(),
                other => format!("<{other}>"),
            };
            pieces.push(Piece {
                surface,
                score: 0.0,
            });
        }
        Self::finish(pieces, ModelType::Unigram, true, card, 0)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let (pieces, model) = parse_model(&bytes)?;
        if pieces.is_empty() {
            return Err(Error::tokenizer("sentencepiece model has no pieces"));
        }
        let unk = pieces
            .iter()
            .position(|piece| piece.surface == "<unk>")
            .unwrap_or(0) as i32;
        let card = pieces.len();
        Ok(Self::finish(pieces, model, false, card, unk))
    }

    fn finish(pieces: Vec<Piece>, model: ModelType, stub: bool, card: usize, unk: i32) -> Self {
        let mut by_first: HashMap<char, Vec<usize>> = HashMap::new();
        let mut by_surface = HashMap::new();
        for (id, piece) in pieces.iter().enumerate() {
            by_surface.insert(piece.surface.clone(), id as i32);
            if let Some(first) = piece.surface.chars().next() {
                by_first.entry(first).or_default().push(id);
            }
        }
        Self {
            pieces,
            by_first,
            by_surface,
            unk,
            model,
            stub,
            card,
        }
    }

    pub fn id_to_piece(&self, id: i32) -> String {
        self.pieces
            .get(id as usize)
            .map(|piece| piece.surface.clone())
            .unwrap_or_else(|| format!("<{id}>"))
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        if text.is_empty() {
            return Vec::new();
        }
        if self.stub {
            let span = (self.card as u32).saturating_sub(4).max(1);
            return text
                .chars()
                .map(|ch| ((ch as u32 % span) + 4) as i32)
                .collect();
        }
        let norm = normalize(text);
        match self.model {
            ModelType::Unigram => self.encode_unigram(&norm),
            ModelType::Bpe => self.encode_bpe(&norm),
        }
    }

    fn encode_unigram(&self, text: &str) -> Vec<i32> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut prev: Vec<Option<(usize, i32)>> = vec![None; n + 1];
        best[0] = 0.0;
        for start in 0..n {
            if !best[start].is_finite() {
                continue;
            }
            let Some(candidates) = self.by_first.get(&chars[start]) else {
                continue;
            };
            for &id in candidates {
                let surface = self.pieces[id].surface.as_str();
                let len = surface.chars().count();
                if len == 0 || start + len > n {
                    continue;
                }
                if chars[start..start + len].iter().collect::<String>() != surface {
                    continue;
                }
                let score = best[start] + self.pieces[id].score;
                let end = start + len;
                if score > best[end] {
                    best[end] = score;
                    prev[end] = Some((start, id as i32));
                }
            }
        }
        if !best[n].is_finite() {
            return self.encode_chars(&chars);
        }
        let mut ids = Vec::new();
        let mut cursor = n;
        while cursor > 0 {
            match prev[cursor] {
                Some((start, id)) => {
                    ids.push(id);
                    cursor = start;
                }
                None => break,
            }
        }
        ids.reverse();
        if ids.is_empty() {
            self.encode_chars(&chars)
        } else {
            ids
        }
    }

    fn encode_chars(&self, chars: &[char]) -> Vec<i32> {
        chars
            .iter()
            .map(|ch| {
                let surface = ch.to_string();
                self.by_surface.get(&surface).copied().unwrap_or(self.unk)
            })
            .collect()
    }

    fn encode_bpe(&self, text: &str) -> Vec<i32> {
        let mut symbols: Vec<String> = text.chars().map(|ch| ch.to_string()).collect();
        if symbols.is_empty() {
            return Vec::new();
        }
        loop {
            let mut best: Option<(usize, f32)> = None;
            for index in 0..symbols.len().saturating_sub(1) {
                let merged = format!("{}{}", symbols[index], symbols[index + 1]);
                let Some(id) = self.by_surface.get(&merged) else {
                    continue;
                };
                let score = self.pieces[*id as usize].score;
                if best.map(|(_, current)| score > current).unwrap_or(true) {
                    best = Some((index, score));
                }
            }
            let Some((index, _)) = best else {
                break;
            };
            let merged = format!("{}{}", symbols[index], symbols[index + 1]);
            symbols[index] = merged;
            symbols.remove(index + 1);
        }
        symbols
            .iter()
            .map(|symbol| self.by_surface.get(symbol).copied().unwrap_or(self.unk))
            .collect()
    }
}

fn normalize(text: &str) -> String {
    let mut out = String::new();
    if !text.starts_with('▁') {
        out.push('▁');
    }
    for ch in text.chars() {
        if ch == ' ' {
            out.push('▁');
        } else {
            out.push(ch);
        }
    }
    out
}

fn parse_model(bytes: &[u8]) -> Result<(Vec<Piece>, ModelType)> {
    let mut pieces = Vec::new();
    let mut model = ModelType::Unigram;
    let mut cursor = 0;
    while cursor < bytes.len() {
        let (key, next) = read_varint(bytes, cursor)?;
        cursor = next;
        let field = key >> 3;
        let wire = key & 7;
        match (field, wire) {
            (1, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                cursor = next;
                let end = cursor + len as usize;
                if end > bytes.len() {
                    return Err(Error::tokenizer("truncated sentencepiece piece"));
                }
                pieces.push(parse_piece(&bytes[cursor..end])?);
                cursor = end;
            }
            (2, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                cursor = next;
                let end = cursor + len as usize;
                if end > bytes.len() {
                    return Err(Error::tokenizer("truncated sentencepiece trainer spec"));
                }
                model = parse_trainer(&bytes[cursor..end])?;
                cursor = end;
            }
            (_, 0) => {
                let (_, next) = read_varint(bytes, cursor)?;
                cursor = next;
            }
            (_, 1) => cursor += 8,
            (_, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                cursor = next + len as usize;
            }
            (_, 5) => cursor += 4,
            _ => {
                return Err(Error::tokenizer(format!(
                    "unsupported protobuf wire type {wire}"
                )))
            }
        }
    }
    Ok((pieces, model))
}

fn parse_piece(bytes: &[u8]) -> Result<Piece> {
    let mut cursor = 0;
    let mut surface = String::new();
    let mut score = 0.0f32;
    while cursor < bytes.len() {
        let (key, next) = read_varint(bytes, cursor)?;
        cursor = next;
        let field = key >> 3;
        let wire = key & 7;
        match (field, wire) {
            (1, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                cursor = next;
                let end = cursor + len as usize;
                surface = String::from_utf8_lossy(&bytes[cursor..end]).into_owned();
                cursor = end;
            }
            (2, 5) => {
                if cursor + 4 > bytes.len() {
                    return Err(Error::tokenizer("truncated piece score"));
                }
                score = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
                cursor += 4;
            }
            (_, 0) => {
                let (_, next) = read_varint(bytes, cursor)?;
                cursor = next;
            }
            (_, 1) => cursor += 8,
            (_, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                cursor = next + len as usize;
            }
            (_, 5) => cursor += 4,
            _ => {
                return Err(Error::tokenizer("bad piece field"));
            }
        }
    }
    Ok(Piece { surface, score })
}

fn parse_trainer(bytes: &[u8]) -> Result<ModelType> {
    let mut cursor = 0;
    let mut model = ModelType::Unigram;
    while cursor < bytes.len() {
        let (key, next) = read_varint(bytes, cursor)?;
        cursor = next;
        let field = key >> 3;
        let wire = key & 7;
        match (field, wire) {
            (11, 0) => {
                let (value, next) = read_varint(bytes, cursor)?;
                cursor = next;
                model = if value == 2 {
                    ModelType::Bpe
                } else {
                    ModelType::Unigram
                };
            }
            (_, 0) => {
                let (_, next) = read_varint(bytes, cursor)?;
                cursor = next;
            }
            (_, 1) => cursor += 8,
            (_, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                cursor = next + len as usize;
            }
            (_, 5) => cursor += 4,
            _ => return Err(Error::tokenizer("bad trainer spec")),
        }
    }
    Ok(model)
}

fn read_varint(bytes: &[u8], mut cursor: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    while cursor < bytes.len() && shift < 64 {
        let byte = bytes[cursor];
        cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, cursor));
        }
        shift += 7;
    }
    Err(Error::tokenizer("truncated varint"))
}
