//! Safetensors loading, the depformer 8→16 expansion, and [`PersonaPlex`].
//!
//! Checkpoints are bf16 or f16 on disk. They are widened to f32 while parsing,
//! then stored as f16 or f32 depending on [`DtypePolicy`] and
//! `device.supports_dtype`. Matrices larger than the WebGPU buffer budget are
//! split by [`crate::util::RowChunks`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::module::Param;
use burn::tensor::{DType, Device, Tensor};
use half::{bf16, f16};
use safetensors::Dtype as SafeDtype;
use safetensors::SafeTensors;

use crate::backend::DtypePolicy;
use crate::config::{LmConfig, ModelConfig};
use crate::error::{Error, Result};
use crate::lm::LmModel;
use crate::mimi::Mimi;
use crate::tokenizer::Tokenizer;
use crate::util::{self, RowChunks};

const DEFAULT_REPO: &str = "nvidia/personaplex-7b-v1";
const LM_FILE: &str = "model.safetensors";
const MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const TOK_FILE: &str = "tokenizer_spm_32k_3.model";

#[derive(Clone)]
struct RawTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// CPU-side checkpoint. Device tensors are created from this on demand.
pub struct Weights {
    tensors: HashMap<String, RawTensor>,
    dtype: DType,
}

impl Weights {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn from_file(path: impl AsRef<Path>, dtype: DType) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|err| Error::checkpoint(format!("mmap {}: {err}", path.display())))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|err| Error::checkpoint(format!("{}: {err}", path.display())))?;
        let mut map = HashMap::new();
        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data = decode_tensor(view.dtype(), view.data())
                .map_err(|err| Error::checkpoint(format!("{name}: {err}")))?;
            let expected: usize = shape.iter().product();
            if data.len() != expected {
                return Err(Error::checkpoint(format!(
                    "{name}: decoded {} values, shape {shape:?} wants {expected}",
                    data.len()
                )));
            }
            map.insert(name.to_string(), RawTensor { shape, data });
        }
        Ok(Self {
            tensors: map,
            dtype,
        })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn copy_f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let raw = self
            .tensors
            .get(key)
            .ok_or_else(|| Error::checkpoint(format!("missing tensor {key}")))?;
        Ok((raw.shape.clone(), raw.data.clone()))
    }

    pub fn param1(&mut self, key: &str, device: &Device) -> Result<Param<Tensor<1>>> {
        Ok(Param::from_tensor(self.tensor(key, device)?))
    }

    pub fn param2(&mut self, key: &str, device: &Device) -> Result<Param<Tensor<2>>> {
        Ok(Param::from_tensor(self.tensor(key, device)?))
    }

    pub fn param3(&mut self, key: &str, device: &Device) -> Result<Param<Tensor<3>>> {
        Ok(Param::from_tensor(self.tensor(key, device)?))
    }

    pub fn tensor3(&mut self, key: &str, device: &Device) -> Result<Tensor<3>> {
        self.tensor(key, device)
    }

    pub fn matrix(&mut self, key: &str, device: &Device) -> Result<RowChunks> {
        let tensor = self.tensor::<2>(key, device)?;
        Ok(RowChunks::from_full(tensor))
    }

    fn tensor<const D: usize>(&mut self, key: &str, device: &Device) -> Result<Tensor<D>> {
        let raw = self
            .tensors
            .get(key)
            .ok_or_else(|| Error::checkpoint(format!("missing tensor {key}")))?;
        if raw.shape.len() != D {
            return Err(Error::checkpoint(format!(
                "{key}: rank {} , expected {D}",
                raw.shape.len()
            )));
        }
        let mut shape = [0usize; D];
        shape.copy_from_slice(&raw.shape);
        let tensor = util::f32_tensor(raw.data.clone(), shape, device);
        Ok(match self.dtype {
            DType::F32 => tensor,
            other => tensor.cast(other),
        })
    }

    /// Match `get_moshi_lm`: tile short depformer attention and copy steps 0..7 onto 8..15.
    pub fn expand_depformer(&mut self, cfg: &LmConfig) {
        let dep_q = cfg.dep_q;
        let dim = cfg.depformer_dim;
        let keys: Vec<String> = self.tensors.keys().cloned().collect();
        for key in keys {
            if !(key.contains("depformer") && key.contains("self_attn")) {
                continue;
            }
            let expected_rows = if key.ends_with("in_proj_weight") {
                dep_q * 3 * dim
            } else if key.ends_with("out_proj.weight") {
                dep_q * dim
            } else {
                continue;
            };
            self.tile_rows(&key, expected_rows);
        }
        let groups = ["gating", "linears", "depformer_in", "depformer_emb"];
        let existing: Vec<String> = self.tensors.keys().cloned().collect();
        for key in existing {
            for group in groups {
                for old in 0..8.min(dep_q) {
                    let needle = format!("{group}.{old}.");
                    let Some(pos) = key.find(&needle) else {
                        continue;
                    };
                    let src = self.tensors.get(&key).cloned();
                    let Some(src) = src else { continue };
                    for new in 8..dep_q {
                        let repl = format!("{group}.{new}.");
                        let new_key =
                            format!("{}{repl}{}", &key[..pos], &key[pos + needle.len()..]);
                        self.tensors.entry(new_key).or_insert_with(|| src.clone());
                    }
                }
            }
        }
    }

    fn tile_rows(&mut self, key: &str, expected_rows: usize) {
        let Some(raw) = self.tensors.get(key) else {
            return;
        };
        if raw.shape.is_empty() || raw.shape[0] == 0 || raw.shape[0] >= expected_rows {
            return;
        }
        if !expected_rows.is_multiple_of(raw.shape[0]) {
            return;
        }
        let repeats = expected_rows / raw.shape[0];
        let row_len = raw.data.len() / raw.shape[0];
        let mut data = Vec::with_capacity(raw.data.len() * repeats);
        for _ in 0..repeats {
            data.extend_from_slice(&raw.data);
        }
        let mut shape = raw.shape.clone();
        shape[0] = expected_rows;
        debug_assert_eq!(data.len(), row_len * expected_rows);
        self.tensors
            .insert(key.to_string(), RawTensor { shape, data });
    }
}

fn decode_tensor(dtype: SafeDtype, bytes: &[u8]) -> std::result::Result<Vec<f32>, String> {
    match dtype {
        SafeDtype::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err("f32 length is not a multiple of 4".into());
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
            Ok(out)
        }
        SafeDtype::F16 => decode_u16(bytes, |bits| f16::from_bits(bits).to_f32()),
        SafeDtype::BF16 => decode_u16(bytes, |bits| bf16::from_bits(bits).to_f32()),
        other => Err(format!("unsupported safetensors dtype {other:?}")),
    }
}

fn decode_u16(bytes: &[u8], map: impl Fn(u16) -> f32) -> std::result::Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("16-bit length is not a multiple of 2".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.as_chunks::<2>().0 {
        out.push(map(u16::from_le_bytes(*chunk)));
    }
    Ok(out)
}

/// Where `nvidia/personaplex-7b-v1` is read from.
#[derive(Debug, Clone)]
pub enum WeightSource {
    /// Hugging Face repo. Requires `HF_TOKEN` after the license is accepted.
    HuggingFace { repo: String, revision: String },
    /// Directory with `model.safetensors`, the Mimi safetensors, and the sentencepiece model.
    Local(PathBuf),
}

impl WeightSource {
    pub fn huggingface() -> Self {
        Self::HuggingFace {
            repo: DEFAULT_REPO.into(),
            revision: "main".into(),
        }
    }

    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }
}

struct ModelInner {
    config: ModelConfig,
    mimi: Mimi,
    lm: LmModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

/// Loaded PersonaPlex weights. Cloning is cheap: the tensors are reference-counted.
///
/// Construct a [`crate::session::Session`] from this. The model itself has no
/// transport and no streaming state.
#[derive(Clone)]
pub struct PersonaPlex {
    inner: Arc<ModelInner>,
}

impl PersonaPlex {
    pub fn load(source: WeightSource, device: Device, policy: DtypePolicy) -> Result<Self> {
        let config = ModelConfig::personaplex_7b();
        config.validate()?;
        let dtype = policy.resolve(&device);
        let (lm_path, mimi_path, tok_path) = locate(source)?;
        let mut lm_weights = Weights::from_file(&lm_path, dtype)?;
        lm_weights.expand_depformer(&config.lm);
        let mut mimi_weights = Weights::from_file(&mimi_path, dtype)?;
        let lm = LmModel::from_weights(&config.lm, &mut lm_weights, &device)?;
        let mimi = Mimi::from_weights(&config.mimi, &mut mimi_weights, &device)?;
        let tokenizer = Tokenizer::from_file(&tok_path)?;
        Ok(Self {
            inner: Arc::new(ModelInner {
                config,
                mimi,
                lm,
                tokenizer,
                device,
                dtype,
            }),
        })
    }

    /// Test and embedding helper. The caller already built the modules.
    pub fn from_parts(
        config: ModelConfig,
        mimi: Mimi,
        lm: LmModel,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            inner: Arc::new(ModelInner {
                config,
                mimi,
                lm,
                tokenizer,
                device,
                dtype,
            }),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.inner.config
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn weight_dtype(&self) -> DType {
        self.inner.dtype
    }

    pub(crate) fn mimi(&self) -> &Mimi {
        &self.inner.mimi
    }

    pub(crate) fn lm(&self) -> &LmModel {
        &self.inner.lm
    }

    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.inner.tokenizer
    }
}

fn locate(source: WeightSource) -> Result<(PathBuf, PathBuf, PathBuf)> {
    match source {
        WeightSource::Local(dir) => {
            let lm = dir.join(LM_FILE);
            let mimi = dir.join(MIMI_FILE);
            let tok = dir.join(TOK_FILE);
            for path in [&lm, &mimi, &tok] {
                if !path.is_file() {
                    return Err(Error::checkpoint(format!(
                        "missing {} in {}",
                        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
                        dir.display()
                    )));
                }
            }
            Ok((lm, mimi, tok))
        }
        WeightSource::HuggingFace { repo, revision } => {
            let token = std::env::var("HF_TOKEN")
                .ok()
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    std::env::var("HUGGING_FACE_HUB_TOKEN")
                        .ok()
                        .filter(|s| !s.is_empty())
                });
            if token.is_none() {
                return Err(Error::gated(
                    "HF_TOKEN is not set, so the gated checkpoint cannot be downloaded",
                ));
            }
            let api = hf_hub::api::sync::ApiBuilder::new()
                .with_token(token)
                .build()
                .map_err(|err| map_hub(err.to_string()))?;
            let handle = api.repo(hf_hub::Repo::with_revision(
                repo,
                hf_hub::RepoType::Model,
                revision,
            ));
            Ok((
                fetch(&handle, LM_FILE)?,
                fetch(&handle, MIMI_FILE)?,
                fetch(&handle, TOK_FILE)?,
            ))
        }
    }
}

fn fetch(repo: &hf_hub::api::sync::ApiRepo, file: &str) -> Result<PathBuf> {
    repo.get(file).map_err(|err| map_hub(err.to_string()))
}

fn map_hub(message: String) -> Error {
    if Error::is_gated(&message) {
        Error::gated(message)
    } else {
        Error::Hub(message)
    }
}
