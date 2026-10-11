//! Small tensor helpers shared by the model modules.
//!
//! Linear and embedding weights are stored as [`RowChunks`] so a single buffer
//! stays under WebGPU's guaranteed 128 MiB storage-buffer binding.

use burn::module::Param;
use burn::tensor::module::embedding;
use burn::tensor::{DType, Device, Int, Tensor, TensorData};

use crate::backend::MAX_WEIGHT_BUFFER_BYTES;

pub fn f32_tensor<const D: usize>(data: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::<D>::from_data(TensorData::new(data, shape), device)
}

pub fn i32_tensor<const D: usize>(
    data: Vec<i32>,
    shape: [usize; D],
    device: &Device,
) -> Tensor<D, Int> {
    Tensor::<D, Int>::from_data(TensorData::new(data, shape), device)
}

pub fn zeros<const D: usize>(shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::<D>::zeros(shape, device)
}

/// PyTorch-layout linear: `y = x W^T` with `weight` stored `[out, in]`.
pub fn linear(x: Tensor<3>, weight_out_in: Tensor<2>) -> Tensor<3> {
    let dtype = weight_out_in.dtype();
    let x = x.cast(dtype);
    let [b, t, c] = x.dims();
    let out = weight_out_in.dims()[0];
    let flat = x.reshape([b * t, c]);
    let y = flat.matmul(weight_out_in.transpose());
    y.reshape([b, t, out])
}

/// How many rows of a `[rows, cols]` matrix fit in one WebGPU-sized buffer.
pub fn rows_per_buffer(cols: usize, elem_size: usize) -> usize {
    let row_bytes = cols.saturating_mul(elem_size).max(1);
    (MAX_WEIGHT_BUFFER_BYTES / row_bytes).max(1)
}

pub fn elem_size(dtype: DType) -> usize {
    match dtype {
        DType::F16 | DType::BF16 => 2,
        DType::F64 => 8,
        _ => 4,
    }
}

/// Matrix stored in row chunks of at most [`MAX_WEIGHT_BUFFER_BYTES`].
#[derive(Clone)]
pub struct RowChunks {
    chunks: Vec<Param<Tensor<2>>>,
    rows_per_chunk: usize,
    total_rows: usize,
}

impl RowChunks {
    pub fn from_tensor(weight: Tensor<2>, max_bytes: usize) -> Self {
        let [rows, cols] = weight.dims();
        let elem = elem_size(weight.dtype());
        let row_bytes = cols.saturating_mul(elem).max(1);
        let rows_per_chunk = (max_bytes / row_bytes).max(1).min(rows.max(1));
        if rows <= rows_per_chunk {
            return Self {
                chunks: vec![Param::from_tensor(weight)],
                rows_per_chunk: rows.max(1),
                total_rows: rows,
            };
        }
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < rows {
            let len = (rows - start).min(rows_per_chunk);
            chunks.push(Param::from_tensor(weight.clone().narrow(0, start, len)));
            start += len;
        }
        Self {
            chunks,
            rows_per_chunk,
            total_rows: rows,
        }
    }

    pub fn from_full(weight: Tensor<2>) -> Self {
        Self::from_tensor(weight, MAX_WEIGHT_BUFFER_BYTES)
    }

    pub fn rows(&self) -> usize {
        self.total_rows
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// `y = x W^T` with `W` the concatenation of the chunks along the output rows.
    pub fn matmul(&self, x: Tensor<3>) -> Tensor<3> {
        if self.chunks.len() == 1 {
            return linear(x, self.chunks[0].val());
        }
        let parts: Vec<Tensor<3>> = self
            .chunks
            .iter()
            .map(|chunk| linear(x.clone(), chunk.val()))
            .collect();
        Tensor::cat(parts, 2)
    }

    /// Contiguous output-row slice. Used by depth-transformer per-step weights.
    pub fn slice_rows(&self, start: usize, len: usize) -> Tensor<2> {
        if len == 0 || self.chunks.is_empty() {
            return self.chunks[0].val().narrow(0, 0, 0);
        }
        let mut parts = Vec::new();
        let mut remaining = len;
        let mut offset = start;
        while remaining > 0 {
            let chunk_index = offset / self.rows_per_chunk;
            let local = offset % self.rows_per_chunk;
            let chunk = &self.chunks[chunk_index];
            let available = chunk.val().dims()[0] - local;
            let take = remaining.min(available);
            parts.push(chunk.val().narrow(0, local, take));
            offset += take;
            remaining -= take;
        }
        if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            Tensor::cat(parts, 0)
        }
    }

    /// Embedding lookup. `zero_idx` (the reference uses -1) yields an all-zero vector.
    pub fn embed(&self, indices: Tensor<2, Int>, zero_idx: i32) -> Tensor<3> {
        let dim = self.chunks[0].val().dims()[1];
        let is_zero = indices.clone().equal_elem(zero_idx);
        let last = (self.total_rows as i32 - 1).max(0);
        let indices = indices.clamp(0, last);
        let looked = if self.chunks.len() == 1 {
            embedding(self.chunks[0].val(), indices)
        } else {
            let rows = self.rows_per_chunk as i32;
            let mut acc: Option<Tensor<3>> = None;
            for (chunk_index, chunk) in self.chunks.iter().enumerate() {
                let start = chunk_index as i32 * rows;
                let nrows = chunk.val().dims()[0] as i32;
                let in_range = indices
                    .clone()
                    .greater_equal_elem(start)
                    .bool_and(indices.clone().lower_elem(start + nrows));
                let local = indices.clone().sub_scalar(start).clamp(0, nrows - 1);
                let part = embedding(chunk.val(), local);
                let keep = in_range.float().unsqueeze_dim::<3>(2).repeat(&[1, 1, dim]);
                let part = part * keep;
                acc = Some(match acc {
                    None => part,
                    Some(prev) => prev + part,
                });
            }
            acc.expect("embedding chunks")
        };
        let drop = is_zero.float().unsqueeze_dim::<3>(2).repeat(&[1, 1, dim]);
        let mask = drop.greater_elem(0.5);
        looked.mask_fill(mask, 0.0)
    }
}

pub fn tensor_to_f32_vec(tensor: Tensor<1>) -> Vec<f32> {
    tensor
        .cast(DType::F32)
        .try_into_vec_as::<f32>()
        .expect("f32 tensor read")
}

pub fn scalar_i32(tensor: Tensor<1, Int>) -> i32 {
    tensor.into_scalar()
}
