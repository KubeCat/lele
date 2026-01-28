use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;
use std::simd::prelude::*;

pub fn layer_norm<'b, 'a>(
    input: &TensorView<'b>,
    scale: &TensorView<'b>,
    bias: &TensorView<'b>,
    axis: i32,
    epsilon: f32,
    out_buf: &'a mut Vec<f32>,
) -> TensorView<'a> {
    let ndim = input.shape.len();
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;
    let outer_dims = &input.shape[..axis];
    let norm_dims = &input.shape[axis..];
    let outer_size: usize = outer_dims.iter().product();
    let norm_size: usize = norm_dims.iter().product();
    utils::ensure_capacity(out_buf, input.data.len());
    unsafe {
        out_buf.set_len(input.data.len());
    }
    let out_slice = out_buf.as_mut_slice();
    let src = &input.data;
    let gamma = &scale.data;
    let beta = &bias.data;
    let inv_norm_size = 1.0 / (norm_size as f32);
    for i in 0..outer_size {
        let start = i * norm_size;
        let end = start + norm_size;
        let row = &src[start..end];
        let out_row = &mut out_slice[start..end];
        let mut sum_vec = f32x4::splat(0.0);
        let mut sum_sq_vec = f32x4::splat(0.0);
        let (prefix, middle, _suffix) = row.as_simd::<4>();
        for chunk in middle {
            sum_vec += *chunk;
            sum_sq_vec += (*chunk) * (*chunk);
        }
        let mut sum = sum_vec.reduce_sum();
        let mut sum_sq = sum_sq_vec.reduce_sum();
        for &x in prefix {
            sum += x;
            sum_sq += x * x;
        }
        for &x in _suffix {
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum * inv_norm_size;
        let var = (sum_sq * inv_norm_size) - (mean * mean);
        let inv_std = 1.0 / (var + epsilon).sqrt();
        for j in 0..norm_size {
            out_row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
    TensorView {
        data: Cow::Borrowed(out_slice),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
