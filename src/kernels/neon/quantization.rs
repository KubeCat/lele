use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;
use std::simd::prelude::*;
use std::simd::StdFloat;

pub fn dynamic_quantize_linear<'a, 'b>(
    x: &TensorView<'b>,
    out_y: &'a mut Vec<f32>,
    out_scale: &'a mut Vec<f32>,
    out_zp: &'a mut Vec<f32>,
) -> (TensorView<'a>, TensorView<'a>, TensorView<'a>) {
    let len = x.data.len();
    if len == 0 {
        return (
            TensorView {
                data: Cow::Borrowed(out_y),
                shape: Cow::Owned(x.shape.to_vec()),
            },
            TensorView {
                data: Cow::Borrowed(out_scale),
                shape: Cow::Owned(vec![1]),
            },
            TensorView {
                data: Cow::Borrowed(out_zp),
                shape: Cow::Owned(vec![1]),
            },
        );
    }
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut i = 0;
    if len >= 4 {
        let mut min_vec = f32x4::splat(f32::MAX);
        let mut max_vec = f32x4::splat(f32::MIN);
        while i + 4 <= len {
            let vx = f32x4::from_slice(&x.data[i..i + 4]);
            min_vec = min_vec.simd_min(vx);
            max_vec = max_vec.simd_max(vx);
            i += 4;
        }
        min_val = min_vec.reduce_min();
        max_val = max_vec.reduce_max();
    }
    while i < len {
        let v = x.data[i];
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
        i += 1;
    }
    let adjusted_max = max_val.max(0.0);
    let adjusted_min = min_val.min(0.0);
    let range = (adjusted_max - adjusted_min).max(1e-5);
    let scale = range / 255.0;
    let zp = (-adjusted_min / scale).round().clamp(0.0, 255.0);
    utils::ensure_capacity(out_scale, 1);
    unsafe {
        out_scale.set_len(1);
    }
    out_scale[0] = scale;
    utils::ensure_capacity(out_zp, 1);
    unsafe {
        out_zp.set_len(1);
    }
    out_zp[0] = zp;
    utils::ensure_capacity(out_y, len);
    unsafe {
        out_y.set_len(len);
    }
    let inv_scale = 1.0 / scale;
    let inv_scale_vec = f32x4::splat(inv_scale);
    let zp_vec = f32x4::splat(zp);
    let zero = f32x4::splat(0.0);
    let two_five_five = f32x4::splat(255.0);
    let x_data = &x.data;
    let y_data = out_y.as_mut_slice();
    let mut i = 0;
    if len >= 4 {
        while i + 4 <= len {
            let vx = f32x4::from_slice(&x_data[i..i + 4]);
            let vy = (vx * inv_scale_vec + zp_vec)
                .round()
                .simd_clamp(zero, two_five_five);
            vy.copy_to_slice(&mut y_data[i..i + 4]);
            i += 4;
        }
    }
    while i < len {
        y_data[i] = (x_data[i] * inv_scale + zp).round().clamp(0.0, 255.0);
        i += 1;
    }
    (
        TensorView {
            data: Cow::Borrowed(out_y),
            shape: Cow::Owned(x.shape.to_vec()),
        },
        TensorView {
            data: Cow::Borrowed(out_scale),
            shape: Cow::Owned(vec![1]),
        },
        TensorView {
            data: Cow::Borrowed(out_zp),
            shape: Cow::Owned(vec![1]),
        },
    )
}
