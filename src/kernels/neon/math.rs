use crate::kernels::utils;
use crate::tensor::TensorView;
use std::borrow::Cow;
use std::simd::prelude::*;

pub fn relu<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();
    let zero = f32x4::splat(0.0);
    for i in 0..prefix.len() {
        out_slice[i] = input.data[i].max(0.0);
    }
    let middle_out = &mut out_slice[prefix.len()..prefix.len() + middle.len() * 4];
    let (_, middle_out_simd, _) = middle_out.as_simd_mut::<4>();
    for i in 0..middle.len() {
        middle_out_simd[i] = middle[i].simd_max(zero);
    }
    let offset = prefix.len() + middle.len() * 4;
    for i in offset..len {
        out_slice[i] = input.data[i].max(0.0);
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn sigmoid<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();
    let one = f32x4::splat(1.0);
    for i in 0..prefix.len() {
        out_slice[i] = 1.0 / (1.0 + (-input.data[i]).exp());
    }
    let offset_mid = prefix.len();
    for i in 0..middle.len() {
        let x = middle[i];
        let neg_x = f32x4::splat(0.0) - x;
        let mut e_arr = [0.0; 4];
        let neg_x_arr: [f32; 4] = neg_x.into();
        for j in 0..4 {
            e_arr[j] = neg_x_arr[j].exp();
        }
        let e = f32x4::from_array(e_arr);
        let y = one / (one + e);
        y.copy_to_slice(&mut out_slice[offset_mid + i * 4..]);
    }
    let offset_suf = prefix.len() + middle.len() * 4;
    for i in offset_suf..len {
        out_slice[i] = 1.0 / (1.0 + (-input.data[i]).exp());
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}

pub fn swish<'a>(input: &TensorView<'_>, output_buf: &'a mut Vec<f32>) -> TensorView<'a> {
    let len = input.data.len();
    utils::ensure_capacity(output_buf, len);
    unsafe {
        output_buf.set_len(len);
    }
    let (prefix, middle, _suffix) = input.data.as_simd::<4>();
    let out_slice = output_buf.as_mut_slice();
    let one = f32x4::splat(1.0);
    for i in 0..prefix.len() {
        let x = input.data[i];
        out_slice[i] = x * (1.0 / (1.0 + (-x).exp()));
    }
    let offset_mid = prefix.len();
    for i in 0..middle.len() {
        let x = middle[i];
        let neg_x = f32x4::splat(0.0) - x;
        let mut e_arr = [0.0; 4];
        let neg_x_arr: [f32; 4] = neg_x.into();
        for j in 0..4 {
            e_arr[j] = neg_x_arr[j].exp();
        }
        let e = f32x4::from_array(e_arr);
        let y = x * (one / (one + e));
        y.copy_to_slice(&mut out_slice[offset_mid + i * 4..]);
    }
    let offset_suf = prefix.len() + middle.len() * 4;
    for i in offset_suf..len {
        let x = input.data[i];
        out_slice[i] = x * (1.0 / (1.0 + (-x).exp()));
    }
    TensorView {
        data: Cow::Borrowed(output_buf),
        shape: Cow::Owned(input.shape.to_vec()),
    }
}
