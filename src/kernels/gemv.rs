use std::simd::prelude::*;

// Specialized GEMV for [M, K] * [K, 1] -> [M, 1]
// This computes y = A * x where A is [M, K] row-major and x is [K]
// Optimized for contiguous memory access of A's rows (Dot Product)
pub fn sgemv_mv(
    m: usize,
    k: usize,
    a: &[f32],       // [M, K]
    x: &[f32],       // [K]
    out: &mut [f32], // [M]
) {
    const LANES: usize = 8;

    for i in 0..m {
        let row_offset = i * k;
        let row_a = &a[row_offset..row_offset + k];

        let mut sum_v = Simd::<f32, LANES>::splat(0.0);
        let mut j = 0;
        while j + LANES <= k {
            let val_a = Simd::<f32, LANES>::from_slice(&row_a[j..j + LANES]);
            let val_x = Simd::<f32, LANES>::from_slice(&x[j..j + LANES]);
            sum_v += val_a * val_x;
            j += LANES;
        }

        let mut sum = sum_v.reduce_sum();

        while j < k {
            unsafe {
                sum += *row_a.get_unchecked(j) * *x.get_unchecked(j);
            }
            j += 1;
        }

        unsafe {
            *out.get_unchecked_mut(i) = sum;
        }
    }
}
