use lele::kernels::{math, neon, quantization};
use lele::tensor::TensorView;
use std::borrow::Cow;
use std::time::Instant;

fn main() {
    let size = 1000 * 1024; // 1M elements
    let input_data = vec![0.5f32; size];
    let input = TensorView {
        data: Cow::Borrowed(&input_data),
        shape: Cow::Owned(vec![size]),
    };

    println!("Benchmarking with {} elements on Apple M2", size);
    println!("--------------------------------------------------");

    // 1. ReLU
    {
        let mut out_scalar = Vec::new();
        let mut out_neon = Vec::new();

        // Warmup
        let _ = math::relu(&input, &mut out_scalar);
        let _ = neon::math::relu(&input, &mut out_neon);

        let start = Instant::now();
        for _ in 0..100 {
            let _ = math::relu(&input, &mut out_scalar);
        }
        let scalar_time = start.elapsed().as_micros() as f64 / 100.0;

        let start = Instant::now();
        for _ in 0..100 {
            let _ = neon::math::relu(&input, &mut out_neon);
        }
        let neon_time = start.elapsed().as_micros() as f64 / 100.0;

        println!(
            "ReLU: Scalar: {:.2}us, NEON: {:.2}us, Speedup: {:.2}x",
            scalar_time,
            neon_time,
            scalar_time / neon_time
        );
    }

    // 2. Sigmoid
    {
        let mut out_scalar = Vec::new();
        let mut out_neon = Vec::new();

        let start = Instant::now();
        for _ in 0..100 {
            // Manual scalar loop for benchmark if not in kernels
            let len = input.data.len();
            lele::kernels::utils::ensure_capacity(&mut out_scalar, len);
            unsafe {
                out_scalar.set_len(len);
            }
            for i in 0..len {
                out_scalar[i] = 1.0 / (1.0 + (-input.data[i]).exp());
            }
        }
        let scalar_time = start.elapsed().as_micros() as f64 / 100.0;

        let start = Instant::now();
        for _ in 0..100 {
            let _ = neon::math::sigmoid(&input, &mut out_neon);
        }
        let neon_time = start.elapsed().as_micros() as f64 / 100.0;

        println!(
            "Sigmoid: Scalar: {:.2}us, NEON: {:.2}us, Speedup: {:.2}x",
            scalar_time,
            neon_time,
            scalar_time / neon_time
        );
    }

    // 3. LayerNorm
    {
        let norm_size = 512;
        let outer = size / norm_size;
        let ln_input = TensorView {
            data: Cow::Borrowed(&input_data),
            shape: Cow::Owned(vec![outer, norm_size]),
        };
        let gamma = vec![1.0f32; norm_size];
        let beta = vec![0.0f32; norm_size];
        let gamma_view = TensorView {
            data: Cow::Borrowed(&gamma),
            shape: Cow::Owned(vec![norm_size]),
        };
        let beta_view = TensorView {
            data: Cow::Borrowed(&beta),
            shape: Cow::Owned(vec![norm_size]),
        };

        let mut out_scalar = Vec::new();
        let mut out_neon = Vec::new();

        let start = Instant::now();
        for _ in 0..20 {
            let _ = lele::kernels::norm::layer_norm(
                &ln_input,
                &gamma_view,
                &beta_view,
                -1,
                1e-5,
                &mut out_scalar,
            );
        }
        let scalar_time = start.elapsed().as_micros() as f64 / 20.0;

        let start = Instant::now();
        for _ in 0..20 {
            let _ = neon::normalization::layer_norm(
                &ln_input,
                &gamma_view,
                &beta_view,
                -1,
                1e-5,
                &mut out_neon,
            );
        }
        let neon_time = start.elapsed().as_micros() as f64 / 20.0;

        println!(
            "LayerNorm: Scalar: {:.2}us, NEON: {:.2}us, Speedup: {:.2}x",
            scalar_time,
            neon_time,
            scalar_time / neon_time
        );
    }

    // 4. DynamicQuantizeLinear
    {
        let mut out_y_s = Vec::new();
        let mut out_s_s = Vec::new();
        let mut out_z_s = Vec::new();

        let mut out_y_n = Vec::new();
        let mut out_s_n = Vec::new();
        let mut out_z_n = Vec::new();

        let start = Instant::now();
        for _ in 0..50 {
            let _ = quantization::dynamic_quantize_linear(
                &input,
                &mut out_y_s,
                &mut out_s_s,
                &mut out_z_s,
            );
        }
        let scalar_time = start.elapsed().as_micros() as f64 / 50.0;

        let start = Instant::now();
        for _ in 0..50 {
            let _ = neon::quantization::dynamic_quantize_linear(
                &input,
                &mut out_y_n,
                &mut out_s_n,
                &mut out_z_n,
            );
        }
        let neon_time = start.elapsed().as_micros() as f64 / 50.0;

        println!(
            "DynamicQuantize: Scalar: {:.2}us, NEON: {:.2}us, Speedup: {:.2}x",
            scalar_time,
            neon_time,
            scalar_time / neon_time
        );
    }

    // 5. Swish
    {
        let mut out_scalar = Vec::new();
        let mut out_neon = Vec::new();

        let start = Instant::now();
        for _ in 0..50 {
            let len = input.data.len();
            lele::kernels::utils::ensure_capacity(&mut out_scalar, len);
            unsafe {
                out_scalar.set_len(len);
            }
            for i in 0..len {
                let x = input.data[i];
                out_scalar[i] = x * (1.0 / (1.0 + (-x).exp()));
            }
        }
        let scalar_time = start.elapsed().as_micros() as f64 / 50.0;

        let start = Instant::now();
        for _ in 0..50 {
            let _ = neon::math::swish(&input, &mut out_neon);
        }
        let neon_time = start.elapsed().as_micros() as f64 / 50.0;

        println!(
            "Swish: Scalar: {:.2}us, NEON: {:.2}us, Speedup: {:.2}x",
            scalar_time,
            neon_time,
            scalar_time / neon_time
        );
    }
}
