use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
pub struct RealFft {
    fft: Arc<dyn Fft<f32>>,
    scratch_len: usize,
}
impl RealFft {
    pub fn new(length: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(length);
        let scratch_len = fft.get_inplace_scratch_len();
        Self { fft, scratch_len }
    }

    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    pub fn process_with_scratch(
        &self,
        input: &[f32],
        output: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) {
        assert_eq!(input.len(), output.len());
        for (i, &val) in input.iter().enumerate() {
            output[i] = Complex { re: val, im: 0.0 };
        }
        self.fft.process_with_scratch(output, scratch);
    }

    pub fn process(&self, input: &[f32], output: &mut [Complex<f32>]) {
        let mut scratch = vec![Complex { re: 0.0, im: 0.0 }; self.scratch_len];
        self.process_with_scratch(input, output, &mut scratch);
    }
}
