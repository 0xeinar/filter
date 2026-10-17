use crate::FilterType;

/// Second-order IIR (biquad) filter for EQ processing
///
/// Supports Bell, Low Shelf, High Shelf, Low Cut, High Cut
/// with configurable slope for cut filters (6/12/24/48 dB/oct)
pub struct BiquadFilter {
    /// Numerator coefficients (b0, b1, b2)
    b0: f32,
    b1: f32,
    b2: f32,
    /// Denominator coefficients (a1, a2) — a0 is normalized to 1
    a1: f32,
    a2: f32,
    /// State variables (direct form II transposed)
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    pub fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Update the filter coefficients for the given parameters
    pub fn update_coefficients(
        &mut self,
        freq: f32,
        gain_db: f32,
        q: f32,
        filter_type: FilterType,
        sample_rate: f32,
    ) {
        let a = 10.0_f32.powf(gain_db / 40.0); // Square root of linear gain
        let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();

        // Alpha = sin(w0) / (2 * Q)
        let alpha = sin_w0 / (2.0 * q);

        match filter_type {
            FilterType::Bell => {
                // Peaking EQ
                let a_minus_1 = a - 1.0;
                let a_plus_1 = a + 1.0;

                let b0 = 1.0 + alpha * a_minus_1;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a_minus_1;
                let a0 = 1.0 + alpha / a;  // Normalization
                let a1_coeff = -2.0 * cos_w0;
                let a2_coeff = 1.0 - alpha / a;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1_coeff / a0;
                self.a2 = a2_coeff / a0;
            }
            FilterType::LowShelf => {
                // Low shelf EQ
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1_coeff = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2_coeff = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1_coeff / a0;
                self.a2 = a2_coeff / a0;
            }
            FilterType::HighShelf => {
                // High shelf EQ
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1_coeff = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2_coeff = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1_coeff / a0;
                self.a2 = a2_coeff / a0;
            }
            FilterType::LowCut => {
                // High-pass filter (low cut)
                // For now, 12 dB/oct (2nd order). Multi-slope support added later.
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1_coeff = -2.0 * cos_w0;
                let a2_coeff = 1.0 - alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1_coeff / a0;
                self.a2 = a2_coeff / a0;
            }
            FilterType::HighCut => {
                // Low-pass filter (high cut)
                // For now, 12 dB/oct (2nd order). Multi-slope support added later.
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1_coeff = -2.0 * cos_w0;
                let a2_coeff = 1.0 - alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1_coeff / a0;
                self.a2 = a2_coeff / a0;
            }
        }
    }

    /// Process a single sample through the filter (Direct Form II Transposed)
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}
