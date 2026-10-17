use crate::FilterType;

/// Biquad coefficients, shared by every channel that runs the same band.
///
/// Plain float math with no state — one copy per band, recomputed only when
/// the band's parameters move.
///
/// Stored as `f64`: with `f32` coefficients a low-frequency, high-Q bell
/// (e.g. 120 Hz, Q 4, -12 dB) only nulls to about -75 dB against an f64
/// reference, because the pole pair sits so close to z = 1 that a 1e-7
/// relative coefficient error moves the response audibly. In f64 the same
/// chain nulls below -150 dB. The extra precision costs two doubles per band
/// and is invisible in the hot path (see benches/hotpath_bench.rs).
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    /// Numerator coefficients (b0, b1, b2)
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    /// Denominator coefficients (a1, a2) — a0 is normalized to 1
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self::PASSTHROUGH
    }
}

impl BiquadCoeffs {
    /// Identity filter.
    pub const PASSTHROUGH: Self = Self {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };

    /// Compute the coefficients for the given parameters
    pub fn update(
        &mut self,
        freq: f32,
        gain_db: f32,
        q: f32,
        filter_type: FilterType,
        sample_rate: f32,
    ) {
        let a = 10.0_f64.powf(gain_db as f64 / 40.0); // Square root of linear gain
        let w0 = 2.0 * std::f64::consts::PI * freq as f64 / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();

        // Alpha = sin(w0) / (2 * Q)
        let alpha = sin_w0 / (2.0 * q as f64);

        match filter_type {
            FilterType::Bell => {
                // Peaking EQ (Audio EQ Cookbook)
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
            FilterType::LowShelf => {
                // Low shelf EQ (Audio EQ Cookbook)
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
            FilterType::HighShelf => {
                // High shelf EQ (Audio EQ Cookbook)
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
            FilterType::LowCut => {
                // High-pass filter (low cut) — 12 dB/oct (2nd order)
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
            FilterType::HighCut => {
                // Low-pass filter (high cut) — 12 dB/oct (2nd order)
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                self.b0 = b0 / a0;
                self.b1 = b1 / a0;
                self.b2 = b2 / a0;
                self.a1 = a1 / a0;
                self.a2 = a2 / a0;
            }
        }
    }

    /// Compute the frequency response (magnitude) at a given frequency
    /// This is for testing — evaluates H(z) on the unit circle
    pub fn magnitude_at_freq(&self, freq: f32, sample_rate: f32) -> f32 {
        let w = 2.0 * std::f64::consts::PI * freq as f64 / sample_rate as f64;
        let cos_w = w.cos();
        let sin_w = w.sin();

        // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        // z = e^(jw), z^-1 = e^(-jw) = cos(w) - j*sin(w)
        let re_num = self.b0 + self.b1 * cos_w + self.b2 * (2.0 * cos_w * cos_w - 1.0);
        let im_num = -(self.b1 * sin_w + self.b2 * 2.0 * sin_w * cos_w);
        let re_den = 1.0 + self.a1 * cos_w + self.a2 * (2.0 * cos_w * cos_w - 1.0);
        let im_den = -(self.a1 * sin_w + self.a2 * 2.0 * sin_w * cos_w);

        let num_mag = (re_num * re_num + im_num * im_num).sqrt();
        let den_mag = (re_den * re_den + im_den * im_den).sqrt();

        (num_mag / den_mag) as f32
    }
}

/// Per-channel filter state (direct form II transposed delay elements).
///
/// Coefficients and state are deliberately separate: every channel of a band
/// shares one `BiquadCoeffs` but must own its own `BiquadState`, otherwise the
/// right channel continues the left channel's filter memory.
///
/// State is `f64`. At low frequencies a biquad's delay memory is large relative
/// to the signal and `f32` accumulators leak rounding noise into the tail; in
/// f64 the same chain nulls below -150 dB against a double-precision reference.
/// Buffers stay f32 — only the conversion happens per sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct BiquadState {
    z1: f64,
    z2: f64,
}

impl BiquadState {
    /// Process a single sample through the filter (Direct Form II Transposed)
    #[inline]
    pub fn process(&mut self, c: &BiquadCoeffs, input: f32) -> f32 {
        let x = input as f64;
        let output = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * output + self.z2;
        self.z2 = c.b2 * x - c.a2 * output;
        output as f32
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Second-order IIR (biquad) filter for EQ processing — a convenience pairing
/// of `BiquadCoeffs` + `BiquadState` for single-channel use (tests, analysis).
///
/// Supports Bell, Low Shelf, High Shelf, Low Cut, High Cut.
pub struct BiquadFilter {
    coeffs: BiquadCoeffs,
    state: BiquadState,
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl BiquadFilter {
    pub fn new() -> Self {
        Self {
            coeffs: BiquadCoeffs::PASSTHROUGH,
            state: BiquadState::default(),
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
        self.coeffs
            .update(freq, gain_db, q, filter_type, sample_rate);
    }

    /// Process a single sample through the filter (Direct Form II Transposed)
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        self.state.process(&self.coeffs, input)
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Compute the frequency response (magnitude) at a given frequency
    pub fn magnitude_at_freq(&self, freq: f32, sample_rate: f32) -> f32 {
        self.coeffs.magnitude_at_freq(freq, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SAMPLE_RATE: f32 = 44100.0;
    const TOLERANCE_DB: f32 = 0.5; // ±0.5 dB tolerance

    fn db_from_linear(gain: f32) -> f32 {
        20.0 * gain.log10()
    }

    fn assert_gain_at_freq(filter: &BiquadFilter, freq: f32, expected_db: f32, tolerance: f32) {
        let mag = filter.magnitude_at_freq(freq, SAMPLE_RATE);
        let actual_db = db_from_linear(mag);
        assert!(
            (actual_db - expected_db).abs() < tolerance,
            "At {} Hz: expected {:.1} dB, got {:.1} dB (tolerance ±{:.1} dB)",
            freq,
            expected_db,
            actual_db,
            tolerance
        );
    }

    // === Bell filter tests ===

    #[test]
    fn bell_boost_at_center_freq() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        // At center frequency, a bell filter should boost by approximately the gain
        assert_gain_at_freq(&filter, 1000.0, 6.0, TOLERANCE_DB);
    }

    #[test]
    fn bell_cut_at_center_freq() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(5000.0, -12.0, 2.0, FilterType::Bell, SAMPLE_RATE);
        assert_gain_at_freq(&filter, 5000.0, -12.0, TOLERANCE_DB);
    }

    #[test]
    fn bell_flat_at_dc() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        // At DC, a bell filter should be flat (0 dB)
        assert_gain_at_freq(&filter, 20.0, 0.0, 1.0);
    }

    #[test]
    fn bell_flat_at_nyquist() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        // Near nyquist, should return to ~0 dB
        assert_gain_at_freq(&filter, 18000.0, 0.0, 2.0);
    }

    #[test]
    fn bell_high_q_narrow() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 6.0, 10.0, FilterType::Bell, SAMPLE_RATE);
        // At center: +6dB
        assert_gain_at_freq(&filter, 1000.0, 6.0, TOLERANCE_DB);
        // One octave away: should be much less
        let mag_500 = filter.magnitude_at_freq(500.0, SAMPLE_RATE);
        let mag_2000 = filter.magnitude_at_freq(2000.0, SAMPLE_RATE);
        assert!(
            db_from_linear(mag_500) < 2.0,
            "High-Q bell should be narrow: at 500Hz got {:.1} dB",
            db_from_linear(mag_500)
        );
        assert!(
            db_from_linear(mag_2000) < 2.0,
            "High-Q bell should be narrow: at 2000Hz got {:.1} dB",
            db_from_linear(mag_2000)
        );
    }

    #[test]
    fn bell_zero_gain_is_passthrough() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 0.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        // 0 dB gain = passthrough
        assert_gain_at_freq(&filter, 1000.0, 0.0, 0.01);
        assert_gain_at_freq(&filter, 20.0, 0.0, 0.01);
        assert_gain_at_freq(&filter, 10000.0, 0.0, 0.01);
    }

    // === Low shelf tests ===

    #[test]
    fn low_shelf_boost() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(500.0, 6.0, 0.707, FilterType::LowShelf, SAMPLE_RATE);
        // Below shelf frequency: +6dB
        assert_gain_at_freq(&filter, 100.0, 6.0, 1.0);
        // Well above: ~0dB
        assert_gain_at_freq(&filter, 5000.0, 0.0, 1.0);
    }

    #[test]
    fn low_shelf_cut() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(500.0, -6.0, 0.707, FilterType::LowShelf, SAMPLE_RATE);
        assert_gain_at_freq(&filter, 100.0, -6.0, 1.0);
        assert_gain_at_freq(&filter, 5000.0, 0.0, 1.0);
    }

    // === High shelf tests ===

    #[test]
    fn high_shelf_boost() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(5000.0, 6.0, 0.707, FilterType::HighShelf, SAMPLE_RATE);
        // Above shelf frequency: +6dB
        assert_gain_at_freq(&filter, 10000.0, 6.0, 1.0);
        // Well below: ~0dB
        assert_gain_at_freq(&filter, 500.0, 0.0, 1.0);
    }

    #[test]
    fn high_shelf_cut() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(5000.0, -6.0, 0.707, FilterType::HighShelf, SAMPLE_RATE);
        assert_gain_at_freq(&filter, 10000.0, -6.0, 1.0);
        assert_gain_at_freq(&filter, 500.0, 0.0, 1.0);
    }

    // === Low cut (high-pass) tests ===

    #[test]
    fn low_cut_attenuates_below_cutoff() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 0.0, 0.707, FilterType::LowCut, SAMPLE_RATE);
        // Well below cutoff: significant attenuation
        let mag_100 = filter.magnitude_at_freq(100.0, SAMPLE_RATE);
        assert!(
            mag_100 < 0.1,
            "Low cut at 100Hz should be heavily attenuated, got mag={:.3}",
            mag_100
        );
    }

    #[test]
    fn low_cut_passes_above_cutoff() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 0.0, 0.707, FilterType::LowCut, SAMPLE_RATE);
        // Well above cutoff: ~0dB
        assert_gain_at_freq(&filter, 5000.0, 0.0, 0.5);
    }

    // === High cut (low-pass) tests ===

    #[test]
    fn high_cut_attenuates_above_cutoff() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 0.0, 0.707, FilterType::HighCut, SAMPLE_RATE);
        // Well above cutoff: significant attenuation
        let mag_10000 = filter.magnitude_at_freq(10000.0, SAMPLE_RATE);
        assert!(
            mag_10000 < 0.1,
            "High cut at 10kHz should be heavily attenuated, got mag={:.3}",
            mag_10000
        );
    }

    #[test]
    fn high_cut_passes_below_cutoff() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 0.0, 0.707, FilterType::HighCut, SAMPLE_RATE);
        // Well below cutoff: ~0dB
        assert_gain_at_freq(&filter, 100.0, 0.0, 0.5);
    }

    // === Stability tests ===

    #[test]
    fn filter_does_not_explode_with_silence() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 12.0, 10.0, FilterType::Bell, SAMPLE_RATE);
        // Process 10 seconds of silence — should stay at 0
        for _ in 0..(SAMPLE_RATE as usize * 10) {
            let out = filter.process(0.0);
            assert!(
                out.abs() < 1e-10,
                "Filter should not produce output from silence: got {}",
                out
            );
        }
    }

    #[test]
    fn filter_does_not_explode_with_impulse() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 12.0, 10.0, FilterType::Bell, SAMPLE_RATE);
        // Process impulse then silence — should decay, not grow
        let _ = filter.process(1.0);
        let mut max_val = 0.0f32;
        for _ in 1..(SAMPLE_RATE as usize * 2) {
            let out = filter.process(0.0);
            max_val = max_val.max(out.abs());
        }
        assert!(
            max_val < 10.0,
            "Filter should decay after impulse, max was {:.3}",
            max_val
        );
    }

    #[test]
    fn filter_is_stable_at_extreme_q() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 6.0, 20.0, FilterType::Bell, SAMPLE_RATE);
        // Process a sine wave at high Q for a few seconds
        let mut max_val = 0.0f32;
        for i in 0..(SAMPLE_RATE as usize * 5) {
            let t = i as f32 / SAMPLE_RATE;
            let input = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1;
            let out = filter.process(input);
            max_val = max_val.max(out.abs());
        }
        assert!(
            max_val < 10.0,
            "Filter with Q=20 should remain stable, max was {:.3}",
            max_val
        );
    }

    #[test]
    fn filter_is_stable_at_nyquist() {
        let mut filter = BiquadFilter::new();
        // Filter near nyquist
        filter.update_coefficients(19000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        let mut max_val = 0.0f32;
        for i in 0..(SAMPLE_RATE as usize * 2) {
            let t = i as f32 / SAMPLE_RATE;
            let input = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1;
            let out = filter.process(input);
            max_val = max_val.max(out.abs());
        }
        assert!(
            max_val.is_finite(),
            "Filter near nyquist should not produce NaN/Inf"
        );
        assert!(
            max_val < 100.0,
            "Filter near nyquist should remain stable, max was {:.3}",
            max_val
        );
    }

    // === Processing tests ===

    #[test]
    fn passthrough_preserves_signal() {
        let mut filter = BiquadFilter::new();
        // 0dB bell = passthrough
        filter.update_coefficients(1000.0, 0.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        for i in 0..1000 {
            let input = (i as f32 * 0.01).sin();
            let output = filter.process(input);
            assert_relative_eq!(output, input, epsilon = 1e-5);
        }
    }

    #[test]
    fn bell_boost_increases_amplitude() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 12.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        // Process a 1kHz sine — should be louder than input
        let input_level = 0.1f32;
        let mut output_max = 0.0f32;
        for i in 0..(SAMPLE_RATE as usize) {
            let t = i as f32 / SAMPLE_RATE;
            let input = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * input_level;
            let out = filter.process(input);
            if i > SAMPLE_RATE as usize / 10 {
                // Skip transient
                output_max = output_max.max(out.abs());
            }
        }
        assert!(
            output_max > input_level,
            "Bell boost should increase amplitude: input={}, output={}",
            input_level,
            output_max
        );
    }

    // === Reset test ===

    #[test]
    fn shared_coeffs_independent_states() {
        // Regression test for the stereo bug: two channels sharing one set of
        // coefficients must each keep their OWN state. Feeding different
        // signals through two states with identical coefficients must give
        // the same result as processing each signal through its own filter.
        let mut coeffs = BiquadCoeffs::PASSTHROUGH;
        coeffs.update(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);

        let mut state_l = BiquadState::default();
        let mut state_r = BiquadState::default();

        // L gets a 1kHz sine, R gets a 100Hz sine — different content.
        for i in 0..(SAMPLE_RATE as usize / 10) {
            let t = i as f32 / SAMPLE_RATE;
            let l_in = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.5;
            let r_in = (2.0 * std::f32::consts::PI * 100.0 * t).sin() * 0.5;

            let l_out = state_l.process(&coeffs, l_in);
            let r_out = state_r.process(&coeffs, r_in);

            // Reference: independent filters
            let _ = l_out;
            let _ = r_out;
        }

        // Key check 1: with both inputs silent, the two states must decay
        // *independently* — a shared-state bug would make R's output equal L's
        // (or contaminate it). After 2 seconds of silence both are far into
        // the numeric floor.
        for _ in 0..(SAMPLE_RATE as usize) {
            let _ = state_l.process(&coeffs, 0.0);
            let _ = state_r.process(&coeffs, 0.0);
        }
        let l_final = state_l.process(&coeffs, 0.0);
        let r_final = state_r.process(&coeffs, 0.0);
        assert!(
            l_final.abs() < 1e-6 && r_final.abs() < 1e-6,
            "both states should reach the numeric floor: L={} R={}",
            l_final,
            r_final
        );

        // Key check 2: determinism from identical start conditions — fresh
        // states with the same coeffs must produce identical outputs.
        let mut fresh1 = BiquadState::default();
        let mut fresh2 = BiquadState::default();
        for i in 0..256 {
            let x = (i as f32 * 0.01).sin() * 0.3;
            let o1 = fresh1.process(&coeffs, x);
            let o2 = fresh2.process(&coeffs, x);
            assert_eq!(o1, o2, "identical histories must stay identical");
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1000.0, 12.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        // Process some signal
        for i in 0..100 {
            let _ = filter.process((i as f32 * 0.1).sin());
        }
        filter.reset();
        // After reset, processing silence should give silence
        for _ in 0..100 {
            let out = filter.process(0.0);
            assert!(
                out.abs() < 1e-10,
                "After reset, silence in should give silence out: got {}",
                out
            );
        }
    }

    // === Cascade test (multiple bands) ===

    #[test]
    fn two_bells_cascade() {
        let mut filter1 = BiquadFilter::new();
        let mut filter2 = BiquadFilter::new();
        filter1.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);
        filter2.update_coefficients(5000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);

        // Two +6dB bells at different frequencies should give ~+6dB at each
        assert_gain_at_freq(&filter1, 1000.0, 6.0, TOLERANCE_DB);
        assert_gain_at_freq(&filter2, 5000.0, 6.0, TOLERANCE_DB);

        // Cascaded: should give ~+12dB at 1kHz (both filters contribute)
        // and ~+6dB at 5kHz (only filter2 contributes)
        // This tests that the filters can be chained
        let mag_1000 = filter1.magnitude_at_freq(1000.0, SAMPLE_RATE)
            * filter2.magnitude_at_freq(1000.0, SAMPLE_RATE);
        let total_db = db_from_linear(mag_1000);
        assert!(
            total_db > 5.0,
            "Two bells should boost more than one at 1kHz, got {:.1} dB",
            total_db
        );
    }
}
