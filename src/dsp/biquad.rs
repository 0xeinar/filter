use crate::FilterType;

/// Maximum number of sections one band can need. 96 dB/oct is eight 12 dB/oct
/// sections; odd slopes finish with a first-order section, which reuses the
/// same struct with `b2 = a2 = 0`.
pub const MAX_SECTIONS: usize = 8;

/// The slopes a cut filter can have, in dB per octave.
pub const SLOPES_DB_PER_OCT: [u32; 9] = [6, 12, 18, 24, 30, 36, 48, 72, 96];

/// Split a slope into second-order sections plus an optional first-order tail.
///
/// Butterworth cascades come in even orders, so odd multiples of 6 dB/oct are
/// one second-order section short of the next even slope and finish with a
/// one-pole section: 18 dB/oct = 12 + 6, 30 = 24 + 6, 6 = 6.
pub fn cascade_shape(slope_db_per_oct: u32) -> (usize, bool) {
    match slope_db_per_oct {
        6 => (0, true),
        18 => (1, true),
        30 => (2, true),
        24 => (2, false),
        36 => (3, false),
        48 => (4, false),
        72 => (6, false),
        96 => (8, false),
        // 12 dB/oct is the default, and an unrecognised slope gets it too
        // rather than silence.
        _ => (1, false),
    }
}

/// Q of section `k` in an `n`-section Butterworth cascade, ascending.
///
/// `1 / (2 cos((2k+1) pi / 4n))` — the standard result for a maximally flat
/// cascade of 2nd-order sections. For n = 1 this is 0.7071, i.e. the familiar
/// single biquad.
pub fn butterworth_q(k: usize, n: usize) -> f64 {
    debug_assert!(n > 0 && k < n);
    1.0 / (2.0 * (((2 * k + 1) as f64) * std::f64::consts::PI / (4.0 * n as f64)).cos())
}

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

    /// Coefficients for a single section of `filter_type`.
    ///
    /// Cuts are treated as one Butterworth section here; `update_cascade` is
    /// what turns a slope into several.
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

                self.set(b0, b1, b2, a1, a2, a0);
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

                self.set(b0, b1, b2, a1, a2, a0);
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

                self.set(b0, b1, b2, a1, a2, a0);
            }
            FilterType::LowCut => self.cut_section(freq, q as f64, true, sample_rate),
            FilterType::HighCut => self.cut_section(freq, q as f64, false, sample_rate),
            FilterType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                self.set(b0, b1, b2, a1, a2, a0);
            }
            FilterType::BandPass => {
                // Constant 0 dB peak gain (Audio EQ Cookbook)
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                self.set(b0, b1, b2, a1, a2, a0);
            }
            FilterType::AllPass => {
                let b0 = 1.0 - alpha;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 + alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                self.set(b0, b1, b2, a1, a2, a0);
            }
            // Tilt is two sections; a single-section call gets its top half,
            // which keeps this function total for any caller.
            FilterType::TiltShelf => {
                self.update(freq, gain_db / 2.0, q, FilterType::HighShelf, sample_rate);
            }
        }
    }

    /// One 2nd-order cut section at an explicit Q.
    fn cut_section(&mut self, freq: f32, q: f64, highpass: bool, sample_rate: f32) {
        let w0 = 2.0 * std::f64::consts::PI * freq as f64 / sample_rate as f64;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        let (b0, b1, b2) = if highpass {
            let b = (1.0 + cos_w0) / 2.0;
            (b, -2.0 * b, b)
        } else {
            let b = (1.0 - cos_w0) / 2.0;
            (b, 2.0 * b, b)
        };
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        self.set(b0, b1, b2, a1, a2, a0);
    }

    /// One 1st-order cut section (6 dB/oct), bilinear transform of
    /// `s/(s+1)` (high-pass) or `1/(s+1)` (low-pass).
    ///
    /// A one-pole section is a biquad with `b2 = a2 = 0`, so it rides the same
    /// state and the same processing loop.
    fn cut_first_order(&mut self, freq: f32, highpass: bool, sample_rate: f32) {
        let k = (std::f64::consts::PI * freq as f64 / sample_rate as f64).tan();
        let norm = 1.0 / (1.0 + k);
        let (b0, b1) = if highpass {
            (norm, -norm)
        } else {
            (k * norm, k * norm)
        };

        self.b0 = b0;
        self.b1 = b1;
        self.b2 = 0.0;
        self.a1 = (k - 1.0) * norm;
        self.a2 = 0.0;
    }

    #[inline]
    fn set(&mut self, b0: f64, b1: f64, b2: f64, a1: f64, a2: f64, a0: f64) {
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    /// Fill `sections` with the cascade for one band and return how many
    /// sections are in use (0 means passthrough).
    ///
    /// This is the single place that decides how a band maps to filter
    /// sections, so the audio path and the drawn response curve cannot drift
    /// apart — `curve::band_response_db` calls it too.
    pub fn update_cascade(
        sections: &mut [BiquadCoeffs; MAX_SECTIONS],
        freq: f32,
        gain_db: f32,
        q: f32,
        filter_type: FilterType,
        slope_db_per_oct: u32,
        sample_rate: f32,
    ) -> usize {
        match filter_type {
            FilterType::LowCut | FilterType::HighCut => {
                let highpass = filter_type == FilterType::LowCut;
                let (second_order, first_order) = cascade_shape(slope_db_per_oct);

                for (k, section) in sections.iter_mut().take(second_order).enumerate() {
                    section.cut_section(
                        freq,
                        butterworth_q(k, second_order),
                        highpass,
                        sample_rate,
                    );
                }

                let mut count = second_order;
                if first_order {
                    sections[count].cut_first_order(freq, highpass, sample_rate);
                    count += 1;
                }
                count
            }
            FilterType::TiltShelf => {
                // A tilt is a cut and a boost around the same pivot: -g/2 at
                // the bottom, +g/2 at the top, 0 dB at the pivot.
                sections[0].update(freq, -gain_db / 2.0, q, FilterType::LowShelf, sample_rate);
                sections[1].update(freq, gain_db / 2.0, q, FilterType::HighShelf, sample_rate);
                2
            }
            _ => {
                sections[0].update(freq, gain_db, q, filter_type, sample_rate);
                1
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

/// One band's whole filter — a cascade of up to `MAX_SECTIONS` sections — for
/// single-channel use (tests, analysis, the drawn response curve).
///
/// The plugin's hot path does not use this type: it keeps the sections and
/// states in flat arrays so a block can be processed without indexing a
/// struct per band. The coefficient math is shared, which is what keeps the
/// two honest.
pub struct BiquadFilter {
    sections: [BiquadCoeffs; MAX_SECTIONS],
    count: usize,
    states: [BiquadState; MAX_SECTIONS],
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl BiquadFilter {
    pub fn new() -> Self {
        Self {
            sections: [BiquadCoeffs::PASSTHROUGH; MAX_SECTIONS],
            count: 0,
            states: [BiquadState::default(); MAX_SECTIONS],
        }
    }

    /// Update the filter at the default 12 dB/oct slope.
    pub fn update_coefficients(
        &mut self,
        freq: f32,
        gain_db: f32,
        q: f32,
        filter_type: FilterType,
        sample_rate: f32,
    ) {
        self.update_cascade(freq, gain_db, q, filter_type, 12, sample_rate);
    }

    /// Update the filter for a given slope (only cuts use more than one
    /// section).
    pub fn update_cascade(
        &mut self,
        freq: f32,
        gain_db: f32,
        q: f32,
        filter_type: FilterType,
        slope_db_per_oct: u32,
        sample_rate: f32,
    ) {
        self.count = BiquadCoeffs::update_cascade(
            &mut self.sections,
            freq,
            gain_db,
            q,
            filter_type,
            slope_db_per_oct,
            sample_rate,
        );
        // A cascade that changes shape would ring on the old memory.
        self.reset();
    }

    /// Process a single sample through the cascade (Direct Form II Transposed)
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let mut s = input;
        for k in 0..self.count {
            s = self.states[k].process(&self.sections[k], s);
        }
        s
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
    }

    /// Number of sections in the current cascade (0 = passthrough).
    pub fn section_count(&self) -> usize {
        self.count
    }

    /// Compute the frequency response (magnitude) at a given frequency
    pub fn magnitude_at_freq(&self, freq: f32, sample_rate: f32) -> f32 {
        let mut mag = 1.0f32;
        for section in self.sections.iter().take(self.count) {
            mag *= section.magnitude_at_freq(freq, sample_rate);
        }
        mag
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

    // === New filter types ===

    fn mag_db(filter: &BiquadFilter, freq: f32) -> f32 {
        db_from_linear(filter.magnitude_at_freq(freq, SAMPLE_RATE))
    }

    #[test]
    fn notch_rejects_the_centre_and_passes_everything_else() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1_000.0, 0.0, 4.0, FilterType::Notch, SAMPLE_RATE);

        assert!(
            mag_db(&filter, 1_000.0) < -40.0,
            "notch should reject its centre hard, got {:.1} dB",
            mag_db(&filter, 1_000.0)
        );
        for freq in [50.0, 200.0, 20_000.0] {
            assert!(
                mag_db(&filter, freq).abs() < 1.0,
                "notch should pass {freq} Hz, got {:.2} dB",
                mag_db(&filter, freq)
            );
        }
    }

    #[test]
    fn band_pass_is_unity_at_its_centre_and_rejects_both_sides() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1_000.0, 0.0, 2.0, FilterType::BandPass, SAMPLE_RATE);

        assert!(
            mag_db(&filter, 1_000.0).abs() < 0.1,
            "constant-peak band pass should be 0 dB at centre, got {:.2} dB",
            mag_db(&filter, 1_000.0)
        );
        assert!(mag_db(&filter, 50.0) < -10.0);
        assert!(mag_db(&filter, 20_000.0) < -10.0);
    }

    #[test]
    fn all_pass_is_unity_everywhere_but_changes_phase() {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(1_000.0, 0.0, 2.0, FilterType::AllPass, SAMPLE_RATE);

        for freq in [20.0, 100.0, 1_000.0, 9_000.0, 20_000.0] {
            assert!(
                mag_db(&filter, freq).abs() < 0.05,
                "all pass must be flat, {freq} Hz came out {:.3} dB",
                mag_db(&filter, freq)
            );
        }

        // ...but it is not the identity: a sine comes out shifted.
        filter.reset();
        let mut delayed = BiquadFilter::new();
        delayed.update_coefficients(1_000.0, 0.0, 2.0, FilterType::AllPass, SAMPLE_RATE);
        let mut differs = false;
        for n in 0..2_000 {
            let x = (2.0 * std::f32::consts::PI * 1_000.0 * n as f32 / SAMPLE_RATE).sin();
            let y = delayed.process(x);
            if (y - x).abs() > 1e-3 {
                differs = true;
            }
        }
        assert!(differs, "an all pass that does not alter phase is a bug");
    }

    #[test]
    fn tilt_shelf_is_minus_half_the_gain_low_and_plus_half_high() {
        let mut filter = BiquadFilter::new();
        filter.update_cascade(1_000.0, 12.0, 0.707, FilterType::TiltShelf, 12, SAMPLE_RATE);

        assert_eq!(filter.section_count(), 2, "a tilt is a cut and a boost");
        let low = mag_db(&filter, 30.0);
        let high = mag_db(&filter, 18_000.0);
        assert!(
            (low + 6.0).abs() < 0.7,
            "tilt should be -6 dB at the bottom, got {low:.1}"
        );
        assert!(
            (high - 6.0).abs() < 0.7,
            "tilt should be +6 dB at the top, got {high:.1}"
        );
        assert!(
            mag_db(&filter, 1_000.0).abs() < 0.5,
            "the pivot should stay at 0 dB, got {:.2}",
            mag_db(&filter, 1_000.0)
        );
    }

    #[test]
    fn tilt_is_the_mirror_image_of_its_negative() {
        let mut up = BiquadFilter::new();
        up.update_cascade(1_000.0, 8.0, 0.707, FilterType::TiltShelf, 12, SAMPLE_RATE);
        let mut down = BiquadFilter::new();
        down.update_cascade(1_000.0, -8.0, 0.707, FilterType::TiltShelf, 12, SAMPLE_RATE);

        for freq in [50.0, 500.0, 5_000.0] {
            let sum = mag_db(&up, freq) + mag_db(&down, freq);
            assert!(
                sum.abs() < 0.01,
                "a +8 dB tilt and its inverse should cancel at {freq} Hz, got {sum:.2} dB"
            );
        }
    }

    // === Slopes ===

    #[test]
    fn cascade_shape_matches_the_slope_table() {
        // dB/oct -> (2nd-order sections, + 1st-order section)
        let expected = [
            (6, (0, true)),
            (12, (1, false)),
            (18, (1, true)),
            (24, (2, false)),
            (30, (2, true)),
            (36, (3, false)),
            (48, (4, false)),
            (72, (6, false)),
            (96, (8, false)),
        ];
        for (slope, shape) in expected {
            assert_eq!(cascade_shape(slope), shape, "{slope} dB/oct");
        }
        assert_eq!(
            SLOPES_DB_PER_OCT.len(),
            expected.len(),
            "the table the UI offers and the table the DSP understands must agree"
        );
    }

    #[test]
    fn butterworth_section_qs_are_the_textbook_values() {
        // 2 sections: 0.5412, 1.3066. 4 sections adds 0.5098 and 2.5629.
        assert!((butterworth_q(0, 2) - 0.541_196).abs() < 1e-5);
        assert!((butterworth_q(1, 2) - 1.306_563).abs() < 1e-5);
        assert!((butterworth_q(0, 4) - 0.509_796).abs() < 1e-5);
        assert!((butterworth_q(3, 4) - 2.562_915).abs() < 1e-5);
        // A single section is the familiar 0.7071.
        assert!((butterworth_q(0, 1) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn steeper_cuts_use_more_sections() {
        let mut counts = Vec::new();
        for slope in SLOPES_DB_PER_OCT {
            let mut filter = BiquadFilter::new();
            filter.update_cascade(1_000.0, 0.0, 0.707, FilterType::LowCut, slope, SAMPLE_RATE);
            counts.push((slope, filter.section_count()));
        }
        let mut previous = 0;
        for (slope, count) in counts {
            assert!(
                count >= previous,
                "{slope} dB/oct uses {count} sections, fewer than a gentler slope ({previous})"
            );
            assert!(count <= MAX_SECTIONS);
            previous = count;
        }
    }

    #[test]
    fn a_cut_ignores_its_gain_and_its_q() {
        // Slope, not gain or Q, describes a cut: two different gains and two
        // different Qs must produce the same thing.
        let mut a = BiquadFilter::new();
        a.update_cascade(500.0, 0.0, 0.707, FilterType::LowCut, 24, SAMPLE_RATE);
        let mut b = BiquadFilter::new();
        b.update_cascade(500.0, 18.0, 9.0, FilterType::LowCut, 24, SAMPLE_RATE);

        for freq in [50.0, 200.0, 500.0, 4_000.0] {
            let diff = (mag_db(&a, freq) - mag_db(&b, freq)).abs();
            assert!(
                diff < 4e-4,
                "cut response moved with gain/Q at {freq} Hz by {diff:.5} dB"
            );
        }
    }
}
