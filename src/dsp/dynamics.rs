//! Dynamic EQ: a band whose gain follows the level inside its own region.
//!
//! Two pieces, both pure math so they can be tested without a plugin:
//!
//! * [`dynamic_gain_db`] — how much of a band's gain applies at a given
//!   detector level (threshold + range, linear in dB).
//! * [`Detector`] — a band-pass at the band's own frequency feeding a peak
//!   envelope follower with separate attack and release.
//!
//! The model is the one Pro-Q uses: with dynamics on, the band's gain knob is
//! the **maximum** gain change, and the band is inert until the level inside
//! it exceeds the threshold. So a −8 dB dynamic bell at −30 dB threshold does
//! nothing to quiet material and reaches the full −8 dB on loud material.
//!
//! Detection is on the **linked** signal (the louder of the channels) and one
//! detector feeds every channel, so the stereo image cannot shift: a reduction
//! that applied to L but not R would move the phantom centre.

use crate::dsp::BiquadFilter;
use crate::FilterType;

/// Below this the level reads as silence rather than −inf dB.
const LEVEL_FLOOR: f32 = 1e-6;

/// Minimum range that still counts as "dynamics active". A range of 0 dB would
/// mean "reach full gain at threshold", i.e. a step — treat it as off.
pub const MIN_RANGE_DB: f32 = 0.1;

/// Gain applied by a dynamic band, in dB.
///
/// `amount` ramps linearly in dB from 0 at the threshold to 1 at
/// `threshold + range`, then holds. Below the threshold the band contributes
/// nothing, which is what makes the EQ frequency-selective in level rather than
/// in time: quiet material passes untouched.
pub fn dynamic_gain_db(
    static_gain_db: f32,
    level_db: f32,
    threshold_db: f32,
    range_db: f32,
) -> f32 {
    if range_db < MIN_RANGE_DB {
        return static_gain_db;
    }
    let amount = ((level_db - threshold_db) / range_db).clamp(0.0, 1.0);
    static_gain_db * amount
}

/// One-pole smoothing coefficient for a time constant in milliseconds.
///
/// `1 - exp(-1/(tau*sr))` reaches 63.2% of a step after exactly `ms`, and is the
/// coefficient form that stays stable for any time constant (the naive
/// `1/(tau*sr)` overshoots to negative values once `tau*sr < 1`).
pub fn envelope_coeff(ms: f32, sample_rate: f32) -> f32 {
    let samples = (ms.max(0.01) * 0.001 * sample_rate).max(1.0);
    // `exp_m1` keeps precision for the long-time-constant end where
    // `1 - exp(-x)` would cancel to zero in f32 and freeze the envelope; the
    // lower clamp keeps the coefficient strictly positive for the same reason.
    let coeff = -(-1.0 / samples).exp_m1();
    coeff.clamp(1e-9, 1.0)
}

/// Level detector for one dynamic band.
///
/// The band-pass is centred on the band's frequency and follows its Q, so a
/// narrow bell listens to a narrow slice of spectrum. The envelope is a peak
/// follower with separate attack and release, which is what makes a dynamic EQ
/// feel like a compressor rather than a level-triggered switch.
pub struct Detector {
    band: BiquadFilter,
    /// Frequency/Q currently baked into `band`, so a static band costs no trig.
    tuned_to: (f32, f32),
    env: f32,
    level_db: f32,
    /// Attack/release in ms currently baked into the coefficients.
    timing: (f32, f32),
    attack_coeff: f32,
    release_coeff: f32,
    sample_rate: f32,
}

impl Detector {
    pub fn new(sample_rate: f32) -> Self {
        let mut detector = Self {
            band: BiquadFilter::new(),
            tuned_to: (f32::NAN, f32::NAN),
            env: 0.0,
            level_db: 20.0 * LEVEL_FLOOR.log10(),
            timing: (f32::NAN, f32::NAN),
            attack_coeff: 0.0,
            release_coeff: 0.0,
            sample_rate,
        };
        detector.set_timing(10.0, 150.0);
        detector
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        // Force both coefficients to be recomputed at the new rate.
        self.timing = (f32::NAN, f32::NAN);
        self.tuned_to = (f32::NAN, f32::NAN);
    }

    /// Forget the envelope and the filter memory. Called when a band is
    /// re-enabled, so a band that comes back after a long silence does not
    /// start from a stale level.
    pub fn reset(&mut self) {
        self.band.reset();
        self.env = 0.0;
        self.level_db = 20.0 * LEVEL_FLOOR.log10();
    }

    /// Current detector level in dB, from the last `update`.
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// Current envelope value (linear, 0..~1 for a full-scale tone).
    pub fn envelope(&self) -> f32 {
        self.env
    }

    fn set_timing(&mut self, attack_ms: f32, release_ms: f32) {
        if self.timing == (attack_ms, release_ms) {
            return;
        }
        self.timing = (attack_ms, release_ms);
        self.attack_coeff = envelope_coeff(attack_ms, self.sample_rate);
        self.release_coeff = envelope_coeff(release_ms, self.sample_rate);
    }

    /// Run the detector over one sub-block and return the level at its end.
    ///
    /// `q` is the listening bandwidth. Bells use their own Q; shelves and cuts
    /// pass `SHELF_DETECT_Q`, because their defining parameter is not a
    /// resonance and a shelf with a Q of 8 would listen to a hairline.
    pub fn update(
        &mut self,
        samples: &[f32],
        freq: f32,
        q: f32,
        attack_ms: f32,
        release_ms: f32,
    ) -> f32 {
        if self.tuned_to != (freq, q) {
            self.tuned_to = (freq, q);
            // A detector band-pass has 0 dB peak gain, so a full-scale sine at
            // the centre frequency reads 0 dBFS — the threshold scale is
            // therefore the same one the meter shows.
            self.band
                .update_cascade(freq, 0.0, q, FilterType::BandPass, 12, self.sample_rate);
        }
        self.set_timing(attack_ms, release_ms);

        let mut env = self.env;
        for &sample in samples {
            let rectified = self.band.process(sample).abs();
            let coeff = if rectified > env {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            env += (rectified - env) * coeff;
        }
        self.env = env;
        self.level_db = 20.0 * env.max(LEVEL_FLOOR).log10();
        self.level_db
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

/// Listening Q for bands whose shape has no resonance of its own (shelves and
/// cuts). Roughly one octave, which is what a corner needs to be judged by ear.
pub const SHELF_DETECT_Q: f32 = 0.7;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: f32 = 48_000.0;

    fn tone(hz: f32, dbfs: f32, n: usize) -> Vec<f32> {
        let amp = 10f32.powf(dbfs / 20.0);
        (0..n)
            .map(|i| (2.0 * PI * hz * i as f32 / SR).sin() * amp)
            .collect()
    }

    #[test]
    fn nothing_below_threshold_full_gain_above_threshold_plus_range() {
        // A -8 dB band, threshold -30, range 12: inert at -30, full at -18.
        assert_eq!(dynamic_gain_db(-8.0, -40.0, -30.0, 12.0), 0.0);
        assert_eq!(dynamic_gain_db(-8.0, -30.0, -30.0, 12.0), 0.0);
        assert_eq!(dynamic_gain_db(-8.0, -18.0, -30.0, 12.0), -8.0);
        // ... and holds there rather than overshooting.
        assert_eq!(dynamic_gain_db(-8.0, -3.0, -30.0, 12.0), -8.0);
    }

    #[test]
    fn the_ramp_between_threshold_and_range_is_linear_in_db() {
        // Halfway (6 dB over a 12 dB range) = half the gain.
        let half = dynamic_gain_db(12.0, -24.0, -30.0, 12.0);
        assert!((half - 6.0).abs() < 1e-4, "expected +6 dB, got {half}");
        // Boosts work the same way: a +6 dB dynamic bell lifts loud material.
        let quarter = dynamic_gain_db(12.0, -27.0, -30.0, 12.0);
        assert!(
            (quarter - 3.0).abs() < 1e-4,
            "expected +3 dB, got {quarter}"
        );
    }

    #[test]
    fn a_zero_range_is_treated_as_static() {
        // Nothing sane to ramp over: fall back to the static gain instead of
        // dividing by zero and producing NaN into the coefficients.
        assert_eq!(dynamic_gain_db(-8.0, -60.0, -30.0, 0.0), -8.0);
        assert_eq!(dynamic_gain_db(-8.0, -60.0, -30.0, 0.01), -8.0);
    }

    #[test]
    fn envelope_reaches_63_percent_in_the_stated_time() {
        for ms in [1.0f32, 10.0, 100.0] {
            let k = envelope_coeff(ms, SR);
            let n = (ms * 0.001 * SR).round() as usize;
            let mut env = 0.0f32;
            for _ in 0..n {
                env += (1.0 - env) * k;
            }
            assert!(
                (env - 0.632).abs() < 0.01,
                "{ms} ms attack reached {env:.3} instead of 0.632"
            );
        }
    }

    #[test]
    fn envelope_coefficients_stay_in_range_for_extreme_times() {
        // A sub-sample time constant must not produce a negative coefficient,
        // which would make the envelope oscillate or go unstable.
        for ms in [0.01f32, 0.1, 1.0, 2000.0, 1e6] {
            let k = envelope_coeff(ms, SR);
            assert!(k > 0.0 && k <= 1.0, "{ms} ms gave coefficient {k}");
        }
    }

    #[test]
    fn detector_reads_a_tone_at_the_band_centre() {
        let mut det = Detector::new(SR);
        // 200 ms of a -20 dBFS 1 kHz tone through a 1 kHz band-pass: the
        // envelope settles on the tone's amplitude, so the level is -20 dBFS.
        let samples = tone(1_000.0, -20.0, (SR * 0.2) as usize);
        det.update(&samples, 1_000.0, 1.0, 5.0, 150.0);
        let level = det.level_db();
        assert!(
            (level - -20.0).abs() < 1.5,
            "expected about -20 dB, got {level:.2}"
        );
    }

    #[test]
    fn detector_rejects_a_tone_outside_its_band() {
        let mut det = Detector::new(SR);
        // Same level, three octaves up: a 1 kHz band-pass must not hear it.
        let samples = tone(8_000.0, -20.0, (SR * 0.2) as usize);
        det.update(&samples, 1_000.0, 1.0, 5.0, 150.0);
        let level = det.level_db();
        assert!(
            level < -35.0,
            "8 kHz should be well below a 1 kHz detector's -20 dB tone, got {level:.2} dB"
        );
    }

    #[test]
    fn detector_tracks_level_changes_with_attack_and_release() {
        let mut det = Detector::new(SR);
        let loud = tone(1_000.0, -6.0, (SR * 0.1) as usize);
        det.update(&loud, 1_000.0, 1.0, 5.0, 200.0);
        let loud_level = det.level_db();
        assert!(
            loud_level > -12.0,
            "loud tone should read high: {loud_level}"
        );

        // Silence for 20 ms with a 200 ms release: still well above the floor,
        // because release is slow by design.
        let silence = vec![0.0f32; (SR * 0.02) as usize];
        det.update(&silence, 1_000.0, 1.0, 5.0, 200.0);
        let after_short_silence = det.level_db();
        assert!(
            after_short_silence > loud_level - 25.0,
            "a 200 ms release should not fall 25 dB in 20 ms ({loud_level:.1} -> {after_short_silence:.1})"
        );

        // A long silence does let it go.
        det.update(&vec![0.0f32; (SR * 2.0) as usize], 1_000.0, 1.0, 5.0, 200.0);
        assert!(
            det.level_db() < -60.0,
            "2 s of silence should reach the floor, got {:.1}",
            det.level_db()
        );
    }

    #[test]
    fn detector_never_produces_nan_or_infinity() {
        let mut det = Detector::new(SR);
        for input in [
            vec![0.0f32; 256],
            vec![1.0f32; 256],
            vec![-1.0f32; 256],
            // Alternating full scale: the worst case for a biquad's state.
            (0..256)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ] {
            let level = det.update(&input, 1_000.0, 20.0, 0.1, 5.0);
            assert!(level.is_finite(), "level went non-finite: {level}");
            assert!(det.envelope().is_finite());
        }
    }

    #[test]
    fn reset_clears_the_envelope() {
        let mut det = Detector::new(SR);
        det.update(&tone(1_000.0, -6.0, 4_800), 1_000.0, 1.0, 5.0, 150.0);
        assert!(det.envelope() > 0.1);
        det.reset();
        assert_eq!(det.envelope(), 0.0);
    }
}
