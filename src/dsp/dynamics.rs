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

use std::sync::atomic::Ordering;

use atomic_float::AtomicF32;

use crate::dsp::BiquadFilter;
use crate::FilterType;

/// Below this the level reads as silence rather than −inf dB.
const LEVEL_FLOOR: f32 = 1e-6;

/// Ranges below this do nothing level-dependent: the band is effectively
/// static and the dynamic code can be skipped.
pub const MIN_RANGE_DB: f32 = 0.05;

/// Knee widths below this are a hard corner (the floating-point equivalent of
/// "knee off").
pub const MIN_KNEE_DB: f32 = 0.1;

/// The knee a band starts with.
///
/// Pro-Q keeps its knee internal and program-dependent, so a Pro-Q band is
/// never a hard corner unless it wants to be. Ours is a control, but it starts
/// at a few dB for the same reason: a band that steps the instant a note
/// crosses the threshold is the sound nobody wants, and it should take a
/// deliberate move to get there.
pub const DEFAULT_KNEE_DB: f32 = 3.0;

/// Listening Q for a sidechain-filtered detector.
///
/// Fixed rather than shared with the band: the point of a sidechain is to
/// listen somewhere *else*, and inheriting the band's Q would make the trigger
/// band as narrow as the band being controlled — exactly what you do not want
/// when ducking a 4-octave bass from a kick.
pub const SIDECHAIN_Q: f32 = 1.0;

/// The dynamic law of one band.
///
/// This follows Pro-Q's model rather than the one #14 shipped, and the
/// difference matters: a band's **static gain** and its **dynamic range** are
/// two independent parameters that are live at the same time. The sign of the
/// range is the direction — negative moves the band down when the trigger
/// exceeds the threshold (compression), positive moves it up (expansion) — so
/// there is no up/down switch, and `+6 dB static with −12 dB range` is a
/// setting a single signed gain knob cannot express.
///
/// Triggering is one-sided: above the threshold, with the knee easing the
/// arrival from `threshold - knee`. That is also Pro-Q's behaviour ("a soft
/// knee is used internally… so it can start triggering a little bit below the
/// selected threshold value").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynLaw {
    pub threshold_db: f32,
    /// dB of movement at full engagement. The sign is the direction.
    pub range_db: f32,
    /// How far below the threshold the trigger eases in. 0 is a hard corner.
    pub knee_db: f32,
}

impl DynLaw {
    pub fn new(threshold_db: f32, range_db: f32, knee_db: f32) -> Self {
        Self {
            threshold_db,
            range_db,
            knee_db,
        }
    }

    /// Whether the band does anything level-dependent at all.
    pub fn is_active(&self) -> bool {
        self.range_db.abs() >= MIN_RANGE_DB
    }

    /// How far into the dynamic range the band is at `level_db`, in 0..=1.
    ///
    /// Zero at `threshold - knee`, a quarter at the knee's midpoint, one at the
    /// threshold and above. The quadratic is the whole point of the knee: a
    /// note decaying through the threshold eases into the band instead of
    /// arriving at it in one step.
    pub fn amount(&self, level_db: f32) -> f32 {
        if self.knee_db < MIN_KNEE_DB {
            return if level_db >= self.threshold_db {
                1.0
            } else {
                0.0
            };
        }
        if level_db <= self.threshold_db - self.knee_db {
            0.0
        } else if level_db >= self.threshold_db {
            1.0
        } else {
            let t = (level_db - (self.threshold_db - self.knee_db)) / self.knee_db;
            t * t
        }
    }

    /// The dynamic part of the band's gain, in dB — what the meter shows and
    /// what the range ring is a scale for.
    pub fn dynamic_gain_db(&self, level_db: f32) -> f32 {
        self.range_db * self.amount(level_db)
    }

    /// The band's total gain: static plus the dynamic part.
    pub fn applied_gain_db(&self, static_gain_db: f32, level_db: f32) -> f32 {
        static_gain_db + self.dynamic_gain_db(level_db)
    }
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

/// Live per-band gain, published by the audio thread for the GUI to draw.
///
/// A dynamic band's applied gain is the one number that tells you whether the
/// thing is working, and it exists only on the audio thread. One relaxed store
/// per sub-block per dynamic band is the whole cost; the GUI reads it once per
/// frame. Values are only meaningful for bands whose dynamics are enabled — the
/// GUI checks that before reading, so a stale entry after switching dynamics
/// off can never be drawn.
pub struct BandMeters {
    db: [AtomicF32; crate::MAX_BANDS],
}

impl BandMeters {
    pub fn new() -> Self {
        Self {
            db: core::array::from_fn(|_| AtomicF32::new(0.0)),
        }
    }

    /// Audio thread: publish the gain currently baked for `band`.
    #[inline]
    pub fn publish(&self, band: usize, db: f32) {
        if let Some(slot) = self.db.get(band) {
            slot.store(db, Ordering::Relaxed);
        }
    }

    /// GUI thread: the gain last baked for `band`.
    pub fn read(&self, band: usize) -> f32 {
        self.db
            .get(band)
            .map(|slot| slot.load(Ordering::Relaxed))
            .unwrap_or(0.0)
    }
}

impl Default for BandMeters {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn nothing_below_the_knee_the_whole_range_at_the_threshold() {
        // -12 dB range, threshold -30, 3 dB knee: inert at -34, the full -12 by
        // -30, and holding there however loud it gets.
        let law = DynLaw::new(-30.0, -12.0, 3.0);
        assert_eq!(law.dynamic_gain_db(-40.0), 0.0);
        assert_eq!(law.dynamic_gain_db(-34.0), 0.0);
        assert_eq!(law.dynamic_gain_db(-30.0), -12.0);
        assert_eq!(law.dynamic_gain_db(-3.0), -12.0);
    }

    #[test]
    fn the_sign_of_the_range_is_the_direction() {
        // Pro-Q's convention: negative compresses, positive expands. No mode
        // switch, and the magnitude is the same either way.
        let down = DynLaw::new(-30.0, -12.0, 3.0);
        let up = DynLaw::new(-30.0, 12.0, 3.0);
        for tenths in -400..-200 {
            let level = tenths as f32 / 10.0;
            assert!(
                (down.dynamic_gain_db(level) + up.dynamic_gain_db(level)).abs() < 1e-5,
                "level {level}: down {} vs up {:+}",
                down.dynamic_gain_db(level),
                up.dynamic_gain_db(level)
            );
        }
    }

    #[test]
    fn static_gain_and_range_add_up() {
        // The whole point of the model: a band can lift always and duck when it
        // gets loud. +6 dB static with a -12 dB range is -6 dB at full
        // engagement and +6 dB below the knee.
        let law = DynLaw::new(-30.0, -12.0, 3.0);
        assert!((law.applied_gain_db(6.0, -40.0) - 6.0).abs() < 1e-5);
        assert!((law.applied_gain_db(6.0, -30.0) - -6.0).abs() < 1e-5);
        // A static band is the same arithmetic with range 0.
        let static_only = DynLaw::new(-30.0, 0.0, 3.0);
        assert!((static_only.applied_gain_db(6.0, -3.0) - 6.0).abs() < 1e-5);
        assert!(!static_only.is_active());
    }

    #[test]
    fn a_six_db_knee_gives_a_quarter_of_the_range_three_db_down() {
        // The number from #20: with a 6 dB knee, 3 dB below the threshold is
        // the midpoint of a quadratic that runs 0 to 1 — exactly a quarter.
        let law = DynLaw::new(-30.0, -12.0, 6.0);
        let amount = law.amount(-33.0);
        assert!(
            (amount - 0.25).abs() < 1e-5,
            "expected 0.25 of the range, got {amount}"
        );
        let gain = law.dynamic_gain_db(-33.0);
        assert!((gain - -3.0).abs() < 1e-4, "expected -3 dB, got {gain}");
    }

    #[test]
    fn the_knee_is_monotonic_with_no_step_at_either_end() {
        let law = DynLaw::new(-30.0, -12.0, 6.0);
        let mut previous = 0.0;
        let mut level = -40.0;
        while level <= -20.0 {
            let amount = law.amount(level);
            assert!(
                amount >= previous - 1e-6,
                "amount fell from {previous} to {amount} at {level} dB"
            );
            assert!(
                amount - previous < 0.05,
                "step of {:.3} at {level} dB — the knee is not smoothing the corner",
                amount - previous
            );
            previous = amount;
            level += 0.1;
        }
        assert_eq!(law.amount(-36.0), 0.0);
        assert_eq!(law.amount(-30.0), 1.0);
    }

    #[test]
    fn a_zero_knee_is_a_hard_corner() {
        let law = DynLaw::new(-30.0, -12.0, 0.0);
        assert_eq!(law.amount(-30.01), 0.0);
        assert_eq!(law.amount(-30.0), 1.0);
        // A knee below the minimum width is the same as none.
        let tiny = DynLaw::new(-30.0, -12.0, 0.05);
        assert_eq!(tiny.amount(-30.01), law.amount(-30.01));
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
    fn meters_are_per_band_and_default_to_zero() {
        let meters = BandMeters::new();
        assert_eq!(meters.read(0), 0.0);
        meters.publish(3, -7.5);
        assert_eq!(meters.read(3), -7.5);
        // Neighbouring bands must not see it: one array indexed by band, or
        // every dynamic band would show the last one drawn.
        assert_eq!(meters.read(2), 0.0);
        assert_eq!(meters.read(4), 0.0);
        // Out of range reads are harmless rather than a panic on the GUI thread.
        assert_eq!(meters.read(crate::MAX_BANDS + 5), 0.0);
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
