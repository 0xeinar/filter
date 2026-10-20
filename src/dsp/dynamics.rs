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

/// Minimum range that still counts as "dynamics active". A range of 0 dB would
/// mean "reach full gain at threshold", i.e. a step — treat it as off.
pub const MIN_RANGE_DB: f32 = 0.1;

/// Knee widths below this are a hard corner (the floating-point equivalent of
/// "knee off").
pub const MIN_KNEE_DB: f32 = 0.1;

/// Listening Q for a sidechain-filtered detector.
///
/// Fixed rather than shared with the band: the point of a sidechain is to
/// listen somewhere *else*, and inheriting the band's Q would make the trigger
/// band as narrow as the band being controlled — exactly what you do not want
/// when ducking a 4-octave bass from a kick.
pub const SIDECHAIN_Q: f32 = 1.0;

/// Which side of the threshold a dynamic band reacts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynMode {
    /// Gain applies *above* the threshold: quiet material passes untouched and
    /// loud material gets the band's full gain. The default, and the one that
    /// makes a cut frequency-selective in level.
    Down,
    /// Gain applies *below* the threshold, reaching full gain at
    /// `threshold - knee` (or `threshold - range` with no knee). This is upward
    /// compression: a positive gain lifts quiet material towards the threshold
    /// instead of boosting everything.
    Up,
}

impl DynMode {
    pub fn from_i32(value: i32) -> Self {
        if value == 1 {
            Self::Up
        } else {
            Self::Down
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Down => "Down",
            Self::Up => "Up",
        }
    }
}

/// The dynamic law of one band: where the transition starts, how wide it is,
/// how it is eased, and which side of the threshold it acts on.
///
/// A struct rather than five positional arguments because two of them are dB
/// quantities that mean entirely different things (`threshold` and `range`),
/// and a transposed pair would be invisible in a call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynLaw {
    pub threshold_db: f32,
    /// Transition width with no knee: full gain at `threshold + range`.
    pub range_db: f32,
    /// Transition width when a knee is set. See [`DynLaw::amount`].
    pub knee_db: f32,
    pub mode: DynMode,
}

impl DynLaw {
    pub fn new(threshold_db: f32, range_db: f32, knee_db: f32, mode: DynMode) -> Self {
        Self {
            threshold_db,
            range_db,
            knee_db,
            mode,
        }
    }

    /// How much of the band's static gain applies at `level_db`, in 0..=1.
    ///
    /// Two shapes, and only one of them can be the transition:
    ///
    /// * **No knee** — linear in dB from 0 at the threshold to 1 at
    ///   `threshold + range`. A hard corner at the threshold.
    /// * **Knee `k`** — quadratic from 0 at `threshold - k` to 1 at the
    ///   threshold, then held. `range` is the transition width *when no knee is
    ///   set*: a knee and a range cannot both define the same transition, and
    ///   silently ignoring one of them would be worse than saying so here.
    ///
    /// The quadratic is what removes the step: a note decaying through the
    /// threshold eases into the band instead of arriving at it.
    pub fn amount(&self, level_db: f32) -> f32 {
        let knee = self.knee_db;
        let use_knee = knee >= MIN_KNEE_DB;
        if !use_knee && self.range_db < MIN_RANGE_DB {
            // Nothing sane to ramp over: the band is static.
            return 1.0;
        }
        // Upward is the same law with the level mirrored about the threshold —
        // one implementation, so the knee and the ramp cannot diverge between
        // the two directions.
        let level = match self.mode {
            DynMode::Down => level_db,
            DynMode::Up => 2.0 * self.threshold_db - level_db,
        };
        if !use_knee {
            return ((level - self.threshold_db) / self.range_db).clamp(0.0, 1.0);
        }
        if level <= self.threshold_db - knee {
            0.0
        } else if level >= self.threshold_db {
            1.0
        } else {
            let t = (level - (self.threshold_db - knee)) / knee;
            t * t
        }
    }

    /// The gain this band applies, in dB.
    pub fn gain_db(&self, static_gain_db: f32, level_db: f32) -> f32 {
        static_gain_db * self.amount(level_db)
    }
}

/// Gain applied by a dynamic band with no knee and no direction, in dB.
///
/// Kept as a free function because it is the common case and the older tests
/// read better for it; it is exactly [`DynLaw`] with `knee = 0`, `mode = Down`.
pub fn dynamic_gain_db(
    static_gain_db: f32,
    level_db: f32,
    threshold_db: f32,
    range_db: f32,
) -> f32 {
    DynLaw::new(threshold_db, range_db, 0.0, DynMode::Down).gain_db(static_gain_db, level_db)
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

    // ---- upward mode -----------------------------------------------------

    #[test]
    fn upward_applies_gain_below_the_threshold_and_not_above_it() {
        // +6 dB band, threshold -30, range 6, no knee: full gain at -36 and
        // below, nothing at -30 and above — the mirror of the downward law.
        let law = DynLaw::new(-30.0, 6.0, 0.0, DynMode::Up);
        assert_eq!(law.gain_db(6.0, -36.0), 6.0);
        assert_eq!(law.gain_db(6.0, -40.0), 6.0);
        assert_eq!(law.gain_db(6.0, -30.0), 0.0);
        // The case that tempted a second implementation: loud material must be
        // left alone *by the dynamic part*. It is the same static gain either
        // way, which for a dynamic band is 0 dB applied.
        assert_eq!(law.gain_db(6.0, -20.0), 0.0);
        // Halfway through the mirrored ramp is half the gain.
        let half = law.gain_db(6.0, -33.0);
        assert!((half - 3.0).abs() < 1e-4, "expected +3 dB, got {half}");
    }

    #[test]
    fn up_and_down_are_mirror_images_of_each_other() {
        // Same law, level reflected about the threshold: every amount matches
        // its mirror. This is what keeps the knee from existing twice.
        let down = DynLaw::new(-30.0, 8.0, 0.0, DynMode::Down);
        let up = DynLaw::new(-30.0, 8.0, 0.0, DynMode::Up);
        for tenths in -450..-150 {
            let level = tenths as f32 / 10.0;
            let mirror = -60.0 - level;
            assert!(
                (down.amount(level) - up.amount(mirror)).abs() < 1e-5,
                "level {level}: down {} vs up at {mirror} {}",
                down.amount(level),
                up.amount(mirror)
            );
        }
    }

    // ---- soft knee -------------------------------------------------------

    #[test]
    fn a_six_db_knee_gives_a_quarter_of_full_gain_three_db_down() {
        // The number from the issue: with a 6 dB knee, 3 dB below the threshold
        // is the midpoint of a quadratic that starts at 0 and ends at 1 —
        // exactly a quarter.
        let law = DynLaw::new(-30.0, 12.0, 6.0, DynMode::Down);
        let amount = law.amount(-33.0);
        assert!(
            (amount - 0.25).abs() < 1e-5,
            "expected 0.25 of full gain, got {amount}"
        );
        let gain = law.gain_db(-12.0, -33.0);
        assert!((gain - -3.0).abs() < 1e-4, "expected -3 dB, got {gain}");
    }

    #[test]
    fn the_knee_is_monotonic_and_has_no_step_at_either_end() {
        let law = DynLaw::new(-30.0, 12.0, 6.0, DynMode::Down);
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
        // Starts at exactly zero, ends at exactly full.
        assert_eq!(law.amount(-36.0), 0.0);
        assert_eq!(law.amount(-30.0), 1.0);
        assert_eq!(law.amount(-30.1).min(1.0), law.amount(-30.1));
    }

    #[test]
    fn without_a_knee_the_corner_is_still_a_corner() {
        // The old behaviour must not change: `range` alone still ramps linearly
        // from the threshold, and there is no easing below it.
        let law = DynLaw::new(-30.0, 12.0, 0.0, DynMode::Down);
        assert_eq!(law.amount(-30.0), 0.0);
        assert!((law.amount(-24.0) - 0.5).abs() < 1e-5);
        assert_eq!(law.amount(-18.0), 1.0);
        // A knee below the minimum width is the same as no knee.
        let tiny = DynLaw::new(-30.0, 12.0, 0.05, DynMode::Down);
        assert!((tiny.amount(-24.0) - law.amount(-24.0)).abs() < 1e-6);
    }

    #[test]
    fn the_knee_works_in_upward_mode_too() {
        // Mirrored: the transition sits *above* the threshold, so the knee runs
        // from full gain at the threshold down to nothing `knee` dB above it.
        let law = DynLaw::new(-30.0, 12.0, 6.0, DynMode::Up);
        assert_eq!(law.amount(-36.0), 1.0);
        assert_eq!(law.amount(-30.0), 1.0, "the threshold is still full gain");
        assert_eq!(law.amount(-24.0), 0.0);
        let midpoint = law.amount(-27.0); // 3 dB above the threshold
        assert!(
            (midpoint - 0.25).abs() < 1e-5,
            "expected a quarter at the knee midpoint, got {midpoint}"
        );
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
