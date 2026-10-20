//! Dynamic EQ: does a band actually change gain with the signal?
//!
//! These tests mirror the shape of `Filter::process` — detector over a 16-sample
//! sub-block, bake a gain, process the sub-block with it — because a
//! plugin-level harness is blocked: `Plugin::process` takes a `ProcessContext`,
//! and nih-plug's `Transport` has no public constructor (the same wall that
//! keeps the spectrum tap out of a full end-to-end test).
//!
//! So what is proven here is the *algorithm* end to end: the detector, the gain
//! law, and a biquad, laid out the way the audio loop lays them out. What is not
//! proven is that `lib.rs` wires the same order — that part is read, not run.
//!
//! The properties under test are the ones that make a dynamic EQ usable:
//! loud material gets the gain, quiet material does not, and a tone outside the
//! band does not trigger the band at all.

use std::f32::consts::PI;

use filter::dsp::biquad::MAX_SECTIONS;
use filter::dsp::dynamics::{Detector, DynLaw, DynMode, SHELF_DETECT_Q, SIDECHAIN_Q};
use filter::dsp::{BiquadCoeffs, BiquadState};
use filter::FilterType;

const SR: f32 = 48_000.0;
/// The audio loop bakes coefficients every 16 samples; matching it here is the
/// point of the exercise.
const COEFF_BLOCK: usize = 16;

fn tone(hz: f32, dbfs: f32, secs: f32) -> Vec<f32> {
    let amp = 10f32.powf(dbfs / 20.0);
    let n = (SR * secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * hz * i as f32 / SR).sin() * amp)
        .collect()
}

/// RMS level of the last `secs` seconds, in dBFS. Restricted to a whole number
/// of cycles of `hz` so a steady tone measures its amplitude rather than a
/// window artefact.
fn steady_db(samples: &[f32], hz: f32, secs: f32) -> f32 {
    let samples_per_cycle = (SR / hz).round() as usize;
    let want = ((SR * secs) as usize / samples_per_cycle).max(1) * samples_per_cycle;
    let tail = &samples[samples.len().saturating_sub(want)..];
    let power: f32 = tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32;
    // RMS of a sine at amplitude A is A/sqrt(2), so a full-scale sine reads
    // about -3 dB here; add it back so the numbers are comparable to dBFS peak.
    let rms = power.sqrt();
    20.0 * (rms * std::f32::consts::SQRT_2).max(1e-9).log10()
}

/// One band, driven exactly the way the audio loop drives it.
struct DynBand {
    /// Coefficients and state are kept apart on purpose: `BiquadFilter`'s
    /// convenience `update_cascade` resets the state on every call, which is
    /// correct for a filter whose *shape* changes and wrong for one whose gain
    /// is re-baked 3000 times a second. The production loop splits them the
    /// same way, and using the convenience wrapper here silently cost ~7 dB of
    /// measured cut until the harness matched it.
    sections: [BiquadCoeffs; MAX_SECTIONS],
    count: usize,
    states: [BiquadState; MAX_SECTIONS],
    detector: Detector,
    freq: f32,
    q: f32,
    static_gain: f32,
    /// The dynamic law: threshold, transition width, knee, direction.
    law: DynLaw,
    /// Detector frequency when a sidechain filter is in use, with
    /// `SIDECHAIN_Q`. `None` means "listen to the band's own region".
    sidechain: Option<f32>,
    attack: f32,
    release: f32,
    filter_type: FilterType,
    /// Last gain actually baked, for tests that look at the trajectory.
    last_gain: f32,
}

impl DynBand {
    fn new(freq: f32, q: f32, static_gain: f32, threshold: f32, range: f32) -> Self {
        Self {
            sections: [BiquadCoeffs::PASSTHROUGH; MAX_SECTIONS],
            count: 0,
            states: [BiquadState::default(); MAX_SECTIONS],
            detector: Detector::new(SR),
            freq,
            q,
            static_gain,
            law: DynLaw::new(threshold, range, 0.0, DynMode::Down),
            sidechain: None,
            attack: 5.0,
            release: 150.0,
            filter_type: FilterType::Bell,
            last_gain: 0.0,
        }
    }

    fn with_timing(mut self, attack_ms: f32, release_ms: f32) -> Self {
        self.attack = attack_ms;
        self.release = release_ms;
        self
    }

    fn with_law(mut self, law: DynLaw) -> Self {
        self.law = law;
        self
    }

    fn with_sidechain(mut self, hz: f32) -> Self {
        self.sidechain = Some(hz);
        self
    }

    /// Process `input` (also used as the linked detector input, i.e. dry) and
    /// return the output.
    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        for sub in input.chunks(COEFF_BLOCK) {
            // Detector input is the dry signal, as in `process()`: the link is
            // measured before any band is applied.
            let (detect_freq, detect_q) = match self.sidechain {
                Some(hz) => (hz, SIDECHAIN_Q),
                None => (
                    self.freq,
                    if self.filter_type.is_bell_like() {
                        self.q
                    } else {
                        SHELF_DETECT_Q
                    },
                ),
            };
            let level = self
                .detector
                .update(sub, detect_freq, detect_q, self.attack, self.release);
            let gain = self.law.gain_db(self.static_gain, level);
            self.last_gain = gain;
            self.count = BiquadCoeffs::update_cascade(
                &mut self.sections,
                self.freq,
                gain,
                self.q,
                self.filter_type,
                12,
                SR,
            );
            out.extend(sub.iter().map(|&s| {
                let mut acc = s;
                for k in 0..self.count {
                    acc = self.states[k].process(&self.sections[k], acc);
                }
                acc
            }));
        }
        out
    }
}

#[test]
fn a_loud_tone_gets_the_full_cut() {
    // -12 dB dynamic bell at 1 kHz; threshold -30, range 12 dB.
    // A -10 dBFS tone is 20 dB over the threshold, i.e. past the top of the
    // range, so the full -12 dB must apply.
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    let input = tone(1_000.0, -10.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 1_000.0, 0.1);
    assert!(
        (level - -22.0).abs() < 1.0,
        "expected about -22 dBFS (in -10, cut -12), measured {level:.1}"
    );
}

#[test]
fn a_quiet_tone_passes_untouched() {
    // -45 dBFS is 15 dB *below* the threshold: the band must be inert, so the
    // output level equals the input level.
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    let input = tone(1_000.0, -45.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 1_000.0, 0.1);
    assert!(
        (level - -45.0).abs() < 1.0,
        "expected the input level back (-45), measured {level:.1}"
    );
    assert!(
        band.last_gain.abs() < 0.05,
        "a band below threshold should be at 0 dB, baked {:.2} dB",
        band.last_gain
    );
}

#[test]
fn a_tone_outside_the_band_does_not_trigger_it() {
    // Loud 8 kHz, but the band listens at 1 kHz with Q 1: the detector must not
    // hear it, so the cut stays closed and 8 kHz passes at full level. This is
    // the property that separates a dynamic EQ from a broadband compressor.
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    let input = tone(8_000.0, -10.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 8_000.0, 0.1);
    assert!(
        (level - -10.0).abs() < 1.0,
        "8 kHz should pass untouched, measured {level:.1}"
    );
}

#[test]
fn the_gain_ramps_in_over_the_range() {
    // 6 dB over a 12 dB range = half the gain: -12 dB * 0.5 = -6 dB.
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    let input = tone(1_000.0, -24.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 1_000.0, 0.1);
    assert!(
        (level - -30.0).abs() < 1.5,
        "expected about -30 dBFS (in -24, half the cut), measured {level:.1}"
    );
}

#[test]
fn attack_delays_the_cut_and_release_holds_it() {
    let mut fast = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0).with_timing(1.0, 50.0);
    let mut slow = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0).with_timing(100.0, 50.0);
    let input = tone(1_000.0, -10.0, 0.05);

    let fast_out = fast.process(&input);
    let slow_out = slow.process(&input);

    // 50 ms in, a 1 ms attack has long settled at the full cut while a 100 ms
    // attack is still on its way down.
    let fast_db = steady_db(&fast_out, 1_000.0, 0.01);
    let slow_db = steady_db(&slow_out, 1_000.0, 0.01);
    assert!(
        slow_db > fast_db + 2.0,
        "a 100 ms attack should still be cutting less than a 1 ms one at 50 ms \
         (slow {slow_db:.1} dB, fast {fast_db:.1} dB)"
    );

    // Release: after the tone stops, a 50 ms release drops the cut quickly.
    let silence = vec![0.0f32; (SR * 0.2) as usize];
    fast.process(&silence);
    assert!(
        fast.last_gain.abs() < 0.5,
        "the cut should have released after 200 ms of silence, baked {:.2} dB",
        fast.last_gain
    );
}

#[test]
fn a_boost_band_lifts_loud_material_only() {
    // Downward mode with a positive gain: the band boosts *above* the
    // threshold. (This is not what `DynMode::Up` does — see the upward tests
    // below, where the gain applies below the threshold instead.)
    let mut band = DynBand::new(1_000.0, 1.0, 6.0, -30.0, 12.0);
    let loud = tone(1_000.0, -10.0, 0.5);
    let loud_out = band.process(&loud);
    let loud_db = steady_db(&loud_out, 1_000.0, 0.1);
    assert!(
        (loud_db - -4.0).abs() < 1.0,
        "expected about -4 dBFS (in -10, boost +6), measured {loud_db:.1}"
    );

    let mut quiet_band = DynBand::new(1_000.0, 1.0, 6.0, -30.0, 12.0);
    let quiet = tone(1_000.0, -50.0, 0.5);
    let quiet_out = quiet_band.process(&quiet);
    let quiet_db = steady_db(&quiet_out, 1_000.0, 0.1);
    assert!(
        (quiet_db - -50.0).abs() < 1.0,
        "a boost below threshold must do nothing, measured {quiet_db:.1}"
    );
}

#[test]
fn extreme_settings_stay_finite() {
    // Widest range, lowest threshold, fastest attack, highest Q: nothing here
    // may produce a NaN, an inf, or a runaway output.
    let mut band = DynBand::new(1_000.0, 20.0, -24.0, -60.0, 30.0).with_timing(0.1, 5.0);
    let input = tone(1_000.0, 0.0, 0.2);
    let out = band.process(&input);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "dynamic band produced a non-finite sample"
    );
    let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(
        peak < 4.0,
        "a -24 dB cut with a 30 dB range should not produce a runaway output (peak {peak})"
    );
}

#[test]
fn a_high_q_band_is_narrow_in_level_as_well_as_in_frequency() {
    // Q 10 at 1 kHz: a 1.5 kHz tone is mostly outside the listening band, so
    // the cut must not be applied to it the way it is to a 1 kHz tone.
    let mut narrow = DynBand::new(1_000.0, 10.0, -12.0, -30.0, 12.0);
    let off_band = tone(1_500.0, -10.0, 0.5);
    let out = narrow.process(&off_band);
    let level = steady_db(&out, 1_500.0, 0.1);
    // The band still attenuates a little at 1.5 kHz, but far less than -12 dB.
    assert!(
        level > -17.0,
        "a Q 10 band should barely trigger on a 1.5 kHz tone, measured {level:.1} dBFS"
    );
}

// ---- upward mode --------------------------------------------------------

#[test]
fn upward_mode_lifts_quiet_material_and_leaves_loud_material_alone() {
    // +6 dB bell, Up, threshold -30, range 6: full gain at -36 and below,
    // nothing at -30 and above. This is upward compression — quiet parts come
    // up, loud parts are not made louder.
    let law = DynLaw::new(-30.0, 6.0, 0.0, DynMode::Up);

    let mut quiet_band = DynBand::new(1_000.0, 1.0, 6.0, -30.0, 6.0).with_law(law);
    let quiet = tone(1_000.0, -40.0, 0.5);
    let quiet_db = steady_db(&quiet_band.process(&quiet), 1_000.0, 0.1);
    assert!(
        (quiet_db - -34.0).abs() < 1.0,
        "a -40 dBFS tone should be lifted by the full +6 dB, measured {quiet_db:.1}"
    );
    assert!(
        (quiet_band.last_gain - 6.0).abs() < 0.5,
        "baked {:.2} dB, expected +6",
        quiet_band.last_gain
    );

    let mut loud_band = DynBand::new(1_000.0, 1.0, 6.0, -30.0, 6.0).with_law(law);
    let loud = tone(1_000.0, -10.0, 0.5);
    let loud_db = steady_db(&loud_band.process(&loud), 1_000.0, 0.1);
    assert!(
        (loud_db - -10.0).abs() < 1.0,
        "loud material must pass untouched in upward mode, measured {loud_db:.1}"
    );
    assert!(
        loud_band.last_gain.abs() < 0.05,
        "baked {:.2} dB on a loud tone, expected 0",
        loud_band.last_gain
    );
}

// ---- soft knee ----------------------------------------------------------

#[test]
fn a_knee_eases_the_cut_instead_of_stepping_at_the_threshold() {
    // -12 dB bell, threshold -30, 6 dB knee: a tone at -33 (3 dB below the
    // threshold, the knee's midpoint) gets a quarter of the cut, i.e. -3 dB.
    let law = DynLaw::new(-30.0, 12.0, 6.0, DynMode::Down);
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0).with_law(law);
    let input = tone(1_000.0, -33.0, 0.5);
    let level = steady_db(&band.process(&input), 1_000.0, 0.1);
    assert!(
        (level - -36.0).abs() < 2.0,
        "expected about -36 dBFS (in -33, cut -3), measured {level:.1}"
    );
    // The exact-quarter property is asserted on the law itself; here the
    // tolerance is the detector's, which reads a steady tone within about a
    // dB of the truth — a dB of level error near the knee midpoint moves the
    // baked gain by roughly a dB too.
    assert!(
        (band.last_gain - -3.0).abs() < 1.5,
        "baked {:.2} dB, expected about -3 (a quarter of -12)",
        band.last_gain
    );

    // ... and the same level with no knee is untouched, which is the step the
    // knee exists to remove.
    let mut hard = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    let hard_level = steady_db(&hard.process(&input), 1_000.0, 0.1);
    assert!(
        hard.last_gain.abs() < 0.1,
        "the hard-corner law should still be inert 3 dB below the threshold, \
         baked {:.2} dB (level {hard_level:.1})",
        hard.last_gain
    );
}

// ---- sidechain filter ---------------------------------------------------

#[test]
fn a_sidechain_filter_triggers_the_band_from_another_part_of_the_spectrum() {
    // The band sits at 1 kHz, where there is nothing to hear. The detector is
    // moved to 8 kHz, where a loud tone is playing: the band must bake its cut
    // even though its own region is silent.
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0).with_sidechain(8_000.0);
    let loud = tone(8_000.0, -10.0, 0.5);
    band.process(&loud);
    assert!(
        (band.last_gain - -12.0).abs() < 1.0,
        "a sidechained band should reach its full cut, baked {:.2} dB",
        band.last_gain
    );

    // Without the sidechain the same input must not move it: that is the
    // difference between this and a broadband detector.
    let mut no_sidechain = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    no_sidechain.process(&loud);
    // A Q 1 band-pass at 1 kHz still passes some 8 kHz — about 1.5% of the
    // power, which is a fraction of a dB of the cut. What must not happen is
    // the band reaching anything like its full range.
    assert!(
        no_sidechain.last_gain.abs() < 2.0,
        "without a sidechain an 8 kHz tone must not trigger a 1 kHz band, baked {:.2} dB",
        no_sidechain.last_gain
    );
}

#[test]
fn a_sidechain_filter_ignores_the_bands_own_content() {
    // The mirror image: band at 1 kHz, detector at 8 kHz, and the *only*
    // content is 1 kHz. The band's own region is loud, but the detector is
    // listening elsewhere, so nothing is applied.
    let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0).with_sidechain(8_000.0);
    let own = tone(1_000.0, -10.0, 0.5);
    band.process(&own);
    assert!(
        band.last_gain.abs() < 0.7,
        "the detector is sidechained away from 1 kHz and must ignore it, baked {:.2} dB",
        band.last_gain
    );
}

#[test]
fn a_knee_moves_the_transition_start_below_the_threshold() {
    // The audible difference from a hard corner: the band begins engaging
    // *before* the threshold is reached, so material approaching it eases in
    // instead of arriving at the corner. Measured across the knee in audio.
    let law = DynLaw::new(-30.0, 12.0, 6.0, DynMode::Down);
    let baked = |dbfs: f32| {
        let mut band = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0).with_law(law);
        band.process(&tone(1_000.0, dbfs, 0.3));
        band.last_gain
    };

    let below_knee = baked(-36.5); // outside the knee: nothing
    let middle = baked(-33.0); // knee midpoint: about a quarter
    let at_threshold = baked(-30.0); // top of the knee: full

    assert!(
        below_knee.abs() < 0.5,
        "below the knee the band must be inert, baked {below_knee:.2} dB"
    );
    let fraction = middle / -12.0;
    assert!(
        (0.15..0.40).contains(&fraction),
        "at the knee midpoint the band should apply about a quarter of its \
         range, applied {fraction:.2}"
    );
    // Not *exactly* -12: the detector reads a steady tone ~0.6 dB low, and the
    // top of the knee is the steepest part of the curve, so that offset shows
    // up here as about 2 dB of gain. Three quarters of the range is the claim.
    assert!(
        at_threshold < -9.0,
        "the top of the knee should be nearly the full -12 dB, baked {at_threshold:.2}"
    );
    assert!(
        below_knee > middle && middle > at_threshold,
        "the knee must engage progressively, got {below_knee:.2} / {middle:.2} / \
         {at_threshold:.2} dB"
    );

    // The hard-corner law at the same midpoint applies nothing at all, which is
    // the step the knee removes.
    let mut hard = DynBand::new(1_000.0, 1.0, -12.0, -30.0, 12.0);
    hard.process(&tone(1_000.0, -33.0, 0.3));
    assert!(
        hard.last_gain.abs() < 0.3,
        "the hard corner should still be inert at -33, baked {:.2} dB",
        hard.last_gain
    );
}
