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
//! (`src/audio_path.rs` runs the real plugin and covers the wiring.)
//!
//! The model under test is Pro-Q's: a band's static gain and its dynamic range
//! are two independent parameters, live at the same time, and the sign of the
//! range is the direction. Triggering is one-sided — above the threshold — with
//! a knee that eases the arrival from below it.

use std::f32::consts::PI;

use filter::dsp::biquad::MAX_SECTIONS;
use filter::dsp::dynamics::{Detector, DynLaw, DEFAULT_KNEE_DB, SHELF_DETECT_Q, SIDECHAIN_Q};
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
    /// The band's static gain — applied at every level, like a normal band.
    static_gain: f32,
    /// The dynamic part: threshold, signed range, knee.
    law: DynLaw,
    /// Detector frequency when a sidechain filter is in use, with
    /// `SIDECHAIN_Q`. `None` means "listen to the band's own region".
    sidechain: Option<f32>,
    attack: f32,
    release: f32,
    filter_type: FilterType,
    /// The total gain last baked, for tests that look at the trajectory.
    last_gain: f32,
    /// The dynamic part of it alone.
    last_dynamic: f32,
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
            law: DynLaw::new(threshold, range, DEFAULT_KNEE_DB),
            sidechain: None,
            attack: 5.0,
            release: 150.0,
            filter_type: FilterType::Bell,
            last_gain: 0.0,
            last_dynamic: 0.0,
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
            self.last_dynamic = self.law.dynamic_gain_db(level);
            let gain = self.law.applied_gain_db(self.static_gain, level);
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

// ---- compression: the band ducks when the trigger is loud -----------------

#[test]
fn a_loud_tone_gets_the_full_cut() {
    // No static gain; a -12 dB range. A -10 dBFS tone is well past the -30 dB
    // threshold, so the whole range applies.
    let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0);
    let input = tone(1_000.0, -10.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 1_000.0, 0.1);
    assert!(
        (level - -22.0).abs() < 1.0,
        "expected about -22 dBFS (in -10, cut -12), measured {level:.1}"
    );
    assert!(
        (band.last_dynamic - -12.0).abs() < 0.5,
        "the dynamic part should be the full -12 dB, applied {:.2}",
        band.last_dynamic
    );
}

#[test]
fn a_quiet_tone_passes_untouched() {
    // -45 dBFS is 15 dB below the threshold: nothing dynamic, and no static
    // gain either, so the output equals the input.
    let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0);
    let input = tone(1_000.0, -45.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 1_000.0, 0.1);
    assert!(
        (level - -45.0).abs() < 1.0,
        "expected the input level back (-45), measured {level:.1}"
    );
    assert!(
        band.last_dynamic.abs() < 0.05,
        "a band below threshold should apply nothing dynamic, applied {:.2} dB",
        band.last_dynamic
    );
}

#[test]
fn a_tone_outside_the_band_does_not_trigger_it() {
    // 8 kHz at -20 dBFS, but the band listens at 1 kHz with Q 1: the detector
    // must not hear it, so the range stays closed and 8 kHz passes at full
    // level. This is the property that separates a dynamic EQ from a broadband
    // compressor.
    //
    // The tone is deliberately *not* at -10 dBFS: a Q 1 band-pass three octaves
    // away only attenuates about 18 dB, and with a threshold at -30 a -10 dBFS
    // input leaks to -28 and lands inside the knee. The property under test is
    // "content the detector is not listening to does not trigger the band",
    // which needs headroom to be measurable at all.
    let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0);
    let input = tone(8_000.0, -20.0, 0.5);
    let out = band.process(&input);

    let level = steady_db(&out, 8_000.0, 0.1);
    assert!(
        (level - -20.0).abs() < 1.0,
        "8 kHz should pass untouched, measured {level:.1}"
    );
    assert!(
        band.last_dynamic.abs() < 0.5,
        "8 kHz must not trigger a 1 kHz band, applied {:.2} dB",
        band.last_dynamic
    );
}

// ---- the two parameters are independent ----------------------------------

/// The property the Pro-Q model exists for: a band can lift all the time and
/// duck when it gets loud, because the static gain and the range are separate.
///
/// Under the model #14 shipped (gain = the maximum change) this setting was
/// inexpressible: the knob either lifted or ducked, never both.
#[test]
fn a_static_lift_and_a_dynamic_duck_are_both_live() {
    let mut band = DynBand::new(1_000.0, 1.0, 6.0, -30.0, -12.0);

    let quiet = tone(1_000.0, -45.0, 0.5);
    let quiet_db = steady_db(&band.process(&quiet), 1_000.0, 0.1);
    assert!(
        (quiet_db - -39.0).abs() < 1.0,
        "quiet material should still get the static +6 dB, measured {quiet_db:.1}"
    );

    // A fresh band: the envelope from the quiet pass would otherwise still be
    // decaying when the loud tone starts.
    let mut loud_band = DynBand::new(1_000.0, 1.0, 6.0, -30.0, -12.0);
    let loud = tone(1_000.0, -10.0, 0.5);
    let loud_db = steady_db(&loud_band.process(&loud), 1_000.0, 0.1);
    assert!(
        (loud_db - -16.0).abs() < 1.5,
        "loud material should get +6 static and -12 dynamic (-6 net), measured {loud_db:.1}"
    );
    assert!(
        (loud_band.last_gain - -6.0).abs() < 1.0,
        "baked {:.2} dB, expected about -6",
        loud_band.last_gain
    );
}

// ---- expansion: the band lifts when the trigger is loud ------------------

/// #20's acceptance criterion, in real audio: "a loud signal is boosted, a
/// quiet one untouched". This is a positive range — Pro-Q's expansion — and it
/// is a different setting from a static boost, which lifts everything.
#[test]
fn a_positive_range_boosts_loud_material_and_leaves_quiet_material_alone() {
    let mut loud_band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, 6.0);
    let loud = tone(1_000.0, -10.0, 0.5);
    let loud_db = steady_db(&loud_band.process(&loud), 1_000.0, 0.1);
    assert!(
        (loud_db - -4.0).abs() < 1.0,
        "expected about -4 dBFS (in -10, expansion +6), measured {loud_db:.1}"
    );

    let mut quiet_band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, 6.0);
    let quiet = tone(1_000.0, -50.0, 0.5);
    let quiet_db = steady_db(&quiet_band.process(&quiet), 1_000.0, 0.1);
    assert!(
        (quiet_db - -50.0).abs() < 1.0,
        "a quiet signal must not be lifted by expansion, measured {quiet_db:.1}"
    );
    assert!(
        quiet_band.last_dynamic.abs() < 0.05,
        "expansion below the threshold should apply nothing, applied {:.2}",
        quiet_band.last_dynamic
    );
}

#[test]
fn the_sign_of_the_range_decides_the_direction() {
    // Same level, same threshold, opposite sign: one lifts, one ducks, and the
    // magnitudes match. There is no mode switch to get out of step with this.
    let mut up = DynBand::new(1_000.0, 1.0, 0.0, -30.0, 8.0);
    let mut down = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -8.0);
    let input = tone(1_000.0, -8.0, 0.4);
    up.process(&input);
    down.process(&input);
    assert!(
        (up.last_dynamic - 8.0).abs() < 0.5 && (down.last_dynamic + 8.0).abs() < 0.5,
        "expected +8 / -8, got {:+.2} / {:+.2}",
        up.last_dynamic,
        down.last_dynamic
    );
}

// ---- knee ---------------------------------------------------------------

#[test]
fn the_gain_ramps_in_over_the_knee() {
    // 6 dB knee, threshold -30, range -12: the knee's midpoint is 3 dB below
    // the threshold, where a quarter of the range applies — about -3 dB.
    let law = DynLaw::new(-30.0, -12.0, 6.0);
    let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_law(law);
    let level = steady_db(&band.process(&tone(1_000.0, -33.0, 0.5)), 1_000.0, 0.1);
    assert!(
        (level - -36.0).abs() < 2.0,
        "expected about -36 dBFS (in -33, cut -3), measured {level:.1}"
    );
    // The exact-quarter property is asserted on the law; here the tolerance is
    // the detector's, which reads a steady tone within about a dB.
    assert!(
        (band.last_dynamic - -3.0).abs() < 1.5,
        "applied {:.2} dB, expected about -3 (a quarter of -12)",
        band.last_dynamic
    );

    // The hard corner at the same level applies nothing at all, which is the
    // step the knee removes.
    let mut hard =
        DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_law(DynLaw::new(-30.0, -12.0, 0.0));
    hard.process(&tone(1_000.0, -33.0, 0.5));
    assert!(
        hard.last_dynamic.abs() < 0.3,
        "a hard corner should be inert at -33, applied {:.2} dB",
        hard.last_dynamic
    );
}

#[test]
fn a_knee_moves_the_transition_start_below_the_threshold() {
    // The audible difference from a hard corner: the band begins engaging
    // *before* the threshold is reached, so material approaching it eases in.
    let law = DynLaw::new(-30.0, -12.0, 6.0);
    let applied = |dbfs: f32| {
        let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_law(law);
        band.process(&tone(1_000.0, dbfs, 0.3));
        band.last_dynamic
    };

    let below_knee = applied(-36.5); // outside the knee: nothing
    let middle = applied(-33.0); // knee midpoint: about a quarter
    let at_threshold = applied(-30.0); // top of the knee: the full range

    assert!(
        below_knee.abs() < 0.5,
        "below the knee the band must be inert, applied {below_knee:.2} dB"
    );
    let fraction = middle / -12.0;
    assert!(
        (0.15..0.40).contains(&fraction),
        "at the knee midpoint the band should apply about a quarter of the \\
         range, applied {fraction:.2}"
    );
    // Not exactly -12: the detector reads a steady tone ~0.6 dB low, and the
    // top of the knee is the steepest part of the curve, so that offset shows
    // up here as about 2 dB. Three quarters of the range is the claim.
    assert!(
        at_threshold < -9.0,
        "the top of the knee should be nearly the full -12 dB, applied {at_threshold:.2}"
    );
    assert!(
        below_knee > middle && middle > at_threshold,
        "the knee must engage progressively, got {below_knee:.2} / {middle:.2} / \\
         {at_threshold:.2} dB"
    );
}

// ---- timing -------------------------------------------------------------

#[test]
fn attack_delays_the_cut_and_release_holds_it() {
    // The attack is visible only while the envelope is still climbing towards
    // the threshold, so the tone sits 10 dB over it and the comparison is the
    // applied gain after 10 ms — long enough for a 1 ms attack to have arrived,
    // short enough for a 100 ms one to be a tenth of the way there.
    let first_10ms = (SR * 0.01) as usize;
    let tone_20 = tone(1_000.0, -20.0, 0.1);

    let mut fast = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_timing(1.0, 50.0);
    fast.process(&tone_20[..first_10ms]);
    let fast_dynamic = fast.last_dynamic;

    let mut slow = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_timing(100.0, 50.0);
    slow.process(&tone_20[..first_10ms]);
    let slow_dynamic = slow.last_dynamic;

    assert!(
        fast_dynamic < -10.0,
        "a 1 ms attack should be at nearly the full range after 10 ms, applied {fast_dynamic:.1} dB"
    );
    assert!(
        slow_dynamic > -6.0,
        "a 100 ms attack should have barely engaged after 10 ms, applied {slow_dynamic:.1} dB"
    );

    // Release: the same band, let go of the signal, drops back.
    let silence = vec![0.0f32; (SR * 0.2) as usize];
    fast.process(&silence);
    assert!(
        fast.last_dynamic.abs() < 0.5,
        "the range should have released after 200 ms of silence, applied {:.2} dB",
        fast.last_dynamic
    );
}

// ---- robustness ---------------------------------------------------------

#[test]
fn extreme_settings_stay_finite() {
    // Widest range, lowest threshold, fastest attack, highest Q: nothing here
    // may produce a NaN, an inf, or a runaway output.
    let mut band = DynBand::new(1_000.0, 20.0, -6.0, -60.0, -30.0).with_timing(0.1, 5.0);
    let input = tone(1_000.0, 0.0, 0.2);
    let out = band.process(&input);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "dynamic band produced a non-finite sample"
    );
    let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(
        peak < 4.0,
        "a -30 dB range should not produce a runaway output (peak {peak})"
    );
}

#[test]
fn a_high_q_band_is_narrow_in_level_as_well_as_in_frequency() {
    // Q 10 at 1 kHz: a 1.5 kHz tone is mostly outside the listening band, so
    // the cut must not be applied to it the way it is to a 1 kHz tone.
    let mut narrow = DynBand::new(1_000.0, 10.0, 0.0, -30.0, -12.0);
    let off_band = tone(1_500.0, -10.0, 0.5);
    let out = narrow.process(&off_band);
    let level = steady_db(&out, 1_500.0, 0.1);
    // The band still attenuates a little at 1.5 kHz, but far less than -12 dB.
    assert!(
        level > -17.0,
        "a Q 10 band should barely trigger on a 1.5 kHz tone, measured {level:.1} dBFS"
    );
}

// ---- sidechain filter ---------------------------------------------------

#[test]
fn a_sidechain_filter_triggers_the_band_from_another_part_of_the_spectrum() {
    // The band sits at 1 kHz, where there is nothing to hear. The detector is
    // moved to 8 kHz, where a loud tone is playing: the band must bake its cut
    // even though its own region is silent.
    let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_sidechain(8_000.0);
    band.process(&tone(8_000.0, -10.0, 0.5));
    assert!(
        (band.last_dynamic - -12.0).abs() < 1.0,
        "a sidechained band should reach its full range, applied {:.2} dB",
        band.last_dynamic
    );

    // Without the sidechain the same input must not move it: that is the
    // difference between this and a broadband detector.
    let mut no_sidechain = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0);
    no_sidechain.process(&tone(8_000.0, -20.0, 0.5));
    // A Q 1 band-pass at 1 kHz still passes some 8 kHz — about 1.5% of the
    // power, a fraction of a dB of the range. What must not happen is the band
    // reaching anything like its full travel.
    assert!(
        no_sidechain.last_dynamic.abs() < 2.0,
        "without a sidechain an 8 kHz tone must not trigger a 1 kHz band, applied {:.2} dB",
        no_sidechain.last_dynamic
    );
}

#[test]
fn a_sidechain_filter_ignores_the_bands_own_content() {
    // The mirror image: band at 1 kHz, detector at 8 kHz, and the *only*
    // content is 1 kHz. The band's own region is loud, but the detector is
    // listening elsewhere, so nothing is applied.
    let mut band = DynBand::new(1_000.0, 1.0, 0.0, -30.0, -12.0).with_sidechain(8_000.0);
    band.process(&tone(1_000.0, -20.0, 0.5));
    assert!(
        band.last_dynamic.abs() < 0.7,
        "the detector is sidechained away from 1 kHz and must ignore it, applied {:.2} dB",
        band.last_dynamic
    );
}
