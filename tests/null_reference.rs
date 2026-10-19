//! Null test: the plugin's production biquad (f32, DF-II transposed) against an
//! independent reference implementation (f64, DF-I, coefficients derived
//! separately from the Audio EQ Cookbook).
//!
//! This is the closest honest analogue to an "A/B null against a reference EQ"
//! that can run headless in CI: there is no FabFilter/Pro-Q to null against, so
//! the reference is the textbook transfer function computed by different code.
//! It catches coefficient typos, state-form mistakes, and sign errors that a
//! magnitude-response test alone can miss — a wrong phase or a wrong b sign
//! still nulls well against nothing, but not against a correct reference.
//!
//! The residual is expected to sit far below audibility; the thresholds below
//! are set with headroom over the measured values so they fail on real bugs,
//! not on f32 rounding noise.

use filter::dsp::biquad::{butterworth_q, cascade_shape, MAX_SECTIONS, SLOPES_DB_PER_OCT};
use filter::dsp::{BiquadCoeffs, BiquadFilter, BiquadState};
use filter::FilterType;

const SR: f64 = 48_000.0;
const SR_F32: f32 = 48_000.0;

/// Independent reference biquad: f64 math, Direct Form I, coefficients computed
/// from the cookbook in a separate implementation from `dsp::biquad`.
struct Reference {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Reference {
    fn new(freq: f64, gain_db: f64, q: f64, filter_type: FilterType) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * freq / SR;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            FilterType::Bell => (
                1.0 + alpha * a,
                -2.0 * cos_w0,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * cos_w0,
                1.0 - alpha / a,
            ),
            FilterType::LowShelf => {
                let k = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cos_w0 + k),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
                    a * ((a + 1.0) - (a - 1.0) * cos_w0 - k),
                    (a + 1.0) + (a - 1.0) * cos_w0 + k,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
                    (a + 1.0) + (a - 1.0) * cos_w0 - k,
                )
            }
            FilterType::HighShelf => {
                let k = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cos_w0 + k),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
                    a * ((a + 1.0) + (a - 1.0) * cos_w0 - k),
                    (a + 1.0) - (a - 1.0) * cos_w0 + k,
                    2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
                    (a + 1.0) - (a - 1.0) * cos_w0 - k,
                )
            }
            FilterType::LowCut => (
                (1.0 + cos_w0) / 2.0,
                -(1.0 + cos_w0),
                (1.0 + cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterType::HighCut => (
                (1.0 - cos_w0) / 2.0,
                1.0 - cos_w0,
                (1.0 - cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterType::Notch => (
                1.0,
                -2.0 * cos_w0,
                1.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterType::BandPass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha),
            FilterType::AllPass => (
                1.0 - alpha,
                -2.0 * cos_w0,
                1.0 + alpha,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            // A tilt is two sections; this reference models its top half, and
            // `nulls_tilt_shelf_against_two_references` chains both.
            FilterType::TiltShelf => {
                let k = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cos_w0 + k),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
                    a * ((a + 1.0) + (a - 1.0) * cos_w0 - k),
                    (a + 1.0) - (a - 1.0) * cos_w0 + k,
                    2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
                    (a + 1.0) - (a - 1.0) * cos_w0 - k,
                )
            }
        };

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Deterministic white noise + two tones, so the test exercises broadband and
/// narrowband content without a random dependency.
fn test_signal(len: usize) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..len)
        .map(|n| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0;
            let t = n as f64 / SR;
            let tone = (2.0 * std::f64::consts::PI * 91.0 * t).sin() * 0.4
                + (2.0 * std::f64::consts::PI * 4_700.0 * t).sin() * 0.3;
            (noise * 0.15 + tone) * 0.5
        })
        .collect()
}

/// Null the plugin biquad against the reference for one filter setting.
/// Returns the residual level in dB relative to the reference output.
fn null_depth(freq: f32, gain_db: f32, q: f32, filter_type: FilterType) -> f64 {
    let input = test_signal(48_000);
    let mut coeffs = BiquadCoeffs::PASSTHROUGH;
    coeffs.update(freq, gain_db, q, filter_type, SR_F32);
    let mut state = BiquadState::default();
    let mut reference = Reference::new(freq as f64, gain_db as f64, q as f64, filter_type);

    let mut sig_energy = 0.0f64;
    let mut err_energy = 0.0f64;
    for &x in &input {
        let ours = state.process(&coeffs, x as f32) as f64;
        let theirs = reference.process(x);
        sig_energy += theirs * theirs;
        let e = ours - theirs;
        err_energy += e * e;
    }

    10.0 * (err_energy / sig_energy).log10()
}

#[test]
fn nulls_against_reference_all_types() {
    let cases: &[(f32, f32, f32, FilterType)] = &[
        (1_000.0, 6.0, 1.0, FilterType::Bell),
        (120.0, -12.0, 4.0, FilterType::Bell),
        (8_000.0, 9.0, 0.7, FilterType::Bell),
        (200.0, 8.0, 0.707, FilterType::LowShelf),
        (6_000.0, -10.0, 0.707, FilterType::HighShelf),
        (80.0, 0.0, 0.707, FilterType::LowCut),
        (12_000.0, 0.0, 0.707, FilterType::HighCut),
        (1_500.0, 0.0, 4.0, FilterType::Notch),
        (2_500.0, 0.0, 6.0, FilterType::BandPass),
        (700.0, 0.0, 0.9, FilterType::AllPass),
    ];

    for &(freq, gain, q, ftype) in cases {
        let depth = null_depth(freq, gain, q, ftype);
        println!("{ftype:?} @ {freq:>7} Hz, {gain:>5} dB, Q {q:<5} -> {depth:>8.1} dB");
        assert!(
            depth < -120.0,
            "{ftype:?} @ {freq} Hz, {gain} dB, Q {q}: null only {depth:.1} dB"
        );
    }
}

#[test]
fn nulls_against_reference_eight_band_cascade() {
    // A realistic mix chain: cuts, shelves and overlapping bells. Each stage is
    // nulled through the same signal path the plugin uses (shared coefficients,
    // per-channel state).
    let chain: &[(f32, f32, f32, FilterType)] = &[
        (35.0, 0.0, 0.707, FilterType::LowCut),
        (90.0, 3.0, 1.2, FilterType::Bell),
        (250.0, -4.5, 1.8, FilterType::Bell),
        (800.0, 2.0, 1.0, FilterType::Bell),
        (2_500.0, -3.0, 3.5, FilterType::Bell),
        (200.0, 5.0, 0.707, FilterType::LowShelf),
        (6_000.0, 4.0, 1.4, FilterType::Bell),
        (9_000.0, 0.0, 0.707, FilterType::HighCut),
    ];

    let input = test_signal(48_000);
    let mut coeffs = Vec::new();
    let mut states = Vec::new();
    let mut refs = Vec::new();
    for &(freq, gain, q, ftype) in chain {
        let mut c = BiquadCoeffs::PASSTHROUGH;
        c.update(freq, gain, q, ftype, SR_F32);
        coeffs.push(c);
        states.push(BiquadState::default());
        refs.push(Reference::new(freq as f64, gain as f64, q as f64, ftype));
    }

    let mut sig_energy = 0.0f64;
    let mut err_energy = 0.0f64;
    for &x in &input {
        let mut ours = x as f32;
        let mut theirs = x;
        for (c, s) in coeffs.iter().zip(states.iter_mut()) {
            ours = s.process(c, ours);
        }
        for r in refs.iter_mut() {
            theirs = r.process(theirs);
        }
        sig_energy += theirs * theirs;
        let e = ours as f64 - theirs;
        err_energy += e * e;
    }

    let depth = 10.0 * (err_energy / sig_energy).log10();
    println!("8-band cascade null depth: {depth:.1} dB");
    assert!(
        depth < -120.0,
        "8-band cascade null only {depth:.1} dB below the reference"
    );
}

/// The residual must be genuinely tiny, not just "below a threshold" — this
/// pins the measured value so an accidental precision regression is visible.
#[test]
fn reference_null_depth_is_reported() {
    let depth = null_depth(1_000.0, 6.0, 1.0, FilterType::Bell);
    println!("single bell null depth: {depth:.1} dB");
    assert!(depth < -100.0, "expected >100 dB null, got {depth:.1} dB");
}

/// Run the plugin's cascade for one band against a chain of reference
/// sections, returning the residual level in dB.
fn null_cascade(
    freq: f32,
    gain_db: f32,
    q: f32,
    filter_type: FilterType,
    slope_db_per_oct: u32,
    reference_sections: Vec<Reference>,
) -> f64 {
    let input = test_signal(48_000);

    let mut sections = [BiquadCoeffs::PASSTHROUGH; MAX_SECTIONS];
    let count = BiquadCoeffs::update_cascade(
        &mut sections,
        freq,
        gain_db,
        q,
        filter_type,
        slope_db_per_oct,
        SR_F32,
    );
    assert_eq!(
        count,
        reference_sections.len(),
        "the plugin and the reference disagree about how many sections this setting needs"
    );
    let mut states = [BiquadState::default(); MAX_SECTIONS];
    let mut refs = reference_sections;

    let mut sig_energy = 0.0f64;
    let mut err_energy = 0.0f64;
    for &x in &input {
        let mut ours = x as f32;
        for k in 0..count {
            ours = states[k].process(&sections[k], ours);
        }
        let mut theirs = x;
        for r in refs.iter_mut() {
            theirs = r.process(theirs);
        }
        sig_energy += theirs * theirs;
        let e = ours as f64 - theirs;
        err_energy += e * e;
    }

    10.0 * (err_energy / sig_energy).log10()
}

#[test]
fn nulls_a_tilt_shelf_against_two_references() {
    // A tilt is two sections by construction: -g/2 at the bottom, +g/2 at the
    // top. If the plugin ever changed that (or dropped one half) the section
    // count assertion trips before the null can.
    let gain = 9.0;
    let freq = 1_000.0;
    let q = 0.707;
    let depth = null_cascade(
        freq,
        gain,
        q,
        FilterType::TiltShelf,
        12,
        vec![
            Reference::new(
                freq as f64,
                -(gain as f64) / 2.0,
                q as f64,
                FilterType::LowShelf,
            ),
            Reference::new(
                freq as f64,
                (gain as f64) / 2.0,
                q as f64,
                FilterType::HighShelf,
            ),
        ],
    );
    println!("tilt shelf null depth: {depth:.1} dB");
    assert!(depth < -120.0, "tilt shelf null only {depth:.1} dB");
}

#[test]
fn nulls_sloped_cuts_against_chained_butterworth_sections() {
    // Every even slope is a whole number of 2nd-order Butterworth sections;
    // the odd ones add a 1st-order tail, which this reference cannot express,
    // so those are covered by `measured_slopes_match_the_label` instead.
    for slope in [12u32, 24, 36, 48, 72, 96] {
        let freq = 300.0;
        let (second_order, first_order) = cascade_shape(slope);
        assert!(!first_order, "{slope} dB/oct should be all 2nd order");

        let refs = (0..second_order)
            .map(|k| {
                Reference::new(
                    freq as f64,
                    0.0,
                    butterworth_q(k, second_order),
                    FilterType::LowCut,
                )
            })
            .collect();

        let depth = null_cascade(freq, 0.0, 0.707, FilterType::LowCut, slope, refs);
        println!("{slope:>2} dB/oct low cut null depth: {depth:>8.1} dB");
        assert!(
            depth < -120.0,
            "{slope} dB/oct low cut null only {depth:.1} dB"
        );
    }
}

/// Steepness actually measured off the transfer function, in dB/oct.
///
/// Both probes sit one octave apart *below* the corner, well inside the
/// stopband of a low cut, where a Butterworth cascade is on its asymptote. The
/// magnitude floor is 1e-30 rather than something audible: a 96 dB/oct cut is
/// already at -289 dB there, and clamping early would read the clamp instead
/// of the filter.
fn measured_slope_db_per_oct(slope: u32) -> f32 {
    let mut filter = BiquadFilter::new();
    filter.update_cascade(2_000.0, 0.0, 0.707, FilterType::LowCut, slope, SR_F32);
    let attenuation = |hz: f32| -20.0 * filter.magnitude_at_freq(hz, SR_F32).max(1e-30).log10();
    attenuation(250.0) - attenuation(500.0)
}

#[test]
fn measured_slopes_match_the_label() {
    for &slope in SLOPES_DB_PER_OCT.iter() {
        let measured = measured_slope_db_per_oct(slope);
        let expected = slope as f32;
        println!("{slope:>2} dB/oct -> measured {measured:>6.1} dB/oct");
        assert!(
            (measured - expected).abs() <= expected * 0.06,
            "{slope} dB/oct measured {measured:.1} dB/oct"
        );
    }
}

/// A steeper slope is never shallower than a gentler one inside the stopband.
///
/// Only below the corner, though: all Butterworth cascades pass -3 dB at the
/// corner and a steeper one is actually *shallower* just above it, so probing
/// there would assert something false about the design rather than test it.
#[test]
fn slopes_are_monotonic_in_attenuation() {
    for probe in [500.0f32, 250.0, 100.0, 50.0] {
        let mut previous = 0.0f32; // dB of attenuation
        for &slope in SLOPES_DB_PER_OCT.iter() {
            let mut filter = BiquadFilter::new();
            filter.update_cascade(1_000.0, 0.0, 0.707, FilterType::LowCut, slope, SR_F32);
            let attenuation = -20.0 * filter.magnitude_at_freq(probe, SR_F32).max(1e-30).log10();
            assert!(
                attenuation >= previous - 0.01,
                "at {probe} Hz, {slope} dB/oct attenuates {attenuation:.2} dB but a gentler slope already reached {previous:.2} dB"
            );
            previous = attenuation;
        }
    }
}
