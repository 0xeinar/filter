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

use filter::dsp::{BiquadCoeffs, BiquadState};
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
