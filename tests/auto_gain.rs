//! Does auto-gain actually keep the level steady?
//!
//! `auto_gain_compensation_db()` estimates the level change from the EQ curve
//! (mean gain over the audio band, weighted uniformly in log frequency = pink).
//! That is a model, not a measurement, so this test measures the real thing:
//! run pink noise through the same biquads the plugin uses, measure the RMS
//! change, and check that the compensation cancels it.

use filter::dsp::curve::{auto_gain_compensation_db, BandSpec};
use filter::dsp::BiquadFilter;
use filter::{FilterType, MAX_BANDS};

const SR: f32 = 48_000.0;
const SETTLE: usize = 4_000;

fn specs(bands: &[(f32, f32, f32, FilterType)]) -> [BandSpec; MAX_BANDS] {
    core::array::from_fn(|i| match bands.get(i) {
        Some(&(freq, gain_db, q, filter_type)) => BandSpec {
            freq,
            gain_db,
            q,
            filter_type,
            enabled: true,
        },
        None => BandSpec {
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        },
    })
}

/// Pink-ish noise (Paul Kellet's refined method) — the weighting auto-gain
/// assumes, and a fair stand-in for music.
fn pink_noise(len: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6) =
        (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let white = ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            b0 = 0.99886 * b0 + white * 0.0555179;
            b1 = 0.99332 * b1 + white * 0.0750759;
            b2 = 0.96900 * b2 + white * 0.153_852;
            b3 = 0.86650 * b3 + white * 0.310_485_6;
            b4 = 0.55000 * b4 + white * 0.5329522;
            b5 = -0.7616 * b5 - white * 0.0168980;
            let pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
            b6 = white * 0.115926;
            pink * 0.11
        })
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Level change in dB caused by the EQ chain, measured on pink noise.
fn measured_level_change_db(bands: &[(f32, f32, f32, FilterType)]) -> f32 {
    let input = pink_noise(SR as usize);
    let mut filters: Vec<BiquadFilter> = bands
        .iter()
        .map(|&(freq, gain, q, t)| {
            let mut f = BiquadFilter::new();
            f.update_coefficients(freq, gain, q, t, SR);
            f
        })
        .collect();

    let mut output = Vec::with_capacity(input.len());
    for &x in &input {
        let mut s = x;
        for f in filters.iter_mut() {
            s = f.process(s);
        }
        output.push(s);
    }

    let in_rms = rms(&input[SETTLE..]);
    let out_rms = rms(&output[SETTLE..]);
    20.0 * (out_rms / in_rms).log10()
}

#[test]
fn auto_gain_cancels_a_broad_shelf_boost() {
    let bands = [(500.0, 12.0, 0.707, FilterType::LowShelf)];
    let change = measured_level_change_db(&bands);
    let comp = auto_gain_compensation_db(&specs(&bands), 1);

    assert!(
        change > 4.0,
        "expected a clear level increase, got {change:.2} dB"
    );
    let residual = (change + comp).abs();
    assert!(
        residual < 1.5,
        "after compensation the level should be within 1.5 dB: \
         measured {change:+.2} dB, compensation {comp:+.2} dB, residual {residual:.2} dB"
    );
}

#[test]
fn auto_gain_cancels_a_broad_cut() {
    let bands = [(1200.0, -12.0, 0.707, FilterType::HighShelf)];
    let change = measured_level_change_db(&bands);
    let comp = auto_gain_compensation_db(&specs(&bands), 1);

    // A -12 dB shelf above 1.2 kHz only covers ~40% of the octaves, so pink
    // weighting puts the expected drop near -2 dB, not -12.
    assert!(
        change < -1.0,
        "expected a clear level drop, got {change:.2} dB"
    );
    let residual = (change + comp).abs();
    assert!(
        residual < 1.5,
        "residual {residual:.2} dB (measured {change:+.2}, comp {comp:+.2})"
    );
}

#[test]
fn auto_gain_is_gentle_on_a_symmetric_mix_curve() {
    // A typical mixing curve: a low cut, a broad tilt, one narrow cut. The net
    // level change should be small and the compensation should not overshoot.
    let bands = [
        (35.0, 0.0, 0.707, FilterType::LowCut),
        (200.0, 2.0, 1.0, FilterType::Bell),
        (3_000.0, -2.5, 2.0, FilterType::Bell),
        (9_000.0, 0.0, 0.707, FilterType::HighCut),
    ];
    let change = measured_level_change_db(&bands);
    let comp = auto_gain_compensation_db(&specs(&bands), bands.len());

    // A cut plus two 12 dB/oct skirts: the pink-noise model is an
    // approximation, so allow 1.5 dB — the point is that it does not overshoot.
    let residual = (change + comp).abs();
    assert!(
        residual < 1.5,
        "residual {residual:.2} dB (measured {change:+.2}, comp {comp:+.2})"
    );
}
