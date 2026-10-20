//! The real audio path, driven directly.
//!
//! Every audio test before this one mirrored the loop in `Filter::process`: the
//! detector over a 16-sample sub-block, a bake, a biquad. That made the thing
//! under test a *copy*. Two bugs in this project were exactly two
//! implementations of one idea drifting apart — a harness that reset filter
//! state on every bake and measured 7 dB wrong, and a spectrum band mapping
//! that disagreed with the axis by a factor of five.
//!
//! `Filter::process_buffer` was split out of `Plugin::process` so that the
//! production code itself is reachable from a test: `Buffer` is constructible
//! outside nih-plug (`set_slices`), while `ProcessContext` is not. What is here
//! therefore exercises parameter reads, coefficient baking, smoothing, the
//! cascade, the dynamics detector, the output stage and both analyzer taps —
//! the wiring, not a model of it.

use std::f32::consts::PI;
use std::sync::Arc;

use nih_plug::prelude::Buffer;

use crate::test_support::{test_params, test_params_with_dynamics};
use crate::{Filter, FilterParams};

const SR: f32 = 48_000.0;

/// A plugin instance with the given parameters, ready to process.
fn plugin(params: Arc<FilterParams>) -> Filter {
    let mut filter = Filter {
        params,
        ..Filter::default()
    };
    // The same setup `Plugin::initialize` performs. Without it the detectors
    // are still tuned for their default rate and dynamics measure wrong.
    filter.configure(SR);
    filter
}

fn tone(hz: f32, dbfs: f32, secs: f32) -> Vec<f32> {
    let amp = 10f32.powf(dbfs / 20.0);
    let n = (SR * secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * hz * i as f32 / SR).sin() * amp)
        .collect()
}

fn rms_db(samples: &[f32]) -> f32 {
    let tail = &samples[samples.len() / 2..];
    let power: f32 = tail.iter().map(|s| s * s).sum::<f32>() / tail.len().max(1) as f32;
    10.0 * power.max(1e-18).log10()
}

/// Run `input` through the production audio path and return the output.
fn run(filter: &mut Filter, input: &[f32]) -> Vec<f32> {
    let mut left = input.to_vec();
    let mut right = input.to_vec();
    let mut out_left = Vec::new();
    let mut out_right = Vec::new();

    // Host-sized blocks, because the loop's sub-block boundaries depend on it.
    for start in (0..input.len()).step_by(128) {
        let end = (start + 128).min(input.len());
        let mut buffer = Buffer::default();
        {
            let l = &mut left[start..end];
            let r = &mut right[start..end];
            let n = l.len();
            // Safety: nih-plug's own contract for `set_slices` — the slices
            // must be exactly `num_samples` long and must outlive the buffer.
            unsafe {
                buffer.set_slices(n, |slices| {
                    slices.push(l);
                    slices.push(r);
                });
            }
            filter.process_buffer(&mut buffer, false);
        }
        out_left.extend_from_slice(&left[start..end]);
        out_right.extend_from_slice(&right[start..end]);
    }

    // Both channels must receive the same treatment; a difference here is the
    // crosstalk bug that shipped once already.
    for (a, b) in out_left.iter().zip(&out_right) {
        assert!((a - b).abs() < 1e-6, "left and right diverged: {a} vs {b}");
    }
    out_left
}

#[test]
fn the_real_path_applies_the_curve() {
    // +6 dB bell at 1 kHz, measured through `process_buffer` with no harness in
    // between.
    let params = test_params(&[(1_000.0, 6.0, 1.0, crate::FilterType::Bell)]);
    let mut filter = plugin(params);

    let input = tone(1_000.0, -18.0, 0.5);
    let output = run(&mut filter, &input);

    let gain = rms_db(&output) - rms_db(&input);
    assert!(
        (gain - 6.0).abs() < 0.3,
        "a +6 dB bell should measure +6 dB through the real path, got {gain:+.2}"
    );
}

#[test]
fn the_real_path_leaves_other_frequencies_alone() {
    let params = test_params(&[(1_000.0, 12.0, 8.0, crate::FilterType::Bell)]);
    let mut filter = plugin(params);

    // Four octaves away from a Q 8 bell: essentially nothing.
    let input = tone(8_000.0, -18.0, 0.5);
    let output = run(&mut filter, &input);
    let gain = rms_db(&output) - rms_db(&input);
    assert!(
        gain.abs() < 0.5,
        "a narrow 1 kHz bell should not touch 8 kHz, got {gain:+.2} dB"
    );
}

#[test]
fn the_real_path_measures_zero_latency_at_1x() {
    let params = test_params(&[(1_000.0, 6.0, 1.0, crate::FilterType::Bell)]);
    let mut filter = plugin(params);
    let input = tone(1_000.0, -18.0, 0.05);
    let mut left = input.clone();
    let mut right = input.clone();
    let mut buffer = Buffer::default();
    let n = left.len();
    unsafe {
        buffer.set_slices(n, |slices| {
            slices.push(&mut left);
            slices.push(&mut right);
        });
    }
    let latency = filter.process_buffer(&mut buffer, false);
    assert_eq!(latency, 0, "no oversampling means no reported latency");
}

#[test]
fn silence_in_is_silence_out_through_the_real_path() {
    let params = test_params(&[(1_000.0, 24.0, 20.0, crate::FilterType::Bell)]);
    let mut filter = plugin(params);
    let input = vec![0.0f32; 9_600];
    let output = run(&mut filter, &input);
    let peak = output.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert_eq!(
        peak, 0.0,
        "a resonant boost must not ring up out of silence"
    );
}

#[test]
fn the_real_path_does_the_dynamics_itself() {
    // The gap this closes: `tests/dynamic_eq.rs` proves the *algorithm* with a
    // mirrored loop. This proves the plugin calls it, with the right threshold,
    // on the right samples, with the gain actually reaching the coefficients.
    let params = test_params_with_dynamics(&[(1_000.0, -12.0, 2.0, crate::FilterType::Bell)]);
    let mut filter = plugin(params);

    let loud = tone(1_000.0, -10.0, 0.6);
    let loud_out = run(&mut filter, &loud);
    let loud_gain = rms_db(&loud_out) - rms_db(&loud);

    let mut filter = plugin(test_params_with_dynamics(&[(
        1_000.0,
        -12.0,
        2.0,
        crate::FilterType::Bell,
    )]));
    let quiet = tone(1_000.0, -45.0, 0.6);
    let quiet_out = run(&mut filter, &quiet);
    let quiet_gain = rms_db(&quiet_out) - rms_db(&quiet);

    assert!(
        loud_gain < -6.0,
        "signal well above a -30 dB threshold should get most of the -12 dB band, got {loud_gain:+.2}"
    );
    assert!(
        quiet_gain.abs() < 1.0,
        "signal 15 dB below the threshold should be untouched, got {quiet_gain:+.2}"
    );
}

#[test]
fn a_disabled_band_does_nothing_through_the_real_path() {
    // `num_bands` gating happens in `process_buffer`, not in the helper.
    let params = test_params(&[]);
    let mut filter = plugin(params);
    let input = tone(1_000.0, -18.0, 0.3);
    let output = run(&mut filter, &input);
    for (a, b) in input.iter().zip(&output) {
        assert!((a - b).abs() < 1e-6, "no bands should pass audio untouched");
    }
}
