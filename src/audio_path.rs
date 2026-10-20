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

/// Largest sample-to-sample step in `samples`, and where it happened.
fn biggest_step(samples: &[f32]) -> (usize, f32) {
    samples
        .windows(2)
        .enumerate()
        .map(|(i, w)| (i, (w[1] - w[0]).abs()))
        .fold(
            (0, 0.0),
            |best, cur| if cur.1 > best.1 { cur } else { best },
        )
}

/// Soloing a band must not step the waveform.
///
/// Bypassing a filter by skipping it is a discontinuity: the cascade's output
/// jumps to the dry signal in one sample. A 1 kHz tone's own biggest step is
/// `A * 2π * f / SR` (~0.016 at -18 dBFS), so anything well above that is the
/// band being switched, not the signal.
///
/// Solo is the case that made this necessary: it changes which bands run, and a
/// solo button that pops on every press is one nobody uses. The same ramp covers
/// the enable toggle, which had the same step before this existed.
///
/// Driven through the solo atomic rather than a parameter because that is what
/// the GUI writes — `BoolParam` has no public setter, and a test that reached
/// for one would be testing a path the UI never takes.
#[test]
fn soloing_a_band_does_not_step_the_waveform() {
    const SOLOED: usize = 0;
    // Two boosts an octave and a half apart: soloing either one is a 12 dB
    // change at the tone's frequency, which is a step you can hear.
    let params = test_params(&[
        (1_000.0, 12.0, 4.0, crate::FilterType::Bell),
        (3_000.0, 12.0, 4.0, crate::FilterType::Bell),
    ]);
    let mut filter = plugin(params.clone());
    let input = tone(1_000.0, -18.0, 0.4);
    let mut left = input.clone();
    let mut right = input.clone();
    // Two traps here, both of which made an earlier version of this test pass
    // with the click still present:
    //
    // 1. The flip is applied at a block start, so it must be a multiple of the
    //    host block size — `SR * 0.2` is not, and the branch simply never fired.
    // 2. It must not land on a zero crossing. 0.2 s is exactly 200 cycles of
    //    1 kHz, where the tone is at zero: a 12 dB change there is a zero
    //    amplitude difference, so the measurement could not see it. 9728 is
    //    128 * 76, i.e. 0.2027 s — three quarters of a cycle off the crossing.
    let flip_block = 9_728usize;
    let flip_at = flip_block;

    for start in (0..input.len()).step_by(128) {
        let end = (start + 128).min(input.len());
        if start == flip_block {
            filter
                .solo
                .store(1 - SOLOED, std::sync::atomic::Ordering::Relaxed);
        }
        let mut buffer = Buffer::default();
        {
            let l = &mut left[start..end];
            let r = &mut right[start..end];
            let n = l.len();
            unsafe {
                buffer.set_slices(n, |slices| {
                    slices.push(l);
                    slices.push(r);
                });
            }
            filter.process_buffer(&mut buffer, false);
        }
    }

    // The tone's own worst step, measured on the first half where the band is
    // steady — the threshold comes from the signal, not from a magic number.
    let (_, steady) = biggest_step(&left[..flip_at]);
    let (at, step) = biggest_step(&left);
    assert!(
        step < steady * 2.0,
        "soloing stepped the waveform by {step:.4} at sample {at} \
         (a steady tone's biggest step is {steady:.4})"
    );
}

/// Solo must leave exactly one band running, and clearing it must restore the
/// rest — the acceptance criterion for solo, measured in audio rather than
/// asserted about a mask.
#[test]
fn solo_leaves_only_the_soloed_band_running() {
    use std::sync::atomic::Ordering;

    let params = test_params(&[
        (1_000.0, 12.0, 4.0, crate::FilterType::Bell),
        (8_000.0, 12.0, 4.0, crate::FilterType::Bell),
    ]);
    let mut filter = plugin(params);
    // Both bands are boosts, and the tone sits inside band 0 only, so the
    // three readings below are distinguishable at one frequency.
    let input = tone(1_000.0, -18.0, 0.4);
    let gain = |filter: &mut Filter| {
        let out = run(filter, &input);
        rms_db(&out) - rms_db(&input)
    };

    let both = gain(&mut filter);
    assert!(
        (both - 12.0).abs() < 0.5,
        "band 0 should boost 1 kHz by 12 dB, got {both:+.2}"
    );

    // Solo band 1, the 8 kHz one: the 1 kHz boost must disappear.
    filter.solo.store(1, Ordering::Relaxed);
    let soloed_other = gain(&mut filter);
    assert!(
        soloed_other.abs() < 0.5,
        "soloing the 8 kHz band should leave 1 kHz alone, got {soloed_other:+.2}"
    );

    // Solo band 0: back to the full boost, because it is the only band running.
    filter.solo.store(0, Ordering::Relaxed);
    let soloed_this = gain(&mut filter);
    assert!(
        (soloed_this - 12.0).abs() < 0.5,
        "soloing band 0 should restore its own boost, got {soloed_this:+.2}"
    );

    // Clearing solo restores every band.
    filter.solo.store(crate::NO_SOLO, Ordering::Relaxed);
    let cleared = gain(&mut filter);
    assert!(
        (cleared - 12.0).abs() < 0.5,
        "clearing solo should restore the full chain, got {cleared:+.2}"
    );
}
