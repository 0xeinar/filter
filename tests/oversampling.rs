//! Oversampling: does it earn its latency?
//!
//! The claim behind 2x/4x is narrow and checkable: a biquad is designed with the
//! bilinear transform, which warps the frequency axis so that a band near
//! Nyquist lands somewhere other than where the editor draws it. Running the
//! cascade faster shrinks that error. So the tests here measure three things:
//!
//! 1. The resampler itself is transparent — a flat EQ stays flat, and the
//!    delay is exactly the number the host is told.
//! 2. The warping error actually falls as the factor rises, against a
//!    reference taken at a rate high enough to be treated as truth.
//! 3. The dynamic detector keeps its ballistics, because its attack and release
//!    are specified in milliseconds and it now runs at the oversampled rate.
//!
//! The harness mirrors the production sub-block shape (16 host samples, bake,
//! filter, downsample) for the same reason the dynamic-EQ tests do: a
//! plugin-level `process()` call needs a `ProcessContext` that nih-plug does not
//! let a test construct. See issue #31.

use std::f32::consts::PI;

use filter::dsp::biquad::MAX_SECTIONS;
use filter::dsp::dynamics::Detector;
use filter::dsp::oversample::{latency_samples, Oversampler};
use filter::dsp::{BiquadCoeffs, BiquadState};
use filter::FilterType;

const SR: f32 = 44_100.0;
const COEFF_BLOCK: usize = 16;

/// One band, driven the way the audio loop drives it: coefficients and state
/// kept apart, because `update_cascade` on the combined type resets the state on
/// every call.
struct Chain {
    os: Oversampler,
    coeffs: [BiquadCoeffs; MAX_SECTIONS],
    state: [BiquadState; MAX_SECTIONS],
    count: usize,
    factor: usize,
    scratch: Vec<f32>,
}

impl Chain {
    fn new(factor: usize, freq: f32, gain: f32, q: f32, ftype: FilterType) -> Self {
        let mut chain = Self {
            os: Oversampler::new(),
            coeffs: [BiquadCoeffs::PASSTHROUGH; MAX_SECTIONS],
            state: [BiquadState::default(); MAX_SECTIONS],
            count: 1,
            factor,
            scratch: vec![0.0; COEFF_BLOCK * 4],
        };
        chain.os.set_factor(factor);
        chain.count = BiquadCoeffs::update_cascade(
            &mut chain.coeffs,
            freq,
            gain,
            q,
            ftype,
            12,
            SR * factor as f32,
        );
        chain
    }

    /// Filter a whole signal, sub-block by sub-block, exactly as `process()`
    /// does.
    fn run(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = input.to_vec();
        for chunk in out.chunks_mut(COEFF_BLOCK) {
            let len = chunk.len();
            if self.factor == 1 {
                for sample in chunk.iter_mut() {
                    let mut v = *sample;
                    for i in 0..self.count {
                        v = self.state[i].process(&self.coeffs[i], v);
                    }
                    *sample = v;
                }
            } else {
                let os_len = len * self.factor;
                self.os.upsample(0, chunk, &mut self.scratch[..os_len]);
                for sample in self.scratch[..os_len].iter_mut() {
                    let mut v = *sample;
                    for i in 0..self.count {
                        v = self.state[i].process(&self.coeffs[i], v);
                    }
                    *sample = v;
                }
                let scratch = &self.scratch[..os_len];
                self.os.downsample(0, scratch, chunk);
            }
        }
        out
    }
}

fn tone(hz: f32, dbfs: f32, secs: f32) -> Vec<f32> {
    let amp = 10f32.powf(dbfs / 20.0);
    let n = (SR * secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * hz * i as f32 / SR).sin() * amp)
        .collect()
}

/// RMS level of the steady tail, in dB.
///
/// RMS rather than peak: above ~15 kHz a cycle at 44.1 kHz is under three
/// samples long, so a peak measurement lands on whatever phase the window
/// happened to start at. For a gain measurement (a difference of two levels on
/// the same signal) RMS is exact and phase-blind.
fn steady_db(samples: &[f32], _hz: f32) -> f32 {
    let tail = &samples[samples.len() / 4..];
    let power: f32 = tail.iter().map(|s| s * s).sum::<f32>() / tail.len().max(1) as f32;
    10.0 * power.max(1e-18).log10()
}

/// Measured gain of a band through the full chain at one frequency.
fn measured_gain_db(factor: usize, freq: f32, measure_hz: f32, gain: f32) -> f32 {
    let mut chain = Chain::new(factor, freq, gain, 0.7, FilterType::HighShelf);
    let input = tone(measure_hz, -12.0, 0.6);
    let output = chain.run(&input);
    steady_db(&output, measure_hz) - steady_db(&input, measure_hz)
}

#[test]
fn a_flat_chain_is_transparent_at_every_factor() {
    // 0 dB bell: the cascade is an identity, so anything measured here is the
    // resampler's own colouration.
    for factor in [1usize, 2, 4] {
        for hz in [
            100.0, 1_000.0, 5_000.0, 10_000.0, 16_000.0, 18_000.0, 20_000.0,
        ] {
            let gain = measured_gain_db(factor, 1_000.0, hz, 0.0);
            assert!(
                gain.abs() < 0.15,
                "{factor}x colours a flat path at {hz} Hz by {gain:+.3} dB"
            );
        }
    }
}

#[test]
fn the_delay_is_the_number_the_host_is_told() {
    for factor in [2usize, 4] {
        let mut chain = Chain::new(factor, 1_000.0, 0.0, 0.7, FilterType::Bell);
        let mut input = vec![0.0f32; 512];
        input[0] = 1.0;
        let output = chain.run(&input);

        let peak = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let reported = latency_samples(factor) as usize;
        assert!(
            peak.abs_diff(reported) <= 1,
            "{factor}x impulse peak at {peak}, host told {reported}"
        );
        assert_eq!(latency_samples(1), 0, "1x must report zero latency");
    }
}

#[test]
fn oversampling_reduces_the_bilinear_warping_error() {
    // The property the feature exists for. A high shelf is prewarped so its
    // *corner* lands exactly where it is asked to — that is why measuring at
    // the corner shows almost no error and proves nothing. The warping shows up
    // everywhere else, and it grows towards Nyquist: at 20 kHz the analog
    // frequency is tan(pi*f/fs), several times the linear value, so the shape
    // of the transition is stretched.
    //
    // The reference is the same band designed at 8x, where the warping is
    // negligible: the response the band converges to.
    let corner = 12_000.0;
    let gain = 9.0;

    let mut reference = [BiquadCoeffs::PASSTHROUGH; MAX_SECTIONS];
    let count = BiquadCoeffs::update_cascade(
        &mut reference,
        corner,
        gain,
        0.7,
        FilterType::HighShelf,
        12,
        SR * 8.0,
    );
    let ideal_at = |hz: f32| -> f32 {
        (0..count)
            .map(|i| reference[i].magnitude_at_freq(hz, SR * 8.0))
            .product::<f32>()
            .max(1e-12)
            .log10()
            * 20.0
    };

    // Above the corner, where the shelf is most of the way to its plateau and
    // the warp has the most to say.
    let probes = [16_000.0, 18_000.0, 20_000.0];

    let max_error = |factor: usize| -> f32 {
        probes
            .iter()
            .map(|&hz| (measured_gain_db(factor, corner, hz, gain) - ideal_at(hz)).abs())
            .fold(0.0f32, f32::max)
    };

    let err_1x = max_error(1);
    let err_2x = max_error(2);
    let err_4x = max_error(4);

    assert!(
        err_4x < err_2x && err_2x < err_1x,
        "error should shrink as the factor rises: 1x {err_1x:.2} dB, \
         2x {err_2x:.2} dB, 4x {err_4x:.2} dB"
    );
    assert!(
        err_4x < 1.0,
        "4x should land within 1 dB of the converged response, got {err_4x:.2} dB"
    );
    assert!(
        err_1x > 1.0,
        "if 1x were already within 1 dB there would be nothing to fix; got {err_1x:.2} dB"
    );
}

#[test]
fn the_detector_keeps_its_ballistics_at_the_oversampled_rate() {
    // Attack is specified in milliseconds. Running the detector at 4x while
    // leaving its timing computed for 1x would make it four times faster — the
    // classic bug in this kind of change, and one that a steady-state
    // measurement cannot see at all.
    //
    // The comparison is the envelope at fixed *wall-clock* instants rather than
    // a threshold crossing: the rectified output of a band-passed tone ripples,
    // so a single crossing index is noisy. Averaging over a short window around
    // each instant removes the ripple without hiding a rate error.
    fn trajectory(factor: usize) -> Vec<(f32, f32)> {
        let mut det = Detector::default();
        let rate = SR * factor as f32;
        det.set_sample_rate(rate);
        let block_len = COEFF_BLOCK * 4;
        let mut block = vec![0.0f32; block_len];
        let attack = 50.0;
        let release = 200.0;

        let blocks = (0.30 * rate / block_len as f32) as usize;
        let mut out = Vec::with_capacity(blocks);
        for block_idx in 0..blocks {
            for (i, slot) in block.iter_mut().enumerate() {
                let t = (block_idx * block_len + i) as f32 / rate;
                *slot = (2.0 * PI * 1000.0 * t).sin();
            }
            let level = det.update(&block, 1000.0, 1.0, attack, release);
            let t_ms = (block_idx * block_len) as f32 / rate * 1000.0;
            out.push((t_ms, level));
        }
        out
    }

    fn level_at(trajectory: &[(f32, f32)], at_ms: f32) -> f32 {
        let window = 2.0;
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for &(t, level) in trajectory {
            if (t - at_ms).abs() <= window {
                sum += level;
                count += 1.0;
            }
        }
        sum / count.max(1.0)
    }

    let one = trajectory(1);
    let four = trajectory(4);
    for at_ms in [10.0f32, 25.0, 50.0, 100.0, 200.0] {
        let a = level_at(&one, at_ms);
        let b = level_at(&four, at_ms);
        assert!(
            (a - b).abs() < 2.0,
            "at {at_ms} ms the envelope is {a:.2} dB at 1x and {b:.2} dB at 4x — \
             the detector rate is not following the filter rate"
        );
    }
}

#[test]
fn the_bypass_crossfade_stays_aligned_with_oversampling() {
    // Bypass mixes dry against wet. With oversampling the wet path is late by
    // the reported latency, so the dry signal has to be held back the same
    // amount; if it is not, the transition mixes two different instants and
    // produces a step.
    let factor = 4usize;
    let latency = latency_samples(factor) as usize;
    let mut chain = Chain::new(factor, 2_000.0, 12.0, 1.0, FilterType::Bell);

    let input = tone(2_000.0, -6.0, 0.25);
    let wet = chain.run(&input);

    // Dry, delayed by the reported latency — what the production loop feeds the
    // crossfade.
    let mut dry_delayed = vec![0.0f32; input.len()];
    dry_delayed[latency..].copy_from_slice(&input[..input.len() - latency]);

    // Crossfade over 20 ms starting a third of the way in.
    let fade = (SR * 0.02) as usize;
    let start = input.len() / 3;
    let mut mixed = vec![0.0f32; input.len()];
    for i in 0..input.len() {
        let mix = if i < start {
            0.0
        } else if i < start + fade {
            (i - start) as f32 / fade as f32
        } else {
            1.0
        };
        mixed[i] = dry_delayed[i] + (wet[i] - dry_delayed[i]) * mix;
    }

    // The signal's own largest step, for scale: a 2 kHz tone at -6 dBFS moves
    // about 0.18 per sample at 44.1 kHz.
    let natural_step = input
        .windows(2)
        .fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()));
    let mixed_step = mixed
        .windows(2)
        .skip(input.len() / 4)
        .fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()));

    assert!(
        mixed_step < natural_step * 4.0,
        "crossfade introduced a step of {mixed_step:.3} against a natural \
         {natural_step:.3} — dry and wet are not time-aligned"
    );
}

#[test]
fn the_factor_can_change_mid_stream_without_a_discontinuity() {
    // Switching factors flushes the FIR history and rebakes every band. A
    // residual click would be a design failure, so it is measured: the output
    // must stay bounded and free of a spike at the switch.
    let mut chain = Chain::new(1, 1_000.0, 9.0, 1.0, FilterType::Bell);
    let first = tone(1_000.0, -9.0, 0.2);
    let mut out = chain.run(&first);

    chain.os.set_factor(4);
    chain.count = BiquadCoeffs::update_cascade(
        &mut chain.coeffs,
        1_000.0,
        9.0,
        1.0,
        FilterType::Bell,
        12,
        SR * 4.0,
    );
    chain.factor = 4;
    chain.state = [BiquadState::default(); MAX_SECTIONS];

    let second = tone(1_000.0, -9.0, 0.2);
    out.extend(chain.run(&second));

    let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    // A 9 dB boost on a -9 dBFS tone peaks near 1.0; anything past 1.5 is a
    // transient, not a filter.
    assert!(peak < 1.5, "factor switch produced a peak of {peak:.2}");
}
