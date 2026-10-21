//! Linear-phase FIR design and partitioned convolution.
//!
//! Why this exists: every filter in the cascade so far is minimum phase, which
//! is the right default — no pre-ringing, low latency — but it means a steep
//! low cut smears transients, and there is no way to get the surgical,
//! symmetric-around-the-hit behaviour a mastering engineer reaches for.
//!
//! # How the response is designed
//!
//! The target magnitude is not re-derived from a different formula: it is
//! sampled from the *same* biquad cascades the minimum-phase path runs, at the
//! bin frequencies of an FFT of length `n`. So the curve drawn in the editor,
//! the sound of minimum-phase mode, and the magnitude of the linear-phase IR
//! are three views of one model rather than three models. `the_ir_matches_the_
//! cascade_magnitude` holds that line against `combined_response_db`.
//!
//! Phase is then *imposed*, not inherited:
//!
//! - **Linear**: `H[k] = |H[k]| * exp(-j*2*pi*k*D/n)` with `D = n/2`. That
//!   exponent is `(-1)^k`, so the spectrum stays real and the impulse response
//!   comes out exactly symmetric about `n/2`. Symmetry *is* linear phase — the
//!   test asserts `ir[D+i] == ir[D-i]` rather than trusting the algebra.
//! - **Minimum**: the homomorphic (cepstrum) method — fold the real cepstrum
//!   onto its causal half and exponentiate. Same magnitude, phase from the
//!   analog prototype, energy at the front. Not offered as a UI mode: the
//!   biquad cascade already *is* minimum phase, so shipping it would be a
//!   second implementation of the same thing, which is a failure mode this
//!   codebase has already paid for twice.
//!
//! # Why the delay is `n/2` and not less
//!
//! A symmetric filter of length `n` has group delay `(n-1)/2`; the design uses
//! `n/2` so the centre lands on an exact sample. Latency is therefore a
//! function of the tap count, which is why it is a reported constant and not a
//! measured guess — the host must compensate it, and a PDC error of a few
//! samples is audible as a phasey top end on parallel material.
//!
//! # Convolution
//!
//! Uniformly-partitioned overlap-save, `2B`-point FFT, hop `B`. Every
//! partition's spectrum is precomputed once per design, so the per-hop cost is
//! one forward FFT, `P` complex multiply-accumulates and one inverse FFT —
//! not `P` transforms. The convolver buffers to a hop boundary and emits its
//! result through a delay line, so its delay is a constant `B` samples for any
//! host block size: `spread_out_calls_have_the_same_response_as_one_whole_call`
//! is what makes that claim testable instead of hopeful.

use rustfft::{num_complex::Complex, FftPlanner};

use super::biquad::{BiquadCoeffs, MAX_SECTIONS};
use super::curve::BandSpec;
use crate::MAX_BANDS;

/// Taps in a linear-phase design.
///
/// 2048 is a latency/accuracy trade, not a round number: it is 1024 host
/// samples of delay (21 ms at 48 kHz) for a filter that holds its magnitude to
/// well under a decibel down to a 30 Hz low cut. Halving it halves the delay
/// and roughly doubles the transition width, which is visible on a 48 dB/oct
/// cut — the case the mode exists for.
pub const DESIGN_TAPS: usize = 2048;

/// Hop size for the convolution engine. The engine's own latency is exactly
/// this, independent of host block size.
pub const HOP: usize = 128;

/// Which phase response to impose on the designed magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirPhase {
    /// Zero phase, symmetric impulse response, delay `taps / 2`.
    Linear,
    /// Minimum phase via the cepstrum, no added delay.
    Minimum,
}

impl FirPhase {
    /// Group delay this design adds, in samples at the design rate.
    pub const fn latency(taps: usize) -> u32 {
        (taps / 2) as u32
    }
}

/// Per-band coefficients, baked once so the magnitude sweep does not recompute
/// them per bin. Bands above `num_active` and disabled bands are dropped here,
/// so the sweep loop has no branch.
fn baked_bands(
    bands: &[BandSpec; MAX_BANDS],
    num_active: usize,
    sample_rate: f32,
) -> Vec<([BiquadCoeffs; MAX_SECTIONS], usize)> {
    bands
        .iter()
        .take(num_active)
        .filter(|spec| spec.enabled)
        .map(|spec| {
            let mut sections = [BiquadCoeffs::default(); MAX_SECTIONS];
            let count = BiquadCoeffs::update_cascade(
                &mut sections,
                spec.freq,
                spec.gain_db,
                spec.q,
                spec.filter_type,
                spec.slope_db_per_oct,
                sample_rate,
            );
            (sections, count)
        })
        .collect()
}

/// Sample the combined cascade magnitude at the `n/2 + 1` non-negative bin
/// frequencies of an `n`-point FFT.
pub fn magnitude_spectrum(
    bands: &[BandSpec; MAX_BANDS],
    num_active: usize,
    sample_rate: f32,
    n: usize,
) -> Vec<f32> {
    let baked = baked_bands(bands, num_active, sample_rate);
    (0..=n / 2)
        .map(|k| {
            let freq = k as f32 * sample_rate / n as f32;
            let mut mag = 1.0f32;
            for (sections, count) in &baked {
                for section in sections.iter().take(*count) {
                    mag *= section.magnitude_at_freq(freq, sample_rate);
                }
            }
            mag
        })
        .collect()
}

/// Design an impulse response of `taps` samples (rounded up to a power of two)
/// for the given band set.
pub fn design(
    bands: &[BandSpec; MAX_BANDS],
    num_active: usize,
    sample_rate: f32,
    taps: usize,
    phase: FirPhase,
) -> Vec<f32> {
    let n = taps.max(8).next_power_of_two();
    let mags = magnitude_spectrum(bands, num_active, sample_rate, n);
    let mut planner = FftPlanner::new();
    let scale = 1.0 / n as f32;

    match phase {
        FirPhase::Linear => {
            // Real, symmetric spectrum with a delay of exactly n/2: the
            // exponent exp(-j*pi*k) is (-1)^k, so no complex phase is needed
            // and the result cannot drift off centre.
            let mut buf = vec![Complex::new(0.0f32, 0.0); n];
            for (k, &m) in mags.iter().enumerate() {
                let signed = if k % 2 == 0 { m } else { -m };
                buf[k] = Complex::new(signed, 0.0);
                if k != 0 && k != n / 2 {
                    buf[n - k] = buf[k];
                }
            }
            planner.plan_fft_inverse(n).process(&mut buf);
            buf.iter().map(|c| c.re * scale).collect()
        }
        FirPhase::Minimum => {
            // Real cepstrum of the log magnitude, folded onto the causal half,
            // then exponentiated back into a spectrum. Folding is what moves
            // the energy from a symmetric pair (one at +D, one at -D) to a
            // single causal one at the front.
            let mut log_mag = vec![Complex::new(0.0f32, 0.0); n];
            for (k, &m) in mags.iter().enumerate() {
                log_mag[k] = Complex::new(m.max(1e-9).ln(), 0.0);
                if k != 0 && k != n / 2 {
                    log_mag[n - k] = log_mag[k];
                }
            }
            planner.plan_fft_inverse(n).process(&mut log_mag);
            let cepstrum: Vec<f32> = log_mag.iter().map(|c| c.re * scale).collect();

            let mut folded = vec![Complex::new(0.0f32, 0.0); n];
            folded[0] = Complex::new(cepstrum[0], 0.0);
            for k in 1..n / 2 {
                folded[k] = Complex::new(2.0 * cepstrum[k], 0.0);
            }
            folded[n / 2] = Complex::new(cepstrum[n / 2], 0.0);

            planner.plan_fft_forward(n).process(&mut folded);
            for c in folded.iter_mut() {
                let gain = c.re.exp();
                *c = Complex::new(gain * c.im.cos(), gain * c.im.sin());
            }
            planner.plan_fft_inverse(n).process(&mut folded);
            folded.iter().map(|c| c.re * scale).collect()
        }
    }
}

/// Uniformly-partitioned overlap-save convolution with a constant `HOP`
/// samples of delay.
///
/// One instance per channel — the state is a filter history, and sharing it
/// between L and R is the stereo-crosstalk bug this project already fixed once.
pub struct Convolver {
    block: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    /// Spectrum of each IR partition, `2 * block` bins each.
    partitions: Vec<Vec<Complex<f32>>>,
    /// Ring of the last `partitions.len()` input spectra.
    history: Vec<Vec<Complex<f32>>>,
    history_pos: usize,
    /// Previous hop, held because overlap-save needs the block before this one.
    prev: Vec<f32>,
    /// Accumulator for the hop currently being filled.
    current: Vec<f32>,
    fill: usize,
    time: Vec<Complex<f32>>,
    acc: Vec<Complex<f32>>,
    /// Output delay line. Writes run up to two hops ahead of reads; the ring is
    /// twice that so a write can never land on an unread sample.
    ring: Vec<f32>,
    written: usize,
    emitted: usize,
}

impl Convolver {
    pub fn new(block: usize) -> Self {
        let block = block.max(16).next_power_of_two();
        let n = block * 2;
        let mut planner = FftPlanner::new();
        Self {
            block,
            fft: planner.plan_fft_forward(n),
            ifft: planner.plan_fft_inverse(n),
            partitions: Vec::new(),
            history: Vec::new(),
            history_pos: 0,
            prev: vec![0.0; block],
            current: vec![0.0; block],
            fill: 0,
            time: vec![Complex::new(0.0, 0.0); n],
            acc: vec![Complex::new(0.0, 0.0); n],
            ring: vec![0.0; block * 4],
            written: 0,
            emitted: 0,
        }
    }

    /// Delay introduced by the engine, in samples. Constant regardless of the
    /// block sizes the caller passes.
    pub fn latency(&self) -> usize {
        self.block
    }

    pub fn block(&self) -> usize {
        self.block
    }

    /// Install a (new) impulse response, discarding any tail in flight: the old
    /// tail describes a filter that no longer exists, and letting it ring out
    /// over the new one would splice two responses together.
    pub fn set_ir(&mut self, ir: &[f32]) {
        let n = self.block * 2;
        self.partitions.clear();
        for chunk in ir.chunks(self.block) {
            let mut buf = vec![Complex::new(0.0f32, 0.0); n];
            for (slot, &h) in buf.iter_mut().zip(chunk) {
                *slot = Complex::new(h, 0.0);
            }
            self.fft.process(&mut buf);
            self.partitions.push(buf);
        }
        // A convolver with no IR is a wire, not a filter.
        if self.partitions.is_empty() {
            let mut buf = vec![Complex::new(0.0f32, 0.0); n];
            buf[0] = Complex::new(1.0, 0.0);
            self.partitions.push(buf);
        }
        self.history = vec![vec![Complex::new(0.0, 0.0); n]; self.partitions.len()];
        self.reset();
    }

    /// Drop all filter memory and restart the output delay line.
    pub fn reset(&mut self) {
        for slot in &mut self.history {
            slot.fill(Complex::new(0.0, 0.0));
        }
        self.history_pos = 0;
        self.prev.fill(0.0);
        self.current.fill(0.0);
        self.fill = 0;
        self.ring.fill(0.0);
        self.written = 0;
        self.emitted = 0;
    }

    /// Process in place, one sample per sample. `input.len()` may be anything;
    /// the delay is still exactly [`Self::latency`].
    pub fn process(&mut self, input: &mut [f32]) {
        for sample in input.iter_mut() {
            self.current[self.fill] = *sample;
            self.fill += 1;
            if self.fill == self.block {
                self.compute_hop();
                self.fill = 0;
                self.prev.copy_from_slice(&self.current);
            }
            // Output index `emitted` carries convolution time
            // `emitted - latency`. The delay is what makes the hop buffering
            // invisible: without it the first hop's samples would be written
            // to ring slots the caller has already passed, and those slots
            // would read back as the silence that was emitted at the time.
            let due = self.emitted >= self.block;
            *sample = if due {
                let t = self.emitted - self.block;
                if t < self.written {
                    self.ring[t % self.ring.len()]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            self.emitted += 1;
        }
    }

    /// Convolve one completed hop and write its `block` outputs into the delay
    /// line at their absolute positions.
    fn compute_hop(&mut self) {
        let n = self.block * 2;
        for (i, &x) in self.prev.iter().chain(self.current.iter()).enumerate() {
            self.time[i] = Complex::new(x, 0.0);
        }
        self.fft.process(&mut self.time);

        self.history[self.history_pos].copy_from_slice(&self.time);
        let parts = self.partitions.len();
        let newest = self.history_pos;

        self.acc.fill(Complex::new(0.0, 0.0));
        for (p, partition) in self.partitions.iter().enumerate() {
            let idx = (newest + parts - p % parts) % parts;
            let spectrum = &self.history[idx];
            for (a, (x, h)) in self.acc.iter_mut().zip(spectrum.iter().zip(partition)) {
                *a += *x * *h;
            }
        }
        self.history_pos = (self.history_pos + 1) % parts;

        self.ifft.process(&mut self.acc);
        let scale = 1.0 / n as f32;
        // Overlap-save: the second half is the alias-free region, and with a
        // hop of `block` it is exactly this hop's output.
        for k in 0..self.block {
            let value = self.acc[self.block + k].re * scale;
            let slot = self.written % self.ring.len();
            self.ring[slot] = value;
            self.written += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilterType;

    fn spec(freq: f32, gain_db: f32, q: f32, filter_type: FilterType) -> BandSpec {
        BandSpec {
            freq,
            gain_db,
            q,
            filter_type,
            slope_db_per_oct: 24,
            enabled: true,
        }
    }

    fn bands(list: &[BandSpec]) -> [BandSpec; MAX_BANDS] {
        core::array::from_fn(|i| list.get(i).copied().unwrap_or_default())
    }

    /// |H| in dB of an impulse response at one frequency, by direct DTFT.
    fn ir_magnitude_db(ir: &[f32], freq: f32, sample_rate: f32) -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq as f64 / sample_rate as f64;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, &h) in ir.iter().enumerate() {
            re += h as f64 * (w * n as f64).cos();
            im -= h as f64 * (w * n as f64).sin();
        }
        20.0 * (re * re + im * im).sqrt().max(1e-12).log10()
    }

    /// Group delay at one frequency, by differentiating the phase. A linear
    /// phase filter must show the same value everywhere.
    fn group_delay(ir: &[f32], freq: f32, sample_rate: f32) -> f64 {
        let phase = |f: f64| {
            let w = 2.0 * std::f64::consts::PI * f / sample_rate as f64;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (n, &h) in ir.iter().enumerate() {
                re += h as f64 * (w * n as f64).cos();
                im -= h as f64 * (w * n as f64).sin();
            }
            im.atan2(re)
        };
        // One hertz is far too small a step to move the phase a full turn at
        // the delays this design has, so a single wrap into (-pi, pi] is exact
        // rather than approximate.
        let df = 1.0;
        let mut dphi = phase(freq as f64 + df) - phase(freq as f64);
        while dphi > std::f64::consts::PI {
            dphi -= 2.0 * std::f64::consts::PI;
        }
        while dphi <= -std::f64::consts::PI {
            dphi += 2.0 * std::f64::consts::PI;
        }
        -dphi / (2.0 * std::f64::consts::PI * df) * sample_rate as f64
    }

    #[test]
    fn the_ir_matches_the_cascade_magnitude() {
        let sr = 48_000.0;
        let set = bands(&[
            spec(1_000.0, 12.0, 1.0, FilterType::Bell),
            spec(120.0, 0.0, 0.707, FilterType::LowCut),
            spec(8_000.0, -6.0, 0.707, FilterType::HighShelf),
        ]);
        let ir = design(&set, 3, sr, DESIGN_TAPS, FirPhase::Linear);

        for freq in [
            30.0, 60.0, 120.0, 250.0, 1_000.0, 3_000.0, 8_000.0, 15_000.0,
        ] {
            let designed = ir_magnitude_db(&ir, freq, sr);
            let target = super::super::curve::combined_response_db(&set, 3, freq) as f64;
            assert!(
                (designed - target).abs() < 0.5,
                "{freq} Hz: IR is {designed:.2} dB, cascade says {target:.2} dB"
            );
        }
    }

    #[test]
    fn linear_phase_ir_is_symmetric_about_half_its_length() {
        let sr = 48_000.0;
        let set = bands(&[spec(200.0, 0.0, 0.707, FilterType::LowCut)]);
        let ir = design(&set, 1, sr, DESIGN_TAPS, FirPhase::Linear);
        let d = ir.len() / 2;
        for i in 1..d {
            assert!(
                (ir[d + i] - ir[d - i]).abs() < 1e-6,
                "asymmetry at offset {i}: {} vs {}",
                ir[d + i],
                ir[d - i]
            );
        }
    }

    #[test]
    fn linear_phase_group_delay_is_the_reported_latency() {
        let sr = 48_000.0;
        let set = bands(&[spec(4_000.0, 9.0, 1.4, FilterType::Bell)]);
        let ir = design(&set, 1, sr, DESIGN_TAPS, FirPhase::Linear);
        let reported = FirPhase::latency(DESIGN_TAPS) as f64;
        for freq in [200.0, 1_000.0, 5_000.0, 12_000.0] {
            let measured = group_delay(&ir, freq, sr);
            assert!(
                (measured - reported).abs() < 1.0,
                "{freq} Hz: group delay {measured:.2} vs reported {reported}"
            );
        }
    }

    #[test]
    fn minimum_phase_puts_its_energy_at_the_front() {
        let sr = 48_000.0;
        let set = bands(&[spec(300.0, 0.0, 0.707, FilterType::LowCut)]);
        let ir = design(&set, 1, sr, DESIGN_TAPS, FirPhase::Minimum);

        let total: f64 = ir.iter().map(|h| (*h as f64).powi(2)).sum();
        let first_quarter: f64 = ir
            .iter()
            .take(ir.len() / 4)
            .map(|h| (*h as f64).powi(2))
            .sum();
        assert!(
            first_quarter / total > 0.9,
            "minimum phase should be causal-front-loaded, got {:.3}",
            first_quarter / total
        );

        // And it is genuinely not the linear-phase answer.
        let linear = design(&set, 1, sr, DESIGN_TAPS, FirPhase::Linear);
        let peak_min = ir
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        let peak_lin = linear
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        assert!(peak_min < peak_lin / 4, "minimum phase must not be centred");
    }

    #[test]
    fn a_flat_design_is_a_delayed_impulse() {
        let set = [BandSpec::default(); MAX_BANDS];
        let ir = design(&set, 0, 48_000.0, 256, FirPhase::Linear);
        let d = ir.len() / 2;
        assert!((ir[d] - 1.0).abs() < 1e-3, "centre tap is {}", ir[d]);
        for (i, &h) in ir.iter().enumerate() {
            if i != d {
                assert!(h.abs() < 1e-3, "flat EQ must be transparent, tap {i} = {h}");
            }
        }
    }

    #[test]
    fn convolver_matches_direct_convolution() {
        // A short, arbitrary IR and a deterministic signal: the check is
        // sample-exact agreement, not a spectral summary that could hide a
        // misaligned partition.
        let ir: Vec<f32> = (0..300)
            .map(|i| (i as f32 * 0.37).sin() * (-i as f32 / 60.0).exp())
            .collect();
        let input: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.11).sin()).collect();

        let mut conv = Convolver::new(HOP);
        conv.set_ir(&ir);
        let mut wet = input.clone();
        conv.process(&mut wet);

        let latency = conv.latency();
        for n in 0..input.len() {
            let mut expected = 0.0f64;
            for (j, &h) in ir.iter().enumerate() {
                if n >= latency + j {
                    expected += h as f64 * input[n - latency - j] as f64;
                }
            }
            assert!(
                (wet[n] as f64 - expected).abs() < 2e-3,
                "sample {n}: {} vs {expected:.6}",
                wet[n]
            );
        }
    }

    #[test]
    fn spread_out_calls_have_the_same_response_as_one_whole_call() {
        let ir: Vec<f32> = (0..200).map(|i| (i as f32 * 0.21).cos() * 0.5).collect();
        let input: Vec<f32> = (0..777).map(|i| ((i as f32) * 0.05).sin()).collect();

        let mut whole = Convolver::new(HOP);
        whole.set_ir(&ir);
        let mut a = input.clone();
        whole.process(&mut a);

        let mut pieces = Convolver::new(HOP);
        pieces.set_ir(&ir);
        let mut b = input.clone();
        for chunk in b.chunks_mut(37) {
            pieces.process(chunk);
        }

        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "block-size dependence at {i}: {x} vs {y}"
            );
        }
    }

    #[test]
    fn an_identity_ir_is_pure_delay() {
        let mut conv = Convolver::new(HOP);
        conv.set_ir(&[1.0]);
        let mut buf: Vec<f32> = (0..500).map(|i| i as f32).collect();
        conv.process(&mut buf);
        let latency = conv.latency();
        for (n, &v) in buf.iter().enumerate() {
            let expected = if n >= latency {
                (n - latency) as f32
            } else {
                0.0
            };
            // 1e-4 rather than 1e-6: an f32 FFT round trip leaves roughly
            // -117 dB of noise on a silent sample, and this test is about
            // *alignment*, not about beating the arithmetic's own floor.
            assert!((v - expected).abs() < 1e-4, "sample {n}: {v} vs {expected}");
        }
    }

    #[test]
    fn changing_the_ir_clears_the_old_tail() {
        let mut conv = Convolver::new(HOP);
        conv.set_ir(&[0.5, 0.5]);
        let mut warm: Vec<f32> = vec![1.0; 400];
        conv.process(&mut warm);

        conv.set_ir(&[1.0]);
        let mut after: Vec<f32> = vec![0.0; 400];
        conv.process(&mut after);
        for (i, &v) in after.iter().enumerate().skip(conv.latency()) {
            assert!(v.abs() < 1e-6, "old tail survived at {i}: {v}");
        }
    }
}
