//! Hot-path microbenchmarks for the plugin's process loop shape.
//!
//! Measures per-sample cost of:
//! (a) the pre-fix shape: per-sample smoother next() + update_coefficients + process,
//! (b) coefficients-only cost (isolate the trig work),
//! (c) per-host-block coefficient baking (128 samples — the previous shape),
//! (d) per-sub-block coefficient baking (16 samples — the current shape while
//!     a band is being automated), and
//! (e) the fully static case: no smoother work, no coefficient updates at all,
//!     which is what an untouched EQ costs, and
//! (f) the spectrum analyzer, which now runs twice (input and post-EQ taps)
//!     while the editor is open, and
//! (g) one dynamic band's level detector — the per-sample cost a band pays for
//!     following the signal instead of a knob.
//!
//! Run with: cargo bench --bench hotpath_bench

use std::hint::black_box;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use filter::dsp::dynamics::{Detector, DynLaw};
use filter::dsp::oversample::Oversampler;
use filter::dsp::spectrum::{SpectrumAnalyzer, SpectrumRing};
use filter::dsp::BiquadFilter;
use filter::FilterType;

const SR: f32 = 48_000.0;
const BLOCK: usize = 128;
const SUB: usize = 16;
const BANDS: usize = 8;

struct Params {
    freq: AtomicU32,
    gain: AtomicU32,
    q: AtomicU32,
}

impl Params {
    fn load(&self) -> (f32, f32, f32) {
        // black_box defeats LLVM hoisting of Relaxed loads, which would
        // otherwise let it precompute the trig once and fake the numbers.
        let f = f32::from_bits(self.freq.load(Ordering::Relaxed));
        let g = f32::from_bits(self.gain.load(Ordering::Relaxed));
        let q = f32::from_bits(self.q.load(Ordering::Relaxed));
        (black_box(f), black_box(g), black_box(q))
    }
}

fn cpu(s: f64) -> f64 {
    s * 48_000.0 * 2.0 * 100.0
}

fn main() {
    let input: Vec<f32> = (0..SR as usize * 2)
        .map(|n| {
            (2.0 * std::f32::consts::PI * 997.0 * n as f32 / SR).sin() * 0.5
                + (2.0 * std::f32::consts::PI * 3_101.0 * n as f32 / SR).sin() * 0.25
        })
        .collect();
    let params = Params {
        freq: AtomicU32::new(1000.0f32.to_bits()),
        gain: AtomicU32::new(3.0f32.to_bits()),
        q: AtomicU32::new(1.0f32.to_bits()),
    };

    // (a) pre-fix shape: per-sample smoother-advance + coefficient update + process.
    // Smoother semantics replicated: SmoothingStyle::next() is a multiply/add —
    // the significant cost being the coefficient update itself.
    let mut bands: Vec<BiquadFilter> = (0..BANDS).map(|_| BiquadFilter::new()).collect();
    let t0 = Instant::now();
    let mut n0 = 0usize;
    for chunk in input.chunks(BLOCK) {
        for _s in chunk {
            let (f, g, q) = params.load();
            let mut s = black_box(0.25f32);
            for b in bands.iter_mut().take(BANDS) {
                b.update_coefficients(f, g, q, FilterType::Bell, SR);
                s = b.process(s);
            }
            black_box(s);
            n0 += 1;
        }
    }
    let per_sample_prefix = t0.elapsed().as_secs_f64() / n0 as f64;

    // (b) coefficients only
    let mut b2 = BiquadFilter::new();
    let t1 = Instant::now();
    let mut n1 = 0usize;
    for chunk in input.chunks(BLOCK) {
        for _s in chunk {
            let (f, g, q) = params.load();
            b2.update_coefficients(f, g, q, FilterType::Bell, SR);
            n1 += 1;
        }
    }
    black_box(&b2);
    let per_update = t1.elapsed().as_secs_f64() / n1 as f64;

    // Shared body: bake coefficients every `every` samples, process every sample.
    let bake_and_process = |every: usize, update: bool| -> f64 {
        let mut chain: Vec<BiquadFilter> = (0..BANDS).map(|_| BiquadFilter::new()).collect();
        let t = Instant::now();
        let mut n = 0usize;
        for chunk in input.chunks(BLOCK) {
            for sub in chunk.chunks(every) {
                if update {
                    let (f, g, q) = params.load();
                    for b in chain.iter_mut().take(BANDS) {
                        b.update_coefficients(f, g, q, FilterType::Bell, SR);
                    }
                }
                for _s in sub {
                    let mut s = black_box(0.25f32);
                    for b in chain.iter_mut().take(BANDS) {
                        s = b.process(s);
                    }
                    black_box(s);
                    n += 1;
                }
            }
        }
        t.elapsed().as_secs_f64() / n as f64
    };

    // (c) per-host-block (128) coefficient baking — previous shape.
    let per_sample_block = bake_and_process(BLOCK, true);
    // (d) per-sub-block (16) coefficient baking — current shape while automating.
    let per_sample_sub = bake_and_process(SUB, true);
    // (e) static: no coefficient updates, filter alone.
    let per_sample_static = bake_and_process(SUB, false);

    println!(
        "== hot path, {} bands, {}-sample host blocks ==",
        BANDS, BLOCK
    );
    println!(
        "(a) per-sample coeffs (pre-fix shape):  {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_prefix * 1e9,
        cpu(per_sample_prefix)
    );
    println!(
        "(b) one update_coefficients alone:      {:>7.2} ns",
        per_update * 1e9
    );
    println!(
        "(c) per-block coeffs (128):             {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_block * 1e9,
        cpu(per_sample_block)
    );
    println!(
        "(d) per-sub-block coeffs (16, moving):  {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_sub * 1e9,
        cpu(per_sample_sub)
    );
    println!(
        "(e) static (nothing updating):          {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_static * 1e9,
        cpu(per_sample_static)
    );
    // (g) one dynamic band: the band-pass detector plus envelope, per sample.
    // A dynamic band also rebakes its coefficients every 16 samples even when
    // nothing is automated, which costs one `update_coefficients` (line (b)) per
    // 16 samples — added below rather than measured twice.
    let mut detector = Detector::new(SR);
    let t3 = Instant::now();
    let mut n3 = 0usize;
    for chunk in input.chunks(BLOCK) {
        for sub in chunk.chunks(SUB) {
            let level = detector.update(sub, 1000.0, 1.0, 10.0, 150.0);
            black_box(DynLaw::new(-30.0, -12.0, 3.0).applied_gain_db(-6.0, level));
            n3 += sub.len();
        }
    }
    let per_sample_detector = t3.elapsed().as_secs_f64() / n3 as f64;

    // (f) analyzer cost per sample. This is the price of the spectrum display —
    // paid twice now (input and output taps), and only while the editor is open.
    let ring = SpectrumRing::new();
    let mut analyzer = SpectrumAnalyzer::new(SR);
    let t2 = Instant::now();
    let mut n2 = 0usize;
    for chunk in input.chunks(BLOCK) {
        analyzer.push_block(&ring, chunk);
        n2 += chunk.len();
    }
    black_box(&analyzer);
    let per_sample_analyzer = t2.elapsed().as_secs_f64() / n2 as f64;

    // (f2) stereo analysis: two transforms and a power sum per frame, which is
    // what the taps now do. Paid only while the editor is open.
    let ring2 = SpectrumRing::new();
    let mut stereo_analyzer = SpectrumAnalyzer::new(SR);
    let t4 = Instant::now();
    let mut n4 = 0usize;
    for chunk in input.chunks(BLOCK) {
        stereo_analyzer.push_stereo_block(&ring2, chunk, chunk);
        n4 += chunk.len();
    }
    black_box(&stereo_analyzer);
    let per_sample_stereo = t4.elapsed().as_secs_f64() / n4 as f64;

    println!(
        "    -> (a) -> (d): {:.1}x   (a) -> (e): {:.1}x",
        per_sample_prefix / per_sample_sub,
        per_sample_prefix / per_sample_static
    );
    println!(
        "(f) one analyzer, per sample:           {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_analyzer * 1e9,
        cpu(per_sample_analyzer)
    );
    println!(
        "(f2) one analyzer, stereo per sample:   {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_stereo * 1e9,
        cpu(per_sample_stereo)
    );
    println!(
        "(g) one dynamic band's detector:        {:>7.2} ns/sample   {:>6.2}% CPU",
        per_sample_detector * 1e9,
        cpu(per_sample_detector)
    );
    println!(
        "    plus one rebake per {SUB} samples:      {:>7.2} ns/sample   -> {:>6.2} ns/sample total",
        per_update / SUB as f64 * 1e9,
        (per_sample_detector + per_update / SUB as f64) * 1e9
    );

    // (h) the resampler pair, per host sample, both channels. This is the whole
    // price of oversampling: the cascade inside it costs the same per
    // oversampled sample as it does at 1x, so the factor multiplies lines (c)-(e).
    for factor in [2usize, 4] {
        let mut os = Oversampler::new();
        os.set_factor(factor);
        let mut up_buf = vec![0.0f32; SUB * 4];
        let mut down_buf = vec![0.0f32; SUB];
        let t = Instant::now();
        let mut n = 0usize;
        for chunk in input.chunks(SUB) {
            let os_len = chunk.len() * factor;
            for channel in 0..2 {
                os.upsample(channel, chunk, &mut up_buf[..os_len]);
                os.downsample(channel, &up_buf[..os_len], &mut down_buf[..chunk.len()]);
            }
            black_box(&down_buf);
            n += chunk.len();
        }
        let per_sample = t.elapsed().as_secs_f64() / n as f64;
        println!(
            "(h) resampler, 2 channels at {factor}x:        {:>7.2} ns/sample   {:>6.2}% CPU",
            per_sample * 1e9,
            cpu(per_sample)
        );
    }
}
