//! Hot-path microbenchmarks for the plugin's process loop shape.
//!
//! Measures per-sample cost of:
//! (a) the pre-fix shape: per-sample smoother next() + update_coefficients + process,
//! (b) coefficients-only cost (isolate the trig work),
//! (c) per-host-block coefficient baking (128 samples — the previous shape),
//! (d) per-sub-block coefficient baking (16 samples — the current shape while
//!     a band is being automated), and
//! (e) the fully static case: no smoother work, no coefficient updates at all,
//!     which is what an untouched EQ costs.
//!
//! Run with: cargo bench --bench hotpath_bench

use std::hint::black_box;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

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
    println!(
        "    -> (a) -> (d): {:.1}x   (a) -> (e): {:.1}x",
        per_sample_prefix / per_sample_sub,
        per_sample_prefix / per_sample_static
    );
}
