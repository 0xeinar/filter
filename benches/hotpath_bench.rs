//! Hot-path microbenchmarks for the plugin's process loop shape.
//!
//! Measures per-sample cost of:
//! (a) the pre-fix shape: per-sample smoother next() + update_coefficients + process,
//!     as the process() loop was structured (per-sample param reads)
//! (b) coefficients-only cost (isolate the trig work)
//! (c) the restructured shape: block-smoothed params, per-block coefficient
//!     updates, per-sample process only
//!
//! Run with: cargo bench --bench hotpath_bench

use std::hint::black_box;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use filter::dsp::BiquadFilter;
use filter::FilterType;

const SR: f32 = 48_000.0;
const BLOCK: usize = 128;
const CHUNK: usize = 64;
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

    // (c) restructured: per-block coefficient updates + per-sample process,
    // params block-smoothed (smoothing math folded into the block fill —
    // approximated here by reading the params once per chunk).
    let mut bands2: Vec<BiquadFilter> = (0..BANDS).map(|_| BiquadFilter::new()).collect();
    let t2 = Instant::now();
    let mut n2 = 0usize;
    for chunk in input.chunks(BLOCK) {
        for sub in chunk.chunks(CHUNK) {
            let (f, g, q) = params.load();
            for b in bands2.iter_mut().take(BANDS) {
                b.update_coefficients(f, g, q, FilterType::Bell, SR);
            }
            for _s in sub {
                let mut s = black_box(0.25f32);
                for b in bands2.iter_mut().take(BANDS) {
                    s = b.process(s);
                }
                black_box(s);
                n2 += 1;
            }
        }
    }
    let per_sample_fixed = t2.elapsed().as_secs_f64() / n2 as f64;

    println!(
        "== hot path, {} bands, {}-sample host blocks ==",
        BANDS, BLOCK
    );
    println!(
        "(a) per-sample coeffs (pre-fix shape):  {:>7.2} ns/sample",
        per_sample_prefix * 1e9
    );
    println!(
        "(b) one update_coefficients alone:      {:>7.2} ns",
        per_update * 1e9
    );
    println!(
        "(c) per-block coeffs (restructured):    {:>7.2} ns/sample",
        per_sample_fixed * 1e9
    );
    println!(
        "    -> reduction: {:.1}x",
        per_sample_prefix / per_sample_fixed
    );
    println!(
        "    CPU load (a) @48kHz stereo: {:.2}%",
        per_sample_prefix * 48_000.0 * 2.0 * 100.0
    );
    println!(
        "    CPU load (c) @48kHz stereo: {:.2}%",
        per_sample_fixed * 48_000.0 * 2.0 * 100.0
    );
}
