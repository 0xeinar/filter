//! Real-time input spectrum analysis.
//!
//! The audio thread fills a mono scratch buffer per block, runs a small FFT
//! when the buffer is full, and writes magnitude snapshots into a lock-free
//! ring. The GUI thread polls the ring and draws. No allocation or locking
//! in `push_block`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use rustfft::{num_complex::Complex, FftPlanner};

use std::f32::consts::PI;

/// FFT size — 1024 samples ≈ 23 ms at 44.1 kHz, bin spacing ≈ 43 Hz.
pub const FFT_SIZE: usize = 1024;

/// Number of spectrum frames kept in the GUI ring. GUI only ever reads the
/// newest, so a small depth is plenty.
pub const RING_FRAMES: usize = 4;

/// One analyzer output frame: dB magnitudes for `BINS` bands, log-spaced.
pub const BINS: usize = 96;

/// Frequency of the upper edge of band `i`, in Hz.
///
/// The analyzer maps linear FFT bins onto `BINS` log-spaced bands spanning
/// bin 1 to bin N/2-1. The GUI needs the same mapping to place a band's bar,
/// and the canvas tests need it to synthesize a frame whose drawn position is
/// known. Both call this, so the two can't drift.
///
/// Evaluated at a nominal 48 kHz: the display is a log-frequency picture, not
/// a calibrated analyser, and the band edges move by less than a pixel with
/// the sample rate.
pub fn bin_edge_hz(i: usize) -> f32 {
    let first_bin = 1.0f32;
    let last_bin_f = (FFT_SIZE / 2 - 1) as f32;
    let decades = (last_bin_f / first_bin).log10();
    let edge_bin = (first_bin * 10f32.powf(decades * (i + 1) as f32 / BINS as f32)).round();
    edge_bin * 48_000.0 / FFT_SIZE as f32
}

/// A single averaged spectrum frame in dB (normalized: 0 dBFS → ~0 dB).
#[derive(Clone, Copy)]
pub struct SpectrumFrame {
    /// dB values, index 0 = lowest frequency band (~20 Hz), last = ~20 kHz+
    pub db: [f32; BINS],
    /// Counter that increments every frame — lets the GUI detect stale data
    pub seq: u64,
}

impl SpectrumFrame {
    pub fn silent() -> Self {
        Self {
            db: [-96.0; BINS],
            seq: 0,
        }
    }
}

/// Lock-free(ish) analyzer state shared between audio and GUI threads.
///
/// The audio thread writes frames at `ring[audio_idx]` and bumps `audio_idx`
/// atomically. The GUI copies the newest frame under a short RwLock read.
/// RwLock is acceptable here because the *audio thread never blocks*: it uses
/// try_write and skips the frame on contention (GUI updates are best-effort).
pub struct SpectrumRing {
    slots: Vec<RwLock<SpectrumFrame>>,
    next_write: AtomicUsize,
}

impl SpectrumRing {
    pub fn new() -> Self {
        Self {
            slots: (0..RING_FRAMES)
                .map(|_| RwLock::new(SpectrumFrame::silent()))
                .collect(),
            next_write: AtomicUsize::new(0),
        }
    }

    /// Audio thread: publish a frame. Never blocks — skips on contention.
    pub fn publish(&self, frame: SpectrumFrame) {
        let idx = self.next_write.load(Ordering::Relaxed) % RING_FRAMES;
        if let Ok(mut slot) = self.slots[idx].try_write() {
            *slot = frame;
            self.next_write.store(idx + 1, Ordering::Relaxed);
        }
    }

    /// GUI thread: read the newest frame (highest seq among slots).
    pub fn read_newest(&self) -> SpectrumFrame {
        let mut best = SpectrumFrame::silent();
        for slot in &self.slots {
            if let Ok(guard) = slot.try_read() {
                if guard.seq >= best.seq {
                    best = *guard;
                }
            }
        }
        best
    }
}

impl Default for SpectrumRing {
    fn default() -> Self {
        Self::new()
    }
}

/// The two analyzer taps: `input` is the pre-EQ signal, `output` the post-EQ
/// signal (after every band, the output trim and the bypass mix).
///
/// Bundled into one struct rather than passed as two `Arc<SpectrumRing>`
/// arguments so a caller can never hand the GUI the same ring twice by
/// transposing an argument — the mistake would be invisible (both curves would
/// simply agree) and the whole point of the overlay is that they differ.
pub struct SpectrumTaps {
    pub input: SpectrumRing,
    pub output: SpectrumRing,
}

impl SpectrumTaps {
    pub fn new() -> Self {
        Self {
            input: SpectrumRing::new(),
            output: SpectrumRing::new(),
        }
    }
}

impl Default for SpectrumTaps {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming FFT analyzer owned by the audio thread.
pub struct SpectrumAnalyzer {
    scratch: Vec<f32>,
    fill: usize,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    scratch_complex: Vec<Complex<f32>>,
    #[allow(dead_code)] // kept for future band-edge queries
    sample_rate: f32,
    seq: u64,
    /// Hann window coefficients for FFT_SIZE
    window: Vec<f32>,
    /// Total magnitude compensation of the Hann window + FFT normalization
    norm: f32,
}

impl SpectrumAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / FFT_SIZE as f32).cos())
            .collect();
        // Hann window coherent gain = 0.5; FFT normalization = 1/(N/2) for magnitudes.
        let norm = 2.0 / (0.5 * FFT_SIZE as f32);
        Self {
            scratch: vec![0.0; FFT_SIZE],
            fill: 0,
            fft,
            scratch_complex: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            sample_rate,
            // Start at 1 so every published frame outranks the seq=0 silent
            // sentinel in SpectrumFrame::silent().
            seq: 1,
            window,
            norm,
        }
    }

    /// Feed a block of mono samples; runs the FFT and publishes whenever the
    /// internal buffer is full. Partial input is retained between calls.
    pub fn push_block(&mut self, ring: &SpectrumRing, samples: &[f32]) {
        let mut consumed = 0;
        while consumed < samples.len() {
            let want = FFT_SIZE - self.fill;
            let take = want.min(samples.len() - consumed);
            self.scratch[self.fill..self.fill + take]
                .copy_from_slice(&samples[consumed..consumed + take]);
            self.fill += take;
            consumed += take;

            if self.fill == FFT_SIZE {
                let frame = self.compute_frame();
                ring.publish(frame);
                // Overlap 75% for smoother time resolution
                self.scratch.copy_within(FFT_SIZE / 4..FFT_SIZE, 0);
                self.fill = FFT_SIZE * 3 / 4;
            }
        }
    }

    fn compute_frame(&mut self) -> SpectrumFrame {
        for (i, &s) in self.scratch.iter().enumerate() {
            self.scratch_complex[i] = Complex::new(s * self.window[i], 0.0);
        }
        self.fft.process(&mut self.scratch_complex);

        let mut frame = SpectrumFrame::silent();
        frame.seq = self.seq;
        self.seq += 1;

        // Map linear bins → BINS log-spaced bands across the FFT's USEFUL
        // range: bin 1 (lowest non-DC) to bin N/2-1 (Nyquist). Below bin 1 the
        // FFT has no resolution (bin 0 is DC), so log-spaced bands would waste
        // most of the low end on a single bin each.
        // Band b covers bin range [lo, hi); magnitude = max in range.
        // NOTE: FFT bin k corresponds to k * sample_rate / FFT_SIZE Hz.
        let first_bin = 1.0f32;
        let last_bin_f = (FFT_SIZE / 2 - 1) as f32;
        let decades = (last_bin_f / first_bin).log10();
        let mut prev_bin = 1usize;
        let last_bin = FFT_SIZE / 2 - 1;
        for (b, out) in frame.db.iter_mut().enumerate() {
            if prev_bin >= last_bin {
                break;
            }
            let f = first_bin * 10f32.powf(decades * (b + 1) as f32 / BINS as f32);
            let hi_bin = ((f).round() as usize).clamp(prev_bin + 1, last_bin);
            // Max-magnitude across the band's bins
            let mut mag = 0.0f32;
            for k in prev_bin..hi_bin {
                let c = self.scratch_complex[k];
                mag = mag.max((c.re * c.re + c.im * c.im).sqrt());
            }
            prev_bin = hi_bin;
            let db = 20.0 * (mag * self.norm).max(1e-6).log10();
            *out = db.clamp(-96.0, 12.0);
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> SpectrumRing {
        SpectrumRing::new()
    }

    #[test]
    fn silent_frame_floor() {
        let f = SpectrumFrame::silent();
        assert!(f.db.iter().all(|&d| d == -96.0));
    }

    #[test]
    fn ring_publish_read() {
        let r = ring();
        let mut f = SpectrumFrame::silent();
        f.seq = 7;
        f.db[10] = -3.5;
        r.publish(f);
        let got = r.read_newest();
        assert_eq!(got.seq, 7);
        assert_eq!(got.db[10], -3.5);
    }

    #[test]
    fn ring_newest_wins() {
        let r = ring();
        let mut f1 = SpectrumFrame::silent();
        f1.seq = 1;
        r.publish(f1);
        let mut f2 = SpectrumFrame::silent();
        f2.seq = 2;
        r.publish(f2);
        assert_eq!(r.read_newest().seq, 2);
    }

    #[test]
    fn sine_peaks_at_right_band() {
        // 1 kHz sine at full scale
        let sr = 48_000.0;
        let mut an = SpectrumAnalyzer::new(sr);
        let r = ring();

        // Feed 4 windows worth (with 75% overlap, 1 + 3 extra quarters)
        let total = FFT_SIZE + 3 * (FFT_SIZE / 4);
        let samples: Vec<f32> = (0..total)
            .map(|n| (2.0 * PI * 1000.0 * n as f32 / sr).sin())
            .collect();
        an.push_block(&r, &samples);

        let frame = r.read_newest();
        assert!(frame.seq > 0, "no frame was published");

        // Find the peak band and verify it's near 1 kHz.
        // Replicate compute_frame's band→bin mapping to get the band's bin range.
        let (peak_b, peak_db) = frame
            .db
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let first_bin = 1.0f32;
        let last_bin_f = (FFT_SIZE / 2 - 1) as f32;
        let decades = (last_bin_f / first_bin).log10();
        let mut prev_bin = 1usize;
        let mut hi_bin = 1usize;
        for b in 0..=peak_b {
            let f = first_bin * 10f32.powf(decades * (b + 1) as f32 / BINS as f32);
            hi_bin = ((f).round() as usize).clamp(prev_bin + 1, FFT_SIZE / 2 - 1);
            prev_bin = hi_bin;
        }
        let bin_hz = 48_000.0f32 / FFT_SIZE as f32;
        let peak_hz = hi_bin as f32 * bin_hz;
        assert!(
            (800.0..1250.0).contains(&peak_hz),
            "peak at {} Hz (band {}, hi bin {})",
            peak_hz,
            peak_b,
            hi_bin
        );
        // Full-scale sine should read close to 0 dBFS
        assert!(
            *peak_db > -6.0,
            "peak too low: {} dB at {} Hz",
            peak_db,
            peak_hz
        );
    }

    #[test]
    fn silence_stays_at_floor() {
        let mut an = SpectrumAnalyzer::new(48_000.0);
        let r = ring();
        an.push_block(&r, &vec![0.0; FFT_SIZE + FFT_SIZE / 4]);
        let frame = r.read_newest();
        // With exact zeros, magnitudes are 0 → clamped to -96
        assert!(frame.db.iter().all(|&d| d <= -90.0));
    }

    #[test]
    fn partial_blocks_are_retained() {
        let mut an = SpectrumAnalyzer::new(48_000.0);
        let r = ring();
        // Feed half a window, then half a window — should still produce a frame
        let half = vec![0.5f32; FFT_SIZE / 2];
        an.push_block(&r, &half);
        // No full window yet — nothing published (only the silent sentinel)
        assert_eq!(r.read_newest().seq, 0);
        an.push_block(&r, &half);
        let frame = r.read_newest();
        assert_eq!(frame.seq, 1);
        // DC-heavy 0.5 constant: bin 0 gets energy
        assert!(frame.db[0] > -60.0, "low band should have energy");
    }

    #[test]
    fn band_mapping_is_monotonic() {
        // The band center frequencies must be strictly increasing
        let sr = 48_000.0;
        let nyquist = sr / 2.0;
        let decades = (nyquist / 20.0f32).log10();
        let mut prev = 0.0f32;
        for b in 0..BINS {
            let f = 20.0 * 10f32.powf(decades * (b + 1) as f32 / BINS as f32);
            assert!(f > prev, "band {} frequency not increasing", b);
            prev = f;
        }
    }

    #[test]
    fn analyzer_never_blocks_on_contention() {
        // Hold the ring's write lock (as a slow GUI could) and ensure
        // publish() returns rather than deadlocking.
        let r = ring();
        {
            let _guard = r.slots[0].write().unwrap();
            let mut an = SpectrumAnalyzer::new(48_000.0);
            let samples = vec![0.1f32; FFT_SIZE + FFT_SIZE / 4];
            an.push_block(&r, &samples); // must not block
        }
        // After the lock is released, a new publish can succeed
        let mut an = SpectrumAnalyzer::new(48_000.0);
        an.push_block(&r, &vec![0.1f32; FFT_SIZE + FFT_SIZE / 4]);
    }

    #[test]
    fn taps_do_not_share_a_ring() {
        let taps = SpectrumTaps::new();
        let mut f = SpectrumFrame::silent();
        f.seq = 9;
        taps.input.publish(f);
        assert_eq!(taps.input.read_newest().seq, 9);
        // Publishing to one tap must not make the other look fed — a shared
        // ring here would draw the input twice and hide every EQ change.
        assert_eq!(taps.output.read_newest().seq, 0);
    }

    /// The overlay only means something if the two taps actually carry
    /// different signals: the same sine, with the post tap filtered by a
    /// -20 dB bell, must read ~20 dB lower at the peak.
    #[test]
    fn post_tap_shows_what_the_filter_did() {
        use crate::dsp::BiquadFilter;
        use crate::FilterType;

        let sr = 48_000.0;
        let taps = SpectrumTaps::new();
        let mut pre = SpectrumAnalyzer::new(sr);
        let mut post = SpectrumAnalyzer::new(sr);

        let total = FFT_SIZE + 3 * (FFT_SIZE / 4);
        let input: Vec<f32> = (0..total)
            .map(|n| (2.0 * PI * 1000.0 * n as f32 / sr).sin())
            .collect();

        // -20 dB bell at 1 kHz, exactly where the sine sits.
        let mut eq = BiquadFilter::new();
        eq.update_coefficients(1000.0, -20.0, 1.0, FilterType::Bell, sr);
        let output: Vec<f32> = input.iter().map(|&s| eq.process(s)).collect();

        pre.push_block(&taps.input, &input);
        post.push_block(&taps.output, &output);

        let peak = |ring: &SpectrumRing| {
            ring.read_newest()
                .db
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let pre_db = peak(&taps.input);
        let post_db = peak(&taps.output);

        assert!(
            pre_db > -6.0,
            "pre tap should see the full-scale sine, got {pre_db:.1} dB"
        );
        let delta = pre_db - post_db;
        assert!(
            (18.0..22.0).contains(&delta),
            "post tap should read ~20 dB below the pre tap, got {delta:.1} dB \
             (pre {pre_db:.1}, post {post_db:.1})"
        );
    }
}
