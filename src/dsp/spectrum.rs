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
/// Rate the axis assumes before the audio thread has published one, and the
/// rate `SpectrumTaps::sample_rate` falls back to.
pub const NOMINAL_SAMPLE_RATE: f32 = 48_000.0;

/// Upper FFT bin of each display band.
///
/// **The single source of truth for the band layout.** The analyzer fills the
/// bands and the axis labels them; they used to compute this independently and
/// disagreed — the analyzer forced each band to cover at least one bin more
/// than the previous, so the bottom third of the display was effectively linear
/// in bin while the axis labelled it log-spaced. A 1 kHz tone landed in band 20,
/// whose label read 187 Hz.
///
/// Logarithmic across bins 1..N/2-1. Below the crossover several bands resolve
/// the same bin and therefore share an edge: that is not a bug, it is the FFT
/// running out of resolution.
pub fn band_upper_bins() -> [usize; BINS] {
    let first_bin = 1.0f32;
    let last_bin = FFT_SIZE / 2 - 1;
    let decades = (last_bin as f32 / first_bin).log10();
    let mut edges = [1usize; BINS];
    for (b, edge) in edges.iter_mut().enumerate() {
        let target = first_bin * 10f32.powf(decades * (b + 1) as f32 / BINS as f32);
        *edge = (target.round() as usize).clamp(1, last_bin);
    }
    // An edge may never go backwards, which the rounding could otherwise make
    // it do between neighbouring bands near the top.
    let mut last = 1usize;
    for edge in edges.iter_mut() {
        last = last.max(*edge);
        *edge = last;
    }
    edges
}

/// Frequency of the upper edge of band `i`, in Hz, at `sample_rate`.
///
/// A bin's frequency is `bin * sample_rate / FFT_SIZE`, so the old nominal
/// 48 kHz made every label 8.8% high at 44.1 kHz and an octave low at 96 kHz.
/// The axis is what tells you where a peak is, so it reads the rate the audio
/// thread actually published.
pub fn bin_edge_hz(i: usize, sample_rate: f32) -> f32 {
    band_upper_bins()[i.min(BINS - 1)] as f32 * sample_rate / FFT_SIZE as f32
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
    /// Host sample rate, published by the audio thread so the frequency axis
    /// can label its bins. Zero until the first `initialize`.
    sample_rate: atomic_float::AtomicF32,
}

impl SpectrumTaps {
    pub fn new() -> Self {
        Self {
            input: SpectrumRing::new(),
            output: SpectrumRing::new(),
            sample_rate: atomic_float::AtomicF32::new(0.0),
        }
    }

    /// Publish the rate the audio thread is running at. Called from
    /// `initialize`, which runs again on a device change.
    pub fn set_sample_rate(&self, sample_rate: f32) {
        if sample_rate > 0.0 && sample_rate.is_finite() {
            self.sample_rate
                .store(sample_rate, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The rate to label the axis with. Falls back to a nominal 48 kHz before
    /// the audio thread has reported one — a zero rate would collapse every
    /// bin onto 0 Hz.
    pub fn sample_rate(&self) -> f32 {
        let rate = self.sample_rate.load(std::sync::atomic::Ordering::Relaxed);
        if rate > 0.0 {
            rate
        } else {
            NOMINAL_SAMPLE_RATE
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
    /// Second channel, used only by `push_stereo_block`. Both channels are
    /// transformed separately and their **power** spectra averaged.
    scratch_right: Vec<f32>,
    /// |X[k]|² for the current frame, summed across the channels present.
    power: Vec<f32>,
    /// Band layout, from `band_upper_bins` — the same table the axis labels.
    edges: [usize; BINS],
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
            scratch_right: vec![0.0; FFT_SIZE],
            power: vec![0.0; FFT_SIZE / 2],
            edges: band_upper_bins(),
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

    /// Push a stereo block. Both channels are transformed and their **power**
    /// spectra averaged, so a boost on one channel is visible and anti-phase
    /// content cannot cancel.
    ///
    /// Not the same as averaging the channels into one buffer first: amplitude
    /// averaging cancels out-of-phase material, and taking the root of the
    /// per-sample mean power rectifies the signal (the first attempt at this
    /// turned a 1 kHz tone into a peak at 47 Hz, which its own test caught).
    pub fn push_stereo_block(&mut self, ring: &SpectrumRing, left: &[f32], right: &[f32]) {
        let len = left.len().min(right.len());
        let mut consumed = 0;
        while consumed < len {
            let want = FFT_SIZE - self.fill;
            let take = want.min(len - consumed);
            self.scratch[self.fill..self.fill + take]
                .copy_from_slice(&left[consumed..consumed + take]);
            self.scratch_right[self.fill..self.fill + take]
                .copy_from_slice(&right[consumed..consumed + take]);
            self.fill += take;
            consumed += take;

            if self.fill == FFT_SIZE {
                let frame = self.compute_frame_stereo();
                ring.publish(frame);
                // Overlap 75% for smoother time resolution
                self.scratch.copy_within(FFT_SIZE / 4..FFT_SIZE, 0);
                self.scratch_right.copy_within(FFT_SIZE / 4..FFT_SIZE, 0);
                self.fill = FFT_SIZE * 3 / 4;
            }
        }
    }

    fn transform(&mut self, samples: &[f32]) {
        for (i, &s) in samples.iter().enumerate() {
            self.scratch_complex[i] = Complex::new(s * self.window[i], 0.0);
        }
        self.fft.process(&mut self.scratch_complex);
    }

    fn accumulate_power(&mut self) {
        for (k, power) in self.power.iter_mut().enumerate() {
            let c = self.scratch_complex[k];
            *power += c.re * c.re + c.im * c.im;
        }
    }

    fn compute_frame(&mut self) -> SpectrumFrame {
        self.power.fill(0.0);
        let scratch = std::mem::take(&mut self.scratch);
        self.transform(&scratch);
        self.scratch = scratch;
        self.accumulate_power();
        self.bands_from_power()
    }

    fn compute_frame_stereo(&mut self) -> SpectrumFrame {
        self.power.fill(0.0);
        let left = std::mem::take(&mut self.scratch);
        self.transform(&left);
        self.scratch = left;
        self.accumulate_power();

        let right = std::mem::take(&mut self.scratch_right);
        self.transform(&right);
        self.scratch_right = right;
        self.accumulate_power();

        // Mean power across channels. A hard-panned source therefore reads 3 dB
        // below a centred one at the same amplitude — it only reaches one
        // channel, so it genuinely carries half the power.
        for power in self.power.iter_mut() {
            *power *= 0.5;
        }
        self.bands_from_power()
    }

    fn bands_from_power(&mut self) -> SpectrumFrame {
        let mut frame = SpectrumFrame::silent();
        frame.seq = self.seq;
        self.seq += 1;

        // Bands come from `band_upper_bins`, the same table the axis labels:
        // the analyzer and the axis must not be able to disagree about which
        // frequency a bar represents. Band b covers bins [lo, end).
        let last_bin = FFT_SIZE / 2 - 1;
        let mut lo = 1usize;
        for (b, out) in frame.db.iter_mut().enumerate() {
            let edge = self.edges[b].clamp(1, last_bin);
            // At least one bin per band. Repeated edges mean several bands
            // resolve the same bin, which is what running out of resolution
            // looks like near the bottom of the range.
            let end = edge.max(lo + 1).min(last_bin + 1);
            let mut mag = 0.0f32;
            for power in &self.power[lo.min(last_bin)..end] {
                mag = mag.max(power.sqrt());
            }
            lo = edge.max(lo);
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
        // Read the band's frequency the way the axis does, rather than
        // recomputing the mapping here: this test used its own copy, which is
        // how the analyzer and the axis came to disagree in the first place.
        let peak_hz = bin_edge_hz(peak_b, 48_000.0);
        assert!(
            (800.0..1250.0).contains(&peak_hz),
            "peak at {} Hz (band {})",
            peak_hz,
            peak_b
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

    // ---- frequency axis -------------------------------------------------

    #[test]
    fn bin_frequencies_scale_with_the_sample_rate() {
        // The bug this replaced: a nominal 48 kHz was hardcoded, so every label
        // was 8.8% high at 44.1 kHz and an octave low at 96 kHz.
        for i in [0usize, 10, 40, BINS - 1] {
            let at_48 = bin_edge_hz(i, 48_000.0);
            let at_96 = bin_edge_hz(i, 96_000.0);
            let at_44 = bin_edge_hz(i, 44_100.0);
            assert!(
                (at_96 - 2.0 * at_48).abs() < 1e-3,
                "bin {i}: 96 kHz should be twice 48 kHz ({at_96} vs {at_48})"
            );
            let expected = at_48 * 44_100.0 / 48_000.0;
            assert!(
                (at_44 - expected).abs() < 1e-3,
                "bin {i}: 44.1 kHz should be scaled, not nominal ({at_44} vs {expected})"
            );
        }
    }

    #[test]
    fn the_axis_never_goes_backwards_at_every_rate() {
        // Non-decreasing, not strictly increasing: below the crossover several
        // display bands resolve the same FFT bin, so equal neighbours are
        // expected and correct.
        for rate in [44_100.0f32, 48_000.0, 96_000.0] {
            let mut previous = 0.0;
            for i in 0..BINS {
                let hz = bin_edge_hz(i, rate);
                assert!(
                    hz >= previous,
                    "bin {i} at {rate} goes backwards: {hz} after {previous}"
                );
                previous = hz;
            }
            assert!(bin_edge_hz(BINS / 2, rate) > bin_edge_hz(0, rate) * 10.0);
        }
    }

    #[test]
    fn the_rate_falls_back_before_the_audio_thread_reports_one() {
        let taps = SpectrumTaps::new();
        assert_eq!(taps.sample_rate(), NOMINAL_SAMPLE_RATE);
        taps.set_sample_rate(44_100.0);
        assert_eq!(taps.sample_rate(), 44_100.0);
        // Garbage must not poison the axis: a zero rate would collapse every
        // bin onto 0 Hz and a NaN would make every label NaN.
        taps.set_sample_rate(0.0);
        taps.set_sample_rate(f32::NAN);
        assert_eq!(taps.sample_rate(), 44_100.0);
    }

    // ---- stereo analysis ------------------------------------------------

    /// Peak display band for a stereo pair, and its level.
    fn analyze(left: &[f32], right: &[f32], rate: f32) -> (usize, f32) {
        let ring = SpectrumRing::new();
        let mut analyzer = SpectrumAnalyzer::new(rate);
        for (l, r) in left.chunks(128).zip(right.chunks(128)) {
            analyzer.push_stereo_block(&ring, l, r);
        }
        let frame = ring.read_newest();
        let peak = (0..BINS)
            .max_by(|a, b| frame.db[*a].total_cmp(&frame.db[*b]))
            .unwrap();
        (peak, frame.db[peak])
    }

    fn tone_at(hz: f32, amp: f32, rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * hz * i as f32 / rate).sin() * amp)
            .collect()
    }

    #[test]
    fn a_hard_panned_tone_is_analysed_like_a_centred_one() {
        // The bug: both taps fed channel 0, so a boost applied to the right
        // channel was invisible and a sound panned right read as silence.
        let rate = 48_000.0f32;
        let n = FFT_SIZE * 4;
        let tone = tone_at(1_000.0, 0.5, rate, n);
        let silence = vec![0.0f32; n];

        let (left_peak, left_db) = analyze(&tone, &silence, rate);
        let (right_peak, right_db) = analyze(&silence, &tone, rate);

        assert_eq!(
            left_peak, right_peak,
            "which channel carries the signal must not matter"
        );
        assert!(
            (left_db - right_db).abs() < 0.5,
            "hard left reads {left_db:.1} dB, hard right reads {right_db:.1} dB"
        );
        let hz = bin_edge_hz(left_peak, rate);
        assert!(
            (hz - 1_000.0).abs() < 200.0,
            "peak at {hz} Hz, expected 1 kHz"
        );
    }

    #[test]
    fn anti_phase_channels_are_not_cancelled() {
        // Averaging amplitude would show silence here while the room is full of
        // sound — a real and easy mistake.
        let rate = 48_000.0f32;
        let n = FFT_SIZE * 4;
        let left = tone_at(1_000.0, 0.5, rate, n);
        let right: Vec<f32> = left.iter().map(|s| -s).collect();

        let (peak, db) = analyze(&left, &right, rate);
        let hz = bin_edge_hz(peak, rate);
        assert!((hz - 1_000.0).abs() < 200.0, "peak at {hz} Hz");
        assert!(db > -20.0, "anti-phase content was cancelled: {db:.1} dB");
    }

    #[test]
    fn a_panned_source_reads_three_db_below_a_centred_one() {
        // Not a bug: the same amplitude in one channel carries half the power
        // of the same amplitude in two, so power averaging puts a hard-panned
        // source 3 dB lower. What must not happen is reading as silence.
        let rate = 48_000.0f32;
        let n = FFT_SIZE * 4;
        let tone = tone_at(1_000.0, 0.5, rate, n);
        let silence = vec![0.0f32; n];

        let (_, centred) = analyze(&tone, &tone, rate);
        let (_, panned) = analyze(&tone, &silence, rate);
        let delta = centred - panned;
        assert!(
            (delta - 3.0).abs() < 1.0,
            "centred {centred:.1} dB vs panned {panned:.1} dB (delta {delta:.1})"
        );
        assert!(panned > -20.0, "a panned source must still be visible");
    }
}
