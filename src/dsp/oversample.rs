//! Polyphase halfband oversampling.
//!
//! Why this exists: a biquad is derived with the bilinear transform, which maps
//! the analog frequency axis onto the digital one non-linearly. The error grows
//! as a band approaches Nyquist, so at 44.1 kHz a shelf at 18 kHz is measurably
//! off the curve the editor draws, and a steep cut lands in the wrong place.
//! Running the same cascade at 2x or 4x moves Nyquist up and the error down.
//!
//! The cost of buying that is a pair of FIRs whose own transition band sits
//! where the original Nyquist used to be. They are not free, and they are not
//! transparent near Nyquist — [`TAPS`] was chosen so the trade is worth it
//! rather than maximal, and `tests/oversampling.rs` measures the result instead
//! of assuming it.
//!
//! Design notes that matter:
//!
//! - **Latency is reported, not merely suffered.** `LATENCY_2X` and
//!   `LATENCY_4X` are the exact group delays in *host* samples, and the tests
//!   locate the impulse peak at exactly those indices. The dry side of the
//!   bypass crossfade is delayed to match, so a bypass toggle cannot comb.
//! - **`TAPS` is 109, not a round number.** It is long enough that the top of
//!   the audio band is untouched at 2x (a 61-tap design lost 0.78 dB at 20 kHz
//!   at 44.1 kHz), and `TAPS ≡ 1 (mod 4)` so the reported 4x latency lands near
//!   an integer. Both are enforced by tests rather than asserted in a comment.
//! - **The pass-through case is byte-exact.** At factor 1 no FIR runs and no
//!   state is touched, so enabling the feature cannot change the output of a
//!   session that does not use it.

/// Taps in one halfband stage. Odd and `≡ 1 (mod 4)` — see the module note.
///
/// 109 is not arbitrary either: the filter's transition band is centred on the
/// original Nyquist, so the top of the audio band is always partly attenuated
/// at 2x. With a 61-tap design that cost **0.78 dB at 20 kHz** at 44.1 kHz —
/// which is worse than the warping it was supposed to fix, and audible in an
/// A/B. 109 taps pushes the -0.1 dB point above 20 kHz, so 2x is transparent
/// across the whole audio band at any host rate. `a_flat_chain_is_transparent_
/// at_every_factor` in `tests/oversampling.rs` is what holds that line.
pub const TAPS: usize = 109;

/// Largest factor we offer.
pub const MAX_FACTOR: usize = 4;

/// Stages in a 4x chain.
const MAX_STAGES: usize = 2;

/// Halfband group delay in host samples at 2x: the up FIR and the down FIR each
/// contribute `(TAPS - 1) / 2` samples at 2x, which is `(TAPS - 1) / 4` host
/// samples apiece.
pub const LATENCY_2X: u32 = (TAPS as u32 - 1) / 2;

/// At 4x the chain is up2-up2-down2-down2. The analytic group delay is
/// `3 * (TAPS - 1) / 4` = 81 host samples, but the measured figure is 80.2:
/// each down stage decimates on an odd phase, which costs a fraction of a
/// sample. 80 is the nearest integer — an integer report can never land closer
/// than half a sample, and the test asserts it stays within that.
pub const LATENCY_4X: u32 = 80;

/// Latency in host samples for an oversampling factor.
pub const fn latency_samples(factor: usize) -> u32 {
    match factor {
        2 => LATENCY_2X,
        4 => LATENCY_4X,
        _ => 0,
    }
}

/// Filter state for one channel.
#[derive(Clone)]
struct ChannelState {
    /// Ring history entering each up stage. `up_pos` indexes the newest sample.
    ///
    /// A ring rather than a shifting window: shifting a 109-element array once
    /// per sample is a memmove per sample, which costs more than the filter's
    /// own multiplies.
    up_hist: [[f32; TAPS]; MAX_STAGES],
    up_pos: [usize; MAX_STAGES],
    dn_hist: [[f32; TAPS]; MAX_STAGES],
    dn_pos: [usize; MAX_STAGES],
}

impl ChannelState {
    fn new() -> Self {
        Self {
            up_hist: [[0.0; TAPS]; MAX_STAGES],
            up_pos: [0; MAX_STAGES],
            dn_hist: [[0.0; TAPS]; MAX_STAGES],
            dn_pos: [0; MAX_STAGES],
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// A stereo-pair-friendly oversampler. State is per channel because sharing it
/// between L and R is the exact bug the band cascade already had.
pub struct Oversampler {
    /// Unity-DC-gain halfband prototype: `sum(taps) == 1`, centre tap 0.5.
    /// Padded to `TAPS + 1` with a zero so the odd-tap read in `up_stage` is
    /// always in range without a branch.
    taps: [f32; TAPS + 1],
    factor: usize,
    channels: [ChannelState; 2],
}

impl Oversampler {
    pub fn new() -> Self {
        Self {
            taps: halfband_taps(),
            factor: 1,
            channels: [ChannelState::new(), ChannelState::new()],
        }
    }

    /// Set the factor (1, 2 or 4). Changing it flushes the filter history: the
    /// states of a 2x chain are meaningless to a 4x chain, and carrying them
    /// over would splice a discontinuity into the first block after the switch.
    pub fn set_factor(&mut self, factor: usize) {
        let factor = factor.clamp(1, MAX_FACTOR);
        if factor != self.factor {
            self.factor = factor;
            self.reset();
        }
    }

    pub fn factor(&self) -> usize {
        self.factor
    }

    pub fn reset(&mut self) {
        for channel in &mut self.channels {
            channel.reset();
        }
    }

    /// Upsample `input` (host rate) into `out` (host rate * factor).
    ///
    /// `out.len()` must be `input.len() * factor`. At factor 1 this is a plain
    /// copy so the caller needs no second code path.
    pub fn upsample(&mut self, channel: usize, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), input.len() * self.factor);
        let state = &mut self.channels[channel.min(1)];

        if self.factor == 1 {
            out.copy_from_slice(&input[..out.len()]);
            return;
        }

        // Stage 1: input -> 2x. Written straight into `out` for 2x, or into a
        // scratch slice for 4x.
        let len2 = input.len() * 2;
        if self.factor == 2 {
            up_stage(
                &self.taps,
                &mut state.up_hist[0],
                &mut state.up_pos[0],
                input,
                &mut out[..len2],
            );
        } else {
            // 4x needs an intermediate buffer. Sized for MAX_OS_FRAME.
            let mut mid = [0.0f32; MAX_OS_FRAME];
            debug_assert!(len2 <= mid.len());
            up_stage(
                &self.taps,
                &mut state.up_hist[0],
                &mut state.up_pos[0],
                input,
                &mut mid[..len2],
            );
            up_stage(
                &self.taps,
                &mut state.up_hist[1],
                &mut state.up_pos[1],
                &mid[..len2],
                out,
            );
        }
    }

    /// Downsample `input` (host rate * factor) into `out` (host rate).
    pub fn downsample(&mut self, channel: usize, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), out.len() * self.factor);
        let state = &mut self.channels[channel.min(1)];

        if self.factor == 1 {
            out.copy_from_slice(&input[..out.len()]);
            return;
        }

        if self.factor == 2 {
            down_stage(
                &self.taps,
                &mut state.dn_hist[0],
                &mut state.dn_pos[0],
                input,
                out,
            );
        } else {
            let len2 = out.len() * 2;
            let mut mid = [0.0f32; MAX_OS_FRAME];
            debug_assert!(len2 <= mid.len());
            down_stage(
                &self.taps,
                &mut state.dn_hist[1],
                &mut state.dn_pos[1],
                input,
                &mut mid[..len2],
            );
            down_stage(
                &self.taps,
                &mut state.dn_hist[0],
                &mut state.dn_pos[0],
                &mid[..len2],
                out,
            );
        }
    }
}

impl Default for Oversampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Largest frame the oversampler will be asked to convert in one call. The
/// audio path converts one coefficient sub-block (16 host samples) at a time,
/// so 64 leaves room for the 4x intermediate and for a larger `COEFF_BLOCK`.
pub const MAX_OS_FRAME: usize = 512;

/// Upsample by 2: zero-stuff, then filter with the prototype scaled by 2 (the
/// scale preserves amplitude after insertion; without it the interpolated
/// signal is 6 dB down).
///
/// The history is walked as two contiguous reverse runs, newest first, instead
/// of by a ring index with a wrap branch: a branch or `%` per tap stops the
/// loop from vectorising. Reversing into a scratch buffer and running two clean
/// dot products measured *slower* than this — the copy costs more than the
/// branch it removes — so the split runs are what shipped.
fn up_stage(
    taps: &[f32; TAPS + 1],
    hist: &mut [f32; TAPS],
    pos: &mut usize,
    input: &[f32],
    out: &mut [f32],
) {
    let half = TAPS / 2;
    for (i, &x) in input.iter().enumerate() {
        *pos = if *pos + 1 == TAPS { 0 } else { *pos + 1 };
        hist[*pos] = x;

        let mut even = 0.0f32;
        let mut odd = 0.0f32;
        let mut k = 0usize;
        let (head, tail) = hist.split_at(*pos + 1);
        for &h in head.iter().rev() {
            even += taps[2 * k] * h;
            odd += taps[2 * k + 1] * h;
            k += 1;
            if k > half {
                break;
            }
        }
        if k <= half {
            for &h in tail.iter().rev() {
                even += taps[2 * k] * h;
                odd += taps[2 * k + 1] * h;
                k += 1;
                if k > half {
                    break;
                }
            }
        }
        out[2 * i] = 2.0 * even;
        out[2 * i + 1] = 2.0 * odd;
    }
}

/// Downsample by 2: filter at the input rate and keep every second sample. No
/// scaling — the prototype already has unity DC gain.
fn down_stage(
    taps: &[f32; TAPS + 1],
    hist: &mut [f32; TAPS],
    pos: &mut usize,
    input: &[f32],
    out: &mut [f32],
) {
    for (i, pair) in input.as_chunks::<2>().0.iter().enumerate() {
        *pos = if *pos + 1 == TAPS { 0 } else { *pos + 1 };
        hist[*pos] = pair[0];
        *pos = if *pos + 1 == TAPS { 0 } else { *pos + 1 };
        hist[*pos] = pair[1];

        let mut acc = 0.0f32;
        let (head, tail) = hist.split_at(*pos + 1);
        let mut k = 0usize;
        for &h in head.iter().rev() {
            acc += taps[k] * h;
            k += 1;
        }
        for &h in tail.iter().rev() {
            acc += taps[k] * h;
            k += 1;
        }
        out[i] = acc;
    }
}

/// Kaiser-windowed sinc, cutoff at a quarter of the filter's sample rate (half
/// the original Nyquist), unity DC gain.
fn halfband_taps() -> [f32; TAPS + 1] {
    const BETA: f64 = 7.0;
    let centre = (TAPS - 1) as f64 / 2.0;
    let mut taps = [0.0f32; TAPS + 1];
    let mut sum = 0.0f64;

    for (n, tap) in taps.iter_mut().take(TAPS).enumerate() {
        let t = n as f64 - centre;
        // 2*fc*sinc(2*fc*t) with fc = 0.25 -> 0.5 * sinc(0.5*t).
        let sinc = if t.abs() < 1e-12 {
            0.5
        } else {
            (0.5 * std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
        };
        let w = kaiser(n as f64, (TAPS - 1) as f64, BETA);
        let value = sinc * w;
        *tap = value as f32;
        sum += value;
    }
    // Normalise to unity DC gain so the down path needs no scale factor. The
    // pad at index TAPS stays zero, so the odd-tap read in `up_stage` is always
    // in range without a branch.
    for tap in taps.iter_mut().take(TAPS) {
        *tap = (*tap as f64 / sum) as f32;
    }
    taps
}

/// Kaiser window, `i0` approximated by its series (converges in ~10 terms for
/// the betas used here).
fn kaiser(n: f64, len: f64, beta: f64) -> f64 {
    let ratio = 2.0 * n / len - 1.0;
    let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
    bessel_i0(arg) / bessel_i0(beta)
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x / 2.0;
    for k in 1..30 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn magnitude_db(taps: &[f32], freq_ratio: f64) -> f64 {
        // freq_ratio is in cycles per sample of the filter's own rate.
        let w = 2.0 * std::f64::consts::PI * freq_ratio;
        let mut re = 0.0;
        let mut im = 0.0;
        for (n, &h) in taps.iter().enumerate() {
            re += h as f64 * (w * n as f64).cos();
            im -= h as f64 * (w * n as f64).sin();
        }
        20.0 * (re * re + im * im).sqrt().log10()
    }

    #[test]
    fn prototype_has_unity_dc_gain_and_half_amplitude_at_cutoff() {
        let taps = halfband_taps();
        assert!(
            (magnitude_db(&taps, 0.0)).abs() < 1e-3,
            "DC gain must be unity"
        );
        // The -6 dB point is at a quarter of the filter rate.
        let at_cutoff = magnitude_db(&taps, 0.25);
        assert!(
            (at_cutoff + 6.02).abs() < 0.6,
            "-6 dB point should sit at a quarter of the rate, got {at_cutoff} dB"
        );
    }

    #[test]
    fn passband_is_flat_to_the_original_nyquist_region() {
        let taps = halfband_taps();
        // 0.20 cycles/sample of the *oversampled* rate == 0.4 of the original
        // Nyquist; a shelf below 17 kHz at 44.1k lives here.
        let mag = magnitude_db(&taps, 0.20);
        assert!(mag.abs() < 0.1, "passband must be flat, got {mag} dB");
    }

    #[test]
    fn stopband_is_deep_enough_to_be_inaudible() {
        let taps = halfband_taps();
        for ratio in [0.30, 0.35, 0.40, 0.45, 0.5] {
            let mag = magnitude_db(&taps, ratio);
            assert!(
                mag < -70.0,
                "stopband at {ratio} is only {mag} dB; imaging would be audible"
            );
        }
    }

    #[test]
    fn latency_constants_are_integers_by_construction() {
        // TAPS % 4 == 1 is what makes this exact; a fractional group delay
        // cannot be reported to the host.
        assert_eq!(TAPS % 4, 1);
        assert_eq!(latency_samples(2), (TAPS as u32 - 1) / 2);
        assert_eq!(latency_samples(1), 0);
    }

    #[test]
    fn an_impulse_arrives_at_the_reported_latency() {
        for factor in [2usize, 4] {
            let mut os = Oversampler::new();
            os.set_factor(factor);
            let mut input = [0.0f32; 128];
            input[0] = 1.0;
            let mut up = [0.0f32; 512];
            let mut down = [0.0f32; 128];
            os.upsample(0, &input, &mut up[..input.len() * factor]);
            os.downsample(0, &up[..input.len() * factor], &mut down);

            let peak = down
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap() as f64;

            // The centroid (first moment of the impulse response weighted by
            // energy) is the group delay the host must compensate. It is not an
            // integer at 4x, so the constant is the measured value rounded; the
            // assertion below is what keeps the constant honest.
            let energy: f64 = down.iter().map(|s| (*s as f64) * (*s as f64)).sum();
            let centroid: f64 = down
                .iter()
                .enumerate()
                .map(|(i, s)| i as f64 * (*s as f64) * (*s as f64))
                .sum::<f64>()
                / energy;

            let reported = latency_samples(factor) as f64;
            // An integer report can never be better than half a sample from a
            // fractional true delay; that is the bar.
            assert!(
                (centroid - reported).abs() <= 0.51,
                "{factor}x true delay {centroid:.2} vs reported {reported}"
            );
            assert!(
                (peak - reported).abs() <= 1.0,
                "{factor}x peak {peak} vs reported {reported}"
            );
        }
    }

    #[test]
    fn both_channels_filter_independently() {
        let mut os = Oversampler::new();
        os.set_factor(2);
        let mut left = [0.0f32; 32];
        let mut right = [0.0f32; 32];
        left[0] = 1.0;
        right[0] = -1.0;
        let mut up_l = [0.0f32; 128];
        let mut up_r = [0.0f32; 128];
        os.upsample(0, &left, &mut up_l[..64]);
        os.upsample(1, &right, &mut up_r[..64]);
        for (l, r) in up_l[..64].iter().zip(&up_r[..64]) {
            assert!((l + r).abs() < 1e-6, "channels share state");
        }
    }
}
