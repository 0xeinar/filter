pub mod biquad;
pub mod curve;
pub mod spectrum;

pub use biquad::{BiquadCoeffs, BiquadFilter, BiquadState};

/// Crossfade between the dry and the processed signal.
///
/// `mix` is 0.0 for fully processed, 1.0 for fully bypassed. Bypass is a
/// crossfade rather than a switch so toggling it cannot click.
#[inline]
pub fn mix_dry_wet(dry: f32, wet: f32, mix: f32) -> f32 {
    // Exact at the endpoints: a fully bypassed EQ must be bit-identical to its
    // input, not "within rounding" of it.
    if mix >= 1.0 {
        return dry;
    }
    if mix <= 0.0 {
        return wet;
    }
    wet + (dry - wet) * mix
}

/// The plugin's output stage for one sample: output trim, then the bypass
/// crossfade. Bypass mixes towards the untouched dry signal, so it stays
/// correct even when the trim is not 0 dB.
#[inline]
pub fn output_stage(dry: f32, wet: f32, gain_lin: f32, bypass_mix: f32) -> f32 {
    mix_dry_wet(dry, wet * gain_lin, bypass_mix)
}

#[cfg(test)]
mod tests {
    use super::{mix_dry_wet, output_stage};

    #[test]
    fn bypass_returns_the_dry_signal_even_with_output_gain() {
        // 6 dB of trim and a heavily filtered wet path: bypassed output must be
        // bit-identical to the input.
        let out = output_stage(0.3, -0.9, 2.0, 1.0);
        assert_eq!(out, 0.3);
    }

    #[test]
    fn fully_wet_applies_the_output_trim() {
        let out = output_stage(0.3, 0.5, 2.0, 0.0);
        assert!((out - 1.0).abs() < 1e-7);
    }

    #[test]
    fn crossfade_is_monotonic_between_dry_and_wet() {
        let a = output_stage(0.0, 1.0, 1.0, 0.25);
        let b = output_stage(0.0, 1.0, 1.0, 0.75);
        assert!(
            a > b,
            "more bypass should move the output towards dry (0.0)"
        );
    }

    #[test]
    fn mix_endpoints_are_exact() {
        assert_eq!(mix_dry_wet(0.25, -0.5, 0.0), -0.5);
        assert_eq!(mix_dry_wet(0.25, -0.5, 1.0), 0.25);
    }

    #[test]
    fn mix_halfway_is_the_average() {
        assert!((mix_dry_wet(1.0, 0.0, 0.5) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn mix_clamps_out_of_range_inputs() {
        assert_eq!(mix_dry_wet(0.25, -0.5, 1.5), 0.25);
        assert_eq!(mix_dry_wet(0.25, -0.5, -0.5), -0.5);
    }
}
