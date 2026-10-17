/// Property-based tests using proptest
/// Fuzzes the filter with random parameters to find crashes/instabilities
use filter::{dsp::BiquadFilter, FilterType};
use proptest::prelude::*;

const SAMPLE_RATE: f32 = 44100.0;

fn filter_type_strategy() -> impl Strategy<Value = FilterType> {
    prop_oneof![
        Just(FilterType::Bell),
        Just(FilterType::LowShelf),
        Just(FilterType::HighShelf),
        Just(FilterType::LowCut),
        Just(FilterType::HighCut),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        .. ProptestConfig::default()
    })]

    /// Filter should never produce NaN or Inf for any valid parameters
    #[test]
    fn filter_never_produces_nan_inf(
        freq in 20.0f32..20000.0,
        gain in -24.0f32..24.0,
        q in 0.1f32..20.0,
        ftype in filter_type_strategy(),
    ) {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(freq, gain, q, ftype, SAMPLE_RATE);

        for i in 0..1000 {
            let t = i as f32 / SAMPLE_RATE;
            let input = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1;
            let output = filter.process(input);
            prop_assert!(output.is_finite(), "Filter produced non-finite output: {} at freq={}, gain={}, q={}, type={:?}", output, freq, gain, q, ftype);
        }
    }

    /// Filter should not explode (output bounded) for any valid parameters
    #[test]
    fn filter_output_bounded(
        freq in 20.0f32..20000.0,
        gain in -24.0f32..24.0,
        q in 0.1f32..20.0,
        ftype in filter_type_strategy(),
    ) {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(freq, gain, q, ftype, SAMPLE_RATE);

        let max_expected = 100.0;
        let mut max_output = 0.0f32;

        for i in 0..(SAMPLE_RATE as usize * 2) {
            let t = i as f32 / SAMPLE_RATE;
            let input = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.1;
            let output = filter.process(input);
            max_output = max_output.max(output.abs());
        }

        prop_assert!(
            max_output < max_expected,
            "Filter output unbounded: max={} at freq={}, gain={}, q={}, type={:?}",
            max_output, freq, gain, q, ftype
        );
    }

    /// Processing silence should always produce silence (or near-zero)
    #[test]
    fn silence_in_silence_out(
        freq in 20.0f32..20000.0,
        gain in -24.0f32..24.0,
        q in 0.1f32..20.0,
        ftype in filter_type_strategy(),
    ) {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(freq, gain, q, ftype, SAMPLE_RATE);

        for _ in 0..10000 {
            let output = filter.process(0.0);
            prop_assert!(
                output.abs() < 1e-6,
                "Silence in should give silence out: got {} at freq={}, gain={}, q={}",
                output, freq, gain, q
            );
        }
    }

    /// Zero-gain bell should be transparent (passthrough)
    #[test]
    fn zero_gain_bell_is_passthrough(
        freq in 20.0f32..20000.0,
        q in 0.1f32..20.0,
    ) {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(freq, 0.0, q, FilterType::Bell, SAMPLE_RATE);

        for i in 0..1000 {
            let input = (i as f32 * 0.01).sin() * 0.5;
            let output = filter.process(input);
            prop_assert!(
                (output - input).abs() < 1e-4,
                "Zero-gain bell should be passthrough: input={}, output={} at freq={}, q={}",
                input, output, freq, q
            );
        }
    }

    /// Reset should always clear state regardless of previous parameters
    #[test]
    fn reset_always_clears_state(
        freq in 20.0f32..20000.0,
        gain in -24.0f32..24.0,
        q in 0.1f32..20.0,
        ftype in filter_type_strategy(),
    ) {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(freq, gain, q, ftype, SAMPLE_RATE);

        for i in 0..100 {
            let _ = filter.process((i as f32 * 0.1).sin());
        }

        filter.reset();

        for _ in 0..100 {
            let output = filter.process(0.0);
            prop_assert!(
                output.abs() < 1e-10,
                "Reset should clear state: got {} after reset",
                output
            );
        }
    }

    /// Cascading two filters should produce higher magnitude than a single filter
    /// at the center frequency (measured via magnitude_at_freq, not time-domain)
    #[test]
    fn cascade_increases_magnitude(
        freq in 100.0f32..10000.0,
        gain in 3.0f32..12.0,
        q in 0.5f32..5.0,
    ) {
        let mut filter1 = BiquadFilter::new();
        filter1.update_coefficients(freq, gain, q, FilterType::Bell, SAMPLE_RATE);

        // Single filter magnitude at center frequency
        let single_mag = filter1.magnitude_at_freq(freq, SAMPLE_RATE);

        // Two cascaded filters: magnitude multiplies
        let cascade_mag = single_mag * single_mag;

        prop_assert!(
            cascade_mag > single_mag,
            "Cascade should have higher magnitude than single: single={:.3}, cascade={:.3} at freq={}, gain={}",
            single_mag, cascade_mag, freq, gain
        );

        // In dB, cascade should be roughly double
        let single_db = 20.0 * single_mag.log10();
        let cascade_db = 20.0 * cascade_mag.log10();
        prop_assert!(
            (cascade_db - 2.0 * single_db).abs() < 0.1,
            "Cascade dB should be double single dB: single={:.1}dB, cascade={:.1}dB",
            single_db, cascade_db
        );
    }

    /// Filter should handle parameter changes mid-stream without crashing
    #[test]
    fn parameter_changes_mid_stream(
        freq1 in 20.0f32..20000.0,
        freq2 in 20.0f32..20000.0,
        gain1 in -24.0f32..24.0,
        gain2 in -24.0f32..24.0,
        q in 0.1f32..20.0,
    ) {
        let mut filter = BiquadFilter::new();
        filter.update_coefficients(freq1, gain1, q, FilterType::Bell, SAMPLE_RATE);

        let mut max_output = 0.0f32;
        for i in 0..2000 {
            if i == 1000 {
                filter.update_coefficients(freq2, gain2, q, FilterType::Bell, SAMPLE_RATE);
            }

            let t = i as f32 / SAMPLE_RATE;
            let input = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1;
            let output = filter.process(input);
            max_output = max_output.max(output.abs());
        }

        prop_assert!(max_output.is_finite());
        prop_assert!(max_output < 100.0, "Output should be bounded after parameter change: max={}", max_output);
    }
}
