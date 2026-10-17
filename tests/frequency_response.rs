/// FFT-based frequency response tests
/// Generates sine sweeps through the filter and verifies the output spectrum
use filter::{dsp::BiquadFilter, FilterType};
use rustfft::{num_complex::Complex, FftPlanner};

const SAMPLE_RATE: f32 = 44100.0;
const FFT_SIZE: usize = 8192;

/// Generate a sine wave at a given frequency
fn generate_sine(freq: f32, num_samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Compute the magnitude spectrum of a signal using FFT
fn magnitude_spectrum(signal: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let mut buffer: Vec<Complex<f32>> = signal.iter().map(|&s| Complex::new(s, 0.0)).collect();
    buffer.resize(FFT_SIZE, Complex::new(0.0, 0.0));

    fft.process(&mut buffer);

    buffer
        .iter()
        .take(FFT_SIZE / 2)
        .map(|c| c.norm() / (FFT_SIZE as f32 / 2.0))
        .collect()
}

/// Find the magnitude at a specific frequency bin
fn magnitude_at_freq(spectrum: &[f32], freq: f32) -> f32 {
    let bin = (freq / SAMPLE_RATE * FFT_SIZE as f32).round() as usize;
    spectrum[bin.min(spectrum.len() - 1)]
}

/// Process a signal through a biquad filter
fn process_through_filter(signal: &[f32], filter: &mut BiquadFilter) -> Vec<f32> {
    signal.iter().map(|&s| filter.process(s)).collect()
}

/// Convert linear gain to dB
fn to_db(gain: f32) -> f32 {
    20.0 * gain.log10()
}

/// Measure the gain change at a specific frequency
fn measure_gain_change(input: &[f32], output: &[f32], freq: f32) -> f32 {
    let in_spectrum = magnitude_spectrum(input);
    let out_spectrum = magnitude_spectrum(output);
    let in_mag = magnitude_at_freq(&in_spectrum, freq);
    let out_mag = magnitude_at_freq(&out_spectrum, freq);
    to_db(out_mag / in_mag)
}

#[test]
fn fft_bell_boost_at_1khz() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);

    let input = generate_sine(1000.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    let gain_change = measure_gain_change(&input, &output, 1000.0);
    assert!(
        (gain_change - 6.0).abs() < 1.0,
        "Bell +6dB at 1kHz: expected ~+6dB, got {:.1}dB",
        gain_change
    );
}

#[test]
fn fft_bell_cut_at_1khz() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, -12.0, 1.0, FilterType::Bell, SAMPLE_RATE);

    let input = generate_sine(1000.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    let gain_change = measure_gain_change(&input, &output, 1000.0);
    assert!(
        (gain_change - (-12.0)).abs() < 1.0,
        "Bell -12dB at 1kHz: expected ~-12dB, got {:.1}dB",
        gain_change
    );
}

#[test]
fn fft_bell_does_not_affect_other_frequencies() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, 12.0, 1.0, FilterType::Bell, SAMPLE_RATE);

    // Generate a signal with content at both 1kHz and 10kHz
    let input: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                + (2.0 * std::f32::consts::PI * 10000.0 * t).sin()
        })
        .collect();

    let output = process_through_filter(&input, &mut filter);

    // 1kHz should be boosted by ~12dB
    let gain_1k = measure_gain_change(&input, &output, 1000.0);
    assert!(
        gain_1k > 10.0,
        "1kHz should be boosted: got {:.1}dB",
        gain_1k
    );

    // 10kHz should be relatively unaffected (< 3dB change)
    let gain_10k = measure_gain_change(&input, &output, 10000.0);
    assert!(
        gain_10k.abs() < 3.0,
        "10kHz should be unaffected: got {:.1}dB",
        gain_10k
    );
}

#[test]
fn fft_low_cut_removes_low_frequencies() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, 0.0, 0.707, FilterType::LowCut, SAMPLE_RATE);

    // Generate low frequency content (100Hz)
    let input = generate_sine(100.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    let gain_change = measure_gain_change(&input, &output, 100.0);
    assert!(
        gain_change < -10.0,
        "Low cut at 1kHz should heavily attenuate 100Hz: got {:.1}dB",
        gain_change
    );
}

#[test]
fn fft_high_cut_removes_high_frequencies() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, 0.0, 0.707, FilterType::HighCut, SAMPLE_RATE);

    // Generate high frequency content (10kHz)
    let input = generate_sine(10000.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    let gain_change = measure_gain_change(&input, &output, 10000.0);
    assert!(
        gain_change < -10.0,
        "High cut at 1kHz should heavily attenuate 10kHz: got {:.1}dB",
        gain_change
    );
}

#[test]
fn fft_low_shelf_boosts_below_cutoff() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(500.0, 6.0, 0.707, FilterType::LowShelf, SAMPLE_RATE);

    let input = generate_sine(100.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    let gain_change = measure_gain_change(&input, &output, 100.0);
    assert!(
        gain_change > 4.0,
        "Low shelf at 500Hz should boost 100Hz: got {:.1}dB",
        gain_change
    );
}

#[test]
fn fft_high_shelf_boosts_above_cutoff() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(5000.0, 6.0, 0.707, FilterType::HighShelf, SAMPLE_RATE);

    let input = generate_sine(10000.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    let gain_change = measure_gain_change(&input, &output, 10000.0);
    assert!(
        gain_change > 4.0,
        "High shelf at 5kHz should boost 10kHz: got {:.1}dB",
        gain_change
    );
}

#[test]
fn fft_cascade_multiple_bands() {
    let mut filter1 = BiquadFilter::new();
    let mut filter2 = BiquadFilter::new();
    filter1.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);
    filter2.update_coefficients(5000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);

    // Generate signal with content at both 1kHz and 5kHz
    let input: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                + (2.0 * std::f32::consts::PI * 5000.0 * t).sin()
        })
        .collect();

    // Process through both filters in series
    let stage1 = process_through_filter(&input, &mut filter1);
    let output = process_through_filter(&stage1, &mut filter2);

    // Both frequencies should be boosted
    let gain_1k = measure_gain_change(&input, &output, 1000.0);
    let gain_5k = measure_gain_change(&input, &output, 5000.0);

    assert!(
        gain_1k > 4.0,
        "1kHz should be boosted: got {:.1}dB",
        gain_1k
    );
    assert!(
        gain_5k > 4.0,
        "5kHz should be boosted: got {:.1}dB",
        gain_5k
    );
}

#[test]
fn fft_passthrough_preserves_spectrum() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, 0.0, 1.0, FilterType::Bell, SAMPLE_RATE);

    // Generate a multi-tone signal
    let input: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                + (2.0 * std::f32::consts::PI * 2000.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 8000.0 * t).sin() * 0.2
        })
        .collect();

    let output = process_through_filter(&input, &mut filter);

    // Check that the spectrum is preserved at all frequencies
    for &freq in &[440.0, 2000.0, 8000.0] {
        let gain_change = measure_gain_change(&input, &output, freq);
        assert!(
            gain_change.abs() < 0.5,
            "Passthrough at {}Hz: expected <0.5dB change, got {:.1}dB",
            freq,
            gain_change
        );
    }
}

#[test]
fn fft_no_dc_offset_introduced() {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(1000.0, 6.0, 1.0, FilterType::Bell, SAMPLE_RATE);

    // Generate a signal with zero DC offset
    let input = generate_sine(1000.0, FFT_SIZE, SAMPLE_RATE);
    let output = process_through_filter(&input, &mut filter);

    // Check DC bin (bin 0)
    let spectrum = magnitude_spectrum(&output);
    let dc_level = spectrum[0];
    assert!(
        dc_level < 0.01,
        "Filter should not introduce DC offset: got {:.6}",
        dc_level
    );
}
