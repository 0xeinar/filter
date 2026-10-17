//! Frequency-response curve computation and plot coordinate mapping.
//!
//! Pure math, no GUI dependencies — fully unit-testable.

use crate::{FilterType, MAX_BANDS};

/// Plot axis ranges: 20 Hz – 20 kHz, ±24 dB.
pub const FREQ_MIN: f32 = 20.0;
pub const FREQ_MAX: f32 = 20_000.0;
pub const DB_MAX: f32 = 24.0;

/// Number of points to evaluate when drawing the response curve.
pub const CURVE_POINTS: usize = 256;

/// Logarithmic x position (0.0–1.0) for a frequency within the plot range.
pub fn freq_to_x(freq: f32) -> f32 {
    let clamped = freq.clamp(FREQ_MIN, FREQ_MAX);
    ((clamped / FREQ_MIN).log10()) / ((FREQ_MAX / FREQ_MIN).log10())
}

/// Frequency for a logarithmic x position (0.0–1.0).
pub fn x_to_freq(x: f32) -> f32 {
    let clamped = x.clamp(0.0, 1.0);
    FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(clamped)
}

/// y position (0.0–1.0, 0.5 = 0 dB) for a gain in dB within ±DB_MAX.
pub fn db_to_y(db: f32) -> f32 {
    let clamped = db.clamp(-DB_MAX, DB_MAX);
    0.5 - clamped / (2.0 * DB_MAX)
}

/// Gain in dB for a y position (0.0–1.0, 0.5 = 0 dB).
pub fn y_to_db(y: f32) -> f32 {
    let clamped = y.clamp(0.0, 1.0);
    (0.5 - clamped) * 2.0 * DB_MAX
}

/// Input parameters for one EQ band.
#[derive(Debug, Clone, Copy)]
pub struct BandSpec {
    pub freq: f32,
    pub gain_db: f32,
    pub q: f32,
    pub filter_type: FilterType,
    pub enabled: bool,
}

/// Combined response of all bands at a given frequency.
///
/// Disabled bands are skipped entirely (not treated as 0 dB peaking filters).
pub fn combined_response_db(bands: &[BandSpec; MAX_BANDS], num_active: usize, freq: f32) -> f32 {
    // Evaluate each band's response analytically via its biquad coefficients.
    let mut total_db = 0.0f32;
    for spec in bands.iter().take(num_active) {
        if !spec.enabled {
            continue;
        }
        total_db += band_response_db(spec, freq);
    }
    total_db
}

/// Analytic magnitude (dB) of one band at one frequency.
///
/// Builds a throwaway biquad with the Audio EQ Cookbook coefficients — the
/// same math as `BiquadFilter::update_coefficients` — and evaluates |H(e^jw)|.
pub fn band_response_db(spec: &BandSpec, freq: f32) -> f32 {
    // Share coefficient computation with the DSP by going through a real filter.
    let mut f = super::biquad::BiquadFilter::new();
    f.update_coefficients(spec.freq, spec.gain_db, spec.q, spec.filter_type, 48_000.0);
    let mag = f.magnitude_at_freq(freq, 48_000.0);
    20.0 * mag.max(1e-6).log10()
}

/// Sample the combined response across the whole plot.
pub fn response_curve(bands: &[BandSpec; MAX_BANDS], num_active: usize, points: usize) -> Vec<f32> {
    (0..points)
        .map(|i| {
            let x = i as f32 / (points - 1).max(1) as f32;
            let freq = x_to_freq(x);
            combined_response_db(bands, num_active, freq)
        })
        .collect()
}

/// A frequency that lands on a nice round value for grid lines.
pub const GRID_FREQS: [f32; 13] = [
    20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10_000.0,
    20_000.0,
];

/// Grid gains for horizontal lines (0 dB line drawn separately, emphasized).
pub const GRID_DBS: [f32; 8] = [-24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilterType;

    fn spec(freq: f32, gain: f32, q: f32, t: FilterType) -> BandSpec {
        BandSpec {
            freq,
            gain_db: gain,
            q,
            filter_type: t,
            enabled: true,
        }
    }

    #[test]
    fn freq_x_roundtrip() {
        for f in [20.0, 100.0, 1000.0, 440.0, 20_000.0] {
            let x = freq_to_x(f);
            let back = x_to_freq(x);
            assert!((back - f).abs() / f < 1e-4, "f={} back={}", f, back);
        }
    }

    #[test]
    fn freq_x_endpoints() {
        assert_eq!(freq_to_x(20.0), 0.0);
        assert!((freq_to_x(20_000.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn freq_x_log_midpoint() {
        // 20Hz→20kHz spans 3 decades; 2kHz is 2/3 across
        let x = freq_to_x(2000.0);
        assert!((x - 2.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn db_y_roundtrip() {
        for db in [-24.0, -6.0, 0.0, 6.0, 24.0] {
            let y = db_to_y(db);
            let back = y_to_db(y);
            assert!((back - db).abs() < 1e-4);
        }
    }

    #[test]
    fn db_y_center_is_zero() {
        assert!((db_to_y(0.0) - 0.5).abs() < 1e-6);
        assert!((y_to_db(0.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn flat_curve_is_zero_db() {
        let bands: [BandSpec; MAX_BANDS] = core::array::from_fn(|_| BandSpec {
            freq: 1000.0,
            gain_db: 6.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        });
        let curve = response_curve(&bands, MAX_BANDS, 64);
        assert!(curve.iter().all(|&db| db.abs() < 1e-4));
    }

    #[test]
    fn disabled_band_contributes_nothing() {
        let mut bands: [BandSpec; MAX_BANDS] = core::array::from_fn(|_| BandSpec {
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        });
        bands[0] = spec(1000.0, 12.0, 1.0, FilterType::Bell);
        bands[0].enabled = false;
        let curve = response_curve(&bands, MAX_BANDS, 64);
        assert!(curve.iter().all(|&db| db.abs() < 1e-3));
    }

    #[test]
    fn bell_peaks_at_its_frequency() {
        let mut bands: [BandSpec; MAX_BANDS] = core::array::from_fn(|_| BandSpec {
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        });
        bands[0] = spec(1000.0, 12.0, 1.0, FilterType::Bell);

        let at_peak = band_response_db(&bands[0], 1000.0);
        assert!((at_peak - 12.0).abs() < 0.05, "got {}", at_peak);

        let away = band_response_db(&bands[0], 200.0);
        assert!(away < 1.0, "should fall off, got {}", away);
    }

    #[test]
    fn low_cut_attenuates_lows() {
        let mut bands: [BandSpec; MAX_BANDS] = core::array::from_fn(|_| BandSpec {
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        });
        bands[0] = spec(100.0, 0.0, 0.707, FilterType::LowCut);

        let low = band_response_db(&bands[0], 30.0);
        let high = band_response_db(&bands[0], 5000.0);
        assert!(low < -6.0, "30Hz should be cut, got {}", low);
        assert!(high > -1.0, "5kHz should pass, got {}", high);
    }

    #[test]
    fn two_bands_sum() {
        let mut bands: [BandSpec; MAX_BANDS] = core::array::from_fn(|_| BandSpec {
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        });
        bands[0] = spec(100.0, 6.0, 2.0, FilterType::Bell);
        bands[1] = spec(10_000.0, 6.0, 2.0, FilterType::Bell);

        // At 1kHz both are far from their centers, so near 0
        let mid = band_response_db(&bands[0], 1000.0) + band_response_db(&bands[1], 1000.0);
        assert!(mid.abs() < 2.0);

        // At each center, near +6
        assert!(band_response_db(&bands[0], 100.0) > 5.0);
        assert!(band_response_db(&bands[1], 10_000.0) > 5.0);
    }

    #[test]
    fn response_curve_matches_point_evaluation() {
        let mut bands: [BandSpec; MAX_BANDS] = core::array::from_fn(|_| BandSpec {
            freq: 1000.0,
            gain_db: 0.0,
            q: 1.0,
            filter_type: FilterType::Bell,
            enabled: false,
        });
        bands[0] = spec(700.0, 9.0, 1.5, FilterType::Bell);
        bands[1] = spec(4000.0, -6.0, 1.0, FilterType::HighShelf);

        let curve = response_curve(&bands, MAX_BANDS, 256);
        let x = freq_to_x(700.0);
        let idx = (x * 255.0).round() as usize;
        let expected = band_response_db(&bands[0], 700.0) + band_response_db(&bands[1], 700.0);
        assert!(
            (curve[idx] - expected).abs() < 0.3,
            "curve[{}] = {} vs {}",
            idx,
            curve[idx],
            expected
        );
    }
}
