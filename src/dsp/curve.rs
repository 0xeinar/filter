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

/// Number of log-spaced points used to average the curve for auto-gain.
pub const AUTO_GAIN_POINTS: usize = 48;

/// Mean *power* gain of the current curve over the audio band, in dB,
/// weighted uniformly in log frequency (i.e. pink noise).
///
/// Averaging dB values would be wrong: energy sums linearly, so +12 dB over
/// half the octaves raises the level by about 9 dB, not 6. Measured on pink
/// noise, the dB-average under-predicts such a shelf by ~3.8 dB — the
/// `tests/auto_gain.rs` integration test is what caught that.
pub fn mean_response_db(bands: &[BandSpec; MAX_BANDS], num_active: usize) -> f32 {
    let mut total = 0.0f64;
    for i in 0..AUTO_GAIN_POINTS {
        // Half-step offset so the endpoints are not weighted double.
        let t = (i as f32 + 0.5) / AUTO_GAIN_POINTS as f32;
        let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(t);
        let db = combined_response_db(bands, num_active, freq) as f64;
        total += 10f64.powf(db / 10.0);
    }
    (10.0 * (total / AUTO_GAIN_POINTS as f64).log10()) as f32
}

/// Gain compensation in dB that keeps the overall level roughly unchanged:
/// the negative of the curve's mean gain, clamped to ±18 dB so a pathological
/// setting cannot slam the output.
pub fn auto_gain_compensation_db(bands: &[BandSpec; MAX_BANDS], num_active: usize) -> f32 {
    -mean_response_db(bands, num_active).clamp(-18.0, 18.0)
}

/// Which bands move when band `removed` is deleted from a bank of
/// `num_active` bands: the tail shifts down one slot, as `(from, to)` pairs in
/// ascending order. Applying them in this order is safe because each write
/// targets a slot whose source has not been overwritten yet.
pub fn compaction_plan(removed: usize, num_active: usize) -> Vec<(usize, usize)> {
    if removed >= num_active {
        return Vec::new();
    }
    (removed + 1..num_active)
        .map(|from| (from, from - 1))
        .collect()
}

/// Grab radius around a band node, in pixels.
pub const NODE_GRAB_RADIUS_PX: f32 = 14.0;

/// Normalized Q change per scroll notch (2% of the Q range).
pub const Q_SCROLL_STEP: f32 = 0.02;
/// Normalized Q change per scroll notch with a fine modifier held (Alt).
pub const Q_SCROLL_FINE_STEP: f32 = 0.005;

/// A grabbable band node under the cursor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeHit {
    pub band_index: usize,
    pub distance_px: f32,
}

/// Find the grabbable node nearest to a cursor position, within the grab
/// radius. Node positions are pixel coordinates `(band_index, x, y)`.
/// Overlapping nodes resolve to the visually nearest one.
pub fn nearest_grabbable_node(
    cursor_x: f32,
    cursor_y: f32,
    nodes_px: &[(usize, f32, f32)],
) -> Option<NodeHit> {
    let mut best: Option<NodeHit> = None;
    for &(band_index, nx, ny) in nodes_px {
        let dx = cursor_x - nx;
        let dy = cursor_y - ny;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > NODE_GRAB_RADIUS_PX {
            continue;
        }
        let closer = match best {
            None => true,
            Some(b) => dist < b.distance_px,
        };
        if closer {
            best = Some(NodeHit {
                band_index,
                distance_px: dist,
            });
        }
    }
    best
}

/// Normalized Q step per scroll notch. `fine` (Alt held) is 4x finer.
pub fn scroll_q_step(fine: bool) -> f32 {
    if fine {
        Q_SCROLL_FINE_STEP
    } else {
        Q_SCROLL_STEP
    }
}

/// Frequency for a pixel x position within a plot of the given width —
/// the inverse of the `freq_to_x` mapping used for drawing.
pub fn px_to_freq(px: f32, width_px: f32) -> f32 {
    let x01 = if width_px > 1.0 {
        (px / width_px).clamp(0.0, 1.0)
    } else {
        0.0
    };
    x_to_freq(x01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilterType;

    fn specs(bands: &[(f32, f32, f32, FilterType)]) -> [BandSpec; MAX_BANDS] {
        core::array::from_fn(|i| match bands.get(i) {
            Some(&(freq, gain, q, t)) => BandSpec {
                freq,
                gain_db: gain,
                q,
                filter_type: t,
                enabled: true,
            },
            None => BandSpec {
                freq: 1000.0,
                gain_db: 0.0,
                q: 1.0,
                filter_type: FilterType::Bell,
                enabled: false,
            },
        })
    }

    #[test]
    fn flat_curve_needs_no_compensation() {
        let b = specs(&[]);
        assert_eq!(auto_gain_compensation_db(&b, 0), 0.0);
        assert!(mean_response_db(&b, 0).abs() < 1e-6);
    }

    #[test]
    fn compensation_inverts_a_broad_boost() {
        // A +12 dB low shelf over most of the band pulls the mean up; the
        // compensation has to pull it back down by roughly the same amount.
        let b = specs(&[(500.0, 12.0, 0.707, FilterType::LowShelf)]);
        let mean = mean_response_db(&b, 1);
        let comp = auto_gain_compensation_db(&b, 1);
        assert!(mean > 3.0, "expected a clear mean boost, got {mean:.2} dB");
        assert!(
            (comp + mean).abs() < 1e-5,
            "compensation {comp:.2} dB should cancel the mean {mean:.2} dB"
        );
    }

    #[test]
    fn wide_shelf_is_measured_by_energy_not_by_db_average() {
        // +12 dB below 500 Hz covers ~46% of the octaves in 20 Hz..20 kHz.
        // Energy-wise that is 0.54*1 + 0.46*15.85 = 7.9x, i.e. ~9 dB, not the
        // 5.6 dB a dB-average would give.
        let b = specs(&[(500.0, 12.0, 0.707, FilterType::LowShelf)]);
        let mean = mean_response_db(&b, 1);
        assert!(
            (mean - 9.0).abs() < 1.5,
            "expected ~9 dB of level change, got {mean:.2} dB"
        );
    }

    #[test]
    fn compensation_pushes_a_broad_cut_up() {
        let b = specs(&[(8000.0, -9.0, 0.707, FilterType::HighShelf)]);
        assert!(auto_gain_compensation_db(&b, 1) > 0.0);
    }

    #[test]
    fn compensation_is_clamped() {
        // Deep, wide cuts stacked up would otherwise ask for a huge make-up.
        let b = specs(&[
            (80.0, -24.0, 0.5, FilterType::LowShelf),
            (800.0, -24.0, 0.5, FilterType::Bell),
            (8000.0, -24.0, 0.5, FilterType::HighShelf),
        ]);
        let comp = auto_gain_compensation_db(&b, 3);
        assert!((-18.0..=18.0).contains(&comp), "got {comp}");
    }

    #[test]
    fn compaction_plan_shifts_the_tail_down() {
        assert_eq!(compaction_plan(0, 3), vec![(1, 0), (2, 1)]);
        assert_eq!(compaction_plan(1, 3), vec![(2, 1)]);
        assert!(
            compaction_plan(2, 3).is_empty(),
            "deleting the last band moves nothing"
        );
        assert!(
            compaction_plan(9, 3).is_empty(),
            "out-of-range delete is a no-op"
        );
        assert!(compaction_plan(0, 1).is_empty());
    }

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

    #[test]
    fn nearest_node_within_radius() {
        // Node at (100, 150), grab radius 14
        let nodes = [(2usize, 100.0f32, 150.0f32)];
        let hit = nearest_grabbable_node(105.0, 148.0, &nodes).unwrap();
        assert_eq!(hit.band_index, 2);
        assert!(hit.distance_px <= NODE_GRAB_RADIUS_PX);
    }

    #[test]
    fn nearest_node_outside_radius_is_none() {
        let nodes = [(0usize, 100.0f32, 150.0f32)];
        // 15 px away diagonally? 10px horizontally + 11px vertically ≈ 14.9 < 14?
        // Use clearly-far point:
        assert!(nearest_grabbable_node(130.0, 190.0, &nodes).is_none());
    }

    #[test]
    fn nearest_node_edge_exact_radius() {
        let nodes = [(1usize, 100.0f32, 100.0f32)];
        // Exactly 14 px right of the node center
        let hit = nearest_grabbable_node(114.0, 100.0, &nodes).unwrap();
        assert_eq!(hit.band_index, 1);
        assert!((hit.distance_px - NODE_GRAB_RADIUS_PX).abs() < 1e-4);
    }

    #[test]
    fn nearest_node_picks_closest_of_two() {
        let nodes = [
            (0usize, 100.0f32, 100.0f32),
            (1usize, 106.0f32, 100.0f32), // closer to cursor
        ];
        let hit = nearest_grabbable_node(110.0, 100.0, &nodes).unwrap();
        assert_eq!(hit.band_index, 1);
        assert!(hit.distance_px < 4.1);
    }

    #[test]
    fn px_to_freq_roundtrip() {
        let width = 800.0f32;
        for f in [20.0, 100.0, 440.0, 1000.0, 5000.0, 20_000.0] {
            let px = freq_to_x(f) * width;
            let back = px_to_freq(px, width);
            assert!((back - f).abs() / f < 1e-4, "f={} back={}", f, back);
        }
    }

    #[test]
    fn px_to_freq_clamps() {
        let width = 800.0f32;
        assert_eq!(px_to_freq(-50.0, width), FREQ_MIN);
        assert_eq!(px_to_freq(2000.0, width), FREQ_MAX);
    }

    #[test]
    fn scroll_q_steps() {
        assert!((scroll_q_step(false) - 0.02).abs() < 1e-6);
        assert!((scroll_q_step(true) - 0.005).abs() < 1e-6);
    }
}
