//! Helpers shared by the crate's unit tests.
//!
//! Two things every GUI-side test needs: a `GuiContext` that goes nowhere, and
//! `FilterParams` with exact values in place. The latter exists because
//! `BandParams::default()` hardcodes 1 kHz / 0 dB / Q 1 and nih-plug's
//! parameter setters are crate-private, so tests have to build the params with
//! the values already set.

use std::sync::Arc;

use nih_plug::prelude::*;

use crate::{BandParams, FilterParams, FilterType, MAX_BANDS};

/// A `GuiContext` that accepts every call and does nothing — enough to drive
/// the GUI code paths with no host and no window.
pub struct NullGuiContext;

impl GuiContext for NullGuiContext {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Standalone
    }
    fn request_resize(&self) -> bool {
        false
    }
    unsafe fn raw_begin_set_parameter(&self, _param: ParamPtr) {}
    unsafe fn raw_set_parameter_normalized(&self, _param: ParamPtr, _normalized: f32) {}
    unsafe fn raw_end_set_parameter(&self, _param: ParamPtr) {}
    fn get_state(&self) -> PluginState {
        PluginState {
            version: String::new(),
            params: Default::default(),
            fields: Default::default(),
        }
    }
    fn set_state(&self, _state: PluginState) {}
}

/// A `GuiContext` that records the normalized values the GUI writes, so a
/// test can assert on what the editor asked the host to do.
#[derive(Default)]
pub struct RecordingGuiContext {
    writes: std::sync::Mutex<Vec<f32>>,
}

impl RecordingGuiContext {
    /// Normalized values written so far, in order.
    pub fn writes(&self) -> Vec<f32> {
        self.writes.lock().unwrap().clone()
    }
}

impl GuiContext for RecordingGuiContext {
    fn plugin_api(&self) -> PluginApi {
        PluginApi::Standalone
    }
    fn request_resize(&self) -> bool {
        false
    }
    unsafe fn raw_begin_set_parameter(&self, _param: ParamPtr) {}
    unsafe fn raw_set_parameter_normalized(&self, _param: ParamPtr, normalized: f32) {
        self.writes.lock().unwrap().push(normalized);
    }
    unsafe fn raw_end_set_parameter(&self, _param: ParamPtr) {}
    fn get_state(&self) -> PluginState {
        PluginState {
            version: String::new(),
            params: Default::default(),
            fields: Default::default(),
        }
    }
    fn set_state(&self, _state: PluginState) {}
}

/// A band with explicit values.
pub fn band(freq: f32, gain: f32, q: f32, ftype: FilterType, enabled: bool) -> BandParams {
    band_with_slope(freq, gain, q, ftype, crate::Slope::DEFAULT as i32, enabled)
}

/// A band with an explicit slope index (`Slope` as i32; 1 = 12 dB/oct).
pub fn band_with_slope(
    freq: f32,
    gain: f32,
    q: f32,
    ftype: FilterType,
    slope_index: i32,
    enabled: bool,
) -> BandParams {
    BandParams {
        freq: FloatParam::new(
            "Freq",
            freq,
            FloatRange::Skewed {
                min: 20.0,
                max: 20_000.0,
                factor: FloatRange::skew_factor(-2.5),
            },
        )
        .with_smoother(SmoothingStyle::Logarithmic(20.0)),
        gain: FloatParam::new(
            "Gain",
            gain,
            FloatRange::Linear {
                min: -24.0,
                max: 24.0,
            },
        )
        .with_smoother(SmoothingStyle::Linear(20.0)),
        q: FloatParam::new(
            "Q",
            q,
            FloatRange::Skewed {
                min: 0.1,
                max: 20.0,
                factor: FloatRange::skew_factor(-1.0),
            },
        )
        .with_smoother(SmoothingStyle::Logarithmic(20.0)),
        filter_type: IntParam::new("Type", ftype as i32, IntRange::Linear { min: 0, max: 8 }),
        slope: IntParam::new(
            "Slope",
            slope_index,
            IntRange::Linear {
                min: 0,
                max: crate::Slope::ALL.len() as i32 - 1,
            },
        ),
        enable: BoolParam::new("Enable", enabled),
        // Dynamics are off unless a test asks for them via `dynamic`.
        ..BandParams::default()
    }
}

/// Turn dynamics on for `band`, with explicit threshold/range/timing.
pub fn dynamic(
    band: BandParams,
    threshold_db: f32,
    range_db: f32,
    attack_ms: f32,
    release_ms: f32,
) -> BandParams {
    BandParams {
        dyn_enable: BoolParam::new("Dynamics", true),
        dyn_threshold: FloatParam::new(
            "Threshold",
            threshold_db,
            FloatRange::Linear {
                min: -60.0,
                max: 0.0,
            },
        ),
        dyn_range: FloatParam::new(
            "Range",
            range_db,
            FloatRange::Linear {
                min: -30.0,
                max: 30.0,
            },
        ),
        dyn_attack: FloatParam::new(
            "Attack",
            attack_ms,
            FloatRange::Skewed {
                min: 0.1,
                max: 200.0,
                factor: FloatRange::skew_factor(-1.5),
            },
        ),
        dyn_release: FloatParam::new(
            "Release",
            release_ms,
            FloatRange::Skewed {
                min: 5.0,
                max: 2000.0,
                factor: FloatRange::skew_factor(-1.5),
            },
        ),
        ..band
    }
}

/// Params holding `bands` (all enabled) with the band count already set to
/// their number.
pub fn test_params(bands: &[(f32, f32, f32, FilterType)]) -> Arc<FilterParams> {
    test_params_with_post_spectrum(bands, true)
}

/// As `test_params`, with the post-EQ spectrum overlay set to `post_spectrum`.
/// Used to prove the overlay layer really is gated on the parameter rather than
/// always drawn.
pub fn test_params_with_post_spectrum(
    bands: &[(f32, f32, f32, FilterType)],
    post_spectrum: bool,
) -> Arc<FilterParams> {
    let mut band_params: [BandParams; MAX_BANDS] = core::array::from_fn(|_| BandParams::default());
    for (i, &(freq, gain, q, ftype)) in bands.iter().enumerate() {
        band_params[i] = band(freq, gain, q, ftype, true);
    }
    Arc::new(FilterParams {
        num_bands: IntParam::new(
            "Bands",
            bands.len() as i32,
            IntRange::Linear {
                min: 0,
                max: MAX_BANDS as i32,
            },
        ),
        band_params,
        spectrum_post: BoolParam::new("Output Spectrum", post_spectrum),
        ..FilterParams::default()
    })
}

/// Params with `bands`, every one of them dynamic. Used by the editor tests,
/// which need a band whose control row is populated.
pub fn test_params_with_dynamics(bands: &[(f32, f32, f32, FilterType)]) -> Arc<FilterParams> {
    let mut band_params: [BandParams; MAX_BANDS] = core::array::from_fn(|_| BandParams::default());
    for (i, &(freq, gain, q, ftype)) in bands.iter().enumerate() {
        // The band's gain becomes its dynamic range and the static gain goes to
        // 0, which is what enabling dynamics in the editor does ("make
        // dynamic"). A fixture that left the gain in place would describe a
        // band nobody creates: static -12 plus a -12 dynamic range.
        band_params[i] = dynamic(band(freq, 0.0, q, ftype, true), -30.0, gain, 10.0, 150.0);
    }
    Arc::new(FilterParams {
        num_bands: IntParam::new(
            "Bands",
            bands.len() as i32,
            IntRange::Linear {
                min: 0,
                max: MAX_BANDS as i32,
            },
        ),
        band_params,
        ..FilterParams::default()
    })
}

/// Params whose bands also carry a cut slope — `test_params` leaves every band
/// at the default 12 dB/oct.
pub fn test_params_with_slopes(
    bands: &[(f32, f32, f32, FilterType, crate::Slope)],
) -> Arc<FilterParams> {
    let mut band_params: [BandParams; MAX_BANDS] = core::array::from_fn(|_| BandParams::default());
    for (i, &(freq, gain, q, ftype, slope)) in bands.iter().enumerate() {
        band_params[i] = band_with_slope(freq, gain, q, ftype, slope as i32, true);
    }
    Arc::new(FilterParams {
        num_bands: IntParam::new(
            "Bands",
            bands.len() as i32,
            IntRange::Linear {
                min: 0,
                max: MAX_BANDS as i32,
            },
        ),
        band_params,
        ..FilterParams::default()
    })
}
