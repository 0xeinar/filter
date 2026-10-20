use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_iced::IcedState;
use std::sync::Arc;

pub mod dsp;
#[cfg(test)]
mod e2e;
mod editor;
#[cfg(test)]
mod test_support;
pub mod ui;

pub const MAX_BANDS: usize = 24;
const PEAK_METER_DECAY_MS: f64 = 150.0;

pub struct Filter {
    params: Arc<FilterParams>,
    peak_meter_decay_weight: f32,
    peak_meter: Arc<AtomicF32>,
    /// Per-band coefficients, one entry per cascade section — shared by every
    /// channel running that band. Only the first `band_sections[i]` are live.
    band_coeffs: [[dsp::BiquadCoeffs; dsp::biquad::MAX_SECTIONS]; MAX_BANDS],
    /// Live section count per band: 1 for a bell or shelf, 2 for a tilt, up to
    /// 8 for a 96 dB/oct cut.
    band_sections: [usize; MAX_BANDS],
    /// Per-channel filter state. Channel 0 uses `band_states[0]`,
    /// channel 1 uses `band_states[1]`, etc. Section state is per channel for
    /// the same reason the band state was: sharing it makes one channel
    /// continue the other's filter memory.
    band_states: [[[dsp::BiquadState; dsp::biquad::MAX_SECTIONS]; MAX_BANDS]; MAX_CHANNELS],
    sample_rate: f32,
    /// Shared with the GUI: the pre-EQ and post-EQ analyzer rings.
    spectrum_taps: Arc<dsp::spectrum::SpectrumTaps>,
    /// Audio-thread-only FFT analyzers (None until initialize). One per tap —
    /// each owns its own window buffer, so they must not share state.
    spectrum_analyzer_in: Option<dsp::spectrum::SpectrumAnalyzer>,
    spectrum_analyzer_out: Option<dsp::spectrum::SpectrumAnalyzer>,
    /// Audio-thread snapshots of band type/enable for the current block.
    band_types: [FilterType; MAX_BANDS],
    band_slopes: [u32; MAX_BANDS],
    band_enabled: [bool; MAX_BANDS],
    /// Last values baked into the current coefficients — lets process()
    /// skip recomputation when nothing moved.
    last_baked_freq: [f32; MAX_BANDS],
    last_baked_gain: [f32; MAX_BANDS],
    last_baked_q: [f32; MAX_BANDS],
    last_baked_type: [FilterType; MAX_BANDS],
    last_baked_slope: [u32; MAX_BANDS],
    /// Smoothed bypass crossfade: 0 = processed, 1 = fully dry.
    bypass_mix: Smoother<f32>,
    /// Current bypass smoother target (0 or 1), so the ramp is not restarted
    /// on every sub-block.
    bypass_target: f32,
    /// Current auto-gain compensation in dB (0 when auto-gain is off).
    comp_db: f32,
    /// A band moved since the compensation was last computed.
    comp_dirty: bool,
    /// Samples since the last compensation update (recompute throttle).
    comp_samples: u32,
    /// One level detector per band, reused across blocks so their envelopes
    /// keep their state. Only bands with dynamics enabled ever read these.
    detectors: [dsp::dynamics::Detector; MAX_BANDS],
    /// Snapshot of "this band is running dynamically" for the current block.
    band_dyn: [bool; MAX_BANDS],
}

/// Maximum audio channels the plugin processes (stereo layout + mono).
const MAX_CHANNELS: usize = 2;
/// Maximum host block size we'll bake coefficients for. nih-plug buffers are
/// host-driven; 128 is the standard upper bound. Larger host blocks are
/// processed in multiple iterations by iter_blocks.
const MAX_BLOCK: usize = 128;
/// Coefficients are re-baked at most every `COEFF_BLOCK` samples while a band
/// is being automated. 16 samples = 0.33 ms @48 kHz, i.e. below the ~1 ms
/// threshold where a staircase becomes audible on a fast gain sweep. Bands
/// that are not moving cost nothing: their smoother is not advanced and their
/// coefficients are not recomputed.
const COEFF_BLOCK: usize = 16;

/// How often auto-gain may recompute while a band is moving.
const AUTO_GAIN_INTERVAL_MS: f64 = 30.0;

/// Per-band parameters
#[derive(Params)]
pub struct BandParams {
    #[id = "freq"]
    pub freq: FloatParam,

    #[id = "gain"]
    pub gain: FloatParam,

    #[id = "q"]
    pub q: FloatParam,

    #[id = "type"]
    pub filter_type: IntParam,

    /// Slope of a cut filter, as a `Slope` index. Ignored by every other
    /// type — a bell has a Q, not a slope.
    #[id = "slope"]
    pub slope: IntParam,

    #[id = "enable"]
    pub enable: BoolParam,

    // ---- dynamics -------------------------------------------------------
    // A dynamic band's gain is its *maximum* change: the band is inert below
    // the threshold and reaches the full gain at threshold + range. Five
    // parameters per band rather than a nested struct because nih-plug's
    // `#[nested]` on an optional sub-struct would change every existing
    // band's serialized shape.
    /// Follow the level inside the band instead of applying the gain always.
    #[id = "dyn-enable"]
    pub dyn_enable: BoolParam,

    /// Level, in dBFS inside the band, at which the gain starts to apply.
    #[id = "dyn-threshold"]
    pub dyn_threshold: FloatParam,

    /// How many dB above the threshold it takes to reach the full gain.
    #[id = "dyn-range"]
    pub dyn_range: FloatParam,

    /// How fast the detector follows a rising level.
    #[id = "dyn-attack"]
    pub dyn_attack: FloatParam,

    /// How fast the detector falls back when the level drops.
    #[id = "dyn-release"]
    pub dyn_release: FloatParam,
}

#[derive(Params)]
pub struct FilterParams {
    #[persist = "editor-state"]
    editor_state: Arc<IcedState>,

    #[id = "num-bands"]
    pub num_bands: IntParam,

    /// Post-EQ output trim. Sits after every band, before the bypass mix.
    #[id = "output-gain"]
    pub output_gain: FloatParam,

    /// Compensate the average level change of the current EQ curve, so A/B
    /// against bypass is a fair loudness comparison.
    #[id = "auto-gain"]
    pub auto_gain: BoolParam,

    /// Hard bypass with a short crossfade; filter state keeps running so
    /// un-bypassing is seamless.
    #[id = "bypass"]
    pub bypass: BoolParam,

    /// Draw the post-EQ spectrum over the input one. Off is the "did I change
    /// anything?" question answered by eyes instead of memory, so it defaults
    /// on; the cost is one extra analyzer, and only while the editor is open.
    #[id = "spectrum-post"]
    pub spectrum_post: BoolParam,

    #[nested(array, group = "Bands")]
    pub band_params: [BandParams; MAX_BANDS],
}

/// Snapshot of the current band parameters as pure specs, for the GUI curve
/// and for auto-gain. Bands past `num_active` come back disabled.
pub fn band_specs(params: &FilterParams, num_active: usize) -> [dsp::curve::BandSpec; MAX_BANDS] {
    core::array::from_fn(|i| {
        if i >= num_active {
            return dsp::curve::BandSpec {
                freq: 1000.0,
                gain_db: 0.0,
                q: 1.0,
                filter_type: FilterType::Bell,
                slope_db_per_oct: Slope::DEFAULT.db_per_oct(),
                enabled: false,
            };
        }
        let bp = &params.band_params[i];
        dsp::curve::BandSpec {
            freq: bp.freq.value(),
            gain_db: bp.gain.value(),
            q: bp.q.value(),
            filter_type: FilterType::from_i32(bp.filter_type.value()),
            slope_db_per_oct: Slope::from_i32(bp.slope.value()).db_per_oct(),
            enabled: bp.enable.value(),
        }
    })
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Bell = 0,
    LowShelf = 1,
    HighShelf = 2,
    LowCut = 3,
    HighCut = 4,
    // Appended, not inserted: the discriminant is what presets and automation
    // store, so 0–4 must keep their meaning forever.
    Notch = 5,
    BandPass = 6,
    AllPass = 7,
    TiltShelf = 8,
}

impl FilterType {
    /// All filter types, in enum order — used by the UI pick list.
    pub const ALL: [FilterType; 9] = [
        Self::Bell,
        Self::LowShelf,
        Self::HighShelf,
        Self::LowCut,
        Self::HighCut,
        Self::Notch,
        Self::BandPass,
        Self::AllPass,
        Self::TiltShelf,
    ];

    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::LowShelf,
            2 => Self::HighShelf,
            3 => Self::LowCut,
            4 => Self::HighCut,
            5 => Self::Notch,
            6 => Self::BandPass,
            7 => Self::AllPass,
            8 => Self::TiltShelf,
            _ => Self::Bell,
        }
    }

    /// Cuts are the only type whose steepness is a slope rather than a Q.
    pub fn is_cut(self) -> bool {
        matches!(self, Self::LowCut | Self::HighCut)
    }

    /// Types where the band's gain means nothing (a cut passes 0 dB).
    pub fn has_gain(self) -> bool {
        !self.is_cut()
    }

    /// Types whose shape is set by a Q, as opposed to a corner (and, for cuts,
    /// a slope). Only used to pick a detector bandwidth: a shelf with a Q of 8
    /// would listen to a hairline instead of to its corner.
    pub fn is_bell_like(self) -> bool {
        !matches!(
            self,
            Self::LowShelf | Self::HighShelf | Self::LowCut | Self::HighCut | Self::TiltShelf
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bell => "Bell",
            Self::LowShelf => "Low Shelf",
            Self::HighShelf => "High Shelf",
            Self::LowCut => "Low Cut",
            Self::HighCut => "High Cut",
            Self::Notch => "Notch",
            Self::BandPass => "Band Pass",
            Self::AllPass => "All Pass",
            Self::TiltShelf => "Tilt Shelf",
        }
    }
}

/// Slope of a cut filter, in dB per octave.
///
/// Stored as a parameter index rather than a raw value so the pick list and
/// the automation lane agree on the same nine options.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slope {
    Db6 = 0,
    Db12 = 1,
    Db18 = 2,
    Db24 = 3,
    Db30 = 4,
    Db36 = 5,
    Db48 = 6,
    Db72 = 7,
    Db96 = 8,
}

impl Slope {
    pub const ALL: [Slope; 9] = [
        Self::Db6,
        Self::Db12,
        Self::Db18,
        Self::Db24,
        Self::Db30,
        Self::Db36,
        Self::Db48,
        Self::Db72,
        Self::Db96,
    ];

    /// Default slope for a new band: 12 dB/oct, one biquad.
    pub const DEFAULT: Slope = Self::Db12;

    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => Self::Db6,
            2 => Self::Db18,
            3 => Self::Db24,
            4 => Self::Db30,
            5 => Self::Db36,
            6 => Self::Db48,
            7 => Self::Db72,
            8 => Self::Db96,
            _ => Self::Db12,
        }
    }

    pub fn db_per_oct(self) -> u32 {
        crate::dsp::biquad::SLOPES_DB_PER_OCT[self as usize]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Db6 => "6 dB/oct",
            Self::Db12 => "12 dB/oct",
            Self::Db18 => "18 dB/oct",
            Self::Db24 => "24 dB/oct",
            Self::Db30 => "30 dB/oct",
            Self::Db36 => "36 dB/oct",
            Self::Db48 => "48 dB/oct",
            Self::Db72 => "72 dB/oct",
            Self::Db96 => "96 dB/oct",
        }
    }
}

impl std::fmt::Display for Slope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Display for FilterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Default for BandParams {
    fn default() -> Self {
        Self {
            freq: FloatParam::new(
                "Freq",
                1000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" Hz")
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0)),

            gain: FloatParam::new(
                "Gain",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            q: FloatParam::new(
                "Q",
                1.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 20.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            filter_type: IntParam::new(
                "Type",
                FilterType::Bell as i32,
                // Nine types now; 0–4 keep their old meaning so existing
                // automation still resolves to the same filter.
                IntRange::Linear { min: 0, max: 8 },
            ),

            slope: IntParam::new(
                "Slope",
                Slope::DEFAULT as i32,
                IntRange::Linear {
                    min: 0,
                    max: Slope::ALL.len() as i32 - 1,
                },
            ),

            // A band is audible the moment it exists. Bands are created from
            // the graph by clicking, or by raising the band count — either way
            // an inactive new band is just confusing.
            enable: BoolParam::new("Enable", true),

            dyn_enable: BoolParam::new("Dynamics", false),
            dyn_threshold: FloatParam::new(
                "Threshold",
                -30.0,
                FloatRange::Linear {
                    min: -60.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            dyn_range: FloatParam::new(
                "Range",
                12.0,
                FloatRange::Linear {
                    min: 1.0,
                    max: 30.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1))
            .with_step_size(0.5),
            dyn_attack: FloatParam::new(
                "Attack",
                10.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 200.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_unit(" ms")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),
            dyn_release: FloatParam::new(
                "Release",
                150.0,
                FloatRange::Skewed {
                    min: 5.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_unit(" ms")
            .with_value_to_string(formatters::v2s_f32_rounded(0)),
        }
    }
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            params: Arc::new(FilterParams::default()),
            peak_meter_decay_weight: 1.0,
            peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            band_coeffs: core::array::from_fn(|_| {
                [dsp::BiquadCoeffs::PASSTHROUGH; dsp::biquad::MAX_SECTIONS]
            }),
            band_sections: [0; MAX_BANDS],
            band_states: core::array::from_fn(|_| {
                core::array::from_fn(|_| core::array::from_fn(|_| dsp::BiquadState::default()))
            }),
            sample_rate: 44100.0,
            spectrum_taps: Arc::new(dsp::spectrum::SpectrumTaps::new()),
            spectrum_analyzer_in: None,
            spectrum_analyzer_out: None,
            band_types: [FilterType::Bell; MAX_BANDS],
            band_slopes: [12; MAX_BANDS],
            band_enabled: [false; MAX_BANDS],
            last_baked_freq: [1000.0; MAX_BANDS],
            last_baked_gain: [0.0; MAX_BANDS],
            last_baked_q: [1.0; MAX_BANDS],
            last_baked_type: [FilterType::Bell; MAX_BANDS],
            last_baked_slope: [0; MAX_BANDS],
            bypass_mix: Smoother::new(SmoothingStyle::Linear(20.0)),
            bypass_target: 0.0,
            comp_db: 0.0,
            comp_dirty: true,
            comp_samples: 0,
            detectors: core::array::from_fn(|_| dsp::dynamics::Detector::default()),
            band_dyn: [false; MAX_BANDS],
        }
    }
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),
            num_bands: IntParam::new(
                "Bands",
                0,
                IntRange::Linear {
                    min: 0,
                    max: MAX_BANDS as i32,
                },
            ),
            output_gain: FloatParam::new(
                "Output",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            auto_gain: BoolParam::new("Auto Gain", false),
            bypass: BoolParam::new("Bypass", false),

            spectrum_post: BoolParam::new("Output Spectrum", true),

            band_params: core::array::from_fn(|_| BandParams::default()),
        }
    }
}

impl Plugin for Filter {
    const NAME: &'static str = "Filter";
    const VENDOR: &'static str = "Einar";
    const URL: &'static str = "https://github.com/einaralex/filter";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.peak_meter.clone(),
            self.spectrum_taps.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.peak_meter_decay_weight = 0.25f64
            .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;
        self.sample_rate = buffer_config.sample_rate;
        for sections in &mut self.band_coeffs {
            sections[0].update(1000.0, 0.0, 1.0, FilterType::Bell, self.sample_rate);
        }
        self.band_sections = [1; MAX_BANDS];
        self.bypass_mix
            .reset(if self.params.bypass.value() { 1.0 } else { 0.0 });
        self.comp_dirty = true;
        self.spectrum_analyzer_in = Some(dsp::spectrum::SpectrumAnalyzer::new(self.sample_rate));
        self.spectrum_analyzer_out = Some(dsp::spectrum::SpectrumAnalyzer::new(self.sample_rate));
        for detector in &mut self.detectors {
            detector.set_sample_rate(self.sample_rate);
            detector.reset();
        }
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_active = self.params.num_bands.value() as usize;

        // Spectrum analysis runs on the INPUT signal, before any band touches
        // the buffer. Channel 0 only (the earlier code averaged L/R; for a
        // log-frequency display the difference is below the pixel grid).
        // Editor-open check: skip the FFT entirely when the GUI is closed.
        let editor_open = self.params.editor_state.is_open();
        if editor_open {
            if let (Some(analyzer), Some(first_channel)) = (
                self.spectrum_analyzer_in.as_mut(),
                buffer.as_slice().first(),
            ) {
                // Safety: nih-plug guarantees the buffer callback runs on the
                // audio thread with no allocator use in our code below.
                // as_slice borrows for this block only.
                let samples: &[f32] = first_channel;
                analyzer.push_block(&self.spectrum_taps.input, samples);
            }
        }

        // Scratch for block-smoothed parameter values (audio-thread only).
        // MAX here is a compile-time bound; only `block_len` entries are used.
        let mut freq_block = [0.0f32; MAX_BLOCK];
        let mut gain_block = [0.0f32; MAX_BLOCK];
        let mut q_block = [0.0f32; MAX_BLOCK];
        let mut out_gain_block = [0.0f32; MAX_BLOCK];
        let mut bypass_block = [0.0f32; MAX_BLOCK];

        // Auto-gain: track the curve's mean level so A/B against bypass is
        // fair. Recomputed at most every AUTO_GAIN_INTERVAL_MS while a band is
        // moving — each evaluation is a few dozen magnitude calculations, far
        // too expensive to run per sub-block.
        if self.params.auto_gain.value() {
            let interval =
                (self.sample_rate as f64 * AUTO_GAIN_INTERVAL_MS / 1000.0).max(1.0) as u32;
            if self.comp_dirty && self.comp_samples >= interval {
                let specs = band_specs(&self.params, num_active);
                self.comp_db = dsp::curve::auto_gain_compensation_db(&specs, num_active);
                self.comp_dirty = false;
                self.comp_samples = 0;
            }
        } else {
            self.comp_db = 0.0;
        }

        // Per-band coefficient-change snapshot. A band's coefficients are only
        // recomputed when its smoothed freq/gain/q actually moved (or its
        // type/enable changed), so a static EQ costs ~zero per block.
        let mut coeffs_changed = [false; MAX_BANDS];

        // Does anything in this block need a level detector? Scanning once per
        // host block keeps the whole linked-input pass out of the static path.
        let any_dynamic = (0..num_active).any(|i| {
            let bp = &self.params.band_params[i];
            bp.enable.value()
                && bp.dyn_enable.value()
                && FilterType::from_i32(bp.filter_type.value()).has_gain()
        });

        for (_block_offset, mut block) in buffer.iter_blocks(MAX_BLOCK) {
            // Sub-block loop: coefficients are re-baked every COEFF_BLOCK
            // samples instead of once per host block, so an automation glide is
            // quantized at 0.33 ms rather than 2.7 ms. A static EQ pays for
            // nothing here — see the `moving` check below.
            for sub_offset in (0..block.samples()).step_by(COEFF_BLOCK) {
                let block_len = (block.samples() - sub_offset).min(COEFF_BLOCK);
                let num_channels = block.channels();

                // --- 0. Linked detector input for this sub-block ---
                // The *louder* of the channels, taken while both are still dry.
                // One detector feeds every channel, so a gain change cannot
                // apply to L but not R — that would move the phantom centre.
                // This pass is skipped entirely when no band has dynamics.
                let mut det_in = [0.0f32; COEFF_BLOCK];
                if any_dynamic {
                    for (i, slot) in det_in[..block_len].iter_mut().enumerate() {
                        let mut peak = 0.0f32;
                        for channel in 0..num_channels {
                            if let Some(samples) = block.get(channel) {
                                peak = peak.max(samples[sub_offset + i].abs());
                            }
                        }
                        *slot = peak;
                    }
                }

                // --- 1. Advance smoothers once per sub-block, per band ---
                let mut any_band_active = false;
                // Multiple parallel fixed-size arrays are indexed by band here;
                // iterator adapters can't express that cleanly.
                #[allow(clippy::needless_range_loop)]
                for band_idx in 0..num_active {
                    let bp = &self.params.band_params[band_idx];
                    let enabled = bp.enable.value();
                    let ftype = FilterType::from_i32(bp.filter_type.value());

                    // Rising edge: the smoother idled while the band was off, so
                    // snap it to the current value instead of gliding in from a
                    // stale one.
                    if enabled && !self.band_enabled[band_idx] {
                        bp.freq.smoothed.reset(bp.freq.value());
                        bp.gain.smoothed.reset(bp.gain.value());
                        bp.q.smoothed.reset(bp.q.value());
                        // A band that was off (or had no detector running) must
                        // not resume with a stale envelope.
                        self.detectors[band_idx].reset();
                    }

                    // Dynamics only mean anything for a type that has a gain.
                    // A cut passes 0 dB at every level, so its "gain" is not a
                    // parameter the detector could modulate.
                    let dyn_on = enabled && ftype.has_gain() && bp.dyn_enable.value();
                    self.band_dyn[band_idx] = dyn_on;

                    self.band_types[band_idx] = ftype;
                    self.band_enabled[band_idx] = enabled;
                    let slope = Slope::from_i32(bp.slope.value()).db_per_oct();

                    if enabled {
                        any_band_active = true;
                        // A band only needs work when a smoother is still
                        // gliding or the target/type moved since the last bake.
                        // Static bands skip both the smoother fill and the trig.
                        // A dynamic band rebakes every sub-block by design: its
                        // gain follows the envelope, which moves even when no
                        // parameter does.
                        let moving = dyn_on
                            || bp.freq.smoothed.is_smoothing()
                            || bp.gain.smoothed.is_smoothing()
                            || bp.q.smoothed.is_smoothing()
                            || self.last_baked_freq[band_idx] != bp.freq.value()
                            || self.last_baked_gain[band_idx] != bp.gain.value()
                            || self.last_baked_q[band_idx] != bp.q.value()
                            || self.last_baked_type[band_idx] != ftype
                            || self.last_baked_slope[band_idx] != slope;

                        self.band_slopes[band_idx] = slope;

                        if !moving {
                            coeffs_changed[band_idx] = false;
                            continue;
                        }

                        bp.freq
                            .smoothed
                            .next_block(&mut freq_block[..block_len], block_len);
                        bp.gain
                            .smoothed
                            .next_block(&mut gain_block[..block_len], block_len);
                        bp.q.smoothed
                            .next_block(&mut q_block[..block_len], block_len);
                        coeffs_changed[band_idx] = true;
                    } else {
                        coeffs_changed[band_idx] = false;
                    }
                }

                // --- 1b. Output stage for this sub-block: one dB -> linear
                // conversion per 16 samples (0.33 ms) keeps `powf` out of the
                // per-sample path, and the bypass crossfade is advanced once
                // for both channels.
                self.params
                    .output_gain
                    .smoothed
                    .next_block(&mut out_gain_block[..block_len], block_len);
                let out_lin = util::db_to_gain(out_gain_block[block_len - 1] + self.comp_db);

                // Only re-target on an actual change: `set_target` restarts the
                // ramp, so calling it every sub-block would reset the glide.
                let bypass_target = if self.params.bypass.value() { 1.0 } else { 0.0 };
                if bypass_target != self.bypass_target {
                    self.bypass_mix.set_target(self.sample_rate, bypass_target);
                    self.bypass_target = bypass_target;
                }
                self.bypass_mix
                    .next_block(&mut bypass_block[..block_len], block_len);

                self.comp_samples = self.comp_samples.saturating_add(block_len as u32);

                // --- 2. Recompute coefficients where needed (per sub-block,
                // per band — shared by all channels). Baked from the
                // block-smoothed values so parameter glides stay sample-accurate
                // in bulk.
                #[allow(clippy::needless_range_loop)]
                for band_idx in 0..num_active {
                    if !self.band_enabled[band_idx] || !coeffs_changed[band_idx] {
                        continue;
                    }
                    let baked_freq = freq_block[block_len - 1];
                    let baked_q = q_block[block_len - 1];
                    let baked_type = self.band_types[band_idx];
                    let baked_slope = self.band_slopes[band_idx];
                    let static_gain = gain_block[block_len - 1];
                    // Dynamic bands bake the gain the *signal* asks for this
                    // sub-block, not the one the knob points at. The detector
                    // runs before the bake so the coefficients for these 16
                    // samples use the level those samples produced — a
                    // sub-block of extra latency, 0.33 ms at 48 kHz.
                    let baked_gain = if self.band_dyn[band_idx] {
                        let bp = &self.params.band_params[band_idx];
                        // Shelves and cuts are defined by a corner, not a
                        // resonance: a Q of 8 there would listen to a hairline.
                        let detect_q = if baked_type.is_bell_like() {
                            baked_q
                        } else {
                            dsp::dynamics::SHELF_DETECT_Q
                        };
                        let level = self.detectors[band_idx].update(
                            &det_in[..block_len],
                            baked_freq,
                            detect_q,
                            bp.dyn_attack.value(),
                            bp.dyn_release.value(),
                        );
                        dsp::dynamics::dynamic_gain_db(
                            static_gain,
                            level,
                            bp.dyn_threshold.value(),
                            bp.dyn_range.value(),
                        )
                    } else {
                        static_gain
                    };
                    self.band_sections[band_idx] = dsp::BiquadCoeffs::update_cascade(
                        &mut self.band_coeffs[band_idx],
                        baked_freq,
                        baked_gain,
                        baked_q,
                        baked_type,
                        baked_slope,
                        self.sample_rate,
                    );
                    self.last_baked_freq[band_idx] = baked_freq;
                    self.last_baked_gain[band_idx] = baked_gain;
                    self.last_baked_q[band_idx] = baked_q;
                    self.last_baked_type[band_idx] = baked_type;
                    self.last_baked_slope[band_idx] = baked_slope;
                    self.comp_dirty = true;
                }

                // --- 3. Process every channel with the SAME coefficients ---
                // Each channel owns its own filter state (z1/z2); sharing one
                // state between L and R made the right channel continue the left
                // channel's filter memory. The sub-block-smoothed parameter
                // values interpolate over the samples: sample-accurate glide
                // without per-sample coefficient math.
                for channel_idx in 0..num_channels {
                    let Some(full_channel) = block.get_mut(channel_idx) else {
                        continue;
                    };
                    // Defensive: host layouts beyond MAX_CHANNELS would panic on
                    // the state indexing below. The plugin declares mono/stereo
                    // only, so this should never trigger — but an audio-thread
                    // panic is far worse than a skipped channel.
                    if channel_idx >= MAX_CHANNELS {
                        continue;
                    }
                    let Some(channel_samples) =
                        full_channel.get_mut(sub_offset..sub_offset + block_len)
                    else {
                        continue;
                    };
                    let is_primary = channel_idx == 0;
                    let states = &mut self.band_states[channel_idx];

                    let meter_open = is_primary && self.params.editor_state.is_open();
                    for (sample_idx, sample) in channel_samples.iter_mut().enumerate() {
                        // Keep the dry signal for the bypass crossfade, and keep
                        // filtering while bypassed so un-bypassing is seamless.
                        let dry = *sample;
                        let mut s = dry;

                        if any_band_active {
                            // Each band is a cascade: one section for a bell or
                            // shelf, two for a tilt, up to eight for a 96 dB/oct
                            // cut. Slicing to the live count keeps the inner
                            // loops bounds-check-free after the first iteration.
                            for (band_idx, band_state) in
                                states.iter_mut().enumerate().take(num_active)
                            {
                                if !self.band_enabled[band_idx] {
                                    continue;
                                }
                                let sections = self.band_sections[band_idx];
                                let coeffs = &self.band_coeffs[band_idx][..sections];
                                for (coeff, state) in
                                    coeffs.iter().zip(band_state[..sections].iter_mut())
                                {
                                    s = state.process(coeff, s);
                                }
                            }
                        }

                        let out = dsp::output_stage(dry, s, out_lin, bypass_block[sample_idx]);
                        *sample = out;

                        // Peak meter: updated per sample (same decay rate as
                        // before) from the first channel. The old code averaged
                        // |L| and |R| per frame; |L| alone is within a hair for a
                        // peak-hold meter and keeps the hot path single-channel.
                        if meter_open {
                            let amplitude = out.abs();
                            let current_peak =
                                self.peak_meter.load(std::sync::atomic::Ordering::Relaxed);
                            let new_peak = if amplitude > current_peak {
                                amplitude
                            } else {
                                current_peak * self.peak_meter_decay_weight
                                    + amplitude * (1.0 - self.peak_meter_decay_weight)
                            };
                            self.peak_meter
                                .store(new_peak, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
        }

        // Post-EQ tap. The buffer now holds the fully processed signal — every
        // band, the output trim and the bypass crossfade — which is exactly
        // what the overlay must show. It runs once per host block rather than
        // per sub-block so the analyzer sees an evenly spaced sample stream;
        // feeding per sub-block would run the FFT up to eight times as often
        // for the same window.
        //
        // Deliberately still fed when the overlay is switched off: skipping it
        // would freeze the analyzer mid-window and splice pre-toggle samples
        // onto post-toggle ones, so re-enabling would flash a bogus frame. The
        // second analyzer measures 0.22% of a core (benches/hotpath_bench.rs),
        // which is not worth a visible artifact.
        if editor_open {
            if let (Some(analyzer), Some(first_channel)) = (
                self.spectrum_analyzer_out.as_mut(),
                buffer.as_slice().first(),
            ) {
                let samples: &[f32] = first_channel;
                analyzer.push_block(&self.spectrum_taps.output, samples);
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Filter {
    const CLAP_ID: &'static str = "com.einaralex.filter";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A surgical parametric EQ plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Equalizer,
    ];
}

impl Vst3Plugin for Filter {
    const VST3_CLASS_ID: [u8; 16] = *b"FilterEQEinar01\0";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Eq];
}

nih_export_clap!(Filter);
nih_export_vst3!(Filter);

#[cfg(test)]
mod param_tests {
    use super::*;

    /// The discriminant is what a preset and an automation lane store, so it
    /// must never move. New types get appended, never inserted.
    #[test]
    fn filter_type_discriminants_are_stable() {
        let expected = [
            (FilterType::Bell, 0, "Bell"),
            (FilterType::LowShelf, 1, "Low Shelf"),
            (FilterType::HighShelf, 2, "High Shelf"),
            (FilterType::LowCut, 3, "Low Cut"),
            (FilterType::HighCut, 4, "High Cut"),
            (FilterType::Notch, 5, "Notch"),
            (FilterType::BandPass, 6, "Band Pass"),
            (FilterType::AllPass, 7, "All Pass"),
            (FilterType::TiltShelf, 8, "Tilt Shelf"),
        ];
        for (filter_type, value, name) in expected {
            assert_eq!(filter_type as i32, value, "{name} moved");
            assert_eq!(FilterType::from_i32(value), filter_type);
            assert_eq!(filter_type.name(), name);
        }
        assert_eq!(FilterType::ALL.len(), expected.len());
    }

    #[test]
    fn every_slope_index_round_trips_and_matches_the_dsp_table() {
        for (index, slope) in Slope::ALL.iter().enumerate() {
            assert_eq!(*slope as i32, index as i32);
            assert_eq!(Slope::from_i32(index as i32), *slope);
            assert_eq!(
                slope.db_per_oct(),
                dsp::biquad::SLOPES_DB_PER_OCT[index],
                "{slope} disagrees with the DSP table"
            );
        }
        assert_eq!(
            Slope::DEFAULT.db_per_oct(),
            12,
            "cuts default to one biquad"
        );
    }

    #[test]
    fn only_cuts_are_steepness_controlled_and_gainless() {
        for filter_type in FilterType::ALL {
            assert_eq!(filter_type.is_cut(), !filter_type.has_gain());
            if filter_type.is_cut() {
                assert!(matches!(
                    filter_type,
                    FilterType::LowCut | FilterType::HighCut
                ));
            }
        }
    }
}
