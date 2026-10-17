use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_iced::IcedState;
use std::sync::Arc;

mod editor;
mod dsp;

pub const MAX_BANDS: usize = 24;
const PEAK_METER_DECAY_MS: f64 = 150.0;

pub struct Filter {
    params: Arc<FilterParams>,
    peak_meter_decay_weight: f32,
    peak_meter: Arc<AtomicF32>,
    bands: [dsp::BiquadFilter; MAX_BANDS],
    sample_rate: f32,
}

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

    #[id = "enable"]
    pub enable: BoolParam,
}

#[derive(Params)]
struct FilterParams {
    #[persist = "editor-state"]
    editor_state: Arc<IcedState>,

    #[id = "num-bands"]
    pub num_bands: IntParam,

    #[nested(array, group = "Bands")]
    pub band_params: [BandParams; MAX_BANDS],
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    Bell = 0,
    LowShelf = 1,
    HighShelf = 2,
    LowCut = 3,
    HighCut = 4,
}

impl FilterType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => Self::Bell,
            1 => Self::LowShelf,
            2 => Self::HighShelf,
            3 => Self::LowCut,
            4 => Self::HighCut,
            _ => Self::Bell,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bell => "Bell",
            Self::LowShelf => "Low Shelf",
            Self::HighShelf => "High Shelf",
            Self::LowCut => "Low Cut",
            Self::HighCut => "High Cut",
        }
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
                FloatRange::Linear { min: -24.0, max: 24.0 },
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
                IntRange::Linear { min: 0, max: 4 },
            ),

            enable: BoolParam::new("Enable", false),
        }
    }
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            params: Arc::new(FilterParams::default()),
            peak_meter_decay_weight: 1.0,
            peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            bands: core::array::from_fn(|_| dsp::BiquadFilter::new()),
            sample_rate: 44100.0,
        }
    }
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),
            num_bands: IntParam::new("Bands", 0, IntRange::Linear { min: 0, max: MAX_BANDS as i32 }),
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
        self.sample_rate = buffer_config.sample_rate as f32;
        for band in &mut self.bands {
            band.update_coefficients(1000.0, 0.0, 1.0, FilterType::Bell, self.sample_rate);
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

        for channel_samples in buffer.iter_samples() {
            let mut amplitude = 0.0f32;
            let num_samples = channel_samples.len();

            for sample in channel_samples {
                let mut s = *sample;

                for i in 0..num_active {
                    if self.params.band_params[i].enable.value() {
                        let freq = self.params.band_params[i].freq.smoothed.next();
                        let gain = self.params.band_params[i].gain.smoothed.next();
                        let q = self.params.band_params[i].q.smoothed.next();
                        let ftype = FilterType::from_i32(
                            self.params.band_params[i].filter_type.value() as i32,
                        );

                        self.bands[i].update_coefficients(freq, gain, q, ftype, self.sample_rate);
                        s = self.bands[i].process(s);
                    }
                }

                *sample = s;
                amplitude += s.abs();
            }

            if self.params.editor_state.is_open() {
                amplitude = (amplitude / num_samples as f32).abs();
                let current_peak = self.peak_meter.load(std::sync::atomic::Ordering::Relaxed);
                let new_peak = if amplitude > current_peak {
                    amplitude
                } else {
                    current_peak * self.peak_meter_decay_weight
                        + amplitude * (1.0 - self.peak_meter_decay_weight)
                };
                self.peak_meter.store(new_peak, std::sync::atomic::Ordering::Relaxed);
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
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Eq,
    ];
}

nih_export_clap!(Filter);
nih_export_vst3!(Filter);
