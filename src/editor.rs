use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext, ParamSetter};
use nih_plug_iced::widgets as nih_widgets;
use nih_plug_iced::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::dsp::dynamics::BandMeters;
use crate::dsp::spectrum::SpectrumTaps;
use crate::patch::PatchState;
use crate::ui;
use crate::ui::selection;
use crate::ChannelMode;
use crate::FilterParams;
use crate::FilterType;
use crate::Slope;
use crate::MAX_BANDS;

pub use colors::band_color;

/// Window size. The layout is fixed-height by construction (see the region
/// constants below), so this is a hard budget, not a starting suggestion —
/// `chrome_fits_in_the_window` asserts it.
pub const EDITOR_WIDTH: u16 = 900;
pub const EDITOR_HEIGHT: u16 = 524;

// Fixed region heights. Every region states its height explicitly so the
// whole column is deterministic and cannot silently overflow the window as
// the band count grows — which is exactly what the old per-band rows did
// (each row ~24 px, so past five bands the list ran off the bottom).
const HEADER_H: u16 = 28;
const STRIP_H: u16 = 24;
const PANEL_H: u16 = 116;
const FOOTER_H: u16 = 14;
/// Vertical gap between the column's children.
const GAP: u16 = 6;
/// Band strip chip.
const CHIP_W: u16 = 24;
const CHIP_H: u16 = 20;

/// Height of everything except the graph.
pub fn chrome_height() -> u16 {
    HEADER_H + ui::graph::GRAPH_HEIGHT as u16 + STRIP_H + PANEL_H + FOOTER_H + GAP * 4
}

pub fn default_state() -> Arc<IcedState> {
    // The layout is fixed-height, so this is a real invariant rather than a
    // style preference: if the regions outgrow the window, the bottom of the
    // editor is clipped.
    debug_assert!(
        chrome_height() <= EDITOR_HEIGHT,
        "editor chrome is {} px tall but the window is {EDITOR_HEIGHT} px",
        chrome_height()
    );
    IcedState::from_size(EDITOR_WIDTH as u32, EDITOR_HEIGHT as u32)
}

pub fn create(
    params: Arc<FilterParams>,
    peak_meter: Arc<AtomicF32>,
    spectrum_taps: Arc<SpectrumTaps>,
    dyn_meters: Arc<BandMeters>,
    solo: Arc<AtomicUsize>,
    patch_state: Arc<Mutex<PatchState>>,
    editor_state: Arc<IcedState>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<FilterEditor>(
        editor_state,
        (
            params,
            peak_meter,
            spectrum_taps,
            dyn_meters,
            solo,
            patch_state,
        ),
    )
}

mod colors {
    use nih_plug_iced::Color;
    pub const BG: Color = Color {
        r: 0.102,
        g: 0.102,
        b: 0.118,
        a: 1.0,
    };
    pub const TEXT: Color = Color {
        r: 0.8,
        g: 0.8,
        b: 0.8,
        a: 1.0,
    };
    pub const TEXT_DIM: Color = Color {
        r: 0.5,
        g: 0.5,
        b: 0.5,
        a: 1.0,
    };
    pub const TEXT_FAINT: Color = Color {
        r: 0.32,
        g: 0.32,
        b: 0.32,
        a: 1.0,
    };

    pub fn band_color(i: usize) -> Color {
        const PALETTE: [[f32; 3]; 12] = [
            [0.3, 0.6, 1.0],
            [1.0, 0.4, 0.3],
            [0.3, 0.9, 0.4],
            [1.0, 0.8, 0.2],
            [0.8, 0.3, 0.9],
            [1.0, 0.5, 0.0],
            [0.0, 0.8, 0.8],
            [1.0, 0.3, 0.6],
            [0.5, 0.8, 0.2],
            [0.6, 0.4, 1.0],
            [0.9, 0.6, 0.3],
            [0.4, 1.0, 0.6],
        ];
        let c = PALETTE[i % 12];
        Color {
            r: c[0],
            g: c[1],
            b: c[2],
            a: 1.0,
        }
    }
}

/// A numbered band chip in the strip under the graph.
///
/// Styled directly rather than themed: the chip's colour *is* the band's
/// identity, so it has to come from the band palette.
struct ChipStyle {
    color: Color,
    selected: bool,
    enabled: bool,
}

/// The A/B slot buttons: filled when live, outlined when not.
struct SlotStyle {
    active: bool,
}

impl button::StyleSheet for SlotStyle {
    fn active(&self) -> button::Style {
        let (bg_alpha, border_alpha, text_alpha) = if self.active {
            (0.85, 1.0, 1.0)
        } else {
            (0.10, 0.25, 0.75)
        };
        button::Style {
            shadow_offset: Vector::default(),
            background: Some(Background::Color(Color {
                r: 0.55,
                g: 0.75,
                b: 1.0,
                a: bg_alpha,
            })),
            border_radius: 3.0,
            border_width: 1.0,
            border_color: Color {
                r: 0.55,
                g: 0.75,
                b: 1.0,
                a: border_alpha,
            },
            // A filled slot needs a dark digit to stay readable.
            text_color: if self.active {
                Color {
                    r: 0.06,
                    g: 0.06,
                    b: 0.08,
                    a: 1.0,
                }
            } else {
                Color {
                    r: 0.75,
                    g: 0.85,
                    b: 1.0,
                    a: text_alpha,
                }
            },
        }
    }

    fn hovered(&self) -> button::Style {
        let active = self.active();
        button::Style {
            border_width: 1.5,
            ..active
        }
    }
}

impl button::StyleSheet for ChipStyle {
    fn active(&self) -> button::Style {
        let (bg_alpha, border_alpha, text_alpha) = if self.selected {
            (0.9, 1.0, 0.0)
        } else if !self.enabled {
            (0.12, 0.18, 0.30)
        } else {
            (0.28, 0.45, 0.95)
        };
        button::Style {
            shadow_offset: Vector::default(),
            background: Some(Background::Color(Color {
                a: bg_alpha,
                ..self.color
            })),
            border_radius: 3.0,
            border_width: 1.0,
            border_color: Color {
                a: border_alpha,
                ..self.color
            },
            // A selected chip is filled with the band colour, so its digit
            // has to go dark to stay readable.
            text_color: if self.selected {
                Color {
                    r: 0.06,
                    g: 0.06,
                    b: 0.08,
                    a: 1.0,
                }
            } else {
                Color {
                    a: text_alpha,
                    ..self.color
                }
            },
        }
    }

    fn hovered(&self) -> button::Style {
        let active = self.active();
        button::Style {
            border_width: 1.5,
            ..active
        }
    }
}

struct FilterEditor {
    params: Arc<FilterParams>,
    context: Arc<dyn GuiContext>,
    peak_meter: Arc<AtomicF32>,
    /// Live dynamic offset per band, so the DYN row can show how far the band
    /// has moved from its static gain right now.
    dyn_meters: Arc<BandMeters>,
    /// Which band is soloed (`NO_SOLO` for none). Shared with the audio thread
    /// and the canvas, which draws the badge and owns the shortcut.
    solo: Arc<AtomicUsize>,
    /// Undo/redo history and the A/B pair. Shared with the canvas, which owns
    /// the keyboard and therefore the shortcut.
    patch_state: Arc<Mutex<PatchState>>,
    slot_a_state: std::cell::UnsafeCell<button::State>,
    slot_b_state: std::cell::UnsafeCell<button::State>,
    slot_copy_state: std::cell::UnsafeCell<button::State>,
    /// Button cache for the oversampling cycle control.
    os_state: std::cell::UnsafeCell<button::State>,
    graph: ui::Graph,
    /// Shared with the graph: one of the editor's few pieces of state that
    /// is not a host parameter.
    selection: Arc<AtomicUsize>,
    num_bands_slider_state: nih_widgets::param_slider::State,
    out_gain_slider_state: nih_widgets::param_slider::State,
    // Wrap in UnsafeCell to allow disjoint mutable access by index
    freq_slider_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    gain_slider_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    q_slider_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    // Dynamics: four sliders per band, one row of state each.
    dyn_threshold_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    dyn_range_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    dyn_attack_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    dyn_release_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    dyn_knee_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    dyn_sc_freq_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    type_pick_states: [std::cell::UnsafeCell<pick_list::State<FilterType>>; MAX_BANDS],
    slope_pick_states: [std::cell::UnsafeCell<pick_list::State<Slope>>; MAX_BANDS],
    channel_pick_states: [std::cell::UnsafeCell<pick_list::State<ChannelMode>>; MAX_BANDS],
    chip_states: [std::cell::UnsafeCell<button::State>; MAX_BANDS],
}

// Safety: We only ever access one index at a time, and each index is unique
unsafe impl Send for FilterEditor {}
unsafe impl Sync for FilterEditor {}

impl FilterEditor {
    /// Get a mutable reference to a slider state by index.
    /// Safety: Caller must ensure no two calls with the same index are active simultaneously.
    #[allow(clippy::mut_from_ref)] // false positive: UnsafeCell::get() is the intended API
    unsafe fn slider_state(
        states: &[std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
        i: usize,
    ) -> &mut nih_widgets::param_slider::State {
        &mut *states[i].get()
    }

    /// Get a mutable reference to a pick list state by index.
    /// Safety: Caller must ensure no two calls with the same index are active simultaneously.
    #[allow(clippy::mut_from_ref)] // false positive: UnsafeCell::get() is the intended API
    unsafe fn pick_state(
        states: &[std::cell::UnsafeCell<pick_list::State<FilterType>>; MAX_BANDS],
        i: usize,
    ) -> &mut pick_list::State<FilterType> {
        &mut *states[i].get()
    }

    /// Get a mutable reference to a slope pick list state by index.
    /// Safety: Caller must ensure no two calls with the same index are active simultaneously.
    #[allow(clippy::mut_from_ref)] // false positive: UnsafeCell::get() is the intended API
    unsafe fn slope_pick_state(
        states: &[std::cell::UnsafeCell<pick_list::State<Slope>>; MAX_BANDS],
        i: usize,
    ) -> &mut pick_list::State<Slope> {
        &mut *states[i].get()
    }

    /// Get a mutable reference to a channel-mode pick list state by index.
    /// Safety: Caller must ensure no two calls with the same index are active simultaneously.
    #[allow(clippy::mut_from_ref)] // false positive: UnsafeCell::get() is the intended API
    unsafe fn channel_pick_state(
        states: &[std::cell::UnsafeCell<pick_list::State<ChannelMode>>; MAX_BANDS],
        i: usize,
    ) -> &mut pick_list::State<ChannelMode> {
        &mut *states[i].get()
    }

    /// Get a mutable reference to a chip state by index.
    /// Safety: Caller must ensure no two calls with the same index are active simultaneously.
    #[allow(clippy::mut_from_ref)] // false positive: UnsafeCell::get() is the intended API
    unsafe fn chip_state(
        states: &[std::cell::UnsafeCell<button::State>; MAX_BANDS],
        i: usize,
    ) -> &mut button::State {
        &mut *states[i].get()
    }

    /// The selection clamped to the bands that actually exist.
    fn selected(&self, num_bands: usize) -> usize {
        selection::clamp_to(self.selection.load(Ordering::Relaxed), num_bands)
    }

    fn select(&self, band: usize) {
        self.selection.store(band, Ordering::Relaxed);
    }

    /// Select `index` only if that band exists. The band count can change
    /// behind the GUI's back (the Bands slider, the host, a preset), so a
    /// queued `SelectBand` can name a slot that is no longer in use.
    fn select_if_live(&self, index: usize) {
        let num_bands = self.params.num_bands.value() as usize;
        if index < num_bands {
            self.select(index);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    ParamUpdate(nih_widgets::ParamMessage),
    /// Band `index` changed filter type.
    TypeChanged(usize, FilterType),
    /// Band `index` changed cut slope.
    SlopeChanged(usize, Slope),
    /// Band `index` changed its channel basis (stereo / mid / side).
    ChannelChanged(usize, ChannelMode),
    /// Band `index` enabled/disabled.
    EnableChanged(usize, bool),
    /// Band `index` selected from the band strip.
    SelectBand(usize),
    BypassToggled(bool),
    AutoGainToggled(bool),
    /// Show/hide the post-EQ spectrum overlay.
    PostSpectrumToggled(bool),
    /// Advance the oversampling factor: off -> 2x -> 4x -> off.
    OversamplingCycled,
    /// Band `index` enabled/disabled its dynamics.
    DynamicsToggled(usize, bool),
    /// Band `index` enabled/disabled its sidechain filter.
    SidechainToggled(usize, bool),
    /// Solo band `index`, or clear solo when the flag is false.
    SoloToggled(usize, bool),
    /// Step back one edit.
    Undo,
    /// Step forward again.
    Redo,
    /// Switch to A/B slot `index`.
    SwitchSlot(usize),
    /// Copy the live state into the other slot.
    CopySlot,
}

impl FilterEditor {
    /// Set a band's filter type as a single automation gesture.
    fn set_band_type(&mut self, index: usize, filter_type: FilterType) -> Command<Message> {
        let param = &self.params.band_params[index].filter_type;
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter(param, filter_type as i32);
        setter.end_set_parameter(param);
        Command::none()
    }

    /// Set a band's channel basis as a single automation gesture.
    fn set_band_channel(&mut self, index: usize, mode: ChannelMode) -> Command<Message> {
        let param = &self.params.band_params[index].channel_mode;
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter(param, mode as i32);
        setter.end_set_parameter(param);
        Command::none()
    }

    /// Set a cut band's slope as a single automation gesture.
    fn set_band_slope(&mut self, index: usize, slope: Slope) -> Command<Message> {
        let param = &self.params.band_params[index].slope;
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter(param, slope as i32);
        setter.end_set_parameter(param);
        Command::none()
    }

    /// Lock the shared history without a poisoned-lock panic bricking the GUI.
    fn patch_state_mut(&self) -> std::sync::MutexGuard<'_, PatchState> {
        self.patch_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Enable/disable a band's dynamics as a single automation gesture, so the
    /// host records one undo step rather than one per parameter touched.
    fn set_band_dynamics(&self, index: usize, on: bool) -> Command<Message> {
        let bp = &self.params.band_params[index];
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(&bp.dyn_enable);
        setter.set_parameter(&bp.dyn_enable, on);
        setter.end_set_parameter(&bp.dyn_enable);

        // Turning dynamics on for a band that has a gain but no range yet moves
        // the gain into the range, which is Pro-Q's "make dynamic" (its
        // Alt-drag trades one for the other). Without it the switch would look
        // broken: the band would go on being static because the range is 0.
        if on && bp.dyn_range.value().abs() < crate::dsp::dynamics::MIN_RANGE_DB {
            let gain = bp.gain.value();
            if gain.abs() >= crate::dsp::dynamics::MIN_RANGE_DB {
                setter.begin_set_parameter(&bp.dyn_range);
                setter.set_parameter(&bp.dyn_range, gain);
                setter.end_set_parameter(&bp.dyn_range);
                setter.begin_set_parameter(&bp.gain);
                setter.set_parameter(&bp.gain, 0.0);
                setter.end_set_parameter(&bp.gain);
            }
        }
        Command::none()
    }

    /// Enable/disable a band's sidechain filter as one gesture.
    fn set_band_sidechain(&self, index: usize, on: bool) -> Command<Message> {
        let param = &self.params.band_params[index].dyn_sc_enable;
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter(param, on);
        setter.end_set_parameter(param);
        Command::none()
    }

    /// Enable/disable a band as a single automation gesture.
    fn set_band_enable(&mut self, index: usize, enabled: bool) -> Command<Message> {
        let param = &self.params.band_params[index].enable;
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter(param, enabled);
        setter.end_set_parameter(param);
        Command::none()
    }
}

impl IcedEditor for FilterEditor {
    type Executor = executor::Default;
    type Message = Message;
    type InitializationFlags = (
        Arc<FilterParams>,
        Arc<AtomicF32>,
        Arc<SpectrumTaps>,
        Arc<BandMeters>,
        Arc<AtomicUsize>,
        Arc<Mutex<PatchState>>,
    );

    fn new(
        (params, peak_meter, spectrum_taps, dyn_meters, solo, patch_state): Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let selection = Arc::new(AtomicUsize::new(selection::NONE));
        let graph = ui::Graph::new(
            params.clone(),
            context.clone(),
            spectrum_taps,
            dyn_meters.clone(),
            solo.clone(),
            patch_state.clone(),
            selection.clone(),
        );
        let editor = FilterEditor {
            params,
            context,
            peak_meter,
            dyn_meters,
            solo,
            patch_state,
            slot_a_state: std::cell::UnsafeCell::new(button::State::new()),
            slot_b_state: std::cell::UnsafeCell::new(button::State::new()),
            slot_copy_state: std::cell::UnsafeCell::new(button::State::new()),
            os_state: std::cell::UnsafeCell::new(button::State::new()),
            graph,
            selection,
            num_bands_slider_state: Default::default(),
            out_gain_slider_state: Default::default(),
            freq_slider_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            gain_slider_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            q_slider_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            dyn_threshold_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            dyn_range_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            dyn_attack_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            dyn_release_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            dyn_knee_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            dyn_sc_freq_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            type_pick_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(pick_list::State::default())
            }),
            slope_pick_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(pick_list::State::default())
            }),
            channel_pick_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(pick_list::State::default())
            }),
            chip_states: core::array::from_fn(|_| std::cell::UnsafeCell::new(button::State::new())),
        };
        (editor, Command::none())
    }

    fn context(&self) -> &dyn GuiContext {
        self.context.as_ref()
    }

    fn update(
        &mut self,
        _window: &mut WindowQueue,
        message: Self::Message,
    ) -> Command<Self::Message> {
        match message {
            Message::ParamUpdate(message) => self.handle_param_message(message),
            Message::TypeChanged(index, filter_type) => {
                let _ = self.set_band_type(index, filter_type);
            }
            Message::SlopeChanged(index, slope) => {
                let _ = self.set_band_slope(index, slope);
            }
            Message::ChannelChanged(index, mode) => {
                let _ = self.set_band_channel(index, mode);
            }
            Message::EnableChanged(index, enabled) => {
                let _ = self.set_band_enable(index, enabled);
            }
            Message::SelectBand(index) => {
                // Selecting a dead index (the band count can change from the
                // host or from another editor) must not leave the panel
                // pointing at nothing.
                self.select_if_live(index);
            }
            Message::BypassToggled(on) => {
                let param = &self.params.bypass;
                let setter = ParamSetter::new(self.context.as_ref());
                setter.set_parameter(param, on);
            }
            Message::AutoGainToggled(on) => {
                let param = &self.params.auto_gain;
                let setter = ParamSetter::new(self.context.as_ref());
                setter.set_parameter(param, on);
            }
            Message::DynamicsToggled(index, on) => {
                let _ = self.set_band_dynamics(index, on);
            }
            Message::SoloToggled(index, on) => {
                // One solo at a time, and `NO_SOLO` rather than 0 for "none":
                // band 0 is a real band.
                let next = if on { index } else { crate::NO_SOLO };
                self.solo.store(next, Ordering::Relaxed);
            }
            Message::Undo => {
                self.patch_state_mut()
                    .undo(&self.params, self.context.as_ref());
            }
            Message::Redo => {
                self.patch_state_mut()
                    .redo(&self.params, self.context.as_ref());
            }
            Message::SwitchSlot(index) => {
                // Bound the lock to a statement: holding a guard across the
                // `if` body and then locking again would deadlock the GUI.
                let active = self.patch_state_mut().ab.active();
                // Clicking the slot you are already in does nothing: switching
                // to yourself would just rewrite the same values.
                if active != index.min(1) {
                    self.patch_state_mut()
                        .toggle_slot(&self.params, self.context.as_ref());
                }
            }
            Message::CopySlot => {
                self.patch_state_mut().copy_slot(&self.params);
            }
            Message::PostSpectrumToggled(on) => {
                let param = &self.params.spectrum_post;
                let setter = ParamSetter::new(self.context.as_ref());
                setter.set_parameter(param, on);
            }
            Message::SidechainToggled(index, on) => {
                let _ = self.set_band_sidechain(index, on);
            }
            Message::OversamplingCycled => {
                let param = &self.params.oversampling;
                let next = (param.value() + 1) % 3;
                let setter = ParamSetter::new(self.context.as_ref());
                setter.set_parameter(param, next);
            }
        }
        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        use nih_plug_iced::{Alignment, Column, Length, Row, Text};

        // Undo granularity is decided here: nih-plug redraws every frame, so
        // watching the parameters once per frame catches edits from every path
        // — canvas drags, panel sliders, host automation, a second editor — and
        // a drag settles into a single step once it stops moving.
        // Observe for undo, and keep the persisted A/B pair in step. The sync
        // is what makes an edit to the active slot survive a reload; the clone
        // into the persist field only happens when the slot actually changed,
        // not once per frame.
        {
            let mut state = self.patch_state_mut();
            let live = crate::patch::Patch::capture(&self.params);
            state.observe_patch(&live, Instant::now());
            let changed = state.sync(&live);
            if changed {
                let snapshot = state.snapshot();
                drop(state);
                if let Ok(mut blob) = self.params.ab_state.lock() {
                    *blob = Some(snapshot);
                }
            }
        }

        let num_bands = self.params.num_bands.value() as usize;
        let selected = self.selected(num_bands);

        let slot_active = self.patch_state_mut().ab.active();
        let os_factor = match self.params.oversampling.value() {
            1 => 2,
            2 => 4,
            _ => 1,
        };
        let os_label = if os_factor == 1 {
            "OS off".to_string()
        } else {
            format!("OS {os_factor}x")
        };
        let header = Row::new()
            .height(Length::Units(HEADER_H))
            .spacing(10)
            .align_items(Alignment::Center)
            .push(
                Text::new("Filter")
                    .font(assets::NOTO_SANS_LIGHT)
                    .size(18)
                    .color(colors::TEXT),
            )
            .push(Text::new("Bands").size(11).color(colors::TEXT_DIM))
            .push(
                nih_widgets::ParamSlider::new(
                    &mut self.num_bands_slider_state,
                    &self.params.num_bands,
                )
                .width(Length::Units(120))
                .height(Length::Units(22))
                .map(Message::ParamUpdate),
            )
            .push(Text::new("Out").size(11).color(colors::TEXT_DIM))
            .push(
                nih_widgets::ParamSlider::new(
                    &mut self.out_gain_slider_state,
                    &self.params.output_gain,
                )
                .width(Length::Units(120))
                .height(Length::Units(22))
                .map(Message::ParamUpdate),
            )
            .push({
                // A/B slots. Clicking the inactive letter switches to it; the
                // arrow copies the live state into the other slot without
                // switching, which is how you make B from A and then edit it.
                let copy_label = if slot_active == 0 { "A>B" } else { "B>A" };
                // Safety: view() is called once per frame and each state is
                // touched once here.
                let slot_a = button::Button::new(
                    unsafe { &mut *self.slot_a_state.get() },
                    Text::new("A").size(11),
                )
                .width(Length::Units(22))
                .height(Length::Units(20))
                .padding(0)
                .style(SlotStyle {
                    active: slot_active == 0,
                })
                .on_press(Message::SwitchSlot(0));
                let slot_b = button::Button::new(
                    unsafe { &mut *self.slot_b_state.get() },
                    Text::new("B").size(11),
                )
                .width(Length::Units(22))
                .height(Length::Units(20))
                .padding(0)
                .style(SlotStyle {
                    active: slot_active == 1,
                })
                .on_press(Message::SwitchSlot(1));
                let slot_copy = button::Button::new(
                    unsafe { &mut *self.slot_copy_state.get() },
                    Text::new(copy_label).size(10),
                )
                .width(Length::Units(34))
                .height(Length::Units(20))
                .padding(0)
                .style(SlotStyle { active: false })
                .on_press(Message::CopySlot);
                Row::new()
                    .spacing(3)
                    .align_items(Alignment::Center)
                    .push(slot_a)
                    .push(slot_b)
                    .push(slot_copy)
            })
            // Oversampling is a three-way choice, not a toggle, so it is a
            // button that cycles and shows where it is. A click is a host
            // gesture, so it also lands in the undo history.
            .push(
                button::Button::new(
                    unsafe { &mut *self.os_state.get() },
                    Text::new(os_label).size(10),
                )
                .width(Length::Units(34))
                .height(Length::Units(20))
                .padding(0)
                .style(SlotStyle {
                    active: os_factor > 1,
                })
                .on_press(Message::OversamplingCycled),
            )
            .push(Text::new("Post").size(11).color(colors::TEXT_DIM))
            .push(
                Toggler::new(
                    self.params.spectrum_post.value(),
                    String::new(),
                    Message::PostSpectrumToggled,
                )
                .size(14),
            )
            .push(Text::new("Auto gain").size(11).color(colors::TEXT_DIM))
            .push(
                Toggler::new(
                    self.params.auto_gain.value(),
                    String::new(),
                    Message::AutoGainToggled,
                )
                .size(14),
            )
            .push(Text::new("Bypass").size(11).color(colors::TEXT_DIM))
            .push(
                Toggler::new(
                    self.params.bypass.value(),
                    String::new(),
                    Message::BypassToggled,
                )
                .size(14),
            );

        // Band strip: one chip per band. Fixed height, so 24 bands cost the
        // same vertical space as one — the thing the old row list got wrong.
        let mut strip = Row::new()
            .height(Length::Units(STRIP_H))
            .spacing(3)
            .align_items(Alignment::Center);
        if num_bands == 0 {
            strip = strip.push(
                Text::new("no bands — click the graph to add one")
                    .size(11)
                    .color(colors::TEXT_FAINT),
            );
        }
        for i in 0..num_bands {
            let bp = &self.params.band_params[i];
            let enabled = bp.enable.value();
            let style = ChipStyle {
                color: colors::band_color(i),
                selected: selected == i,
                enabled,
            };
            let text_color = if style.selected {
                Color {
                    r: 0.06,
                    g: 0.06,
                    b: 0.08,
                    a: 1.0,
                }
            } else {
                Color {
                    a: if enabled { 0.95 } else { 0.30 },
                    ..style.color
                }
            };
            // Safety: each index i is visited once per frame
            let state = unsafe { Self::chip_state(&self.chip_states, i) };
            strip = strip.push(
                Button::new(
                    state,
                    Text::new(format!("{}", i + 1)).size(11).color(text_color),
                )
                .width(Length::Units(CHIP_W))
                .height(Length::Units(CHIP_H))
                .padding(0)
                .style(style)
                .on_press(Message::SelectBand(i)),
            );
        }

        // Selected-band panel. This replaces the per-band rows: whatever the
        // band count, the editor shows exactly one band's controls, so the
        // layout never grows past the window.
        let panel: Element<'_, Message> = if selection::is_none(selected) {
            Column::new()
                .height(Length::Units(PANEL_H))
                .push(
                    Text::new("Select a band on the graph or in the strip above")
                        .size(11)
                        .color(colors::TEXT_FAINT),
                )
                .push(
                    Text::new("click: add · drag: freq/gain · scroll: Q · right-click: delete · double-click: flat")
                        .size(11)
                        .color(colors::TEXT_FAINT),
                )
                .into()
        } else {
            let i = selected;
            let bp = &self.params.band_params[i];
            let ftype = FilterType::from_i32(bp.filter_type.value());
            let color = colors::band_color(i);

            // Safety: each index i is visited once per frame
            let type_pick = unsafe {
                PickList::new(
                    Self::pick_state(&self.type_pick_states, i),
                    &FilterType::ALL,
                    Some(ftype),
                    move |t| Message::TypeChanged(i, t),
                )
                .text_size(11)
                .width(Length::Units(96))
            };
            // Which signal this band works on. Next to the type because the
            // two together describe what the band *is*, and because a mode set
            // by accident is otherwise silent — a Mid band on a wide source
            // looks exactly like a band doing nothing.
            let channel_pick = unsafe {
                PickList::new(
                    Self::channel_pick_state(&self.channel_pick_states, i),
                    &ChannelMode::ALL,
                    Some(ChannelMode::from_i32(bp.channel_mode.value())),
                    move |m| Message::ChannelChanged(i, m),
                )
                .text_size(11)
                .width(Length::Units(78))
            };
            let enable_toggle = Toggler::new(bp.enable.value(), String::new(), move |on| {
                Message::EnableChanged(i, on)
            })
            .size(14);

            // Solo: exclusive by construction — the audio path holds one index,
            // so pressing it elsewhere moves the solo rather than adding to it.
            // Toggling the soloed band's own switch clears it.
            let solo_on = self.solo.load(Ordering::Relaxed) == i;
            let solo_toggle = Toggler::new(solo_on, String::new(), move |on| {
                Message::SoloToggled(i, on)
            })
            .size(14);

            // A cut is described by its slope, everything else by gain and Q.
            // Showing a gain that nothing reads would be a lie about the state.
            let mut readouts = Row::new().spacing(16).align_items(Alignment::Center).push(
                Text::new(selection::format_freq(bp.freq.value()))
                    .size(12)
                    .color(colors::TEXT),
            );
            if ftype.has_gain() {
                readouts = readouts.push(
                    Text::new(selection::format_gain(bp.gain.value()))
                        .size(12)
                        .color(colors::TEXT),
                );
            }
            if ftype.is_cut() {
                readouts = readouts.push(
                    Text::new(Slope::from_i32(bp.slope.value()).name())
                        .size(12)
                        .color(colors::TEXT),
                );
            } else {
                readouts = readouts.push(
                    Text::new(selection::format_q(bp.q.value()))
                        .size(12)
                        .color(colors::TEXT),
                );
            }
            if bp.dyn_enable.value() && ftype.has_gain() {
                // The gain above is the *maximum* this band can apply, so show
                // what it is applying right now — otherwise the panel reads as
                // if that gain were always on, and you cannot see the band work.
                readouts = readouts.push(
                    Text::new(format!("DYN {:+.1} dB", self.dyn_meters.read(i)))
                        .size(11)
                        .color(colors::band_color(i)),
                );
            }

            let top = Row::new()
                .spacing(10)
                .align_items(Alignment::Center)
                .push(Text::new(format!("Band {}", i + 1)).size(12).color(color))
                .push(type_pick)
                .push(channel_pick)
                .push(enable_toggle)
                .push(Text::new("Enable").size(11).color(colors::TEXT_DIM))
                .push(solo_toggle)
                .push(Text::new("Solo").size(11).color(if solo_on {
                    color
                } else {
                    colors::TEXT_DIM
                }))
                .push(Column::new().width(Length::Fill))
                .push(readouts);

            let freq_slider = unsafe {
                nih_widgets::ParamSlider::new(
                    Self::slider_state(&self.freq_slider_states, i),
                    &self.params.band_params[i].freq,
                )
                .width(Length::Fill)
                .map(Message::ParamUpdate)
            };
            let mut controls = Row::new()
                .spacing(10)
                .align_items(Alignment::Center)
                .push(freq_slider);
            if ftype.has_gain() {
                let gain_slider = unsafe {
                    nih_widgets::ParamSlider::new(
                        Self::slider_state(&self.gain_slider_states, i),
                        &self.params.band_params[i].gain,
                    )
                    .width(Length::Fill)
                    .map(Message::ParamUpdate)
                };
                controls = controls.push(gain_slider);
            }
            if ftype.is_cut() {
                // Slope is the cut's steepness control; there is no Q to show.
                // Safety: each index i is visited once per frame
                let slope_pick = unsafe {
                    PickList::new(
                        Self::slope_pick_state(&self.slope_pick_states, i),
                        &Slope::ALL,
                        Some(Slope::from_i32(bp.slope.value())),
                        move |s| Message::SlopeChanged(i, s),
                    )
                    .text_size(11)
                    .width(Length::Fill)
                };
                controls = controls.push(slope_pick);
            } else {
                let q_slider = unsafe {
                    nih_widgets::ParamSlider::new(
                        Self::slider_state(&self.q_slider_states, i),
                        &self.params.band_params[i].q,
                    )
                    .width(Length::Fill)
                    .map(Message::ParamUpdate)
                };
                controls = controls.push(q_slider);
            }

            // Dynamics rows. Only a type with a gain can be dynamic: a cut
            // passes 0 dB at every level, so there is nothing for a detector to
            // modulate. Say that rather than showing controls that do nothing.
            // The panel keeps its fixed height either way.
            let dynamics_row: Element<'_, Message> = if ftype.has_gain() {
                let dyn_on = bp.dyn_enable.value();
                let mut row = Row::new()
                    .spacing(8)
                    .align_items(Alignment::Center)
                    .push(
                        Toggler::new(dyn_on, String::new(), move |on| {
                            Message::DynamicsToggled(i, on)
                        })
                        .size(14),
                    )
                    .push(Text::new("Dyn").size(11).color(colors::TEXT_DIM));
                let mut second: Option<Row<'_, Message>> = None;
                if dyn_on {
                    // Safety: each index i is visited once per frame
                    let threshold = unsafe {
                        nih_widgets::ParamSlider::new(
                            Self::slider_state(&self.dyn_threshold_states, i),
                            &self.params.band_params[i].dyn_threshold,
                        )
                        .width(Length::Fill)
                        .map(Message::ParamUpdate)
                    };
                    let range = unsafe {
                        nih_widgets::ParamSlider::new(
                            Self::slider_state(&self.dyn_range_states, i),
                            &self.params.band_params[i].dyn_range,
                        )
                        .width(Length::Fill)
                        .map(Message::ParamUpdate)
                    };
                    let attack = unsafe {
                        nih_widgets::ParamSlider::new(
                            Self::slider_state(&self.dyn_attack_states, i),
                            &self.params.band_params[i].dyn_attack,
                        )
                        .width(Length::Fill)
                        .map(Message::ParamUpdate)
                    };
                    let release = unsafe {
                        nih_widgets::ParamSlider::new(
                            Self::slider_state(&self.dyn_release_states, i),
                            &self.params.band_params[i].dyn_release,
                        )
                        .width(Length::Fill)
                        .map(Message::ParamUpdate)
                    };
                    row = row
                        .push(Text::new("Thr").size(11).color(colors::TEXT_DIM))
                        .push(threshold)
                        .push(Text::new("Range").size(11).color(colors::TEXT_DIM))
                        .push(range)
                        .push(Text::new("Atk").size(11).color(colors::TEXT_DIM))
                        .push(attack)
                        .push(Text::new("Rel").size(11).color(colors::TEXT_DIM))
                        .push(release);

                    // Second row: the knee, and where the detector listens.
                    // Both only mean something while dynamics are on, so they
                    // live here rather than taking panel space when they would
                    // do nothing.
                    let knee = unsafe {
                        nih_widgets::ParamSlider::new(
                            Self::slider_state(&self.dyn_knee_states, i),
                            &self.params.band_params[i].dyn_knee,
                        )
                        .width(Length::Fill)
                        .map(Message::ParamUpdate)
                    };
                    let sc_on = bp.dyn_sc_enable.value();
                    let mut sc_row = Row::new()
                        .spacing(8)
                        .align_items(Alignment::Center)
                        .push(Text::new("Knee").size(11).color(colors::TEXT_DIM))
                        .push(knee)
                        .push(
                            Toggler::new(sc_on, String::new(), move |on| {
                                Message::SidechainToggled(i, on)
                            })
                            .size(14),
                        )
                        .push(Text::new("Sidechain").size(11).color(colors::TEXT_DIM));
                    if sc_on {
                        // Safety: each index i is visited once per frame
                        let sc_freq = unsafe {
                            nih_widgets::ParamSlider::new(
                                Self::slider_state(&self.dyn_sc_freq_states, i),
                                &self.params.band_params[i].dyn_sc_freq,
                            )
                            .width(Length::Fill)
                            .map(Message::ParamUpdate)
                        };
                        sc_row = sc_row
                            .push(Text::new("at").size(11).color(colors::TEXT_DIM))
                            .push(sc_freq);
                    } else {
                        sc_row = sc_row.push(
                            Text::new("detector follows the band")
                                .size(11)
                                .color(colors::TEXT_FAINT),
                        );
                    }
                    second = Some(sc_row);
                } else {
                    row = row.push(
                        Text::new(
                            "the range moves the band when the level is past the threshold; the gain stays static",
                        )
                        .size(11)
                        .color(colors::TEXT_FAINT),
                    );
                }
                let mut column = Column::new().spacing(4).push(row);
                if let Some(second) = second {
                    column = column.push(second);
                }
                column.into()
            } else {
                Row::new()
                    .align_items(Alignment::Center)
                    .push(
                        Text::new("Dynamics need a type with a gain")
                            .size(11)
                            .color(colors::TEXT_FAINT),
                    )
                    .into()
            };

            Column::new()
                .height(Length::Units(PANEL_H))
                .spacing(4)
                .push(top)
                .push(controls)
                .push(dynamics_row)
                .into()
        };

        let peak_db = util::gain_to_db(self.peak_meter.load(Ordering::Relaxed));
        let footer = Row::new().height(Length::Units(FOOTER_H)).push(
            Text::new(if peak_db > -60.0 {
                format!("Peak: {:.1} dB", peak_db)
            } else {
                "Peak: -inf dB".to_string()
            })
            .size(11)
            .color(colors::TEXT_DIM),
        );

        let content = Column::new()
            .spacing(GAP)
            .push(header)
            .push(self.graph.view())
            .push(strip)
            .push(panel)
            .push(footer);

        content.into()
    }

    fn background_color(&self) -> nih_plug_iced::Color {
        colors::BG
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        test_params, test_params_with_dynamics, NullGuiContext, RecordingGuiContext,
    };
    use nih_plug::prelude::Param;

    /// An editor over explicit bands, plus the context that records its writes.
    fn editor_with_bands(
        bands: &[(f32, f32, f32, FilterType)],
    ) -> (FilterEditor, Arc<RecordingGuiContext>) {
        let params = test_params(bands);
        let context = Arc::new(RecordingGuiContext::default());
        let patch_state = Arc::new(std::sync::Mutex::new(PatchState::new(&params)));
        let flags = (
            params,
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            Arc::new(SpectrumTaps::new()),
            Arc::new(BandMeters::new()),
            Arc::new(AtomicUsize::new(crate::NO_SOLO)),
            patch_state,
        );
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx: Arc<dyn GuiContext> = context.clone();
        let (editor, _) = <FilterEditor as IcedEditor>::new(flags, ctx);
        (editor, context)
    }

    /// An editor over params the caller built, with band 0 selected.
    fn editor_over(params: Arc<FilterParams>) -> FilterEditor {
        let context = Arc::new(RecordingGuiContext::default());
        let patch_state = Arc::new(std::sync::Mutex::new(PatchState::new(&params)));
        let flags = (
            params,
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            Arc::new(SpectrumTaps::new()),
            Arc::new(BandMeters::new()),
            Arc::new(AtomicUsize::new(crate::NO_SOLO)),
            patch_state,
        );
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx: Arc<dyn GuiContext> = context;
        let (editor, _) = <FilterEditor as IcedEditor>::new(flags, ctx);
        editor.select(0);
        editor
    }

    #[test]
    fn the_panel_builds_for_every_type_with_and_without_dynamics() {
        // The dynamics row changes shape by band type (gain types get controls,
        // cuts get an explanation). Twelve sliders' worth of state indexed by
        // band is exactly where an out-of-range index would hide.
        for ftype in FilterType::ALL {
            for params in [
                test_params(&[(1_000.0, -6.0, 1.0, ftype)]),
                test_params_with_dynamics(&[(1_000.0, -6.0, 1.0, ftype)]),
            ] {
                let mut editor = editor_over(params);
                let _ = editor.view();
            }
        }
    }

    #[test]
    fn the_panel_builds_with_a_knee_a_sidechain_and_upward_mode() {
        // The widest the dynamics rows ever get: two rows, every slider and
        // both toggles live. A missing per-band state or a duplicated borrow
        // panics here rather than on someone's session.
        let params = test_params_with_dynamics(&[(1_000.0, -6.0, 1.0, FilterType::Bell)]);
        {
            let ctx = NullGuiContext;
            let setter = ParamSetter::new(&ctx);
            let bp = &params.band_params[0];
            for _ in 0..1 {
                setter.begin_set_parameter(&bp.dyn_knee);
                setter.set_parameter(&bp.dyn_knee, 6.0);
                setter.end_set_parameter(&bp.dyn_knee);
                setter.begin_set_parameter(&bp.dyn_sc_enable);
                setter.set_parameter(&bp.dyn_sc_enable, true);
                setter.end_set_parameter(&bp.dyn_sc_enable);
                setter.begin_set_parameter(&bp.dyn_sc_freq);
                setter.set_parameter(&bp.dyn_sc_freq, 80.0);
                setter.end_set_parameter(&bp.dyn_sc_freq);
            }
        }
        let mut editor = editor_over(params);
        let _ = editor.view();
    }

    #[test]
    fn the_sidechain_toggle_is_its_own_gesture() {
        let (editor, ctx) = editor_with_bands(&[(1_000.0, -6.0, 1.0, FilterType::Bell)]);
        let _ = editor.set_band_sidechain(0, true);
        assert_eq!(ctx.writes(), vec![1.0], "one write, normalized true");
    }

    #[test]
    fn enabling_dynamics_moves_the_gain_into_the_range() {
        // Pro-Q's "make dynamic": the band's gain becomes its dynamic range, so
        // the switch does something on a band that was static a moment ago. Its
        // three writes are enable + range + gain, each its own gesture so the
        // host records them contiguously.
        let (editor, ctx) = editor_with_bands(&[(1_000.0, -6.0, 1.0, FilterType::Bell)]);
        let _ = editor.set_band_dynamics(0, true);
        let writes = ctx.writes();
        assert_eq!(writes.len(), 3, "enable, range, gain: got {writes:?}");
        // -6 dB of gain over a -30..+30 range normalizes to 0.4.
        let range_write = writes[1];
        assert!(
            (range_write - 0.4).abs() < 0.01,
            "expected the -6 dB gain to land in the range, normalized {range_write}"
        );
        // 0 dB over the gain's range is its centre.
        assert!(
            (writes[2] - 0.5).abs() < 0.01,
            "expected the gain to go to 0 dB, normalized {}",
            writes[2]
        );
    }

    #[test]
    fn enabling_dynamics_leaves_an_existing_range_alone() {
        let params = test_params_with_dynamics(&[(1_000.0, -6.0, 1.0, FilterType::Bell)]);
        let patch_state = Arc::new(std::sync::Mutex::new(PatchState::new(&params)));
        let ctx = Arc::new(RecordingGuiContext::default());
        let flags = (
            params,
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            Arc::new(SpectrumTaps::new()),
            Arc::new(BandMeters::new()),
            Arc::new(AtomicUsize::new(crate::NO_SOLO)),
            patch_state,
        );
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx_dyn: Arc<dyn GuiContext> = ctx.clone();
        let (editor, _) = <FilterEditor as IcedEditor>::new(flags, ctx_dyn);
        let _ = editor.set_band_dynamics(0, true);
        assert_eq!(
            ctx.writes().len(),
            1,
            "a band that already has a range must only be switched on"
        );
    }

    fn editor_with(num_bands: usize) -> FilterEditor {
        let bands: Vec<(f32, f32, f32, FilterType)> = (0..num_bands)
            .map(|i| {
                let hz = 40.0 * 1.6f32.powi(i as i32);
                (hz.min(20_000.0), 3.0, 1.0, FilterType::Bell)
            })
            .collect();
        let params = test_params(&bands);
        let patch_state = Arc::new(std::sync::Mutex::new(PatchState::new(&params)));
        let flags = (
            params,
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            Arc::new(SpectrumTaps::new()),
            Arc::new(BandMeters::new()),
            Arc::new(AtomicUsize::new(crate::NO_SOLO)),
            patch_state,
        );
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx: Arc<dyn GuiContext> = Arc::new(NullGuiContext);
        let (editor, _) = <FilterEditor as IcedEditor>::new(flags, ctx);
        editor
    }

    #[test]
    fn chrome_fits_in_the_window() {
        // Every region above is given an explicit height, so this sum *is*
        // the layout height. With the old per-band rows this was unbounded in
        // the band count; now it is a constant that has to fit.
        assert!(
            chrome_height() <= EDITOR_HEIGHT,
            "chrome is {} px tall but the window is only {} px",
            chrome_height(),
            EDITOR_HEIGHT
        );
    }

    #[test]
    fn band_strip_fits_horizontally_at_full_capacity() {
        let width = MAX_BANDS as u16 * CHIP_W + (MAX_BANDS as u16 - 1) * 3;
        assert!(
            width <= EDITOR_WIDTH,
            "{MAX_BANDS} chips need {width} px but the window is {EDITOR_WIDTH} px wide"
        );
    }

    /// The view is a real widget tree; building it exercises the band strip,
    /// the selected-band panel and the empty state. It cannot check pixels,
    /// but it is the only thing that runs the editor code at all without a
    /// window, and it is what catches an out-of-range chip index or a panic in
    /// the panel path.
    #[test]
    fn view_builds_at_every_band_count() {
        for count in [0, 1, 2, 5, 12, MAX_BANDS] {
            let mut editor = editor_with(count);
            let _ = editor.view();
        }
    }

    #[test]
    fn view_builds_with_a_selected_band() {
        let mut editor = editor_with(3);
        editor.select(1);
        let _ = editor.view();
    }

    /// A selection left over from a larger band count (the "Bands" slider can
    /// lower it at any time) must not make the panel point past the end.
    #[test]
    fn a_stale_selection_is_clamped_to_a_live_band() {
        let mut editor = editor_with(1);
        editor.select(9);
        let _ = editor.view();
        assert_eq!(editor.selected(1), 0);
        assert_eq!(editor.selected(0), selection::NONE);
    }

    #[test]
    fn select_band_message_ignores_a_dead_index() {
        // The graph and the "Bands" slider both change the count; a queued
        // SelectBand for a band that no longer exists must be dropped rather
        // than select a parameter slot that is not in use.
        let editor = editor_with(1);
        editor.select(0);
        editor.select_if_live(5);
        assert_eq!(editor.selected(1), 0);
        editor.select_if_live(0);
        assert_eq!(editor.selected(1), 0);
    }

    #[test]
    fn changing_slope_is_one_single_gesture() {
        let (mut editor, ctx) = editor_with_bands(&[(100.0, 0.0, 0.707, FilterType::LowCut)]);
        let _ = editor.set_band_slope(0, Slope::Db48);

        let writes = ctx.writes();
        assert_eq!(writes.len(), 1, "one begin/set/end, one write: {writes:?}");
        let expected = editor.params.band_params[0]
            .slope
            .preview_normalized(Slope::Db48 as i32);
        assert!((writes[0] - expected).abs() < 1e-6);
    }

    /// A cut has no meaningful gain and no Q: the panel must build from the
    /// slope branch instead, with the gain and Q sliders absent.
    #[test]
    fn the_panel_builds_for_every_filter_type() {
        for filter_type in FilterType::ALL {
            let (mut editor, _) = editor_with_bands(&[(500.0, 3.0, 1.0, filter_type)]);
            editor.select(0);
            let _ = editor.view();
        }
    }

    /// Building the view must not write to the host — a repaint is not an edit.
    #[test]
    fn rendering_never_writes_parameters() {
        let (mut editor, ctx) = editor_with_bands(&[(500.0, 3.0, 1.0, FilterType::LowCut)]);
        editor.select(0);
        let _ = editor.view();
        assert!(ctx.writes().is_empty(), "view() wrote {:?}", ctx.writes());
    }
}
