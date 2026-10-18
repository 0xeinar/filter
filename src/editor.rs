use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext, ParamSetter};
use nih_plug_iced::widgets as nih_widgets;
use nih_plug_iced::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::dsp::spectrum::SpectrumRing;
use crate::ui;
use crate::ui::selection;
use crate::FilterParams;
use crate::FilterType;
use crate::MAX_BANDS;

pub use colors::band_color;

/// Window size. The layout is fixed-height by construction (see the region
/// constants below), so this is a hard budget, not a starting suggestion —
/// `chrome_fits_in_the_window` asserts it.
pub const EDITOR_WIDTH: u16 = 900;
pub const EDITOR_HEIGHT: u16 = 500;

// Fixed region heights. Every region states its height explicitly so the
// whole column is deterministic and cannot silently overflow the window as
// the band count grows — which is exactly what the old per-band rows did
// (each row ~24 px, so past five bands the list ran off the bottom).
const HEADER_H: u16 = 28;
const STRIP_H: u16 = 24;
const PANEL_H: u16 = 66;
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
    spectrum_ring: Arc<SpectrumRing>,
    editor_state: Arc<IcedState>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<FilterEditor>(editor_state, (params, peak_meter, spectrum_ring))
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
    type_pick_states: [std::cell::UnsafeCell<pick_list::State<FilterType>>; MAX_BANDS],
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
    /// Band `index` enabled/disabled.
    EnableChanged(usize, bool),
    /// Band `index` selected from the band strip.
    SelectBand(usize),
    BypassToggled(bool),
    AutoGainToggled(bool),
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
    type InitializationFlags = (Arc<FilterParams>, Arc<AtomicF32>, Arc<SpectrumRing>);

    fn new(
        (params, peak_meter, spectrum_ring): Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let selection = Arc::new(AtomicUsize::new(selection::NONE));
        let graph = ui::Graph::new(
            params.clone(),
            context.clone(),
            peak_meter.clone(),
            spectrum_ring,
            selection.clone(),
        );
        let editor = FilterEditor {
            params,
            context,
            peak_meter,
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
            type_pick_states: core::array::from_fn(|_| {
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
        }
        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        use nih_plug_iced::{Alignment, Column, Length, Row, Text};

        let num_bands = self.params.num_bands.value() as usize;
        let selected = self.selected(num_bands);

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
            .push(Column::new().width(Length::Fill))
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
            let enable_toggle = Toggler::new(bp.enable.value(), String::new(), move |on| {
                Message::EnableChanged(i, on)
            })
            .size(14);

            let readouts = Row::new()
                .spacing(16)
                .align_items(Alignment::Center)
                .push(
                    Text::new(selection::format_freq(bp.freq.value()))
                        .size(12)
                        .color(colors::TEXT),
                )
                .push(
                    Text::new(selection::format_gain(bp.gain.value()))
                        .size(12)
                        .color(colors::TEXT),
                )
                .push(
                    Text::new(selection::format_q(bp.q.value()))
                        .size(12)
                        .color(colors::TEXT),
                );

            let top = Row::new()
                .spacing(10)
                .align_items(Alignment::Center)
                .push(Text::new(format!("Band {}", i + 1)).size(12).color(color))
                .push(type_pick)
                .push(enable_toggle)
                .push(Text::new("Enable").size(11).color(colors::TEXT_DIM))
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
            let gain_slider = unsafe {
                nih_widgets::ParamSlider::new(
                    Self::slider_state(&self.gain_slider_states, i),
                    &self.params.band_params[i].gain,
                )
                .width(Length::Fill)
                .map(Message::ParamUpdate)
            };
            let q_slider = unsafe {
                nih_widgets::ParamSlider::new(
                    Self::slider_state(&self.q_slider_states, i),
                    &self.params.band_params[i].q,
                )
                .width(Length::Fill)
                .map(Message::ParamUpdate)
            };

            Column::new()
                .height(Length::Units(PANEL_H))
                .spacing(4)
                .push(top)
                .push(
                    Row::new()
                        .spacing(10)
                        .align_items(Alignment::Center)
                        .push(freq_slider)
                        .push(gain_slider)
                        .push(q_slider),
                )
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
    use crate::test_support::{test_params, NullGuiContext};

    fn editor_with(num_bands: usize) -> FilterEditor {
        let bands: Vec<(f32, f32, f32, FilterType)> = (0..num_bands)
            .map(|i| {
                let hz = 40.0 * 1.6f32.powi(i as i32);
                (hz.min(20_000.0), 3.0, 1.0, FilterType::Bell)
            })
            .collect();
        let params = test_params(&bands);
        let flags = (
            params,
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            Arc::new(SpectrumRing::new()),
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
}
