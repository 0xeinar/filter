use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext};
use nih_plug_iced::widgets as nih_widgets;
use nih_plug_iced::*;
use std::sync::Arc;

use crate::ui;
use crate::FilterParams;
use crate::FilterType;
use crate::MAX_BANDS;

pub use colors::band_color;

pub fn default_state() -> Arc<IcedState> {
    IcedState::from_size(900, 500)
}

pub fn create(
    params: Arc<FilterParams>,
    peak_meter: Arc<AtomicF32>,
    editor_state: Arc<IcedState>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<FilterEditor>(editor_state, (params, peak_meter))
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

struct FilterEditor {
    params: Arc<FilterParams>,
    context: Arc<dyn GuiContext>,
    peak_meter: Arc<AtomicF32>,
    graph: ui::Graph,
    num_bands_slider_state: nih_widgets::param_slider::State,
    // Wrap in UnsafeCell to allow disjoint mutable access by index
    freq_slider_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    gain_slider_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
    q_slider_states: [std::cell::UnsafeCell<nih_widgets::param_slider::State>; MAX_BANDS],
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
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    ParamUpdate(nih_widgets::ParamMessage),
}

impl IcedEditor for FilterEditor {
    type Executor = executor::Default;
    type Message = Message;
    type InitializationFlags = (Arc<FilterParams>, Arc<AtomicF32>);

    fn new(
        (params, peak_meter): Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let graph = ui::Graph::new(params.clone(), peak_meter.clone());
        let editor = FilterEditor {
            params,
            context,
            peak_meter,
            graph,
            num_bands_slider_state: Default::default(),
            freq_slider_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            gain_slider_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
            q_slider_states: core::array::from_fn(|_| {
                std::cell::UnsafeCell::new(Default::default())
            }),
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
        }
        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        use nih_plug_iced::{Alignment, Column, Length, Row, Space, Text};

        let num_bands = self.params.num_bands.value() as usize;

        let header = Column::new()
            .push(
                Text::new("Filter")
                    .font(assets::NOTO_SANS_LIGHT)
                    .size(28)
                    .color(colors::TEXT)
                    .height(36.into())
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Left),
            )
            .push(Space::with_height(4.into()))
            .push(
                Row::new()
                    .push(Text::new("Bands: ").size(13).color(colors::TEXT_DIM))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.num_bands_slider_state,
                            &self.params.num_bands,
                        )
                        .map(Message::ParamUpdate),
                    ),
            );

        let mut bands_col = Column::new().spacing(2);

        for i in 0..num_bands {
            let band_color = colors::band_color(i);
            let ftype = FilterType::from_i32(self.params.band_params[i].filter_type.value());

            // Safety: Each index i is unique in this loop iteration
            let freq_slider = unsafe {
                nih_widgets::ParamSlider::new(
                    Self::slider_state(&self.freq_slider_states, i),
                    &self.params.band_params[i].freq,
                )
                .map(Message::ParamUpdate)
            };

            let gain_slider = unsafe {
                nih_widgets::ParamSlider::new(
                    Self::slider_state(&self.gain_slider_states, i),
                    &self.params.band_params[i].gain,
                )
                .map(Message::ParamUpdate)
            };

            let q_slider = unsafe {
                nih_widgets::ParamSlider::new(
                    Self::slider_state(&self.q_slider_states, i),
                    &self.params.band_params[i].q,
                )
                .map(Message::ParamUpdate)
            };

            let band_row = Row::new()
                .spacing(8)
                .align_items(Alignment::Center)
                .push(Text::new(format!("{} ", i + 1)).size(12).color(band_color))
                .push(
                    Text::new(format!("{:10} ", ftype.name()))
                        .size(11)
                        .color(colors::TEXT_DIM),
                )
                .push(freq_slider)
                .push(gain_slider)
                .push(q_slider);

            bands_col = bands_col.push(band_row);
        }

        let peak_db = util::gain_to_db(self.peak_meter.load(std::sync::atomic::Ordering::Relaxed));

        let content = Column::new()
            .spacing(8)
            .push(header)
            .push(self.graph.view())
            .push(Space::with_height(8.into()))
            .push(bands_col)
            .push(Space::with_height(8.into()))
            .push(
                Text::new(if peak_db > -60.0 {
                    format!("Peak: {:.1} dB", peak_db)
                } else {
                    "Peak: -inf dB".to_string()
                })
                .size(11)
                .color(colors::TEXT_DIM),
            );

        content.into()
    }

    fn background_color(&self) -> nih_plug_iced::Color {
        colors::BG
    }
}
