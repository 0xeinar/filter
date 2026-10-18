//! End-to-end tests: a GUI gesture → the values the host receives → audio.
//!
//! What this covers that the unit tests do not: the *links* between the layers.
//! A unit test can prove the canvas computes +12 dB or that a biquad boosts by
//! 12 dB; only a test like this proves the number the GUI sends is the number
//! the filter ends up running with.
//!
//! One link is deliberately simulated rather than executed: nih-plug keeps the
//! parameter setters crate-private (`ParamMut` is `pub(crate)`), so no test can
//! write a value into a `FloatParam` — that is the host's job, inside nih-plug's
//! wrapper. Here the recorded normalized value goes through
//! `Param::preview_plain`, which is the same conversion the wrapper applies, and
//! then through the production biquad. Everything we own is real: the event, the
//! gesture, the normalized value, the plain value, the coefficients, the samples.

#![cfg(test)]

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_iced::backend::widget::canvas::Cursor;
use nih_plug_iced::{keyboard, mouse, Point, Rectangle};

use crate::dsp::biquad::BiquadFilter;
use crate::dsp::curve::{db_to_y, freq_to_x};
use crate::dsp::spectrum::SpectrumRing;
use crate::test_support::test_params;
use crate::ui::graph::{Graph, GRAPH_HEIGHT};
use crate::ui::selection;
use crate::{FilterParams, FilterType};

const SAMPLE_RATE: f32 = 48_000.0;
const WIDTH: f32 = 900.0;

/// Records the normalized values the GUI sends to the host.
#[derive(Default)]
struct RecordingContext {
    writes: Mutex<Vec<f32>>,
}

impl RecordingContext {
    fn writes(&self) -> Vec<f32> {
        self.writes.lock().unwrap().clone()
    }
}

impl GuiContext for RecordingContext {
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

struct Harness {
    params: Arc<FilterParams>,
    context: Arc<RecordingContext>,
    graph: Graph,
}

impl Harness {
    fn new(bands: &[(f32, f32, f32, FilterType)]) -> Self {
        let params = test_params(bands);
        let context = Arc::new(RecordingContext::default());
        let graph = Graph::new(
            params.clone(),
            context.clone(),
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            Arc::new(SpectrumRing::new()),
            Arc::new(AtomicUsize::new(selection::NONE)),
        );
        Self {
            params,
            context,
            graph,
        }
    }

    fn bounds() -> Rectangle {
        Rectangle {
            x: 0.0,
            y: 0.0,
            width: WIDTH,
            height: GRAPH_HEIGHT,
        }
    }

    fn press_at(&self, hz: f32, db: f32) {
        let b = Self::bounds();
        let cursor = Cursor::Available(Point::new(freq_to_x(hz) * b.width, db_to_y(db) * b.height));
        self.graph.driver().handle_mouse_event(
            mouse::Event::ButtonPressed(mouse::Button::Left),
            b,
            cursor,
        );
    }

    fn drag_to(&self, hz: f32, db: f32) {
        let b = Self::bounds();
        let cursor = Cursor::Available(Point::new(freq_to_x(hz) * b.width, db_to_y(db) * b.height));
        self.graph.driver().handle_mouse_event(
            mouse::Event::CursorMoved {
                position: cursor.position().unwrap(),
            },
            b,
            cursor,
        );
    }

    fn release(&self) {
        let b = Self::bounds();
        self.graph.driver().handle_mouse_event(
            mouse::Event::ButtonReleased(mouse::Button::Left),
            b,
            Cursor::Unavailable,
        );
    }

    fn press_key(&self, key: keyboard::KeyCode, modifiers: keyboard::Modifiers) {
        self.graph
            .driver()
            .handle_keyboard_event(keyboard::Event::KeyPressed {
                key_code: key,
                modifiers,
            });
    }

    /// The plain value the host would store for a write to `param`.
    fn plain(&self, band: usize, which: Which, normalized: f32) -> f32 {
        let bp = &self.params.band_params[band];
        match which {
            Which::Freq => bp.freq.preview_plain(normalized),
            Which::Gain => bp.gain.preview_plain(normalized),
            Which::Q => bp.q.preview_plain(normalized),
        }
    }
}

#[derive(Clone, Copy)]
enum Which {
    Freq,
    Gain,
    Q,
}

/// Steady-state gain (dB) of one biquad at `probe_hz`, measured on a sine.
fn measured_gain_db(
    freq: f32,
    gain_db: f32,
    q: f32,
    filter_type: FilterType,
    probe_hz: f32,
) -> f32 {
    let mut filter = BiquadFilter::new();
    filter.update_coefficients(freq, gain_db, q, filter_type, SAMPLE_RATE);
    filter.reset();

    let settle = 8_192;
    let measure = 4_096;
    let mut input_sq = 0.0f64;
    let mut output_sq = 0.0f64;
    for n in 0..settle + measure {
        let phase = 2.0 * std::f32::consts::PI * probe_hz * n as f32 / SAMPLE_RATE;
        let input = 0.5 * phase.sin();
        let output = filter.process(input);
        if n >= settle {
            input_sq += (input as f64).powi(2);
            output_sq += (output as f64).powi(2);
        }
    }
    let ratio = (output_sq / input_sq).sqrt();
    20.0 * ratio.log10() as f32
}

#[test]
fn click_drag_to_plus_12_db_actually_boosts_that_frequency() {
    let h = Harness::new(&[]);

    // One gesture: press on empty graph at 1 kHz, drag up to +12 dB.
    h.press_at(1_000.0, 0.0);
    h.drag_to(1_000.0, 12.0);
    h.release();

    let writes = h.context.writes();
    // Creation writes freq/gain/q/type/enable and then the count; the drag
    // that follows re-writes freq and gain. In a host the last write wins, so
    // the final freq/gain pair is what the audio thread would end up with.
    assert_eq!(writes.len(), 8, "create + drag, got {writes:?}");
    assert!(
        (writes[5] - 1.0 / crate::MAX_BANDS as f32).abs() < 1e-6,
        "the band count should be raised"
    );

    // What the host would store, converted exactly as it converts.
    let freq = h.plain(0, Which::Freq, writes[6]);
    let gain = h.plain(0, Which::Gain, writes[7]);
    let q = h.plain(0, Which::Q, writes[2]);
    assert!(
        (freq - 1_000.0).abs() / 1_000.0 < 0.02,
        "the band should land on the click frequency, got {freq:.1} Hz"
    );
    assert!(
        (gain - 12.0).abs() < 0.2,
        "the drag should land on +12 dB, got {gain:.2} dB"
    );

    // The audio the plugin would produce for that band.
    let at_target = measured_gain_db(freq, gain, q, FilterType::Bell, 1_000.0);
    let away = measured_gain_db(freq, gain, q, FilterType::Bell, 100.0);
    assert!(
        (at_target - gain).abs() < 0.5,
        "a +{gain:.1} dB band should measure {gain:.1} dB at its centre, got {at_target:.2} dB"
    );
    assert!(
        away.abs() < 0.5,
        "100 Hz is two octaves below and should be untouched, got {away:.2} dB"
    );
}

#[test]
fn keyboard_nudge_moves_the_audio_not_just_the_parameter() {
    let h = Harness::new(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);
    h.graph.driver();

    // Select the only band with a click on its node, as a user would.
    h.press_at(1_000.0, 0.0);
    h.release();

    // Shift+Up is the coarse nudge: 1 dB.
    h.press_key(keyboard::KeyCode::Up, keyboard::Modifiers::SHIFT);

    let writes = h.context.writes();
    let gain_write = *writes.last().expect("a gain nudge should write");
    let gain = h.plain(0, Which::Gain, gain_write);
    assert!((gain - 1.0).abs() < 1e-4, "expected +1.0 dB, got {gain}");

    let measured = measured_gain_db(1_000.0, gain, 1.0, FilterType::Bell, 1_000.0);
    assert!(
        (measured - 1.0).abs() < 0.1,
        "a +1.0 dB nudge should measure +1.0 dB, got {measured:.3}"
    );
}

#[test]
fn clicking_a_node_does_not_create_another_band() {
    // A regression guard for the click-to-create interaction: the second click
    // on an existing node must select and drag it, not spawn a duplicate.
    let h = Harness::new(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);
    h.press_at(1_000.0, 0.0);
    h.release();

    let writes = h.context.writes();
    assert!(
        writes.is_empty(),
        "clicking an existing node wrote {writes:?}, expected nothing"
    );
}

#[test]
fn deleting_the_last_band_leaves_the_audio_untouched() {
    // Keyboard delete is the interactive path; the count write is what the
    // audio thread reads, so a count of zero means the DSP is bypassed.
    let h = Harness::new(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]);
    h.press_at(1_000.0, 12.0);
    h.release();
    h.press_key(keyboard::KeyCode::Delete, keyboard::Modifiers::default());

    let writes = h.context.writes();
    assert_eq!(
        *writes.last().expect("delete should write the count"),
        0.0,
        "the count should drop to zero, got {writes:?}"
    );
}
