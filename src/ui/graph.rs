//! EQ curve graph — canvas widget drawing grid, response curve, and band
//! handles.
//!
//! Interaction:
//! - click empty graph: create a band at that frequency (0 dB, Bell, enabled)
//!   and grab it, so the same click-drag also sets its gain
//! - click a node: select that band — the editor's band panel follows it
//! - drag a node: frequency (x) and gain (y)
//! - scroll over a node: Q (Alt = fine)
//! - double-click a node: gain back to 0 dB
//! - right-click a node: delete the band and compact the remaining bands

use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nih_plug::prelude::*;
use nih_plug_iced::backend::widget::canvas::{
    self, Canvas as IcedCanvas, Cursor, Fill, Frame, Geometry, Path, Program, Stroke,
    Text as CanvasText,
};
use nih_plug_iced::{keyboard, mouse, Color, Length, Point, Rectangle, Size};

use crate::dsp::curve::{
    band_response_db, db_to_y, freq_to_x, nearest_grabbable_node, px_to_freq, scroll_q_step,
    y_to_db, BandSpec, DB_MAX, FREQ_MAX, FREQ_MIN, GRID_DBS, GRID_FREQS,
};
use crate::dsp::spectrum::{
    tilt_db, SpectrumFrame, SpectrumRing, SpectrumView, TraceBallistics, BINS, TILT_DB_PER_OCT,
};
use crate::ui::selection;
use crate::{editor, FilterParams, FilterType, MAX_BANDS};

/// Height of the graph area in points.
pub const GRAPH_HEIGHT: f32 = 300.0;

/// Two clicks on the same node inside this window count as a double-click
/// (resets the band's gain to 0 dB).
const DOUBLE_CLICK: Duration = Duration::from_millis(400);

/// Keyboard nudge steps for the selected band. One press is a semitone or
/// 0.1 dB; Shift makes it an octave or 1 dB.
const NUDGE_SEMITONES: f32 = 1.0;
const NUDGE_COARSE_SEMITONES: f32 = 12.0;
const NUDGE_DB: f32 = 0.1;
const NUDGE_COARSE_DB: f32 = 1.0;

/// Collect the current band parameters as pure `BandSpec` values.
///
/// Deliberately the crate-level function rather than a copy: the graph used to
/// duplicate this, and the copy silently fell behind when a field was added.
fn band_specs(params: &FilterParams, num_active: usize) -> [BandSpec; MAX_BANDS] {
    crate::band_specs(params, num_active)
}

/// x position (0..1) of each spectrum bin's upper edge at `sample_rate`.
///
/// Not a constant: a bin's frequency scales with the rate, so the bars move
/// when the host switches from 44.1 to 96 kHz. Recomputed when the published
/// rate changes, cached otherwise.
fn spectrum_bin_x(sample_rate: f32) -> Vec<f32> {
    (0..BINS)
        .map(|i| freq_to_x(crate::dsp::spectrum::bin_edge_hz(i, sample_rate).min(FREQ_MAX)))
        .collect()
}

/// Maps an analyzer dB value to a canvas y coordinate.
///
/// Shared by both taps so the two curves stay directly comparable. `floor_db`
/// comes from the display range selector (0 to -60 / -90 / -120); the top of
/// the display is always 0 dB at three quarters height, so changing the range
/// deepens the floor rather than rescaling the signal.
fn spectrum_y(bounds: Rectangle, db: f32, floor_db: f32) -> f32 {
    let span = (-floor_db).max(1.0);
    let y01 = ((db - floor_db) / span).clamp(0.0, 1.0) * 0.75;
    bounds.y + bounds.height * (1.0 - y01)
}

/// Map a frequency + gain to pixel coordinates within bounds.
fn to_px(bounds: Rectangle, freq: f32, db: f32) -> Point {
    Point {
        x: bounds.x + freq_to_x(freq) * bounds.width,
        y: bounds.y + db_to_y(db) * bounds.height,
    }
}

/// The spectrum display controls, drawn as chips in the graph's top-right
/// corner.
///
/// They live in the canvas rather than the header because the header is
/// already at the window's width budget (`chrome_fits_in_the_window`), and
/// because they are about the picture the cursor is sitting on. Every one of
/// them is display state, not audio state: they are deliberately not host
/// parameters, so they are not automated and not saved with the project.
const CHIP_H: f32 = 16.0;
const CHIP_GAP: f32 = 4.0;
const CHIP_PAD: f32 = 6.0;
/// Rough advance width of the 9 px font used for the chip labels.
const CHIP_CHAR_W: f32 = 5.4;
const CHIP_MARGIN: f32 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chip {
    Freeze,
    Tilt,
    Range,
    PreGain,
}

impl Chip {
    /// Right-to-left on screen.
    const ALL: [Chip; 4] = [Chip::Freeze, Chip::Tilt, Chip::Range, Chip::PreGain];

    fn label(self, view: &SpectrumView) -> String {
        match self {
            Chip::Freeze => if view.frozen() { "Frozen" } else { "Freeze" }.to_string(),
            Chip::Tilt => if view.tilted() { "Tilt on" } else { "Tilt" }.to_string(),
            Chip::Range => format!("{:.0} dB", view.floor_db()),
            Chip::PreGain => format!("Pre {:+.0}", view.pre_gain_db()),
        }
    }

    /// Whether the chip should read as engaged. `Range` and `Pre` cycle
    /// through values instead of toggling, so they are "engaged" whenever they
    /// are off their default — a chip that looked dead while the display was
    /// in its default range would be indistinguishable from a disabled one.
    fn active(self, view: &SpectrumView) -> bool {
        match self {
            Chip::Freeze => view.frozen(),
            Chip::Tilt => view.tilted(),
            Chip::Range => {
                view.floor_db()
                    != crate::dsp::spectrum::DB_RANGES[crate::dsp::spectrum::DEFAULT_DB_RANGE]
            }
            Chip::PreGain => view.pre_gain_db() != 0.0,
        }
    }

    /// What a click does: toggles for the two switches, a cycle for the two
    /// selectors.
    fn activate(self, view: &SpectrumView) {
        match self {
            Chip::Freeze => view.set_frozen(!view.frozen()),
            Chip::Tilt => view.set_tilted(!view.tilted()),
            Chip::Range => {
                view.cycle_range();
            }
            Chip::PreGain => {
                view.cycle_pre_gain();
            }
        }
    }
}

/// Screen rectangles for the chips, laid out right-to-left from the top-right
/// corner. Widths follow the label, so "Frozen" does not overlap "Tilt on".
fn chip_rects(view: &SpectrumView, bounds: Rectangle) -> [(Chip, Rectangle); 4] {
    let mut right = bounds.x + bounds.width - CHIP_MARGIN;
    let y = bounds.y + CHIP_MARGIN;
    let mut out = [(
        Chip::Freeze,
        Rectangle {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        },
    ); 4];
    for (slot, chip) in out.iter_mut().zip(Chip::ALL.iter().rev()) {
        let width = chip.label(view).chars().count() as f32 * CHIP_CHAR_W + CHIP_PAD * 2.0;
        *slot = (
            *chip,
            Rectangle {
                x: right - width,
                y,
                width,
                height: CHIP_H,
            },
        );
        right -= width + CHIP_GAP;
    }
    out
}

/// The chip under `pos`, if any. Checked before the nodes so a chip is never
/// shadowed by a band handle sitting underneath it.
fn chip_at(pos: Point, view: &SpectrumView, bounds: Rectangle) -> Option<Chip> {
    chip_rects(view, bounds)
        .into_iter()
        .find(|(_, rect)| rect.contains(pos))
        .map(|(chip, _)| chip)
}

/// The gain a band's node is drawn and grabbed at.
///
/// A cut passes 0 dB and ignores its gain parameter, so its node belongs on
/// the 0 dB line where its curve actually starts — not floating wherever the
/// (unused) gain happens to sit.
/// Whether the band's dynamics will actually move it.
///
/// `dyn_enable` alone is not enough: a range of 0 dB is Pro-Q's "clear
/// dynamics", and a band in that state is static however the switch is set.
fn dyn_is_active(bp: &crate::BandParams) -> bool {
    bp.dyn_enable.value()
        && bp.dyn_range.value().abs() >= crate::dsp::dynamics::MIN_RANGE_DB
        && FilterType::from_i32(bp.filter_type.value()).has_gain()
}

fn node_gain_db(bp: &crate::BandParams) -> f32 {
    if FilterType::from_i32(bp.filter_type.value()).has_gain() {
        bp.gain.value()
    } else {
        0.0
    }
}

/// Node positions in canvas-local pixel coordinates `(band_index, x, y)`
/// for all enabled, active bands.
fn node_positions(
    params: &FilterParams,
    num_active: usize,
    bounds: Rectangle,
) -> Vec<(usize, f32, f32)> {
    (0..num_active)
        .filter(|&i| params.band_params[i].enable.value())
        .map(|i| {
            let p = to_px(
                bounds,
                params.band_params[i].freq.value(),
                node_gain_db(&params.band_params[i]),
            );
            (i, p.x, p.y)
        })
        .collect()
}

/// Drag state for the interactive node handles. Only the band index is
/// stored — the params are reached through the stable `Arc<FilterParams>`.
#[derive(Debug, Clone, Copy)]
struct DragState {
    band: usize,
}

struct GraphProgram {
    params: Arc<FilterParams>,
    context: Arc<dyn GuiContext>,
    _peak_meter: Arc<AtomicF32>,
    /// Both analyzer taps: input (pre-EQ) and output (post-EQ).
    taps: Arc<crate::dsp::spectrum::SpectrumTaps>,
    /// Live dynamic gain per band. Read for the node readout so a dynamic band
    /// shows how far it has moved, not just where the knob sits.
    dyn_meters: Arc<crate::dsp::dynamics::BandMeters>,
    /// Undo history and A/B slots. The canvas owns the keyboard, so the undo
    /// shortcut lands here and is forwarded to the editor as a message.
    patch_state: Arc<std::sync::Mutex<crate::patch::PatchState>>,
    /// Display ballistics + frozen frame, one per tap. GUI-thread only: the
    /// audio thread publishes raw measurements and never holds display state.
    /// `RefCell` because `GraphView` (and thus every `&self` method here) is
    /// rebuilt each frame, so an `&mut self` field would not survive.
    ballistics_in: RefCell<TraceBallistics>,
    ballistics_out: RefCell<TraceBallistics>,
    /// Precomputed x position (0..1 across the graph) for every spectrum bin.
    /// The bin → frequency mapping never changes, so this saves a `powf` and a
    /// `log10` per bin per frame — 192 of them at 60 fps.
    spec_x: RefCell<Vec<f32>>,
    /// Sample rate `spec_x` was built for. The axis is rebuilt when the audio
    /// thread reports a different one, which happens on a device change.
    spec_rate: std::cell::Cell<f32>,
    /// Cached geometry for the layers that only change when a parameter does
    /// (grid, band curves, combined curve). Without this, every frame rebuilt
    /// up to 24 × 120 `band_response_db` evaluations — each with a `powf`,
    /// two `ln` and a `sqrt` — for a picture that was identical to the last
    /// frame. The spectrum and nodes stay uncached by design.
    cache: RefCell<Option<CachedLayers>>,
    /// Active node drag, if any. GUI-thread only; `Cell` because `GraphView`
    /// (and thus `&self` program methods) is rebuilt every frame.
    drag: Cell<Option<DragState>>,
    /// Node currently under the cursor, for hover highlighting.
    hover: Cell<Option<usize>>,
    /// Alt held — fine (4x) scroll steps for Q.
    fine_scroll: Cell<bool>,
    /// Last click `(band, instant)` — used to detect a double-click on a node
    /// (reset its gain) without needing iced's event stream to expose one.
    last_click: Cell<Option<(usize, std::time::Instant)>>,
    /// Selected band, shared with the editor (`selection::NONE` for none).
    /// An atomic instead of a `Cell` because the editor renders from it too,
    /// and nih-plug rebuilds the whole widget tree every frame.
    selected: Arc<AtomicUsize>,
}

/// Geometry that is only valid for one parameter/bounds state.
struct CachedLayers {
    key: u64,
    grid: Geometry,
    curves: Geometry,
}

impl CachedLayers {
    /// Hash everything the cached layers are drawn from: the active bands'
    /// plain values, plus the canvas bounds (pixel mapping depends on both
    /// size and origin).
    fn key_for(params: &FilterParams, bounds: Rectangle, tilted: bool) -> u64 {
        let mut h = DefaultHasher::new();
        let num_active = params.num_bands.value() as usize;
        num_active.hash(&mut h);
        // The curve itself is tilted, so toggling the tilt invalidates it.
        tilted.hash(&mut h);
        for i in 0..num_active {
            let bp = &params.band_params[i];
            bp.freq.value().to_bits().hash(&mut h);
            bp.gain.value().to_bits().hash(&mut h);
            bp.q.value().to_bits().hash(&mut h);
            bp.filter_type.value().hash(&mut h);
            // Slopes change the drawn curve, so a slope edit has to invalidate
            // the cached geometry just like a freq/gain edit does.
            bp.slope.value().hash(&mut h);
            bp.enable.value().hash(&mut h);
            // The range curve is drawn from these, so a dynamic edit has to
            // invalidate the cached geometry like any other.
            bp.dyn_enable.value().hash(&mut h);
            bp.dyn_range.value().to_bits().hash(&mut h);
        }
        for v in [bounds.x, bounds.y, bounds.width, bounds.height] {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }
}

impl GraphProgram {
    /// Write a plain value to a float param as a complete automation gesture.
    /// Per-move gestures keep the DAW's undo stack sane and match how single
    /// clicks on the built-in sliders behave.
    fn set_plain(&self, param: &FloatParam, plain: f32) {
        let setter = ParamSetter::new(self.context.as_ref());
        setter.set_parameter(param, plain);
    }

    /// Write a parameter as a complete automation gesture (begin → set → end),
    /// so the host records one clean undo step per parameter instead of a
    /// loose value change.
    fn set_gesture<P: Param>(&self, param: &P, value: P::Plain) {
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter(param, value);
        setter.end_set_parameter(param);
    }

    /// Copy every parameter of band `from` into band `to`.
    fn copy_band(&self, from: usize, to: usize) {
        let src = &self.params.band_params[from];
        let dst = &self.params.band_params[to];
        self.set_gesture(&dst.freq, src.freq.value());
        self.set_gesture(&dst.gain, src.gain.value());
        self.set_gesture(&dst.q, src.q.value());
        self.set_gesture(&dst.filter_type, src.filter_type.value());
        self.set_gesture(&dst.slope, src.slope.value());
        self.set_gesture(&dst.enable, src.enable.value());
    }

    /// Append a band at `freq` with 0 dB gain and a neutral Q, enabled.
    ///
    /// The band's parameters are written *before* the count is raised so the
    /// audio thread never sees a half-initialised band. Returns the new band's
    /// index, or `None` when the bank is full.
    fn create_band(&self, freq: f32, gain_db: f32) -> Option<usize> {
        let num_active = self.params.num_bands.value() as usize;
        if num_active >= MAX_BANDS {
            return None;
        }
        let bp = &self.params.band_params[num_active];
        self.set_gesture(&bp.freq, freq.clamp(FREQ_MIN, FREQ_MAX));
        self.set_gesture(&bp.gain, gain_db.clamp(-DB_MAX, DB_MAX));
        self.set_gesture(&bp.q, 1.0);
        self.set_gesture(&bp.filter_type, FilterType::Bell as i32);
        self.set_gesture(&bp.slope, crate::Slope::DEFAULT as i32);
        self.set_gesture(&bp.enable, true);
        self.set_gesture(&self.params.num_bands, num_active as i32 + 1);
        Some(num_active)
    }

    /// Delete band `index` and compact the remaining bands down one slot.
    ///
    /// The tail is shifted first and the count lowered last, so no band is ever
    /// visible twice or lost mid-update.
    fn delete_band(&self, index: usize) {
        let num_active = self.params.num_bands.value() as usize;
        if index >= num_active {
            return;
        }
        for (from, to) in crate::dsp::curve::compaction_plan(index, num_active) {
            self.copy_band(from, to);
        }
        self.set_gesture(&self.params.num_bands, num_active as i32 - 1);
    }

    /// Selected band, or `None`.
    /// Lock the shared history. A panic elsewhere must not brick the editor,
    /// so a poisoned lock is taken back rather than unwrapped.
    fn lock_patch_state(&self) -> std::sync::MutexGuard<'_, crate::patch::PatchState> {
        self.patch_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn selected(&self) -> Option<usize> {
        let sel = self.selected.load(Ordering::Relaxed);
        if selection::is_none(sel) {
            None
        } else {
            Some(sel)
        }
    }

    /// Select a band (or clear the selection with `None`).
    fn select(&self, band: Option<usize>) {
        let value = band.unwrap_or(selection::NONE);
        self.selected.store(value, Ordering::Relaxed);
    }

    /// Delete a band and move the selection the way the compaction moved the
    /// bands.
    ///
    /// Both the right-click and the Delete key go through here: a deleted band
    /// releases its slot, the tail slides down, and a selection above it has to
    /// follow — otherwise the panel starts silently editing a different band.
    fn delete_and_follow(&self, band: usize, num_active: usize) {
        self.delete_band(band);
        let next = selection::after_delete(
            self.selected.load(Ordering::Relaxed),
            band,
            num_active.saturating_sub(1),
        );
        self.selected.store(next, Ordering::Relaxed);
    }

    /// Set a band's Q from a normalized value, as a complete gesture.
    fn set_q_normalized(&self, band: usize, normalized: f32) {
        let param = &self.params.band_params[band].q;
        let setter = ParamSetter::new(self.context.as_ref());
        setter.begin_set_parameter(param);
        setter.set_parameter_normalized(param, normalized);
        setter.end_set_parameter(param);
    }

    fn draw_grid(&self, frame: &mut Frame, bounds: Rectangle) {
        let grid = Color::from_rgba(1.0, 1.0, 1.0, 0.06);
        let label = Color::from_rgba(1.0, 1.0, 1.0, 0.25);

        // Grid lines are stroked, not filled: filling a zero-area `Path::line`
        // tessellates to nothing, so the original `frame.fill` drew no lines at
        // all — only the labels. (Found by the headless draw test: the grid
        // layer had 12 text runs and 0 mesh vertices.)
        let hairline = Stroke::default().with_color(grid).with_width(1.0);

        // Vertical frequency lines + labels
        for &f in &GRID_FREQS {
            let x = bounds.x + freq_to_x(f) * bounds.width;
            frame.stroke(
                &Path::line(
                    Point::new(x, bounds.y),
                    Point::new(x, bounds.y + bounds.height),
                ),
                hairline,
            );
            let text = if f >= 1000.0 {
                format!("{}k", f / 1000.0)
            } else {
                format!("{}", f as i32)
            };
            frame.fill_text(CanvasText {
                content: text,
                position: Point::new(x + 2.0, bounds.y + bounds.height - 12.0),
                color: label,
                size: 9.0,
                ..CanvasText::default()
            });
        }

        // Horizontal dB lines + labels (0 dB drawn separately, emphasized)
        for &db in &GRID_DBS {
            let y = bounds.y + db_to_y(db) * bounds.height;
            frame.stroke(
                &Path::line(
                    Point::new(bounds.x, y),
                    Point::new(bounds.x + bounds.width, y),
                ),
                hairline,
            );
            frame.fill_text(CanvasText {
                content: format!("{:+}", db),
                position: Point::new(bounds.x + 3.0, y + 2.0),
                color: label,
                size: 9.0,
                ..CanvasText::default()
            });
        }

        // 0 dB line, brighter
        let y0 = bounds.y + db_to_y(0.0) * bounds.height;
        frame.stroke(
            &Path::line(
                Point::new(bounds.x, y0),
                Point::new(bounds.x + bounds.width, y0),
            ),
            Stroke::default()
                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.14))
                .with_width(1.0),
        );
    }

    /// Update the smoothed spectrum from the ring and draw it as a filled
    /// area along the bottom of the graph (the INPUT signal).
    ///
    /// The input trace also carries the per-trace offset, so it can be lifted
    /// or dropped out from under the post trace instead of disappearing beneath
    /// it.
    fn draw_input_spectrum(&self, frame: &mut Frame, bounds: Rectangle) {
        let offset = self.taps.view.pre_gain_db();
        let db = self.shown_spectrum(&self.taps.input, &self.ballistics_in, offset);
        let floor = self.taps.view.floor_db();
        let spec_x = self.spec_x.borrow();

        let mut path_pts: Vec<Point> = Vec::with_capacity(BINS + 2);
        path_pts.push(Point::new(bounds.x, bounds.y + bounds.height));
        for (i, &value) in db.iter().enumerate() {
            let x = bounds.x + spec_x[i] * bounds.width;
            path_pts.push(Point::new(x, spectrum_y(bounds, value, floor)));
        }
        // Close down to the bottom-right corner; the bottom-left corner is the
        // first point.
        path_pts.push(Point::new(
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        ));

        let area = Path::new(|p| {
            p.move_to(path_pts[0]);
            for &pt in &path_pts[1..] {
                p.line_to(pt);
            }
            p.close();
        });
        frame.fill(&area, Fill::from(Color::from_rgba(0.35, 0.85, 0.6, 0.10)));
        // Faint top edge stroke
        let line = Path::new(|p| {
            p.move_to(path_pts[1]);
            for &pt in &path_pts[2..path_pts.len() - 1] {
                p.line_to(pt);
            }
        });
        frame.stroke(
            &line,
            Stroke::default()
                .with_color(Color::from_rgba(0.35, 0.85, 0.6, 0.25))
                .with_width(1.0),
        );
    }

    /// The post-EQ spectrum (what leaves the plugin), drawn as a line over the
    /// input's filled area.
    ///
    /// A line, not a second filled area: two translucent fills overlapping
    /// would read as a third colour wherever the curves cross, and the useful
    /// information is the *gap* between them. A green fill plus an amber line
    /// answers "what did the EQ do" at a glance.
    fn draw_output_spectrum(&self, frame: &mut Frame, bounds: Rectangle) {
        let db = self.shown_spectrum(&self.taps.output, &self.ballistics_out, 0.0);
        let floor = self.taps.view.floor_db();
        let spec_x = self.spec_x.borrow();

        let line = Path::new(|p| {
            for (i, &value) in db.iter().enumerate() {
                let pt = Point::new(
                    bounds.x + spec_x[i] * bounds.width,
                    spectrum_y(bounds, value, floor),
                );
                if i == 0 {
                    p.move_to(pt);
                } else {
                    p.line_to(pt);
                }
            }
        });
        frame.stroke(
            &line,
            Stroke::default()
                .with_color(Color::from_rgba(1.0, 0.72, 0.35, 0.85))
                .with_width(1.5),
        );
    }

    /// Advance one trace's ballistics and return the dB values to draw.
    ///
    /// Smoothing lives on the GUI side rather than in the analyzer so the audio
    /// thread never holds display state and the two taps can decay
    /// independently. `offset` shifts the trace in dB after smoothing: the
    /// per-trace offset is a display preference and must not leak back into the
    /// measurement.
    fn shown_spectrum(
        &self,
        ring: &SpectrumRing,
        ballistics: &RefCell<TraceBallistics>,
        offset: f32,
    ) -> [f32; BINS] {
        let latest: SpectrumFrame = ring.read_newest();
        let frozen = self.taps.view.frozen();
        let mut ballistics = ballistics.borrow_mut();
        let mut db = ballistics.update(&latest.db, frozen);
        if offset != 0.0 {
            for value in db.iter_mut() {
                *value += offset;
            }
        }
        db
    }

    fn draw_curve(&self, frame: &mut Frame, bounds: Rectangle) {
        let num_active = self.params.num_bands.value() as usize;
        let specs = band_specs(&self.params, num_active);
        const POINTS: usize = 160;

        let mut combined: Vec<Point> = Vec::with_capacity(POINTS);
        for i in 0..POINTS {
            let x01 = i as f32 / (POINTS - 1) as f32;
            let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(x01);
            let mut db = 0.0f32;
            for spec in specs.iter().take(num_active) {
                if spec.enabled {
                    db += band_response_db(spec, freq);
                }
            }
            // The curve shares the spectrum's tilted reference, so a flat EQ
            // still reads as "the post trace sits on the pre trace" at every
            // frequency. Tilt here is the same `tilt_db` the analyzer applies.
            if self.taps.view.tilted() {
                db += tilt_db(freq, TILT_DB_PER_OCT);
            }
            combined.push(to_px(bounds, freq, db.clamp(-DB_MAX - 6.0, DB_MAX + 6.0)));
        }

        // Subtle fill under curve down to the 0 dB line
        let y0 = bounds.y + db_to_y(0.0) * bounds.height;
        let area = Path::new(|p| {
            p.move_to(combined[0]);
            for &pt in &combined[1..] {
                p.line_to(pt);
            }
            p.line_to(Point::new(combined[POINTS - 1].x, y0));
            p.line_to(Point::new(combined[0].x, y0));
            p.close();
        });
        frame.fill(&area, Fill::from(Color::from_rgba(0.45, 0.7, 1.0, 0.07)));

        // Bright combined curve stroke
        let line = Path::new(|p| {
            p.move_to(combined[0]);
            for &pt in &combined[1..] {
                p.line_to(pt);
            }
        });
        frame.stroke(
            &line,
            Stroke::default()
                .with_color(Color::from_rgba(0.55, 0.75, 1.0, 0.95))
                .with_width(2.0),
        );
    }

    fn draw_band_curves(&self, frame: &mut Frame, bounds: Rectangle) {
        let num_active = self.params.num_bands.value() as usize;
        let specs = band_specs(&self.params, num_active);
        const POINTS: usize = 120;

        for (i, spec) in specs.iter().enumerate().take(num_active) {
            if !spec.enabled {
                continue;
            }
            let color = editor::band_color(i);
            let stroke_color = Color { a: 0.35, ..color };

            let mut first = true;
            let line = Path::new(|p| {
                for j in 0..POINTS {
                    let x01 = j as f32 / (POINTS - 1) as f32;
                    let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(x01);
                    let db = band_response_db(spec, freq);
                    let pt = to_px(bounds, freq, db.clamp(-DB_MAX - 6.0, DB_MAX + 6.0));
                    if first {
                        p.move_to(pt);
                        first = false;
                    } else {
                        p.line_to(pt);
                    }
                }
            });
            frame.stroke(
                &line,
                Stroke::default().with_color(stroke_color).with_width(1.0),
            );

            // A dynamic band gets a second, dashed-ish faint curve at the far
            // end of its range, so the drawing says where the band can go as
            // well as where it sits. Without it the range is invisible: the
            // static curve is all you would see, and the band would look
            // static while audibly ducking.
            let range_db = self.params.band_params[i].dyn_range.value();
            if dyn_is_active(&self.params.band_params[i]) && range_db.abs() > 0.0 {
                let far = BandSpec {
                    gain_db: spec.gain_db + range_db,
                    ..*spec
                };
                let line = Path::new(|p| {
                    for j in 0..POINTS {
                        let x01 = j as f32 / (POINTS - 1) as f32;
                        let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(x01);
                        let db = band_response_db(&far, freq);
                        let pt = to_px(bounds, freq, db.clamp(-DB_MAX - 6.0, DB_MAX + 6.0));
                        if j == 0 {
                            p.move_to(pt);
                        } else {
                            p.line_to(pt);
                        }
                    }
                });
                frame.stroke(
                    &line,
                    Stroke::default()
                        .with_color(Color { a: 0.18, ..color })
                        .with_width(1.0),
                );
            }
        }
    }

    fn draw_nodes(
        &self,
        frame: &mut Frame,
        bounds: Rectangle,
        hover: Option<usize>,
        dragging: Option<usize>,
        selected: Option<usize>,
    ) {
        let num_active = self.params.num_bands.value() as usize;
        for i in 0..num_active {
            let bp = &self.params.band_params[i];
            if !bp.enable.value() {
                continue;
            }
            let color = editor::band_color(i);
            let center = to_px(bounds, bp.freq.value(), node_gain_db(bp));

            let is_hover = hover == Some(i);
            let is_drag = dragging == Some(i);
            let ring_alpha = if is_drag {
                0.6
            } else if is_hover {
                0.45
            } else {
                0.25
            };
            let inner_radius = if is_drag { 4.5 } else { 3.5 };

            // Outer ring
            frame.fill(
                &Path::circle(center, 7.0),
                Fill::from(Color {
                    a: ring_alpha,
                    ..color
                }),
            );
            // Inner dot
            frame.fill(&Path::circle(center, inner_radius), Fill::from(color));

            // The selected band gets a full-opacity ring so it reads as
            // "active" from across the room — the panel below edits it.
            let is_selected = selected == Some(i);
            if is_selected {
                frame.stroke(
                    &Path::circle(center, 9.0),
                    Stroke::default()
                        .with_color(Color {
                            a: 0.95,
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                        })
                        .with_width(1.5),
                );
            }

            // Frequency + gain readout while dragging or while selected.
            //
            // A dynamic band shows the static gain *and* the dynamic part: with
            // Pro-Q's model the knob stays a static gain and the range moves
            // the band away from it, so a label that showed only one of the two
            // would be wrong about the sound in one direction or the other.
            if is_drag || is_selected {
                let dynamic = bp.dyn_enable.value()
                    && FilterType::from_i32(bp.filter_type.value()).has_gain()
                    && bp.dyn_range.value().abs() >= crate::dsp::dynamics::MIN_RANGE_DB;
                let static_gain = bp.gain.value();
                let label = if dynamic {
                    format!(
                        "{:.0} Hz  {:+.1} dB ({:+.1} dyn)",
                        bp.freq.value(),
                        static_gain,
                        self.dyn_meters.read(i)
                    )
                } else {
                    format!("{:.0} Hz  {:+.1} dB", bp.freq.value(), static_gain)
                };
                let text_x = (center.x + 10.0).min(bounds.x + bounds.width - 140.0);
                frame.fill_text(CanvasText {
                    content: label,
                    position: Point::new(text_x, center.y - 16.0),
                    color: Color { a: 0.9, ..color },
                    size: 10.0,
                    ..CanvasText::default()
                });
            }
        }
    }

    /// Draw the display controls: freeze, tilt, range, pre-trace offset.
    fn draw_spectrum_chips(&self, frame: &mut Frame, bounds: Rectangle) {
        let view = &self.taps.view;
        for (chip, rect) in chip_rects(view, bounds) {
            let (bg, fg) = if chip.active(view) {
                (
                    Color::from_rgba(0.55, 0.75, 1.0, 0.85),
                    Color::from_rgba(0.06, 0.06, 0.08, 1.0),
                )
            } else {
                (
                    Color::from_rgba(1.0, 1.0, 1.0, 0.08),
                    Color::from_rgba(1.0, 1.0, 1.0, 0.55),
                )
            };
            frame.fill(
                &Path::rectangle(
                    Point::new(rect.x, rect.y),
                    Size::new(rect.width, rect.height),
                ),
                Fill::from(bg),
            );
            frame.fill_text(CanvasText {
                content: chip.label(view),
                position: Point::new(rect.x + CHIP_PAD, rect.y + 2.0),
                color: fg,
                size: 9.0,
                ..CanvasText::default()
            });
        }
    }

    fn draw(
        &self,
        bounds: Rectangle,
        hover: Option<usize>,
        dragging: Option<usize>,
    ) -> Vec<Geometry> {
        // A device change can move the whole frequency axis; do this before the
        // cache key is computed, because it invalidates every cached point.
        self.refresh_axis();

        // The grid, per-band curves and the combined curve only change when a
        // parameter (or the canvas geometry) does. nih-plug redraws the whole
        // editor every frame with `always_redraw`, but the EQ picture is
        // usually static — so keep those layers and rebuild them on change
        // only. Rebuilding cost the trig-heavy `band_response_db` up to 24 ×
        // 120 times per frame.
        let key = CachedLayers::key_for(&self.params, bounds, self.taps.view.tilted());
        let mut cache = self.cache.borrow_mut();
        if !matches!(&*cache, Some(c) if c.key == key) {
            let mut grid_frame = Frame::new(bounds.size());
            self.draw_grid(&mut grid_frame, bounds);
            let mut curves_frame = Frame::new(bounds.size());
            self.draw_band_curves(&mut curves_frame, bounds);
            self.draw_curve(&mut curves_frame, bounds);
            *cache = Some(CachedLayers {
                key,
                grid: grid_frame.into_geometry(),
                curves: curves_frame.into_geometry(),
            });
        }

        // The spectrum moves continuously; the nodes move while dragging.
        let mut input_frame = Frame::new(bounds.size());
        self.draw_input_spectrum(&mut input_frame, bounds);
        let mut output_frame = Frame::new(bounds.size());
        if self.params.spectrum_post.value() {
            self.draw_output_spectrum(&mut output_frame, bounds);
        }
        let mut nodes_frame = Frame::new(bounds.size());
        self.draw_nodes(&mut nodes_frame, bounds, hover, dragging, self.selected());
        // On top of everything: a chip you cannot see is a chip you cannot
        // click, and a band node sitting over it would make both unusable.
        let mut chips_frame = Frame::new(bounds.size());
        self.draw_spectrum_chips(&mut chips_frame, bounds);

        let cached = cache
            .as_ref()
            .expect("cache was just populated in this same call");
        vec![
            cached.grid.clone(),
            input_frame.into_geometry(),
            output_frame.into_geometry(),
            cached.curves.clone(),
            nodes_frame.into_geometry(),
            chips_frame.into_geometry(),
        ]
    }
}

/// Public handle used by the editor's view.
pub struct Graph {
    program: Arc<GraphProgram>,
}

impl GraphProgram {
    /// Rebuild the spectrum bar positions if the audio thread has published a
    /// different sample rate. Called once per draw, before either overlay: a
    /// stale axis would place every bar at the wrong frequency.
    fn refresh_axis(&self) {
        let rate = self.taps.sample_rate();
        if rate != self.spec_rate.get() {
            *self.spec_x.borrow_mut() = spectrum_bin_x(rate);
            self.spec_rate.set(rate);
            // The curves are cached geometry; a new axis means every cached
            // point is now at the wrong x.
            *self.cache.borrow_mut() = None;
        }
    }
}

impl Graph {
    pub fn new(
        params: Arc<FilterParams>,
        context: Arc<dyn GuiContext>,
        peak_meter: Arc<AtomicF32>,
        taps: Arc<crate::dsp::spectrum::SpectrumTaps>,
        dyn_meters: Arc<crate::dsp::dynamics::BandMeters>,
        patch_state: Arc<std::sync::Mutex<crate::patch::PatchState>>,
        selected: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            // The Cell fields are GUI-thread-only interaction state, but the
            // program is shared through an Arc inside the editor, which
            // declares (unsafely, with justification) Send+Sync. Access stays
            // on the GUI thread — same contract as the slider states.
            #[allow(clippy::arc_with_non_send_sync)]
            program: Arc::new({
                let spec_x = RefCell::new(spectrum_bin_x(taps.sample_rate()));
                let spec_rate = Cell::new(taps.sample_rate());
                GraphProgram {
                    params,
                    context,
                    _peak_meter: peak_meter,
                    taps,
                    dyn_meters,
                    patch_state,
                    ballistics_in: RefCell::new(TraceBallistics::new()),
                    ballistics_out: RefCell::new(TraceBallistics::new()),
                    spec_x,
                    spec_rate,
                    cache: RefCell::new(None),
                    drag: Cell::new(None),
                    hover: Cell::new(None),
                    fine_scroll: Cell::new(false),
                    last_click: Cell::new(None),
                    selected,
                }
            }),
        }
    }

    pub fn view(&self) -> nih_plug_iced::Element<'_, Message> {
        // Parameters animate every frame (smoothers) and nih-plug redraws the
        // whole tree each frame, so a canvas Cache would freeze the drawing.
        // Geometry for ~160 points is cheap to rebuild — bypass the cache.
        //
        // GraphView is likewise rebuilt every frame; all mutable interaction
        // state lives on the shared GraphProgram behind Cell so drags survive
        // the rebuild.
        IcedCanvas::new(GraphView {
            program: self.program.clone(),
        })
        .width(Length::Fill)
        .height(Length::Units(GRAPH_HEIGHT as u16))
        .into()
    }
}

/// Message type used by the editor.
use crate::editor::Message;

#[cfg(test)]
impl Graph {
    /// The canvas exactly as iced would drive it, so end-to-end tests go
    /// through the real event handlers instead of a copy of their logic.
    pub(crate) fn driver(&self) -> GraphView {
        GraphView {
            program: self.program.clone(),
        }
    }
}

/// Wrapper implementing Program (forwarded to the shared GraphProgram).
///
/// Rebuilt every frame — all mutable interaction state lives on the
/// `GraphProgram` behind interior mutability.
pub(crate) struct GraphView {
    program: Arc<GraphProgram>,
}

impl GraphView {
    pub(crate) fn handle_mouse_event(
        &self,
        event: mouse::Event,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        let program = &self.program;

        match event {
            mouse::Event::ButtonPressed(mouse::Button::Left) => {
                let Some(pos) = cursor.position_in(&bounds) else {
                    return (canvas::event::Status::Ignored, None);
                };
                // A chip wins over whatever is underneath it.
                if let Some(chip) = chip_at(pos, &program.taps.view, bounds) {
                    chip.activate(&program.taps.view);
                    return (canvas::event::Status::Captured, None);
                }
                let num_active = program.params.num_bands.value() as usize;
                let nodes = node_positions(&program.params, num_active, bounds);
                match nearest_grabbable_node(pos.x, pos.y, &nodes) {
                    Some(hit) => {
                        // Double-click a node: back to 0 dB (PLAN's "reset to
                        // flat"). iced doesn't hand the canvas a double-click
                        // event, so time the clicks here.
                        let now = Instant::now();
                        let is_double = matches!(
                            program.last_click.get(),
                            Some((band, at))
                                if band == hit.band_index && now.duration_since(at) < DOUBLE_CLICK
                        );
                        if is_double {
                            let gain = &program.params.band_params[hit.band_index].gain;
                            program.set_plain(gain, 0.0);
                            program.select(Some(hit.band_index));
                            program.last_click.set(None);
                            return (canvas::event::Status::Captured, None);
                        }

                        program.select(Some(hit.band_index));
                        program.last_click.set(Some((hit.band_index, now)));
                        program.drag.set(Some(DragState {
                            band: hit.band_index,
                        }));
                        program.hover.set(Some(hit.band_index));
                        (canvas::event::Status::Captured, None)
                    }
                    None => {
                        // Click on empty graph: create a band at that frequency
                        // (0 dB, so the curve does not jump) and immediately
                        // start dragging it — a click-drag also sets the gain.
                        let freq = px_to_freq(pos.x, bounds.width);
                        match program.create_band(freq, 0.0) {
                            Some(band) => {
                                let now = Instant::now();
                                program.select(Some(band));
                                program.drag.set(Some(DragState { band }));
                                program.hover.set(Some(band));
                                program.last_click.set(Some((band, now)));
                                (canvas::event::Status::Captured, None)
                            }
                            None => (canvas::event::Status::Ignored, None),
                        }
                    }
                }
            }

            // Right-click a node: delete that band and compact the rest.
            mouse::Event::ButtonPressed(mouse::Button::Right) => {
                let Some(pos) = cursor.position_in(&bounds) else {
                    return (canvas::event::Status::Ignored, None);
                };
                let num_active = program.params.num_bands.value() as usize;
                let nodes = node_positions(&program.params, num_active, bounds);
                match nearest_grabbable_node(pos.x, pos.y, &nodes) {
                    Some(hit) => {
                        program.delete_and_follow(hit.band_index, num_active);
                        program.drag.set(None);
                        program.hover.set(None);
                        program.last_click.set(None);
                        (canvas::event::Status::Captured, None)
                    }
                    None => (canvas::event::Status::Ignored, None),
                }
            }

            mouse::Event::CursorMoved { position } => {
                // position is window-absolute; convert to canvas-local.
                let local_x = position.x - bounds.x;
                let local_y = position.y - bounds.y;

                if let Some(state) = program.drag.get() {
                    // Move the grabbed node: x → frequency, y → gain.
                    let freq = px_to_freq(local_x, bounds.width);
                    let gain_db = y_to_db((local_y / bounds.height).clamp(0.0, 1.0));
                    let bp = &program.params.band_params[state.band];
                    program.set_plain(&bp.freq, freq);
                    // A cut has no gain to drag; moving vertically would write
                    // a value nothing reads.
                    if FilterType::from_i32(bp.filter_type.value()).has_gain() {
                        program.set_plain(&bp.gain, gain_db);
                    }
                    (canvas::event::Status::Captured, None)
                } else {
                    // Hover highlight when near a node
                    let num_active = program.params.num_bands.value() as usize;
                    let nodes = node_positions(&program.params, num_active, bounds);
                    let new_hover =
                        if chip_at(Point::new(local_x, local_y), &program.taps.view, bounds)
                            .is_some()
                        {
                            // No band highlight under a chip: it would suggest a
                            // node is grabbable where the click goes to the chip.
                            None
                        } else {
                            nearest_grabbable_node(local_x, local_y, &nodes)
                                .map(|hit| hit.band_index)
                        };
                    if new_hover != program.hover.get() {
                        program.hover.set(new_hover);
                    }
                    (canvas::event::Status::Ignored, None)
                }
            }

            mouse::Event::ButtonReleased(mouse::Button::Left) => {
                if program.drag.get().is_some() {
                    program.drag.set(None);
                    (canvas::event::Status::Captured, None)
                } else {
                    (canvas::event::Status::Ignored, None)
                }
            }

            mouse::Event::WheelScrolled { delta } => {
                let Some(pos) = cursor.position_in(&bounds) else {
                    return (canvas::event::Status::Ignored, None);
                };
                let num_active = program.params.num_bands.value() as usize;
                let nodes = node_positions(&program.params, num_active, bounds);
                let Some(hit) = nearest_grabbable_node(pos.x, pos.y, &nodes) else {
                    return (canvas::event::Status::Ignored, None);
                };

                let lines_y = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => y,
                    mouse::ScrollDelta::Pixels { y, .. } => y / 40.0,
                };
                if lines_y == 0.0 {
                    return (canvas::event::Status::Ignored, None);
                }

                // A cut has no Q; scrolling it steps the slope instead, which
                // is the control that actually shapes it.
                let bp = &program.params.band_params[hit.band_index];
                let ftype = FilterType::from_i32(bp.filter_type.value());
                if ftype.is_cut() {
                    let current = crate::Slope::from_i32(bp.slope.value()) as i32;
                    let next = (current + lines_y.signum() as i32)
                        .clamp(0, crate::Slope::ALL.len() as i32 - 1);
                    program.set_gesture(&bp.slope, next);
                    return (canvas::event::Status::Captured, None);
                }

                let fine = program.fine_scroll.get();
                let step = scroll_q_step(fine);
                let q_param = &program.params.band_params[hit.band_index].q;
                let q_range = q_param.range();
                let normalized = q_range.normalize(q_param.value());
                let new_normalized = (normalized + lines_y.signum() * step).clamp(0.0, 1.0);
                program.set_q_normalized(hit.band_index, new_normalized);
                (canvas::event::Status::Captured, None)
            }

            _ => (canvas::event::Status::Ignored, None),
        }
    }

    /// Handle a key press on the selected band.
    ///
    /// | Key | Action |
    /// |---|---|
    /// | ← → | Frequency by a semitone (Shift: an octave) |
    /// | ↑ ↓ | Gain by 0.1 dB (Shift: 1 dB) |
    /// | Tab / Shift+Tab | Select the next / previous band |
    /// | Backspace / Delete | Delete the selected band |
    /// | Esc | Clear the selection |
    pub(crate) fn handle_keyboard_event(&self, event: keyboard::Event) -> canvas::event::Status {
        let program = &self.program;

        // The canvas receives raw keyboard events, including modifier changes,
        // so Alt can gate fine Q scrolling and Shift can change the nudge step.
        if let keyboard::Event::ModifiersChanged(modifiers) = event {
            program.fine_scroll.set(modifiers.alt());
            return canvas::event::Status::Ignored;
        }

        let keyboard::Event::KeyPressed {
            key_code,
            modifiers,
        } = event
        else {
            return canvas::event::Status::Ignored;
        };

        use keyboard::KeyCode;

        // Undo/redo live before the selection check: undo must work with no
        // band selected, which is exactly when you have just deleted one.
        // The canvas owns the keyboard, so the work happens here rather than
        // being forwarded as a message — `handle_keyboard_event` returns only a
        // status, and the canvas already holds the params, the context and the
        // shared history.
        if modifiers.command() || modifiers.control() {
            let handled = match key_code {
                KeyCode::Z if modifiers.shift() => {
                    let mut state = program.lock_patch_state();
                    state.redo(&program.params, program.context.as_ref())
                }
                KeyCode::Z => {
                    let mut state = program.lock_patch_state();
                    state.undo(&program.params, program.context.as_ref())
                }
                KeyCode::Y => {
                    let mut state = program.lock_patch_state();
                    state.redo(&program.params, program.context.as_ref())
                }
                _ => false,
            };
            if handled {
                return canvas::event::Status::Captured;
            }
        }

        let num_active = program.params.num_bands.value() as usize;
        let Some(band) = program.selected().filter(|&b| b < num_active) else {
            return canvas::event::Status::Ignored;
        };

        match key_code {
            KeyCode::Left | KeyCode::Right => {
                let semitones = if modifiers.shift() {
                    NUDGE_COARSE_SEMITONES
                } else {
                    NUDGE_SEMITONES
                };
                let factor = 2f32.powf(semitones / 12.0);
                let param = &program.params.band_params[band].freq;
                let next = if key_code == KeyCode::Left {
                    param.value() / factor
                } else {
                    param.value() * factor
                };
                let next = next.clamp(FREQ_MIN, FREQ_MAX);
                program.set_gesture(param, next);
                canvas::event::Status::Captured
            }
            KeyCode::Up | KeyCode::Down => {
                let step = if modifiers.shift() {
                    NUDGE_COARSE_DB
                } else {
                    NUDGE_DB
                };
                let delta = if key_code == KeyCode::Up { step } else { -step };
                let param = &program.params.band_params[band].gain;
                let next = (param.value() + delta).clamp(-DB_MAX, DB_MAX);
                program.set_gesture(param, next);
                canvas::event::Status::Captured
            }
            KeyCode::Tab => {
                let count = num_active.max(1);
                let next = if modifiers.shift() {
                    (band + count - 1) % count
                } else {
                    (band + 1) % count
                };
                program.select(Some(next));
                canvas::event::Status::Captured
            }
            KeyCode::Backspace | KeyCode::Delete => {
                program.delete_and_follow(band, num_active);
                program.drag.set(None);
                program.hover.set(None);
                program.last_click.set(None);
                canvas::event::Status::Captured
            }
            KeyCode::Escape => {
                program.select(None);
                canvas::event::Status::Captured
            }
            _ => canvas::event::Status::Ignored,
        }
    }
}

impl Program<Message> for GraphView {
    fn update(
        &mut self,
        event: canvas::Event,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        match event {
            canvas::Event::Mouse(mouse_event) => {
                self.handle_mouse_event(mouse_event, bounds, cursor)
            }
            canvas::Event::Keyboard(keyboard_event) => {
                let status = self.handle_keyboard_event(keyboard_event);
                (status, None)
            }
        }
    }

    fn draw(&self, bounds: Rectangle, _cursor: Cursor) -> Vec<Geometry> {
        let hover = self.program.hover.get();
        let dragging = self.program.drag.get().map(|s| s.band);
        self.program.draw(bounds, hover, dragging)
    }

    fn mouse_interaction(
        &self,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> nih_plug_iced::mouse::Interaction {
        let Some(pos) = cursor.position_in(&bounds) else {
            return nih_plug_iced::mouse::Interaction::default();
        };

        if self.program.drag.get().is_some() {
            return nih_plug_iced::mouse::Interaction::Grabbing;
        }

        // A chip is a button, not empty graph: a crosshair over it would say
        // "a click here creates a band", which is exactly what it does not do.
        if chip_at(pos, &self.program.taps.view, bounds).is_some() {
            return nih_plug_iced::mouse::Interaction::Pointer;
        }

        let num_active = self.program.params.num_bands.value() as usize;
        let nodes = node_positions(&self.program.params, num_active, bounds);
        match nearest_grabbable_node(pos.x, pos.y, &nodes) {
            Some(_) => nih_plug_iced::mouse::Interaction::Grab,
            // Crosshair over empty space: this is where a click creates a band.
            None => nih_plug_iced::mouse::Interaction::Crosshair,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Headless tests for the canvas draw path.
    //!
    //! The editor cannot be opened in CI, but `Program::draw()` is pure: it
    //! turns parameters into a `Vec<Geometry>`, and `Geometry::into_primitive()`
    //! exposes the meshes and text runs iced will rasterize. These tests drive
    //! the real `GraphProgram::draw` and assert on what actually comes out, so a
    //! layer that stops being drawn, a curve drawn at the wrong gain, or a stale
    //! cache is a test failure rather than something only an eyeball would catch.

    use super::*;
    use crate::dsp::spectrum::SpectrumTaps;
    use crate::test_support::{
        test_params, test_params_with_dynamics, test_params_with_post_spectrum, NullGuiContext,
        RecordingGuiContext,
    };
    use atomic_float::AtomicF32;
    use iced_graphics::Primitive;
    use nih_plug::prelude::util;
    use nih_plug::prelude::Param;

    // Layer indices, named. `draw()` returns one geometry per layer in render
    // order; a bare integer here once broke three tests when a layer was
    // inserted, so the indices are spelled out in exactly one place.
    const GRID: usize = 0;
    const INPUT_SPECTRUM: usize = 1;
    const OUTPUT_SPECTRUM: usize = 2;
    const CURVES: usize = 3;
    const NODES: usize = 4;
    const CHIPS: usize = 5;

    /// The default display floor, as the range selector starts on it.
    const FLOOR: f32 = crate::dsp::spectrum::DB_RANGES[crate::dsp::spectrum::DEFAULT_DB_RANGE];

    fn bounds() -> Rectangle {
        Rectangle {
            x: 0.0,
            y: 0.0,
            width: 900.0,
            height: GRAPH_HEIGHT,
        }
    }

    #[allow(clippy::arc_with_non_send_sync)] // GUI-thread-only, same contract as Graph::new
    fn test_program_with(params: Arc<FilterParams>, context: Arc<dyn GuiContext>) -> GraphProgram {
        test_program_with_taps(params, context, Arc::new(SpectrumTaps::new()))
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn test_program_with_taps(
        params: Arc<FilterParams>,
        context: Arc<dyn GuiContext>,
        taps: Arc<SpectrumTaps>,
    ) -> GraphProgram {
        let patch_state = Arc::new(std::sync::Mutex::new(crate::patch::PatchState::new(
            &params,
        )));
        test_program_with_history(params, context, taps, patch_state)
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn test_program_with_history(
        params: Arc<FilterParams>,
        context: Arc<dyn GuiContext>,
        taps: Arc<SpectrumTaps>,
        patch_state: Arc<std::sync::Mutex<crate::patch::PatchState>>,
    ) -> GraphProgram {
        let rate = taps.sample_rate();
        GraphProgram {
            params,
            context,
            _peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            taps,
            dyn_meters: Arc::new(crate::dsp::dynamics::BandMeters::new()),
            patch_state,
            ballistics_in: RefCell::new(TraceBallistics::new()),
            ballistics_out: RefCell::new(TraceBallistics::new()),
            spec_x: RefCell::new(spectrum_bin_x(rate)),
            spec_rate: Cell::new(rate),
            cache: RefCell::new(None),
            drag: Cell::new(None),
            hover: Cell::new(None),
            fine_scroll: Cell::new(false),
            last_click: Cell::new(None),
            selected: Arc::new(AtomicUsize::new(selection::NONE)),
        }
    }

    fn test_program(bands: &[(f32, f32, f32, FilterType)]) -> GraphProgram {
        test_program_with(test_params(bands), Arc::new(NullGuiContext))
    }

    #[test]
    fn the_axis_follows_the_published_sample_rate() {
        // The wiring, not the arithmetic: `bin_edge_hz` is unit-tested, but it
        // only helps if the canvas actually reads the rate the audio thread
        // published instead of a constant. A device change from 48 to 96 kHz
        // must move every bar one octave to the right.
        let taps = Arc::new(SpectrumTaps::new());
        taps.set_sample_rate(48_000.0);
        let program = test_program_with_taps(
            test_params(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]),
            Arc::new(NullGuiContext),
            taps.clone(),
        );

        let mid = BINS / 2;
        let before = program.spec_x.borrow()[mid];
        assert_eq!(program.spec_rate.get(), 48_000.0);

        taps.set_sample_rate(96_000.0);
        // The draw path refreshes the axis once per frame; call the same hook.
        program.refresh_axis();

        let after = program.spec_x.borrow()[mid];
        assert_eq!(program.spec_rate.get(), 96_000.0);
        assert!(
            after > before + 0.05,
            "band {mid} sits at x={before} at 48 kHz and x={after} at 96 kHz — \
             one octave is a measurable shift on a log axis"
        );

        // And it does not rebuild when nothing changed.
        let cached = program.spec_x.borrow().clone();
        program.refresh_axis();
        assert_eq!(*program.spec_x.borrow(), cached);
    }

    /// Every vertex of every mesh in a primitive tree, as
    /// `(x, y, r, g, b, a)` — linear-space colour, exactly what the renderer
    /// receives.
    fn vertices(primitive: &Primitive, out: &mut Vec<[f32; 6]>) {
        match primitive {
            Primitive::Mesh2D { buffers, .. } => {
                for v in &buffers.vertices {
                    out.push([
                        v.position[0],
                        v.position[1],
                        v.color[0],
                        v.color[1],
                        v.color[2],
                        v.color[3],
                    ]);
                }
            }
            Primitive::Group { primitives } => {
                for p in primitives {
                    vertices(p, out);
                }
            }
            Primitive::Clip { content, .. } | Primitive::Translate { content, .. } => {
                vertices(content, out)
            }
            Primitive::Cached { cache } => vertices(cache, out),
            _ => {}
        }
    }

    /// Every triangle in a primitive tree, expanded from the index buffer so
    /// the dump can be rasterized without reimplementing iced's mesh handling.
    fn triangles(primitive: &Primitive, out: &mut Vec<[[f32; 6]; 3]>) {
        match primitive {
            Primitive::Mesh2D { buffers, .. } => {
                for idx in buffers.indices.as_chunks::<3>().0 {
                    let mut tri = [[0.0f32; 6]; 3];
                    for (k, &i) in idx.iter().enumerate() {
                        let v = buffers.vertices[i as usize];
                        tri[k] = [
                            v.position[0],
                            v.position[1],
                            v.color[0],
                            v.color[1],
                            v.color[2],
                            v.color[3],
                        ];
                    }
                    out.push(tri);
                }
            }
            Primitive::Group { primitives } => {
                for p in primitives {
                    triangles(p, out);
                }
            }
            Primitive::Clip { content, .. } | Primitive::Translate { content, .. } => {
                triangles(content, out)
            }
            Primitive::Cached { cache } => triangles(cache, out),
            _ => {}
        }
    }

    /// Every text run in a primitive tree.
    fn texts(primitive: &Primitive, out: &mut Vec<String>) {
        match primitive {
            Primitive::Text { content, .. } => out.push(content.clone()),
            Primitive::Group { primitives } => {
                for p in primitives {
                    texts(p, out);
                }
            }
            Primitive::Clip { content, .. } | Primitive::Translate { content, .. } => {
                texts(content, out)
            }
            Primitive::Cached { cache } => texts(cache, out),
            _ => {}
        }
    }

    fn layer_vertices(layers: &[Geometry], index: usize) -> Vec<[f32; 6]> {
        let mut out = Vec::new();
        vertices(&layers[index].clone().into_primitive(), &mut out);
        out
    }

    #[test]
    fn draws_grid_spectrum_curves_and_nodes() {
        let program = test_program(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]);
        let layers = program.draw(bounds(), None, None);

        // Grid | input spectrum | output spectrum | curves | nodes — the order
        // the renderer composites.
        assert_eq!(layers.len(), 6, "expected six canvas layers");

        let mut labels = Vec::new();
        texts(&layers[0].clone().into_primitive(), &mut labels);
        for expected in ["20", "10k", "-24", "+24"] {
            assert!(
                labels.iter().any(|l| l == expected),
                "grid is missing the {expected} label; got {labels:?}"
            );
        }

        // Regression guard: `frame.fill()` on a `Path::line` tessellates to
        // nothing, so grid lines silently disappeared while their labels kept
        // drawing. The grid layer must contain mesh geometry, not just text.
        assert!(
            !layer_vertices(&layers, GRID).is_empty(),
            "grid layer has text but no mesh — grid lines are not being drawn"
        );
        assert!(
            !layer_vertices(&layers, INPUT_SPECTRUM).is_empty(),
            "input spectrum layer has no geometry"
        );
        assert!(
            !layer_vertices(&layers, OUTPUT_SPECTRUM).is_empty(),
            "output spectrum layer has no geometry"
        );
        assert!(
            !layer_vertices(&layers, CURVES).is_empty(),
            "curve layer has no geometry"
        );
        assert!(
            !layer_vertices(&layers, NODES).is_empty(),
            "node layer has no geometry for an enabled band"
        );
    }

    #[test]
    fn nodes_layer_is_empty_without_bands() {
        let program = test_program(&[]);
        let layers = program.draw(bounds(), None, None);
        assert!(
            layer_vertices(&layers, NODES).is_empty(),
            "no bands should draw no nodes"
        );
    }

    /// A synthetic frame, as the analyzer would publish it.
    fn frame_with(db: &[f32; BINS], seq: u64) -> SpectrumFrame {
        let mut f = SpectrumFrame::silent();
        f.db = *db;
        f.seq = seq;
        f
    }

    /// A flat synthetic frame at `db`.
    fn frame(db: f32, seq: u64) -> SpectrumFrame {
        frame_with(&[db; BINS], seq)
    }

    /// Topmost vertex of a layer — for a flat spectrum that is where the
    /// curve was drawn.
    fn top_y(layers: &[Geometry], index: usize) -> f32 {
        layer_vertices(layers, index)
            .iter()
            .map(|v| v[1])
            .fold(f32::INFINITY, f32::min)
    }

    #[test]
    fn input_and_output_overlays_follow_their_own_taps() {
        let b = bounds();
        let taps = Arc::new(SpectrumTaps::new());
        // A flat 0 dB input with a 20 dB-quieter output: a broadband cut.
        taps.input.publish(frame(0.0, 1));
        taps.output.publish(frame(-20.0, 1));

        let program =
            test_program_with_taps(test_params(&[]), Arc::new(NullGuiContext), taps.clone());

        // The ballistics need several frames to converge (rise is 0.55/frame),
        // so draw a handful before measuring.
        let mut layers = program.draw(b, None, None);
        for _ in 0..40 {
            layers = program.draw(b, None, None);
        }

        let input_y = top_y(&layers, INPUT_SPECTRUM);
        let output_y = top_y(&layers, OUTPUT_SPECTRUM);

        // y grows downward: the quieter output must sit below the input.
        assert!(
            output_y > input_y + 5.0,
            "output overlay is not below the input overlay \
             (input y {input_y:.1}, output y {output_y:.1})"
        );
        // And each must land where the dB→y mapping says, not merely in the
        // right order — this catches both curves being drawn from one tap.
        assert!(
            (input_y - spectrum_y(b, 0.0, FLOOR)).abs() < 2.0,
            "input overlay is at y {input_y:.1}, expected {:.1}",
            spectrum_y(b, 0.0, FLOOR)
        );
        assert!(
            (output_y - spectrum_y(b, -20.0, FLOOR)).abs() < 2.0,
            "output overlay is at y {output_y:.1}, expected {:.1}",
            spectrum_y(b, -20.0, FLOOR)
        );
    }

    #[test]
    fn output_overlay_is_gated_by_its_parameter() {
        let b = bounds();
        let taps = Arc::new(SpectrumTaps::new());
        taps.input.publish(frame(0.0, 1));
        taps.output.publish(frame(-12.0, 1));
        let params = test_params_with_post_spectrum(&[], false);
        let program = test_program_with_taps(params, Arc::new(NullGuiContext), taps);
        let layers = program.draw(b, None, None);

        // The layer stays in the list (indices are a render order contract)
        // but must be empty when the overlay is off.
        assert_eq!(layers.len(), 6);
        assert!(
            layer_vertices(&layers, OUTPUT_SPECTRUM).is_empty(),
            "output overlay was drawn although the parameter is off"
        );
        assert!(
            !layer_vertices(&layers, INPUT_SPECTRUM).is_empty(),
            "the input spectrum must keep drawing when the overlay is off"
        );
    }

    /// A dynamic band's readout must carry both numbers.
    ///
    /// With the Pro-Q model the knob is a static gain and the range moves the
    /// band away from it, so a label showing only one of the two says the wrong
    /// thing about the sound in one direction or the other.
    #[test]
    fn a_dynamic_band_labels_its_static_gain_and_the_live_part() {
        let params = test_params_with_dynamics(&[(1_000.0, -12.0, 1.0, FilterType::Bell)]);
        let program = test_program_with(params, Arc::new(NullGuiContext));
        program.select(Some(0));
        program.dyn_meters.publish(0, -7.5);

        let layers = program.draw(bounds(), None, None);
        let mut labels = Vec::new();
        texts(&layers[NODES].clone().into_primitive(), &mut labels);

        let label = labels
            .iter()
            .find(|l| l.contains("1000 Hz"))
            .unwrap_or_else(|| panic!("no readout drawn; got {labels:?}"));
        // The fixture makes the band dynamic by moving its gain into the range,
        // so the static gain is 0 and the live part is what the meter says.
        assert!(
            label.contains("+0.0 dB"),
            "the static gain is missing from {label:?}"
        );
        assert!(
            label.contains("(-7.5 dyn)"),
            "the live dynamic part is missing from {label:?}"
        );
    }

    /// Cmd/Ctrl+Z on the canvas must reach the shared history and write the
    /// previous values back to the host. The canvas owns the keyboard, so this
    /// is the only path the shortcut can take.
    #[test]
    fn command_z_undoes_through_the_canvas() {
        use crate::patch::{Patch, PatchState, SETTLE};
        use std::time::Instant;

        let before = test_params(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);
        // The "live" parameters are one edit ahead of the history.
        let live = test_params(&[(1_000.0, 6.0, 1.0, FilterType::Bell)]);
        let ctx = Arc::new(RecordingGuiContext::default());

        let state = Arc::new(std::sync::Mutex::new(PatchState::new(&before)));
        {
            // Settle one edit into the history: A -> B.
            let mut state = state.lock().unwrap();
            let t0 = Instant::now();
            let edited = Patch::capture(&live);
            state.history.observe(&edited, t0);
            state.history.observe(&edited, t0 + SETTLE);
            assert!(state.history.can_undo(), "the fixture needs a step");
        }

        let program = {
            let ctx_dyn: Arc<dyn GuiContext> = ctx.clone();
            test_program_with_history(live, ctx_dyn, Arc::new(SpectrumTaps::new()), state.clone())
        };
        // GUI-thread-only, same contract as Graph::new.
        #[allow(clippy::arc_with_non_send_sync)]
        let view = GraphView {
            program: Arc::new(program),
        };
        let status = view.handle_keyboard_event(keyboard::Event::KeyPressed {
            key_code: keyboard::KeyCode::Z,
            modifiers: keyboard::Modifiers::COMMAND,
        });

        assert_eq!(status, canvas::event::Status::Captured);
        assert_eq!(
            ctx.writes().len(),
            1,
            "undo should write exactly the parameter that changed"
        );
        // The gain goes back to 0 dB.
        let gain = &test_params(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]).band_params[0].gain;
        assert!((ctx.writes()[0] - gain.preview_normalized(0.0)).abs() < 1e-6);

        // Redo puts it back.
        let program_redo = {
            let ctx_dyn: Arc<dyn GuiContext> = ctx.clone();
            test_program_with_history(
                test_params(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]),
                ctx_dyn,
                Arc::new(SpectrumTaps::new()),
                state,
            )
        };
        #[allow(clippy::arc_with_non_send_sync)]
        let view = GraphView {
            program: Arc::new(program_redo),
        };
        let status = view.handle_keyboard_event(keyboard::Event::KeyPressed {
            key_code: keyboard::KeyCode::Z,
            modifiers: keyboard::Modifiers::COMMAND | keyboard::Modifiers::SHIFT,
        });
        assert_eq!(status, canvas::event::Status::Captured);
        assert_eq!(ctx.writes().len(), 2, "redo should write the gain back");
    }

    #[test]
    fn command_z_with_nothing_to_undo_is_left_for_the_host() {
        // Falling through matters: the host has its own undo, and swallowing
        // the key with an empty stack would break it.
        let program = test_program(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);
        // GUI-thread-only, same contract as Graph::new.
        #[allow(clippy::arc_with_non_send_sync)]
        let view = GraphView {
            program: Arc::new(program),
        };
        let status = view.handle_keyboard_event(keyboard::Event::KeyPressed {
            key_code: keyboard::KeyCode::Z,
            modifiers: keyboard::Modifiers::COMMAND,
        });
        assert_eq!(status, canvas::event::Status::Ignored);
    }

    #[test]
    fn a_static_band_ignores_the_meter() {
        // Same meter value, dynamics off: the label must be the knob's value.
        let params = test_params(&[(1_000.0, -12.0, 1.0, FilterType::Bell)]);
        let program = test_program_with(params, Arc::new(NullGuiContext));
        program.select(Some(0));
        program.dyn_meters.publish(0, -7.5);

        let layers = program.draw(bounds(), None, None);
        let mut labels = Vec::new();
        texts(&layers[NODES].clone().into_primitive(), &mut labels);

        assert!(
            labels.iter().any(|l| l.contains("-12.0 dB")),
            "a static band should still report its own gain, got {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l.contains("dyn")),
            "a static band must not be labelled dynamic: {labels:?}"
        );
    }

    #[test]
    fn curve_peak_lands_on_the_band_gain() {
        // A single +12 dB bell at 1 kHz: the drawn curve must pass through
        // (+12 dB, 1 kHz) in pixel space. This is what catches an inverted y
        // axis, a stale cache or a curve built from the wrong values.
        let program = test_program(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]);
        let b = bounds();
        let layers = program.draw(b, None, None);
        let verts = layer_vertices(&layers, CURVES);

        let expected_x = b.x + freq_to_x(1_000.0) * b.width;
        let expected_y = b.y + db_to_y(12.0) * b.height;

        let nearest = verts
            .iter()
            .map(|v| ((v[0] - expected_x).abs(), (v[1] - expected_y).abs()))
            .filter(|(dx, _)| *dx < 3.0)
            .map(|(_, dy)| dy)
            .fold(f32::INFINITY, f32::min);

        assert!(
            nearest < 3.0,
            "curve does not pass through the +12 dB point at 1 kHz: \
             closest vertex within 3 px horizontally is {nearest:.1} px away vertically"
        );
    }

    #[test]
    fn cache_reuses_layers_until_a_parameter_changes() {
        let program = test_program(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]);
        let b = bounds();

        // Two draws with unchanged params must reuse the cached layers and
        // produce byte-identical geometry.
        let first = layer_vertices(&program.draw(b, None, None), CURVES);
        let second = layer_vertices(&program.draw(b, None, None), CURVES);
        assert_eq!(first, second, "unchanged params must redraw identically");

        // A different gain has to produce a different curve: if the cache key
        // ever stops covering the band values, this is what fails.
        let flipped = test_program(&[(1_000.0, -12.0, 1.0, FilterType::Bell)]);
        let third = layer_vertices(&flipped.draw(b, None, None), CURVES);
        assert_ne!(
            first, third,
            "curve layer does not respond to the band gain"
        );
    }

    #[test]
    fn cache_key_tracks_params_and_bounds() {
        let b = bounds();
        // The curve is tilted along with the spectrum, so the tilt is part of
        // what the cached geometry depends on.
        let key = |params: &FilterParams, b: Rectangle| CachedLayers::key_for(params, b, false);

        let base = key(&test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]), b);
        assert_eq!(
            base,
            key(&test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]), b),
            "identical params must produce the same key"
        );
        assert_ne!(
            base,
            key(&test_params(&[(2_000.0, 12.0, 1.0, FilterType::Bell)]), b),
            "freq not tracked"
        );
        assert_ne!(
            base,
            key(&test_params(&[(1_000.0, 12.0, 3.0, FilterType::Bell)]), b),
            "q not tracked"
        );
        assert_ne!(
            base,
            key(
                &test_params(&[(1_000.0, 12.0, 1.0, FilterType::LowShelf)]),
                b
            ),
            "filter type not tracked"
        );
        assert_ne!(
            CachedLayers::key_for(
                &crate::test_support::test_params_with_slopes(&[(
                    1_000.0,
                    0.0,
                    0.707,
                    FilterType::LowCut,
                    crate::Slope::Db12
                )]),
                b,
                false
            ),
            CachedLayers::key_for(
                &crate::test_support::test_params_with_slopes(&[(
                    1_000.0,
                    0.0,
                    0.707,
                    FilterType::LowCut,
                    crate::Slope::Db48
                )]),
                b,
                false
            ),
            "cut slope not tracked — the drawn curve would go stale"
        );
        assert_ne!(base, key(&test_params(&[]), b), "band count not tracked");
        assert_ne!(
            base,
            CachedLayers::key_for(
                &test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]),
                b,
                true
            ),
            "tilt not tracked — the tilted curve would be drawn from a cached flat one"
        );

        let mut resized = b;
        resized.width += 1.0;
        assert_ne!(
            base,
            key(
                &test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]),
                resized
            ),
            "canvas bounds not tracked"
        );
    }

    fn click_at(
        view: &GraphView,
        b: Rectangle,
        x: f32,
        y: f32,
        button: mouse::Button,
    ) -> canvas::event::Status {
        let cursor = Cursor::Available(Point::new(x, y));
        let (status, _) = view.handle_mouse_event(mouse::Event::ButtonPressed(button), b, cursor);
        status
    }

    #[test]
    fn clicking_empty_graph_creates_a_band_at_that_frequency() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let (view, program) = {
            #[allow(clippy::arc_with_non_send_sync)]
            let p = Arc::new(test_program_with(test_params(&[]), ctx.clone()));
            (GraphView { program: p.clone() }, p)
        };
        let b = bounds();
        let x = b.x + freq_to_x(1_000.0) * b.width;
        let y = b.y + b.height * 0.5;

        assert_eq!(
            click_at(&view, b, x, y, mouse::Button::Left),
            canvas::event::Status::Captured
        );

        let writes = ctx.writes();
        assert_eq!(
            writes.len(),
            7,
            "6 band params (freq, gain, q, type, slope, enable) + the band count"
        );
        assert!(
            (writes[5] - 1.0).abs() < 1e-6,
            "created band must be enabled"
        );
        assert!(
            (writes[6] - 1.0 / MAX_BANDS as f32).abs() < 1e-4,
            "band count must be raised to 1, got normalized {}",
            writes[6]
        );
        let expected_freq = program.params.band_params[0]
            .freq
            .preview_normalized(1_000.0);
        assert!(
            (writes[0] - expected_freq).abs() < 2e-3,
            "band should be created at the click frequency"
        );
        assert!(
            program.drag.get().is_some(),
            "a freshly created band should be grabbed so the same click-drag sets its gain"
        );
    }

    #[test]
    fn right_clicking_a_node_deletes_it_and_compacts_the_tail() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let (view, program) = {
            #[allow(clippy::arc_with_non_send_sync)]
            let p = Arc::new(test_program_with(
                test_params(&[
                    (1_000.0, 6.0, 1.0, FilterType::Bell),
                    (5_000.0, -6.0, 1.0, FilterType::Bell),
                ]),
                ctx.clone(),
            ));
            (GraphView { program: p.clone() }, p)
        };
        let b = bounds();
        let x = b.x + freq_to_x(5_000.0) * b.width;
        let y = b.y + db_to_y(-6.0) * b.height;

        assert_eq!(
            click_at(&view, b, x, y, mouse::Button::Right),
            canvas::event::Status::Captured
        );

        let writes = ctx.writes();
        assert_eq!(
            writes.len(),
            1,
            "deleting the last band only lowers the count"
        );
        assert!(
            (writes[0] - 1.0 / MAX_BANDS as f32).abs() < 1e-4,
            "band count must drop to 1"
        );
        assert!(program.drag.get().is_none());
    }

    #[test]
    fn deleting_a_middle_band_copies_the_tail_down_before_lowering_the_count() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let program = test_program_with(
            test_params(&[
                (100.0, 1.0, 1.0, FilterType::Bell),
                (1_000.0, 2.0, 1.0, FilterType::Bell),
                (5_000.0, 3.0, 1.0, FilterType::Bell),
            ]),
            ctx.clone(),
        );
        program.delete_band(1);

        let writes = ctx.writes();
        assert_eq!(writes.len(), 7, "one band copied (6 params) + the count");
        // The copied frequency must be the third band's 5 kHz, not the deleted one.
        let expected = program.params.band_params[1]
            .freq
            .preview_normalized(5_000.0);
        assert!(
            (writes[0] - expected).abs() < 1e-4,
            "tail band was not copied into the freed slot"
        );
        assert!(
            (writes[6] - 2.0 / MAX_BANDS as f32).abs() < 1e-4,
            "band count must drop to 2"
        );
    }

    #[test]
    fn double_clicking_a_node_resets_its_gain() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let (view, _program) = {
            #[allow(clippy::arc_with_non_send_sync)]
            let p = Arc::new(test_program_with(
                test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]),
                ctx.clone(),
            ));
            (GraphView { program: p.clone() }, p)
        };
        let b = bounds();
        let x = b.x + freq_to_x(1_000.0) * b.width;
        let y = b.y + db_to_y(12.0) * b.height;

        assert_eq!(
            click_at(&view, b, x, y, mouse::Button::Left),
            canvas::event::Status::Captured
        );
        let before = ctx.writes().len();
        assert_eq!(before, 0, "a single click on a node only starts a drag");

        assert_eq!(
            click_at(&view, b, x, y, mouse::Button::Left),
            canvas::event::Status::Captured
        );
        let writes = ctx.writes();
        assert_eq!(writes.len(), 1, "double-click writes exactly the gain");
        assert!(
            (writes[0] - 0.5).abs() < 1e-6,
            "gain must be reset to 0 dB (normalized 0.5 in a ±24 dB range)"
        );
    }

    #[test]
    fn creating_a_band_when_the_bank_is_full_is_a_no_op() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let full: Vec<(f32, f32, f32, FilterType)> = (0..MAX_BANDS)
            .map(|i| (100.0 + i as f32 * 100.0, 0.0, 1.0, FilterType::Bell))
            .collect();
        let program = test_program_with(test_params(&full), ctx.clone());
        assert!(program.create_band(1_000.0, 0.0).is_none());
        assert!(ctx.writes().is_empty(), "nothing may be written when full");
    }

    /// Not an assertion — dumps the layers as JSON for visual inspection
    /// (`FILTER_CANVAS_DUMP=/tmp/layers.json cargo test -- --nocapture`).
    #[test]
    fn dump_layers_for_visual_inspection() {
        let Some(path) = std::env::var_os("FILTER_CANVAS_DUMP") else {
            return;
        };
        let bands = [
            (100.0, 4.0, 1.0, FilterType::LowShelf),
            (1_000.0, 12.0, 1.6, FilterType::Bell),
            (6_000.0, -8.0, 2.0, FilterType::Bell),
        ];
        let params = test_params(&bands);

        // Give both taps a plausible signal so the dump shows what the overlay
        // looks like: a pink-ish input, and the same signal with this curve
        // applied. Sine-free and deterministic, so the dump is diffable.
        let taps = Arc::new(SpectrumTaps::new());
        let specs = band_specs(&params, bands.len());
        let mut input = [0.0f32; BINS];
        let mut output = [0.0f32; BINS];
        for i in 0..BINS {
            // Same band→frequency mapping the canvas uses to place the bar, so
            // the dump's geometry is self-consistent (and measurable).
            let freq = crate::dsp::spectrum::bin_edge_hz(i, taps.sample_rate());
            let level = -18.0 - 12.0 * (i as f32 / BINS as f32);
            input[i] = level;
            let delta: f32 = specs
                .iter()
                .take(bands.len())
                .map(|s| band_response_db(s, freq))
                .sum();
            output[i] = level + delta;
        }
        taps.input.publish(frame_with(&input, 1));
        taps.output.publish(frame_with(&output, 1));

        let program = test_program_with_taps(params, Arc::new(NullGuiContext), taps);
        let b = bounds();
        // Let the ballistics settle before dumping, otherwise the dump shows
        // curves on their way up from the floor.
        let mut layers = program.draw(b, None, None);
        for _ in 0..40 {
            layers = program.draw(b, None, None);
        }

        let mut json = String::from("{\n");
        json += &format!(
            "  \"bounds\": [{}, {}, {}, {}],\n",
            b.x, b.y, b.width, b.height
        );
        json += "  \"layers\": [\n";
        for (i, layer) in layers.iter().enumerate() {
            let primitive = layer.clone().into_primitive();
            let mut tris = Vec::new();
            triangles(&primitive, &mut tris);
            let mut labels = Vec::new();
            texts(&primitive, &mut labels);

            json += &format!(
                "    {{\"triangles\": [{}], \"texts\": [{}]}}",
                tris.iter()
                    .map(|t| format!(
                        "[[{:.2},{:.2},{:.5},{:.5},{:.5},{:.5}],\
                         [{:.2},{:.2},{:.5},{:.5},{:.5},{:.5}],\
                         [{:.2},{:.2},{:.5},{:.5},{:.5},{:.5}]]",
                        t[0][0],
                        t[0][1],
                        t[0][2],
                        t[0][3],
                        t[0][4],
                        t[0][5],
                        t[1][0],
                        t[1][1],
                        t[1][2],
                        t[1][3],
                        t[1][4],
                        t[1][5],
                        t[2][0],
                        t[2][1],
                        t[2][2],
                        t[2][3],
                        t[2][4],
                        t[2][5],
                    ))
                    .collect::<Vec<_>>()
                    .join(","),
                labels
                    .iter()
                    .map(|t| format!("\"{t}\""))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            json += if i + 1 == layers.len() { "\n" } else { ",\n" };
        }
        json += "  ]\n}\n";
        std::fs::write(path, json).expect("failed to write canvas dump");
    }

    // ---- selection ------------------------------------------------------
    //
    // Selection is shared state between the canvas and the editor, so these
    // tests assert both halves: that a click moves it, and that deleting a
    // band moves it to a band that still exists (the compaction hazard).

    fn with_view(bands: &[(f32, f32, f32, FilterType)]) -> (GraphView, Arc<GraphProgram>) {
        let ctx = Arc::new(NullGuiContext);
        #[allow(clippy::arc_with_non_send_sync)]
        let p = Arc::new(test_program_with(test_params(bands), ctx));
        (GraphView { program: p.clone() }, p)
    }

    fn node_xy(program: &GraphProgram, band: usize) -> (f32, f32) {
        let bp = &program.params.band_params[band];
        let p = to_px(bounds(), bp.freq.value(), bp.gain.value());
        (p.x, p.y)
    }

    #[test]
    fn clicking_a_node_selects_that_band() {
        let (view, program) = with_view(&[
            (100.0, 6.0, 1.0, FilterType::Bell),
            (2_000.0, -4.0, 1.0, FilterType::Bell),
        ]);
        assert_eq!(program.selected(), None, "nothing selected initially");

        let (x, y) = node_xy(&program, 1);
        assert_eq!(
            click_at(&view, bounds(), x, y, mouse::Button::Left),
            canvas::event::Status::Captured
        );
        assert_eq!(program.selected(), Some(1));
    }

    #[test]
    fn creating_a_band_selects_it() {
        let (view, program) = with_view(&[]);
        let b = bounds();
        click_at(
            &view,
            b,
            b.x + freq_to_x(500.0) * b.width,
            b.y + b.height * 0.5,
            mouse::Button::Left,
        );
        assert_eq!(
            program.selected(),
            Some(0),
            "a band you just created is the one you want to edit"
        );
    }

    #[test]
    fn deleting_a_band_below_the_selection_follows_the_compaction() {
        let (view, program) = with_view(&[
            (100.0, 6.0, 1.0, FilterType::Bell),
            (500.0, 3.0, 1.0, FilterType::Bell),
            (2_000.0, -4.0, 1.0, FilterType::Bell),
        ]);
        program.select(Some(2));

        let (x, y) = node_xy(&program, 0);
        click_at(&view, bounds(), x, y, mouse::Button::Right);

        assert_eq!(
            program.selected(),
            Some(1),
            "band 2 slid into slot 1, so the selection must follow it down"
        );
    }

    #[test]
    fn deleting_the_selected_band_leaves_a_live_band_selected() {
        let (view, program) = with_view(&[
            (100.0, 6.0, 1.0, FilterType::Bell),
            (2_000.0, -4.0, 1.0, FilterType::Bell),
        ]);
        program.select(Some(1));

        let (x, y) = node_xy(&program, 1);
        click_at(&view, bounds(), x, y, mouse::Button::Right);

        assert_eq!(
            program.selected(),
            Some(0),
            "deleting the last band must fall back to the new last band"
        );
        assert!(program.selected().unwrap() < program.params.num_bands.value() as usize);
    }

    #[test]
    fn deleting_the_only_band_clears_the_selection() {
        let (view, program) = with_view(&[(100.0, 6.0, 1.0, FilterType::Bell)]);
        program.select(Some(0));

        let (x, y) = node_xy(&program, 0);
        click_at(&view, bounds(), x, y, mouse::Button::Right);

        assert_eq!(program.selected(), None);
    }

    #[test]
    fn the_selected_band_shows_its_readout_without_dragging() {
        let program = test_program(&[(440.0, -3.0, 1.0, FilterType::Bell)]);
        let plain = program.draw(bounds(), None, None);
        let mut before = Vec::new();
        texts(&plain[NODES].clone().into_primitive(), &mut before);
        assert!(
            !before.iter().any(|t| t.contains("440 Hz")),
            "an unselected node should not label itself: {before:?}"
        );

        program.select(Some(0));
        let selected = program.draw(bounds(), None, None);
        let mut after = Vec::new();
        texts(&selected[NODES].clone().into_primitive(), &mut after);
        assert!(
            after
                .iter()
                .any(|t| t.contains("440 Hz") && t.contains("-3.0 dB")),
            "the selected node should label itself in the panel's terms: {after:?}"
        );
    }

    #[test]
    fn the_selection_ring_is_drawn_only_for_the_selected_band() {
        let program = test_program(&[(440.0, 0.0, 1.0, FilterType::Bell)]);
        let plain = layer_vertices(&program.draw(bounds(), None, None), NODES).len();

        program.select(Some(0));
        let selected = layer_vertices(&program.draw(bounds(), None, None), NODES).len();

        assert!(
            selected > plain,
            "the selected node should add a ring ({selected} vs {plain} vertices)"
        );
    }

    // ---- keyboard -------------------------------------------------------

    fn press(view: &GraphView, key: keyboard::KeyCode) -> canvas::event::Status {
        press_with(view, key, keyboard::Modifiers::default())
    }

    fn press_with(
        view: &GraphView,
        key: keyboard::KeyCode,
        modifiers: keyboard::Modifiers,
    ) -> canvas::event::Status {
        view.handle_keyboard_event(keyboard::Event::KeyPressed {
            key_code: key,
            modifiers,
        })
    }

    /// Params + a recording context, with band 0 selected, ready for a key.
    fn keyboard_fixture(
        bands: &[(f32, f32, f32, FilterType)],
    ) -> (GraphView, Arc<GraphProgram>, Arc<RecordingGuiContext>) {
        let ctx = Arc::new(RecordingGuiContext::default());
        #[allow(clippy::arc_with_non_send_sync)]
        let p = Arc::new(test_program_with(test_params(bands), ctx.clone()));
        p.select(Some(0));
        (GraphView { program: p.clone() }, p, ctx)
    }

    #[test]
    fn right_arrow_moves_the_band_up_a_semitone() {
        let (view, program, ctx) = keyboard_fixture(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);

        assert_eq!(
            press(&view, keyboard::KeyCode::Right),
            canvas::event::Status::Captured
        );

        let expected = 1_000.0 * 2f32.powf(1.0 / 12.0);
        let normalized = program.params.band_params[0]
            .freq
            .preview_normalized(expected);
        let writes = ctx.writes();
        assert_eq!(writes.len(), 1, "one gesture, one write");
        assert!(
            (writes[0] - normalized).abs() < 1e-4,
            "expected {expected:.1} Hz, got normalized {} (want {normalized})",
            writes[0]
        );
    }

    #[test]
    fn left_arrow_moves_the_band_down_a_semitone() {
        let (view, program, ctx) = keyboard_fixture(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);

        press(&view, keyboard::KeyCode::Left);

        let expected = 1_000.0 / 2f32.powf(1.0 / 12.0);
        let normalized = program.params.band_params[0]
            .freq
            .preview_normalized(expected);
        assert!((ctx.writes()[0] - normalized).abs() < 1e-4);
    }

    #[test]
    fn shift_arrow_moves_a_whole_octave() {
        let (view, program, ctx) = keyboard_fixture(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);

        press_with(&view, keyboard::KeyCode::Right, keyboard::Modifiers::SHIFT);

        let normalized = program.params.band_params[0]
            .freq
            .preview_normalized(2_000.0);
        assert!((ctx.writes()[0] - normalized).abs() < 1e-4);
    }

    #[test]
    fn up_and_down_arrows_nudge_gain_by_a_tenth_of_a_db() {
        let (view, program, ctx) = keyboard_fixture(&[(1_000.0, 3.0, 1.0, FilterType::Bell)]);
        let gain = &program.params.band_params[0].gain;

        press(&view, keyboard::KeyCode::Up);
        assert!((ctx.writes()[0] - gain.preview_normalized(3.1)).abs() < 1e-6);

        // The recording context does not store the value (nih-plug keeps the
        // parameter setters crate-private), so the next press starts from the
        // same 3.0 dB. Accumulation across presses is covered by the
        // end-to-end test, which runs the value through the DSP.
        press(&view, keyboard::KeyCode::Down);
        assert!((ctx.writes()[1] - gain.preview_normalized(2.9)).abs() < 1e-6);
    }

    #[test]
    fn arrows_are_clamped_to_the_parameter_range() {
        let (view, program, ctx) = keyboard_fixture(&[(1_000.0, 24.0, 1.0, FilterType::Bell)]);

        press_with(&view, keyboard::KeyCode::Up, keyboard::Modifiers::SHIFT);

        let gain = &program.params.band_params[0].gain;
        assert!((ctx.writes()[0] - gain.preview_normalized(24.0)).abs() < 1e-6);
    }

    #[test]
    fn arrows_do_nothing_without_a_selection() {
        let (view, _program, ctx) = keyboard_fixture(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);
        // Undo the fixture's selection.
        view.program.select(None);

        assert_eq!(
            press(&view, keyboard::KeyCode::Right),
            canvas::event::Status::Ignored
        );
        assert!(ctx.writes().is_empty(), "no selection, no writes");
    }

    #[test]
    fn tab_and_shift_tab_cycle_the_selection() {
        let (view, program, _ctx) = keyboard_fixture(&[
            (100.0, 0.0, 1.0, FilterType::Bell),
            (1_000.0, 0.0, 1.0, FilterType::Bell),
            (5_000.0, 0.0, 1.0, FilterType::Bell),
        ]);

        press(&view, keyboard::KeyCode::Tab);
        assert_eq!(program.selected(), Some(1));
        press(&view, keyboard::KeyCode::Tab);
        assert_eq!(program.selected(), Some(2));
        press(&view, keyboard::KeyCode::Tab);
        assert_eq!(program.selected(), Some(0), "wraps to the first band");
        press_with(&view, keyboard::KeyCode::Tab, keyboard::Modifiers::SHIFT);
        assert_eq!(program.selected(), Some(2), "wraps backwards");
    }

    #[test]
    fn delete_removes_the_selected_band_and_moves_the_selection() {
        let (view, program, ctx) = keyboard_fixture(&[
            (100.0, 0.0, 1.0, FilterType::Bell),
            (1_000.0, 0.0, 1.0, FilterType::Bell),
        ]);
        program.select(Some(1));

        assert_eq!(
            press(&view, keyboard::KeyCode::Delete),
            canvas::event::Status::Captured
        );

        let writes = ctx.writes();
        assert!(
            (writes.last().copied().unwrap() - 1.0 / MAX_BANDS as f32).abs() < 1e-6,
            "the band count should be lowered to 1, got {writes:?}"
        );
        assert_eq!(program.selected(), Some(0));
    }

    #[test]
    fn backspace_deletes_like_delete() {
        let (view, program, ctx) = keyboard_fixture(&[(100.0, 0.0, 1.0, FilterType::Bell)]);

        press(&view, keyboard::KeyCode::Backspace);

        assert_eq!(ctx.writes(), vec![0.0], "the count must drop to 0");
        assert_eq!(program.selected(), None);
    }

    #[test]
    fn escape_clears_the_selection() {
        let (view, program, _ctx) = keyboard_fixture(&[(100.0, 0.0, 1.0, FilterType::Bell)]);
        assert_eq!(program.selected(), Some(0));

        press(&view, keyboard::KeyCode::Escape);

        assert_eq!(program.selected(), None);
    }

    // ---- cuts and slopes ------------------------------------------------

    /// A cut's node is drawn and grabbed on the 0 dB line, so clicking where
    /// the node appears must grab it — not create a second band on top.
    #[test]
    fn a_cut_node_is_grabbed_on_the_zero_db_line() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let params = crate::test_support::test_params_with_slopes(&[(
            1_000.0,
            0.0,
            0.707,
            FilterType::LowCut,
            crate::Slope::Db24,
        )]);
        #[allow(clippy::arc_with_non_send_sync)]
        let program = Arc::new(test_program_with(params.clone(), ctx.clone()));
        let view = GraphView {
            program: program.clone(),
        };
        let b = bounds();

        // The band's gain parameter says +12 dB, but a cut is drawn at 0 dB.
        let (x, y) = node_xy(&program, 0);
        assert!(
            (y - (b.y + db_to_y(0.0) * b.height)).abs() < 1.0,
            "a cut node belongs on the 0 dB line, got y = {y}"
        );

        click_at(&view, b, x, y, mouse::Button::Left);
        assert!(
            ctx.writes().is_empty(),
            "clicking a cut node created a band instead of grabbing it: {:?}",
            ctx.writes()
        );
        assert!(program.drag.get().is_some(), "the node should be grabbed");
    }

    #[test]
    fn scrolling_over_a_cut_steps_the_slope_instead_of_the_q() {
        let ctx = Arc::new(RecordingGuiContext::default());
        let params = crate::test_support::test_params_with_slopes(&[(
            1_000.0,
            0.0,
            0.707,
            FilterType::LowCut,
            crate::Slope::Db12,
        )]);
        #[allow(clippy::arc_with_non_send_sync)]
        let program = Arc::new(test_program_with(params.clone(), ctx.clone()));
        let view = GraphView {
            program: program.clone(),
        };
        let b = bounds();
        let (x, y) = node_xy(&program, 0);
        let cursor = Cursor::Available(Point::new(x, y));

        view.handle_mouse_event(
            mouse::Event::WheelScrolled {
                delta: mouse::ScrollDelta::Lines { x: 0.0, y: 1.0 },
            },
            b,
            cursor,
        );

        let writes = ctx.writes();
        assert_eq!(writes.len(), 1, "one slope step, one write: {writes:?}");
        let slope = &params.band_params[0].slope;
        let expected = slope.preview_normalized(crate::Slope::Db18 as i32);
        assert!(
            (writes[0] - expected).abs() < 1e-6,
            "12 dB/oct + 1 step should be 18 dB/oct"
        );
    }

    #[test]
    fn scrolling_over_a_bell_still_steps_the_q() {
        let (view, program, ctx) = keyboard_fixture(&[(1_000.0, 0.0, 1.0, FilterType::Bell)]);
        let b = bounds();
        let (x, y) = node_xy(&program, 0);

        view.handle_mouse_event(
            mouse::Event::WheelScrolled {
                delta: mouse::ScrollDelta::Lines { x: 0.0, y: 1.0 },
            },
            b,
            Cursor::Available(Point::new(x, y)),
        );

        let writes = ctx.writes();
        assert_eq!(writes.len(), 1);
        // Scrolling up opens the Q: the normalized value must move up from
        // where it was, and the slope must not have been touched.
        let q = &program.params.band_params[0].q;
        assert!(
            writes[0] > q.preview_normalized(q.value()),
            "expected a wider Q, got normalized {} from {}",
            writes[0],
            q.preview_normalized(q.value())
        );
        assert_eq!(
            crate::Slope::from_i32(program.params.band_params[0].slope.value()),
            crate::Slope::DEFAULT,
            "scrolling a bell must not change the slope"
        );
    }

    // ==== spectrum display controls =====================================

    /// Centre of a chip, laid out exactly as the canvas lays it out.
    fn chip_center(view: &SpectrumView, chip: Chip) -> (f32, f32) {
        let (_, rect) = chip_rects(view, bounds())
            .into_iter()
            .find(|(c, _)| *c == chip)
            .expect("every chip has a rectangle");
        (rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
    }

    #[test]
    fn the_chips_are_drawn_with_their_labels() {
        let program = test_program(&[]);
        let layers = program.draw(bounds(), None, None);
        let mut labels = Vec::new();
        texts(&layers[CHIPS].clone().into_primitive(), &mut labels);

        for expected in ["Freeze", "Tilt", "-90 dB", "Pre +0"] {
            assert!(
                labels.iter().any(|l| l == expected),
                "chip {expected} is not drawn; got {labels:?}"
            );
        }
        assert!(
            !layer_vertices(&layers, CHIPS).is_empty(),
            "chips drew text but no background geometry"
        );
    }

    #[test]
    fn the_chips_fit_in_the_corner_without_overlapping() {
        // The chip widths follow their labels, which change as the state does
        // ("Tilt" -> "Tilt on", "-90 dB" -> "-120 dB"), so this has to hold for
        // every combination of states rather than for the default one.
        let view = SpectrumView::new();
        let b = bounds();
        for frozen in [false, true] {
            for tilted in [false, true] {
                for _ in crate::dsp::spectrum::DB_RANGES {
                    view.set_frozen(frozen);
                    view.set_tilted(tilted);
                    let rects = chip_rects(&view, b);
                    for (chip, rect) in rects {
                        assert!(
                            rect.x >= b.x && rect.x + rect.width <= b.x + b.width,
                            "{chip:?} ({}) leaves the canvas: {rect:?}",
                            chip.label(&view)
                        );
                        assert!(
                            rect.y >= b.y && rect.y + rect.height <= b.y + b.height,
                            "{chip:?} leaves the canvas vertically: {rect:?}"
                        );
                    }
                    // Right-to-left layout: each chip must end before the next
                    // one starts, leaving the gap between them.
                    let mut left_of = f32::INFINITY;
                    for (chip, rect) in rects {
                        assert!(
                            rect.x + rect.width <= left_of,
                            "{chip:?} overlaps its neighbour ({rect:?})"
                        );
                        left_of = rect.x;
                    }
                    view.cycle_range();
                }
            }
        }
    }

    #[test]
    fn clicking_the_freeze_chip_holds_the_frame() {
        let taps = Arc::new(SpectrumTaps::new());
        taps.input.publish(frame(-20.0, 1));
        #[allow(clippy::arc_with_non_send_sync)]
        let program = Arc::new(test_program_with_taps(
            test_params(&[]),
            Arc::new(NullGuiContext),
            taps.clone(),
        ));
        let view = GraphView {
            program: program.clone(),
        };
        let b = bounds();
        let (x, y) = chip_center(&taps.view, Chip::Freeze);

        assert!(!taps.view.frozen());
        assert_eq!(
            click_at(&view, b, x, y, mouse::Button::Left),
            canvas::event::Status::Captured,
            "the chip must consume the click"
        );
        assert!(taps.view.frozen());

        // A click on the chip must not also create a band behind it.
        assert_eq!(
            program.params.num_bands.value(),
            0,
            "clicking a chip created a band"
        );

        // Freeze the picture at -20 dB, then let audio drop away: the drawn
        // trace must stay where it was.
        let frozen_y = top_y(&program.draw(b, None, None), INPUT_SPECTRUM);
        taps.input.publish(frame(-80.0, 2));
        for _ in 0..10 {
            program.draw(b, None, None);
        }
        let held_y = top_y(&program.draw(b, None, None), INPUT_SPECTRUM);
        assert!(
            (held_y - frozen_y).abs() < 0.5,
            "frozen trace moved from {frozen_y:.1} to {held_y:.1}"
        );

        // Clicking again releases it, and the trace follows the new level.
        click_at(&view, b, x, y, mouse::Button::Left);
        assert!(!taps.view.frozen());
        let mut after = held_y;
        for _ in 0..20 {
            after = top_y(&program.draw(b, None, None), INPUT_SPECTRUM);
        }
        assert!(
            after > held_y + 5.0,
            "-80 dB should draw lower than -20 dB ({held_y:.1} -> {after:.1})"
        );
    }

    #[test]
    fn clicking_the_range_chip_walks_the_floors() {
        let taps = Arc::new(SpectrumTaps::new());
        #[allow(clippy::arc_with_non_send_sync)]
        let program = Arc::new(test_program_with_taps(
            test_params(&[]),
            Arc::new(NullGuiContext),
            taps.clone(),
        ));
        let view = GraphView { program };
        let b = bounds();

        let start = taps.view.floor_db();
        let (x, y) = chip_center(&taps.view, Chip::Range);
        click_at(&view, b, x, y, mouse::Button::Left);
        let next = taps.view.floor_db();
        assert_ne!(next, start, "the range chip did not change the floor");
        assert!(
            crate::dsp::spectrum::DB_RANGES.contains(&next),
            "{next} is not one of the offered ranges"
        );

        // The label follows the state, so the button says what it will do.
        let mut labels = Vec::new();
        texts(
            &view.program.draw(b, None, None)[CHIPS]
                .clone()
                .into_primitive(),
            &mut labels,
        );
        assert!(
            labels.iter().any(|l| l == &format!("{next:.0} dB")),
            "chip still shows the old range: {labels:?}"
        );
    }

    #[test]
    fn a_deeper_range_maps_the_same_level_higher() {
        // The top of the display is pinned at 0 dB, so widening the range to
        // -120 dB pushes the floor down rather than rescaling the signal: a
        // given level ends up higher up the canvas, with more room beneath it.
        let b = bounds();
        let quiet = -80.0;
        let shallow = spectrum_y(b, quiet, -60.0);
        let deep = spectrum_y(b, quiet, -120.0);
        assert!(
            deep < shallow,
            "-80 dB should sit higher on a -120 dB scale ({deep:.1}) than on a \
             -60 dB one, where it is already below the floor ({shallow:.1})"
        );
        assert!(
            (spectrum_y(b, 0.0, -120.0) - spectrum_y(b, 0.0, -60.0)).abs() < 0.01,
            "0 dB must not move when the range changes"
        );
        // The floor always lands exactly on the bottom edge, for every range.
        for floor in crate::dsp::spectrum::DB_RANGES {
            assert!((spectrum_y(b, floor, floor) - (b.y + b.height)).abs() < 0.01);
        }
    }

    #[test]
    fn the_pre_offset_moves_the_input_trace_only() {
        let taps = Arc::new(SpectrumTaps::new());
        taps.input.publish(frame(0.0, 1));
        taps.output.publish(frame(-20.0, 1));
        let taps_for_program = taps.clone();
        #[allow(clippy::arc_with_non_send_sync)]
        let program = Arc::new(test_program_with_taps(
            test_params(&[]),
            Arc::new(NullGuiContext),
            taps_for_program,
        ));
        let view = GraphView {
            program: program.clone(),
        };
        let b = bounds();

        let before = program.draw(b, None, None);
        let input_before = top_y(&before, INPUT_SPECTRUM);
        let output_before = top_y(&before, OUTPUT_SPECTRUM);

        // Two steps: 0 -> -6 -> -12 dB.
        let (x, y) = chip_center(&taps.view, Chip::PreGain);
        click_at(&view, b, x, y, mouse::Button::Left);
        click_at(&view, b, x, y, mouse::Button::Left);
        assert_eq!(taps.view.pre_gain_db(), -12.0);

        let after = program.draw(b, None, None);
        let input_after = top_y(&after, INPUT_SPECTRUM);
        let output_after = top_y(&after, OUTPUT_SPECTRUM);

        let expected = spectrum_y(b, -12.0, taps.view.floor_db());
        assert!(
            (input_after - expected).abs() < 1.0,
            "input trace should be drawn at -12 dB (y {expected:.1}), got {input_after:.1}"
        );
        assert!(
            input_after > input_before,
            "a negative offset must move the input trace down"
        );
        assert!(
            (output_after - output_before).abs() < 0.5,
            "the pre-gain offset must not move the post trace"
        );
    }

    #[test]
    fn tilt_tilts_the_drawn_curve_with_the_spectrum() {
        // A flat EQ under tilt is no longer a flat line: the curve shares the
        // spectrum's reference, so it rises with it. If it did not, a flat EQ
        // would appear to be cutting the top end by 19 dB.
        let taps = Arc::new(SpectrumTaps::new());
        #[allow(clippy::arc_with_non_send_sync)]
        let program = Arc::new(test_program_with_taps(
            test_params(&[]),
            Arc::new(NullGuiContext),
            taps.clone(),
        ));
        let view = GraphView {
            program: program.clone(),
        };
        let b = bounds();

        let y_at = |program: &GraphProgram, hz: f32| {
            let layers = program.draw(b, None, None);
            let want_x = b.x + freq_to_x(hz) * b.width;
            layer_vertices(&layers, CURVES)
                .iter()
                .filter(|v| (v[0] - want_x).abs() < 3.0)
                .map(|v| v[1])
                .fold(f32::INFINITY, f32::min)
        };

        let flat_low = y_at(&program, 1_000.0);
        let flat_high = y_at(&program, 10_000.0);
        assert!(
            (flat_low - flat_high).abs() < 2.0,
            "a flat EQ should draw flat before tilt ({flat_low:.1} vs {flat_high:.1})"
        );

        let (x, y) = chip_center(&taps.view, Chip::Tilt);
        click_at(&view, b, x, y, mouse::Button::Left);
        assert!(taps.view.tilted());

        let tilted_low = y_at(&program, 1_000.0);
        let tilted_high = y_at(&program, 10_000.0);
        // The tilt pivots at 1 kHz: that point must not move.
        assert!(
            (tilted_low - flat_low).abs() < 2.0,
            "the tilt pivot moved from {flat_low:.1} to {tilted_low:.1}"
        );
        assert!(
            tilted_high < flat_high - 10.0,
            "10 kHz should rise under tilt: {flat_high:.1} -> {tilted_high:.1}"
        );
    }
}
