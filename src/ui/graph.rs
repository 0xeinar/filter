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
use nih_plug_iced::{keyboard, mouse, Color, Length, Point, Rectangle};

use crate::dsp::curve::{
    band_response_db, db_to_y, freq_to_x, nearest_grabbable_node, px_to_freq, scroll_q_step,
    y_to_db, BandSpec, DB_MAX, FREQ_MAX, FREQ_MIN, GRID_DBS, GRID_FREQS,
};
use crate::dsp::spectrum::{SpectrumFrame, BINS};
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
fn band_specs(params: &FilterParams, num_active: usize) -> [BandSpec; MAX_BANDS] {
    core::array::from_fn(|i| {
        if i >= num_active {
            return BandSpec {
                freq: 1000.0,
                gain_db: 0.0,
                q: 1.0,
                filter_type: FilterType::Bell,
                enabled: false,
            };
        }
        let bp = &params.band_params[i];
        BandSpec {
            freq: bp.freq.value(),
            gain_db: bp.gain.value(),
            q: bp.q.value(),
            filter_type: FilterType::from_i32(bp.filter_type.value()),
            enabled: bp.enable.value(),
        }
    })
}

/// x position (0..1) of each spectrum bin's upper edge, precomputed once.
/// Depends only on the FFT bin → frequency mapping, which is fixed.
fn spectrum_bin_x() -> Vec<f32> {
    let first_bin = 1.0f32;
    let last_bin_f = (crate::dsp::spectrum::FFT_SIZE / 2 - 1) as f32;
    let decades = (last_bin_f / first_bin).log10();
    (0..BINS)
        .map(|i| {
            let edge_bin = (first_bin * 10f32.powf(decades * (i + 1) as f32 / BINS as f32)).round();
            let edge_hz = edge_bin * 48_000.0 / crate::dsp::spectrum::FFT_SIZE as f32;
            freq_to_x(edge_hz.min(FREQ_MAX))
        })
        .collect()
}

/// Map a frequency + gain to pixel coordinates within bounds.
fn to_px(bounds: Rectangle, freq: f32, db: f32) -> Point {
    Point {
        x: bounds.x + freq_to_x(freq) * bounds.width,
        y: bounds.y + db_to_y(db) * bounds.height,
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
                params.band_params[i].gain.value(),
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
    spectrum_ring: Arc<crate::dsp::spectrum::SpectrumRing>,
    /// Visually-smoothed spectrum bars, shared lock-free (GUI thread writes,
    /// same thread reads; atomic to satisfy Send+Sync on the Arc).
    smoothed: Vec<AtomicF32>,
    /// Precomputed x position (0..1 across the graph) for every spectrum bin.
    /// The bin → frequency mapping never changes, so this saves a `powf` and a
    /// `log10` per bin per frame — 192 of them at 60 fps.
    spec_x: Vec<f32>,
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
    fn key_for(params: &FilterParams, bounds: Rectangle) -> u64 {
        let mut h = DefaultHasher::new();
        let num_active = params.num_bands.value() as usize;
        num_active.hash(&mut h);
        for i in 0..num_active {
            let bp = &params.band_params[i];
            bp.freq.value().to_bits().hash(&mut h);
            bp.gain.value().to_bits().hash(&mut h);
            bp.q.value().to_bits().hash(&mut h);
            bp.filter_type.value().hash(&mut h);
            bp.enable.value().hash(&mut h);
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
    /// area along the bottom of the graph. dB range: -90..0 mapped to
    /// graph height fraction 0..0.75 (0 dB at 3/4 height, linear in dB).
    fn update_and_draw_spectrum(&self, frame: &mut Frame, bounds: Rectangle) {
        let latest: SpectrumFrame = self.spectrum_ring.read_newest();

        // Rise fast, fall slow (classic analyzer ballistics)
        const RISE: f32 = 0.55;
        const FALL: f32 = 0.08;
        let mut path_pts: Vec<Point> = Vec::with_capacity(BINS + 2);

        // Map: bin i → precomputed x position (log, same as curve); dB → y
        // with -90 dB at the bottom and 0 dB at 3/4 height. Values above 0 dB
        // clamp.
        let db_floor = -90.0f32;
        path_pts.push(Point::new(bounds.x, bounds.y + bounds.height));
        for (i, slot) in self.smoothed.iter().enumerate() {
            let cur = slot.load(std::sync::atomic::Ordering::Relaxed);
            let target = latest.db[i];
            let k = if target > cur { RISE } else { FALL };
            let next = cur + (target - cur) * k;
            slot.store(next, std::sync::atomic::Ordering::Relaxed);

            let y01 = ((next - db_floor) / (0.0 - db_floor)).clamp(0.0, 1.0) * 0.75;
            let x = bounds.x + self.spec_x[i] * bounds.width;
            let y = bounds.y + bounds.height * (1.0 - y01);
            path_pts.push(Point::new(x, y));
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
            let center = to_px(bounds, bp.freq.value(), bp.gain.value());

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

            // Frequency + gain readout while dragging or while selected
            if is_drag || is_selected {
                let label = format!("{:.0} Hz  {:+.1} dB", bp.freq.value(), bp.gain.value());
                let text_x = (center.x + 10.0).min(bounds.x + bounds.width - 95.0);
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

    fn draw(
        &self,
        bounds: Rectangle,
        hover: Option<usize>,
        dragging: Option<usize>,
    ) -> Vec<Geometry> {
        // The grid, per-band curves and the combined curve only change when a
        // parameter (or the canvas geometry) does. nih-plug redraws the whole
        // editor every frame with `always_redraw`, but the EQ picture is
        // usually static — so keep those layers and rebuild them on change
        // only. Rebuilding cost the trig-heavy `band_response_db` up to 24 ×
        // 120 times per frame.
        let key = CachedLayers::key_for(&self.params, bounds);
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
        let mut spectrum_frame = Frame::new(bounds.size());
        self.update_and_draw_spectrum(&mut spectrum_frame, bounds);
        let mut nodes_frame = Frame::new(bounds.size());
        self.draw_nodes(&mut nodes_frame, bounds, hover, dragging, self.selected());

        let cached = cache
            .as_ref()
            .expect("cache was just populated in this same call");
        vec![
            cached.grid.clone(),
            spectrum_frame.into_geometry(),
            cached.curves.clone(),
            nodes_frame.into_geometry(),
        ]
    }
}

/// Public handle used by the editor's view.
pub struct Graph {
    program: Arc<GraphProgram>,
}

impl Graph {
    pub fn new(
        params: Arc<FilterParams>,
        context: Arc<dyn GuiContext>,
        peak_meter: Arc<AtomicF32>,
        spectrum_ring: Arc<crate::dsp::spectrum::SpectrumRing>,
        selected: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            // The Cell fields are GUI-thread-only interaction state, but the
            // program is shared through an Arc inside the editor, which
            // declares (unsafely, with justification) Send+Sync. Access stays
            // on the GUI thread — same contract as the slider states.
            #[allow(clippy::arc_with_non_send_sync)]
            program: Arc::new(GraphProgram {
                params,
                context,
                _peak_meter: peak_meter,
                spectrum_ring,
                smoothed: (0..BINS).map(|_| AtomicF32::new(-90.0)).collect(),
                spec_x: spectrum_bin_x(),
                cache: RefCell::new(None),
                drag: Cell::new(None),
                hover: Cell::new(None),
                fine_scroll: Cell::new(false),
                last_click: Cell::new(None),
                selected,
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
                    program.set_plain(&bp.gain, gain_db);
                    (canvas::event::Status::Captured, None)
                } else {
                    // Hover highlight when near a node
                    let num_active = program.params.num_bands.value() as usize;
                    let nodes = node_positions(&program.params, num_active, bounds);
                    let new_hover =
                        nearest_grabbable_node(local_x, local_y, &nodes).map(|hit| hit.band_index);
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

        let num_active = program.params.num_bands.value() as usize;
        let Some(band) = program.selected().filter(|&b| b < num_active) else {
            return canvas::event::Status::Ignored;
        };

        use keyboard::KeyCode;
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
    use crate::dsp::spectrum::SpectrumRing;
    use crate::test_support::{test_params, NullGuiContext};
    use atomic_float::AtomicF32;
    use iced_graphics::Primitive;
    use nih_plug::prelude::util;
    use nih_plug::prelude::Param;

    /// Records the normalized values the draw/interaction code writes to the
    /// host, so tests can assert on *what* the GUI asks the host to do.
    #[derive(Default)]
    struct TestGuiContext {
        writes: std::sync::Mutex<Vec<f32>>,
    }

    impl TestGuiContext {
        fn writes(&self) -> Vec<f32> {
            self.writes.lock().unwrap().clone()
        }
    }

    impl GuiContext for TestGuiContext {
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
        GraphProgram {
            params,
            context,
            _peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            spectrum_ring: Arc::new(SpectrumRing::new()),
            smoothed: (0..BINS).map(|_| AtomicF32::new(-90.0)).collect(),
            spec_x: spectrum_bin_x(),
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
                for idx in buffers.indices.chunks_exact(3) {
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

        // Grid | spectrum | curves | nodes — the order the renderer composites.
        assert_eq!(layers.len(), 4, "expected four canvas layers");

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
            !layer_vertices(&layers, 0).is_empty(),
            "grid layer has text but no mesh — grid lines are not being drawn"
        );
        assert!(
            !layer_vertices(&layers, 1).is_empty(),
            "spectrum layer has no geometry"
        );
        assert!(
            !layer_vertices(&layers, 2).is_empty(),
            "curve layer has no geometry"
        );
        assert!(
            !layer_vertices(&layers, 3).is_empty(),
            "node layer has no geometry for an enabled band"
        );
    }

    #[test]
    fn nodes_layer_is_empty_without_bands() {
        let program = test_program(&[]);
        let layers = program.draw(bounds(), None, None);
        assert!(
            layer_vertices(&layers, 3).is_empty(),
            "no bands should draw no nodes"
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
        let verts = layer_vertices(&layers, 2);

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
        let first = layer_vertices(&program.draw(b, None, None), 2);
        let second = layer_vertices(&program.draw(b, None, None), 2);
        assert_eq!(first, second, "unchanged params must redraw identically");

        // A different gain has to produce a different curve: if the cache key
        // ever stops covering the band values, this is what fails.
        let flipped = test_program(&[(1_000.0, -12.0, 1.0, FilterType::Bell)]);
        let third = layer_vertices(&flipped.draw(b, None, None), 2);
        assert_ne!(
            first, third,
            "curve layer does not respond to the band gain"
        );
    }

    #[test]
    fn cache_key_tracks_params_and_bounds() {
        let b = bounds();
        let base =
            CachedLayers::key_for(&test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]), b);
        assert_eq!(
            base,
            CachedLayers::key_for(&test_params(&[(1_000.0, 12.0, 1.0, FilterType::Bell)]), b),
            "identical params must produce the same key"
        );
        assert_ne!(
            base,
            CachedLayers::key_for(&test_params(&[(2_000.0, 12.0, 1.0, FilterType::Bell)]), b),
            "freq not tracked"
        );
        assert_ne!(
            base,
            CachedLayers::key_for(&test_params(&[(1_000.0, 12.0, 3.0, FilterType::Bell)]), b),
            "q not tracked"
        );
        assert_ne!(
            base,
            CachedLayers::key_for(
                &test_params(&[(1_000.0, 12.0, 1.0, FilterType::LowShelf)]),
                b
            ),
            "filter type not tracked"
        );
        assert_ne!(
            base,
            CachedLayers::key_for(&test_params(&[]), b),
            "band count not tracked"
        );

        let mut resized = b;
        resized.width += 1.0;
        assert_ne!(
            base,
            CachedLayers::key_for(
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
        let ctx = Arc::new(TestGuiContext::default());
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
        assert_eq!(writes.len(), 6, "5 band params + the band count");
        assert!(
            (writes[4] - 1.0).abs() < 1e-6,
            "created band must be enabled"
        );
        assert!(
            (writes[5] - 1.0 / MAX_BANDS as f32).abs() < 1e-4,
            "band count must be raised to 1, got normalized {}",
            writes[5]
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
        let ctx = Arc::new(TestGuiContext::default());
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
        let ctx = Arc::new(TestGuiContext::default());
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
        assert_eq!(writes.len(), 6, "one band copied (5 params) + the count");
        // The copied frequency must be the third band's 5 kHz, not the deleted one.
        let expected = program.params.band_params[1]
            .freq
            .preview_normalized(5_000.0);
        assert!(
            (writes[0] - expected).abs() < 1e-4,
            "tail band was not copied into the freed slot"
        );
        assert!(
            (writes[5] - 2.0 / MAX_BANDS as f32).abs() < 1e-4,
            "band count must drop to 2"
        );
    }

    #[test]
    fn double_clicking_a_node_resets_its_gain() {
        let ctx = Arc::new(TestGuiContext::default());
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
        let ctx = Arc::new(TestGuiContext::default());
        let full: Vec<(f32, f32, f32, FilterType)> = (0..MAX_BANDS)
            .map(|i| (100.0 + i as f32 * 100.0, 0.0, 1.0, FilterType::Bell))
            .collect();
        let program = test_program_with(test_params(&full), ctx.clone());
        assert!(program.create_band(1_000.0, 0.0).is_none());
        assert!(ctx.writes().is_empty(), "nothing may be written when full");
    }

    /// Not an assertion — dumps the four layers as JSON for visual inspection
    /// (`FILTER_CANVAS_DUMP=/tmp/layers.json cargo test -- --nocapture`).
    #[test]
    fn dump_layers_for_visual_inspection() {
        let Some(path) = std::env::var_os("FILTER_CANVAS_DUMP") else {
            return;
        };
        let program = test_program(&[
            (100.0, 4.0, 1.0, FilterType::LowShelf),
            (1_000.0, 12.0, 1.6, FilterType::Bell),
            (6_000.0, -8.0, 2.0, FilterType::Bell),
        ]);
        let b = bounds();
        let layers = program.draw(b, None, None);

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
        texts(&plain[3].clone().into_primitive(), &mut before);
        assert!(
            !before.iter().any(|t| t.contains("440 Hz")),
            "an unselected node should not label itself: {before:?}"
        );

        program.select(Some(0));
        let selected = program.draw(bounds(), None, None);
        let mut after = Vec::new();
        texts(&selected[3].clone().into_primitive(), &mut after);
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
        let plain = layer_vertices(&program.draw(bounds(), None, None), 3).len();

        program.select(Some(0));
        let selected = layer_vertices(&program.draw(bounds(), None, None), 3).len();

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
    ) -> (GraphView, Arc<GraphProgram>, Arc<TestGuiContext>) {
        let ctx = Arc::new(TestGuiContext::default());
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
}
