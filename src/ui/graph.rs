//! EQ curve graph — canvas widget drawing grid, response curve, and band handles.

use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_iced::backend::widget::canvas::{
    self, Canvas as IcedCanvas, Cursor, Fill, Frame, Geometry, Path, Program, Stroke,
    Text as CanvasText,
};
use nih_plug_iced::{Color, Length, Point, Rectangle};

use crate::dsp::curve::{
    band_response_db, db_to_y, freq_to_x, BandSpec, DB_MAX, FREQ_MAX, FREQ_MIN, GRID_DBS,
    GRID_FREQS,
};
use crate::{editor, FilterParams, FilterType, MAX_BANDS};

/// Height of the graph area in points.
pub const GRAPH_HEIGHT: f32 = 300.0;

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

/// Map a frequency + gain to pixel coordinates within bounds.
fn to_px(bounds: Rectangle, freq: f32, db: f32) -> Point {
    Point {
        x: bounds.x + freq_to_x(freq) * bounds.width,
        y: bounds.y + db_to_y(db) * bounds.height,
    }
}

struct GraphProgram {
    params: Arc<FilterParams>,
    _peak_meter: Arc<AtomicF32>,
}

impl GraphProgram {
    fn draw_grid(&self, frame: &mut Frame, bounds: Rectangle) {
        let grid = Color::from_rgba(1.0, 1.0, 1.0, 0.06);
        let label = Color::from_rgba(1.0, 1.0, 1.0, 0.25);

        // Vertical frequency lines + labels
        for &f in &GRID_FREQS {
            let x = bounds.x + freq_to_x(f) * bounds.width;
            frame.fill(
                &Path::line(
                    Point::new(x, bounds.y),
                    Point::new(x, bounds.y + bounds.height),
                ),
                Fill::from(grid),
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
            frame.fill(
                &Path::line(
                    Point::new(bounds.x, y),
                    Point::new(bounds.x + bounds.width, y),
                ),
                Fill::from(grid),
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
        frame.fill(
            &Path::line(
                Point::new(bounds.x, y0),
                Point::new(bounds.x + bounds.width, y0),
            ),
            Fill::from(Color::from_rgba(1.0, 1.0, 1.0, 0.14)),
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
        const POINTS: usize = 120;

        for i in 0..num_active {
            let spec = band_specs(&self.params, num_active)[i];
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
                    let db = band_response_db(&spec, freq);
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

    fn draw_nodes(&self, frame: &mut Frame, bounds: Rectangle) {
        let num_active = self.params.num_bands.value() as usize;
        for i in 0..num_active {
            let bp = &self.params.band_params[i];
            if !bp.enable.value() {
                continue;
            }
            let color = editor::band_color(i);
            let center = to_px(bounds, bp.freq.value(), bp.gain.value());

            // Outer ring
            frame.fill(
                &Path::circle(center, 7.0),
                Fill::from(Color { a: 0.25, ..color }),
            );
            // Inner dot
            frame.fill(&Path::circle(center, 3.5), Fill::from(color));
        }
    }
}

impl GraphProgram {
    fn draw(&self, bounds: Rectangle) -> Vec<Geometry> {
        let mut frame = Frame::new(bounds.size());
        self.draw_grid(&mut frame, bounds);
        self.draw_band_curves(&mut frame, bounds);
        self.draw_curve(&mut frame, bounds);
        self.draw_nodes(&mut frame, bounds);
        vec![frame.into_geometry()]
    }
}

/// Public handle used by the editor's view.
pub struct Graph {
    program: Arc<GraphProgram>,
}

impl Graph {
    pub fn new(params: Arc<FilterParams>, peak_meter: Arc<AtomicF32>) -> Self {
        Self {
            program: Arc::new(GraphProgram {
                params,
                _peak_meter: peak_meter,
            }),
        }
    }

    pub fn view(&self) -> nih_plug_iced::Element<'_, Message> {
        // Parameters animate every frame (smoothers) and nih-plug redraws the
        // whole tree each frame, so a canvas Cache would freeze the drawing.
        // Geometry for ~160 points is cheap to rebuild — bypass the cache.
        IcedCanvas::new(GraphView {
            program: self.program.clone(),
        })
        .width(Length::Fill)
        .height(Length::Units(GRAPH_HEIGHT as u16))
        .into()
    }
}

/// Wrapper implementing Program (forwarded to the shared GraphProgram).
struct GraphView {
    program: Arc<GraphProgram>,
}

impl Program<Message> for GraphView {
    fn update(
        &mut self,
        _event: canvas::Event,
        _bounds: Rectangle,
        _cursor: Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        // Node dragging lands in step 3 — drag state will live on GraphView
        (canvas::event::Status::Ignored, None)
    }

    fn draw(&self, bounds: Rectangle, _cursor: Cursor) -> Vec<Geometry> {
        self.program.draw(bounds)
    }

    fn mouse_interaction(
        &self,
        _bounds: Rectangle,
        _cursor: Cursor,
    ) -> nih_plug_iced::mouse::Interaction {
        nih_plug_iced::mouse::Interaction::default()
    }
}

/// Message type used by the editor — graph only emits these from step 3 on.
use crate::editor::Message;
