//! Chart primitives drawn straight into the buffer: braille line and area
//! charts (2x4 dots per cell), stacked half-block area charts (two colored
//! pixels per cell), gauges, and inline sparklines.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::Span;

/// Braille dot bits, indexed `[column][row from top]`.
const BRAILLE_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
const EIGHTHS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
const PARTIAL_BLOCKS: [&str; 8] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];

/// One braille series: a filled area or a line. `values` has two entries per
/// cell column; `None` leaves a gap.
pub struct BrailleSeries<'a> {
    pub values: &'a [Option<f64>],
    pub color: Color,
    pub fill: bool,
}

/// Draws braille series scaled to `0..=max`. Later series draw over earlier
/// ones, and a cell takes the color of the last series with a dot in it.
pub fn braille(buffer: &mut Buffer, area: Rect, series: &[BrailleSeries], max: f64) {
    if area.is_empty() {
        return;
    }
    let width = area.width as usize;
    let height = area.height as usize;
    let dots_high = height * 4;
    let mut masks = vec![0u8; width * height];
    let mut colors: Vec<Option<Color>> = vec![None; width * height];

    for line in series {
        let mut previous: Option<usize> = None;
        let mut touched = vec![false; width * height];
        for (dot_x, value) in line.values.iter().enumerate().take(width * 2) {
            let Some(value) = value else {
                previous = None;
                continue;
            };
            let level = dot_level(*value, max, dots_high);
            let (low, high) = if line.fill {
                (0, level)
            } else {
                // A line: the dot at the value, joined to the previous column.
                let top = level.max(1);
                match previous {
                    Some(previous) => (top.min(previous.max(1)) - 1, top.max(previous.max(1))),
                    None => (top - 1, top),
                }
            };
            previous = Some(level);
            for dot_y in low..high {
                let cell_x = dot_x / 2;
                let cell_y = height - 1 - dot_y / 4;
                let index = cell_y * width + cell_x;
                masks[index] |= BRAILLE_BITS[dot_x % 2][3 - dot_y % 4];
                touched[index] = true;
            }
        }
        for (index, touched) in touched.iter().enumerate() {
            if *touched {
                colors[index] = Some(line.color);
            }
        }
    }

    for (index, mask) in masks.iter().enumerate() {
        if *mask == 0 {
            continue;
        }
        let x = area.x + (index % width) as u16;
        let y = area.y + (index / width) as u16;
        let symbol = char::from_u32(0x2800 + *mask as u32).unwrap_or(' ');
        if let Some(cell) = buffer.cell_mut((x, y)) {
            cell.set_char(symbol);
            if let Some(color) = colors[index] {
                cell.set_fg(color);
            }
        }
    }
}

/// Dots lit for `value` out of `dots` (at least one for any positive value).
fn dot_level(value: f64, max: f64, dots: usize) -> usize {
    if value.is_nan() || value <= 0.0 || max <= 0.0 {
        return 0;
    }
    let level = (value / max * dots as f64).round() as usize;
    level.clamp(1, dots)
}

/// Draws stacked layers (bottom first) as a half-block area chart scaled to
/// `0..=max`, one bucket per cell column. Columns with no data stay blank; a
/// total under half a pixel shows as a thin baseline so activity is visible.
pub fn stacked(buffer: &mut Buffer, area: Rect, layers: &[(Color, Vec<Option<f64>>)], max: f64) {
    if area.is_empty() || max <= 0.0 {
        return;
    }
    let pixels = area.height as usize * 2;
    for column in 0..area.width as usize {
        let mut tops = Vec::with_capacity(layers.len());
        let mut total = 0.0;
        let mut any = false;
        for (color, values) in layers {
            if let Some(Some(value)) = values.get(column) {
                any = true;
                total += value.max(0.0);
            }
            tops.push((*color, total));
        }
        if !any {
            continue;
        }
        let pixel_value = max / pixels as f64;
        let color_at = |pixel: usize| {
            let center = (pixel as f64 + 0.5) * pixel_value;
            tops.iter()
                .find(|(_, top)| *top > center)
                .map(|(color, _)| *color)
        };
        let x = area.x + column as u16;
        if total > 0.0 && total < pixel_value / 2.0 {
            let color = tops
                .iter()
                .scan(0.0, |below, (color, top)| {
                    let size = top - *below;
                    *below = *top;
                    Some((*color, size))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(color, _)| color);
            if let (Some(cell), Some(color)) =
                (buffer.cell_mut((x, area.y + area.height - 1)), color)
            {
                cell.set_symbol("▁").set_fg(color);
            }
            continue;
        }
        for row in 0..area.height as usize {
            let bottom_pixel = (area.height as usize - 1 - row) * 2;
            let lower = color_at(bottom_pixel);
            let upper = color_at(bottom_pixel + 1);
            let Some(cell) = buffer.cell_mut((x, area.y + row as u16)) else {
                continue;
            };
            match (upper, lower) {
                (None, None) => {}
                (Some(upper), Some(lower)) if upper == lower => {
                    cell.set_symbol("█").set_fg(upper);
                }
                (None, Some(lower)) => {
                    cell.set_symbol("▄").set_fg(lower);
                }
                (Some(upper), lower) => {
                    cell.set_symbol("▀").set_fg(upper);
                    if let Some(lower) = lower {
                        cell.set_bg(lower);
                    }
                }
            }
        }
    }
}

/// A horizontal gauge `width` cells wide filled to `fraction` with eighth-cell
/// precision.
pub fn gauge(fraction: f64, width: usize, color: Color) -> Vec<Span<'static>> {
    let fraction = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eighths = (fraction * width as f64 * 8.0).round() as usize;
    let full = eighths / 8;
    let partial = eighths % 8;
    let mut filled = "█".repeat(full);
    if partial > 0 && full < width {
        filled.push_str(PARTIAL_BLOCKS[partial]);
    }
    let used = full + usize::from(partial > 0 && full < width);
    vec![
        Span::styled(filled, Style::new().fg(color).bg(Color::Indexed(236))),
        Span::styled(
            " ".repeat(width.saturating_sub(used)),
            Style::new().bg(Color::Indexed(236)),
        ),
    ]
}

/// An inline sparkline: one eighth-block glyph per value, scaled to `max`.
/// Any positive value shows at least the lowest block; zero and gaps are blank.
pub fn sparkline(values: &[Option<f64>], max: f64) -> String {
    values
        .iter()
        .map(|value| match value {
            Some(value) if *value > 0.0 && max > 0.0 => {
                let level = ((value / max) * 8.0).round().clamp(1.0, 8.0) as usize;
                EIGHTHS[level]
            }
            _ => " ",
        })
        .collect()
}

/// A vertical histogram of `values` (each 0..=max) in one line of eighth blocks.
pub fn histogram(values: &[f64], max: f64) -> String {
    values
        .iter()
        .map(|value| {
            if *value <= 0.0 || max <= 0.0 {
                "▁"
            } else {
                EIGHTHS[((value / max) * 8.0).round().clamp(1.0, 8.0) as usize]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols(buffer: &Buffer) -> Vec<String> {
        let area = buffer.area;
        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn braille_fill_and_gaps() {
        let area = Rect::new(0, 0, 2, 1);
        let mut buffer = Buffer::empty(area);
        let values = [Some(100.0), Some(50.0), None, Some(0.0)];
        braille(
            &mut buffer,
            area,
            &[BrailleSeries {
                values: &values,
                color: Color::Cyan,
                fill: true,
            }],
            100.0,
        );
        // Left column full (4 dots), right column half (2 dots): 0x47 | 0xA0.
        assert_eq!(symbols(&buffer)[0], "⣧ ");
        assert_eq!(buffer[(0, 0)].fg, Color::Cyan);
    }

    #[test]
    fn braille_line_joins_columns() {
        let area = Rect::new(0, 0, 1, 2);
        let mut buffer = Buffer::empty(area);
        let values = [Some(0.0), Some(100.0)];
        braille(
            &mut buffer,
            area,
            &[BrailleSeries {
                values: &values,
                color: Color::Blue,
                fill: false,
            }],
            100.0,
        );
        // The right column rises through both cells.
        assert_eq!(symbols(&buffer), vec!["⢸".to_string(), "⣸".to_string()]);
    }

    #[test]
    fn stacked_layers_and_baseline() {
        let area = Rect::new(0, 0, 3, 2);
        let mut buffer = Buffer::empty(area);
        let layers = vec![
            (Color::Red, vec![Some(50.0), Some(0.1), None]),
            (Color::Green, vec![Some(25.0), Some(0.0), None]),
        ];
        stacked(&mut buffer, area, &layers, 100.0);
        let rows = symbols(&buffer);
        // Column 0: red 2 pixels, green 1 pixel: bottom cell full red, top cell lower half green.
        assert_eq!(rows[1].chars().next(), Some('█'));
        assert_eq!(buffer[(0, 1)].fg, Color::Red);
        assert_eq!(rows[0].chars().next(), Some('▄'));
        assert_eq!(buffer[(0, 0)].fg, Color::Green);
        // Column 1: a tiny total shows as a baseline; column 2 has no data.
        assert_eq!(rows[1].chars().nth(1), Some('▁'));
        assert_eq!(rows[1].chars().nth(2), Some(' '));
    }

    #[test]
    fn gauges_and_sparklines() {
        let spans = gauge(0.5, 4, Color::Cyan);
        assert_eq!(spans[0].content, "██");
        assert_eq!(spans[1].content, "  ");
        let spans = gauge(0.53, 4, Color::Cyan);
        assert_eq!(spans[0].content, "██▏");
        assert_eq!(
            sparkline(
                &[None, Some(0.0), Some(1.0), Some(1000.0), Some(5000.0)],
                1000.0
            ),
            "  ▁██"
        );
        assert_eq!(histogram(&[0.0, 50.0, 100.0], 100.0), "▁▄█");
    }
}
