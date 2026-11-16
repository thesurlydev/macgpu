//! Colors for the interactive view.
//!
//! One accent per metric, used everywhere that metric appears: GPU time and
//! active % cyan, frequency-normalized % blue, power yellow, clock magenta,
//! memory green, temperature light red. Red is reserved for limits, alerts,
//! and errors.

use ratatui::style::{Color, Modifier, Style};

pub const GPU: Color = Color::Cyan;
pub const NORMALIZED: Color = Color::Blue;
pub const POWER: Color = Color::Yellow;
pub const CLOCK: Color = Color::Magenta;
pub const MEMORY: Color = Color::Green;
pub const TEMPERATURE: Color = Color::LightRed;
pub const ALERT: Color = Color::Red;
pub const WARNING: Color = Color::LightYellow;
pub const LABEL: Color = Color::Gray;
pub const DIM: Color = Color::DarkGray;
pub const SELECTED_BG: Color = Color::Indexed(237);

/// Colors for the "who used the GPU" series, in rank order, and for the rest.
pub const SERIES: [Color; 6] = [
    Color::Indexed(39),  // blue
    Color::Indexed(214), // orange
    Color::Indexed(77),  // green
    Color::Indexed(170), // orchid
    Color::Indexed(44),  // teal
    Color::Indexed(180), // tan
];
pub const SERIES_OTHER: Color = Color::Indexed(242);

pub fn dim() -> Style {
    Style::new().fg(DIM)
}

pub fn label() -> Style {
    Style::new().fg(LABEL)
}

pub fn accent(color: Color) -> Style {
    Style::new().fg(color)
}

pub fn strong(color: Color) -> Style {
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

pub fn title() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}
