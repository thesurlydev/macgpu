//! Key and mouse handling for the interactive view.

use super::app::{
    App, DOUBLE_CLICK, Effect, Overlay, PAGE_ROWS, SIGNALS, Screen, StatusLevel, WHEEL_ROWS,
};
use crate::view::Filter;
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use std::time::Instant;

impl App {
    pub fn handle_key(&mut self, key: KeyEvent) {
        if key.kind == KeyEventKind::Release {
            return;
        }
        self.dirty = true;
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('q'))
        {
            self.effects.push(Effect::Quit);
            return;
        }
        match &self.overlay {
            Overlay::Help { .. } => return self.help_key(key),
            Overlay::Filter { .. } => return self.filter_key(key),
            Overlay::Confirm(_) => return self.confirm_key(key),
            Overlay::None => {}
        }
        if matches!(self.screen, Screen::Detail(_)) && self.detail_key(key) {
            return;
        }
        self.main_key(key);
    }

    /// Keys shared by both screens.
    fn main_key(&mut self, key: KeyEvent) {
        let page = self.page_rows();
        match key.code {
            KeyCode::Char('q') => self.effects.push(Effect::Quit),
            KeyCode::Char('?') | KeyCode::F(1) => self.overlay = Overlay::Help { scroll: 0 },
            KeyCode::Char('/') => {
                self.overlay = Overlay::Filter {
                    input: self.filter_text.clone(),
                    previous: self.filter_text.clone(),
                }
            }
            KeyCode::Esc if !self.filter.is_empty() => self.apply_filter(""),
            KeyCode::Char('s') | KeyCode::Char('.') | KeyCode::Char('>') => {
                self.set_sort(self.sort.step(true), false)
            }
            KeyCode::Char(',') | KeyCode::Char('<') => self.set_sort(self.sort.step(false), false),
            KeyCode::Char('S') | KeyCode::Char('r') => {
                self.reverse = !self.reverse;
                self.rebuild_rows();
                self.sort_status();
            }
            KeyCode::Char('g') => self.set_group(self.group.next()),
            KeyCode::Char('G') => self.set_group(self.group.prev()),
            KeyCode::Char('t') => {
                self.tree = !self.tree;
                self.rebuild_rows();
                self.set_status(
                    if self.tree { "tree view" } else { "flat view" },
                    StatusLevel::Info,
                );
            }
            KeyCode::Char('f') | KeyCode::Char(' ') => self.toggle_freeze(),
            KeyCode::Char('z') => self.step_zoom(true),
            KeyCode::Char('Z') => self.step_zoom(false),
            KeyCode::Char('+') | KeyCode::Char('=') => self.step_interval(true),
            KeyCode::Char('-') | KeyCode::Char('_') => self.step_interval(false),
            KeyCode::Char('a') => {
                self.include_idle = !self.include_idle;
                self.effects.push(Effect::SetIncludeIdle(self.include_idle));
                let what = if self.include_idle {
                    "showing idle GPU clients"
                } else {
                    "hiding idle GPU clients"
                };
                let note = if self.live {
                    ""
                } else {
                    " (a recording decides this)"
                };
                self.set_status(format!("{what}{note}"), StatusLevel::Info);
            }
            KeyCode::Char('k') => self.prompt_signal(),
            KeyCode::Enter => self.open_detail(),
            KeyCode::Up => self.move_selection(-1),
            KeyCode::Down => self.move_selection(1),
            KeyCode::PageUp => self.move_selection(-(page as isize)),
            KeyCode::PageDown => self.move_selection(page as isize),
            KeyCode::Home => self.select_index(0),
            KeyCode::End => self.select_index(self.rows.len().saturating_sub(1)),
            _ => {}
        }
    }

    /// Detail-screen keys. Returns false to fall through to the shared keys.
    fn detail_key(&mut self, key: KeyEvent) -> bool {
        let Screen::Detail(detail) = &mut self.screen else {
            return false;
        };
        match key.code {
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Left | KeyCode::Enter => {
                self.screen = Screen::Main
            }
            KeyCode::Up => detail.scroll = detail.scroll.saturating_sub(1),
            KeyCode::Down => detail.scroll = detail.scroll.saturating_add(1),
            KeyCode::PageUp => detail.scroll = detail.scroll.saturating_sub(PAGE_ROWS),
            KeyCode::PageDown => detail.scroll = detail.scroll.saturating_add(PAGE_ROWS),
            KeyCode::Home => detail.scroll = 0,
            // The draw clamps this to the last line of the current breakdown.
            KeyCode::End => detail.scroll = u16::MAX,
            KeyCode::Char('g' | 'G' | 't' | '/' | 's' | 'S' | '.' | ',' | '<' | '>' | 'r') => {}
            _ => return false,
        }
        true
    }

    fn help_key(&mut self, key: KeyEvent) {
        let Overlay::Help { scroll } = &mut self.overlay else {
            return;
        };
        match key.code {
            KeyCode::Up => *scroll = scroll.saturating_sub(1),
            KeyCode::Down => *scroll = scroll.saturating_add(1),
            KeyCode::PageUp => *scroll = scroll.saturating_sub(PAGE_ROWS),
            KeyCode::PageDown | KeyCode::Char(' ') => *scroll = scroll.saturating_add(PAGE_ROWS),
            KeyCode::Home => *scroll = 0,
            _ => self.overlay = Overlay::None,
        }
    }

    fn filter_key(&mut self, key: KeyEvent) {
        let Overlay::Filter { input, previous } = &mut self.overlay else {
            return;
        };
        match key.code {
            KeyCode::Enter => {
                let text = input.clone();
                self.overlay = Overlay::None;
                self.apply_filter(&text);
                return;
            }
            KeyCode::Esc => {
                let text = previous.clone();
                self.overlay = Overlay::None;
                self.apply_filter(&text);
                return;
            }
            KeyCode::Backspace => {
                input.pop();
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => input.clear(),
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let trimmed = input.trim_end().len();
                let cut = input[..trimmed]
                    .rfind([' ', ','])
                    .map(|index| index + 1)
                    .unwrap_or(0);
                input.truncate(cut);
            }
            KeyCode::Char(character)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                input.push(character)
            }
            _ => return,
        }
        // Apply as you type.
        let text = input.clone();
        self.filter = Filter::parse(&text);
        self.rebuild_rows();
    }

    fn confirm_key(&mut self, key: KeyEvent) {
        let Overlay::Confirm(confirm) = &mut self.overlay else {
            return;
        };
        match key.code {
            KeyCode::Left | KeyCode::BackTab | KeyCode::Up => {
                confirm.signal = (confirm.signal + SIGNALS.len() - 1) % SIGNALS.len()
            }
            KeyCode::Right | KeyCode::Tab | KeyCode::Down => {
                confirm.signal = (confirm.signal + 1) % SIGNALS.len()
            }
            KeyCode::Char('9') | KeyCode::Char('K') => confirm.signal = 1,
            KeyCode::Enter | KeyCode::Char('y') => {
                let confirm = confirm.clone();
                self.overlay = Overlay::None;
                self.effects.push(Effect::Signal {
                    pid: confirm.pid,
                    start_us: confirm.start_us,
                    signal: SIGNALS[confirm.signal].0,
                    name: confirm.name,
                });
            }
            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('q') => {
                self.overlay = Overlay::None;
                self.set_status("signal cancelled", StatusLevel::Info);
            }
            _ => {}
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        if self.overlay != Overlay::None {
            return;
        }
        if let Screen::Detail(detail) = &mut self.screen {
            match mouse.kind {
                MouseEventKind::ScrollDown => {
                    detail.scroll = detail.scroll.saturating_add(WHEEL_ROWS as u16)
                }
                MouseEventKind::ScrollUp => {
                    detail.scroll = detail.scroll.saturating_sub(WHEEL_ROWS as u16)
                }
                _ => return,
            }
            self.dirty = true;
            return;
        }
        let table = self.hitboxes.table;
        match mouse.kind {
            MouseEventKind::ScrollDown => self.move_selection(WHEEL_ROWS),
            MouseEventKind::ScrollUp => self.move_selection(-WHEEL_ROWS),
            MouseEventKind::Down(MouseButton::Left) => {
                if mouse.row == self.hitboxes.header_y && table.height > 0 {
                    let column = self
                        .hitboxes
                        .header
                        .iter()
                        .find(|(x, width, _)| mouse.column >= *x && mouse.column < x + width)
                        .and_then(|(_, _, column)| column.sort_key());
                    if let Some(sort) = column {
                        if sort == self.sort {
                            self.reverse = !self.reverse;
                            self.rebuild_rows();
                            self.sort_status();
                        } else {
                            self.set_sort(sort, true);
                        }
                    }
                    return;
                }
                let inside = mouse.column >= table.x
                    && mouse.column < table.x + table.width
                    && mouse.row >= table.y
                    && mouse.row < table.y + table.height;
                if !inside {
                    return;
                }
                let index = self.scroll + (mouse.row - table.y) as usize;
                if index >= self.rows.len() {
                    return;
                }
                self.select_index(index);
                let key = self.rows[index].row.key;
                let now = Instant::now();
                let double = self.last_click.is_some_and(|(at, last)| {
                    last == key && now.duration_since(at) <= DOUBLE_CLICK
                });
                self.last_click = Some((now, key));
                if double {
                    self.last_click = None;
                    self.open_detail();
                }
            }
            _ => {}
        }
    }

    /// Rows that fit the table, for paging.
    fn page_rows(&self) -> usize {
        (self.hitboxes.table.height as usize).max(1)
    }
}
