//! Sort keys and the comparator that orders rows.

use crate::view::Row;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortKey {
    Gpu,
    Power,
    Memory,
    Children,
    Last,
    Lifetime,
    Pid,
    Name,
}

impl SortKey {
    pub const ALL: [SortKey; 8] = [
        SortKey::Gpu,
        SortKey::Power,
        SortKey::Memory,
        SortKey::Children,
        SortKey::Last,
        SortKey::Lifetime,
        SortKey::Pid,
        SortKey::Name,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SortKey::Gpu => "GPU ms/s",
            SortKey::Power => "power",
            SortKey::Memory => "GPU memory",
            SortKey::Children => "queues",
            SortKey::Last => "last submit",
            SortKey::Lifetime => "lifetime",
            SortKey::Pid => "PID",
            SortKey::Name => "name",
        }
    }

    pub(super) fn step(self, forward: bool) -> SortKey {
        let index = Self::ALL.iter().position(|key| *key == self).unwrap_or(0);
        let len = Self::ALL.len();
        Self::ALL[if forward {
            (index + 1) % len
        } else {
            (index + len - 1) % len
        }]
    }

    /// Orders `a` before `b` in this key's natural direction: biggest first for
    /// amounts, most recent first for submissions, ascending for PID and name.
    /// Missing values sort last in either direction.
    pub(super) fn compare(self, a: &Row, b: &Row, reverse: bool) -> Ordering {
        fn optional<T: PartialOrd>(
            a: Option<T>,
            b: Option<T>,
            descending: bool,
            reverse: bool,
        ) -> Ordering {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let order = a.partial_cmp(&b).unwrap_or(Ordering::Equal);
                    let order = if descending { order.reverse() } else { order };
                    if reverse { order.reverse() } else { order }
                }
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            }
        }
        let primary = match self {
            SortKey::Gpu => optional(Some(a.gpu_ns), Some(b.gpu_ns), true, reverse),
            SortKey::Power => optional(a.power_w, b.power_w, true, reverse),
            SortKey::Memory => optional(a.memory_bytes, b.memory_bytes, true, reverse),
            SortKey::Children => optional(Some(a.children), Some(b.children), true, reverse),
            SortKey::Last => optional(a.last_submit_age_ms, b.last_submit_age_ms, false, reverse),
            SortKey::Lifetime => optional(
                Some(a.lifetime_gpu_ns),
                Some(b.lifetime_gpu_ns),
                true,
                reverse,
            ),
            SortKey::Pid => optional(a.pid, b.pid, false, reverse),
            SortKey::Name => optional(
                Some(a.name.to_lowercase()),
                Some(b.name.to_lowercase()),
                false,
                reverse,
            ),
        };
        primary
            .then_with(|| b.gpu_ns.cmp(&a.gpu_ns))
            .then_with(|| b.lifetime_gpu_ns.cmp(&a.lifetime_gpu_ns))
            .then_with(|| a.key.cmp(&b.key))
    }
}
