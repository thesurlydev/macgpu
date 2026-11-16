//! IOReport (private `libIOReport.dylib`): GPU P-state residency, GPU energy,
//! power-controller limiter states, throttle and issue counters.
//!
//! The reader subscribes only to the GPU channels macgpu uses (about 30
//! channels rather than the 12,000 the system publishes) and converts every
//! sample into plain cumulative Rust data in [`IoReportSample`]. Deltas are
//! pure functions of two samples, so the analysis is unit tested with
//! hand-built samples. Root is not needed.
//!
//! Channel metadata (group, name, unit, state names) is read from each returned
//! sample, not from the subscription request: a subscription can drop channels
//! the driver does not support, and samples can reorder them.

use super::cf::{self, Owned};
use super::clock;
use anyhow::{Result, bail};
use core_foundation::base::TCFType;
use core_foundation_sys::array::{
    CFArrayAppendValue, CFArrayCreateMutable, CFArrayGetCount, CFArrayGetValueAtIndex, CFArrayRef,
    kCFTypeArrayCallBacks,
};
use core_foundation_sys::base::{CFEqual, CFHash, CFTypeRef, kCFAllocatorDefault};
use core_foundation_sys::dictionary::{
    CFDictionaryCreateMutableCopy, CFDictionaryGetValue, CFDictionaryRef, CFDictionarySetValue,
    CFMutableDictionaryRef,
};
use core_foundation_sys::string::CFStringRef;
use std::ffi::c_void;

type IoReportSubscriptionRef = *const c_void;

#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyAllChannels(group_id: u64, channel_id: u64) -> CFDictionaryRef;
    fn IOReportCreateSubscription(
        allocator: *const c_void,
        desired_channels: CFMutableDictionaryRef,
        subscribed: *mut CFMutableDictionaryRef,
        channel_id: u64,
        options: CFTypeRef,
    ) -> IoReportSubscriptionRef;
    fn IOReportCreateSamples(
        subscription: IoReportSubscriptionRef,
        subscribed: CFMutableDictionaryRef,
        options: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportChannelGetGroup(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetSubGroup(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetChannelName(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetUnitLabel(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetDriverID(channel: CFDictionaryRef) -> u64;
    fn IOReportChannelGetChannelID(channel: CFDictionaryRef) -> u64;
    fn IOReportChannelGetFormat(channel: CFDictionaryRef) -> i32;
    fn IOReportSimpleGetIntegerValue(channel: CFDictionaryRef, index: i32) -> i64;
    fn IOReportStateGetCount(channel: CFDictionaryRef) -> i32;
    fn IOReportStateGetNameForIndex(channel: CFDictionaryRef, index: i32) -> CFStringRef;
    fn IOReportStateGetResidency(channel: CFDictionaryRef, index: i32) -> i64;
}

/// `IOReportChannelGetFormat` values.
const FORMAT_SIMPLE: i32 = 1;
const FORMAT_STATE: i32 = 2;

/// `IOReportCopyAllChannels`: every group.
const ALL_GROUPS: u64 = 0;
/// `IOReportCopyAllChannels`: every channel in those groups.
const ALL_CHANNELS: u64 = 0;
/// `IOReportCreateSubscription`: let IOReport assign the channel ID.
const AUTO_CHANNEL_ID: u64 = 0;
/// Null selects IOReport's default allocator.
const DEFAULT_ALLOCATOR: *const c_void = std::ptr::null();
/// Null selects IOReport's default options.
const DEFAULT_OPTIONS: CFTypeRef = std::ptr::null();

/// Key of the channel array in IOReport channel and sample dictionaries.
const CHANNELS_KEY: &str = "IOReportChannels";

pub const GPU_STATS: &str = "GPU Stats";
pub const ENERGY_MODEL: &str = "Energy Model";

/// `GPU Stats` subgroups and channels macgpu reads.
pub mod channel {
    /// P-state residency: `OFF`, `P1`..`Pn`, in 24 MHz ticks.
    pub const PERFORMANCE_STATES: (&str, &str) = ("GPU Performance States", "GPUPH");
    /// Power-controller state: why the GPU runs at its current clock.
    pub const POWER_CONTROLLER: (&str, &str) = ("GPU Power Controller States", "PWRCTRL");
    /// Thermal (CLTM) performance cap: `NO_CLTM` when uncapped.
    pub const CLTM: (&str, &str) = ("CLTM-induced GPU Performance States", "GPU_CLTM");
    /// Whole-GPU energy from the AGX driver, in nJ. Unlike the PMGR `GPU`
    /// channel (mJ), it updates continuously on macOS 27 without an entitlement.
    pub const GPU_ENERGY: &str = "GPU Energy";
    pub const THROTTLE_COUNTERS: &str = "GPU Throttler Counters";
    pub const THM_COUNTERS: &str = "THM Counters";
    pub const ISSUE_COUNTS: &str = "Issue Counts";
}

/// Channels to subscribe to: (group, subgroup, channel name or `None` for
/// every channel of the subgroup). An empty subgroup matches any subgroup.
const SUBSCRIPTIONS: &[(&str, &str, Option<&str>)] = &[
    (
        GPU_STATS,
        channel::PERFORMANCE_STATES.0,
        Some(channel::PERFORMANCE_STATES.1),
    ),
    (
        GPU_STATS,
        channel::POWER_CONTROLLER.0,
        Some(channel::POWER_CONTROLLER.1),
    ),
    (GPU_STATS, channel::CLTM.0, Some(channel::CLTM.1)),
    (GPU_STATS, channel::THROTTLE_COUNTERS, None),
    (GPU_STATS, channel::THM_COUNTERS, None),
    (GPU_STATS, channel::ISSUE_COUNTS, None),
    (ENERGY_MODEL, "", Some(channel::GPU_ENERGY)),
];

/// True when a channel is one of [`SUBSCRIPTIONS`].
fn wanted(group: &str, subgroup: &str, name: &str) -> bool {
    SUBSCRIPTIONS
        .iter()
        .any(|(want_group, want_subgroup, want_name)| {
            *want_group == group
                && (want_subgroup.is_empty() || *want_subgroup == subgroup)
                && want_name.is_none_or(|want_name| want_name == name)
        })
}

/// Cumulative IOReport counters at one instant, converted to plain Rust data.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IoReportSample {
    /// `mach_absolute_time` in nanoseconds when the sample was taken.
    pub taken_at_ns: u64,
    pub channels: Vec<Channel>,
}

/// One IOReport channel's cumulative value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Channel {
    pub group: String,
    pub subgroup: String,
    /// Channel name with surrounding whitespace trimmed (some drivers pad names).
    pub name: String,
    /// Unit label, such as `nJ`, `24Mticks` or `events`. Empty when the driver gives none.
    pub unit: String,
    pub driver_id: u64,
    pub value: ChannelValue,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum ChannelValue {
    /// A single cumulative integer. `None` when IOReport returned its invalid
    /// (negative) sentinel.
    Simple(Option<u64>),
    /// Cumulative residency per state, in the channel's unit.
    State(Vec<StateResidency>),
    /// A format macgpu does not read (histograms and arrays).
    #[default]
    Other,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StateResidency {
    pub name: String,
    /// `None` when IOReport returned its invalid (negative) sentinel.
    pub residency: Option<u64>,
}

impl IoReportSample {
    /// Channels in `subgroup` (any group) named `name`.
    pub fn find<'a>(
        &'a self,
        subgroup: &'a str,
        name: &'a str,
    ) -> impl Iterator<Item = &'a Channel> {
        self.channels
            .iter()
            .filter(move |channel| channel.subgroup == subgroup && channel.name == name)
    }

    /// Channels in `subgroup` (any group).
    pub fn subgroup<'a>(&'a self, subgroup: &'a str) -> impl Iterator<Item = &'a Channel> {
        self.channels
            .iter()
            .filter(move |channel| channel.subgroup == subgroup)
    }

    /// The channel in this sample with the same identity as `other`.
    pub fn matching(&self, other: &Channel) -> Option<&Channel> {
        self.channels
            .iter()
            .find(|channel| channel.same_channel(other))
    }
}

impl Channel {
    /// Same group, subgroup, name, and driver.
    pub fn same_channel(&self, other: &Channel) -> bool {
        self.driver_id == other.driver_id
            && self.name == other.name
            && self.subgroup == other.subgroup
            && self.group == other.group
    }

    /// Residency per state between `earlier` and `self`, by state name.
    ///
    /// Returns `None` when either side is not a state channel, the state lists
    /// differ, a value is the invalid sentinel, or any state went backwards (a
    /// counter reset): the window cannot be trusted then.
    pub fn state_delta(&self, earlier: &Channel) -> Option<Vec<(String, u64)>> {
        let (ChannelValue::State(after), ChannelValue::State(before)) =
            (&self.value, &earlier.value)
        else {
            return None;
        };
        if after.len() != before.len() {
            return None;
        }
        after
            .iter()
            .zip(before)
            .map(|(after, before)| {
                if after.name != before.name {
                    return None;
                }
                let delta = after.residency?.checked_sub(before.residency?)?;
                Some((after.name.clone(), delta))
            })
            .collect()
    }

    /// Increase of a simple counter between `earlier` and `self`. `None` for a
    /// sentinel or a decrease (counter reset).
    pub fn simple_delta(&self, earlier: &Channel) -> Option<u64> {
        match (&self.value, &earlier.value) {
            (ChannelValue::Simple(after), ChannelValue::Simple(before)) => {
                after.as_ref()?.checked_sub((*before)?)
            }
            _ => None,
        }
    }
}

/// Converts an energy delta in the channel's unit to joules. `None` for an
/// unknown unit.
pub fn energy_to_joules(value: u64, unit: &str) -> Option<f64> {
    let scale = match unit.trim() {
        "J" => 1.0,
        "mJ" => 1e-3,
        "uJ" | "µJ" => 1e-6,
        "nJ" => 1e-9,
        "pJ" => 1e-12,
        _ => return None,
    };
    Some(value as f64 * scale)
}

/// An IOReport subscription to the GPU channels. Holds CoreFoundation objects,
/// so it stays on the thread that created it.
pub struct IoReportReader {
    subscription: Owned,
    subscribed: Owned,
    /// Converted metadata of each channel position in the previous sample.
    metadata: Vec<Metadata>,
}

/// A channel's identity and converted strings, reused while the channel at the
/// same position keeps the same driver, channel ID, format, state count and
/// state names. Hashing the CFString names detects a rename without converting
/// the strings on every sample.
struct Metadata {
    driver_id: u64,
    channel_id: u64,
    format: i32,
    state_count: i32,
    state_names_hash: u64,
    channel: Channel,
}

impl IoReportReader {
    pub fn new() -> Result<Self> {
        let desired = desired_channels()?;
        let mut subscribed: CFMutableDictionaryRef = std::ptr::null_mut();
        // SAFETY: the allocator and options are null by contract, the input dictionary is owned by `desired`, and the output pointer is valid; both returned references are owned by `Owned`.
        let subscription = unsafe {
            IOReportCreateSubscription(
                DEFAULT_ALLOCATOR,
                desired.as_ptr() as CFMutableDictionaryRef,
                &mut subscribed,
                AUTO_CHANNEL_ID,
                DEFAULT_OPTIONS,
            )
        };
        // SAFETY: the create rule returns a +1 reference that `Owned` releases on drop.
        let subscribed = unsafe { Owned::from_create(subscribed as CFTypeRef) };
        // SAFETY: the create rule returns a +1 reference that `Owned` releases on drop.
        let subscription = unsafe { Owned::from_create(subscription) };
        match (subscription, subscribed) {
            (Some(subscription), Some(subscribed)) => Ok(Self {
                subscription,
                subscribed,
                metadata: Vec::new(),
            }),
            _ => bail!("IOReportCreateSubscription failed"),
        }
    }

    pub fn sample(&mut self) -> Result<IoReportSample> {
        let taken_at_ns = clock::mach_now_ns();
        // SAFETY: both dictionaries are owned references from `new`, and the returned samples dictionary is owned by `Owned`.
        let samples = unsafe {
            Owned::from_create(IOReportCreateSamples(
                self.subscription.as_ptr(),
                self.subscribed.as_ptr() as CFMutableDictionaryRef,
                std::ptr::null(),
            ) as CFTypeRef)
        };
        let Some(samples) = samples else {
            bail!("IOReportCreateSamples returned null");
        };
        // SAFETY: `samples` is a valid dictionary returned by `IOReportCreateSamples` and still owned here.
        let channels = unsafe { self.convert_channels(samples.as_ptr() as CFDictionaryRef) };
        Ok(IoReportSample {
            taken_at_ns,
            channels,
        })
    }

    /// Converts every channel of a samples dictionary into plain data. Identity
    /// is checked on every sample; strings are converted again only when the
    /// channel at a position changed.
    ///
    /// # Safety
    /// `samples` must be a valid samples dictionary from `IOReportCreateSamples`.
    unsafe fn convert_channels(&mut self, samples: CFDictionaryRef) -> Vec<Channel> {
        // SAFETY: `samples` is an owned, valid samples dictionary.
        let Some(array) = (unsafe { channel_array(samples) }) else {
            return Vec::new();
        };
        // SAFETY: `array` is a valid CFArray owned by the samples dictionary.
        let count = usize::try_from(unsafe { CFArrayGetCount(array) }).unwrap_or(0);
        let mut channels = Vec::with_capacity(count);
        for index in 0..count {
            // SAFETY: `index` came from `CFArrayGetCount`, and `array` stays alive while it is read.
            let item = unsafe { CFArrayGetValueAtIndex(array, index as isize) } as CFDictionaryRef;
            if item.is_null() {
                continue;
            }
            // SAFETY: `item` is a valid channel dictionary from the samples array.
            let (driver_id, channel_id, format) = unsafe {
                (
                    IOReportChannelGetDriverID(item),
                    IOReportChannelGetChannelID(item),
                    IOReportChannelGetFormat(item),
                )
            };
            let state_count = if format == FORMAT_STATE {
                // SAFETY: `item` is a valid state channel dictionary from the samples array.
                unsafe { IOReportStateGetCount(item) }.max(0)
            } else {
                0
            };
            // State names can change while the count stays fixed, so include
            // their cheap CFString hash in the cache key.
            let names_hash = if format == FORMAT_STATE {
                // SAFETY: `item` is a valid state channel dictionary with `state_count` states.
                unsafe { state_names_hash(item, state_count) }
            } else {
                0
            };
            let cached = self.metadata.get(index).filter(|metadata| {
                metadata.driver_id == driver_id
                    && metadata.channel_id == channel_id
                    && metadata.format == format
                    && metadata.state_count == state_count
                    && metadata.state_names_hash == names_hash
            });
            let channel = match cached {
                // SAFETY: `item` has the cached format and state count, checked above.
                Some(metadata) => unsafe { refresh_values(&metadata.channel, item) },
                None => {
                    // SAFETY: `item` is a valid channel dictionary from the samples array.
                    let channel = unsafe { convert_channel(item) };
                    let metadata = Metadata {
                        driver_id,
                        channel_id,
                        format,
                        state_count,
                        state_names_hash: names_hash,
                        channel: channel.clone(),
                    };
                    if index < self.metadata.len() {
                        self.metadata[index] = metadata;
                    } else {
                        self.metadata.push(metadata);
                    }
                    channel
                }
            };
            channels.push(channel);
        }
        self.metadata.truncate(count);
        channels
    }
}

/// Builds the subscription request: the channels of [`SUBSCRIPTIONS`] in one
/// `IOReportChannels` array.
///
/// One `IOReportCopyAllChannels` call enumerates every group and channel.
/// Groups are compared as CFStrings first, so only channels of the two wanted
/// groups are converted to Rust strings.
fn desired_channels() -> Result<Owned> {
    // SAFETY: zero selects every group and channel; the +1 dictionary reference is owned by `Owned`.
    let Some(all) = (unsafe {
        Owned::from_create(IOReportCopyAllChannels(ALL_GROUPS, ALL_CHANNELS) as CFTypeRef)
    }) else {
        bail!("IOReportCopyAllChannels returned null");
    };
    // SAFETY: `all` owns a valid channel dictionary.
    let Some(array) = (unsafe { channel_array(all.as_ptr() as CFDictionaryRef) }) else {
        bail!("IOReport returned no channel list");
    };
    let groups = [cf::cfstring(GPU_STATS), cf::cfstring(ENERGY_MODEL)];
    // SAFETY: all CF objects here are valid, `selected` owns its +1 array and dictionary references, and IOReport channel strings stay alive while `all` is in scope.
    unsafe {
        let selected =
            Owned::from_create(
                CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks) as CFTypeRef,
            );
        let Some(selected) = selected else {
            bail!("CFArrayCreateMutable failed");
        };
        for index in 0..CFArrayGetCount(array) {
            let item = CFArrayGetValueAtIndex(array, index) as CFDictionaryRef;
            if item.is_null() {
                continue;
            }
            let group = IOReportChannelGetGroup(item);
            if group.is_null()
                || !groups
                    .iter()
                    .any(|wanted| CFEqual(group as CFTypeRef, wanted.as_CFTypeRef()) != 0)
            {
                continue;
            }
            // The channel strings are borrowed from `all`, which is still owned here.
            let strings = (
                cf::cfstring_to_string(group),
                cf::cfstring_to_string(IOReportChannelGetSubGroup(item)),
                cf::cfstring_to_string(IOReportChannelGetChannelName(item)),
            );
            if wanted(strings.0.trim(), strings.1.trim(), strings.2.trim()) {
                CFArrayAppendValue(selected.as_ptr() as _, item as *const c_void);
            }
        }
        if CFArrayGetCount(selected.as_ptr() as CFArrayRef) == 0 {
            bail!("IOReport has none of the GPU channels macgpu reads");
        }

        let desired = Owned::from_create(CFDictionaryCreateMutableCopy(
            kCFAllocatorDefault,
            0,
            all.as_ptr() as CFDictionaryRef,
        ) as CFTypeRef);
        let Some(desired) = desired else {
            bail!("CFDictionaryCreateMutableCopy failed");
        };
        let key = cf::cfstring(CHANNELS_KEY);
        CFDictionarySetValue(
            desired.as_ptr() as CFMutableDictionaryRef,
            key.as_concrete_TypeRef() as *const c_void,
            selected.as_ptr(),
        );
        Ok(desired)
    }
}

/// The `IOReportChannels` array of a channel or sample dictionary.
///
/// # Safety
/// `dictionary` must be a valid `CFDictionaryRef`.
unsafe fn channel_array(dictionary: CFDictionaryRef) -> Option<CFArrayRef> {
    let key = cf::cfstring(CHANNELS_KEY);
    // SAFETY: `dictionary` is valid and `key` stays alive for the borrow, whose result is used before either goes away.
    let array =
        unsafe { CFDictionaryGetValue(dictionary, key.as_concrete_TypeRef() as *const c_void) };
    (!array.is_null()).then_some(array as CFArrayRef)
}

/// A copy of `template` (same identity and state names) with the values of `item`.
///
/// # Safety
/// `item` must be a valid IOReport channel dictionary with the same format and
/// state count as `template`.
unsafe fn refresh_values(template: &Channel, item: CFDictionaryRef) -> Channel {
    let value = match &template.value {
        ChannelValue::Simple(_) => ChannelValue::Simple(non_negative(
            // SAFETY: `item` is a valid simple channel dictionary from the samples array.
            unsafe { IOReportSimpleGetIntegerValue(item, 0) },
        )),
        ChannelValue::State(states) => ChannelValue::State(
            states
                .iter()
                .enumerate()
                .map(|(index, state)| StateResidency {
                    name: state.name.clone(),
                    residency: non_negative(
                        // SAFETY: `index` is within the cached state count of this valid channel dictionary.
                        unsafe { IOReportStateGetResidency(item, index as i32) },
                    ),
                })
                .collect(),
        ),
        ChannelValue::Other => ChannelValue::Other,
    };
    Channel {
        value,
        ..template.clone()
    }
}

/// # Safety
/// `item` must be a valid IOReport channel dictionary.
unsafe fn convert_channel(item: CFDictionaryRef) -> Channel {
    // SAFETY: the caller guarantees a valid channel dictionary whose borrowed CFStrings outlive this call.
    unsafe {
        // IOReport string references are valid while the sample dictionary is owned.
        let text = |string: CFStringRef| cf::cfstring_to_string(string).trim().to_string();
        let value = match IOReportChannelGetFormat(item) {
            FORMAT_SIMPLE => {
                ChannelValue::Simple(non_negative(IOReportSimpleGetIntegerValue(item, 0)))
            }
            FORMAT_STATE => {
                let count = IOReportStateGetCount(item).max(0);
                ChannelValue::State(
                    (0..count)
                        .map(|state| {
                            let name = IOReportStateGetNameForIndex(item, state);
                            StateResidency {
                                name: if name.is_null() {
                                    format!("S{state}")
                                } else {
                                    text(name)
                                },
                                residency: non_negative(IOReportStateGetResidency(item, state)),
                            }
                        })
                        .collect(),
                )
            }
            _ => ChannelValue::Other,
        };
        Channel {
            group: text(IOReportChannelGetGroup(item)),
            subgroup: text(IOReportChannelGetSubGroup(item)),
            name: text(IOReportChannelGetChannelName(item)),
            unit: text(IOReportChannelGetUnitLabel(item)),
            driver_id: IOReportChannelGetDriverID(item),
            value,
        }
    }
}

/// Cheap hash of a state channel's names, from CFString's own hash.
///
/// # Safety
/// `item` must be a valid state channel dictionary with `state_count` states.
unsafe fn state_names_hash(item: CFDictionaryRef, state_count: i32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for index in 0..state_count {
        // SAFETY: `index` is within the state count of this valid channel dictionary.
        let name = unsafe { IOReportStateGetNameForIndex(item, index) };
        let code = if name.is_null() {
            0
        } else {
            // SAFETY: `name` is a valid CFString borrowed from the channel dictionary.
            let value = unsafe { CFHash(name as CFTypeRef) };
            value as u64
        };
        hash = (hash ^ code).wrapping_mul(1_099_511_628_211u64);
    }
    hash
}

/// IOReport marks invalid values with a negative sentinel (`INT64_MIN`).
fn non_negative(value: i64) -> Option<u64> {
    u64::try_from(value).ok()
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub fn state_channel(subgroup: &str, name: &str, states: &[(&str, u64)]) -> Channel {
        Channel {
            group: GPU_STATS.into(),
            subgroup: subgroup.into(),
            name: name.into(),
            unit: "24Mticks".into(),
            driver_id: 7,
            value: ChannelValue::State(
                states
                    .iter()
                    .map(|(name, residency)| StateResidency {
                        name: name.to_string(),
                        residency: Some(*residency),
                    })
                    .collect(),
            ),
        }
    }

    pub fn simple_channel(
        group: &str,
        subgroup: &str,
        name: &str,
        unit: &str,
        value: u64,
    ) -> Channel {
        Channel {
            group: group.into(),
            subgroup: subgroup.into(),
            name: name.into(),
            unit: unit.into(),
            driver_id: 7,
            value: ChannelValue::Simple(Some(value)),
        }
    }

    pub fn energy(nanojoules: u64) -> Channel {
        simple_channel(ENERGY_MODEL, "", channel::GPU_ENERGY, "nJ", nanojoules)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn state_delta_by_name() {
        let before = state_channel(
            "GPU Performance States",
            "GPUPH",
            &[("OFF", 100), ("P1", 10)],
        );
        let after = state_channel(
            "GPU Performance States",
            "GPUPH",
            &[("OFF", 150), ("P1", 60)],
        );
        assert_eq!(
            after.state_delta(&before),
            Some(vec![("OFF".to_string(), 50), ("P1".to_string(), 50)])
        );
    }

    #[test]
    fn state_delta_rejects_resets_sentinels_and_shape_changes() {
        let before = state_channel("s", "c", &[("OFF", 100), ("P1", 10)]);
        let reset = state_channel("s", "c", &[("OFF", 5), ("P1", 20)]);
        assert_eq!(reset.state_delta(&before), None);

        let mut sentinel = state_channel("s", "c", &[("OFF", 200), ("P1", 20)]);
        if let ChannelValue::State(states) = &mut sentinel.value {
            states[1].residency = None;
        }
        assert_eq!(sentinel.state_delta(&before), None);

        let renamed = state_channel("s", "c", &[("OFF", 200), ("P2", 20)]);
        assert_eq!(renamed.state_delta(&before), None);
        let longer = state_channel("s", "c", &[("OFF", 200), ("P1", 20), ("P2", 1)]);
        assert_eq!(longer.state_delta(&before), None);
    }

    #[test]
    fn simple_delta_handles_sentinel_and_reset() {
        let before = energy(1_000);
        assert_eq!(energy(4_000).simple_delta(&before), Some(3_000));
        assert_eq!(energy(500).simple_delta(&before), None);
        let mut sentinel = energy(0);
        sentinel.value = ChannelValue::Simple(None);
        assert_eq!(sentinel.simple_delta(&before), None);
        assert_eq!(before.simple_delta(&sentinel), None);
        assert_eq!(non_negative(i64::MIN), None);
        assert_eq!(non_negative(42), Some(42));
    }

    #[test]
    fn selects_only_wanted_channels() {
        assert!(wanted(GPU_STATS, "GPU Performance States", "GPUPH"));
        assert!(!wanted(GPU_STATS, "GPU Performance States", "GPU_SW"));
        assert!(wanted(GPU_STATS, channel::ISSUE_COUNTS, "Total Restarts"));
        assert!(wanted(ENERGY_MODEL, "", channel::GPU_ENERGY));
        assert!(!wanted(ENERGY_MODEL, "", "GPU"));
        assert!(!wanted(
            "CPU Stats",
            channel::ISSUE_COUNTS,
            "Total Restarts"
        ));
    }

    #[test]
    fn converts_energy_units() {
        assert_eq!(energy_to_joules(2_000_000_000, "nJ"), Some(2.0));
        assert_eq!(energy_to_joules(1_500, "mJ"), Some(1.5));
        assert_eq!(energy_to_joules(3_000_000, "uJ"), Some(3.0));
        assert_eq!(energy_to_joules(1, "events"), None);
    }

    #[test]
    fn finds_channels_by_identity() {
        let sample = IoReportSample {
            taken_at_ns: 0,
            channels: vec![
                energy(5),
                state_channel("GPU Performance States", "GPUPH", &[]),
            ],
        };
        assert_eq!(sample.find("", channel::GPU_ENERGY).count(), 1);
        assert!(sample.matching(&energy(9)).is_some());
        let mut other_driver = energy(9);
        other_driver.driver_id = 8;
        assert!(sample.matching(&other_driver).is_none());
    }

    #[test]
    #[ignore = "needs Apple Silicon hardware and the IOReport dylib"]
    fn samples_live_gpu_channels() {
        let mut reader = IoReportReader::new().expect("subscribe to IOReport");
        let first = reader.sample().expect("first sample");
        std::thread::sleep(std::time::Duration::from_millis(200));
        let started = std::time::Instant::now();
        let cpu = clock::thread_cpu_ns();
        let second = reader.sample().expect("second sample");
        let wall = started.elapsed();
        let cpu = clock::thread_cpu_ns().saturating_sub(cpu);
        eprintln!(
            "IOReport sample: {} channels, {:.3} ms wall, {:.3} ms cpu",
            second.channels.len(),
            wall.as_secs_f64() * 1e3,
            cpu as f64 / 1e6
        );

        let started = std::time::Instant::now();
        for _ in 0..50 {
            reader.sample().expect("sample");
        }
        eprintln!(
            "IOReport sample, back to back: {:.3} ms",
            started.elapsed().as_secs_f64() * 1e3 / 50.0
        );

        let (subgroup, name) = channel::PERFORMANCE_STATES;
        let before = first
            .find(subgroup, name)
            .next()
            .expect("GPUPH in first sample");
        let after = second.matching(before).expect("GPUPH in second sample");
        let delta = after.state_delta(before).expect("GPUPH delta");
        assert_eq!(delta[0].0, "OFF");
        let ticks: u64 = delta.iter().map(|(_, ticks)| ticks).sum();
        assert!(ticks > 0, "GPUPH did not advance");

        let energy = second
            .find("", channel::GPU_ENERGY)
            .next()
            .expect("GPU Energy channel");
        assert_eq!(energy.unit, "nJ");
        assert!(second.subgroup(channel::ISSUE_COUNTS).count() > 0);
        assert!(second.channels.len() < 100, "subscription is not GPU-only");
    }
}
