//! SMC temperature sensors (GPU `Tg*` keys).
//!
//! Talks to the `AppleSMC` user client with `IOConnectCallStructMethod`
//! selector 2, which needs no root. The SMC exposes thousands of four-letter
//! keys; the GPU die sensors are the 4-byte float (`flt `) keys starting with
//! `Tg`. They come in pairs about 5.6 C apart and track GPU load (about 60 C
//! idle and 93 C average under full load on an M4 Max).
//!
//! The key list is sorted, so the reader binary-searches the `Tg` range once at
//! startup, caches each key's info, and then needs one SMC call per sensor per
//! read.

use super::iokit::{self, IoObject, KERN_SUCCESS, io_object_t};
use anyhow::{Context, Result, bail};
use std::ffi::c_void;

/// The `AppleSMC` user-client method that takes and returns [`KeyData`].
const SELECTOR_HANDLE_YPC_EVENT: u32 = 2;
/// `KeyData::data8` commands.
const CMD_READ_BYTES: u8 = 5;
const CMD_READ_INDEX: u8 = 8;
const CMD_READ_KEY_INFO: u8 = 9;

/// Prefix of the GPU temperature keys.
const GPU_PREFIX: &[u8; 2] = b"Tg";
/// `flt `: a little-endian f32.
const TYPE_FLOAT: u32 = u32::from_be_bytes(*b"flt ");

/// Readings outside this range are treated as invalid rather than averaged.
///
/// While the GPU idles its sensors sometimes stop reporting for about a second
/// at a time, all at once: on an M4 Max (macOS 27), twice in 4 minutes of idle,
/// every `Tg` key read one of the fixed values -4.0, 0.0, 1.6, or 1.8 C between
/// readings of 50-60 C. A running die is never colder than its surroundings,
/// so anything below 10 C is one of these placeholders, not a temperature.
pub const MIN_PLAUSIBLE_C: f64 = 10.0;
const MAX_PLAUSIBLE_C: f64 = 150.0;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Version {
    major: u8,
    minor: u8,
    build: u8,
    reserved: u8,
    release: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct PowerLimit {
    version: u16,
    length: u16,
    cpu: u32,
    gpu: u32,
    memory: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct KeyInfo {
    data_size: u32,
    data_type: u32,
    data_attributes: u8,
}

/// `SMCKeyData_t`: the 80-byte struct the SMC user client exchanges.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct KeyData {
    key: u32,
    version: Version,
    power_limit: PowerLimit,
    key_info: KeyInfo,
    result: u8,
    status: u8,
    data8: u8,
    data32: u32,
    bytes: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<KeyData>() == 80);

/// One GPU sensor reading.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SensorReading {
    /// SMC key, such as `Tg0K`.
    pub key: String,
    pub celsius: f64,
}

/// GPU die sensor readings at one instant.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuTemperatures {
    pub sensors: Vec<SensorReading>,
}

impl GpuTemperatures {
    /// Mean and maximum over the plausible readings, and how many there were.
    /// `None` when no reading is plausible.
    pub fn summary(&self) -> Option<(f64, f64, u32)> {
        let valid: Vec<f64> = self
            .sensors
            .iter()
            .map(|sensor| sensor.celsius)
            .filter(|celsius| (MIN_PLAUSIBLE_C..=MAX_PLAUSIBLE_C).contains(celsius))
            .collect();
        if valid.is_empty() {
            return None;
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        let max = valid.iter().copied().fold(f64::MIN, f64::max);
        Some((mean, max, valid.len() as u32))
    }

    /// Whether every sensor read below [`MIN_PLAUSIBLE_C`]: the sensors are not
    /// reporting, as happens for about a second at a time while the GPU idles.
    pub fn not_reporting(&self) -> bool {
        !self.sensors.is_empty()
            && self
                .sensors
                .iter()
                .all(|sensor| sensor.celsius < MIN_PLAUSIBLE_C)
    }
}

/// An open SMC connection, closed on drop.
struct Connection(io_object_t);

impl Drop for Connection {
    fn drop(&mut self) {
        unsafe { iokit::IOServiceClose(self.0) };
    }
}

impl Connection {
    fn open() -> Result<Self> {
        let service: IoObject =
            iokit::matching_service("AppleSMC").context("no AppleSMC service")?;
        let mut connection = 0;
        let result = unsafe {
            iokit::IOServiceOpen(
                service.raw(),
                mach2::traps::mach_task_self(),
                0,
                &mut connection,
            )
        };
        if result != KERN_SUCCESS || connection == 0 {
            bail!("opening the AppleSMC user client failed: {result:#x}");
        }
        Ok(Self(connection))
    }

    fn call(&self, input: &KeyData) -> Option<KeyData> {
        let mut output = KeyData::default();
        let mut output_size = std::mem::size_of::<KeyData>();
        let result = unsafe {
            iokit::IOConnectCallStructMethod(
                self.0,
                SELECTOR_HANDLE_YPC_EVENT,
                (input as *const KeyData).cast::<c_void>(),
                std::mem::size_of::<KeyData>(),
                (&raw mut output).cast::<c_void>(),
                &mut output_size,
            )
        };
        (result == KERN_SUCCESS && output.result == 0).then_some(output)
    }

    fn key_at(&self, index: u32) -> Option<u32> {
        self.call(&KeyData {
            data8: CMD_READ_INDEX,
            data32: index,
            ..KeyData::default()
        })
        .map(|output| output.key)
    }

    fn key_info(&self, key: u32) -> Option<KeyInfo> {
        self.call(&KeyData {
            key,
            data8: CMD_READ_KEY_INFO,
            ..KeyData::default()
        })
        .map(|output| output.key_info)
    }

    fn read_bytes(&self, key: u32, info: KeyInfo) -> Option<[u8; 32]> {
        self.call(&KeyData {
            key,
            key_info: info,
            data8: CMD_READ_BYTES,
            ..KeyData::default()
        })
        .map(|output| output.bytes)
    }

    fn key_count(&self) -> Option<u32> {
        let key = u32::from_be_bytes(*b"#KEY");
        let info = self.key_info(key)?;
        let bytes = self.read_bytes(key, info)?;
        Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

struct Sensor {
    key: u32,
    name: String,
    info: KeyInfo,
}

/// Reads the GPU temperature sensors through a cached SMC connection.
pub struct SmcReader {
    connection: Connection,
    sensors: Vec<Sensor>,
}

impl SmcReader {
    pub fn new() -> Result<Self> {
        let connection = Connection::open()?;
        let count = connection
            .key_count()
            .context("reading the SMC key count")?;
        let keys = gpu_keys(&connection, count);
        let sensors: Vec<Sensor> = keys
            .into_iter()
            .filter_map(|key| {
                let info = connection.key_info(key)?;
                (info.data_type == TYPE_FLOAT && info.data_size == 4).then(|| Sensor {
                    key,
                    name: key_name(key),
                    info,
                })
            })
            .collect();
        if sensors.is_empty() {
            bail!("the SMC has no Tg* float GPU temperature keys");
        }
        Ok(Self {
            connection,
            sensors,
        })
    }

    /// Number of GPU sensors found at startup.
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    pub fn read(&mut self) -> Result<GpuTemperatures> {
        let sensors: Vec<SensorReading> = self
            .sensors
            .iter()
            .filter_map(|sensor| {
                let bytes = self.connection.read_bytes(sensor.key, sensor.info)?;
                let celsius = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                celsius.is_finite().then(|| SensorReading {
                    key: sensor.name.clone(),
                    celsius: f64::from(celsius),
                })
            })
            .collect();
        if sensors.is_empty() {
            bail!("no GPU temperature sensor could be read");
        }
        Ok(GpuTemperatures { sensors })
    }
}

/// Keys starting with `Tg`. The SMC key index is sorted by key, so the range
/// is found by binary search; if that finds nothing (an unsorted SMC), every
/// key is scanned, which costs about 30 ms once.
fn gpu_keys(connection: &Connection, count: u32) -> Vec<u32> {
    let lower = u32::from_be_bytes([GPU_PREFIX[0], GPU_PREFIX[1], 0, 0]);
    let mut low = 0u32;
    let mut high = count;
    while low < high {
        let middle = low + (high - low) / 2;
        match connection.key_at(middle) {
            Some(key) if key < lower => low = middle + 1,
            Some(_) => high = middle,
            None => break,
        }
    }
    let keys: Vec<u32> = (low..count)
        .map_while(|index| connection.key_at(index))
        .take_while(|key| has_gpu_prefix(*key))
        .collect();
    if !keys.is_empty() {
        return keys;
    }
    (0..count)
        .filter_map(|index| connection.key_at(index))
        .filter(|key| has_gpu_prefix(*key))
        .collect()
}

fn has_gpu_prefix(key: u32) -> bool {
    key.to_be_bytes()[..2] == GPU_PREFIX[..]
}

fn key_name(key: u32) -> String {
    String::from_utf8_lossy(&key.to_be_bytes()).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(key: &str, celsius: f64) -> SensorReading {
        SensorReading {
            key: key.into(),
            celsius,
        }
    }

    #[test]
    fn summarizes_plausible_readings_only() {
        let temperatures = GpuTemperatures {
            sensors: vec![
                reading("Tg04", 50.0),
                reading("Tg05", 56.0),
                reading("Tg0K", 0.0),
                reading("Tg0L", 400.0),
            ],
        };
        assert_eq!(temperatures.summary(), Some((53.0, 56.0, 2)));
        assert!(!temperatures.not_reporting());
        assert_eq!(GpuTemperatures::default().summary(), None);
        assert!(!GpuTemperatures::default().not_reporting());
    }

    #[test]
    fn idle_placeholders_are_not_temperatures() {
        // What all 44 sensors of an idle M4 Max read for about a second.
        let placeholders = GpuTemperatures {
            sensors: [-4.0, 1.6, 0.0, 1.8, 1.6, -4.0]
                .iter()
                .enumerate()
                .map(|(index, celsius)| reading(&format!("Tg{index}"), *celsius))
                .collect(),
        };
        assert_eq!(placeholders.summary(), None);
        assert!(placeholders.not_reporting());

        let mut mixed = placeholders.clone();
        mixed.sensors[0].celsius = 52.0;
        assert_eq!(mixed.summary(), Some((52.0, 52.0, 1)));
        assert!(!mixed.not_reporting());
    }

    #[test]
    fn key_names_and_prefix() {
        assert_eq!(key_name(u32::from_be_bytes(*b"Tg0K")), "Tg0K");
        assert!(has_gpu_prefix(u32::from_be_bytes(*b"Tg0K")));
        assert!(!has_gpu_prefix(u32::from_be_bytes(*b"TG0B")));
        assert!(!has_gpu_prefix(u32::from_be_bytes(*b"Tp01")));
    }

    #[test]
    #[ignore]
    fn reads_live_gpu_sensors() {
        let started = std::time::Instant::now();
        let mut reader = SmcReader::new().expect("open SMC");
        let startup = started.elapsed();
        reader.read().expect("warm-up read");
        let started = std::time::Instant::now();
        let cpu = crate::sys::clock::thread_cpu_ns();
        let temperatures = reader.read().expect("read sensors");
        let read = started.elapsed();
        let cpu = crate::sys::clock::thread_cpu_ns().saturating_sub(cpu);
        let (mean, max, count) = temperatures.summary().expect("plausible readings");
        eprintln!(
            "SMC: {} sensors, startup {:.1} ms, read {:.2} ms wall {:.2} ms cpu, avg {mean:.1} C, max {max:.1} C",
            reader.sensor_count(),
            startup.as_secs_f64() * 1e3,
            read.as_secs_f64() * 1e3,
            cpu as f64 / 1e6
        );
        for sensor in &temperatures.sensors {
            eprint!("{}={:.1} ", sensor.key, sensor.celsius);
        }
        eprintln!();
        assert!(count > 0);
        assert!(mean <= max);
    }
}
