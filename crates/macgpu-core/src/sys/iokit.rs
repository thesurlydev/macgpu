//! IOKit FFI and the GPU registry reader.
//!
//! The reader walks each `IOAccelerator` service's children in-process and
//! reads the properties macgpu needs. Static properties are cached per
//! accelerator; performance counters and client queues are read every sample.

use super::cf::{self, Owned, Value};
use anyhow::{Result, bail};
use core_foundation::base::TCFType;
use core_foundation_sys::base::{CFAllocatorRef, CFTypeRef, kCFAllocatorDefault};
use core_foundation_sys::dictionary::{CFDictionaryRef, CFMutableDictionaryRef};
use core_foundation_sys::string::CFStringRef;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Arc;

pub use mach2::kern_return::{KERN_SUCCESS, kern_return_t};
pub use mach2::port::mach_port_t;

#[allow(non_camel_case_types)]
/// An IOKit object handle, represented by the same integer type as a Mach port.
pub type io_object_t = mach_port_t;
#[allow(non_camel_case_types)]
/// A user-client connection returned by `IOServiceOpen`.
pub type io_connect_t = mach_port_t;
/// `kIOMainPortDefault` is `MACH_PORT_NULL`.
pub const MAIN_PORT_DEFAULT: mach_port_t = mach2::port::MACH_PORT_NULL;
const SERVICE_PLANE: &CStr = c"IOService";

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    pub fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
    pub fn IOServiceGetMatchingServices(
        main_port: mach_port_t,
        matching: CFDictionaryRef,
        existing: *mut io_object_t,
    ) -> kern_return_t;
    pub fn IOServiceGetMatchingService(
        main_port: mach_port_t,
        matching: CFDictionaryRef,
    ) -> io_object_t;
    pub fn IOIteratorNext(iterator: io_object_t) -> io_object_t;
    pub fn IOIteratorIsValid(iterator: io_object_t) -> u32;
    pub fn IOObjectRelease(object: io_object_t) -> kern_return_t;
    pub fn IOObjectGetClass(object: io_object_t, class_name: *mut c_char) -> kern_return_t;
    pub fn IORegistryEntryGetChildIterator(
        entry: io_object_t,
        plane: *const c_char,
        iterator: *mut io_object_t,
    ) -> kern_return_t;
    pub fn IORegistryEntryCreateCFProperty(
        entry: io_object_t,
        key: CFStringRef,
        allocator: CFAllocatorRef,
        options: u32,
    ) -> CFTypeRef;
    pub fn IORegistryEntryGetRegistryEntryID(entry: io_object_t, id: *mut u64) -> kern_return_t;
    pub fn IORegistryEntryFromPath(main_port: mach_port_t, path: *const c_char) -> io_object_t;
    pub fn IOServiceOpen(
        service: io_object_t,
        owning_task: mach_port_t,
        kind: u32,
        connect: *mut io_connect_t,
    ) -> kern_return_t;
    pub fn IOServiceClose(connect: io_connect_t) -> kern_return_t;
    pub fn IOConnectCallStructMethod(
        connection: io_connect_t,
        selector: u32,
        input: *const c_void,
        input_size: usize,
        output: *mut c_void,
        output_size: *mut usize,
    ) -> kern_return_t;
}

/// An owned IOKit object handle, released on drop.
pub struct IoObject(io_object_t);

impl IoObject {
    /// Wraps a handle the caller owns. Returns `None` for `IO_OBJECT_NULL`.
    pub fn new(raw: io_object_t) -> Option<Self> {
        if raw == 0 { None } else { Some(Self(raw)) }
    }

    pub fn raw(&self) -> io_object_t {
        self.0
    }

    pub fn class_name(&self) -> Option<String> {
        let mut buffer = [0 as c_char; 128];
        // SAFETY: `buffer` has room for the class name, which IOKit NUL-terminates on success.
        let result = unsafe { IOObjectGetClass(self.0, buffer.as_mut_ptr()) };
        (result == KERN_SUCCESS).then(|| {
            // SAFETY: the successful call above NUL-terminated `buffer`.
            unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
    }

    pub fn registry_id(&self) -> Option<u64> {
        let mut id = 0u64;
        // SAFETY: `id` is a valid, writable u64 for the kernel to fill.
        let result = unsafe { IORegistryEntryGetRegistryEntryID(self.0, &mut id) };
        (result == KERN_SUCCESS).then_some(id)
    }

    /// Reads one property. Cheaper than reading the whole property table.
    pub fn property(&self, key: &str) -> Option<Value> {
        let key = cf::cfstring(key);
        // SAFETY: `key` stays alive for the call; `Owned` releases the +1 reference returned by the create rule.
        unsafe {
            let object = IORegistryEntryCreateCFProperty(
                self.0,
                key.as_concrete_TypeRef(),
                kCFAllocatorDefault,
                0,
            );
            Owned::from_create(object).map(|owned| owned.to_value())
        }
    }

    /// Iterates the entry's children in the IOService plane.
    pub fn children(&self) -> Result<IoIterator> {
        let mut iterator = 0;
        // SAFETY: `iterator` is a valid out-parameter and `SERVICE_PLANE` is NUL-terminated.
        let result = unsafe {
            IORegistryEntryGetChildIterator(self.0, SERVICE_PLANE.as_ptr(), &mut iterator)
        };
        if result != KERN_SUCCESS {
            bail!("IORegistryEntryGetChildIterator failed: {result:#x}");
        }
        IoIterator::new(iterator)
    }
}

impl Drop for IoObject {
    fn drop(&mut self) {
        // SAFETY: `self.0` is an owned IOKit handle, released exactly once here.
        unsafe { IOObjectRelease(self.0) };
    }
}

/// An owned IOKit iterator.
pub struct IoIterator(IoObject);

impl IoIterator {
    fn new(raw: io_object_t) -> Result<Self> {
        match IoObject::new(raw) {
            Some(object) => Ok(Self(object)),
            None => bail!("IOKit returned a null iterator"),
        }
    }

    /// False when the registry changed during iteration and the walk must restart.
    pub fn is_valid(&self) -> bool {
        // SAFETY: the iterator handle is owned by `self` and valid while borrowed.
        unsafe { IOIteratorIsValid(self.0.raw()) != 0 }
    }
}

impl Iterator for IoIterator {
    type Item = IoObject;

    fn next(&mut self) -> Option<IoObject> {
        // SAFETY: the iterator handle is owned by `self`; the next object is wrapped for release.
        IoObject::new(unsafe { IOIteratorNext(self.0.raw()) })
    }
}

/// Returns every service matching an IOKit class name.
pub fn matching_services(class: &str) -> Result<IoIterator> {
    let class = CString::new(class)?;
    let mut iterator = 0;
    // SAFETY: `class` is NUL-terminated; `IOServiceGetMatchingServices` consumes the matching dictionary and returns an iterator owned by `IoIterator`.
    unsafe {
        let matching = IOServiceMatching(class.as_ptr());
        if matching.is_null() {
            bail!("IOServiceMatching returned null");
        }
        // IOServiceGetMatchingServices consumes `matching`.
        let result = IOServiceGetMatchingServices(
            MAIN_PORT_DEFAULT,
            matching as CFDictionaryRef,
            &mut iterator,
        );
        if result != KERN_SUCCESS {
            bail!("IOServiceGetMatchingServices failed: {result:#x}");
        }
    }
    IoIterator::new(iterator)
}

/// Returns the first service matching an IOKit class name.
pub fn matching_service(class: &str) -> Option<IoObject> {
    let class = CString::new(class).ok()?;
    // SAFETY: `class` is NUL-terminated; `IOServiceGetMatchingService` consumes the matching dictionary and its result is wrapped for release.
    unsafe {
        let matching = IOServiceMatching(class.as_ptr());
        if matching.is_null() {
            return None;
        }
        IoObject::new(IOServiceGetMatchingService(
            MAIN_PORT_DEFAULT,
            matching as CFDictionaryRef,
        ))
    }
}

/// Looks up a registry entry by path, such as `IODeviceTree:/arm-io/sgx`.
pub fn entry_from_path(path: &str) -> Option<IoObject> {
    let path = CString::new(path).ok()?;
    // SAFETY: `path` is NUL-terminated and the returned object is wrapped in `IoObject` for release.
    IoObject::new(unsafe { IORegistryEntryFromPath(MAIN_PORT_DEFAULT, path.as_ptr()) })
}

// ---------------------------------------------------------------------------
// GPU registry data
// ---------------------------------------------------------------------------

/// Static description of one GPU, read once per accelerator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuStatic {
    pub model: Option<String>,
    pub core_count: Option<u32>,
    pub generation: Option<u32>,
    pub variant: Option<String>,
    pub num_gps: Option<u32>,
    pub num_mgpus: Option<u32>,
    pub driver_version: Option<String>,
}

/// The accelerator's `PerformanceStatistics` dictionary.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerformanceStatistics {
    pub device_utilization_pct: Option<u8>,
    pub renderer_utilization_pct: Option<u8>,
    pub tiler_utilization_pct: Option<u8>,
    pub in_use_system_memory: Option<u64>,
    pub alloc_system_memory: Option<u64>,
    pub recovery_count: Option<u64>,
}

/// One `AppUsage` entry. There is one per Metal command queue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueueUsage {
    pub api: String,
    /// Cumulative GPU time in nanoseconds.
    pub accumulated_ns: u64,
    /// Last submission, in `mach_absolute_time` converted to nanoseconds. 0 if never.
    pub last_submitted_ns: u64,
}

/// One `AGXDeviceUserClient`: a GPU connection owned by a process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuClient {
    /// IORegistry entry ID. Unique for the life of the boot.
    pub registry_id: u64,
    pub pid: i32,
    /// Process name from `IOUserClientCreator`, truncated by the kernel.
    pub creator_name: String,
    pub queues: Vec<QueueUsage>,
    pub command_queue_count: Option<u32>,
    pub command_queue_count_max: Option<u32>,
}

impl GpuClient {
    pub fn accumulated_ns(&self) -> u64 {
        self.queues.iter().fold(0u64, |total, queue| {
            total.saturating_add(queue.accumulated_ns)
        })
    }

    pub fn last_submitted_ns(&self) -> u64 {
        self.queues
            .iter()
            .map(|queue| queue.last_submitted_ns)
            .max()
            .unwrap_or(0)
    }
}

/// One `IOAccelerator` service and its clients.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Accelerator {
    pub registry_id: u64,
    pub info: Arc<GpuStatic>,
    pub performance: PerformanceStatistics,
    pub clients: Vec<GpuClient>,
}

/// Everything read from the GPU registry in one sample.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Registry {
    pub accelerators: Vec<Accelerator>,
}

impl Registry {
    pub fn clients(&self) -> impl Iterator<Item = &GpuClient> {
        self.accelerators
            .iter()
            .flat_map(|accelerator| accelerator.clients.iter())
    }
}

/// Reads the GPU registry, caching static per-accelerator information.
#[derive(Default)]
pub struct RegistryReader {
    statics: HashMap<u64, Arc<GpuStatic>>,
}

const MAX_WALK_ATTEMPTS: usize = 3;

impl RegistryReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read(&mut self) -> Result<Registry> {
        let mut last_error = None;
        for _ in 0..MAX_WALK_ATTEMPTS {
            match self.walk() {
                Ok(Some(registry)) => return Ok(registry),
                Ok(None) => continue,
                Err(error) => last_error = Some(error),
            }
        }
        match last_error {
            Some(error) => Err(error),
            None => bail!("GPU registry kept changing during {MAX_WALK_ATTEMPTS} walks"),
        }
    }

    /// Reads only `PerformanceStatistics` from every accelerator. Cheap enough to
    /// poll several times per window so instantaneous utilization can be averaged.
    pub fn read_performance(&self) -> Result<Vec<PerformanceStatistics>> {
        Ok(matching_services("IOAccelerator")?
            .filter_map(|service| service.property("PerformanceStatistics"))
            .map(|value| parse_performance_statistics(&value))
            .collect())
    }

    /// One walk. Returns `Ok(None)` when the registry changed mid-walk.
    fn walk(&mut self) -> Result<Option<Registry>> {
        let mut registry = Registry::default();
        let services = matching_services("IOAccelerator")?;
        for service in services {
            // A failed ID lookup cannot identify a cache entry.
            let Some(registry_id) = service.registry_id() else {
                continue;
            };
            let info = self
                .statics
                .entry(registry_id)
                .or_insert_with(|| Arc::new(read_static(&service)))
                .clone();
            let performance = service
                .property("PerformanceStatistics")
                .map(|value| parse_performance_statistics(&value))
                .unwrap_or_default();

            let mut clients = Vec::new();
            let mut children = service.children()?;
            for child in children.by_ref() {
                if child.class_name().as_deref() != Some("AGXDeviceUserClient") {
                    continue;
                }
                if let Some(client) = read_client(&child) {
                    clients.push(client);
                }
            }
            if !children.is_valid() {
                return Ok(None);
            }

            registry.accelerators.push(Accelerator {
                registry_id,
                info,
                performance,
                clients,
            });
        }
        Ok(Some(registry))
    }
}

fn read_static(service: &IoObject) -> GpuStatic {
    let config = service.property("GPUConfigurationVariable");
    let config_u32 = |key: &str| {
        config
            .as_ref()
            .and_then(|config| config.get(key))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };
    GpuStatic {
        model: service
            .property("model")
            .and_then(|value| value_to_string(&value)),
        core_count: service
            .property("gpu-core-count")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
            .or_else(|| config_u32("num_cores")),
        generation: config_u32("gpu_gen"),
        variant: config
            .as_ref()
            .and_then(|config| config.get("gpu_var"))
            .and_then(value_to_string),
        num_gps: config_u32("num_gps"),
        num_mgpus: config_u32("num_mgpus"),
        driver_version: service
            .property("IOSourceVersion")
            .and_then(|value| value_to_string(&value)),
    }
}

fn read_client(entry: &IoObject) -> Option<GpuClient> {
    let creator = entry.property("IOUserClientCreator")?;
    let (pid, creator_name) = parse_creator(creator.as_str()?)?;
    let queues = entry
        .property("AppUsage")
        .map(|value| parse_app_usage(&value))
        .unwrap_or_default();
    let count = |key: &str| {
        entry
            .property(key)
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
    };
    Some(GpuClient {
        registry_id: entry.registry_id().unwrap_or_default(),
        pid,
        creator_name,
        queues,
        command_queue_count: count("CommandQueueCount"),
        command_queue_count_max: count("CommandQueueCountMax"),
    })
}

/// Strings in the registry are sometimes NUL-terminated `CFData`.
fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Data(bytes) => {
            let end = bytes
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(bytes.len());
            let text = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Parses `IOUserClientCreator`, for example `"pid 431, WindowServer"`.
pub fn parse_creator(text: &str) -> Option<(i32, String)> {
    let rest = text.strip_prefix("pid ")?;
    let (pid, name) = rest.split_once(',')?;
    Some((pid.trim().parse().ok()?, name.trim().to_string()))
}

pub fn parse_app_usage(value: &Value) -> Vec<QueueUsage> {
    let Some(entries) = value.as_array() else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let accumulated_ns = entry.get("accumulatedGPUTime").and_then(Value::as_u64)?;
            Some(QueueUsage {
                api: entry
                    .get("API")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                accumulated_ns,
                last_submitted_ns: entry
                    .get("lastSubmittedTime")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            })
        })
        .collect()
}

pub fn parse_performance_statistics(value: &Value) -> PerformanceStatistics {
    let u8_stat = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u8::try_from(value).ok())
    };
    let u64_stat = |key: &str| value.get(key).and_then(Value::as_u64);
    PerformanceStatistics {
        device_utilization_pct: u8_stat("Device Utilization %"),
        renderer_utilization_pct: u8_stat("Renderer Utilization %"),
        tiler_utilization_pct: u8_stat("Tiler Utilization %"),
        in_use_system_memory: u64_stat("In use system memory"),
        alloc_system_memory: u64_stat("Alloc system memory"),
        recovery_count: u64_stat("recoveryCount"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn dict(pairs: &[(&str, Value)]) -> Value {
        Value::Dict(
            pairs
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone()))
                .collect::<BTreeMap<_, _>>(),
        )
    }

    #[test]
    fn parses_creator() {
        assert_eq!(
            parse_creator("pid 431, WindowServer"),
            Some((431, "WindowServer".to_string()))
        );
        assert_eq!(
            parse_creator("pid 3159, Google Chrome He"),
            Some((3159, "Google Chrome He".to_string()))
        );
        assert_eq!(parse_creator("garbage"), None);
    }

    #[test]
    fn parses_app_usage_entries() {
        let value = Value::Array(vec![
            dict(&[
                ("API", Value::String("Metal".into())),
                ("accumulatedGPUTime", Value::Int(24_797_695_140_083)),
                ("lastSubmittedTime", Value::Int(590_749_484_213_791)),
            ]),
            dict(&[
                ("API", Value::String("Metal".into())),
                ("accumulatedGPUTime", Value::Int(34_803_750)),
            ]),
            dict(&[("API", Value::String("Metal".into()))]),
        ]);
        let queues = parse_app_usage(&value);
        assert_eq!(queues.len(), 2);
        assert_eq!(queues[0].accumulated_ns, 24_797_695_140_083);
        assert_eq!(queues[0].last_submitted_ns, 590_749_484_213_791);
        assert_eq!(queues[1].last_submitted_ns, 0);
    }

    #[test]
    fn parses_performance_statistics() {
        let value = dict(&[
            ("Device Utilization %", Value::Int(32)),
            ("Renderer Utilization %", Value::Int(31)),
            ("Tiler Utilization %", Value::Int(8)),
            ("In use system memory", Value::Int(3_221_749_760)),
            ("Alloc system memory", Value::Int(9_315_614_720)),
            ("recoveryCount", Value::Int(2)),
        ]);
        let stats = parse_performance_statistics(&value);
        assert_eq!(stats.device_utilization_pct, Some(32));
        assert_eq!(stats.renderer_utilization_pct, Some(31));
        assert_eq!(stats.tiler_utilization_pct, Some(8));
        assert_eq!(stats.in_use_system_memory, Some(3_221_749_760));
        assert_eq!(stats.alloc_system_memory, Some(9_315_614_720));
        assert_eq!(stats.recovery_count, Some(2));
    }

    #[test]
    #[ignore = "needs Apple Silicon hardware with an IOAccelerator service"]
    fn reads_live_registry() {
        let registry = RegistryReader::new().read().expect("read GPU registry");
        assert!(!registry.accelerators.is_empty(), "no IOAccelerator found");
        let accelerator = &registry.accelerators[0];
        assert!(accelerator.info.core_count.unwrap_or(0) > 0);
        assert!(registry.clients().any(|client| client.pid > 0));
    }
}
