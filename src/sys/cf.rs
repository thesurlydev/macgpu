//! CoreFoundation helpers: convert CF property trees into plain Rust values.
//!
//! Every reader converts CF objects into [`Value`] at the boundary so that the
//! rest of the crate works with owned, `Send` data that unit tests can build by
//! hand.

use core_foundation::base::TCFType;
use core_foundation::string::CFString;
use core_foundation_sys::array::{
    CFArrayGetCount, CFArrayGetTypeID, CFArrayGetValueAtIndex, CFArrayRef,
};
use core_foundation_sys::base::{CFGetTypeID, CFRelease, CFTypeRef};
use core_foundation_sys::data::{CFDataGetBytePtr, CFDataGetLength, CFDataGetTypeID, CFDataRef};
use core_foundation_sys::dictionary::{
    CFDictionaryGetCount, CFDictionaryGetKeysAndValues, CFDictionaryGetTypeID, CFDictionaryRef,
};
use core_foundation_sys::number::{
    CFBooleanGetTypeID, CFBooleanGetValue, CFBooleanRef, CFNumberGetTypeID, CFNumberGetValue,
    CFNumberIsFloatType, CFNumberRef, kCFNumberFloat64Type, kCFNumberSInt64Type,
};
use core_foundation_sys::string::{CFStringGetTypeID, CFStringRef};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// An owned copy of a CoreFoundation property-list value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Data(Vec<u8>),
    Array(Vec<Value>),
    Dict(BTreeMap<String, Value>),
    /// A CF type this module does not convert.
    Other,
}

impl Value {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            Value::Float(value) if value.is_finite() => Some(*value as i64),
            Value::Bool(value) => Some(i64::from(*value)),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_i64().and_then(|value| u64::try_from(value).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(value) => Some(*value as f64),
            Value::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_data(&self) -> Option<&[u8]> {
        match self {
            Value::Data(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Dict(value) => Some(value),
            _ => None,
        }
    }

    /// Looks up `key` when this value is a dictionary.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_dict().and_then(|dict| dict.get(key))
    }
}

/// Converts a borrowed CF object into a [`Value`]. Does not consume `object`.
///
/// # Safety
/// `object` must be null or a valid CF object for the duration of the call.
pub unsafe fn to_value(object: CFTypeRef) -> Value {
    if object.is_null() {
        return Value::Other;
    }
    unsafe {
        let type_id = CFGetTypeID(object);
        if type_id == CFNumberGetTypeID() {
            let number = object as CFNumberRef;
            if CFNumberIsFloatType(number) != 0 {
                let mut value = 0f64;
                CFNumberGetValue(
                    number,
                    kCFNumberFloat64Type,
                    (&raw mut value).cast::<c_void>(),
                );
                Value::Float(value)
            } else {
                let mut value = 0i64;
                CFNumberGetValue(
                    number,
                    kCFNumberSInt64Type,
                    (&raw mut value).cast::<c_void>(),
                );
                Value::Int(value)
            }
        } else if type_id == CFStringGetTypeID() {
            Value::String(cfstring_to_string(object as CFStringRef))
        } else if type_id == CFBooleanGetTypeID() {
            Value::Bool(CFBooleanGetValue(object as CFBooleanRef))
        } else if type_id == CFDataGetTypeID() {
            let data = object as CFDataRef;
            let length = usize::try_from(CFDataGetLength(data)).unwrap_or(0);
            let pointer = CFDataGetBytePtr(data);
            if pointer.is_null() || length == 0 {
                Value::Data(Vec::new())
            } else {
                Value::Data(std::slice::from_raw_parts(pointer, length).to_vec())
            }
        } else if type_id == CFArrayGetTypeID() {
            let array = object as CFArrayRef;
            let count = CFArrayGetCount(array);
            let mut values = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
            for index in 0..count {
                values.push(to_value(CFArrayGetValueAtIndex(array, index)));
            }
            Value::Array(values)
        } else if type_id == CFDictionaryGetTypeID() {
            Value::Dict(dictionary_to_map(object as CFDictionaryRef))
        } else {
            Value::Other
        }
    }
}

/// Converts a CF dictionary with string keys into a map. Non-string keys are skipped.
///
/// # Safety
/// `dictionary` must be a valid `CFDictionaryRef`.
pub unsafe fn dictionary_to_map(dictionary: CFDictionaryRef) -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    unsafe {
        let count = usize::try_from(CFDictionaryGetCount(dictionary)).unwrap_or(0);
        if count == 0 {
            return map;
        }
        let mut keys: Vec<*const c_void> = vec![std::ptr::null(); count];
        let mut values: Vec<*const c_void> = vec![std::ptr::null(); count];
        CFDictionaryGetKeysAndValues(dictionary, keys.as_mut_ptr(), values.as_mut_ptr());
        for (key, value) in keys.into_iter().zip(values) {
            if key.is_null() || CFGetTypeID(key) != CFStringGetTypeID() {
                continue;
            }
            map.insert(cfstring_to_string(key as CFStringRef), to_value(value));
        }
    }
    map
}

/// Copies a borrowed `CFStringRef` into a Rust string.
///
/// # Safety
/// `string` must be null or a valid `CFStringRef`.
pub unsafe fn cfstring_to_string(string: CFStringRef) -> String {
    if string.is_null() {
        return String::new();
    }
    unsafe { CFString::wrap_under_get_rule(string).to_string() }
}

/// An owned CF object released on drop (create rule).
pub struct Owned(CFTypeRef);

impl Owned {
    /// Takes ownership of a +1 CF reference. Returns `None` for null.
    ///
    /// # Safety
    /// `object` must be null or a CF object the caller owns.
    pub unsafe fn from_create(object: CFTypeRef) -> Option<Self> {
        if object.is_null() {
            None
        } else {
            Some(Self(object))
        }
    }

    pub fn as_ptr(&self) -> CFTypeRef {
        self.0
    }

    pub fn to_value(&self) -> Value {
        unsafe { to_value(self.0) }
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { CFRelease(self.0) };
    }
}

/// Creates a CFString that lives as long as the returned value.
pub fn cfstring(value: &str) -> CFString {
    CFString::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_foundation::array::CFArray;
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::number::CFNumber;

    #[test]
    fn converts_nested_property_lists() {
        let entry = CFDictionary::from_CFType_pairs(&[
            (
                CFString::new("API").as_CFType(),
                CFString::new("Metal").as_CFType(),
            ),
            (
                CFString::new("accumulatedGPUTime").as_CFType(),
                CFNumber::from(1_234_567_890_123i64).as_CFType(),
            ),
            (
                CFString::new("ratio").as_CFType(),
                CFNumber::from(0.5f64).as_CFType(),
            ),
            (
                CFString::new("flag").as_CFType(),
                CFBoolean::true_value().as_CFType(),
            ),
        ]);
        let array = CFArray::from_CFTypes(&[entry.as_CFType()]);
        let value = unsafe { to_value(array.as_CFTypeRef()) };

        let entries = value.as_array().expect("array");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].get("API").and_then(Value::as_str), Some("Metal"));
        assert_eq!(
            entries[0].get("accumulatedGPUTime").and_then(Value::as_u64),
            Some(1_234_567_890_123)
        );
        assert_eq!(entries[0].get("ratio").and_then(Value::as_f64), Some(0.5));
        assert_eq!(entries[0].get("flag"), Some(&Value::Bool(true)));
    }
}
