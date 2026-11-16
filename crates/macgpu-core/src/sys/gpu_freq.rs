//! GPU DVFS frequency table from the device tree.
//!
//! The GPU node (`sgx`, compatible `gpu,<soc>`) carries `perf-states`: little
//! endian u32 pairs of (frequency in Hz, voltage in mV). Entry 0 is the `OFF`
//! state (0 Hz) and entry n is IOReport `GPUPH` state `Pn`. Chips with several
//! GPU partitions repeat the table once per partition (`perf-state-table-count`
//! tables of `perf-state-count` entries); the copies share frequencies and
//! differ only in voltage, so the first copy is used. The `pmgr` node's
//! `voltage-states9` holds the same table on M1 to M5 and is the fallback.
//!
//! The table is sorted by voltage, not frequency, and a voltage step can carry
//! two clocks: a lower and a higher entry at the same voltage. Both are kept in
//! IOReport state order so `Pn` maps to the right table entry.

use super::iokit::{self, IoObject};

/// Frequencies in MHz for the active P-states P1..Pn, in IOReport state order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrequencyTable {
    /// `mhz[n - 1]` is the clock of state `Pn`. Not sorted (see the module docs).
    pub mhz: Vec<f64>,
    /// Where the table came from, for diagnostics.
    pub source: String,
}

impl FrequencyTable {
    /// Highest clock in the table.
    pub fn max_mhz(&self) -> Option<f64> {
        self.mhz.iter().copied().reduce(f64::max)
    }

    /// Clock of the IOReport state named `state` (`P1`..`Pn`).
    pub fn mhz_for_state(&self, state: &str) -> Option<f64> {
        let index: usize = state.strip_prefix('P')?.parse().ok()?;
        self.mhz.get(index.checked_sub(1)?).copied()
    }
}

/// Lowest and highest plausible GPU clocks, to reject a misread table.
const MIN_PLAUSIBLE_MHZ: f64 = 50.0;
const MAX_PLAUSIBLE_MHZ: f64 = 5_000.0;

/// Reads the table, or `None` when no valid table exists (for example in a VM).
pub fn read() -> Option<FrequencyTable> {
    read_gpu_node().or_else(read_pmgr)
}

/// `perf-states` on the GPU's device-tree node, `sgx` under `arm-io`.
fn read_gpu_node() -> Option<FrequencyTable> {
    let gpu = iokit::entry_from_path("IODeviceTree:/arm-io/sgx")?;
    let data = gpu.property("perf-states")?;
    let mhz = parse_table(
        data.as_data()?,
        u32_property(&gpu, "perf-state-count"),
        u32_property(&gpu, "perf-state-table-count"),
    )?;
    Some(FrequencyTable {
        mhz,
        source: "IODeviceTree sgx perf-states".into(),
    })
}

/// `voltage-states9` on the power manager node, parsed as little-endian pairs.
fn read_pmgr() -> Option<FrequencyTable> {
    let pmgr = iokit::entry_from_path("IODeviceTree:/arm-io/pmgr")?;
    let data = pmgr.property("voltage-states9")?;
    let mhz = parse_table(data.as_data()?, None, None)?;
    Some(FrequencyTable {
        mhz,
        source: "IODeviceTree pmgr voltage-states9".into(),
    })
}

fn u32_le(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

fn u32_property(entry: &IoObject, key: &str) -> Option<usize> {
    let value = entry.property(key)?;
    usize::try_from(u32_le(value.as_data()?)?).ok()
}

/// Parses (Hz, mV) pairs into MHz for P1..Pn, dropping the `OFF` entry.
///
/// `count` is the number of entries per table and `tables` the number of
/// copies; when they are missing the whole property is one table. Returns
/// `None` unless entry 0 is `OFF` (0 Hz), every other entry is a plausible
/// clock, and every copy has the same frequencies: a table that fails these
/// checks would put wrong MHz on screen, which is worse than showing none.
pub fn parse_table(data: &[u8], count: Option<usize>, tables: Option<usize>) -> Option<Vec<f64>> {
    let pairs: Vec<(u32, u32)> = data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| Some((u32_le(&chunk[..4])?, u32_le(&chunk[4..])?)))
        .collect::<Option<Vec<_>>>()?;
    let tables = tables.unwrap_or(1).max(1);
    let count = count.unwrap_or(pairs.len() / tables);
    if count < 2 || pairs.len() < count * tables {
        return None;
    }
    let frequencies: Vec<u32> = pairs[..count].iter().map(|(hz, _)| *hz).collect();
    for copy in pairs.chunks_exact(count).take(tables).skip(1) {
        if copy
            .iter()
            .map(|(hz, _)| *hz)
            .ne(frequencies.iter().copied())
        {
            return None;
        }
    }
    if frequencies[0] != 0 {
        return None;
    }
    let mhz: Vec<f64> = frequencies[1..]
        .iter()
        .map(|hz| f64::from(*hz) / 1e6)
        .collect();
    mhz.iter()
        .all(|mhz| (MIN_PLAUSIBLE_MHZ..=MAX_PLAUSIBLE_MHZ).contains(mhz))
        .then_some(mhz)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two copies of a `perf-states` table with out-of-order clocks, in MHz.
    const SAMPLE_MHZ: [u32; 16] = [
        0, 338, 618, 796, 924, 952, 1056, 1062, 1182, 1182, 1312, 1242, 1380, 1326, 1470, 1578,
    ];

    fn encode(tables: &[(&[u32], u32)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (mhz, millivolts) in tables {
            for (index, mhz) in mhz.iter().enumerate() {
                bytes.extend_from_slice(&(mhz * 1_000_000).to_le_bytes());
                bytes.extend_from_slice(&(millivolts + index as u32 * 10).to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn parses_repeated_tables_without_sorting() {
        let data = encode(&[(&SAMPLE_MHZ, 600), (&SAMPLE_MHZ, 610)]);
        let mhz = parse_table(&data, Some(16), Some(2)).expect("table");
        assert_eq!(mhz.len(), 15);
        assert_eq!(mhz[0], 338.0);
        assert_eq!(mhz[9], 1312.0);
        assert_eq!(mhz[10], 1242.0);
        let table = FrequencyTable {
            mhz,
            source: String::new(),
        };
        assert_eq!(table.max_mhz(), Some(1578.0));
        assert_eq!(table.mhz_for_state("P11"), Some(1242.0));
        assert_eq!(table.mhz_for_state("P16"), None);
        assert_eq!(table.mhz_for_state("OFF"), None);
        assert_eq!(table.mhz_for_state("P0"), None);
    }

    #[test]
    fn whole_property_is_one_table_without_counts() {
        let data = encode(&[(&SAMPLE_MHZ, 600)]);
        assert_eq!(
            parse_table(&data, None, None).map(|mhz| mhz.len()),
            Some(15)
        );
    }

    #[test]
    fn rejects_inconsistent_tables() {
        let mut other = SAMPLE_MHZ;
        other[3] = 800;
        let data = encode(&[(&SAMPLE_MHZ, 600), (&other, 600)]);
        assert_eq!(parse_table(&data, Some(16), Some(2)), None);

        let no_off = [338, 618, 796];
        assert_eq!(parse_table(&encode(&[(&no_off, 600)]), None, None), None);

        let zero_state = [0, 338, 0];
        assert_eq!(
            parse_table(&encode(&[(&zero_state, 600)]), None, None),
            None
        );

        assert_eq!(parse_table(&[1, 2, 3], None, None), None);
        let data = encode(&[(&SAMPLE_MHZ, 600)]);
        assert_eq!(parse_table(&data, Some(16), Some(2)), None);
    }

    #[test]
    #[ignore = "needs Apple Silicon hardware with a device-tree GPU node"]
    fn reads_live_table() {
        let table = read().expect("GPU frequency table");
        assert!(table.mhz.len() >= 2);
        eprintln!("{}: {:?}", table.source, table.mhz);
        let max = table.max_mhz().expect("max");
        assert!((300.0..=3000.0).contains(&max), "max {max} MHz");
        if let (Some(gpu), Some(pmgr)) = (read_gpu_node(), read_pmgr()) {
            assert_eq!(gpu.mhz, pmgr.mhz, "sgx and pmgr tables disagree");
        }
    }
}
