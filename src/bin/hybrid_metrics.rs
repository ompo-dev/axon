use std::time::Duration;

#[derive(Clone, Copy)]
pub(super) struct CompilerTiming {
    pub(super) total: Duration,
    pub(super) validate_and_index: Duration,
    pub(super) classify_and_materialize: Duration,
}

#[derive(Clone, Copy)]
pub(super) struct Measurement {
    pub(super) duration: Duration,
    pub(super) total: u64,
}

#[derive(Clone, Copy)]
pub(super) struct HybridMeasurement {
    pub(super) duration: Duration,
    pub(super) compiler: CompilerTiming,
    pub(super) execution: Duration,
    pub(super) verification: Duration,
    pub(super) total: u64,
}

#[derive(Clone, Copy)]
pub(super) struct OracleMeasurement {
    pub(super) execution: Duration,
    pub(super) verification: Duration,
    pub(super) total: u64,
}

pub fn duration_samples_nanos(samples: &[Duration]) -> Vec<u64> {
    samples
        .iter()
        .map(|sample| sample.as_nanos().min(u64::MAX as u128) as u64)
        .collect()
}
