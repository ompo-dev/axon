use crate::config::{MAX_CPU_THREADS, RAM_SOFT_CAP_DEFAULT, VRAM_SOFT_CAP_DEFAULT};
use crate::system_info::detect_total_ram_bytes;

#[derive(Clone, Debug)]
pub struct ResourceCaps {
    pub cpu_threads: usize,
    pub ram_soft_cap: u64,
    pub vram_soft_cap: u64,
    pub disk_soft_cap: u64,
}

pub fn detect_resource_caps() -> ResourceCaps {
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1).min(MAX_CPU_THREADS))
        .unwrap_or(1);
    let ram_total = detect_total_ram_bytes().unwrap_or(RAM_SOFT_CAP_DEFAULT);
    let ram_soft_cap = (ram_total as f64 * 0.35) as u64;
    let ram_soft_cap = ram_soft_cap
        .min(RAM_SOFT_CAP_DEFAULT)
        .max(512 * 1024 * 1024);
    let vram_soft_cap = VRAM_SOFT_CAP_DEFAULT;
    let disk_soft_cap = detect_free_disk_bytes().unwrap_or(20 * 1024 * 1024 * 1024);
    ResourceCaps {
        cpu_threads,
        ram_soft_cap,
        vram_soft_cap,
        disk_soft_cap,
    }
}

fn detect_free_disk_bytes() -> Option<u64> {
    None
}
