use crate::config::{MAX_CPU_THREADS, RAM_SOFT_CAP_DEFAULT, VRAM_SOFT_CAP_DEFAULT};

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
    let ram_soft_cap = ram_soft_cap.min(RAM_SOFT_CAP_DEFAULT).max(512 * 1024 * 1024);
    let vram_soft_cap = VRAM_SOFT_CAP_DEFAULT;
    let disk_soft_cap = detect_free_disk_bytes().unwrap_or(20 * 1024 * 1024 * 1024);
    ResourceCaps {
        cpu_threads,
        ram_soft_cap,
        vram_soft_cap,
        disk_soft_cap,
    }
}

#[cfg(target_os = "windows")]
fn detect_total_ram_bytes() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
    }
    unsafe {
        let mut info = MemoryStatusEx {
            dwLength: std::mem::size_of::<MemoryStatusEx>() as u32,
            dwMemoryLoad: 0,
            ullTotalPhys: 0,
            ullAvailPhys: 0,
            ullTotalPageFile: 0,
            ullAvailPageFile: 0,
            ullTotalVirtual: 0,
            ullAvailVirtual: 0,
            ullAvailExtendedVirtual: 0,
        };
        let ok = GlobalMemoryStatusEx(&mut info as *mut MemoryStatusEx);
        if ok == 0 {
            None
        } else {
            Some(info.ullTotalPhys)
        }
    }
}

#[cfg(target_os = "linux")]
fn detect_total_ram_bytes() -> Option<u64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok())?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn detect_total_ram_bytes() -> Option<u64> {
    None
}

fn detect_free_disk_bytes() -> Option<u64> {
    None
}
