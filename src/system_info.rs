//! Informações do host que podem ser coletadas sem depender do runtime.

#[cfg(target_os = "windows")]
pub fn detect_total_ram_bytes() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    unsafe extern "system" {
        #[link_name = "GlobalMemoryStatusEx"]
        fn global_memory_status_ex(info: *mut MemoryStatusEx) -> i32;
    }

    let mut info = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    // `info` tem layout C e o tamanho exigido pela API do Windows.
    let success = unsafe { global_memory_status_ex(&mut info) };
    (success != 0).then_some(info.ull_total_phys)
}

#[cfg(target_os = "linux")]
pub fn detect_total_ram_bytes() -> Option<u64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    raw.lines().find_map(|line| {
        line.strip_prefix("MemTotal:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
            .map(|kilobytes| kilobytes * 1024)
    })
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub fn detect_total_ram_bytes() -> Option<u64> {
    None
}
