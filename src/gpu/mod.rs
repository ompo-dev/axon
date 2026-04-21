use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct GpuBackend {
    pub available: bool,
    pub driver_name: String,
    pub note: String,
}

impl GpuBackend {
    pub fn probe() -> Arc<Self> {
        #[cfg(target_os = "windows")]
        {
            return Arc::new(probe_cuda_windows());
        }
        #[cfg(target_os = "linux")]
        {
            return Arc::new(probe_cuda_linux());
        }
        #[allow(unreachable_code)]
        Arc::new(Self {
            available: false,
            driver_name: "none".to_string(),
            note: "unsupported platform".to_string(),
        })
    }
}

impl Display for GpuBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.available {
            write!(f, "GPU({})", self.driver_name)
        } else {
            write!(f, "CPU fallback ({})", self.note)
        }
    }
}

#[cfg(target_os = "windows")]
fn probe_cuda_windows() -> GpuBackend {
    unsafe {
        use std::ffi::CString;
        type HMODULE = *mut core::ffi::c_void;
        unsafe extern "system" {
            fn LoadLibraryA(name: *const i8) -> HMODULE;
            fn GetProcAddress(module: HMODULE, name: *const i8) -> *mut core::ffi::c_void;
            fn FreeLibrary(module: HMODULE) -> i32;
        }
        type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
        let dll_name = CString::new("nvcuda.dll").expect("nvcuda.dll");
        let module = LoadLibraryA(dll_name.as_ptr());
        if module.is_null() {
            return GpuBackend {
                available: false,
                driver_name: "none".to_string(),
                note: "nvcuda.dll not found".to_string(),
            };
        }
        let sym_name = CString::new("cuInit").expect("cuInit");
        let proc = GetProcAddress(module, sym_name.as_ptr());
        if proc.is_null() {
            let _ = FreeLibrary(module);
            return GpuBackend {
                available: false,
                driver_name: "none".to_string(),
                note: "cuInit symbol not found".to_string(),
            };
        }
        let func: CuInit = std::mem::transmute(proc);
        let rc = func(0);
        let _ = FreeLibrary(module);
        if rc == 0 {
            GpuBackend {
                available: true,
                driver_name: "CUDA Driver API".to_string(),
                note: "cuInit ok".to_string(),
            }
        } else {
            GpuBackend {
                available: false,
                driver_name: "none".to_string(),
                note: format!("cuInit returned {}", rc),
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn probe_cuda_linux() -> GpuBackend {
    unsafe {
        use std::ffi::CString;
        const RTLD_NOW: i32 = 2;
        unsafe extern "C" {
            fn dlopen(filename: *const i8, flag: i32) -> *mut core::ffi::c_void;
            fn dlsym(handle: *mut core::ffi::c_void, symbol: *const i8) -> *mut core::ffi::c_void;
            fn dlclose(handle: *mut core::ffi::c_void) -> i32;
        }
        type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
        let so_name = CString::new("libcuda.so.1").expect("libcuda.so.1");
        let handle = dlopen(so_name.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            return GpuBackend {
                available: false,
                driver_name: "none".to_string(),
                note: "libcuda.so.1 not found".to_string(),
            };
        }
        let sym_name = CString::new("cuInit").expect("cuInit");
        let proc = dlsym(handle, sym_name.as_ptr());
        if proc.is_null() {
            let _ = dlclose(handle);
            return GpuBackend {
                available: false,
                driver_name: "none".to_string(),
                note: "cuInit symbol not found".to_string(),
            };
        }
        let func: CuInit = std::mem::transmute(proc);
        let rc = func(0);
        let _ = dlclose(handle);
        if rc == 0 {
            GpuBackend {
                available: true,
                driver_name: "CUDA Driver API".to_string(),
                note: "cuInit ok".to_string(),
            }
        } else {
            GpuBackend {
                available: false,
                driver_name: "none".to_string(),
                note: format!("cuInit returned {}", rc),
            }
        }
    }
}
