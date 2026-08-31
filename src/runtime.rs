/// Optimizations are optional. Exact execution remains the authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationFailure {
    Unavailable,
    InvalidCertificate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    CertifiedOptimization,
    ExactFallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedExecution<T> {
    mode: ExecutionMode,
    value: T,
}

impl<T> CheckedExecution<T> {
    pub const fn mode(&self) -> ExecutionMode {
        self.mode
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

/// Verification-mode executor. It returns optimized work only when it equals the exact fallback.
pub fn run_checked<T: Eq>(
    optimized: Result<T, OptimizationFailure>,
    exact: impl FnOnce() -> T,
) -> CheckedExecution<T> {
    let exact_value = exact();
    match optimized {
        Ok(value) if value == exact_value => CheckedExecution {
            mode: ExecutionMode::CertifiedOptimization,
            value,
        },
        Ok(_) | Err(_) => CheckedExecution {
            mode: ExecutionMode::ExactFallback,
            value: exact_value,
        },
    }
}
