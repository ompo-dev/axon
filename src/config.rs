pub const PAGE_SIZE: usize = 4096;
pub const PAGE_HEADER_SIZE: usize = 64;
pub const PAGE_PAYLOAD_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

pub const MAGIC_FILE: [u8; 4] = *b"AXON";
pub const MAGIC_PAGE: [u8; 4] = *b"PAGE";
pub const MAGIC_SNAPSHOT: [u8; 4] = *b"SNAP";

pub const FORMAT_VERSION: u16 = 1;

pub const TICK_HZ: u64 = 100;
pub const TICK_MILLIS: u64 = 10;
pub const JOURNAL_FLUSH_MILLIS: u64 = 250;
pub const JOURNAL_FLUSH_BYTES: usize = 64 * 1024;
pub const CHECKPOINT_MILLIS: u64 = 5_000;
pub const CHECKPOINT_JOURNAL_BYTES: usize = 128 * 1024 * 1024;

pub const DEFAULT_RANDOM_SEED: u64 = 0xA10A_2026;
pub const MAX_CPU_THREADS: usize = 4;
pub const RAM_SOFT_CAP_DEFAULT: u64 = 3 * 1024 * 1024 * 1024;
pub const VRAM_SOFT_CAP_DEFAULT: u64 = 6 * 1024 * 1024 * 1024;
