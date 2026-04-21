use std::fmt::{Display, Formatter};

use crate::config::{
    FORMAT_VERSION, MAGIC_FILE, MAGIC_PAGE, PAGE_HEADER_SIZE, PAGE_PAYLOAD_SIZE, PAGE_SIZE,
};
use crate::error::AxonError;

pub const SUPERBLOCK_A_PAGE_ID: u64 = 0;
pub const SUPERBLOCK_B_PAGE_ID: u64 = 1;
pub const FIRST_DATA_PAGE_ID: u64 = 2;
pub const REGION_ROOT_COUNT: usize = 8;

pub const SUPERBLOCK_SIZE: usize = PAGE_PAYLOAD_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum PageType {
    Free = 0,
    Meta = 1,
    AssemblyNode = 2,
    EdgeCsr = 3,
    EdgeDelta = 4,
    Episode = 5,
    Concept = 6,
    Journal = 7,
    ObsTile = 8,
    AllocMap = 9,
}

impl PageType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Free),
            1 => Some(Self::Meta),
            2 => Some(Self::AssemblyNode),
            3 => Some(Self::EdgeCsr),
            4 => Some(Self::EdgeDelta),
            5 => Some(Self::Episode),
            6 => Some(Self::Concept),
            7 => Some(Self::Journal),
            8 => Some(Self::ObsTile),
            9 => Some(Self::AllocMap),
            _ => None,
        }
    }
}

impl Display for PageType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Free => "FREE",
            Self::Meta => "META",
            Self::AssemblyNode => "ASSEMBLY_NODE",
            Self::EdgeCsr => "EDGE_CSR",
            Self::EdgeDelta => "EDGE_DELTA",
            Self::Episode => "EPISODE",
            Self::Concept => "CONCEPT",
            Self::Journal => "JOURNAL",
            Self::ObsTile => "OBS_TILE",
            Self::AllocMap => "ALLOC_MAP",
        };
        write!(f, "{name}")
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RegionRoot {
    pub root_page_id: u64,
    pub page_count: u64,
}

#[derive(Clone, Debug)]
pub struct BrainMeta {
    pub name: String,
    pub language: String,
    pub mode: u8,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
}

impl Default for BrainMeta {
    fn default() -> Self {
        let now = unix_ms_now();
        Self {
            name: "brain".to_string(),
            language: "pt-BR".to_string(),
            mode: 0,
            created_unix_ms: now,
            updated_unix_ms: now,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Superblock {
    pub magic: [u8; 4],
    pub version: u16,
    pub generation: u64,
    pub commit_lsn: u64,
    pub page_size: u32,
    pub page_count: u64,
    pub region_roots: [RegionRoot; REGION_ROOT_COUNT],
    pub caps: u64,
    pub flags: u64,
    pub brain_meta: BrainMeta,
    pub checksum: u32,
}

impl Superblock {
    pub fn new(page_count: u64, brain_name: String, mode: u8) -> Self {
        let mut sb = Self {
            magic: MAGIC_FILE,
            version: FORMAT_VERSION,
            generation: 1,
            commit_lsn: 0,
            page_size: PAGE_SIZE as u32,
            page_count,
            region_roots: [RegionRoot::default(); REGION_ROOT_COUNT],
            caps: 0,
            flags: 0,
            brain_meta: BrainMeta {
                name: brain_name,
                mode,
                ..BrainMeta::default()
            },
            checksum: 0,
        };
        sb.checksum = sb.calculate_checksum();
        sb
    }

    pub fn serialize(&self) -> [u8; SUPERBLOCK_SIZE] {
        let mut out = [0u8; SUPERBLOCK_SIZE];
        out[0..4].copy_from_slice(&self.magic);
        put_u16(&mut out, 4, self.version);
        put_u64(&mut out, 8, self.generation);
        put_u64(&mut out, 16, self.commit_lsn);
        put_u32(&mut out, 24, self.page_size);
        put_u64(&mut out, 28, self.page_count);
        let mut offset = 36;
        for root in self.region_roots {
            put_u64(&mut out, offset, root.root_page_id);
            put_u64(&mut out, offset + 8, root.page_count);
            offset += 16;
        }
        put_u64(&mut out, offset, self.caps);
        put_u64(&mut out, offset + 8, self.flags);
        offset += 16;
        put_fixed_str(&mut out, offset, 64, &self.brain_meta.name);
        put_fixed_str(&mut out, offset + 64, 16, &self.brain_meta.language);
        out[offset + 80] = self.brain_meta.mode;
        put_u64(&mut out, offset + 88, self.brain_meta.created_unix_ms);
        put_u64(&mut out, offset + 96, self.brain_meta.updated_unix_ms);
        put_u32(&mut out, SUPERBLOCK_SIZE - 4, self.checksum);
        out
    }

    pub fn deserialize(bytes: &[u8; SUPERBLOCK_SIZE]) -> Result<Self, AxonError> {
        if bytes[0..4] != MAGIC_FILE {
            return Err(AxonError::InvalidFormat(
                "superblock magic mismatch".to_string(),
            ));
        }
        let mut sb = Self {
            magic: MAGIC_FILE,
            version: get_u16(bytes, 4),
            generation: get_u64(bytes, 8),
            commit_lsn: get_u64(bytes, 16),
            page_size: get_u32(bytes, 24),
            page_count: get_u64(bytes, 28),
            region_roots: [RegionRoot::default(); REGION_ROOT_COUNT],
            caps: 0,
            flags: 0,
            brain_meta: BrainMeta::default(),
            checksum: get_u32(bytes, SUPERBLOCK_SIZE - 4),
        };
        let mut offset = 36;
        for root in &mut sb.region_roots {
            *root = RegionRoot {
                root_page_id: get_u64(bytes, offset),
                page_count: get_u64(bytes, offset + 8),
            };
            offset += 16;
        }
        sb.caps = get_u64(bytes, offset);
        sb.flags = get_u64(bytes, offset + 8);
        offset += 16;
        sb.brain_meta = BrainMeta {
            name: get_fixed_str(bytes, offset, 64),
            language: get_fixed_str(bytes, offset + 64, 16),
            mode: bytes[offset + 80],
            created_unix_ms: get_u64(bytes, offset + 88),
            updated_unix_ms: get_u64(bytes, offset + 96),
        };
        let expected = sb.calculate_checksum();
        if sb.checksum != expected {
            return Err(AxonError::InvalidFormat(format!(
                "superblock checksum mismatch: expected {expected:#x}, got {:#x}",
                sb.checksum
            )));
        }
        Ok(sb)
    }

    pub fn calculate_checksum(&self) -> u32 {
        let mut cloned = self.clone();
        cloned.checksum = 0;
        checksum32(&cloned.serialize())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PageHeader {
    pub magic: [u8; 4],
    pub page_id: u64,
    pub page_type: PageType,
    pub generation: u64,
    pub lsn_begin: u64,
    pub lsn_end: u64,
    pub payload_len: u16,
    pub reserved: [u8; 17],
    pub checksum: u32,
}

impl PageHeader {
    pub fn new(
        page_id: u64,
        page_type: PageType,
        generation: u64,
        lsn_begin: u64,
        lsn_end: u64,
        payload_len: u16,
    ) -> Self {
        Self {
            magic: MAGIC_PAGE,
            page_id,
            page_type,
            generation,
            lsn_begin,
            lsn_end,
            payload_len,
            reserved: [0u8; 17],
            checksum: 0,
        }
    }

    pub fn serialize(&self) -> [u8; PAGE_HEADER_SIZE] {
        let mut out = [0u8; PAGE_HEADER_SIZE];
        out[0..4].copy_from_slice(&self.magic);
        put_u64(&mut out, 4, self.page_id);
        out[12] = self.page_type as u8;
        put_u64(&mut out, 13, self.generation);
        put_u64(&mut out, 21, self.lsn_begin);
        put_u64(&mut out, 29, self.lsn_end);
        put_u16(&mut out, 37, self.payload_len);
        out[39..56].copy_from_slice(&self.reserved);
        put_u32(&mut out, 60, self.checksum);
        out
    }

    pub fn deserialize(bytes: &[u8; PAGE_HEADER_SIZE]) -> Result<Self, AxonError> {
        if bytes[0..4] != MAGIC_PAGE {
            return Err(AxonError::InvalidFormat("page magic mismatch".to_string()));
        }
        let page_type = PageType::from_u8(bytes[12]).ok_or_else(|| {
            AxonError::InvalidFormat(format!("unknown page type byte {}", bytes[12]))
        })?;
        let mut reserved = [0u8; 17];
        reserved.copy_from_slice(&bytes[39..56]);
        Ok(Self {
            magic: MAGIC_PAGE,
            page_id: get_u64(bytes, 4),
            page_type,
            generation: get_u64(bytes, 13),
            lsn_begin: get_u64(bytes, 21),
            lsn_end: get_u64(bytes, 29),
            payload_len: get_u16(bytes, 37),
            reserved,
            checksum: get_u32(bytes, 60),
        })
    }

    pub fn calculate_checksum(&self, payload: &[u8]) -> u32 {
        let mut h = *self;
        h.checksum = 0;
        let mut data = Vec::with_capacity(PAGE_HEADER_SIZE + payload.len());
        data.extend_from_slice(&h.serialize());
        data.extend_from_slice(payload);
        checksum32(&data)
    }
}

#[derive(Clone, Debug)]
pub struct Page {
    pub header: PageHeader,
    pub payload: [u8; PAGE_PAYLOAD_SIZE],
}

impl Page {
    pub fn new(
        page_id: u64,
        page_type: PageType,
        generation: u64,
        lsn_begin: u64,
        lsn_end: u64,
        payload: &[u8],
    ) -> Result<Self, AxonError> {
        if payload.len() > PAGE_PAYLOAD_SIZE {
            return Err(AxonError::InvalidFormat(format!(
                "payload too large: {}",
                payload.len()
            )));
        }
        let mut page = Self {
            header: PageHeader::new(
                page_id,
                page_type,
                generation,
                lsn_begin,
                lsn_end,
                payload.len() as u16,
            ),
            payload: [0u8; PAGE_PAYLOAD_SIZE],
        };
        page.payload[..payload.len()].copy_from_slice(payload);
        page.header.checksum = page
            .header
            .calculate_checksum(&page.payload[..payload.len()]);
        Ok(page)
    }

    pub fn serialize(&self) -> [u8; PAGE_SIZE] {
        let mut out = [0u8; PAGE_SIZE];
        out[..PAGE_HEADER_SIZE].copy_from_slice(&self.header.serialize());
        out[PAGE_HEADER_SIZE..].copy_from_slice(&self.payload);
        out
    }

    pub fn deserialize(bytes: &[u8; PAGE_SIZE]) -> Result<Self, AxonError> {
        let mut header_bytes = [0u8; PAGE_HEADER_SIZE];
        header_bytes.copy_from_slice(&bytes[..PAGE_HEADER_SIZE]);
        let header = PageHeader::deserialize(&header_bytes)?;
        if (header.payload_len as usize) > PAGE_PAYLOAD_SIZE {
            return Err(AxonError::InvalidFormat(format!(
                "invalid payload len {}",
                header.payload_len
            )));
        }
        let mut payload = [0u8; PAGE_PAYLOAD_SIZE];
        payload.copy_from_slice(&bytes[PAGE_HEADER_SIZE..]);
        let expected = header.calculate_checksum(&payload[..header.payload_len as usize]);
        if expected != header.checksum {
            return Err(AxonError::InvalidFormat(format!(
                "page checksum mismatch for page {}",
                header.page_id
            )));
        }
        Ok(Self { header, payload })
    }
}

pub fn checksum32(data: &[u8]) -> u32 {
    const OFFSET_BASIS: u32 = 0x811C9DC5;
    const PRIME: u32 = 0x01000193;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub fn unix_ms_now() -> u64 {
    let now = std::time::SystemTime::now();
    let epoch = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));
    epoch.as_millis() as u64
}

fn put_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(inp: &[u8], offset: usize) -> u16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&inp[offset..offset + 2]);
    u16::from_le_bytes(bytes)
}

fn get_u32(inp: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&inp[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn get_u64(inp: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&inp[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

fn put_fixed_str(out: &mut [u8], offset: usize, len: usize, value: &str) {
    let bytes = value.as_bytes();
    let n = bytes.len().min(len);
    out[offset..offset + n].copy_from_slice(&bytes[..n]);
}

fn get_fixed_str(inp: &[u8], offset: usize, len: usize) -> String {
    let chunk = &inp[offset..offset + len];
    let end = chunk.iter().position(|b| *b == 0).unwrap_or(chunk.len());
    String::from_utf8_lossy(&chunk[..end]).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_roundtrip() {
        let payload = b"hello";
        let page = Page::new(3, PageType::Journal, 9, 10, 11, payload).unwrap();
        let encoded = page.serialize();
        let decoded = Page::deserialize(&encoded).unwrap();
        assert_eq!(decoded.header.page_id, 3);
        assert_eq!(decoded.header.page_type, PageType::Journal);
        assert_eq!(&decoded.payload[..5], payload);
    }

    #[test]
    fn superblock_roundtrip() {
        let sb = Superblock::new(128, "brain".to_string(), 1);
        let bytes = sb.serialize();
        let decoded = Superblock::deserialize(&bytes).unwrap();
        assert_eq!(decoded.magic, MAGIC_FILE);
        assert_eq!(decoded.page_size as usize, PAGE_SIZE);
        assert_eq!(decoded.brain_meta.name, "brain");
    }
}
