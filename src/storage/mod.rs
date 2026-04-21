use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::axon_format::{
    checksum32, unix_ms_now, Page, PageType, Superblock, FIRST_DATA_PAGE_ID,
    SUPERBLOCK_A_PAGE_ID, SUPERBLOCK_B_PAGE_ID,
};
use crate::config::{MAGIC_SNAPSHOT, PAGE_PAYLOAD_SIZE, PAGE_SIZE};
use crate::error::AxonError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MutationKind {
    InputChar = 1,
    OutputChar = 2,
    EdgeUpdate = 3,
    Spawn = 4,
    Prune = 5,
    Merge = 6,
    SemanticLink = 7,
    TempUpdate = 8,
    LinkCreate = 9,
    LinkStrengthen = 10,
    LinkWeaken = 11,
    TemporalRebind = 12,
    CorrectionApplied = 13,
}

impl MutationKind {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::InputChar),
            2 => Some(Self::OutputChar),
            3 => Some(Self::EdgeUpdate),
            4 => Some(Self::Spawn),
            5 => Some(Self::Prune),
            6 => Some(Self::Merge),
            7 => Some(Self::SemanticLink),
            8 => Some(Self::TempUpdate),
            9 => Some(Self::LinkCreate),
            10 => Some(Self::LinkStrengthen),
            11 => Some(Self::LinkWeaken),
            12 => Some(Self::TemporalRebind),
            13 => Some(Self::CorrectionApplied),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MutationRecord {
    pub kind: MutationKind,
    pub flags: u8,
    pub tick: u64,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub value: f32,
    pub extra: f32,
}

impl MutationRecord {
    pub const SIZE: usize = 32;

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0] = self.kind as u8;
        out[1] = self.flags;
        out[2..4].copy_from_slice(&0u16.to_le_bytes());
        out[4..12].copy_from_slice(&self.tick.to_le_bytes());
        out[12..16].copy_from_slice(&self.a.to_le_bytes());
        out[16..20].copy_from_slice(&self.b.to_le_bytes());
        out[20..24].copy_from_slice(&self.c.to_le_bytes());
        out[24..28].copy_from_slice(&self.value.to_le_bytes());
        out[28..32].copy_from_slice(&self.extra.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AxonError> {
        if bytes.len() != Self::SIZE {
            return Err(AxonError::InvalidFormat(
                "invalid mutation record size".to_string(),
            ));
        }
        let kind = MutationKind::from_u8(bytes[0]).ok_or_else(|| {
            AxonError::InvalidFormat(format!("unknown mutation kind {}", bytes[0]))
        })?;
        let tick = u64::from_le_bytes(bytes[4..12].try_into().unwrap_or([0u8; 8]));
        let a = u32::from_le_bytes(bytes[12..16].try_into().unwrap_or([0u8; 4]));
        let b = u32::from_le_bytes(bytes[16..20].try_into().unwrap_or([0u8; 4]));
        let c = u32::from_le_bytes(bytes[20..24].try_into().unwrap_or([0u8; 4]));
        let value = f32::from_le_bytes(bytes[24..28].try_into().unwrap_or([0u8; 4]));
        let extra = f32::from_le_bytes(bytes[28..32].try_into().unwrap_or([0u8; 4]));
        Ok(Self {
            kind,
            flags: bytes[1],
            tick,
            a,
            b,
            c,
            value,
            extra,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PageStatus {
    pub page_id: u64,
    pub page_type: PageType,
    pub payload_len: u16,
    pub generation: u64,
    pub lsn_begin: u64,
    pub lsn_end: u64,
    pub checksum_ok: bool,
}

pub struct BrainFile {
    path: PathBuf,
    file: File,
    pub superblock: Superblock,
    active_superblock_slot: u64,
}

impl BrainFile {
    pub fn open_or_create(
        path: &Path,
        create_if_missing: bool,
        brain_name: &str,
        mode: u8,
    ) -> Result<Self, AxonError> {
        let exists = path.exists();
        if !exists && !create_if_missing {
            return Err(AxonError::State(format!(
                "brain '{}' does not exist (use --create-if-missing)",
                path.display()
            )));
        }
        if !exists {
            Self::create_new(path, brain_name, mode)
        } else {
            Self::open_existing(path)
        }
    }

    pub fn open_readonly(path: &Path) -> Result<Self, AxonError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let (superblock, slot) = load_latest_superblock(&file)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            superblock,
            active_superblock_slot: slot,
        })
    }

    fn create_new(path: &Path, brain_name: &str, mode: u8) -> Result<Self, AxonError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;
        let mut superblock = Superblock::new(3, brain_name.to_string(), mode);
        superblock.brain_meta.updated_unix_ms = unix_ms_now();
        let alloc_payload = [0u8; 32];
        let alloc_page = Page::new(FIRST_DATA_PAGE_ID, PageType::AllocMap, 1, 0, 0, &alloc_payload)?;
        write_raw_page(&mut file, FIRST_DATA_PAGE_ID, &alloc_page.serialize())?;
        write_superblock_slot(&mut file, SUPERBLOCK_A_PAGE_ID, &superblock)?;
        write_superblock_slot(&mut file, SUPERBLOCK_B_PAGE_ID, &superblock)?;
        file.sync_all()?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            superblock,
            active_superblock_slot: SUPERBLOCK_B_PAGE_ID,
        })
    }

    fn open_existing(path: &Path) -> Result<Self, AxonError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let (superblock, slot) = load_latest_superblock(&file)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            superblock,
            active_superblock_slot: slot,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn page_count(&self) -> u64 {
        self.superblock.page_count
    }

    pub fn append_page(
        &mut self,
        page_type: PageType,
        generation: u64,
        lsn_begin: u64,
        lsn_end: u64,
        payload: &[u8],
    ) -> Result<u64, AxonError> {
        let page_id = self.superblock.page_count;
        let page = Page::new(page_id, page_type, generation, lsn_begin, lsn_end, payload)?;
        write_raw_page(&mut self.file, page_id, &page.serialize())?;
        self.superblock.page_count = self.superblock.page_count.saturating_add(1);
        self.superblock.commit_lsn = self.superblock.commit_lsn.max(lsn_end);
        self.superblock.brain_meta.updated_unix_ms = unix_ms_now();
        update_region_counter(&mut self.superblock, page_type, page_id);
        Ok(page_id)
    }

    pub fn write_journal_records(
        &mut self,
        generation: u64,
        lsn_begin: u64,
        lsn_end: u64,
        records: &[MutationRecord],
    ) -> Result<usize, AxonError> {
        if records.is_empty() {
            return Ok(0);
        }
        let mut bytes = Vec::with_capacity(records.len() * MutationRecord::SIZE);
        for record in records {
            bytes.extend_from_slice(&record.to_bytes());
        }
        let mut offset = 0usize;
        let mut page_written = 0usize;
        while offset < bytes.len() {
            let end = (offset + PAGE_PAYLOAD_SIZE).min(bytes.len());
            self.append_page(
                PageType::Journal,
                generation,
                lsn_begin,
                lsn_end,
                &bytes[offset..end],
            )?;
            offset = end;
            page_written += 1;
        }
        Ok(page_written)
    }

    pub fn write_snapshot(
        &mut self,
        generation: u64,
        checkpoint_lsn: u64,
        snapshot_blob: &[u8],
    ) -> Result<(), AxonError> {
        let header_size = 32usize;
        let chunk_capacity = PAGE_PAYLOAD_SIZE.saturating_sub(header_size);
        if chunk_capacity == 0 {
            return Err(AxonError::State("invalid snapshot chunk capacity".to_string()));
        }
        let chunk_count = snapshot_blob.len().div_ceil(chunk_capacity).max(1);
        for chunk_idx in 0..chunk_count {
            let start = chunk_idx * chunk_capacity;
            let end = (start + chunk_capacity).min(snapshot_blob.len());
            let chunk = &snapshot_blob[start..end];
            let mut payload = Vec::with_capacity(header_size + chunk.len());
            payload.extend_from_slice(&MAGIC_SNAPSHOT);
            payload.extend_from_slice(&checkpoint_lsn.to_le_bytes());
            payload.extend_from_slice(&(chunk_idx as u32).to_le_bytes());
            payload.extend_from_slice(&(chunk_count as u32).to_le_bytes());
            payload.extend_from_slice(&(snapshot_blob.len() as u32).to_le_bytes());
            payload.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            payload.extend_from_slice(&checksum32(chunk).to_le_bytes());
            payload.extend_from_slice(chunk);
            self.append_page(
                PageType::Meta,
                generation,
                checkpoint_lsn,
                checkpoint_lsn,
                &payload,
            )?;
        }
        Ok(())
    }

    pub fn read_page(&mut self, page_id: u64) -> Result<Option<Page>, AxonError> {
        if page_id < FIRST_DATA_PAGE_ID || page_id >= self.superblock.page_count {
            return Ok(None);
        }
        let bytes = read_raw_page(&mut self.file, page_id)?;
        let page = Page::deserialize(&bytes)?;
        Ok(Some(page))
    }

    pub fn load_latest_snapshot(&mut self) -> Result<Option<(u64, Vec<u8>)>, AxonError> {
        let statuses = self.scan_pages()?;
        let mut checkpoints: BTreeMap<u64, Vec<(u32, u32, u32, u32, Vec<u8>)>> = BTreeMap::new();
        for status in statuses {
            if status.page_type != PageType::Meta || !status.checksum_ok {
                continue;
            }
            if let Some(page) = self.read_page(status.page_id)? {
                if page.header.payload_len < 28 {
                    continue;
                }
                let payload = &page.payload[..page.header.payload_len as usize];
                if payload[0..4] != MAGIC_SNAPSHOT {
                    continue;
                }
                let checkpoint_lsn = u64::from_le_bytes(payload[4..12].try_into().unwrap_or([0; 8]));
                let chunk_index = u32::from_le_bytes(payload[12..16].try_into().unwrap_or([0; 4]));
                let chunk_count = u32::from_le_bytes(payload[16..20].try_into().unwrap_or([0; 4]));
                let total_len = u32::from_le_bytes(payload[20..24].try_into().unwrap_or([0; 4]));
                let chunk_len = u32::from_le_bytes(payload[24..28].try_into().unwrap_or([0; 4]));
                let checksum_idx = 28;
                if payload.len() < checksum_idx + 4 {
                    continue;
                }
                let chunk_checksum =
                    u32::from_le_bytes(payload[checksum_idx..checksum_idx + 4].try_into().unwrap_or([0; 4]));
                let data_idx = checksum_idx + 4;
                if payload.len() < data_idx + chunk_len as usize {
                    continue;
                }
                let chunk = payload[data_idx..data_idx + chunk_len as usize].to_vec();
                if checksum32(&chunk) != chunk_checksum {
                    continue;
                }
                checkpoints
                    .entry(checkpoint_lsn)
                    .or_default()
                    .push((chunk_index, chunk_count, total_len, chunk_len, chunk));
            }
        }
        if let Some((lsn, mut chunks)) = checkpoints.pop_last() {
            chunks.sort_by_key(|x| x.0);
            let expected_count = chunks.first().map(|x| x.1).unwrap_or(0);
            if expected_count == 0 || chunks.len() as u32 != expected_count {
                return Ok(None);
            }
            let total_len = chunks.first().map(|x| x.2 as usize).unwrap_or(0);
            let mut blob = Vec::with_capacity(total_len);
            for (_, _, _, _, chunk) in chunks {
                blob.extend_from_slice(&chunk);
            }
            blob.truncate(total_len);
            return Ok(Some((lsn, blob)));
        }
        Ok(None)
    }

    pub fn read_journal_after(
        &mut self,
        after_lsn: u64,
    ) -> Result<Vec<MutationRecord>, AxonError> {
        let mut out = Vec::new();
        for status in self.scan_pages()? {
            if status.page_type != PageType::Journal || !status.checksum_ok {
                continue;
            }
            if status.lsn_end <= after_lsn {
                continue;
            }
            if let Some(page) = self.read_page(status.page_id)? {
                let payload = &page.payload[..page.header.payload_len as usize];
                let mut idx = 0usize;
                while idx + MutationRecord::SIZE <= payload.len() {
                    let record =
                        MutationRecord::from_bytes(&payload[idx..idx + MutationRecord::SIZE])?;
                    out.push(record);
                    idx += MutationRecord::SIZE;
                }
            }
        }
        out.sort_by_key(|record| record.tick);
        Ok(out)
    }

    pub fn commit_superblock(&mut self) -> Result<(), AxonError> {
        self.superblock.generation = self.superblock.generation.saturating_add(1);
        self.superblock.checksum = self.superblock.calculate_checksum();
        let next_slot = if self.active_superblock_slot == SUPERBLOCK_A_PAGE_ID {
            SUPERBLOCK_B_PAGE_ID
        } else {
            SUPERBLOCK_A_PAGE_ID
        };
        write_superblock_slot(&mut self.file, next_slot, &self.superblock)?;
        self.file.sync_data()?;
        self.active_superblock_slot = next_slot;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), AxonError> {
        self.file.flush()?;
        Ok(())
    }

    pub fn sync_all(&mut self) -> Result<(), AxonError> {
        self.file.sync_all()?;
        Ok(())
    }

    pub fn scan_pages(&mut self) -> Result<Vec<PageStatus>, AxonError> {
        let mut statuses = Vec::new();
        let count = self.superblock.page_count;
        let mut page_id = FIRST_DATA_PAGE_ID;
        while page_id < count {
            let raw = read_raw_page(&mut self.file, page_id)?;
            match Page::deserialize(&raw) {
                Ok(page) => statuses.push(PageStatus {
                    page_id,
                    page_type: page.header.page_type,
                    payload_len: page.header.payload_len,
                    generation: page.header.generation,
                    lsn_begin: page.header.lsn_begin,
                    lsn_end: page.header.lsn_end,
                    checksum_ok: true,
                }),
                Err(_) => {
                    let header_type = raw[12];
                    statuses.push(PageStatus {
                        page_id,
                        page_type: PageType::from_u8(header_type).unwrap_or(PageType::Free),
                        payload_len: 0,
                        generation: 0,
                        lsn_begin: 0,
                        lsn_end: 0,
                        checksum_ok: false,
                    });
                }
            }
            page_id += 1;
        }
        Ok(statuses)
    }
}

pub fn summarize_page_types(statuses: &[PageStatus]) -> HashMap<PageType, usize> {
    let mut map = HashMap::new();
    for status in statuses {
        *map.entry(status.page_type).or_insert(0usize) += 1usize;
    }
    map
}

fn write_raw_page(file: &mut File, page_id: u64, bytes: &[u8; PAGE_SIZE]) -> Result<(), AxonError> {
    file.seek(SeekFrom::Start(page_id * PAGE_SIZE as u64))?;
    file.write_all(bytes)?;
    Ok(())
}

fn read_raw_page(file: &mut File, page_id: u64) -> Result<[u8; PAGE_SIZE], AxonError> {
    let mut out = [0u8; PAGE_SIZE];
    file.seek(SeekFrom::Start(page_id * PAGE_SIZE as u64))?;
    file.read_exact(&mut out)?;
    Ok(out)
}

fn write_superblock_slot(file: &mut File, slot_page_id: u64, sb: &Superblock) -> Result<(), AxonError> {
    let mut page = [0u8; PAGE_SIZE];
    let sb_bytes = sb.serialize();
    page[..sb_bytes.len()].copy_from_slice(&sb_bytes);
    write_raw_page(file, slot_page_id, &page)
}

fn load_superblock_slot(file: &mut File, slot_page_id: u64) -> Result<Superblock, AxonError> {
    let raw = read_raw_page(file, slot_page_id)?;
    let mut sb_bytes = [0u8; PAGE_PAYLOAD_SIZE];
    sb_bytes.copy_from_slice(&raw[..PAGE_PAYLOAD_SIZE]);
    Superblock::deserialize(&sb_bytes)
}

fn load_latest_superblock(file: &File) -> Result<(Superblock, u64), AxonError> {
    let mut cloned = file.try_clone()?;
    let mut candidates = Vec::new();
    if let Ok(sb_a) = load_superblock_slot(&mut cloned, SUPERBLOCK_A_PAGE_ID) {
        candidates.push((sb_a, SUPERBLOCK_A_PAGE_ID));
    }
    if let Ok(sb_b) = load_superblock_slot(&mut cloned, SUPERBLOCK_B_PAGE_ID) {
        candidates.push((sb_b, SUPERBLOCK_B_PAGE_ID));
    }
    if candidates.is_empty() {
        return Err(AxonError::InvalidFormat(
            "unable to load any valid superblock".to_string(),
        ));
    }
    candidates.sort_by_key(|(sb, _)| sb.generation);
    Ok(candidates.pop().unwrap())
}

fn update_region_counter(superblock: &mut Superblock, page_type: PageType, page_id: u64) {
    let idx = match page_type {
        PageType::Meta => 0usize,
        PageType::AssemblyNode => 1usize,
        PageType::EdgeCsr => 2usize,
        PageType::EdgeDelta => 3usize,
        PageType::Episode => 4usize,
        PageType::Concept => 5usize,
        PageType::Journal => 6usize,
        PageType::ObsTile | PageType::AllocMap | PageType::Free => 7usize,
    };
    let root = &mut superblock.region_roots[idx];
    if root.root_page_id == 0 {
        root.root_page_id = page_id;
    }
    root.page_count = root.page_count.saturating_add(1);
}
