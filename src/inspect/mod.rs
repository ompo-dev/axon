use std::collections::BTreeMap;

use crate::cli::{DumpHeaderArgs, DumpRegionArgs, InspectArgs, VerifyArgs};
use crate::error::AxonError;
use crate::storage::{BrainFile, PageStatus, summarize_page_types};

pub fn inspect_brain(args: InspectArgs) -> Result<(), AxonError> {
    let mut brain = BrainFile::open_readonly(&args.brain)?;
    let statuses = brain.scan_pages()?;
    let counts = summarize_page_types(&statuses);
    println!("brain: {}", args.brain.display());
    println!("generation: {}", brain.superblock.generation);
    println!("page_count: {}", brain.superblock.page_count);
    println!("commit_lsn: {}", brain.superblock.commit_lsn);
    println!("regions:");
    for (idx, root) in brain.superblock.region_roots.iter().enumerate() {
        if root.page_count > 0 {
            println!(
                "  region[{idx}] root={} pages={}",
                root.root_page_id, root.page_count
            );
        }
    }
    println!("page type counts:");
    for (ty, count) in counts {
        println!("  {:<14} {}", ty, count);
    }
    let invalid = statuses.iter().filter(|status| !status.checksum_ok).count();
    println!("invalid pages: {invalid}");
    Ok(())
}

pub fn dump_header(args: DumpHeaderArgs) -> Result<(), AxonError> {
    let brain = BrainFile::open_readonly(&args.brain)?;
    println!(
        "magic: {}",
        String::from_utf8_lossy(&brain.superblock.magic)
    );
    println!("version: {}", brain.superblock.version);
    println!("generation: {}", brain.superblock.generation);
    println!("page_size: {}", brain.superblock.page_size);
    println!("page_count: {}", brain.superblock.page_count);
    println!("commit_lsn: {}", brain.superblock.commit_lsn);
    println!("caps: {}", brain.superblock.caps);
    println!("flags: {}", brain.superblock.flags);
    println!("name: {}", brain.superblock.brain_meta.name);
    println!("language: {}", brain.superblock.brain_meta.language);
    println!("mode: {}", brain.superblock.brain_meta.mode);
    println!(
        "created_unix_ms: {}",
        brain.superblock.brain_meta.created_unix_ms
    );
    println!(
        "updated_unix_ms: {}",
        brain.superblock.brain_meta.updated_unix_ms
    );
    Ok(())
}

pub fn dump_region(args: DumpRegionArgs) -> Result<(), AxonError> {
    let mut brain = BrainFile::open_readonly(&args.brain)?;
    let statuses = brain.scan_pages()?;
    let normalized = args.region.to_lowercase();
    let filtered: Vec<PageStatus> = statuses
        .into_iter()
        .filter(|status| match normalized.as_str() {
            "semantic" => matches!(status.page_type.to_string().as_str(), "CONCEPT" | "META"),
            "memory" => status.page_type.to_string() == "EPISODE",
            "cortex" => matches!(
                status.page_type.to_string().as_str(),
                "ASSEMBLY_NODE" | "EDGE_CSR" | "EDGE_DELTA"
            ),
            "journal" => status.page_type.to_string() == "JOURNAL",
            "obs" => status.page_type.to_string() == "OBS_TILE",
            _ => false,
        })
        .collect();
    if filtered.is_empty() {
        println!("no pages found for region '{}'", args.region);
        return Ok(());
    }
    println!("region '{}' pages (count={}):", args.region, filtered.len());
    for status in filtered.iter().take(64) {
        println!(
            "  page={} type={} payload={} gen={} lsn=[{},{}] checksum_ok={}",
            status.page_id,
            status.page_type,
            status.payload_len,
            status.generation,
            status.lsn_begin,
            status.lsn_end,
            status.checksum_ok
        );
    }
    if filtered.len() > 64 {
        println!("  ... truncated");
    }
    Ok(())
}

pub fn verify_brain(args: VerifyArgs) -> Result<(), AxonError> {
    let mut brain = BrainFile::open_readonly(&args.brain)?;
    let statuses = brain.scan_pages()?;
    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for status in &statuses {
        let entry = by_type
            .entry(status.page_type.to_string())
            .or_insert((0usize, 0usize));
        entry.0 += 1;
        if status.checksum_ok {
            entry.1 += 1;
        }
    }
    println!("verify: {}", args.brain.display());
    println!(
        "superblock generation={} page_count={} commit_lsn={}",
        brain.superblock.generation, brain.superblock.page_count, brain.superblock.commit_lsn
    );
    let mut invalid = 0usize;
    for status in &statuses {
        if !status.checksum_ok {
            invalid += 1;
            println!(
                "invalid page {} type={} (checksum/header failed)",
                status.page_id, status.page_type
            );
        }
    }
    for (kind, (total, ok)) in by_type {
        println!("type={kind:<14} ok={ok} total={total}");
    }
    if invalid > 0 {
        return Err(AxonError::InvalidFormat(format!(
            "verify failed: {} invalid pages found",
            invalid
        )));
    }
    println!("verify ok");
    Ok(())
}
