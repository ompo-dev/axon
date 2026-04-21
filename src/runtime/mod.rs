use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::cli::{CompactArgs, IngestArgs, RunMode, TuiArgs};
use crate::config::{
    CHECKPOINT_JOURNAL_BYTES, CHECKPOINT_MILLIS, DEFAULT_RANDOM_SEED, JOURNAL_FLUSH_BYTES,
    JOURNAL_FLUSH_MILLIS, TICK_MILLIS,
};
use crate::cortex::BrainState;
use crate::error::AxonError;
use crate::gpu::GpuBackend;
use crate::memory::MemoryState;
use crate::semantic::SemanticState;
use crate::storage::{BrainFile, MutationKind, MutationRecord};
use crate::tui::{self, AssemblyView, ConceptView, InputEvent, RenderSnapshot, UiMode};

#[derive(Clone)]
struct RuntimeShared {
    brain: BrainState,
    memory: MemoryState,
    semantic: SemanticState,
    ui_mode: UiMode,
    running: bool,
    use_gpu: bool,
    pending_mutations: Vec<MutationRecord>,
    pending_bytes: usize,
    next_lsn: u64,
    last_checkpoint_lsn: u64,
    force_flush: bool,
    force_checkpoint: bool,
    brain_path: String,
    ticks_total: u64,
    ticks_last_second: u64,
    last_tick_latency_us: u64,
}

pub fn run_tui(args: TuiArgs) -> Result<(), AxonError> {
    let brain_name = args
        .brain
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("brain");
    let mut brain_file = BrainFile::open_or_create(
        &args.brain,
        args.create_if_missing,
        brain_name,
        mode_to_u8(args.mode),
    )?;
    let mut loaded = load_state(&mut brain_file, args.mode)?;
    if loaded.brain.mode != args.mode {
        loaded.brain.mode = args.mode;
    }
    if let Some(dict_path) = args.dict.as_ref() {
        let inserted = loaded.semantic.ingest_txt_file(dict_path)?;
        if inserted > 0 {
            seed_brain_from_semantic(
                &mut loaded.brain,
                &loaded.semantic,
                &mut loaded.pending_mutations,
            );
        }
    }
    let gpu_backend = GpuBackend::probe();
    let shared = Arc::new(Mutex::new(RuntimeShared {
        brain: loaded.brain,
        memory: loaded.memory,
        semantic: loaded.semantic,
        ui_mode: UiMode::Chat,
        running: true,
        use_gpu: gpu_backend.available,
        pending_mutations: loaded.pending_mutations,
        pending_bytes: 0,
        next_lsn: loaded.last_lsn.saturating_add(1),
        last_checkpoint_lsn: loaded.last_lsn,
        force_flush: false,
        force_checkpoint: false,
        brain_path: args.brain.display().to_string(),
        ticks_total: 0,
        ticks_last_second: 0,
        last_tick_latency_us: 0,
    }));
    let storage = Arc::new(Mutex::new(brain_file));

    let (input_tx, input_rx) = mpsc::channel::<InputEvent>();
    let input_handle = tui::spawn_input_thread(input_tx);
    let render_handle = spawn_render_thread(shared.clone(), gpu_backend.clone());
    let telemetry_handle = spawn_telemetry_thread(shared.clone());
    let persist_handle = spawn_persist_thread(shared.clone(), storage.clone());

    let tick_result = run_tick_loop(shared.clone(), input_rx);
    {
        let mut state = shared.lock().map_err(|_| AxonError::State("runtime lock poisoned".to_string()))?;
        state.running = false;
        state.force_flush = true;
        state.force_checkpoint = true;
    }

    let _ = input_handle.join();
    let _ = render_handle.join();
    let _ = telemetry_handle.join();
    let _ = persist_handle.join();
    tick_result
}

pub fn run_ingest(args: IngestArgs) -> Result<(), AxonError> {
    let mut brain_file = BrainFile::open_or_create(&args.brain, true, "brain", 1)?;
    let mut loaded = load_state(&mut brain_file, RunMode::Stochastic)?;
    let inserted = loaded.semantic.ingest_txt_file(&args.dict)?;
    seed_brain_from_semantic(
        &mut loaded.brain,
        &loaded.semantic,
        &mut loaded.pending_mutations,
    );
    persist_all(
        &mut brain_file,
        &loaded.brain,
        &loaded.memory,
        &loaded.semantic,
        &mut loaded.pending_mutations,
        &mut loaded.last_lsn,
    )?;
    println!(
        "ingested {} concept entries into {}",
        inserted,
        brain_file.path().display()
    );
    Ok(())
}

pub fn run_compact(args: CompactArgs) -> Result<(), AxonError> {
    let mut source = BrainFile::open_or_create(&args.brain, false, "brain", 1)?;
    let mut loaded = load_state(&mut source, RunMode::Stochastic)?;
    let compact_path = compact_temp_path(&args.brain);
    if compact_path.exists() {
        fs::remove_file(&compact_path)?;
    }
    let mut target = BrainFile::open_or_create(&compact_path, true, "brain", 1)?;
    persist_all(
        &mut target,
        &loaded.brain,
        &loaded.memory,
        &loaded.semantic,
        &mut loaded.pending_mutations,
        &mut loaded.last_lsn,
    )?;
    target.sync_all()?;
    drop(target);
    fs::rename(&compact_path, &args.brain)?;
    println!("compacted brain file {}", args.brain.display());
    Ok(())
}

fn compact_temp_path(path: &PathBuf) -> PathBuf {
    let mut temp = path.clone();
    if let Some(ext) = path.extension() {
        let mut ext_owned = ext.to_os_string();
        ext_owned.push(".compact");
        temp.set_extension(ext_owned);
    } else {
        temp.set_extension("compact.axon");
    }
    temp
}

fn run_tick_loop(shared: Arc<Mutex<RuntimeShared>>, input_rx: Receiver<InputEvent>) -> Result<(), AxonError> {
    let tick_interval = Duration::from_millis(TICK_MILLIS);
    loop {
        let started = Instant::now();
        {
            let mut state = shared
                .lock()
                .map_err(|_| AxonError::State("runtime lock poisoned".to_string()))?;
            if !state.running {
                break;
            }
            drain_input_events(&mut state, &input_rx);
            let mut mutations = Vec::new();
            let use_gpu = state.use_gpu;
            let RuntimeShared { brain, memory, semantic, .. } = &mut *state;
            let _ = brain.step(memory, semantic, &mut mutations, use_gpu);
            state.pending_bytes = state
                .pending_bytes
                .saturating_add(mutations.len() * MutationRecord::SIZE);
            state.pending_mutations.extend(mutations);
            state.ticks_total = state.ticks_total.saturating_add(1);
            state.ticks_last_second = state.ticks_last_second.saturating_add(1);
            state.last_tick_latency_us = started.elapsed().as_micros() as u64;
        }
        let elapsed = started.elapsed();
        if elapsed < tick_interval {
            thread::sleep(tick_interval - elapsed);
        }
    }
    Ok(())
}

fn drain_input_events(state: &mut RuntimeShared, input_rx: &Receiver<InputEvent>) {
    while let Ok(event) = input_rx.try_recv() {
        match event {
            InputEvent::Text(text) => state.brain.queue_user_text(&text),
            InputEvent::SwitchMode(mode) => state.ui_mode = mode,
            InputEvent::ForceFlush => state.force_flush = true,
            InputEvent::ForceCheckpoint => {
                state.force_flush = true;
                state.force_checkpoint = true;
            }
            InputEvent::Quit => state.running = false,
        }
    }
}

fn spawn_persist_thread(
    shared: Arc<Mutex<RuntimeShared>>,
    storage: Arc<Mutex<BrainFile>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut last_flush = Instant::now();
        let mut last_checkpoint = Instant::now();
        let mut journal_since_checkpoint = 0usize;
        loop {
            thread::sleep(Duration::from_millis(50));
            let mut maybe_records = Vec::new();
            let mut maybe_snapshot = None::<(BrainState, MemoryState, SemanticState, u64)>;
            let mut lsn_begin = 0u64;
            let mut lsn_end = 0u64;
            let should_break;
            {
                let mut state = match shared.lock() {
                    Ok(guard) => guard,
                    Err(_) => break,
                };
                let time_due = last_flush.elapsed() >= Duration::from_millis(JOURNAL_FLUSH_MILLIS);
                let size_due = state.pending_bytes >= JOURNAL_FLUSH_BYTES;
                let forced = state.force_flush;
                let should_flush = forced || time_due || size_due || !state.running;
                if should_flush && !state.pending_mutations.is_empty() {
                    let records = std::mem::take(&mut state.pending_mutations);
                    state.pending_bytes = 0;
                    lsn_begin = records.first().map(|r| r.tick).unwrap_or(state.next_lsn);
                    lsn_end = records.last().map(|r| r.tick).unwrap_or(state.next_lsn);
                    maybe_records = records;
                    state.force_flush = false;
                }
                let checkpoint_due_time =
                    last_checkpoint.elapsed() >= Duration::from_millis(CHECKPOINT_MILLIS);
                let checkpoint_due_size = journal_since_checkpoint >= CHECKPOINT_JOURNAL_BYTES;
                let checkpoint_forced = state.force_checkpoint;
                let should_checkpoint =
                    checkpoint_forced || checkpoint_due_time || checkpoint_due_size || !state.running;
                if should_checkpoint {
                    maybe_snapshot = Some((
                        state.brain.clone(),
                        state.memory.clone(),
                        state.semantic.clone(),
                        state.next_lsn,
                    ));
                    state.force_checkpoint = false;
                }
                should_break = !state.running && state.pending_mutations.is_empty();
            }

            if !maybe_records.is_empty() {
                if let Ok(mut file) = storage.lock() {
                    let generation = file.superblock.generation.saturating_add(1);
                    if file
                        .write_journal_records(generation, lsn_begin, lsn_end, &maybe_records)
                        .is_ok()
                    {
                        let _ = file.commit_superblock();
                        journal_since_checkpoint = journal_since_checkpoint
                            .saturating_add(maybe_records.len() * MutationRecord::SIZE);
                        last_flush = Instant::now();
                    }
                }
            }

            if let Some((brain, memory, semantic, checkpoint_lsn)) = maybe_snapshot {
                if let Ok(mut file) = storage.lock() {
                    let snapshot = encode_snapshot(&brain, &memory, &semantic);
                    let generation = file.superblock.generation.saturating_add(1);
                    if file
                        .write_snapshot(generation, checkpoint_lsn, &snapshot)
                        .and_then(|_| file.commit_superblock())
                        .is_ok()
                    {
                        journal_since_checkpoint = 0;
                        last_checkpoint = Instant::now();
                    }
                }
            }
            if should_break {
                break;
            }
        }
    })
}

fn spawn_telemetry_thread(shared: Arc<Mutex<RuntimeShared>>) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(1));
        let mut guard = match shared.lock() {
            Ok(guard) => guard,
            Err(_) => break,
        };
        if !guard.running && guard.ticks_last_second == 0 {
            break;
        }
        guard.ticks_last_second = 0;
    })
}

fn spawn_render_thread(
    shared: Arc<Mutex<RuntimeShared>>,
    gpu: Arc<GpuBackend>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(120));
        let snapshot = {
            let state = match shared.lock() {
                Ok(guard) => guard,
                Err(_) => break,
            };
            if !state.running && state.pending_mutations.is_empty() {
                break;
            }
            build_render_snapshot(&state)
        };
        tui::render_frame(&snapshot, &gpu);
    })
}

fn build_render_snapshot(state: &RuntimeShared) -> RenderSnapshot {
    let transcript = state
        .brain
        .transcript
        .iter()
        .map(|msg| {
            if msg.from_user {
                format!("you> {}", msg.text)
            } else {
                format!("axon> {}", msg.text)
            }
        })
        .collect();
    let top_assemblies = state
        .brain
        .top_active_assemblies(8)
        .into_iter()
        .map(|a| AssemblyView {
            id: a.id,
            symbol: a.symbol,
            activation: a.activation,
            stability: a.stability,
        })
        .collect();
    let top_concepts = state
        .semantic
        .top_concepts(8)
        .into_iter()
        .map(|c| ConceptView {
            id: c.id,
            lemma: c.lemma.clone(),
            stability: c.stability,
            connectivity: c.connectivity,
        })
        .collect();
    RenderSnapshot {
        mode: state.ui_mode,
        brain_path: state.brain_path.clone(),
        tick: state.brain.tick,
        use_gpu: state.use_gpu,
        pending_mutations: state.pending_mutations.len(),
        pending_input: state.brain.pending_input.len(),
        latency_us: state.last_tick_latency_us,
        spawn_count: state.brain.stats.spawn_count,
        merge_count: state.brain.stats.merge_count,
        prune_count: state.brain.stats.prune_count,
        transcript,
        top_assemblies,
        top_concepts,
    }
}

struct LoadedState {
    brain: BrainState,
    memory: MemoryState,
    semantic: SemanticState,
    pending_mutations: Vec<MutationRecord>,
    last_lsn: u64,
}

fn load_state(file: &mut BrainFile, requested_mode: RunMode) -> Result<LoadedState, AxonError> {
    let mut brain = BrainState::new(requested_mode, DEFAULT_RANDOM_SEED);
    let mut memory = MemoryState::new();
    let mut semantic = SemanticState::new();
    let mut last_lsn = 0u64;

    if let Some((snapshot_lsn, blob)) = file.load_latest_snapshot()? {
        if let Ok((b, m, s)) = decode_snapshot(&blob, requested_mode) {
            brain = b;
            memory = m;
            semantic = s;
            last_lsn = snapshot_lsn;
        }
    }
    let journal_records = file.read_journal_after(last_lsn)?;
    replay_mutations(&mut brain, &journal_records);
    if let Some(max_tick) = journal_records.iter().map(|r| r.tick).max() {
        last_lsn = last_lsn.max(max_tick);
    }
    Ok(LoadedState {
        brain,
        memory,
        semantic,
        pending_mutations: Vec::new(),
        last_lsn,
    })
}

fn replay_mutations(brain: &mut BrainState, records: &[MutationRecord]) {
    let mut last_char: Option<char> = None;
    for record in records {
        match record.kind {
            MutationKind::InputChar => {
                if let Some(ch) = char::from_u32(record.a) {
                    *brain.char_frequency.entry(ch as u32).or_insert(0) += 1;
                    if let Some(prev) = last_char {
                        *brain
                            .transitions
                            .entry((prev as u32, ch as u32))
                            .or_insert(0) += 1;
                    }
                    last_char = Some(ch);
                }
                brain.tick = brain.tick.max(record.tick);
            }
            MutationKind::EdgeUpdate => {
                let from = record.a;
                let to = record.b;
                brain.ensure_assembly_capacity(from.max(to) as usize + 1);
                brain.set_delta_edge_weight(from, to, record.value);
                brain.tick = brain.tick.max(record.tick);
            }
            _ => {
                brain.tick = brain.tick.max(record.tick);
            }
        }
    }
}

fn seed_brain_from_semantic(
    brain: &mut BrainState,
    semantic: &SemanticState,
    pending_mutations: &mut Vec<MutationRecord>,
) {
    for concept in &semantic.concepts {
        for ch in concept.canonical.chars() {
            brain.queue_user_text(&ch.to_string());
            pending_mutations.push(MutationRecord {
                kind: MutationKind::SemanticLink,
                flags: 0,
                tick: brain.tick,
                a: concept.id,
                b: ch as u32,
                c: 0,
                value: concept.stability,
                extra: concept.connectivity,
            });
        }
    }
}

fn persist_all(
    file: &mut BrainFile,
    brain: &BrainState,
    memory: &MemoryState,
    semantic: &SemanticState,
    pending_mutations: &mut Vec<MutationRecord>,
    last_lsn: &mut u64,
) -> Result<(), AxonError> {
    if !pending_mutations.is_empty() {
        let lsn_begin = pending_mutations.first().map(|r| r.tick).unwrap_or(*last_lsn);
        let lsn_end = pending_mutations.last().map(|r| r.tick).unwrap_or(*last_lsn);
        file.write_journal_records(
            file.superblock.generation.saturating_add(1),
            lsn_begin,
            lsn_end,
            pending_mutations,
        )?;
        *last_lsn = (*last_lsn).max(lsn_end);
        pending_mutations.clear();
    }
    let blob = encode_snapshot(brain, memory, semantic);
    file.write_snapshot(
        file.superblock.generation.saturating_add(1),
        (*last_lsn).max(brain.tick),
        &blob,
    )?;
    file.commit_superblock()?;
    file.sync_all()?;
    Ok(())
}

fn mode_to_u8(mode: RunMode) -> u8 {
    match mode {
        RunMode::Deterministic => 0,
        RunMode::Stochastic => 1,
    }
}

fn encode_snapshot(brain: &BrainState, memory: &MemoryState, semantic: &SemanticState) -> Vec<u8> {
    let mut writer = BinaryWriter::new();
    writer.put_u32(0x42534E50);
    writer.put_u16(1);
    writer.put_u8(mode_to_u8(brain.mode));
    writer.put_u8(0);
    writer.put_u64(brain.tick);

    writer.put_u32(brain.assemblies.len() as u32);
    for assembly in &brain.assemblies {
        writer.put_u32(assembly.id);
        writer.put_u32(assembly.symbol.map(|c| c as u32).unwrap_or(0));
        writer.put_f32(assembly.activation);
        writer.put_f32(assembly.stability);
        writer.put_u32(assembly.support_count);
        writer.put_f32(assembly.novelty);
        writer.put_u64(assembly.last_tick);
    }
    writer.put_u32(brain.delta_edges.len() as u32);
    for edge in &brain.delta_edges {
        writer.put_u32(edge.from);
        writer.put_u32(edge.to);
        writer.put_f32(edge.weight);
        writer.put_f32(edge.utility);
        writer.put_u32(edge.weak_ticks);
    }
    writer.put_u32(brain.char_frequency.len() as u32);
    for (code, count) in &brain.char_frequency {
        writer.put_u32(*code);
        writer.put_u64(*count);
    }
    writer.put_u32(brain.transitions.len() as u32);
    for ((from, to), count) in &brain.transitions {
        writer.put_u32(*from);
        writer.put_u32(*to);
        writer.put_u64(*count);
    }
    writer.put_u32(memory.episodes.len() as u32);
    for episode in &memory.episodes {
        writer.put_u64(episode.id);
        writer.put_u64(episode.tick);
        writer.put_f32(episode.salience);
        writer.put_f32(episode.recall_score);
        writer.put_string(&episode.trace);
    }
    writer.put_u32(semantic.concepts.len() as u32);
    for concept in &semantic.concepts {
        writer.put_u32(concept.id);
        writer.put_string(&concept.lemma);
        writer.put_string(&concept.canonical);
        writer.put_string(&concept.definition);
        writer.put_u32(concept.recurrence);
        writer.put_f32(concept.connectivity);
        writer.put_f32(concept.stability);
    }
    writer.into_inner()
}

fn decode_snapshot(
    bytes: &[u8],
    requested_mode: RunMode,
) -> Result<(BrainState, MemoryState, SemanticState), AxonError> {
    let mut reader = BinaryReader::new(bytes);
    let magic = reader.get_u32()?;
    if magic != 0x42534E50 {
        return Err(AxonError::InvalidFormat("snapshot magic mismatch".to_string()));
    }
    let version = reader.get_u16()?;
    if version != 1 {
        return Err(AxonError::InvalidFormat(format!(
            "unsupported snapshot version {version}"
        )));
    }
    let mode_u8 = reader.get_u8()?;
    let _reserved = reader.get_u8()?;
    let mut brain = BrainState::new(
        match mode_u8 {
            0 => RunMode::Deterministic,
            _ => requested_mode,
        },
        DEFAULT_RANDOM_SEED,
    );
    brain.tick = reader.get_u64()?;

    let assemblies = reader.get_u32()? as usize;
    for _ in 0..assemblies {
        let id = reader.get_u32()?;
        let symbol_raw = reader.get_u32()?;
        let symbol = if symbol_raw == 0 {
            None
        } else {
            char::from_u32(symbol_raw)
        };
        let activation = reader.get_f32()?;
        let stability = reader.get_f32()?;
        let support_count = reader.get_u32()?;
        let novelty = reader.get_f32()?;
        let last_tick = reader.get_u64()?;
        brain.ensure_assembly_capacity(id as usize + 1);
        if brain.assemblies.len() <= id as usize {
            continue;
        }
        brain.assemblies[id as usize].id = id;
        brain.assemblies[id as usize].symbol = symbol;
        brain.assemblies[id as usize].activation = activation;
        brain.assemblies[id as usize].stability = stability;
        brain.assemblies[id as usize].support_count = support_count;
        brain.assemblies[id as usize].novelty = novelty;
        brain.assemblies[id as usize].last_tick = last_tick;
        brain.activations[id as usize] = activation;
        if let Some(ch) = symbol {
            brain.char_nodes.insert(ch as u32, id);
        }
    }
    let edge_count = reader.get_u32()? as usize;
    brain.delta_edges.clear();
    for _ in 0..edge_count {
        brain.delta_edges.push(crate::cortex::DeltaEdge {
            from: reader.get_u32()?,
            to: reader.get_u32()?,
            weight: reader.get_f32()?,
            utility: reader.get_f32()?,
            weak_ticks: reader.get_u32()?,
        });
    }
    let char_count = reader.get_u32()? as usize;
    for _ in 0..char_count {
        brain.char_frequency.insert(reader.get_u32()?, reader.get_u64()?);
    }
    let transition_count = reader.get_u32()? as usize;
    for _ in 0..transition_count {
        let from = reader.get_u32()?;
        let to = reader.get_u32()?;
        let count = reader.get_u64()?;
        brain.transitions.insert((from, to), count);
    }
    let mut memory = MemoryState::new();
    let episode_count = reader.get_u32()? as usize;
    for _ in 0..episode_count {
        memory.episodes.push(crate::memory::Episode {
            id: reader.get_u64()?,
            tick: reader.get_u64()?,
            salience: reader.get_f32()?,
            recall_score: reader.get_f32()?,
            trace: reader.get_string()?,
        });
    }
    let mut semantic = SemanticState::new();
    let concept_count = reader.get_u32()? as usize;
    for _ in 0..concept_count {
        let id = reader.get_u32()?;
        let lemma = reader.get_string()?;
        let canonical = reader.get_string()?;
        let definition = reader.get_string()?;
        let recurrence = reader.get_u32()?;
        let connectivity = reader.get_f32()?;
        let stability = reader.get_f32()?;
        if semantic.add_or_update_concept(&lemma, &definition) {
            if let Some(concept) = semantic.concepts.last_mut() {
                concept.id = id;
                concept.canonical = canonical;
                concept.recurrence = recurrence;
                concept.connectivity = connectivity;
                concept.stability = stability;
            }
        }
    }
    Ok((brain, memory, semantic))
}

struct BinaryWriter {
    bytes: Vec<u8>,
}

impl BinaryWriter {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn put_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn put_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn put_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn put_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn put_f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn put_string(&mut self, value: &str) {
        self.put_u32(value.len() as u32);
        self.bytes.extend_from_slice(value.as_bytes());
    }
    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }
    fn get_slice(&mut self, n: usize) -> Result<&'a [u8], AxonError> {
        if self.cursor + n > self.bytes.len() {
            return Err(AxonError::InvalidFormat("snapshot truncated".to_string()));
        }
        let slice = &self.bytes[self.cursor..self.cursor + n];
        self.cursor += n;
        Ok(slice)
    }
    fn get_u8(&mut self) -> Result<u8, AxonError> {
        Ok(self.get_slice(1)?[0])
    }
    fn get_u16(&mut self) -> Result<u16, AxonError> {
        Ok(u16::from_le_bytes(self.get_slice(2)?.try_into().unwrap_or([0; 2])))
    }
    fn get_u32(&mut self) -> Result<u32, AxonError> {
        Ok(u32::from_le_bytes(self.get_slice(4)?.try_into().unwrap_or([0; 4])))
    }
    fn get_u64(&mut self) -> Result<u64, AxonError> {
        Ok(u64::from_le_bytes(self.get_slice(8)?.try_into().unwrap_or([0; 8])))
    }
    fn get_f32(&mut self) -> Result<f32, AxonError> {
        Ok(f32::from_le_bytes(self.get_slice(4)?.try_into().unwrap_or([0; 4])))
    }
    fn get_string(&mut self) -> Result<String, AxonError> {
        let len = self.get_u32()? as usize;
        let data = self.get_slice(len)?;
        Ok(String::from_utf8_lossy(data).into_owned())
    }
}
