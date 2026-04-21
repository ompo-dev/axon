use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::axon_format::FIRST_DATA_PAGE_ID;
use crate::cli::{CompactArgs, IngestArgs, RunMode, TuiArgs};
use crate::config::{
    CHECKPOINT_JOURNAL_BYTES, CHECKPOINT_MILLIS, DEFAULT_RANDOM_SEED, JOURNAL_FLUSH_BYTES,
    JOURNAL_FLUSH_MILLIS, TICK_MILLIS,
};
use crate::cortex::{BrainState, DeltaEdge};
use crate::error::AxonError;
use crate::gpu::GpuBackend;
use crate::memory::{EdgeKind, MemoryEdge, MemoryNode, MemoryState, NodeKind, TemporalAnchor};
use crate::semantic::SemanticState;
use crate::storage::{BrainFile, MutationKind, MutationRecord};
use crate::tui::{
    self, AssemblyView, ConceptView, InputEvent, KeyCode, RenderSnapshot, SlashSuggestion,
    TerminalRenderer, TuiCommand, UiMode,
};

struct RuntimeShared {
    brain: BrainState,
    memory: MemoryState,
    semantic: SemanticState,
    ui_mode: UiMode,
    running: bool,
    use_gpu: bool,
    pending_mutations: Vec<MutationRecord>,
    pending_bytes: usize,
    force_flush: bool,
    force_checkpoint: bool,
    brain_path: String,
    ticks_last_second: u64,
    last_tick_latency_us: u64,
    render_dirty: bool,
    input_buffer: String,
    input_cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    history_draft: String,
    slash_catalog: Vec<SlashSuggestion>,
    slash_suggestions: Vec<SlashSuggestion>,
    slash_selected: usize,
    status_message: String,
    frame_id: u64,
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
        force_flush: false,
        force_checkpoint: false,
        brain_path: args.brain.display().to_string(),
        ticks_last_second: 0,
        last_tick_latency_us: 0,
        render_dirty: true,
        input_buffer: String::new(),
        input_cursor: 0,
        history: Vec::new(),
        history_index: None,
        history_draft: String::new(),
        slash_catalog: tui::default_slash_catalog(),
        slash_suggestions: Vec::new(),
        slash_selected: 0,
        status_message: "pronto".to_string(),
        frame_id: 0,
    }));
    let storage = Arc::new(Mutex::new(brain_file));

    let (input_tx, input_rx) = mpsc::channel::<InputEvent>();
    let _input_handle = tui::spawn_input_thread(input_tx);
    let render_handle = spawn_render_thread(shared.clone(), gpu_backend.clone());
    let telemetry_handle = spawn_telemetry_thread(shared.clone());
    let persist_handle = spawn_persist_thread(shared.clone(), storage.clone());

    let tick_result = run_tick_loop(shared.clone(), input_rx);
    {
        let mut state = shared
            .lock()
            .map_err(|_| AxonError::State("runtime lock poisoned".to_string()))?;
        state.running = false;
        state.force_flush = true;
        state.force_checkpoint = true;
        state.render_dirty = true;
    }

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

            let input_changed = drain_input_events(&mut state, &input_rx);
            let mut mutations = Vec::new();
            let use_gpu = state.use_gpu;
            let RuntimeShared {
                brain,
                memory,
                semantic,
                ..
            } = &mut *state;
            let emitted = brain.step(memory, semantic, &mut mutations, use_gpu);

            state.pending_bytes = state
                .pending_bytes
                .saturating_add(mutations.len() * MutationRecord::SIZE);
            state.pending_mutations.extend(mutations);
            state.ticks_last_second = state.ticks_last_second.saturating_add(1);
            state.last_tick_latency_us = started.elapsed().as_micros() as u64;
            if input_changed || emitted.is_some() {
                state.render_dirty = true;
            }
        }
        let elapsed = started.elapsed();
        if elapsed < tick_interval {
            thread::sleep(tick_interval - elapsed);
        }
    }
    Ok(())
}

fn drain_input_events(state: &mut RuntimeShared, input_rx: &Receiver<InputEvent>) -> bool {
    let mut changed = false;
    while let Ok(event) = input_rx.try_recv() {
        changed = true;
        match event {
            InputEvent::Key(code) => handle_key_event(state, code),
            InputEvent::Paste(text) => {
                begin_manual_edit(state);
                insert_text(state, &text);
                refresh_slash(state);
            }
            InputEvent::Resize { cols, rows } => {
                state.status_message = format!("resize {cols}x{rows}");
            }
            InputEvent::Submit(line) => submit_line(state, line),
            InputEvent::Command(command) => apply_command(state, command),
            InputEvent::Quit => {
                state.running = false;
                state.status_message = "encerrando".to_string();
            }
        }
    }
    if changed {
        state.render_dirty = true;
    }
    changed
}

fn handle_key_event(state: &mut RuntimeShared, code: KeyCode) {
    match code {
        KeyCode::Char(ch) => {
            begin_manual_edit(state);
            insert_char(state, ch);
            refresh_slash(state);
        }
        KeyCode::Backspace => {
            begin_manual_edit(state);
            delete_char_before_cursor(state);
            refresh_slash(state);
        }
        KeyCode::Delete => {
            begin_manual_edit(state);
            delete_char_at_cursor(state);
            refresh_slash(state);
        }
        KeyCode::Left => {
            if state.input_cursor > 0 {
                state.input_cursor -= 1;
            }
        }
        KeyCode::Right => {
            let len = char_count(&state.input_buffer);
            if state.input_cursor < len {
                state.input_cursor += 1;
            }
        }
        KeyCode::Up => {
            if slash_visible(state) {
                if !state.slash_suggestions.is_empty() {
                    if state.slash_selected == 0 {
                        state.slash_selected = state.slash_suggestions.len() - 1;
                    } else {
                        state.slash_selected -= 1;
                    }
                }
            } else {
                history_prev(state);
                refresh_slash(state);
            }
        }
        KeyCode::Down => {
            if slash_visible(state) {
                if !state.slash_suggestions.is_empty() {
                    state.slash_selected = (state.slash_selected + 1) % state.slash_suggestions.len();
                }
            } else {
                history_next(state);
                refresh_slash(state);
            }
        }
        KeyCode::Tab => {
            autocomplete_slash(state);
        }
        KeyCode::Esc => {
            state.slash_suggestions.clear();
            state.slash_selected = 0;
        }
        KeyCode::Enter => {
            let current = state.input_buffer.clone();
            submit_line(state, current);
        }
        KeyCode::F1 => apply_command(state, TuiCommand::SwitchMode(UiMode::Chat)),
        KeyCode::F2 => apply_command(state, TuiCommand::SwitchMode(UiMode::Observatory)),
        KeyCode::F5 => apply_command(state, TuiCommand::ForceFlush),
        KeyCode::F6 => apply_command(state, TuiCommand::ForceCheckpoint),
        KeyCode::CtrlC => {
            state.running = false;
            state.status_message = "ctrl+c recebido".to_string();
        }
        KeyCode::Unknown => {}
    }
}

fn submit_line(state: &mut RuntimeShared, mut line: String) {
    line = line.trim().to_string();
    if line.is_empty() {
        state.input_buffer.clear();
        state.input_cursor = 0;
        state.history_index = None;
        state.history_draft.clear();
        refresh_slash(state);
        return;
    }

    push_history(state, &line);
    state.history_index = None;
    state.history_draft.clear();

    if line.starts_with('/') {
        if let Some(command) = tui::parse_inline_command(&line) {
            apply_command(state, command);
        } else if let Some(selected) = state.slash_suggestions.get(state.slash_selected).cloned() {
            if let Some(command) = tui::parse_inline_command(&selected.command) {
                apply_command(state, command);
            } else {
                state.status_message = format!("comando invalido: {}", line);
            }
        } else {
            state.status_message = format!("comando desconhecido: {}", line);
        }
    } else {
        state.brain.queue_user_text(&line);
        state.status_message = "input enfileirado".to_string();
    }

    state.input_buffer.clear();
    state.input_cursor = 0;
    refresh_slash(state);
}

fn apply_command(state: &mut RuntimeShared, command: TuiCommand) {
    match command {
        TuiCommand::SwitchMode(mode) => {
            state.ui_mode = mode;
            state.status_message = format!("modo {:?}", mode);
        }
        TuiCommand::ForceFlush => {
            state.force_flush = true;
            state.status_message = "flush solicitado".to_string();
        }
        TuiCommand::ForceCheckpoint => {
            state.force_flush = true;
            state.force_checkpoint = true;
            state.status_message = "checkpoint solicitado".to_string();
        }
        TuiCommand::Quit => {
            state.running = false;
            state.status_message = "encerrando".to_string();
        }
        TuiCommand::SetRunMode(mode) => {
            state.brain.mode = mode;
            state.status_message = format!("run mode {:?}", mode);
        }
        TuiCommand::Correction { wrong, correct } => {
            let mut mutations = Vec::new();
            state
                .brain
                .apply_correction(&mut state.memory, &wrong, &correct, &mut mutations);
            state.pending_bytes = state
                .pending_bytes
                .saturating_add(mutations.len() * MutationRecord::SIZE);
            state.pending_mutations.extend(mutations);
            state.status_message = format!("correcao aplicada: {} -> {}", wrong, correct);
        }
        TuiCommand::Help => {
            state.status_message =
                "atalhos: F1 Chat | F2 Observatorio | F5 Flush | F6 Checkpoint".to_string();
        }
    }
    state.render_dirty = true;
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
                    lsn_begin = records.first().map(|r| r.tick).unwrap_or(0);
                    lsn_end = records.last().map(|r| r.tick).unwrap_or(lsn_begin);
                    maybe_records = records;
                    state.force_flush = false;
                    state.render_dirty = true;
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
                        state.brain.tick,
                    ));
                    state.force_checkpoint = false;
                    state.render_dirty = true;
                }
                should_break = !state.running && state.pending_mutations.is_empty();
            }

            if !maybe_records.is_empty() {
                if let Ok(mut file) = storage.lock() {
                    let generation = file.superblock.generation.saturating_add(1);
                    if file
                        .write_journal_records(generation, lsn_begin, lsn_end, &maybe_records)
                        .and_then(|_| file.commit_superblock())
                        .is_ok()
                    {
                        journal_since_checkpoint = journal_since_checkpoint
                            .saturating_add(maybe_records.len() * MutationRecord::SIZE);
                        last_flush = Instant::now();
                    }
                }
            }

            if let Some((brain, memory, semantic, checkpoint_lsn)) = maybe_snapshot {
                if let Ok(mut file) = storage.lock() {
                    refresh_region_roots(&mut file, &brain, &memory, &semantic);
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
    thread::spawn(move || {
        let frame_interval = Duration::from_millis(33);
        let mut renderer = TerminalRenderer::new();
        loop {
            thread::sleep(frame_interval);
            let mut should_break = false;
            let snapshot = {
                let mut state = match shared.lock() {
                    Ok(guard) => guard,
                    Err(_) => break,
                };
                if !state.running && state.pending_mutations.is_empty() && !state.render_dirty {
                    should_break = true;
                    None
                } else if !state.render_dirty {
                    None
                } else {
                    state.frame_id = state.frame_id.saturating_add(1);
                    let snap = build_render_snapshot(&state);
                    state.render_dirty = false;
                    Some(snap)
                }
            };
            if let Some(snapshot) = snapshot {
                let _ = renderer.render_frame(&snapshot, &gpu);
            }
            if should_break {
                break;
            }
        }
        let _ = renderer.shutdown();
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

    let mut top_concepts: Vec<ConceptView> = state
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
    if top_concepts.len() < 8 {
        for node in state.memory.top_hot_nodes(24) {
            if node.kind != NodeKind::Concept {
                continue;
            }
            if top_concepts.iter().any(|c| c.lemma == node.label) {
                continue;
            }
            top_concepts.push(ConceptView {
                id: node.id as u32,
                lemma: node.label.clone(),
                stability: node.temperature,
                connectivity: node.salience,
            });
            if top_concepts.len() >= 8 {
                break;
            }
        }
    }

    RenderSnapshot {
        mode: state.ui_mode,
        brain_path: state.brain_path.clone(),
        tick: state.brain.tick,
        run_mode: state.brain.mode,
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
        input_buffer: state.input_buffer.clone(),
        input_cursor: state.input_cursor,
        slash_suggestions: state.slash_suggestions.clone(),
        slash_selected: state.slash_selected,
        status_message: state.status_message.clone(),
        frame_id: state.frame_id,
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
        if let Ok((loaded_brain, loaded_memory, loaded_semantic)) =
            decode_snapshot(&blob, requested_mode)
        {
            brain = loaded_brain;
            memory = loaded_memory;
            semantic = loaded_semantic;
            last_lsn = snapshot_lsn;
        }
    }
    let journal_records = file.read_journal_after(last_lsn)?;
    replay_mutations(&mut brain, &mut memory, &journal_records);
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

fn replay_mutations(brain: &mut BrainState, memory: &mut MemoryState, records: &[MutationRecord]) {
    let mut last_char: Option<char> = None;
    let mut input_line = String::new();
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
                    if ch == '\n' {
                        if !input_line.trim().is_empty() {
                            let mut sink = Vec::new();
                            memory.observe_text(record.tick, &input_line, &mut sink);
                        }
                        input_line.clear();
                        last_char = None;
                    } else {
                        input_line.push(ch);
                        last_char = Some(ch);
                    }
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
    if !input_line.trim().is_empty() {
        let mut sink = Vec::new();
        memory.observe_text(brain.tick, &input_line, &mut sink);
    }
    brain.last_input_char = last_char;
}

fn seed_brain_from_semantic(
    brain: &mut BrainState,
    semantic: &SemanticState,
    pending_mutations: &mut Vec<MutationRecord>,
) {
    for concept in &semantic.concepts {
        if !concept.canonical.is_empty() {
            brain.queue_user_text(&concept.canonical);
        }
        for ch in concept.canonical.chars() {
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

    refresh_region_roots(file, brain, memory, semantic);
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

fn refresh_region_roots(
    file: &mut BrainFile,
    brain: &BrainState,
    memory: &MemoryState,
    semantic: &SemanticState,
) {
    let assembly_count = brain.assemblies.len() as u64;
    let edge_csr_count = brain.csr.weights.len() as u64;
    let edge_delta_count = brain.delta_edges.len() as u64;
    let episode_count = memory
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Episode)
        .count() as u64;
    let concept_count = semantic.concepts.len() as u64;

    let roots = &mut file.superblock.region_roots;
    roots[1].page_count = assembly_count;
    roots[2].page_count = edge_csr_count;
    roots[3].page_count = edge_delta_count;
    roots[4].page_count = episode_count;
    roots[5].page_count = concept_count;

    if assembly_count > 0 && roots[1].root_page_id == 0 {
        roots[1].root_page_id = FIRST_DATA_PAGE_ID;
    }
    if edge_csr_count > 0 && roots[2].root_page_id == 0 {
        roots[2].root_page_id = FIRST_DATA_PAGE_ID;
    }
    if edge_delta_count > 0 && roots[3].root_page_id == 0 {
        roots[3].root_page_id = FIRST_DATA_PAGE_ID;
    }
    if episode_count > 0 && roots[4].root_page_id == 0 {
        roots[4].root_page_id = FIRST_DATA_PAGE_ID;
    }
    if concept_count > 0 && roots[5].root_page_id == 0 {
        roots[5].root_page_id = FIRST_DATA_PAGE_ID;
    }
}

fn mode_to_u8(mode: RunMode) -> u8 {
    match mode {
        RunMode::Deterministic => 0,
        RunMode::Stochastic => 1,
    }
}

fn mode_from_u8(mode: u8, fallback: RunMode) -> RunMode {
    match mode {
        0 => RunMode::Deterministic,
        1 => RunMode::Stochastic,
        _ => fallback,
    }
}

fn encode_snapshot(brain: &BrainState, memory: &MemoryState, semantic: &SemanticState) -> Vec<u8> {
    let mut writer = BinaryWriter::new();
    writer.put_u32(0x42534E50);
    writer.put_u16(2);
    writer.put_u8(mode_to_u8(brain.mode));
    writer.put_u8(0);
    writer.put_u64(brain.tick);
    writer.put_u32(brain.last_input_char.map(|c| c as u32).unwrap_or(0));

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

    writer.put_u32(memory.nodes.len() as u32);
    for node in &memory.nodes {
        writer.put_u64(node.id);
        writer.put_u8(node.kind as u8);
        writer.put_u8(0);
        writer.put_u16(0);
        writer.put_u64(node.last_tick);
        writer.put_u32(node.recurrence);
        writer.put_f32(node.frequency);
        writer.put_f32(node.salience);
        writer.put_f32(node.temperature);
        writer.put_string(&node.label);
    }

    writer.put_u32(memory.edges.len() as u32);
    for edge in &memory.edges {
        writer.put_u64(edge.from);
        writer.put_u64(edge.to);
        writer.put_u8(edge.kind as u8);
        writer.put_u8(0);
        writer.put_u16(0);
        writer.put_u64(edge.last_tick);
        writer.put_u32(edge.recurrence);
        writer.put_f32(edge.strength);
        writer.put_f32(edge.frequency);
        writer.put_f32(edge.salience);
        writer.put_f32(edge.temperature);
    }

    writer.put_u32(memory.temporal_anchors.len() as u32);
    for anchor in &memory.temporal_anchors {
        writer.put_string(&anchor.cue);
        writer.put_u64(anchor.node_id);
        writer.put_u64(anchor.last_rebind_tick);
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
    match version {
        1 => decode_snapshot_v1(&mut reader, requested_mode),
        2 => decode_snapshot_v2(&mut reader, requested_mode),
        other => Err(AxonError::InvalidFormat(format!(
            "unsupported snapshot version {other}"
        ))),
    }
}

fn decode_snapshot_v2(
    reader: &mut BinaryReader<'_>,
    requested_mode: RunMode,
) -> Result<(BrainState, MemoryState, SemanticState), AxonError> {
    let mode_u8 = reader.get_u8()?;
    let _reserved = reader.get_u8()?;
    let mut brain = BrainState::new(mode_from_u8(mode_u8, requested_mode), DEFAULT_RANDOM_SEED);
    brain.tick = reader.get_u64()?;
    let last_input = reader.get_u32()?;
    brain.last_input_char = if last_input == 0 {
        None
    } else {
        char::from_u32(last_input)
    };

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
        brain.delta_edges.push(DeltaEdge {
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
    let node_count = reader.get_u32()? as usize;
    for _ in 0..node_count {
        let id = reader.get_u64()?;
        let kind = NodeKind::from_u8(reader.get_u8()?).unwrap_or(NodeKind::Concept);
        let _reserved0 = reader.get_u8()?;
        let _reserved1 = reader.get_u16()?;
        let last_tick = reader.get_u64()?;
        let recurrence = reader.get_u32()?;
        let frequency = reader.get_f32()?;
        let salience = reader.get_f32()?;
        let temperature = reader.get_f32()?;
        let label = reader.get_string()?;
        memory.nodes.push(MemoryNode {
            id,
            kind,
            label,
            last_tick,
            recurrence,
            frequency,
            salience,
            temperature,
        });
    }

    let memory_edge_count = reader.get_u32()? as usize;
    for _ in 0..memory_edge_count {
        let from = reader.get_u64()?;
        let to = reader.get_u64()?;
        let kind = EdgeKind::from_u8(reader.get_u8()?).unwrap_or(EdgeKind::CoActivation);
        let _reserved0 = reader.get_u8()?;
        let _reserved1 = reader.get_u16()?;
        let last_tick = reader.get_u64()?;
        let recurrence = reader.get_u32()?;
        let strength = reader.get_f32()?;
        let frequency = reader.get_f32()?;
        let salience = reader.get_f32()?;
        let temperature = reader.get_f32()?;
        memory.edges.push(MemoryEdge {
            from,
            to,
            kind,
            strength,
            last_tick,
            recurrence,
            frequency,
            salience,
            temperature,
        });
    }

    let anchor_count = reader.get_u32()? as usize;
    for _ in 0..anchor_count {
        memory.temporal_anchors.push(TemporalAnchor {
            cue: reader.get_string()?,
            node_id: reader.get_u64()?,
            last_rebind_tick: reader.get_u64()?,
        });
    }
    memory.rebuild_indexes();

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

fn decode_snapshot_v1(
    reader: &mut BinaryReader<'_>,
    requested_mode: RunMode,
) -> Result<(BrainState, MemoryState, SemanticState), AxonError> {
    let mode_u8 = reader.get_u8()?;
    let _reserved = reader.get_u8()?;
    let mut brain = BrainState::new(mode_from_u8(mode_u8, requested_mode), DEFAULT_RANDOM_SEED);
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
        brain.delta_edges.push(DeltaEdge {
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
        let _id = reader.get_u64()?;
        let tick = reader.get_u64()?;
        let salience = reader.get_f32()?;
        let recall_score = reader.get_f32()?;
        let trace = reader.get_string()?;
        memory.ingest_legacy_episode(tick, &trace, salience, recall_score);
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

fn slash_visible(state: &RuntimeShared) -> bool {
    state.input_buffer.starts_with('/') && !state.slash_suggestions.is_empty()
}

fn refresh_slash(state: &mut RuntimeShared) {
    state.slash_suggestions = tui::filter_slash_suggestions(&state.input_buffer, &state.slash_catalog);
    if state.slash_selected >= state.slash_suggestions.len() {
        state.slash_selected = 0;
    }
}

fn autocomplete_slash(state: &mut RuntimeShared) {
    if let Some(suggestion) = state.slash_suggestions.get(state.slash_selected) {
        state.input_buffer = suggestion.command.clone();
        state.input_cursor = char_count(&state.input_buffer);
        refresh_slash(state);
    }
}

fn begin_manual_edit(state: &mut RuntimeShared) {
    if state.history_index.is_some() {
        state.history_index = None;
        state.history_draft.clear();
    }
}

fn history_prev(state: &mut RuntimeShared) {
    if state.history.is_empty() {
        return;
    }
    if state.history_index.is_none() {
        state.history_draft = state.input_buffer.clone();
        state.history_index = Some(state.history.len() - 1);
    } else if let Some(idx) = state.history_index {
        if idx > 0 {
            state.history_index = Some(idx - 1);
        }
    }
    if let Some(idx) = state.history_index {
        state.input_buffer = state.history[idx].clone();
        state.input_cursor = char_count(&state.input_buffer);
    }
}

fn history_next(state: &mut RuntimeShared) {
    let Some(idx) = state.history_index else {
        return;
    };
    if idx + 1 < state.history.len() {
        let next_idx = idx + 1;
        state.history_index = Some(next_idx);
        state.input_buffer = state.history[next_idx].clone();
        state.input_cursor = char_count(&state.input_buffer);
    } else {
        state.history_index = None;
        state.input_buffer = state.history_draft.clone();
        state.input_cursor = char_count(&state.input_buffer);
        state.history_draft.clear();
    }
}

fn push_history(state: &mut RuntimeShared, line: &str) {
    if line.is_empty() {
        return;
    }
    if state.history.last().map(|last| last == line).unwrap_or(false) {
        return;
    }
    state.history.push(line.to_string());
    if state.history.len() > 512 {
        state.history.remove(0);
    }
}

fn insert_char(state: &mut RuntimeShared, ch: char) {
    let idx = byte_index_at_char(&state.input_buffer, state.input_cursor);
    state.input_buffer.insert(idx, ch);
    state.input_cursor += 1;
}

fn insert_text(state: &mut RuntimeShared, text: &str) {
    for ch in text.chars() {
        insert_char(state, ch);
    }
}

fn delete_char_before_cursor(state: &mut RuntimeShared) {
    if state.input_cursor == 0 {
        return;
    }
    let start = byte_index_at_char(&state.input_buffer, state.input_cursor - 1);
    let end = byte_index_at_char(&state.input_buffer, state.input_cursor);
    state.input_buffer.replace_range(start..end, "");
    state.input_cursor -= 1;
}

fn delete_char_at_cursor(state: &mut RuntimeShared) {
    let len = char_count(&state.input_buffer);
    if state.input_cursor >= len {
        return;
    }
    let start = byte_index_at_char(&state.input_buffer, state.input_cursor);
    let end = byte_index_at_char(&state.input_buffer, state.input_cursor + 1);
    state.input_buffer.replace_range(start..end, "");
}

fn char_count(value: &str) -> usize {
    value.chars().count()
}

fn byte_index_at_char(value: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }
    value
        .char_indices()
        .nth(char_idx)
        .map(|(idx, _)| idx)
        .unwrap_or(value.len())
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
        let mut out = [0u8; 2];
        out.copy_from_slice(self.get_slice(2)?);
        Ok(u16::from_le_bytes(out))
    }

    fn get_u32(&mut self) -> Result<u32, AxonError> {
        let mut out = [0u8; 4];
        out.copy_from_slice(self.get_slice(4)?);
        Ok(u32::from_le_bytes(out))
    }

    fn get_u64(&mut self) -> Result<u64, AxonError> {
        let mut out = [0u8; 8];
        out.copy_from_slice(self.get_slice(8)?);
        Ok(u64::from_le_bytes(out))
    }

    fn get_f32(&mut self) -> Result<f32, AxonError> {
        let mut out = [0u8; 4];
        out.copy_from_slice(self.get_slice(4)?);
        Ok(f32::from_le_bytes(out))
    }

    fn get_string(&mut self) -> Result<String, AxonError> {
        let len = self.get_u32()? as usize;
        let data = self.get_slice(len)?;
        Ok(String::from_utf8_lossy(data).into_owned())
    }
}
