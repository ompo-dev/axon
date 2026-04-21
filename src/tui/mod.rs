use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::sync::mpsc::Sender;
use std::thread;

use crate::cli::RunMode;
use crate::gpu::GpuBackend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UiMode {
    Chat,
    Observatory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyCode {
    Char(char),
    Enter,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    F1,
    F2,
    F5,
    F6,
    Tab,
    Esc,
    CtrlC,
    Unknown,
}

#[derive(Clone, Debug)]
pub enum TuiCommand {
    SwitchMode(UiMode),
    ForceFlush,
    ForceCheckpoint,
    Quit,
    SetRunMode(RunMode),
    Correction { wrong: String, correct: String },
    Help,
}

#[derive(Clone, Debug)]
pub enum InputEvent {
    Key(KeyCode),
    Paste(String),
    Resize { cols: u16, rows: u16 },
    Submit(String),
    Command(TuiCommand),
    Quit,
}

#[derive(Clone, Debug)]
pub struct SlashSuggestion {
    pub command: String,
    pub description: String,
}

#[derive(Clone, Debug)]
pub struct AssemblyView {
    pub id: u32,
    pub symbol: Option<char>,
    pub activation: f32,
    pub stability: f32,
}

#[derive(Clone, Debug)]
pub struct ConceptView {
    pub id: u32,
    pub lemma: String,
    pub stability: f32,
    pub connectivity: f32,
}

#[derive(Clone, Debug)]
pub struct RenderSnapshot {
    pub mode: UiMode,
    pub brain_path: String,
    pub tick: u64,
    pub run_mode: RunMode,
    pub use_gpu: bool,
    pub pending_mutations: usize,
    pub pending_input: usize,
    pub latency_us: u64,
    pub spawn_count: u64,
    pub merge_count: u64,
    pub prune_count: u64,
    pub transcript: Vec<String>,
    pub top_assemblies: Vec<AssemblyView>,
    pub top_concepts: Vec<ConceptView>,
    pub input_buffer: String,
    pub input_cursor: usize,
    pub slash_suggestions: Vec<SlashSuggestion>,
    pub slash_selected: usize,
    pub status_message: String,
    pub frame_id: u64,
}

#[derive(Clone, Debug)]
pub enum PatchOp {
    MoveCursor { row: usize, col: usize },
    WriteText(String),
    ClearLine,
    SetStyle(&'static str),
    HideCursor,
    ShowCursor,
}

pub struct TerminalRenderer {
    prev_lines: Vec<String>,
    entered_alt: bool,
    sync_supported: bool,
    tty: bool,
}

impl TerminalRenderer {
    pub fn new() -> Self {
        Self {
            prev_lines: Vec::new(),
            entered_alt: false,
            sync_supported: is_sync_output_supported(),
            tty: std::io::stdout().is_terminal(),
        }
    }

    pub fn render_frame(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu: &GpuBackend,
    ) -> Result<(), io::Error> {
        if !self.entered_alt {
            self.enter_alt_screen()?;
        }
        let (width, height) = terminal_size();
        let lines = build_lines(snapshot, gpu, width as usize, height as usize);
        let mut ops = Vec::new();
        if !self.entered_alt {
            ops.push(PatchOp::HideCursor);
        }
        let max = self.prev_lines.len().max(lines.len());
        for row in 0..max {
            let old = self.prev_lines.get(row);
            let new = lines.get(row);
            if old == new {
                continue;
            }
            ops.push(PatchOp::MoveCursor { row, col: 0 });
            ops.push(PatchOp::ClearLine);
            if let Some(content) = new {
                ops.push(PatchOp::WriteText(content.clone()));
            }
        }
        let input_row = lines.len().saturating_sub(1);
        let cursor_col = 2usize.saturating_add(snapshot.input_cursor);
        ops.push(PatchOp::MoveCursor {
            row: input_row,
            col: cursor_col,
        });
        ops.push(PatchOp::ShowCursor);
        self.apply_patch_ops(&ops)?;
        self.prev_lines = lines;
        self.entered_alt = true;
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<(), io::Error> {
        if !self.entered_alt {
            return Ok(());
        }
        let mut out = io::stdout();
        if self.sync_supported {
            out.write_all(b"\x1b[?2026h")?;
        }
        out.write_all(b"\x1b[?25h")?;
        out.write_all(b"\x1b[?1049l")?;
        if self.sync_supported {
            out.write_all(b"\x1b[?2026l")?;
        }
        out.flush()?;
        self.entered_alt = false;
        Ok(())
    }

    fn enter_alt_screen(&mut self) -> Result<(), io::Error> {
        if !self.tty {
            return Ok(());
        }
        let mut out = io::stdout();
        if self.sync_supported {
            out.write_all(b"\x1b[?2026h")?;
        }
        out.write_all(b"\x1b[?1049h")?;
        out.write_all(b"\x1b[2J\x1b[H")?;
        out.write_all(b"\x1b[?25l")?;
        if self.sync_supported {
            out.write_all(b"\x1b[?2026l")?;
        }
        out.flush()?;
        Ok(())
    }

    fn apply_patch_ops(&self, ops: &[PatchOp]) -> Result<(), io::Error> {
        let mut out = io::stdout();
        if self.sync_supported {
            out.write_all(b"\x1b[?2026h")?;
        }
        for op in ops {
            match op {
                PatchOp::MoveCursor { row, col } => {
                    write!(out, "\x1b[{};{}H", row + 1, col + 1)?;
                }
                PatchOp::WriteText(text) => {
                    out.write_all(text.as_bytes())?;
                }
                PatchOp::ClearLine => {
                    out.write_all(b"\x1b[2K")?;
                }
                PatchOp::SetStyle(style) => {
                    out.write_all(style.as_bytes())?;
                }
                PatchOp::HideCursor => out.write_all(b"\x1b[?25l")?,
                PatchOp::ShowCursor => out.write_all(b"\x1b[?25h")?,
            }
        }
        if self.sync_supported {
            out.write_all(b"\x1b[?2026l")?;
        }
        out.flush()?;
        Ok(())
    }
}

pub fn spawn_input_thread(tx: Sender<InputEvent>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut mode = TerminalModeGuard::new();
        if mode.enable_raw().is_ok() {
            let mut stdin = io::stdin();
            let mut first = [0u8; 1];
            let _ = tx.send(InputEvent::Resize {
                cols: terminal_size().0,
                rows: terminal_size().1,
            });
            loop {
                match stdin.read(&mut first) {
                    Ok(0) => continue,
                    Ok(_) => {
                        if let Some(event) = decode_key_event(first[0], &mut stdin) {
                            match event {
                                KeyCode::CtrlC => {
                                    let _ = tx.send(InputEvent::Quit);
                                    break;
                                }
                                _ => {
                                    let _ = tx.send(InputEvent::Key(event));
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = tx.send(InputEvent::Quit);
                        break;
                    }
                }
            }
        } else {
            // Fallback para terminais sem suporte raw.
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            let mut line = String::new();
            loop {
                line.clear();
                if reader.read_line(&mut line).is_err() {
                    let _ = tx.send(InputEvent::Quit);
                    break;
                }
                let text = line.trim_end_matches(['\r', '\n']).to_string();
                if text.is_empty() {
                    continue;
                }
                if text == "/quit" || text == "/exit" {
                    let _ = tx.send(InputEvent::Quit);
                    break;
                }
                let _ = tx.send(InputEvent::Submit(text));
            }
        }
    })
}

pub fn default_slash_catalog() -> Vec<SlashSuggestion> {
    vec![
        SlashSuggestion {
            command: "/f1".to_string(),
            description: "mudar para modo Chat".to_string(),
        },
        SlashSuggestion {
            command: "/f2".to_string(),
            description: "mudar para modo Observatorio".to_string(),
        },
        SlashSuggestion {
            command: "/f5".to_string(),
            description: "forcar flush de journal".to_string(),
        },
        SlashSuggestion {
            command: "/f6".to_string(),
            description: "forcar checkpoint".to_string(),
        },
        SlashSuggestion {
            command: "/mode deterministic".to_string(),
            description: "modo deterministico".to_string(),
        },
        SlashSuggestion {
            command: "/mode stochastic".to_string(),
            description: "modo estocastico".to_string(),
        },
        SlashSuggestion {
            command: "/corrigir errado|certo".to_string(),
            description: "corrigir associacao enviesada".to_string(),
        },
        SlashSuggestion {
            command: "/help".to_string(),
            description: "mostrar ajuda".to_string(),
        },
        SlashSuggestion {
            command: "/quit".to_string(),
            description: "encerrar TUI".to_string(),
        },
    ]
}

pub fn filter_slash_suggestions(input: &str, catalog: &[SlashSuggestion]) -> Vec<SlashSuggestion> {
    if !input.starts_with('/') {
        return Vec::new();
    }
    let prefix = input.trim().to_lowercase();
    let mut out: Vec<SlashSuggestion> = catalog
        .iter()
        .filter(|item| item.command.starts_with(&prefix))
        .cloned()
        .collect();
    if out.is_empty() {
        out = catalog.to_vec();
    }
    out
}

fn build_lines(
    snapshot: &RenderSnapshot,
    gpu: &GpuBackend,
    width: usize,
    height: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(trim_to_width(
        &format!(
            "AXON v3.1 | mode={:?} | run={:?} | tick={} | backend={} | frame={}",
            snapshot.mode, snapshot.run_mode, snapshot.tick, gpu, snapshot.frame_id
        ),
        width,
    ));
    lines.push(trim_to_width(
        &format!(
            "brain={} | pending_mutations={} | input_queue={} | tick_latency={}us",
            snapshot.brain_path, snapshot.pending_mutations, snapshot.pending_input, snapshot.latency_us
        ),
        width,
    ));
    lines.push(trim_to_width(
        &format!(
            "spawn={} merge={} prune={} | status: {}",
            snapshot.spawn_count, snapshot.merge_count, snapshot.prune_count, snapshot.status_message
        ),
        width,
    ));
    lines.push("-".repeat(width.min(80)));

    match snapshot.mode {
        UiMode::Chat => {
            let reserved = 8usize
                .saturating_add(snapshot.slash_suggestions.len().min(6))
                .saturating_add(1);
            let available = height.saturating_sub(reserved).max(4);
            for line in snapshot.transcript.iter().rev().take(available).rev() {
                lines.push(trim_to_width(line, width));
            }
        }
        UiMode::Observatory => {
            lines.push("Top assemblies:".to_string());
            for asm in &snapshot.top_assemblies {
                let symbol = asm.symbol.unwrap_or(' ');
                lines.push(trim_to_width(
                    &format!(
                        "  #{:04} [{}] act={:+.3} stability={:.3}",
                        asm.id, symbol, asm.activation, asm.stability
                    ),
                    width,
                ));
            }
            lines.push("Top concepts:".to_string());
            for concept in &snapshot.top_concepts {
                lines.push(trim_to_width(
                    &format!(
                        "  #{:04} {} stability={:.3} conn={:.3}",
                        concept.id, concept.lemma, concept.stability, concept.connectivity
                    ),
                    width,
                ));
            }
        }
    }

    if !snapshot.slash_suggestions.is_empty() {
        lines.push("-".repeat(width.min(80)));
        lines.push("Slash suggestions:".to_string());
        for (idx, item) in snapshot.slash_suggestions.iter().take(6).enumerate() {
            let marker = if idx == snapshot.slash_selected { '>' } else { ' ' };
            lines.push(trim_to_width(
                &format!("{marker} {} - {}", item.command, item.description),
                width,
            ));
        }
    }

    lines.push("-".repeat(width.min(80)));
    lines.push(trim_to_width(
        &format!("> {}", snapshot.input_buffer),
        width,
    ));

    while lines.len() > height {
        if lines.len() > 2 {
            lines.remove(4.min(lines.len() - 1));
        } else {
            break;
        }
    }
    lines
}

fn trim_to_width(input: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    for ch in input.chars().take(width) {
        out.push(ch);
    }
    out
}

fn decode_key_event(first: u8, reader: &mut dyn Read) -> Option<KeyCode> {
    match first {
        0x03 => Some(KeyCode::CtrlC),
        b'\r' | b'\n' => Some(KeyCode::Enter),
        0x7F | 0x08 => Some(KeyCode::Backspace),
        0x09 => Some(KeyCode::Tab),
        0x1B => decode_escape_sequence(reader),
        byte if byte.is_ascii() => {
            if byte.is_ascii_control() {
                Some(KeyCode::Unknown)
            } else {
                Some(KeyCode::Char(byte as char))
            }
        }
        byte => decode_utf8(byte, reader).map(KeyCode::Char).or(Some(KeyCode::Unknown)),
    }
}

fn decode_escape_sequence(reader: &mut dyn Read) -> Option<KeyCode> {
    let mut second = [0u8; 1];
    if reader.read(&mut second).ok()? == 0 {
        return Some(KeyCode::Esc);
    }
    if second[0] == b'O' {
        let mut third = [0u8; 1];
        if reader.read(&mut third).ok()? == 0 {
            return Some(KeyCode::Esc);
        }
        return match third[0] {
            b'P' => Some(KeyCode::F1),
            b'Q' => Some(KeyCode::F2),
            _ => Some(KeyCode::Unknown),
        };
    }
    if second[0] != b'[' {
        return Some(KeyCode::Esc);
    }
    let mut third = [0u8; 1];
    if reader.read(&mut third).ok()? == 0 {
        return Some(KeyCode::Esc);
    }
    match third[0] {
        b'A' => Some(KeyCode::Up),
        b'B' => Some(KeyCode::Down),
        b'C' => Some(KeyCode::Right),
        b'D' => Some(KeyCode::Left),
        b'1' | b'3' => {
            let mut fourth = [0u8; 1];
            if reader.read(&mut fourth).ok()? == 0 {
                return Some(KeyCode::Unknown);
            }
            match (third[0], fourth[0]) {
                (b'3', b'~') => Some(KeyCode::Delete),
                (b'1', b'5') => {
                    let mut tilde = [0u8; 1];
                    if reader.read(&mut tilde).ok()? == 0 {
                        return Some(KeyCode::Unknown);
                    }
                    if tilde[0] == b'~' {
                        Some(KeyCode::F5)
                    } else {
                        Some(KeyCode::Unknown)
                    }
                }
                (b'1', b'7') => {
                    let mut tilde = [0u8; 1];
                    if reader.read(&mut tilde).ok()? == 0 {
                        return Some(KeyCode::Unknown);
                    }
                    if tilde[0] == b'~' {
                        Some(KeyCode::F6)
                    } else {
                        Some(KeyCode::Unknown)
                    }
                }
                _ => Some(KeyCode::Unknown),
            }
        }
        _ => Some(KeyCode::Unknown),
    }
}

fn decode_utf8(first: u8, reader: &mut dyn Read) -> Option<char> {
    let needed = if first & 0b1110_0000 == 0b1100_0000 {
        2usize
    } else if first & 0b1111_0000 == 0b1110_0000 {
        3usize
    } else if first & 0b1111_1000 == 0b1111_0000 {
        4usize
    } else {
        1usize
    };
    let mut bytes = vec![first];
    if needed > 1 {
        let mut rest = vec![0u8; needed - 1];
        if reader.read_exact(&mut rest).is_err() {
            return None;
        }
        bytes.extend_from_slice(&rest);
    }
    std::str::from_utf8(&bytes).ok()?.chars().next()
}

fn is_sync_output_supported() -> bool {
    if std::env::var_os("TMUX").is_some() {
        return false;
    }
    if std::env::var_os("WT_SESSION").is_some() {
        return true;
    }
    if let Ok(term_program) = std::env::var("TERM_PROGRAM") {
        let lower = term_program.to_lowercase();
        return lower.contains("iterm")
            || lower.contains("wezterm")
            || lower.contains("ghostty")
            || lower.contains("vscode");
    }
    false
}

fn terminal_size() -> (u16, u16) {
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(120);
    let rows = std::env::var("LINES")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(40);
    (cols, rows)
}

struct TerminalModeGuard {
    #[cfg(target_os = "windows")]
    input_handle: *mut core::ffi::c_void,
    #[cfg(target_os = "windows")]
    output_handle: *mut core::ffi::c_void,
    #[cfg(target_os = "windows")]
    original_input_mode: u32,
    #[cfg(target_os = "windows")]
    original_output_mode: u32,
    raw_enabled: bool,
}

impl TerminalModeGuard {
    fn new() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            input_handle: std::ptr::null_mut(),
            #[cfg(target_os = "windows")]
            output_handle: std::ptr::null_mut(),
            #[cfg(target_os = "windows")]
            original_input_mode: 0,
            #[cfg(target_os = "windows")]
            original_output_mode: 0,
            raw_enabled: false,
        }
    }

    fn enable_raw(&mut self) -> Result<(), io::Error> {
        #[cfg(target_os = "windows")]
        {
            unsafe {
                type Handle = *mut core::ffi::c_void;
                const STD_INPUT_HANDLE: i32 = -10;
                const STD_OUTPUT_HANDLE: i32 = -11;
                const ENABLE_ECHO_INPUT: u32 = 0x0004;
                const ENABLE_LINE_INPUT: u32 = 0x0002;
                const ENABLE_PROCESSED_INPUT: u32 = 0x0001;
                const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;
                const ENABLE_WINDOW_INPUT: u32 = 0x0008;
                const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

                unsafe extern "system" {
                    fn GetStdHandle(nStdHandle: i32) -> Handle;
                    fn GetConsoleMode(hConsoleHandle: Handle, lpMode: *mut u32) -> i32;
                    fn SetConsoleMode(hConsoleHandle: Handle, dwMode: u32) -> i32;
                }

                let input = GetStdHandle(STD_INPUT_HANDLE);
                let output = GetStdHandle(STD_OUTPUT_HANDLE);
                if input.is_null() || output.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let mut input_mode = 0u32;
                if GetConsoleMode(input, &mut input_mode as *mut u32) == 0 {
                    return Err(io::Error::last_os_error());
                }
                let mut output_mode = 0u32;
                if GetConsoleMode(output, &mut output_mode as *mut u32) == 0 {
                    return Err(io::Error::last_os_error());
                }
                self.input_handle = input;
                self.output_handle = output;
                self.original_input_mode = input_mode;
                self.original_output_mode = output_mode;

                let new_input = (input_mode
                    & !(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT))
                    | ENABLE_VIRTUAL_TERMINAL_INPUT
                    | ENABLE_WINDOW_INPUT;
                if SetConsoleMode(input, new_input) == 0 {
                    return Err(io::Error::last_os_error());
                }
                let new_output = output_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                let _ = SetConsoleMode(output, new_output);
                self.raw_enabled = true;
                return Ok(());
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            self.raw_enabled = false;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "raw mode not implemented on this platform yet",
            ))
        }
    }
}

impl Drop for TerminalModeGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            if !self.input_handle.is_null() {
                unsafe extern "system" {
                    fn SetConsoleMode(
                        hConsoleHandle: *mut core::ffi::c_void,
                        dwMode: u32,
                    ) -> i32;
                }
                let _ = SetConsoleMode(self.input_handle, self.original_input_mode);
            }
            if !self.output_handle.is_null() {
                unsafe extern "system" {
                    fn SetConsoleMode(
                        hConsoleHandle: *mut core::ffi::c_void,
                        dwMode: u32,
                    ) -> i32;
                }
                let _ = SetConsoleMode(self.output_handle, self.original_output_mode);
            }
        }
    }
}

pub fn parse_inline_command(line: &str) -> Option<TuiCommand> {
    let trimmed = line.trim();
    match trimmed {
        "/f1" => Some(TuiCommand::SwitchMode(UiMode::Chat)),
        "/f2" => Some(TuiCommand::SwitchMode(UiMode::Observatory)),
        "/f5" => Some(TuiCommand::ForceFlush),
        "/f6" => Some(TuiCommand::ForceCheckpoint),
        "/quit" | "/exit" => Some(TuiCommand::Quit),
        "/help" => Some(TuiCommand::Help),
        "/mode deterministic" => Some(TuiCommand::SetRunMode(RunMode::Deterministic)),
        "/mode stochastic" => Some(TuiCommand::SetRunMode(RunMode::Stochastic)),
        _ => parse_correction_command(trimmed),
    }
}

fn parse_correction_command(line: &str) -> Option<TuiCommand> {
    if !line.starts_with("/corrigir ") {
        return None;
    }
    let payload = line.trim_start_matches("/corrigir ").trim();
    let splitter = if payload.contains('|') {
        '|'
    } else if payload.contains("->") {
        '\0'
    } else {
        '|'
    };
    let parts: Vec<String> = if splitter == '\0' {
        payload
            .splitn(2, "->")
            .map(|s| s.trim().to_string())
            .collect()
    } else {
        payload
            .splitn(2, splitter)
            .map(|s| s.trim().to_string())
            .collect()
    };
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return None;
    }
    Some(TuiCommand::Correction {
        wrong: parts[0].clone(),
        correct: parts[1].clone(),
    })
}
