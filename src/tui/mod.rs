use std::io::{self, BufRead, Write};
use std::sync::mpsc::Sender;
use std::thread;

use crate::gpu::GpuBackend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UiMode {
    Chat,
    Observatory,
}

#[derive(Clone, Debug)]
pub enum InputEvent {
    Text(String),
    SwitchMode(UiMode),
    ForceFlush,
    ForceCheckpoint,
    Quit,
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
}

pub fn spawn_input_thread(tx: Sender<InputEvent>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        print_input_help();
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).is_err() {
                let _ = tx.send(InputEvent::Quit);
                break;
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                continue;
            }
            match trimmed {
                "/f1" => {
                    let _ = tx.send(InputEvent::SwitchMode(UiMode::Chat));
                }
                "/f2" => {
                    let _ = tx.send(InputEvent::SwitchMode(UiMode::Observatory));
                }
                "/f5" => {
                    let _ = tx.send(InputEvent::ForceFlush);
                }
                "/f6" => {
                    let _ = tx.send(InputEvent::ForceCheckpoint);
                }
                "/quit" | "/exit" => {
                    let _ = tx.send(InputEvent::Quit);
                    break;
                }
                _ => {
                    let _ = tx.send(InputEvent::Text(trimmed.to_string()));
                }
            }
        }
    })
}

pub fn render_frame(snapshot: &RenderSnapshot, gpu: &GpuBackend) {
    print!("\x1B[2J\x1B[H");
    let _ = io::stdout().flush();
    println!(
        "AXON TUI | mode={:?} | tick={} | backend={} | pending_mutations={} | input_queue={} | tick_latency={}us",
        snapshot.mode,
        snapshot.tick,
        if snapshot.use_gpu { gpu.to_string() } else { "CPU fallback".to_string() },
        snapshot.pending_mutations,
        snapshot.pending_input,
        snapshot.latency_us
    );
    println!(
        "brain={} | spawn={} merge={} prune={}",
        snapshot.brain_path, snapshot.spawn_count, snapshot.merge_count, snapshot.prune_count
    );
    println!("commands: /f1 /f2 /f5 /f6 /quit");
    println!();
    match snapshot.mode {
        UiMode::Chat => {
            println!("--- Chat ---");
            for line in snapshot.transcript.iter().rev().take(20).rev() {
                println!("{line}");
            }
        }
        UiMode::Observatory => {
            println!("--- Observatory ---");
            println!("top assemblies:");
            for asm in &snapshot.top_assemblies {
                let symbol = asm.symbol.unwrap_or(' ');
                println!(
                    "  #{:04} [{}] act={:+.3} stability={:.3}",
                    asm.id, symbol, asm.activation, asm.stability
                );
            }
            println!();
            println!("top concepts:");
            for concept in &snapshot.top_concepts {
                println!(
                    "  #{:04} {} stability={:.3} conn={:.3}",
                    concept.id, concept.lemma, concept.stability, concept.connectivity
                );
            }
        }
    }
    let _ = io::stdout().flush();
}

fn print_input_help() {
    println!("AXON input ready. type text and press enter.");
    println!("shortcuts: /f1 chat, /f2 observatory, /f5 flush, /f6 checkpoint, /quit");
}
