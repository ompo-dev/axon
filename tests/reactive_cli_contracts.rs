use axon_uic::reactive::agent::*;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-reactive-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let (Ok(p), Ok(root)) = (self.0.canonicalize(), std::env::temp_dir().canonicalize()) {
            assert!(p.starts_with(root));
            let _ = fs::remove_dir_all(p);
        }
    }
}
struct Session {
    child: Child,
    input: ChildStdin,
    output: mpsc::Receiver<String>,
}
impl Session {
    fn new(extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_axon-uic-reactive-agent"))
            .args(["--context", "cli/test", "--arity", "1"])
            .args(extra)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let out = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else {
                    break;
                };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let s = Self {
            child,
            input,
            output: rx,
        };
        s.wait("\"kind\":\"ready\"");
        s
    }
    fn wait(&self, needle: &str) -> String {
        for _ in 0..200 {
            let line = self
                .output
                .recv_timeout(Duration::from_secs(30))
                .expect("process did not respond with stdin open");
            if line.contains(needle) {
                return line;
            }
        }
        panic!("event missing: {needle}")
    }
    fn event(&mut self, s: &str, needle: &str) -> String {
        writeln!(self.input, "{s}").unwrap();
        self.input.flush().unwrap();
        self.wait(needle)
    }
    fn warm(&mut self) -> u64 {
        let mut value = 0;
        for id in 1..=64 {
            let x = if id <= 8 {
                [0, 1, 2, 0, 1, 1, 0, 2][id - 1]
            } else {
                let probe = self.event("P", "\"kind\":\"probe\"");
                if probe.contains("\"values\":[\"0\"]") {
                    0
                } else if probe.contains("\"values\":[\"1\"]") {
                    1
                } else {
                    assert!(probe.contains("\"values\":[\"2\"]"));
                    2
                }
            };
            value = if x > 0 { value + 1 } else { 0 };
            self.event(&format!("O\t{id}\t{x}"), "\"kind\":\"prediction\"");
            let result = self.event(&format!("F\t{id}\t{value}"), "\"kind\":\"feedback\"");
            if result.contains("\"status\":\"promoted\"") {
                return id as u64;
            }
        }
        panic!("process never admitted")
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
#[test]
fn actual_process_creates_a_stateful_procedure_before_eof_and_recovers_after_kill() {
    let temp = Temp::new();
    let checkpoint = temp.0.join("state.axra");
    let mut first = Session::new(&[]);
    let id = first.warm() + 1;
    let p = first.event(&format!("O\t{id}\t0"), "\"kind\":\"prediction\"");
    assert!(p.contains("\"value\":\"0\""));
    first.event(
        &format!("C\t{}", checkpoint.display()),
        "\"kind\":\"error\"",
    );
    first.event(&format!("F\t{id}\t0"), "\"kind\":\"feedback\"");
    first.event(
        &format!("C\t{}", checkpoint.display()),
        "\"kind\":\"checkpoint\"",
    );
    assert!(first.child.try_wait().unwrap().is_none());
    drop(first);
    let bytes = fs::read(&checkpoint).unwrap();
    let model = ReactiveAgent::resume("cli/test", &checkpoint).unwrap();
    assert_eq!(model.statistics().observations, id);
    let mut second = Session::new(&["--restore", checkpoint.to_str().unwrap()]);
    let p = second.event(&format!("O\t{}\t1", id + 1), "\"kind\":\"prediction\"");
    assert!(p.contains("\"value\":\"1\""));
    second.event(&format!("F\t{}\t1", id + 1), "\"kind\":\"feedback\"");
    second.event(
        &format!("C\t{}", checkpoint.display()),
        "\"kind\":\"error\"",
    );
    assert_eq!(fs::read(&checkpoint).unwrap(), bytes);
    second.event("Q", "\"kind\":\"stopped\"");
    assert!(second.child.wait().unwrap().success());
}
#[test]
fn invalid_live_events_do_not_create_fake_observations_or_evidence() {
    let mut s = Session::new(&[]);
    for event in [
        "O\t1\tNaN",
        "O\t1\t1,2",
        "O\t1\t9223372036854775808",
        "F\t1\t2",
        "P\t2:1",
        "P\t1:2,0:2",
        "P\t9223372036854775808:0",
        "P\tNaN",
        "unknown",
    ] {
        s.event(event, "\"kind\":\"error\"");
    }
    let report = s.event("S", "\"kind\":\"status\"");
    assert!(report.contains("\"observations\":0"));
    assert!(report.contains("\"promotions\":0"));
    let proposal = s.event("P\t-32:32", "\"kind\":\"probe\"");
    assert!(proposal.contains("\"values\":null"));
    s.event("Q", "\"kind\":\"stopped\"");
}
#[test]
fn complete_lab_emits_six_reopened_procedures_including_shared_acquired_state() {
    let temp = Temp::new();
    let output = temp.0.join("lab");
    let result = Command::new(env!("CARGO_BIN_EXE_axon-uic-reactive-lab"))
        .arg(&output)
        .args(["37691853", "1000"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for case in 0..6 {
        let a = ReactiveAgent::resume(
            &format!("reactive-lab/{case}"),
            &output.join(format!("case-{case}.agent.axra")),
        )
        .unwrap();
        assert_eq!(a.statistics().promotions, 1);
        assert!(a.statistics().observations <= 64);
    }
    let results = fs::read_to_string(output.join("results.csv")).unwrap();
    assert_eq!(results.lines().count(), 7);
    let summary = fs::read_to_string(output.join("summary.txt")).unwrap();
    assert!(summary.contains("heldout_events=6000"));
    assert!(summary.contains("continuation_events=6000"));
    assert!(summary.contains("errors=0"));
}
