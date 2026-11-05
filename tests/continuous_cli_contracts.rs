use axon_uic::{EpisodeStore, numeric::continuous::*};
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
            "axon-continuous-{}-{}",
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
        let mut child = Command::new(env!("CARGO_BIN_EXE_axon-uic-live-agent"))
            .args(["--context", "live/test", "--arity", "1"])
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
        s.wait_for("\"kind\":\"ready\"");
        s
    }
    fn send(&mut self, event: &str) {
        writeln!(self.input, "{event}").unwrap();
        self.input.flush().unwrap();
    }
    fn wait_for(&self, needle: &str) -> String {
        for _ in 0..1000 {
            let line = self
                .output
                .recv_timeout(Duration::from_secs(20))
                .expect("live process failed to answer while stdin remained open");
            if line.contains(needle) {
                return line;
            }
        }
        panic!("event not found: {needle}")
    }
    fn event(&mut self, event: &str, needle: &str) -> String {
        self.send(event);
        self.wait_for(needle)
    }
    fn warm(&mut self) {
        for id in 1..=8 {
            self.event(&format!("O\t{id}\t{id}"), "\"kind\":\"prediction\"");
            self.event(&format!("F\t{id}\t{}", 2 * id + 1), "\"kind\":\"feedback\"");
        }
        self.wait_for("\"status\":\"candidate\"");
        for id in 9..=14 {
            self.event(&format!("O\t{id}\t{id}"), "\"kind\":\"prediction\"");
            self.event(&format!("F\t{id}\t{}", 2 * id + 1), "\"kind\":\"feedback\"");
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
#[test]
fn the_real_process_learns_before_eof_and_resumes_delayed_feedback_after_a_kill() {
    let temp = Temp::new();
    let checkpoint = temp.0.join("resume.axcont");
    let journal = temp.0.join("milestones.axep");
    let mut first = Session::new(&["--episodes", journal.to_str().unwrap()]);
    first.warm();
    let prediction = first.event("O\t15\t25", "\"kind\":\"prediction\"");
    assert!(prediction.contains("\"value\":51"));
    let status = first.event("S", "\"kind\":\"status\"");
    assert!(status.contains("\"phase\":\"active\""));
    first.event(
        &format!("C\t{}", checkpoint.display()),
        "\"kind\":\"checkpoint\"",
    );
    assert!(first.child.try_wait().unwrap().is_none());
    drop(first);
    let bytes = fs::read(&checkpoint).unwrap();
    assert_eq!(
        ContinuousAgent::from_bytes(&bytes).unwrap().pending_count(),
        1
    );
    let mut second = Session::new(&[
        "--restore",
        checkpoint.to_str().unwrap(),
        "--episodes",
        journal.to_str().unwrap(),
    ]);
    let receipt = second.event("F\t15\t51", "\"kind\":\"feedback\"");
    assert!(receipt.contains("\"correct\":true"));
    let duplicate = second.event("F\t15\t51", "\"kind\":\"feedback\"");
    assert!(duplicate.contains("\"status\":\"duplicate\""));
    second.event(
        &format!("C\t{}", checkpoint.display()),
        "\"kind\":\"error\"",
    );
    assert_eq!(fs::read(&checkpoint).unwrap(), bytes);
    second.send("Q");
    second.wait_for("\"kind\":\"stopped\"");
    assert!(second.child.wait().unwrap().success());
    drop(second);
    let (journal, _) = EpisodeStore::open(&journal).unwrap();
    assert!(
        journal
            .traces("live/test", "numeric.continuous/v1")
            .unwrap()
            .len()
            >= 2
    );
}
#[test]
fn live_input_errors_are_explicit_and_do_not_create_synthetic_predictions() {
    let mut session = Session::new(&[]);
    for event in ["O\t1\tNaN", "O\t1\t1,2", "F\t8\t3", "invalid", "O\t1\tinf"] {
        session.event(event, "\"kind\":\"error\"");
    }
    let status = session.event("S", "\"kind\":\"status\"");
    assert!(status.contains("\"observations\":0"));
    assert!(status.contains("\"promotions\":0"));
    session.send("Q");
    session.wait_for("\"kind\":\"stopped\"");
}

#[test]
fn the_live_process_learns_from_raw_readings_without_explicit_labels_or_eof() {
    let mut session = Session::new(&[]);
    for id in 1..=9 {
        session.event(&format!("T\t{id}\t{id}"), "\"kind\":\"frame\"");
    }
    session.wait_for("\"status\":\"candidate\"");
    for id in 10..=16 {
        session.event(&format!("T\t{id}\t{id}"), "\"kind\":\"frame\"");
    }
    let next = session.event("T\t17\t17", "\"kind\":\"frame\"");
    assert!(next.contains("\"next_prediction\":18"));
    assert!(next.contains("\"previous_correct\":true"));
    let status = session.event("S", "\"kind\":\"status\"");
    assert!(status.contains("\"observations\":17"));
    assert!(status.contains("\"feedback\":16"));
    assert!(status.contains("\"phase\":\"active\""));
    assert!(session.child.try_wait().unwrap().is_none());
    session.send("Q");
    session.wait_for("\"kind\":\"stopped\"");
}

#[test]
fn continuous_lab_recovers_all_three_streams_and_reuses_models() {
    let temp = Temp::new();
    let output = temp.0.join("campaign");
    let run = Command::new(env!("CARGO_BIN_EXE_axon-uic-continuous-lab"))
        .arg(&output)
        .args(["58419327", "1000"])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    for channel in 0..3 {
        for stage in ["midpoint", "final"] {
            let bytes = fs::read(output.join(format!("channel-{channel}.{stage}.axcont"))).unwrap();
            let agent = ContinuousAgent::from_bytes(&bytes).unwrap();
            let stats = agent.statistics();
            assert_eq!(
                stats.observations,
                if stage == "midpoint" { 500 } else { 1000 }
            );
            assert!(stats.promotions >= 3 && stats.reuses > 0);
            assert!(stats.drifts > 0 && stats.correct > 0);
            assert!(stats.correct < stats.scored && stats.scored < stats.feedback);
            assert!(agent.retained_bytes() < 65536);
            if channel == 2 {
                assert_eq!(stats.feedback, stats.observations - 1);
                assert_eq!(agent.pending_count(), 1);
            }
        }
        assert!(
            output
                .join(format!("channel-{channel}.active.axnum"))
                .exists()
        );
    }
    for name in [
        "results.csv",
        "transitions.csv",
        "events.csv",
        "summary.txt",
    ] {
        assert!(fs::metadata(output.join(name)).unwrap().len() > 0);
    }
}
