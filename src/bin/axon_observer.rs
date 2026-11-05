//! Local, instrumented adapter. Existing learners remain unchanged.
use axon_uic::{
    Authority, Effect,
    behavior::{
        BehaviorBudget, BehaviorError, BehaviorLearner, BehaviorModel, BehaviorTracker, Reply,
        Resettable, SequenceOracle,
    },
    behavior_lab::{CachedOracle, Rng, audit_machine},
    factor::{
        FactorModel, FactorTracker, FrameBudget, FrameCache, FrameOracle, FrameReply,
        FrameResettable, PlanBudget,
    },
    factor_lab,
    transfer::{TransferBudget, TransferLearner, conformance_probe},
    transfer_lab::{Environment, Family, Rewired},
};
use std::{
    cell::RefCell,
    error::Error,
    io::{self, BufRead, Write},
    path::Path,
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn string(value: &str) -> String {
    let mut result = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c <= '\u{1f}' => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}
fn emit(event: &str, fields: &[(&str, String)]) {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    static CLOCK: OnceLock<Instant> = OnceLock::new();
    let mut line = format!("{{\"event\":{}}}", string(event));
    line.pop();
    line.push_str(&format!(
        ",\"coreSeq\":{},\"coreNs\":{},\"unixMs\":{}",
        SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1,
        CLOCK.get_or_init(Instant::now).elapsed().as_nanos(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    for (key, value) in fields {
        line.push_str(&format!(",{}:{value}", string(key)));
    }
    line.push('}');
    println!("{line}");
    let _ = io::stdout().flush();
}
fn stage(method: &str, detail: &str) {
    emit(
        "stage",
        &[("method", string(method)), ("detail", string(detail))],
    );
}
fn graph(models: &[&BehaviorModel], phase: &str) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut tables = Vec::new();
    for (factor, model) in models.iter().enumerate() {
        let mut rows = Vec::new();
        for state in 0..model.states() {
            let source = format!("{}:s{state}", model.context());
            nodes.push(format!(
                "{{\"id\":{},\"factor\":{factor},\"state\":{state},\"context\":{}}}",
                string(&source),
                string(model.context())
            ));
            for action in 0..model.actions() {
                let edge = model.edge(state as u16, action).expect("validated");
                edges.push(format!("{{\"source\":{},\"target\":{},\"action\":{action},\"output\":{},\"factor\":{factor}}}",string(&source),string(&format!("{}:s{}",model.context(),edge.next)),edge.output));
                rows.push(format!(
                    "s{state:03} a{action:02} -> s{:03} / {:05}",
                    edge.next, edge.output
                ));
            }
        }
        tables.push(format!(
            "{{\"name\":\"f{factor}\",\"states\":{},\"bytes\":{},\"context\":{},\"rows\":[{}]}}",
            model.states(),
            model.encoded_bytes(),
            string(model.context()),
            rows.iter().map(|r| string(r)).collect::<Vec<_>>().join(",")
        ));
    }
    emit(
        "graph",
        &[
            ("nodes", format!("[{}]", nodes.join(","))),
            ("edges", format!("[{}]", edges.join(","))),
            ("models", format!("[{}]", tables.join(","))),
            ("phase", string(phase)),
        ],
    );
}
struct Gate {
    rx: Receiver<String>,
    paused: bool,
    stopped: bool,
    pending_steps: usize,
    pace: Duration,
}
impl Gate {
    fn new(paused: bool, pace: u64) -> Self {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in io::stdin().lock().lines().map_while(Result::ok) {
                if line.len() <= 16 {
                    let _ = tx.send(line);
                }
            }
        });
        Self {
            rx,
            paused,
            stopped: false,
            pending_steps: 0,
            pace: Duration::from_millis(pace),
        }
    }
    fn command(&mut self, command: &str) {
        match command {
            "pause" => {
                self.paused = true;
                self.pending_steps = 0;
                emit("control", &[("status", string("paused"))]);
            }
            "resume" => {
                self.paused = false;
                self.pending_steps = 0;
                emit("control", &[("status", string("running"))]);
            }
            "stop" => self.stopped = true,
            "step" => {
                self.paused = true;
                self.pending_steps = self.pending_steps.saturating_add(1);
                emit("control", &[("status", string("paused"))]);
            }
            _ => {}
        }
    }
    fn before(&mut self, paced: bool) -> Result<bool, BehaviorError> {
        let deadline = Instant::now() + if paced { self.pace } else { Duration::ZERO };
        loop {
            while let Ok(command) = self.rx.try_recv() {
                self.command(&command);
            }
            if self.stopped {
                return Err(BehaviorError::Invalid("observer stopped"));
            }
            if self.pending_steps > 0 {
                self.pending_steps -= 1;
                return Ok(true);
            }
            if self.paused {
                match self.rx.recv() {
                    Ok(command) => self.command(&command),
                    Err(_) => return Err(BehaviorError::Invalid("observer input closed")),
                }
            } else if Instant::now() < deadline {
                if let Ok(command) = self
                    .rx
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                {
                    self.command(&command);
                }
            } else {
                return Ok(false);
            }
        }
    }
}
type SharedGate = Rc<RefCell<Gate>>;
struct TraceScalar {
    oracle: CachedOracle<Rewired>,
    gate: SharedGate,
    method: String,
    calls: usize,
    context: String,
}
impl SequenceOracle for TraceScalar {
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        self.gate
            .borrow_mut()
            .before(self.calls.is_multiple_of(96) || self.method == "TransferLearner::observe")?;
        let before = self.oracle.raw.queries;
        let clock = Instant::now();
        self.calls += 1;
        let reply = match self.oracle.query(word) {
            Ok(reply) => reply,
            Err(error) => {
                emit(
                    "query_error",
                    &[
                        ("context", string(&self.context)),
                        ("actions", format!("{word:?}")),
                        ("calls", self.calls.to_string()),
                        ("queries", self.oracle.raw.queries.to_string()),
                        ("detail", string(&error.to_string())),
                    ],
                );
                return Err(error);
            }
        };
        {
            emit(
                "query",
                &[
                    ("method", string(&self.method)),
                    ("actions", format!("{word:?}")),
                    ("reply", format!("{reply:?}")),
                    ("queries", self.oracle.raw.queries.to_string()),
                    ("steps", self.oracle.raw.steps.to_string()),
                    ("latencyNs", clock.elapsed().as_nanos().to_string()),
                    ("context", string(&self.context)),
                    (
                        "actionsCount",
                        self.oracle.raw.adapter.action_map().len().to_string(),
                    ),
                    ("calls", self.calls.to_string()),
                    ("cacheHit", (self.oracle.raw.queries == before).to_string()),
                    ("cacheNodes", self.oracle.trie_nodes().to_string()),
                ],
            );
        }
        Ok(reply)
    }
}
impl Drop for TraceScalar {
    fn drop(&mut self) {
        emit(
            "cache_release",
            &[
                ("context", string(&self.context)),
                ("cacheNodes", self.oracle.trie_nodes().to_string()),
            ],
        );
    }
}
struct TraceFrames {
    oracle: FrameCache<factor_lab::Environment>,
    gate: SharedGate,
    calls: usize,
    method: String,
    actions: u8,
}
impl FrameOracle for TraceFrames {
    fn query(&mut self, word: &[u8]) -> Result<FrameReply, BehaviorError> {
        self.gate
            .borrow_mut()
            .before(self.calls.is_multiple_of(256))?;
        let before = self.oracle.queries();
        let clock = Instant::now();
        self.calls += 1;
        let reply = match self.oracle.query(word) {
            Ok(reply) => reply,
            Err(error) => {
                emit(
                    "query_error",
                    &[
                        ("context", string("observer/factor")),
                        ("actions", format!("{word:?}")),
                        ("calls", self.calls.to_string()),
                        ("queries", self.oracle.queries().to_string()),
                        ("detail", string(&error.to_string())),
                    ],
                );
                return Err(error);
            }
        };
        {
            emit(
                "query",
                &[
                    ("method", string(&self.method)),
                    ("actions", format!("{word:?}")),
                    (
                        "frames",
                        format!(
                            "[{}]",
                            reply
                                .iter()
                                .map(|f| format!("{:?}", f.values()))
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                    ),
                    ("queries", self.oracle.queries().to_string()),
                    ("steps", self.oracle.steps().to_string()),
                    ("cacheNodes", self.oracle.cache_nodes().to_string()),
                    ("latencyNs", clock.elapsed().as_nanos().to_string()),
                    ("context", string("observer/factor")),
                    ("actionsCount", self.actions.to_string()),
                    ("calls", self.calls.to_string()),
                    ("cacheHit", (self.oracle.queries() == before).to_string()),
                ],
            );
        }
        Ok(reply)
    }
}
impl Drop for TraceFrames {
    fn drop(&mut self) {
        emit(
            "cache_release",
            &[
                ("context", string("observer/factor")),
                ("cacheNodes", self.oracle.cache_nodes().to_string()),
            ],
        );
    }
}
struct Projection<'a> {
    oracle: &'a mut TraceFrames,
    dimension: usize,
}
fn discover_traced(
    actions: u8,
    context: &str,
    oracle: &mut impl SequenceOracle,
    mut publish: impl FnMut(&BehaviorModel, &str),
) -> Result<BehaviorModel, Box<dyn Error>> {
    let mut learner = BehaviorLearner::new(actions, context, BehaviorBudget::default())?;
    for round in 0..64 {
        stage(
            "BehaviorLearner::hypothesize",
            &format!("{context} / round {round}"),
        );
        let model = learner.hypothesize(oracle)?;
        publish(&model, "hypothesis");
        stage("conformance_probe", context);
        if let Some(word) = conformance_probe(&model, oracle, 65_536)? {
            emit(
                "refine",
                &[
                    ("context", string(context)),
                    ("actions", format!("{word:?}")),
                    ("round", round.to_string()),
                ],
            );
            learner.refine(&model, &word, oracle)?;
        } else {
            publish(&model, "conformance-passed");
            return Ok(model);
        }
    }
    Err("discovery round limit".into())
}
impl SequenceOracle for Projection<'_> {
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        Ok(self
            .oracle
            .query(word)?
            .iter()
            .map(|f| f.values()[self.dimension])
            .collect())
    }
}
fn transfer(seed: u64, novel: bool, gate: SharedGate, out: &Path) -> Result<(), Box<dyn Error>> {
    let mut sources = Vec::new();
    let mut rng = Rng::new(seed);
    let mut source_queries = 0;
    for (i, family) in [Family::Grid, Family::Protocol, Family::Pipeline]
        .into_iter()
        .enumerate()
    {
        let extent = if family == Family::Protocol { 3 } else { 2 };
        stage(
            "BehaviorLearner::hypothesize",
            &format!("learning prototype {i}: {}", family.name()),
        );
        let env = Rewired::random(Environment::new(family, extent, 0), &mut rng);
        let mut oracle = TraceScalar {
            oracle: CachedOracle::new(env, BehaviorBudget::default()),
            gate: gate.clone(),
            method: format!("prototype-{i}"),
            calls: 0,
            context: format!("observer/source-{i}"),
        };
        let model = discover_traced(
            4,
            &format!("observer/source-{i}"),
            &mut oracle,
            |candidate, phase| {
                let mut refs = sources.iter().collect::<Vec<_>>();
                refs.push(candidate);
                graph(&refs, phase);
            },
        )?;
        source_queries += oracle.oracle.raw.queries;
        model.save_new(out.join(format!("source-{i}.axbehavior")))?;
        sources.push(model);
        graph(&sources.iter().collect::<Vec<_>>(), "source-atlas");
    }
    let (family, extent) = if novel {
        (Family::Grid, 3)
    } else {
        (Family::Pipeline, 2)
    };
    let mut target = Rewired::random(Environment::new(family, extent, 0), &mut rng);
    let mut oracle = TraceScalar {
        oracle: CachedOracle::new(target.clone(), BehaviorBudget::default()),
        gate: gate.clone(),
        method: "TransferLearner::observe".into(),
        calls: 0,
        context: "observer/adapted".into(),
    };
    let refs = sources.iter().collect::<Vec<_>>();
    let mut learner =
        TransferLearner::new(&refs, 4, "observer/adapted", TransferBudget::default())?;
    let mut fallback = false;
    let mut model = loop {
        if let Some(result) = learner.compile()? {
            break result.model;
        }
        let probe = learner.choose_probe()?.ok_or("no useful probe")?;
        stage(
            "TransferLearner::choose_probe",
            &format!(
                "separate {} pairs / {} actions",
                probe.eliminated_pairs,
                probe.word.len()
            ),
        );
        let reply = oracle.query(&probe.word)?;
        let observation = learner.observe(&probe.word, &reply);
        let stats = learner.stats();
        emit(
            "probe",
            &[
                ("method", string("TransferLearner::observe")),
                ("actions", format!("{:?}", probe.word)),
                ("reply", format!("{reply:?}")),
                ("hypotheses", stats.surviving_hypotheses.to_string()),
                ("initialHypotheses", stats.initial_hypotheses.to_string()),
                ("work", stats.work.to_string()),
                ("probes", stats.queries.to_string()),
                ("queries", oracle.oracle.raw.queries.to_string()),
                ("steps", oracle.oracle.raw.steps.to_string()),
            ],
        );
        if observation.is_err() {
            fallback = true;
            stage("discover", "prior refuted; learning new representation");
            break discover_traced(4, "observer/adapted", &mut oracle, |m, p| graph(&[m], p))?;
        }
    };
    oracle.method = "conformance_probe".into();
    stage(
        "conformance_probe",
        "bounded validation; independent audit follows",
    );
    if let Some(word) = conformance_probe(&model, &mut oracle, 65536)? {
        fallback = true;
        stage(
            "discover",
            &format!(
                "prior contradicted by conformance witness {word:?}; learning new representation"
            ),
        );
        oracle.method = "discover".into();
        model = discover_traced(4, "observer/adapted", &mut oracle, |m, p| graph(&[m], p))?;
    }
    graph(&[&model], "conformance-passed");
    stage(
        "audit_machine",
        "compare every reachable learned/true state pair",
    );
    let audit = audit_machine(&model, 4, 0, |s, a| target.transition(s, a))?;
    if audit.counterexample.is_some() {
        return Err("exact audit mismatch".into());
    }
    model.save_new(out.join("adapted.axbehavior"))?;
    graph(&[&model], "audited");
    let goal = target.output_for(if novel { 1008 } else { 201 });
    let plan = model
        .plan_output(&[], goal, &Default::default())?
        .ok_or("goal unreachable")?;
    stage(
        "BehaviorModel::plan_output",
        "shortest path to supplied goal token",
    );
    emit(
        "plan",
        &[("actions", format!("{plan:?}")), ("goal", goal.to_string())],
    );
    let mut tracker = BehaviorTracker::new(&model, model.context())?;
    for (i, &action) in plan.iter().enumerate() {
        gate.borrow_mut().before(true)?;
        let from = tracker.state();
        let actual = target.step(action)?;
        tracker.observe(action, actual)?;
        emit(
            "step",
            &[
                ("index", i.to_string()),
                ("action", action.to_string()),
                ("reply", format!("[{actual}]")),
                ("state", tracker.state().to_string()),
                ("residual", "0".into()),
                ("method", string("BehaviorTracker::observe")),
                ("context", string(model.context())),
                ("fromStates", format!("[{from}]")),
                ("toStates", format!("[{}]", tracker.state())),
            ],
        );
    }
    emit(
        "result",
        &[
            ("exact", "true".into()),
            ("residual", "0".into()),
            ("states", model.states().to_string()),
            ("bytes", model.encoded_bytes().to_string()),
            ("queries", oracle.oracle.raw.queries.to_string()),
            ("sourceQueries", source_queries.to_string()),
            ("steps", oracle.oracle.raw.steps.to_string()),
            ("checkedEdges", audit.checked_edges.to_string()),
            ("fallback", fallback.to_string()),
        ],
    );
    Ok(())
}
fn factor(
    seed: u64,
    dimensions: usize,
    gate: SharedGate,
    out: &Path,
) -> Result<(), Box<dyn Error>> {
    let env = factor_lab::Environment::random(factor_lab::Kind::Workflow, dimensions, seed)?
        .zero_initial();
    let mut oracle = TraceFrames {
        oracle: FrameCache::new(
            env.clone(),
            env.actions(),
            dimensions,
            FrameBudget::default(),
            Authority::from([Effect::Simulate]),
        )?,
        gate: gate.clone(),
        calls: 0,
        method: String::new(),
        actions: env.actions(),
    };
    let mut factors = Vec::new();
    for dimension in 0..dimensions {
        oracle.method = format!("projection-{dimension}");
        stage(
            "factor::learn",
            &format!(
                "learn channel {dimension}; all {} global actions",
                env.actions()
            ),
        );
        let model = discover_traced(
            env.actions(),
            &format!("observer/factor/factor-{dimension}"),
            &mut Projection {
                oracle: &mut oracle,
                dimension,
            },
            |candidate, phase| {
                let mut refs = factors.iter().collect::<Vec<_>>();
                refs.push(candidate);
                graph(&refs, phase);
            },
        )?;
        factors.push(model);
        graph(&factors.iter().collect::<Vec<_>>(), "conformance-passed");
    }
    let model = FactorModel::new("observer/factor", factors)?;
    stage(
        "factor_lab::audit",
        "exact product audit; no refinement feedback",
    );
    let audit = factor_lab::audit(&model, &env, 65536)?;
    if audit.counterexample.is_some() {
        return Err("factor audit mismatch".into());
    }
    graph(&model.factors().iter().collect::<Vec<_>>(), "audited");
    model.save_bundle_new(
        &out.join("models"),
        &Authority::from([Effect::ReadData, Effect::WriteData]),
    )?;
    let goal = model.goal(&env.goal_tokens())?;
    let plan = model
        .plan(&[], &goal, PlanBudget::default())?
        .ok_or("no factor plan")?;
    stage("FactorModel::plan", "joint BFS over predictive states");
    emit(
        "plan",
        &[
            ("actions", format!("{plan:?}")),
            ("productStates", model.product_states().to_string()),
        ],
    );
    let mut physical = env;
    let mut tracker = FactorTracker::new(&model, model.context())?;
    for (index, &action) in plan.iter().enumerate() {
        gate.borrow_mut().before(true)?;
        let before = tracker.states()[..dimensions].to_vec();
        let frame = physical.step(action)?;
        tracker.observe(action, frame)?;
        emit(
            "step",
            &[
                ("index", index.to_string()),
                ("action", action.to_string()),
                ("reply", format!("{:?}", frame.values())),
                ("residual", "0".into()),
                ("method", string("FactorTracker::observe")),
                ("context", string(model.context())),
                ("fromStates", format!("{before:?}")),
                ("toStates", format!("{:?}", &tracker.states()[..dimensions])),
            ],
        );
    }
    emit(
        "result",
        &[
            ("exact", "true".into()),
            ("residual", "0".into()),
            ("states", model.factor_states().to_string()),
            ("productStates", model.product_states().to_string()),
            ("bytes", model.encoded_model_bytes().to_string()),
            ("queries", oracle.oracle.queries().to_string()),
            ("steps", oracle.oracle.steps().to_string()),
            ("checkedEdges", audit.checked_edges.to_string()),
        ],
    );
    Ok(())
}
fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|s| s == "snapshot") && args.len() == 3 {
        let model = FactorModel::load_bundle(
            Path::new(&args[1]),
            &args[2],
            &Authority::from([Effect::ReadData]),
        )?;
        graph(
            &model.factors().iter().collect::<Vec<_>>(),
            "stored-snapshot",
        );
        return Ok(());
    }
    if args.len() != 6 || !["transfer", "factor"].contains(&args[0].as_str()) {
        return Err("usage: axon-observer <transfer|factor> <seed> <2..5 dimensions> <novel:0|1> <paused:0|1> <new-directory>".into());
    }
    let seed = args[1].parse::<u64>()?;
    let dimensions = args[2].parse::<usize>()?;
    if !(2..=5).contains(&dimensions)
        || !["0", "1"].contains(&args[3].as_str())
        || !["0", "1"].contains(&args[4].as_str())
    {
        return Err("invalid observer options".into());
    }
    let gate = Rc::new(RefCell::new(Gate::new(args[4] == "1", 75)));
    let out = Path::new(&args[5]);
    std::fs::create_dir(out)?;
    emit(
        "start",
        &[
            ("mode", string(&args[0])),
            ("seed", seed.to_string()),
            ("dimensions", dimensions.to_string()),
            ("telemetry", string("complete-query-trace")),
            (
                "status",
                string(if args[4] == "1" { "paused" } else { "running" }),
            ),
            (
                "scope",
                string(
                    "finite resettable lab adapter; instrumented algorithm events, not consciousness",
                ),
            ),
        ],
    );
    if args[0] == "transfer" {
        transfer(seed, args[3] == "1", gate, out)
    } else {
        factor(seed, dimensions, gate, out)
    }
}
fn main() {
    if let Err(e) = run() {
        emit("error", &[("detail", string(&e.to_string()))]);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_strings_escape_control_and_unicode_without_rust_debug_syntax() {
        assert_eq!(string("a\n\"\\\u{0001}ç"), "\"a\\n\\\"\\\\\\u0001ç\"");
    }
    #[test]
    fn each_queued_step_releases_exactly_one_checkpoint_and_preserves_pause() {
        let (tx, rx) = mpsc::channel();
        let mut gate = Gate {
            rx,
            paused: true,
            stopped: false,
            pending_steps: 0,
            pace: Duration::ZERO,
        };
        tx.send("step".into()).unwrap();
        tx.send("step".into()).unwrap();
        assert!(gate.before(false).unwrap());
        assert!(gate.paused);
        assert_eq!(gate.pending_steps, 1);
        assert!(gate.before(false).unwrap());
        assert_eq!(gate.pending_steps, 0);
        tx.send("resume".into()).unwrap();
        assert!(!gate.before(false).unwrap());
        assert!(!gate.paused);
    }
    #[test]
    fn stop_interrupts_a_paused_checkpoint() {
        let (tx, rx) = mpsc::channel();
        let mut gate = Gate {
            rx,
            paused: true,
            stopped: false,
            pending_steps: 0,
            pace: Duration::ZERO,
        };
        tx.send("stop".into()).unwrap();
        assert!(gate.before(false).is_err());
    }
}
