//! Explicit long-context joint-pointer experiments; no tool authority or teacher at rollout.
use super::*;
use pointer_model::context_read::{ContextModel, Read};
use pointer_model::{Budget, Example, Input, Measurement, Model as PointerModel};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Layout {
    Legacy,
    Blind,
    Working,
    WorkingUniform,
    WorkingAttention,
}
impl Layout {
    pub(super) fn module(self) -> &'static str {
        match self {
            Self::Legacy => pointer_model::MODULE,
            Self::Blind => "semantic-working-pointer-blind-v1",
            Self::Working => "semantic-working-pointer-v1",
            Self::WorkingUniform => "semantic-uniform-working-pointer-v1",
            Self::WorkingAttention => "semantic-attention-working-pointer-v1",
        }
    }
    pub(super) fn file(self) -> &'static str {
        match self {
            Self::Legacy => "relational-pointer.axrp",
            Self::Blind => "working-pointer-blind.axrp",
            Self::Working => "working-pointer.axrp",
            Self::WorkingUniform => "uniform-working-pointer.axrp",
            Self::WorkingAttention => "attention-working-pointer.axrp",
        }
    }
    pub(super) fn read(self) -> Option<Read> {
        match self {
            Self::WorkingUniform => Some(Read::Uniform),
            Self::WorkingAttention => Some(Read::Attention),
            _ => None,
        }
    }
    pub(super) fn kind(self) -> ModuleKind {
        if self.read().is_some() {
            ModuleKind::ContextPointer
        } else {
            ModuleKind::RelationalPointer
        }
    }
    fn context(self) -> &'static str {
        if self.read().is_some() {
            pointer_model::context_read::CONTEXT
        } else {
            pointer_model::CONTEXT
        }
    }
}
fn command(name: &str) -> Option<(Layout, &str)> {
    let (layout, action) = if let Some(action) = name.strip_prefix("uniform-working-pointer-") {
        (Layout::WorkingUniform, action)
    } else if let Some(action) = name.strip_prefix("attention-working-pointer-") {
        (Layout::WorkingAttention, action)
    } else if let Some(action) = name.strip_prefix("working-pointer-blind-") {
        (Layout::Blind, action)
    } else if let Some(action) = name.strip_prefix("working-pointer-") {
        (Layout::Working, action)
    } else {
        (Layout::Legacy, name.strip_prefix("pointer-")?)
    };
    matches!(
        action,
        "train" | "pack" | "batch" | "frames-batch" | "session" | "frames-session"
    )
    .then_some((layout, action))
}
pub fn handles(name: &str) -> bool {
    command(name).is_some()
}
pub fn run(name: &str, number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    let (layout, action) = command(name).ok_or("pointer command")?;
    match action {
        "train" => train_layout(layout, number, birth, data, output),
        "pack" => super::pointer_pack_layout(number, birth, data, output, layout),
        "batch" => batch_layout(layout, number, birth, data, output),
        "frames-batch" => frames_batch_layout(layout, number, birth, data, output),
        "session" => session(layout, false, number, birth, data, output),
        "frames-session" => session(layout, true, number, birth, data, output),
        _ => Err("pointer command".into()),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: Budget,
    examples: Vec<Example>,
}
enum Trained {
    Legacy(PointerModel),
    Context(ContextModel),
}
impl Trained {
    fn base(&self) -> &PointerModel {
        match self {
            Self::Legacy(m) => m,
            Self::Context(m) => m.base(),
        }
    }
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(match self {
            Self::Legacy(m) => m.to_bytes()?,
            Self::Context(m) => m.to_bytes()?,
        })
    }
    fn measure(&self, input: &Input, work: usize, workspace: usize) -> Measurement {
        match self {
            Self::Legacy(m) => m.measure(input, work, workspace),
            Self::Context(m) => m.measure(input, work, workspace),
        }
    }
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Legacy(m) => m.retained_bytes(),
            Self::Context(m) => m.retained_bytes(),
        }
    }
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_layout(Layout::Legacy, number, birth, data, output)
}
fn train_layout(
    layout: Layout,
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
) -> Result<()> {
    for path in [number, birth, data, output] {
        bounded_path(path)?;
    }
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let mut telemetry = LabTelemetry::from_environment(layout.module());
    let mut events = Vec::new();
    let started = Instant::now();
    let mut emit = |e: &pointer_model::Event| {
        telemetry
            .emit(
                "pointer-training",
                &[
                    ("epoch", V::Count(e.epoch as u64)),
                    ("examples", V::Count(e.examples as u64)),
                    ("errors", V::Count(e.errors as u64)),
                    ("optimizer_steps", V::Count(e.updates as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))?;
        events.push(
            serde_json::to_value(e)
                .map_err(|e| crate::structural::Error::Storage(e.to_string()))?,
        );
        Ok(())
    };
    let model = if let Some(read) = layout.read() {
        Trained::Context(ContextModel::learn(
            &training.examples,
            training.budget,
            read,
            &mut emit,
        )?)
    } else {
        Trained::Legacy(PointerModel::learn(
            &training.examples,
            training.budget,
            &mut emit,
        )?)
    };
    let mut correct = 0;
    let mut work = 0;
    let mut refusals = 0;
    for e in &training.examples {
        let measured = model.measure(
            &e.input,
            2_000_000,
            262144usize.saturating_sub(model.retained_bytes() + std::mem::size_of::<Module>()),
        );
        work += measured.work;
        match measured.result {
            Ok(p) => {
                correct += usize::from(e.targets.contains(&p.selection));
            }
            Err(_) => refusals += 1,
        }
    }
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("pointer teaching changed parent".into());
    }
    fs::create_dir(output)?;
    let bytes = model.to_bytes()?;
    let genome = packed_payloads(
        &[
            ("structural", ModuleKind::Structural, &before),
            (layout.module(), layout.kind(), &bytes),
        ],
        output,
    )?;
    let report = serde_json::json!({"schema":"axon-relational-pointer-training/v1","budget":training.budget,"total":training.examples.len(),"words":model.base().words(),"candidate_lexemes_retained":model.base().candidate_lexemes_retained(&training.examples)?,"page_bytes":bytes.len(),"retained_model_bytes":model.retained_bytes(),"encoded_bytes":genome.encoded_bytes(),"parent_unchanged":true,"events":events,"teacher_fit":{"correct":correct,"refused":refusals,"total":training.examples.len(),"work":work,"quantized_same_joint_decoder":true},"elapsed_ns":started.elapsed().as_nanos(),"scope":"Joint supervised pointers; provided candidates/trace/label arity, learned scores and embeddings. Teacher states only in training, not autonomous task success or M1. Exact argmax uses deterministic ties and reports margin, not calibrated semantic confidence. Training workspace/Adam not inherited; order fingerprint is noncryptographic metadata."});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
fn frame_input(text: String) -> Result<Input> {
    let words = pointer_model::tokens(&text)?;
    let candidates = words
        .iter()
        .enumerate()
        .filter_map(|(i, w)| scalar(w).ok().map(|_| i))
        .collect::<Vec<_>>();
    if candidates.is_empty() || candidates.len() > 32 {
        return Err("pointer scalar candidates".into());
    }
    Ok(Input { text, candidates })
}
fn measure(
    layout: Layout,
    runtime: &mut DemandRuntime,
    input: &Input,
    auth: &Authority,
    cap: usize,
) -> Result<Measurement> {
    let circuit_reserve = runtime
        .catalog()
        .entries()
        .iter()
        .filter(|e| actor::program_id(&e.id))
        .map(|e| e.retained_bytes)
        .max()
        .unwrap_or(0);
    Ok(
        runtime.with_module(layout.module(), layout.context(), auth, |page| {
            let budget = cap.saturating_sub(
                page.retained_bytes()
                    + circuit_reserve
                    + input.text.capacity()
                    + input.candidates.capacity() * std::mem::size_of::<usize>(),
            );
            match page {
                Module::RelationalPointer(model) if layout.read().is_none() => {
                    Ok(model.measure(input, 2_000_000, budget))
                }
                Module::ContextPointer(model) if Some(model.read()) == layout.read() => {
                    Ok(model.measure(input, 2_000_000, budget))
                }
                _ => Err(crate::demand::DemandError::Invalid(
                    "pointer page kind/read",
                )),
            }
        })?,
    )
}
pub fn frames_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    frames_batch_layout(Layout::Legacy, number, birth, data, receipt)
}
fn validate_requests(inputs: &[InterpretationInput], maximum: usize) -> Result<()> {
    let mut ids = BTreeSet::new();
    if inputs.is_empty()
        || inputs.len() > maximum
        || inputs
            .iter()
            .any(|r| r.id.is_empty() || r.id.len() > 96 || !ids.insert(&r.id))
    {
        return Err("pointer batch quota/identity".into());
    }
    Ok(())
}
fn frames_batch_layout(
    layout: Layout,
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
) -> Result<()> {
    for path in [number, birth, data, receipt] {
        bounded_path(path)?;
    }
    let inputs: Vec<InterpretationInput> = serde_json::from_slice(&read_bounded(data, 300_000)?)?;
    let mut ids = BTreeSet::new();
    if inputs.is_empty()
        || inputs.len() > 32
        || inputs
            .iter()
            .any(|r| r.id.is_empty() || r.id.len() > 96 || !ids.insert(&r.id))
    {
        return Err("pointer frame batch quota/identity".into());
    }
    let prepared = inputs
        .iter()
        .map(|r| frame_input(r.text.clone()))
        .collect::<Result<Vec<_>>>()?;
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let cap = genome.residency().active_bytes.min(262144);
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = born.controller.runtime_mut();
    let (rows, work) = measured_rows(layout, runtime, &inputs, &prepared, &auth, cap)?;
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("pointer frame inference mutated genome".into());
    }
    publish_new(receipt, &serde_json::to_vec(&rows)?)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"supplied_states":true,"additional_teacher_queries":0,"seed_unchanged":true,"arithmetic_evaluations":0,"work":work,"loads":runtime.stats().loads,"scope":"Supplied teacher-state diagnostic, not autonomous success; NN workspace accounting excludes caller state/history/JSON/process."})
    );
    Ok(())
}
fn measured_rows(
    layout: Layout,
    runtime: &mut DemandRuntime,
    inputs: &[InterpretationInput],
    prepared: &[Input],
    auth: &Authority,
    cap: usize,
) -> Result<(Vec<serde_json::Value>, usize)> {
    let mut rows = Vec::new();
    let mut work = 0;
    for (r, input) in inputs.iter().zip(prepared) {
        let measured = measure(layout, runtime, input, auth, cap)?;
        work += measured.work;
        if work > 100_000_000 {
            return Err("pointer batch work quota".into());
        }
        let (prediction, error, workspace, margin) = match measured.result {
            Ok(p) => (Some(p.selection), None, Some(p.workspace_bytes), p.margin),
            Err(e) => (None, Some(e.to_string()), None, None),
        };
        rows.push(serde_json::json!({"id":r.id,"frame":r.text,"prediction":prediction,"error":error,"work":measured.work,"workspace_bytes":workspace,"margin":margin}));
    }
    Ok((rows, work))
}
#[derive(Clone, Copy, serde::Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Origin {
    Question { index: usize },
    Register { index: usize },
}
impl Origin {
    fn token(self, side: char) -> String {
        match self {
            Self::Question { index } => format!("{side}:q{index}"),
            Self::Register { index } => format!("{side}:r{index}"),
        }
    }
}
struct History {
    program: String,
    left: Origin,
    right: Origin,
}
struct State {
    words: Vec<String>,
    registers: Vec<f64>,
    history: Vec<History>,
    layout: Layout,
}
impl State {
    fn new(text: &str) -> Result<Self> {
        Self::new_layout(text, Layout::Legacy)
    }
    fn new_layout(text: &str, layout: Layout) -> Result<Self> {
        if text.len() > 4096 {
            return Err("pointer question bytes".into());
        }
        let words = pointer_model::tokens(text)?;
        if words.len() > 512 {
            return Err("pointer question words".into());
        }
        Ok(Self {
            words,
            registers: Vec::new(),
            history: Vec::new(),
            layout,
        })
    }
    fn frame(&self) -> Result<Input> {
        if self.layout != Layout::Legacy {
            if self.history.len() != self.registers.len() || self.registers.len() > 8 {
                return Err("working pointer history shape".into());
            }
            let mut words = self.words.clone();
            for (index, (node, n)) in self.history.iter().zip(&self.registers).enumerate() {
                if !n.is_finite() || n.abs() > 1e9 {
                    return Err("pointer register bounds".into());
                }
                let valid = |origin| match origin {
                    Origin::Question { index } => {
                        self.words.get(index).is_some_and(|w| scalar(w).is_ok())
                    }
                    Origin::Register { index: source } => source < index,
                };
                if !actor::program_id(&node.program) || !valid(node.left) || !valid(node.right) {
                    return Err("working pointer history source".into());
                }
                if self.layout != Layout::Blind {
                    words.extend([
                        format!("op:{}", node.program),
                        node.left.token('a'),
                        node.right.token('b'),
                    ]);
                } else {
                    words.extend(["work".into(), "left".into(), "right".into()]);
                }
                words.push(if *n == 0.0 { "0".into() } else { n.to_string() });
            }
            return frame_input(words.join(" "));
        }
        let mut words = vec![
            format!("axon-step-{}", self.registers.len()),
            "axon-question".into(),
        ];
        words.extend(self.words.iter().cloned());
        words.push("axon-memory".into());
        for (i, n) in self.registers.iter().enumerate() {
            if !n.is_finite() || n.abs() > 1e9 {
                return Err("pointer register bounds".into());
            }
            words.push(format!("r{i}"));
            words.push(if *n == 0.0 { "0".into() } else { n.to_string() });
        }
        frame_input(words.join(" "))
    }
    fn origin(&self, position: usize) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(self.source(position)?)?)
    }
    fn source(&self, position: usize) -> Result<Origin> {
        if self.layout != Layout::Legacy {
            if position < self.words.len() {
                return Ok(Origin::Question { index: position });
            }
            let start = self.words.len() + 3;
            if position >= start
                && (position - start).is_multiple_of(4)
                && (position - start) / 4 < self.registers.len()
            {
                return Ok(Origin::Register {
                    index: (position - start) / 4,
                });
            }
            return Err("pointer source position".into());
        }
        if (2..self.words.len() + 2).contains(&position) {
            return Ok(Origin::Question {
                index: position - 2,
            });
        }
        let start = self.words.len() + 4;
        if position >= start
            && (position - start).is_multiple_of(2)
            && (position - start) / 2 < self.registers.len()
        {
            return Ok(Origin::Register {
                index: (position - start) / 2,
            });
        }
        Err("pointer source position".into())
    }
}
fn solve(
    layout: Layout,
    runtime: &mut DemandRuntime,
    text: &str,
    auth: &Authority,
    total: &mut usize,
    cap: usize,
) -> serde_json::Value {
    let mut trace = Vec::new();
    let mut work = 0;
    let mut calculations = 0;
    let mut peak_workspace = 0;
    let result = (|| -> Result<f64> {
        let mut state = if layout == Layout::Legacy {
            State::new(text)?
        } else {
            State::new_layout(text, layout)?
        };
        for step in 0..=8 {
            let input = state.frame()?;
            let measured = measure(layout, runtime, &input, auth, cap)?;
            work += measured.work;
            *total += measured.work;
            if work > 18_000_000 || *total > 100_000_000 {
                return Err("pointer aggregate work quota".into());
            }
            let p = match measured.result {
                Ok(p) => p,
                Err(e) => {
                    trace.push(serde_json::json!({"step":step,"frame":input.text,"work":measured.work,"error":e.to_string()}));
                    return Err(e.into());
                }
            };
            peak_workspace = peak_workspace.max(p.workspace_bytes);
            let mut event = serde_json::json!({"step":step,"frame":input.text,"prediction":p.selection,"work":p.work,"workspace_bytes":p.workspace_bytes,"margin":p.margin,"blocks":p.blocks});
            let words = pointer_model::tokens(&input.text)?;
            let selection = p.selection;
            let read = |i: usize| -> Result<f64> {
                if !input.candidates.contains(&i) {
                    return Err("pointer selected unavailable source".into());
                }
                Ok(scalar(&words[i])?)
            };
            let a = read(selection.left)?;
            event["left_origin"] = state.origin(selection.left)?;
            if selection.label == "math-done" && selection.right.is_none() {
                event["answer"] = serde_json::json!(a);
                trace.push(event);
                return Ok(a);
            }
            if !actor::program_id(&selection.label) || selection.right.is_none() {
                event["error"] = serde_json::json!("pointer action shape");
                trace.push(event);
                return Err("pointer action shape".into());
            }
            if step == 8 {
                event["error"] = serde_json::json!("pointer action quota");
                trace.push(event);
                return Err("pointer action quota".into());
            }
            let b_position = selection.right.unwrap();
            let b = read(b_position)?;
            event["right_origin"] = state.origin(b_position)?;
            if !runtime
                .catalog()
                .get(&selection.label)
                .is_some_and(|e| e.kind == ModuleKind::Circuit && e.input == "numbers/2")
            {
                return Err("pointer circuit unavailable".into());
            }
            calculations += 1;
            match runtime.predict_numeric(
                &selection.label,
                crate::numeric::evolution::constructive::CONTEXT,
                &[a, b],
                auth,
            ) {
                Ok(n) if n.is_finite() && n.abs() <= 1e9 => {
                    event["computed"] = serde_json::json!(n);
                    if layout != Layout::Legacy {
                        state.history.push(History {
                            program: selection.label.clone(),
                            left: state.source(selection.left)?,
                            right: state.source(b_position)?,
                        });
                    }
                    state.registers.push(n);
                }
                Ok(_) => {
                    event["error"] = serde_json::json!("pointer result bounds");
                    trace.push(event);
                    return Err("pointer result bounds".into());
                }
                Err(e) => {
                    event["error"] = serde_json::json!(e.to_string());
                    trace.push(event);
                    return Err(e.into());
                }
            }
            trace.push(event);
        }
        Err("pointer did not stop".into())
    })();
    let (answer, error) = match result {
        Ok(n) => (Some(n), None),
        Err(e) => (None, Some(e.to_string())),
    };
    serde_json::json!({"answer":answer,"error":error,"language_work":work,"arithmetic_evaluations":calculations,"peak_model_workspace_bytes":peak_workspace,"trace":trace})
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_layout(Layout::Legacy, number, birth, data, receipt)
}
fn batch_layout(
    layout: Layout,
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
) -> Result<()> {
    for path in [number, birth, data, receipt] {
        bounded_path(path)?;
    }
    let inputs: Vec<InterpretationInput> = serde_json::from_slice(&read_bounded(data, 65536)?)?;
    let mut ids = BTreeSet::new();
    if inputs.is_empty()
        || inputs.len() > 4
        || inputs
            .iter()
            .any(|i| i.id.is_empty() || i.id.len() > 96 || !ids.insert(&i.id))
    {
        return Err("pointer batch quota/identity".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let cap = genome.residency().active_bytes.min(262144);
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = born.controller.runtime_mut();
    let (rows, work) = own_rows(layout, runtime, &inputs, &auth, cap);
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("pointer inference mutated genome".into());
    }
    let bytes = serde_json::to_vec(&rows)?;
    if bytes.len() > 1_000_000 {
        return Err("pointer receipt quota".into());
    }
    publish_new(receipt, &bytes)?;
    let stats = runtime.stats();
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"seed_unchanged":true,"teacher_queries":0,"loads":stats.loads,"peak_active_modules":stats.peak_active_modules,"peak_active_payload_bytes":stats.peak_active_bytes,"language_work":work,"scope":"Own-result joint pointers over complete bounded text, copied-source arithmetic only. Model workspace accounted separately and preflighted with model/circuit payload; caller state/history/JSON/process excluded. Deterministic score ties, no calibrated semantic confidence, units or full M1 certificate."})
    );
    Ok(())
}
fn own_rows(
    layout: Layout,
    runtime: &mut DemandRuntime,
    inputs: &[InterpretationInput],
    auth: &Authority,
    cap: usize,
) -> (Vec<serde_json::Value>, usize) {
    let mut work = 0;
    let rows = inputs
        .iter()
        .map(|input| {
            let mut row = solve(layout, runtime, &input.text, auth, &mut work, cap);
            row["id"] = serde_json::json!(input.id);
            row
        })
        .collect();
    (rows, work)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Session {
    batches: Vec<Vec<InterpretationInput>>,
}
fn session(
    layout: Layout,
    frames: bool,
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
) -> Result<()> {
    for p in [number, birth, data, receipt] {
        bounded_path(p)?;
    }
    let session: Session = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    if session.batches.is_empty() || session.batches.len() > 512 {
        return Err("pointer session batch quota".into());
    }
    let mut prepared = Vec::new();
    for batch in &session.batches {
        validate_requests(batch, if frames { 32 } else { 4 })?;
        if serde_json::to_vec(batch)?.len() > if frames { 300_000 } else { 65536 } {
            return Err("pointer logical batch byte quota".into());
        }
        if frames {
            prepared.push(
                batch
                    .iter()
                    .map(|r| frame_input(r.text.clone()))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let cap = genome.residency().active_bytes.min(262144);
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = born.controller.runtime_mut();
    let mut bytes = b"{\"batches\":[".to_vec();
    let mut total_work = 0u64;
    let mut total = 0usize;
    for (i, batch) in session.batches.iter().enumerate() {
        let (rows, work) = if frames {
            measured_rows(layout, runtime, batch, &prepared[i], &auth, cap)?
        } else {
            own_rows(layout, runtime, batch, &auth, cap)
        };
        let encoded = serde_json::to_vec(&rows)?;
        if encoded.len() > 1_000_000 || bytes.len() + encoded.len() + 4 > 64_000_000 {
            return Err("pointer session receipt quota".into());
        }
        if i > 0 {
            bytes.push(b',');
        }
        bytes.extend_from_slice(&encoded);
        total_work = total_work
            .checked_add(work as u64)
            .ok_or("pointer session work overflow")?;
        total += rows.len();
    }
    bytes.extend_from_slice(b"]}");
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("pointer session mutated genome".into());
    }
    publish_new(receipt, &bytes)?;
    let stats = runtime.stats();
    println!(
        "{}",
        serde_json::json!({"session":true,"native_births":1,"logical_batches":session.batches.len(),"total":total,"supplied_states":frames,"teacher_queries":0,"arithmetic_evaluations":if frames{Some(0)}else{None},"seed_unchanged":true,"language_work":total_work,"loads":stats.loads,"peak_active_modules":stats.peak_active_modules,"peak_active_payload_bytes":stats.peak_active_bytes,"scope":"Bounded sequential logical batches with one real birth and immutable model cache. Query state resets per request; existing work budgets reset only per logical batch as with separate calls. Input/receipt buffers and process are not accounted query workspace/RSS; supplied states are diagnostic only."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{NumericBudget, NumericExample, NumericSkill, NumericTolerance};
    #[test]
    fn long_frames_keep_distinct_global_origins_for_equal_values() {
        let text = format!("3 {} 3", vec!["word"; 80].join(" "));
        let mut state = State::new(&text).unwrap();
        let input = state.frame().unwrap();
        assert_eq!(input.candidates, vec![2, 83]);
        assert_eq!(
            state.origin(2).unwrap(),
            serde_json::json!({"kind":"question","index":0})
        );
        assert_eq!(
            state.origin(83).unwrap(),
            serde_json::json!({"kind":"question","index":81})
        );
        state.registers.push(6.0);
        let input = state.frame().unwrap();
        assert_eq!(input.candidates, vec![2, 83, 86]);
        assert_eq!(
            state.origin(86).unwrap(),
            serde_json::json!({"kind":"register","index":0})
        );
        assert!(state.origin(85).is_err());
        assert!(State::new(&vec!["x"; 513].join(" ")).is_err());
    }
    #[test]
    fn cold_pointer_genome_computes_own_result_and_preserves_parent() {
        cold_layout(Layout::Legacy);
    }
    #[test]
    fn working_and_blind_genomes_keep_own_provenance_and_session_equivalence() {
        cold_layout(Layout::Working);
        cold_layout(Layout::Blind);
    }
    #[test]
    fn context_read_genomes_compute_own_results_and_keep_session_identity() {
        for layout in [Layout::WorkingUniform, Layout::WorkingAttention] {
            cold_layout(layout);
        }
    }
    #[test]
    fn history_distinguishes_equal_values_within_existing_maximum_context() {
        let mut a = State::new_layout("2 and 2", Layout::Working).unwrap();
        a.history.push(History {
            program: "math-p032".into(),
            left: Origin::Question { index: 0 },
            right: Origin::Question { index: 2 },
        });
        a.registers.push(4.0);
        let before = a.frame().unwrap();
        a.history[0].program = "math-p034".into();
        let after = a.frame().unwrap();
        assert_ne!(before.text, after.text);
        assert_eq!(before.candidates, after.candidates);
        a.layout = Layout::Blind;
        let blind = a.frame().unwrap();
        a.history[0].program = "math-p032".into();
        assert_eq!(blind.text, a.frame().unwrap().text);
        let question = format!("{} 3", vec!["x"; 511].join(" "));
        let mut state = State::new_layout(&question, Layout::Working).unwrap();
        for i in 0..8 {
            state.history.push(History {
                program: "math-p032".into(),
                left: Origin::Question { index: 511 },
                right: if i == 0 {
                    Origin::Question { index: 511 }
                } else {
                    Origin::Register { index: i - 1 }
                },
            });
            state.registers.push((i + 1) as f64);
        }
        let frame = state.frame().unwrap();
        assert_eq!(frame.text.split_whitespace().count(), 544);
        assert_eq!(
            frame.candidates,
            vec![511, 515, 519, 523, 527, 531, 535, 539, 543]
        );
        assert_eq!(
            state.origin(543).unwrap(),
            serde_json::json!({"kind":"register","index":7})
        );
        assert!(state.origin(512).is_err());
        state.history[0].right = Origin::Register { index: 0 };
        assert!(state.frame().is_err());
    }
    fn cold_layout(layout: Layout) {
        let root = std::env::temp_dir().join(format!(
            "axon-pointer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let parent = root.join("parent");
        fs::create_dir(&parent).unwrap();
        let seed = StructuralSeed::default();
        packed(&seed, &parent).unwrap();
        let number = parent.join("structural.axnum");
        let original = fs::read(&number).unwrap();
        let mut examples = Vec::new();
        for n in 0..24 {
            for done in [false, true] {
                let (a, b) = (n + 2, 3 * n + 11);
                let mut state =
                    State::new_layout(&format!("combine {a} with {b}"), layout).unwrap();
                let (left, right) = if layout == Layout::Legacy {
                    (3, 5)
                } else {
                    (1, 3)
                };
                if done {
                    if layout != Layout::Legacy {
                        state.history.push(History {
                            program: "math-p032".into(),
                            left: Origin::Question { index: 1 },
                            right: Origin::Question { index: 3 },
                        });
                    }
                    state.registers.push(f64::from(a + b));
                }
                let input = state.frame().unwrap();
                let target = *input.candidates.last().unwrap();
                examples.push(serde_json::json!({"input":{"text":input.text,"candidates":input.candidates},"targets":if done{serde_json::json!([{"label":"math-done","left":target,"right":null}])}else{serde_json::json!([{"label":"math-p032","left":left,"right":right},{"label":"math-p032","left":right,"right":left}])}}));
            }
        }
        let config = root.join("training.json");
        save(&config,&serde_json::json!({"budget":{"seed":47,"vocabulary":128,"dimensions":8,"epochs":32,"learning_rate":0.01,"minibatch":4},"examples":examples})).unwrap();
        let trained = root.join("trained");
        train_layout(
            layout,
            &number,
            &root.join("training-birth"),
            &config,
            &trained,
        )
        .unwrap();
        let observations = (0..8)
            .map(|i| {
                let (a, b) = (f64::from(i + 1), f64::from(i * 3 + 11));
                NumericExample::new(vec![a, b], a + b)
            })
            .collect();
        let numeric = NumericSkill::learn(
            "pointer-add",
            "pointer/test",
            observations,
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )
        .unwrap();
        let circuit = root.join("add.axcg");
        publish_new(
            &circuit,
            &CircuitDna::from_program(numeric.program())
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
        let manifest = root.join("manifest.json");
        save(&manifest,&serde_json::json!({"encoder":trained.join(format!("{}.axrp",layout.module())),"circuits":[{"id":"math-p032","file":circuit}]})).unwrap();
        let combined = root.join("combined");
        if layout.read().is_some() {
            let wrong = if layout == Layout::WorkingUniform {
                Layout::WorkingAttention
            } else {
                Layout::WorkingUniform
            };
            assert!(
                super::super::pointer_pack_layout(
                    &number,
                    &root.join("wrong-read-birth"),
                    &manifest,
                    &root.join("wrong-read-output"),
                    wrong
                )
                .is_err()
            );
            assert!(!root.join("wrong-read-birth").exists());
        }
        super::super::pointer_pack_layout(
            &number,
            &root.join("pack-birth"),
            &manifest,
            &combined,
            layout,
        )
        .unwrap();
        let child = combined.join("structural.axnum");
        let inputs = root.join("inputs.json");
        save(
            &inputs,
            &serde_json::json!([{"id":"own","text":"combine 2 with 11"}]),
        )
        .unwrap();
        let receipt = root.join("receipt.json");
        batch_layout(layout, &child, &root.join("fresh-birth"), &inputs, &receipt).unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert_eq!(rows[0]["answer"], 13.0);
        assert_eq!(rows[0]["trace"][0]["computed"], 13.0);
        assert_eq!(rows[0]["arithmetic_evaluations"], 1);
        assert_eq!(
            rows[0]["trace"][1]["left_origin"],
            serde_json::json!({"kind":"register","index":0})
        );
        assert_eq!(fs::read(number).unwrap(), original);
        let session_input = root.join("session.json");
        save(&session_input,&serde_json::json!({"batches":[[{"id":"own","text":"combine 2 with 11"}],[{"id":"own","text":"combine 2 with 11"}]]})).unwrap();
        let session_receipt = root.join("session-receipt.json");
        session(
            layout,
            false,
            &child,
            &root.join("session-birth"),
            &session_input,
            &session_receipt,
        )
        .unwrap();
        let session_rows: serde_json::Value =
            serde_json::from_slice(&fs::read(session_receipt).unwrap()).unwrap();
        assert_eq!(session_rows["batches"][0], rows);
        assert_eq!(session_rows["batches"][1], rows);
        let invalid_session = root.join("invalid-session.json");
        save(&invalid_session,&serde_json::json!({"batches":[[{"id":"ok","text":"combine 2 with 11"}],(0..5).map(|i|serde_json::json!({"id":i.to_string(),"text":"3"})).collect::<Vec<_>>()]})).unwrap();
        assert!(
            session(
                layout,
                false,
                &child,
                &root.join("invalid-session-birth"),
                &invalid_session,
                &root.join("invalid-session-output.json")
            )
            .is_err()
        );
        assert!(!root.join("invalid-session-birth").exists());
        let bad = root.join("bad.json");
        save(
            &bad,
            &serde_json::json!([{"id":"bad","text":"combine 2 with 11","expected":13}]),
        )
        .unwrap();
        assert!(
            batch_layout(
                layout,
                &child,
                &root.join("bad-birth"),
                &bad,
                &root.join("bad-output.json")
            )
            .is_err()
        );
        assert!(!root.join("bad-birth").exists());
    }
}
