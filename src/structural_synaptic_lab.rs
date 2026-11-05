//! Vector synaptic mechanism assay through the real genome and demand runtime.
use super::*;
use crate::live_store::checksum;
use crate::structural::sequence::lexical_encoder::synaptic::{self, Genes, Mode, Model};
const LIVE: usize = 262144;
const WORK: usize = 2_000_000;
const SESSION_WORK: usize = 500_000_000;

pub fn seed(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    for p in [number, birth, data, output] {
        bounded_path(p)?;
    }
    if output.exists() || birth.exists() {
        return Err("synaptic output/birth already exists".into());
    }
    let genes: Genes = serde_json::from_slice(&read_bounded(data, 65536)?)?;
    let model = Model::new(genes)?;
    let bytes = model.to_bytes()?;
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let parent = Genome::open_number(number, &auth)?;
    let born = parent.birth(birth, BirthLimits::default(), &auth)?;
    if born
        .controller
        .runtime()
        .catalog()
        .get(synaptic::MODULE)
        .is_some()
    {
        return Err("synaptic module already installed".into());
    }
    fs::create_dir(output)?;
    let mut catalog = Catalog::new(output);
    for entry in born.controller.runtime().catalog().entries() {
        let page = read_bounded(&birth.join(&entry.file), 262144)?;
        publish_new(&output.join(&entry.file), &page)?;
        catalog.register(
            ModuleSpec {
                id: &entry.id,
                kind: entry.kind,
                file: &entry.file,
                input: &entry.input,
                output: &entry.output,
                cost: entry.cost,
            },
            &auth,
        )?;
    }
    let file = "bdh-synaptic.axsyn";
    publish_new(&output.join(file), &bytes)?;
    catalog.register(
        ModuleSpec {
            id: synaptic::MODULE,
            kind: ModuleKind::Synaptic,
            file,
            input: "vector",
            output: "latent-state",
            cost: 1,
        },
        &auth,
    )?;
    let mut runtime = DemandRuntime::new(catalog, parent.residency())?;
    let child = parent.inherit_runtime(&mut runtime, &[], &auth)?;
    child.save_number_new(&output.join("structural.axnum"), &auth)?;
    if original != read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? {
        return Err("synaptic parent changed".into());
    }
    let parameters = model.reference_parameters();
    save(
        &output.join("model-spec.json"),
        &serde_json::json!({"revision":1,"genes":genes,"encoder":parameters[0],"encoder_v":parameters[1],"decoder":parameters[2],"slow_training":false,"reference":"Pathway BDH-GPU cell inference; vector-input quantized one-head adaptation, not pretrained BDH language weights"}),
    )?;
    let report = serde_json::json!({"revision":1,"parent_unchanged":true,"old_modules_preserved":parent.module_count(),"modules":child.module_count(),"encoded_bytes":child.encoded_bytes(),"model_bytes":bytes.len(),"model_retained_bytes":model.retained_bytes(),"private_state_in_genome":false,"generation":child.generation(),"genes":genes});
    save(&output.join("seed-report.json"), &report)?;
    println!("{report}");
    Ok(())
}
fn default_work() -> usize {
    WORK
}
fn default_live() -> usize {
    LIVE
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    episodes: Vec<Episode>,
    #[serde(default = "default_work")]
    max_work: usize,
    #[serde(default = "default_live")]
    max_live_bytes: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Episode {
    scope: String,
    mode: Mode,
    #[serde(default)]
    resume: Option<String>,
    events: Vec<Event>,
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
enum Event {
    Ingest {
        id: String,
        vector: Vec<f32>,
    },
    Learn {
        id: String,
        key: Vec<f32>,
        value: Vec<f32>,
    },
    Query {
        id: String,
        key: Vec<f32>,
    },
    Checkpoint {
        id: String,
        name: String,
    },
}
impl Event {
    fn id(&self) -> &str {
        match self {
            Self::Ingest { id, .. }
            | Self::Learn { id, .. }
            | Self::Query { id, .. }
            | Self::Checkpoint { id, .. } => id,
        }
    }
}
fn validate(input: &Input) -> Result<()> {
    if input.episodes.is_empty()
        || input.episodes.len() > 128
        || input.max_work == 0
        || input.max_work > WORK
        || input.max_live_bytes == 0
        || input.max_live_bytes > LIVE
    {
        return Err("synaptic input quota".into());
    }
    let mut scopes = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut count = 0;
    for ep in &input.episodes {
        if ep.scope.is_empty()
            || ep.scope.len() > 96
            || !ep
                .scope
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-./".contains(&b))
            || !scopes.insert(&ep.scope)
            || ep.events.is_empty()
            || ep.events.len() > 512
        {
            return Err("synaptic episode identity/quota".into());
        }
        if let Some(resume) = &ep.resume {
            bounded_path(Path::new(resume))?;
        }
        let mut ids = BTreeSet::new();
        for event in &ep.events {
            if event.id().is_empty() || event.id().len() > 96 || !ids.insert(event.id()) {
                return Err("synaptic event identity".into());
            }
            count += 1;
            let vectors = match event {
                Event::Ingest { vector, .. } if ep.mode == Mode::Sequence => vec![vector],
                Event::Learn { key, value, .. } if ep.mode == Mode::Association => vec![key, value],
                Event::Query { key, .. } if ep.mode == Mode::Association => vec![key],
                Event::Checkpoint { name, .. } => {
                    crate::demand::file_name(name)?;
                    if name.len() > 64 || !name.ends_with(".axss") || !names.insert(name) {
                        return Err("synaptic checkpoint name".into());
                    }
                    Vec::new()
                }
                _ => return Err("synaptic event mode".into()),
            };
            if vectors.iter().any(|v| {
                v.is_empty() || v.len() > 32 || v.iter().any(|n| !n.is_finite() || n.abs() > 1e6)
            }) {
                return Err("synaptic vector bounds".into());
            }
        }
    }
    if count > 8192 {
        return Err("synaptic session event quota".into());
    }
    Ok(())
}
pub fn session(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    for p in [number, birth, data, output] {
        bounded_path(p)?;
    }
    if output.exists() || birth.exists() {
        return Err("synaptic output/birth already exists".into());
    }
    let input: Input = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    validate(&input)?;
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    fs::create_dir(output)?;
    let start = Instant::now();
    let mut rows = Vec::new();
    let mut charged = 0usize;
    let mut model_hash = None;
    let mut telemetry = LabTelemetry::from_environment("bdh-synaptic-v1");
    for ep in &input.episodes {
        let result=born.controller.runtime_mut().with_module(synaptic::MODULE,synaptic::CONTEXT,&auth,|page|{Ok((||->Result<()>{
            let Module::Synaptic(model)=page else{return Err("synaptic module kind".into());};
            let before_model=model.to_bytes()?;let model_checksum=checksum(&before_model);if model_hash.is_some_and(|old|old!=model_checksum){return Err("synaptic model identity changed".into());}model_hash=Some(model_checksum);
            let live=input.max_live_bytes.checked_sub(page.retained_bytes()).ok_or("synaptic module live quota")?;
            let mut state=if let Some(path)=&ep.resume{model.restore(&ep.scope,ep.mode,&read_bounded(Path::new(path),196608)?)?}else{model.begin(&ep.scope,ep.mode)?};
            for event in &ep.events{
                if charged.checked_add(input.max_work).is_none_or(|v|v>SESSION_WORK){return Err("synaptic session work quota".into());}
                let before=model.snapshot(&state)?;
                if let Event::Checkpoint{id,name}=event{charged=charged.checked_add(before.len()).ok_or("synaptic work overflow")?;publish_new(&output.join(name),&before)?;rows.push(serde_json::json!({"scope":ep.scope,"id":id,"op":"checkpoint","file":name,"snapshot_bytes":before.len(),"state_checksum":format!("{:016X}",checksum(&before)),"position":state.position()}));continue;}
                let transition=match event{Event::Ingest{vector,..}=>model.step(&mut state,vector,input.max_work,live),Event::Learn{key,value,..}=>model.associate(&mut state,key,value,input.max_work,live),Event::Query{key,..}=>model.query(&state,key,input.max_work,live),Event::Checkpoint{..}=>unreachable!()};
                let after=model.snapshot(&state)?;
                match transition{Ok(t)=>{charged+=t.work;telemetry.emit("synaptic_transition",&[("method",V::Text("BDH::transition")),("scope",V::Text(&ep.scope)),("case",V::Text(event.id())),("position",V::Count(t.position as u64)),("synapses_written",V::Count(t.synapses_written as u64)),("work",V::Count(t.work as u64)),("live_bytes",V::Count((t.live_bytes+page.retained_bytes()) as u64))])?;rows.push(serde_json::json!({"scope":ep.scope,"id":event.id(),"transition":t,"state_unchanged":before==after,"state_checksum":format!("{:016X}",checksum(&after)),"module_retained_bytes":page.retained_bytes()}));},Err(e)=>{charged+=input.max_work;if before!=after{return Err("synaptic failure mutated state".into());}rows.push(serde_json::json!({"scope":ep.scope,"id":event.id(),"error":e.to_string(),"state_unchanged":true,"charged_work_upper_bound":input.max_work}));}}
            }if before_model!=model.to_bytes()?{return Err("synaptic operation mutated hereditary model".into());}Ok(())
        })())})?;
        result?;
    }
    if original != read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? {
        return Err("synaptic operation mutated number".into());
    }
    let errors = rows.iter().filter(|r| r.get("error").is_some()).count();
    let receipt = serde_json::json!({"revision":1,"episodes":input.episodes.len(),"events":rows.len(),"errors":errors,"rows":rows,"charged_work":charged,"module_loads":born.controller.runtime().stats().loads,"model_checksum":model_hash.map(|h|format!("{h:016X}")),"hereditary_number_unchanged":true,"slow_weights_unchanged":true,"private_state_in_genome":false,"duration_ms":start.elapsed().as_secs_f64()*1000.0,"scope":"Real vector synaptic mechanism; duration is whole native session, work is counted scalar operations, core live bytes are model/state/transition scratch estimates; exclude codec/checkpoint audit, JSON, IO and process RSS. No measured energy."});
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    if bytes.len() > 64_000_000 {
        return Err("synaptic receipt quota".into());
    }
    publish_new(&output.join("receipt.json"), &bytes)?;
    println!(
        "{}",
        serde_json::json!({"episodes":input.episodes.len(),"events":rows.len(),"errors":errors,"charged_work":charged,"receipt":output.join("receipt.json")})
    );
    Ok(())
}
