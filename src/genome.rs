//! Self-contained mixed-domain inheritance. A genome is not an RNG seed or live experience.
use super::{controller::*, *};
use crate::behavior_lab::Rng;

const MAGIC: &[u8; 8] = b"AXGN0001";
pub const MAX_BYTES: usize = 262_144;
pub const MAX_DIGITS: usize = 1 + 3 * MAX_BYTES;
const CATALOG_FILE: &str = "birth.axseed";
const RECEIPT_FILE: &str = "birth.receipt";

#[derive(Clone, Debug)]
pub struct Genome {
    context: String,
    generation: u64,
    parent: u64,
    byte_cap: usize,
    residency: Seed,
    composition: CompositionBudget,
    catalog: Vec<u8>,
    payloads: Vec<Vec<u8>>,
    procedures: Vec<Vec<u8>>,
}

/// Host-owned ceilings, never authority encoded by an evolving individual.
#[derive(Clone, Copy)]
pub struct BirthLimits {
    pub bytes: usize,
    pub residency: Seed,
    pub composition: CompositionBudget,
}
impl Default for BirthLimits {
    fn default() -> Self {
        Self {
            bytes: MAX_BYTES,
            residency: Seed {
                active_bytes: 262_144,
                active_modules: 8,
                ..Seed::default()
            },
            composition: CompositionBudget::default(),
        }
    }
}

pub struct Birth {
    pub genome: u64,
    pub controller: GoalController,
}
fn blob(out: &mut Vec<u8>, value: &[u8]) {
    out.extend((value.len() as u32).to_le_bytes());
    out.extend(value);
}
fn read_blob(r: &mut Reader<'_>, limit: usize) -> Result<Vec<u8>, DemandError> {
    let len = u32::from_le_bytes(r.take(4)?.try_into().expect("width")) as usize;
    if len > limit {
        return Err(DemandError::Budget("genome section"));
    }
    Ok(r.take(len)?.to_vec())
}
pub(crate) fn bounded_path(path: &Path) -> Result<(), DemandError> {
    let absolute = std::path::absolute(path)?;
    let mut part = PathBuf::new();
    for component in absolute.components() {
        part.push(component);
        // A Windows drive/verbatim prefix alone is not a filesystem directory.
        if matches!(component, std::path::Component::Prefix(_)) {
            continue;
        }
        match std::fs::symlink_metadata(&part) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(DemandError::Invalid("genome symlink path"));
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
impl Genome {
    pub fn capture(
        context: &str,
        runtime: &mut DemandRuntime,
        composition: CompositionBudget,
        procedures: &[Procedure],
        byte_cap: usize,
        authority: &Authority,
    ) -> Result<Self, DemandError> {
        require(authority, Effect::ReadData)?;
        let mut payloads = Vec::new();
        let mut total = 0usize;
        for e in runtime.catalog().entries() {
            total = total
                .checked_add(e.encoded_bytes)
                .ok_or(DemandError::Budget("genome size"))?;
            if total > byte_cap.min(MAX_BYTES) {
                return Err(DemandError::Budget("genome payloads"));
            }
            let path = runtime.catalog().root().join(&e.file);
            bounded_path(&path)?;
            let bytes = read_bounded(&path, e.encoded_bytes as u64).map_err(storage)?;
            if checksum(&bytes) != e.identity {
                return Err(DemandError::Identity);
            }
            payloads.push(bytes);
        }
        for p in procedures {
            p.replay(runtime, authority)?;
        }
        let g = Self {
            context: context.into(),
            generation: 0,
            parent: 0,
            byte_cap,
            residency: runtime.seed(),
            composition,
            catalog: runtime.catalog().to_bytes(),
            payloads,
            procedures: procedures.iter().map(Procedure::to_bytes).collect(),
        };
        g.validate()?;
        Ok(g)
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn parent(&self) -> u64 {
        self.parent
    }
    pub fn composition(&self) -> CompositionBudget {
        self.composition
    }
    pub fn residency(&self) -> Seed {
        self.residency
    }
    pub fn module_count(&self) -> usize {
        self.payloads.len()
    }
    pub fn procedure_count(&self) -> usize {
        self.procedures.len()
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.to_bytes())
    }
    pub fn encoded_bytes(&self) -> usize {
        self.to_bytes().len()
    }
    fn catalog(&self, root: &Path) -> Result<Catalog, DemandError> {
        Catalog::from_bytes(&self.catalog, root)
    }
    pub fn procedures(&self) -> Result<Vec<Procedure>, DemandError> {
        let c = self.catalog(Path::new("."))?;
        self.procedures
            .iter()
            .map(|b| Procedure::from_bytes(b, &c))
            .collect()
    }
    fn validate(&self) -> Result<(), DemandError> {
        name(&self.context)?;
        self.residency.validate()?;
        self.composition.validate()?;
        if !(512..=MAX_BYTES).contains(&self.byte_cap)
            || self.to_bytes().len() > self.byte_cap
            || self.generation > 1_000_000
            || (self.generation == 0) != (self.parent == 0)
            || self.procedures.len() > 8
        {
            return Err(DemandError::Invalid("genome bounds/lineage"));
        }
        let catalog = self.catalog(Path::new("."))?;
        if catalog.entries.len() != self.payloads.len() || self.payloads.is_empty() {
            return Err(DemandError::Invalid("genome module count"));
        }
        let mut files = BTreeSet::from([CATALOG_FILE, RECEIPT_FILE]);
        for (e, bytes) in catalog.entries.iter().zip(&self.payloads) {
            if !files.insert(&e.file)
                || e.file.starts_with("procedure-")
                || bytes.len() != e.encoded_bytes
                || checksum(bytes) != e.identity
            {
                return Err(DemandError::Identity);
            }
            let m = Module::decode_reserved(e.kind, bytes, e.retained_bytes)?;
            if m.context() != e.context || m.retained_bytes() != e.retained_bytes {
                return Err(DemandError::Identity);
            }
        }
        let mut contexts = BTreeSet::new();
        for p in self.procedures()? {
            if !contexts.insert(p.context().to_owned()) {
                return Err(DemandError::Invalid("duplicate genome goal"));
            }
        }
        Ok(())
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        b.extend((self.context.len() as u16).to_le_bytes());
        b.extend(self.context.as_bytes());
        for v in [
            self.generation,
            self.parent,
            self.byte_cap as u64,
            self.residency.active_bytes as u64,
            self.residency.active_modules as u64,
            self.residency.max_rows as u64,
            self.residency.max_value_bytes as u64,
            self.residency.batch_bytes as u64,
            self.residency.max_plan_cost,
            self.composition.max_steps as u64,
            self.composition.max_candidates as u64,
            self.composition.max_work,
            self.composition.max_cost,
        ] {
            b.extend(v.to_le_bytes());
        }
        blob(&mut b, &self.catalog);
        b.extend((self.payloads.len() as u16).to_le_bytes());
        for value in &self.payloads {
            blob(&mut b, value);
        }
        b.push(self.procedures.len() as u8);
        for value in &self.procedures {
            blob(&mut b, value);
        }
        b.extend(checksum(&b).to_le_bytes());
        b
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, DemandError> {
        if b.len() < 16
            || b.len() > MAX_BYTES
            || &b[..8] != MAGIC
            || checksum(&b[..b.len() - 8])
                != u64::from_le_bytes(b[b.len() - 8..].try_into().expect("width"))
        {
            return Err(DemandError::Invalid("genome header/checksum"));
        }
        let mut r = Reader(&b[8..b.len() - 8]);
        let context = r.text()?;
        let generation = r.u64()?;
        let parent = r.u64()?;
        fn width(r: &mut Reader<'_>) -> Result<usize, DemandError> {
            usize::try_from(r.u64()?).map_err(|_| DemandError::Budget("genome width"))
        }
        let byte_cap = width(&mut r)?;
        let residency = Seed {
            active_bytes: width(&mut r)?,
            active_modules: width(&mut r)?,
            max_rows: width(&mut r)?,
            max_value_bytes: width(&mut r)?,
            batch_bytes: width(&mut r)?,
            max_plan_cost: r.u64()?,
        };
        let composition = CompositionBudget {
            max_steps: width(&mut r)?,
            max_candidates: width(&mut r)?,
            max_work: r.u64()?,
            max_cost: r.u64()?,
        };
        let catalog = read_blob(&mut r, MAX_BYTES)?;
        let count = r.u16()? as usize;
        if count > MAX_MODULES {
            return Err(DemandError::Budget("genome modules"));
        }
        let mut payloads = Vec::new();
        for _ in 0..count {
            payloads.push(read_blob(&mut r, MAX_BYTES)?);
        }
        let count = r.take(1)?[0] as usize;
        if count > 8 {
            return Err(DemandError::Budget("genome procedures"));
        }
        let mut procedures = Vec::new();
        for _ in 0..count {
            procedures.push(read_blob(&mut r, 65536)?);
        }
        if !r.0.is_empty() {
            return Err(DemandError::Invalid("trailing genome data"));
        }
        let g = Self {
            context,
            generation,
            parent,
            byte_cap,
            residency,
            composition,
            catalog,
            payloads,
            procedures,
        };
        g.validate()?;
        Ok(g)
    }
    /// Linear-time decimal bytecode: 1 followed by one canonical 000..255 word per byte.
    /// This is reversible representation, not compression or a floating-point number.
    pub fn number(&self) -> String {
        let b = self.to_bytes();
        let mut s = String::with_capacity(1 + b.len() * 3);
        s.push('1');
        for v in b {
            s.push(char::from(b'0' + v / 100));
            s.push(char::from(b'0' + v / 10 % 10));
            s.push(char::from(b'0' + v % 10));
        }
        s
    }
    pub fn from_number(s: &str) -> Result<Self, DemandError> {
        if s.len() > MAX_DIGITS
            || s.len() < 49
            || !s.starts_with('1')
            || !(s.len() - 1).is_multiple_of(3)
            || !s.bytes().all(|c| c.is_ascii_digit())
        {
            return Err(DemandError::Invalid("canonical genome number"));
        }
        let mut b = Vec::with_capacity((s.len() - 1) / 3);
        for word in s.as_bytes()[1..].as_chunks::<3>().0 {
            let v = u16::from(word[0] - b'0') * 100
                + u16::from(word[1] - b'0') * 10
                + u16::from(word[2] - b'0');
            b.push(u8::try_from(v).map_err(|_| DemandError::Invalid("decimal genome word"))?);
        }
        Self::from_bytes(&b)
    }
    pub fn open_number(path: &Path, authority: &Authority) -> Result<Self, DemandError> {
        require(authority, Effect::ReadData)?;
        bounded_path(path)?;
        let b = read_bounded(path, MAX_DIGITS as u64).map_err(storage)?;
        Self::from_number(
            std::str::from_utf8(&b).map_err(|_| DemandError::Invalid("number UTF-8"))?,
        )
    }
    pub fn save_number_new(&self, path: &Path, authority: &Authority) -> Result<(), DemandError> {
        require(authority, Effect::WriteData)?;
        bounded_path(path)?;
        self.validate()?;
        publish_new(path, self.number().as_bytes()).map_err(storage)
    }
    pub fn offspring(&self, random: u64) -> Result<Self, DemandError> {
        let mut g = self.clone();
        g.generation = g
            .generation
            .checked_add(1)
            .ok_or(DemandError::Budget("genome generation"))?;
        g.parent = self.fingerprint();
        let mut rng = Rng::new(random);
        // Local policy mutations. Work/growth caps, evidence and permissions never mutate.
        g.residency.active_modules = match rng.pick(3) {
            0 => g.residency.active_modules.saturating_sub(1).max(1),
            1 => (g.residency.active_modules + 1).min(8),
            _ => g.residency.active_modules,
        };
        g.composition.max_steps = match rng.pick(3) {
            0 => g.composition.max_steps.saturating_sub(1).max(4),
            1 => (g.composition.max_steps + 1).min(6),
            _ => g.composition.max_steps,
        };
        g.validate()?;
        Ok(g)
    }
    /// Consolidate only replayed procedures in a new immutable hereditary revision.
    /// Live cursor values and pending external feedback are deliberately not inherited.
    pub fn consolidate(
        &self,
        procedures: &[Procedure],
        runtime: &mut DemandRuntime,
        authority: &Authority,
    ) -> Result<Self, DemandError> {
        if runtime.catalog().to_bytes() != self.catalog {
            return Err(DemandError::Identity);
        }
        let mut g = self.clone();
        g.generation = g
            .generation
            .checked_add(1)
            .ok_or(DemandError::Budget("genome generation"))?;
        g.parent = self.fingerprint();
        for p in procedures {
            p.replay(runtime, authority)?;
            let index = g.procedures.iter().position(|b| {
                Procedure::from_bytes(b, runtime.catalog())
                    .is_ok_and(|old| old.context() == p.context())
            });
            if let Some(i) = index {
                g.procedures[i] = p.to_bytes();
            } else {
                g.procedures.push(p.to_bytes());
            }
        }
        g.validate()?;
        Ok(g)
    }
    /// Preserve old executable modules/goals; only a checked study descendant may replace its parent.
    pub fn inherit_runtime(
        &self,
        runtime: &mut DemandRuntime,
        new_goals: &[Procedure],
        authority: &Authority,
    ) -> Result<Self, DemandError> {
        require(authority, Effect::ReadData)?;
        let old = self.catalog(Path::new("."))?;
        for (entry, bytes) in old.entries.iter().zip(&self.payloads) {
            let new = runtime
                .catalog()
                .get(&entry.id)
                .ok_or(DemandError::Identity)?;
            if new.identity != entry.identity {
                if entry.kind != ModuleKind::Study
                    || new.kind != entry.kind
                    || new.context != entry.context
                    || new.input != entry.input
                    || new.output != entry.output
                    || new.cost != entry.cost
                {
                    return Err(DemandError::Identity);
                }
                let Module::Study(parent) = Module::decode(entry.kind, bytes)? else {
                    unreachable!()
                };
                let context = new.context.clone();
                runtime.with_module(&entry.id, &context, authority, |module| {
                    let Module::Study(child) = module else {
                        return Err(DemandError::Identity);
                    };
                    child
                        .check_descendant(&parent)
                        .map_err(|e| DemandError::Module(e.to_string()))
                })?;
            } else if new != entry {
                return Err(DemandError::Identity);
            }
        }
        let mut goals = self.procedures()?;
        for goal in new_goals {
            if let Some(old) = goals.iter().find(|old| old.context() == goal.context()) {
                if old.fingerprint() != goal.fingerprint() {
                    return Err(DemandError::Invalid("old goal replacement"));
                }
            } else {
                goals.push(goal.clone());
            }
        }
        let mut child = Self::capture(
            &self.context,
            runtime,
            self.composition,
            &goals,
            self.byte_cap,
            authority,
        )?;
        child.generation = self
            .generation
            .checked_add(1)
            .ok_or(DemandError::Budget("genome generation"))?;
        child.parent = self.fingerprint();
        child.validate()?;
        Ok(child)
    }
    fn check_limits(&self, host: BirthLimits) -> Result<(), DemandError> {
        host.residency.validate()?;
        host.composition.validate()?;
        if self.encoded_bytes() > host.bytes
            || self.residency.active_bytes > host.residency.active_bytes
            || self.residency.active_modules > host.residency.active_modules
            || self.residency.max_rows > host.residency.max_rows
            || self.residency.max_value_bytes > host.residency.max_value_bytes
            || self.residency.batch_bytes > host.residency.batch_bytes
            || self.residency.max_plan_cost > host.residency.max_plan_cost
            || self.composition.max_steps > host.composition.max_steps
            || self.composition.max_candidates > host.composition.max_candidates
            || self.composition.max_work > host.composition.max_work
            || self.composition.max_cost > host.composition.max_cost
        {
            return Err(DemandError::Budget("host birth ceilings"));
        }
        Ok(())
    }
    pub fn birth(
        &self,
        root: &Path,
        host: BirthLimits,
        authority: &Authority,
    ) -> Result<Birth, DemandError> {
        require(authority, Effect::ReadData)?;
        require(authority, Effect::WriteData)?;
        self.validate()?;
        self.check_limits(host)?;
        bounded_path(root)?;
        let catalog = self.catalog(root)?;
        let procedures = self.procedures()?;
        // A new directory is an installation, not an atomic multi-file transaction.
        // Partial births stay visible on failure; receipt is published only after evidence replay.
        std::fs::create_dir(root)?;
        for (e, b) in catalog.entries.iter().zip(&self.payloads) {
            publish_new(&root.join(&e.file), b).map_err(storage)?;
        }
        catalog.save_new(root.join(CATALOG_FILE), authority)?;
        let mut controller = GoalController::new(DemandRuntime::new(catalog, self.residency)?);
        for (i, p) in procedures.into_iter().enumerate() {
            p.save_new(&root.join(format!("procedure-{i}.axgoal")), authority)?;
            controller.install(p, authority)?;
        }
        let genome = self.fingerprint();
        publish_new(
            &root.join(RECEIPT_FILE),
            format!(
                "AXON_BIRTH_V1\ngenome={genome:016X}\nmodules={}\nprocedures={}\n",
                self.module_count(),
                self.procedure_count()
            )
            .as_bytes(),
        )
        .map_err(storage)?;
        Ok(Birth { genome, controller })
    }
}
