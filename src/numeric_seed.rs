//! Heritable numerical development policy and acquired programs, not a general brain.
use super::{
    NumericError, Reader,
    evolution::{EvolutionBudget, EvolvedSkill},
    identifier,
};
use crate::{
    behavior_lab::Rng,
    live_store::checksum,
    text_skill::{publish_new, read_bounded},
};
use std::{collections::BTreeSet, path::Path};

const MAGIC: &[u8; 8] = b"AXDNA001";
const FILE_LIMIT: u64 = 262_144;
const MAX_SKILLS: usize = 8;

#[derive(Clone, Debug)]
pub struct DevelopmentSeed {
    context: String,
    policy: EvolutionBudget,
    byte_cap: usize,
    generation: u64,
    parents: [u64; 2],
    skills: Vec<EvolvedSkill>,
}
impl DevelopmentSeed {
    pub fn new(
        context: &str,
        policy: EvolutionBudget,
        byte_cap: usize,
    ) -> Result<Self, NumericError> {
        let seed = Self {
            context: context.into(),
            policy,
            byte_cap,
            generation: 0,
            parents: [0; 2],
            skills: Vec::new(),
        };
        seed.validate()?;
        Ok(seed)
    }
    fn validate(&self) -> Result<(), NumericError> {
        identifier(&self.context)?;
        self.policy.validate()?;
        if !(512..=FILE_LIMIT as usize).contains(&self.byte_cap)
            || self.skills.len() > MAX_SKILLS
            || self.generation > 1_000_000
            || (self.generation == 0 && self.parents != [0; 2])
            || (self.generation > 0 && self.parents[0] == 0)
        {
            return Err(NumericError::Invalid("development seed schema/lineage"));
        }
        let unique: BTreeSet<_> = self.skills.iter().map(EvolvedSkill::fingerprint).collect();
        if unique.len() != self.skills.len() || self.to_bytes().len() > self.byte_cap {
            return Err(NumericError::Invalid("development seed growth/duplicates"));
        }
        Ok(())
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn policy(&self) -> EvolutionBudget {
        self.policy
    }
    pub fn byte_cap(&self) -> usize {
        self.byte_cap
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn parents(&self) -> [u64; 2] {
        self.parents
    }
    pub fn skills(&self) -> &[EvolvedSkill] {
        &self.skills
    }
    pub fn budget(&self, work_cap: u64) -> Result<EvolutionBudget, NumericError> {
        if work_cap == 0 || work_cap > 200_000_000 {
            return Err(NumericError::Invalid("host work cap"));
        }
        Ok(EvolutionBudget {
            max_work: self.policy.max_work.min(work_cap),
            ..self.policy
        })
    }
    fn offspring(
        &self,
        policy: EvolutionBudget,
        skills: Vec<EvolvedSkill>,
        second: Option<&Self>,
    ) -> Result<Self, NumericError> {
        let child = Self {
            context: self.context.clone(),
            policy,
            byte_cap: self.byte_cap,
            generation: self
                .generation
                .max(second.map_or(0, |s| s.generation))
                .saturating_add(1),
            parents: [self.revision_id(), second.map_or(0, Self::revision_id)],
            skills,
        };
        child.validate()?;
        Ok(child)
    }
    /// Growth never replaces an inherited skill, and overflow leaves the parent unchanged.
    pub fn extend(&self, context: &str, skill: EvolvedSkill) -> Result<Self, NumericError> {
        if context != self.context {
            return Err(NumericError::Context);
        }
        if self
            .skills
            .iter()
            .any(|s| s.fingerprint() == skill.fingerprint())
        {
            return Err(NumericError::Invalid("duplicate acquired skill"));
        }
        let mut skills = self.skills.clone();
        skills.push(skill);
        self.offspring(self.policy, skills, None)
    }
    /// Mutates search traits only. Evidence, programs, tolerance and hard caps are immutable.
    pub fn mutate(&self, context: &str, random_seed: u64) -> Result<Self, NumericError> {
        if context != self.context {
            return Err(NumericError::Context);
        }
        let mut rng = Rng::new(random_seed);
        let mut p = self.policy;
        match rng.pick(8) {
            0 => p.population = if p.population == 32 { 64 } else { 32 },
            1 => p.genes = if p.genes == 24 { 32 } else { 24 },
            2 => p.mutations = p.mutations % 8 + 1,
            3 => p.local_refinement = !p.local_refinement,
            4 => p.refine_inherited_calls = !p.refine_inherited_calls,
            5 => {
                p.refinement_share_percent = if p.refinement_share_percent == 25 {
                    50
                } else {
                    25
                }
            }
            6 => {
                p.refinement_slice_work = if p.refinement_slice_work == 1_000_000 {
                    4_000_000
                } else {
                    1_000_000
                }
            }
            _ => p.generations = if p.generations == 1000 { 2000 } else { 1000 },
        }
        self.offspring(p, self.skills.clone(), None)
    }
    /// Policies recombine; the ordered union keeps every acquired skill or refuses growth.
    pub fn crossover(
        &self,
        context: &str,
        other: &Self,
        random_seed: u64,
    ) -> Result<Self, NumericError> {
        if context != self.context || other.context != self.context {
            return Err(NumericError::Context);
        }
        if self.byte_cap != other.byte_cap || self.policy.max_work != other.policy.max_work {
            return Err(NumericError::Invalid("incompatible seed caps"));
        }
        let mut rng = Rng::new(random_seed);
        let mut p = self.policy;
        macro_rules! inherit {
            ($field:ident) => {
                if rng.pick(2) == 0 {
                    p.$field = other.policy.$field;
                }
            };
        }
        inherit!(population);
        inherit!(generations);
        inherit!(genes);
        inherit!(mutations);
        inherit!(local_refinement);
        inherit!(refine_inherited_calls);
        inherit!(refinement_slice_work);
        inherit!(refinement_share_percent);
        let mut skills = self.skills.clone();
        for skill in &other.skills {
            if !skills
                .iter()
                .any(|s| s.fingerprint() == skill.fingerprint())
            {
                skills.push(skill.clone());
            }
        }
        self.offspring(p, skills, Some(other))
    }
    pub fn fingerprint(&self) -> u64 {
        let mut bytes = Vec::new();
        self.write_traits(&mut bytes);
        for skill in &self.skills {
            bytes.extend(skill.fingerprint().to_le_bytes());
        }
        checksum(&bytes)
    }
    pub fn revision_id(&self) -> u64 {
        let mut bytes = self.fingerprint().to_le_bytes().to_vec();
        bytes.extend(self.generation.to_le_bytes());
        for parent in self.parents {
            bytes.extend(parent.to_le_bytes());
        }
        checksum(&bytes)
    }
    fn write_traits(&self, bytes: &mut Vec<u8>) {
        bytes.extend((self.context.len() as u32).to_le_bytes());
        bytes.extend(self.context.as_bytes());
        let p = self.policy;
        for value in [
            p.population as u64,
            p.generations as u64,
            p.genes as u64,
            p.mutations as u64,
            p.max_work,
            p.refinement_slice_work,
            self.byte_cap as u64,
        ] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend([
            u8::from(p.local_refinement),
            p.refinement_share_percent,
            u8::from(p.refine_inherited_calls),
        ]);
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        self.write_traits(&mut bytes);
        bytes.extend(self.generation.to_le_bytes());
        for parent in self.parents {
            bytes.extend(parent.to_le_bytes());
        }
        bytes.extend((self.skills.len() as u32).to_le_bytes());
        for skill in &self.skills {
            let payload = skill.to_bytes();
            bytes.extend((payload.len() as u32).to_le_bytes());
            bytes.extend(payload);
        }
        bytes.extend(checksum(&bytes).to_le_bytes());
        bytes
    }
    pub fn save_new(&self, path: &Path) -> Result<(), NumericError> {
        self.validate()?;
        publish_new(path, &self.to_bytes()).map_err(Into::into)
    }
    pub fn open(path: &Path, context: &str) -> Result<Self, NumericError> {
        let seed = Self::from_bytes(&read_bounded(path, FILE_LIMIT)?)?;
        if context != seed.context {
            return Err(NumericError::Context);
        }
        Ok(seed)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() < 16 || bytes.len() as u64 > FILE_LIMIT || &bytes[..8] != MAGIC {
            return Err(NumericError::Invalid("development seed header"));
        }
        let end = bytes.len() - 8;
        if checksum(&bytes[..end]) != u64::from_le_bytes(bytes[end..].try_into().expect("eight")) {
            return Err(NumericError::Invalid("development seed checksum"));
        }
        let mut reader = Reader(&bytes[8..end]);
        let context = reader.text()?;
        let mut sized = || {
            usize::try_from(reader.u64()?)
                .map_err(|_| NumericError::Invalid("development trait width"))
        };
        let population = sized()?;
        let generations = sized()?;
        let genes = sized()?;
        let mutations = sized()?;
        let max_work = reader.u64()?;
        let refinement_slice_work = reader.u64()?;
        let byte_cap =
            usize::try_from(reader.u64()?).map_err(|_| NumericError::Invalid("growth width"))?;
        let local = reader.byte()?;
        let refinement_share_percent = reader.byte()?;
        let calls = reader.byte()?;
        if local > 1 || calls > 1 {
            return Err(NumericError::Invalid("development flags"));
        }
        let policy = EvolutionBudget {
            population,
            generations,
            genes,
            mutations,
            max_work,
            refinement_slice_work,
            local_refinement: local == 1,
            refinement_share_percent,
            refine_inherited_calls: calls == 1,
        };
        policy.validate()?;
        let generation = reader.u64()?;
        let parents = [reader.u64()?, reader.u64()?];
        let mut skills = Vec::new();
        for _ in 0..reader.count(MAX_SKILLS)? {
            let length = reader.count(65_536)?;
            skills.push(EvolvedSkill::from_bytes(reader.take(length)?)?);
        }
        if !reader.0.is_empty() {
            return Err(NumericError::Invalid("development trailing data"));
        }
        let seed = Self {
            context,
            policy,
            byte_cap,
            generation,
            parents,
            skills,
        };
        seed.validate()?;
        Ok(seed)
    }
}

/// Development scores only. Do not feed final reserve labels into this selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedScore {
    pub capabilities: u64,
    pub tasks: u8,
    pub retention_ok: bool,
    pub violations: u32,
    pub work: u64,
    pub bytes: u64,
    pub instructions: u64,
}
impl SeedScore {
    fn validate(self) -> Result<(), NumericError> {
        if self.tasks == 0
            || self.tasks > 64
            || (self.tasks < 64 && self.capabilities >> self.tasks != 0)
            || self.bytes == 0
            || (self.capabilities != 0 && self.instructions == 0)
        {
            return Err(NumericError::Invalid("seed score schema"));
        }
        Ok(())
    }
    pub fn feasible(self) -> bool {
        self.retention_ok && self.violations == 0 && self.capabilities != 0
    }
    fn dominates(self, other: Self) -> bool {
        self.capabilities & other.capabilities == other.capabilities
            && self.work <= other.work
            && self.bytes <= other.bytes
            && self.instructions <= other.instructions
            && (self.capabilities != other.capabilities
                || self.work < other.work
                || self.bytes < other.bytes
                || self.instructions < other.instructions)
    }
}
/// Equal capability masks with different tradeoffs remain eligible; input order is stable.
pub fn pareto_front(scores: &[SeedScore]) -> Result<Vec<usize>, NumericError> {
    if scores.is_empty() || scores.len() > 128 {
        return Err(NumericError::Invalid("seed population size"));
    }
    for score in scores {
        score.validate()?;
        if score.tasks != scores[0].tasks {
            return Err(NumericError::Invalid("mixed seed score tasks"));
        }
    }
    Ok((0..scores.len())
        .filter(|&i| {
            scores[i].feasible()
                && !(0..scores.len())
                    .any(|j| j != i && scores[j].feasible() && scores[j].dominates(scores[i]))
        })
        .collect())
}
