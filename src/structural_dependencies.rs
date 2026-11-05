//! Version witnesses for closed learned calls. No authority is inherited here.
use super::{Error, Rule, StructuralSeed, algorithm::Program};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Dependency {
    pub rule: String,
    pub expected: Program,
}

impl StructuralSeed {
    pub(super) fn dependencies_for(&self, program: &Program) -> Result<Vec<Dependency>, Error> {
        let mut result: Vec<Dependency> = Vec::new();
        for call in program.captured_calls() {
            let mut found = false;
            for source in &self.rules {
                if let Some(p) = source.algorithm_program()
                    && p.expression() == call
                {
                    found = true;
                    if !result.iter().any(|d| d.rule == source.id()) {
                        result.push(Dependency {
                            rule: source.id().into(),
                            expected: p.clone(),
                        });
                    }
                }
            }
            if !found {
                return Err(Error::Invalid("closed call has no source version witness"));
            }
            if result.len() > 32 {
                return Err(Error::Budget);
            }
        }
        result.sort_by(|a, b| a.rule.cmp(&b.rule));
        Ok(result)
    }

    pub(super) fn validate_dependencies(&self, strict: bool) -> Result<(), Error> {
        for rule in &self.rules {
            if rule.dependencies.len() > 32 {
                return Err(Error::Budget);
            }
            let programs = rule.algorithm_programs();
            if programs.is_empty() {
                if !rule.dependencies.is_empty() {
                    return Err(Error::Invalid("nonprocedural dependency"));
                }
                continue;
            }
            let calls: Vec<_> = programs
                .into_iter()
                .flat_map(Program::captured_calls)
                .collect();
            let mut seen = BTreeSet::new();
            for dependency in &rule.dependencies {
                if dependency.rule == rule.id() || !seen.insert(&dependency.rule) {
                    return Err(Error::Invalid("self or duplicate dependency"));
                }
                let source = self
                    .rules
                    .iter()
                    .find(|r| r.id() == dependency.rule)
                    .and_then(Rule::algorithm_program)
                    .ok_or(Error::Invalid("missing dependency"))?;
                if source != &dependency.expected {
                    return Err(Error::Invalid("stale dependency version"));
                }
                if !calls.contains(&dependency.expected.expression()) {
                    return Err(Error::Invalid("dependency is not a captured call"));
                }
            }
            if strict
                && calls.iter().any(|call| {
                    !rule
                        .dependencies
                        .iter()
                        .any(|d| d.expected.expression() == *call)
                })
            {
                return Err(Error::Invalid("untracked closed call"));
            }
        }
        fn visit<'a>(
            seed: &'a StructuralSeed,
            id: &'a str,
            active: &mut BTreeSet<&'a str>,
            done: &mut BTreeSet<&'a str>,
        ) -> Result<(), Error> {
            if done.contains(id) {
                return Ok(());
            }
            if !active.insert(id) {
                return Err(Error::Invalid("cyclic dependencies"));
            }
            let rule = seed
                .rules
                .iter()
                .find(|r| r.id() == id)
                .ok_or(Error::Invalid("missing dependency"))?;
            for dependency in &rule.dependencies {
                visit(seed, &dependency.rule, active, done)?;
            }
            active.remove(id);
            done.insert(id);
            Ok(())
        }
        let mut active = BTreeSet::new();
        let mut done = BTreeSet::new();
        for rule in &self.rules {
            visit(self, rule.id(), &mut active, &mut done)?;
        }
        Ok(())
    }

    pub(super) fn prepare_algorithm_dependencies(&mut self) -> Result<(), Error> {
        self.validate_dependencies(false)?;
        let mut additions = Vec::new();
        for (i, rule) in self.rules.iter().enumerate() {
            if rule.dependencies.is_empty() {
                let mut dependencies: Vec<Dependency> = Vec::new();
                for program in rule.algorithm_programs() {
                    for dependency in self.dependencies_for(program)? {
                        if !dependencies.iter().any(|d| d.rule == dependency.rule) {
                            dependencies.push(dependency);
                        }
                    }
                }
                if dependencies.len() > 32 {
                    return Err(Error::Budget);
                }
                dependencies.sort_by(|a, b| a.rule.cmp(&b.rule));
                additions.push((i, dependencies));
            }
        }
        for (i, dependencies) in additions {
            self.rules[i].dependencies = dependencies;
        }
        self.validate_dependencies(true)
    }

    pub(super) fn remove_algorithm_family(&mut self, id: &str) -> Vec<String> {
        let mut affected = BTreeSet::from([id.to_owned()]);
        loop {
            let mut changed = false;
            for rule in &self.rules {
                if rule.dependencies.iter().any(|d| affected.contains(&d.rule)) {
                    changed |= affected.insert(rule.id().to_owned());
                }
            }
            if !changed {
                break;
            }
        }
        let removed = self
            .rules
            .iter()
            .filter(|r| affected.contains(r.id()))
            .map(|r| r.id().to_owned())
            .collect();
        self.rules.retain(|r| !affected.contains(r.id()));
        removed
    }
}
