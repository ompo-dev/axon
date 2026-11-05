//! Bounded anti-unification of admitted programs; templates are not new VM opcodes.
use super::{Error, Expr as E, Meter, Program, Value};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Kind {
    Any,
    Integer,
    Boolean,
    List,
}

struct Hole {
    paths: Vec<Vec<usize>>,
    scope: u8,
    kind: Kind,
    left: E,
    right: E,
    alternatives: Vec<E>,
}

pub(super) struct Template {
    skeleton: E,
    holes: Vec<Hole>,
    fixed: usize,
    key: Vec<u8>,
}

fn children(e: &E) -> Vec<&E> {
    match e {
        E::Get(a, _) => vec![a],
        E::List(v) => v.iter().collect(),
        E::Add(a, b)
        | E::Subtract(a, b)
        | E::Multiply(a, b)
        | E::Remainder(a, b)
        | E::Equal(a, b)
        | E::Less(a, b)
        | E::Push(a, b)
        | E::Call(a, b) => vec![a, b],
        E::If(a, b, c) | E::Fold(a, b, c) => vec![a, b, c],
        _ => vec![],
    }
}

fn child_mut(e: &mut E, index: usize) -> &mut E {
    match e {
        E::Get(a, _) => a,
        E::List(v) => &mut v[index],
        E::Add(a, b)
        | E::Subtract(a, b)
        | E::Multiply(a, b)
        | E::Remainder(a, b)
        | E::Equal(a, b)
        | E::Less(a, b)
        | E::Push(a, b)
        | E::Call(a, b) => {
            if index == 0 {
                a
            } else {
                b
            }
        }
        E::If(a, b, c) | E::Fold(a, b, c) => match index {
            0 => a,
            1 => b,
            _ => c,
        },
        _ => unreachable!("paths originate from this skeleton"),
    }
}

fn replace(root: &mut E, path: &[usize], expression: &E) {
    if let Some((&index, tail)) = path.split_first() {
        replace(child_mut(root, index), tail, expression);
    } else {
        *root = expression.clone();
    }
}

fn compatible(a: &E, b: &E) -> bool {
    if std::mem::discriminant(a) != std::mem::discriminant(b) {
        return false;
    }
    match (a, b) {
        (E::Get(_, i), E::Get(_, j)) => i == j,
        (E::List(a), E::List(b)) => a.len() == b.len(),
        // Closed call bodies are immutable version witnesses, not open binders.
        (E::Call(a, _), E::Call(b, _)) => a == b,
        _ => !children(a).is_empty(),
    }
}

fn walk(
    a: &E,
    b: &E,
    scope: u8,
    kind: Kind,
    path: &mut Vec<usize>,
    holes: &mut Vec<Hole>,
    meter: &mut Meter,
) -> Result<bool, Error> {
    meter.charge((a.nodes() + b.nodes()) as u64)?;
    if a == b {
        return Ok(true);
    }
    if compatible(a, b) {
        let right = children(b);
        for (i, left) in children(a).into_iter().enumerate() {
            if matches!(a, E::Call(_, _)) && i == 0 {
                continue;
            }
            let expected = match a {
                E::If(_, _, _) if i == 0 => Kind::Boolean,
                E::Add(_, _)
                | E::Subtract(_, _)
                | E::Multiply(_, _)
                | E::Remainder(_, _)
                | E::Less(_, _) => Kind::Integer,
                E::Get(_, _) => Kind::List,
                E::Push(_, _) | E::Fold(_, _, _) if i == 0 => Kind::List,
                E::If(_, _, _) => kind,
                _ => Kind::Any,
            };
            let nested_scope = scope + u8::from(matches!(a, E::Fold(_, _, _)) && i == 2);
            path.push(i);
            let ok = walk(left, right[i], nested_scope, expected, path, holes, meter)?;
            path.pop();
            if !ok {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if let Some(hole) = holes
        .iter_mut()
        .find(|h| h.scope == scope && h.kind == kind && h.left == *a && h.right == *b)
    {
        hole.paths.push(path.clone());
    } else {
        if holes.len() == 2 {
            return Ok(false);
        }
        holes.push(Hole {
            paths: vec![path.clone()],
            scope,
            kind,
            left: a.clone(),
            right: b.clone(),
            alternatives: vec![a.clone(), b.clone()],
        });
    }
    Ok(true)
}

pub(super) fn mine(library: &[Program], meter: &mut Meter) -> Result<Vec<Template>, Error> {
    let mut unique: Vec<&Program> = Vec::new();
    for program in library {
        meter.charge(program.expression().nodes() as u64)?;
        if !unique.contains(&program) {
            unique.push(program);
        }
    }
    let mut seen: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
    let mut templates: Vec<Template> = Vec::new();
    for (i, a) in unique.iter().enumerate() {
        for b in &unique[i + 1..] {
            let mut holes = Vec::new();
            if !walk(
                a.expression(),
                b.expression(),
                0,
                Kind::Any,
                &mut vec![],
                &mut holes,
                meter,
            )? || holes.is_empty()
            {
                continue;
            }
            let removed: usize = holes.iter().map(|h| h.paths.len() * h.left.nodes()).sum();
            let fixed = a.expression().nodes() - removed;
            if fixed < 4 {
                continue;
            }
            let mut erased = a.expression().clone();
            for h in &holes {
                for path in &h.paths {
                    replace(&mut erased, path, &E::Empty);
                }
            }
            let descriptors: Vec<_> = holes.iter().map(|h| (&h.paths, h.scope, h.kind)).collect();
            let key = serde_json::to_vec(&(erased, descriptors))
                .map_err(|e| Error::Storage(e.to_string()))?;
            if let Some(&index) = seen.get(&key) {
                for (old, new) in templates[index].holes.iter_mut().zip(&holes) {
                    for expression in [&new.left, &new.right] {
                        meter.charge(expression.nodes() as u64)?;
                        if !old.alternatives.contains(expression) {
                            old.alternatives.push(expression.clone());
                        }
                    }
                }
            } else {
                seen.insert(key.clone(), templates.len());
                templates.push(Template {
                    skeleton: a.expression().clone(),
                    holes,
                    fixed,
                    key,
                });
            }
        }
    }
    for template in &mut templates {
        for hole in &mut template.holes {
            // Keep a compact, deterministic vocabulary from every corroborating pair.
            let mut ranked = hole
                .alternatives
                .drain(..)
                .map(|e| {
                    let key = serde_json::to_vec(&e).map_err(|e| Error::Storage(e.to_string()))?;
                    Ok((e.nodes(), key, e))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            hole.alternatives = ranked.into_iter().take(8).map(|(_, _, e)| e).collect();
        }
    }
    templates.sort_by(|a, b| {
        b.fixed
            .cmp(&a.fixed)
            .then_with(|| a.holes.len().cmp(&b.holes.len()))
            .then_with(|| a.key.cmp(&b.key))
    });
    templates.truncate(64);
    Ok(templates)
}

fn possible(e: &E, kind: Kind) -> bool {
    let actual = match e {
        E::Literal(Value::Integer(_))
        | E::Add(_, _)
        | E::Subtract(_, _)
        | E::Multiply(_, _)
        | E::Remainder(_, _) => Kind::Integer,
        E::Literal(Value::Boolean(_)) | E::Equal(_, _) | E::Less(_, _) => Kind::Boolean,
        E::Literal(Value::List(_)) | E::List(_) | E::Empty | E::Push(_, _) => Kind::List,
        E::Literal(Value::Text(_)) => return kind == Kind::Any,
        _ => Kind::Any,
    };
    kind == Kind::Any || actual == Kind::Any || kind == actual
}

impl Template {
    pub(super) fn choices(
        &self,
        values: &[E],
        predicates: &[E],
        meter: &mut Meter,
    ) -> Result<Vec<Vec<E>>, Error> {
        let mut result = Vec::new();
        for hole in &self.holes {
            let mut choices = Vec::new();
            let mut local = hole.alternatives.clone();
            for a in &hole.alternatives {
                for b in &hole.alternatives {
                    for operation in [
                        E::Add,
                        E::Subtract,
                        E::Multiply,
                        E::Remainder,
                        E::Equal,
                        E::Less,
                    ] {
                        local.push(operation(Box::new(a.clone()), Box::new(b.clone())));
                    }
                }
            }
            for e in local.iter().chain(values).chain(predicates) {
                meter.charge(e.nodes() as u64)?;
                if possible(e, hole.kind)
                    && Program::validate_at_scope(e, hole.scope).is_ok()
                    && !choices.contains(e)
                {
                    choices.push(e.clone());
                }
                if choices.len() == 1024 {
                    break;
                }
            }
            result.push(choices);
        }
        Ok(result)
    }

    pub(super) fn instantiate(&self, replacements: &[&E]) -> E {
        let mut expression = self.skeleton.clone();
        for (hole, replacement) in self.holes.iter().zip(replacements) {
            for path in &hole.paths {
                replace(&mut expression, path, replacement);
            }
        }
        expression
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn repeated(n: i64) -> Program {
        let part = E::Multiply(
            Box::new(E::Item(0)),
            Box::new(E::Literal(Value::Integer(n))),
        );
        Program::new(E::Fold(
            Box::new(E::Input),
            Box::new(E::Empty),
            Box::new(E::Push(
                Box::new(E::Acc(0)),
                Box::new(E::Add(Box::new(part.clone()), Box::new(part))),
            )),
        ))
        .unwrap()
    }
    #[test]
    fn repeated_holes_share_one_binding_and_preserve_scope() {
        let mut meter = Meter::new(100_000);
        let templates = mine(&[repeated(2), repeated(3)], &mut meter).unwrap();
        assert_eq!(templates.len(), 1);
        assert_eq!(templates[0].holes.len(), 1);
        assert_eq!(templates[0].holes[0].paths.len(), 2);
        let choices = templates[0]
            .choices(&[E::Item(0), E::Item(1)], &[], &mut meter)
            .unwrap();
        assert!(choices[0].contains(&E::Item(0)));
        assert!(!choices[0].contains(&E::Item(1)));
        let program =
            Program::new(templates[0].instantiate(&[&E::Literal(Value::Integer(5))])).unwrap();
        assert_eq!(
            program
                .evaluate(&Value::List(vec![Value::Integer(2)]), 1000)
                .unwrap()
                .value,
            Value::List(vec![Value::Integer(20)])
        );
    }

    #[test]
    fn closed_call_bodies_are_never_opened_by_template_substitution() {
        let source = |n| {
            Program::new(E::Fold(
                Box::new(E::Input),
                Box::new(E::Empty),
                Box::new(E::Push(
                    Box::new(E::Acc(0)),
                    Box::new(E::Call(
                        Box::new(E::Add(
                            Box::new(E::Input),
                            Box::new(E::Literal(Value::Integer(n))),
                        )),
                        Box::new(E::Item(0)),
                    )),
                )),
            ))
            .unwrap()
        };
        let templates = mine(&[source(2), source(3)], &mut Meter::new(100_000)).unwrap();
        assert_eq!(templates.len(), 1);
        assert_eq!(templates[0].holes[0].paths, vec![vec![2, 1]]);
        let copied = Program::new(templates[0].instantiate(&[&E::Item(0)])).unwrap();
        assert!(copied.captured_calls().is_empty());
        assert_eq!(
            copied
                .evaluate(&Value::List(vec![Value::Integer(7)]), 1000)
                .unwrap()
                .value,
            Value::List(vec![Value::Integer(7)])
        );
    }
}
