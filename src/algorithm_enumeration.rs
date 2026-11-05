//! Observational equivalence prunes search, not the independent admission gate.
use super::{E, Error, Program, Search, SearchReceipt, V, boxed, call, int};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
struct Node {
    expression: E,
    inherited: bool,
    priority: u8,
}
type Pool = BTreeMap<Vec<V>, Node>;

fn goals(values: Vec<V>, depth: usize, out: &mut Vec<Vec<V>>) {
    if depth > 6 || out.len() >= 128 || out.contains(&values) {
        return;
    }
    out.push(values.clone());
    let lists: Option<Vec<_>> = values
        .iter()
        .map(|v| if let V::List(v) = v { Some(v) } else { None })
        .collect();
    let Some(lists) = lists else {
        return;
    };
    if lists.iter().all(|v| v.len() == lists[0].len()) && lists[0].len() <= 4 {
        for i in 0..lists[0].len() {
            goals(lists.iter().map(|v| v[i].clone()).collect(), depth + 1, out);
        }
    }
    if lists.iter().all(|v| !v.is_empty()) {
        goals(
            lists
                .iter()
                .map(|v| V::List(v[..v.len() - 1].to_vec()))
                .collect(),
            depth + 1,
            out,
        );
        goals(
            lists.iter().map(|v| v.last().unwrap().clone()).collect(),
            depth + 1,
            out,
        );
    }
}

fn construct(values: &[V], pool: &Pool, depth: usize) -> Option<Node> {
    if let Some(node) = pool.get(values) {
        return Some(node.clone());
    }
    if depth > 6 {
        return None;
    }
    let lists: Option<Vec<_>> = values
        .iter()
        .map(|v| if let V::List(v) = v { Some(v) } else { None })
        .collect();
    let lists = lists?;
    if lists.iter().all(|v| v.len() == lists[0].len()) && lists[0].len() <= 4 {
        let parts: Option<Vec<_>> = (0..lists[0].len())
            .map(|i| {
                construct(
                    &lists.iter().map(|v| v[i].clone()).collect::<Vec<_>>(),
                    pool,
                    depth + 1,
                )
            })
            .collect();
        if let Some(parts) = parts {
            return Some(Node {
                inherited: parts.iter().any(|n| n.inherited),
                expression: E::List(parts.into_iter().map(|n| n.expression).collect()),
                priority: 0,
            });
        }
    }
    if lists.iter().all(|v| !v.is_empty()) {
        let prefix = lists
            .iter()
            .map(|v| V::List(v[..v.len() - 1].to_vec()))
            .collect::<Vec<_>>();
        let last = lists
            .iter()
            .map(|v| v.last().unwrap().clone())
            .collect::<Vec<_>>();
        let (prefix, last) = (
            construct(&prefix, pool, depth + 1)?,
            construct(&last, pool, depth + 1)?,
        );
        return Some(Node {
            expression: E::Push(boxed(prefix.expression), boxed(last.expression)),
            inherited: prefix.inherited || last.inherited,
            priority: 0,
        });
    }
    None
}

struct Enumerator<'a, 'b> {
    search: &'a mut Search<'b>,
    pool: Pool,
    tried: BTreeSet<Vec<u8>>,
    goals: Vec<Vec<V>>,
    nodes: usize,
}
impl Enumerator<'_, '_> {
    fn propose(
        &mut self,
        expression: E,
        inherited: bool,
        priority: u8,
    ) -> Result<Option<SearchReceipt>, Error> {
        if expression.nodes() > 256 {
            return Ok(None);
        }
        let key = serde_json::to_vec(&expression).map_err(|e| Error::Storage(e.to_string()))?;
        if !self.tried.insert(key) {
            return Ok(None);
        }
        let (receipt, outputs) = self
            .search
            .observe(expression.clone(), inherited, false, true)?;
        if receipt.is_some() {
            return Ok(receipt);
        }
        let Some(outputs) = outputs else {
            return Ok(None);
        };
        self.search
            .meter
            .charge(self.pool.len() as u64 + outputs.iter().map(V::nodes).sum::<usize>() as u64)?;
        let priority = if self.goals.contains(&outputs) {
            0
        } else {
            priority
        };
        if let Some(old) = self.pool.get_mut(&outputs) {
            if expression.nodes() < old.expression.nodes() {
                *old = Node {
                    expression,
                    inherited,
                    priority: priority.min(old.priority),
                };
            }
            return Ok(None);
        }
        self.nodes += outputs.iter().map(V::nodes).sum::<usize>();
        self.pool.insert(
            outputs,
            Node {
                expression,
                inherited,
                priority,
            },
        );
        while self.pool.len() > 128 || self.nodes > 32768 {
            let key = self
                .pool
                .iter()
                .max_by_key(|(_, n)| (n.priority, n.expression.nodes()))
                .map(|(v, _)| v.clone())
                .unwrap();
            self.nodes -= key.iter().map(V::nodes).sum::<usize>();
            self.pool.remove(&key);
        }
        Ok(None)
    }
}

pub(super) fn learn(
    search: &mut Search<'_>,
    library: &[Program],
) -> Result<Option<SearchReceipt>, Error> {
    let inputs = search
        .examples
        .iter()
        .map(|e| e.input.nodes() + e.output.nodes())
        .sum::<usize>();
    if inputs > 4096 {
        return Ok(None);
    }
    let mut targets = Vec::new();
    goals(
        search.examples.iter().map(|e| e.output.clone()).collect(),
        0,
        &mut targets,
    );
    if targets.iter().flatten().map(V::nodes).sum::<usize>() > 32768 {
        return Ok(None);
    }
    let mut enumerator = Enumerator {
        search,
        pool: BTreeMap::new(),
        tried: BTreeSet::new(),
        goals: targets,
        nodes: 0,
    };
    for e in [E::Input, E::Empty, int(0), int(1)] {
        if let Some(r) = enumerator.propose(e, false, 1)? {
            return Ok(Some(r));
        }
    }
    for _ in 0..6 {
        for target in enumerator.goals.clone() {
            if !enumerator.pool.contains_key(&target)
                && let Some(node) = construct(&target, &enumerator.pool, 0)
                && let Some(r) = enumerator.propose(node.expression, node.inherited, 0)?
            {
                return Ok(Some(r));
            }
        }
        let mut nodes: Vec<_> = enumerator
            .pool
            .iter()
            .map(|(values, n)| (values.clone(), n.clone()))
            .collect();
        nodes.sort_by_key(|(_, n)| (n.priority, n.expression.nodes()));
        for (values, node) in &nodes {
            for i in 0..3 {
                if values.iter().all(|v| matches!(v, V::List(v) if v.len()>i))
                    && let Some(r) = enumerator.propose(
                        E::Get(boxed(node.expression.clone()), i),
                        node.inherited,
                        1,
                    )?
                {
                    return Ok(Some(r));
                }
            }
        }
        let integer: Vec<_> = nodes
            .iter()
            .filter(|(v, _)| v.iter().all(|v| matches!(v, V::Integer(_))))
            .take(16)
            .collect();
        for (_, a) in &integer {
            for (_, b) in &integer {
                for op in [E::Add, E::Subtract, E::Multiply, E::Remainder] {
                    if let Some(r) = enumerator.propose(
                        op(boxed(a.expression.clone()), boxed(b.expression.clone())),
                        a.inherited || b.inherited,
                        3,
                    )? {
                        return Ok(Some(r));
                    }
                }
            }
        }
        let list: Vec<_> = nodes
            .iter()
            .filter(|(v, _)| v.iter().all(|v| matches!(v, V::List(_))))
            .take(8)
            .collect();
        let scalar: Vec<_> = nodes
            .iter()
            .filter(|(v, _)| v.iter().all(|v| !matches!(v, V::List(_))))
            .take(8)
            .collect();
        for p in library {
            for (_, a) in &list {
                if let Some(r) = enumerator.propose(call(p, a.expression.clone()), true, 2)? {
                    return Ok(Some(r));
                }
            }
            for (_, a) in list.iter().chain(&scalar) {
                for (_, b) in &scalar {
                    if let Some(r) = enumerator.propose(
                        call(p, E::List(vec![a.expression.clone(), b.expression.clone()])),
                        true,
                        2,
                    )? {
                        return Ok(Some(r));
                    }
                }
            }
        }
    }
    Ok(None)
}
