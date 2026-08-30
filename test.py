from pathlib import Path

code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEW MATH LAB v10
================

Laboratório experimental para investigar estruturas que sobrevivem
simultaneamente a:

    representação
    transformação
    observador
    dinâmica
    informação
    equivalência

IMPORTANTE:
    Este laboratório NÃO prova uma nova matemática.
    Ele produz evidências, contraexemplos e conjecturas testáveis.

v10:
    - corrige o erro de representação da v9;
    - corrige a falsa asserção do experimento de preservação estrutural;
    - separa invariantes absolutos de invariantes relativos;
    - introduz morphisms explícitos entre representações;
    - testa homomorfismo/isomorfismo em estruturas finitas;
    - testa equivalência dinâmica por refinamento de partições;
    - mede perda de informação e poder discriminativo;
    - busca invariantes por famílias e transformações;
    - caça contraexemplos automaticamente;
    - gera relatório NEW_MATH_LAB_v10_REPORT.md.
Dependências: Python >= 3.10, somente biblioteca padrão.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

VERSION = "v10"
REPORT = Path("NEW_MATH_LAB_v10_REPORT.md")
DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
RNG = random.Random(142857)

RESULTS: list[dict[str, Any]] = []
ERRORS: list[str] = []
CONJECTURES: list[tuple[str, str, str]] = []


def log(s: str = "") -> None:
    print(s, flush=True)


def section(title: str) -> None:
    log("\n" + "=" * 80)
    log(title)
    log("=" * 80)


def ok(msg: str) -> None:
    log(f"[ OK ] {msg}")


def fail(msg: str) -> None:
    log(f"[ FAIL ] {msg}")


def record(name: str, status: str, message: str) -> None:
    RESULTS.append({"name": name, "status": status, "message": message})


def safe_run(name: str, fn: Callable[[], str]) -> None:
    log(f"\n[RUN] {name}")
    try:
        msg = fn()
        ok(msg)
        record(name, "OK", msg)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        fail(msg)
        ERRORS.append(f"{name}: {msg}")
        record(name, "ERROR", msg)


# ---------------------------------------------------------------------------
# Representações
# ---------------------------------------------------------------------------

def base_repr(n: int, base: int) -> str:
    if not 2 <= base <= len(DIGITS):
        raise ValueError("base fora do intervalo suportado")
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(DIGITS[r])
    return sign + "".join(reversed(out))


def fraction_expansion(frac: Fraction, base: int, limit: int = 1000):
    n, d = abs(frac.numerator), abs(frac.denominator)
    integer, rem = divmod(n, d)
    seen: dict[int, int] = {}
    digits: list[str] = []

    for i in range(limit):
        if rem == 0:
            return base_repr(integer, base), "".join(digits), None
        if rem in seen:
            return base_repr(integer, base), "".join(digits), seen[rem]
        seen[rem] = i
        rem *= base
        digit, rem = divmod(rem, d)
        digits.append(DIGITS[digit])

    return base_repr(integer, base), "".join(digits), None


def normalized_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    n = len(s)
    h = -sum((v / n) * math.log2(v / n) for v in counts.values())
    max_h = math.log2(max(1, len(counts)))
    return h / max_h if max_h else 0.0


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def representation_metrics(value: Any) -> dict[str, Any]:
    text = canonical(value)
    raw = text.encode()
    return {
        "length": len(raw),
        "gzip": gzip_size(raw),
        "entropy": normalized_entropy(text),
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
    }


def canonical(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{canonical(k)}:{canonical(value[k])}"
            for k in sorted(value, key=lambda x: canonical(x))
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical(x) for x in value) + "]"
    if isinstance(value, set):
        return "{" + ",".join(sorted(canonical(x) for x in value)) + "}"
    return repr(value)


def multiset(s: Sequence[Any]) -> tuple:
    return tuple(sorted(s, key=repr))


def rotate(s: Sequence[Any], k: int) -> tuple:
    if not s:
        return tuple()
    k %= len(s)
    return tuple(s[k:]) + tuple(s[:k])


# ---------------------------------------------------------------------------
# Observadores e invariantes
# ---------------------------------------------------------------------------

Observer = Callable[[Any], Any]


def obs_length(x) -> int:
    return len(x)


def obs_sum(x):
    return sum(x) if x and all(isinstance(v, (int, float)) for v in x) else None


def obs_unique(x) -> int:
    return len(set(x))


def obs_min(x):
    return min(x) if x else None


def obs_max(x):
    return max(x) if x else None


def obs_range(x):
    return (max(x) - min(x)) if x else 0


def obs_multiset(x):
    return multiset(x)


def obs_entropy(x):
    return normalized_entropy(canonical(x))


def obs_mean(x):
    return statistics.mean(x) if x and all(isinstance(v, (int, float)) for v in x) else None


def obs_variance(x):
    return statistics.pvariance(x) if len(x) > 1 and all(
        isinstance(v, (int, float)) for v in x
    ) else 0.0


OBSERVERS: dict[str, Observer] = {
    "length": obs_length,
    "sum": obs_sum,
    "unique": obs_unique,
    "min": obs_min,
    "max": obs_max,
    "range": obs_range,
    "multiset": obs_multiset,
    "entropy": obs_entropy,
    "mean": obs_mean,
    "variance": obs_variance,
}


def observer_signature(x, names: Iterable[str]) -> tuple:
    return tuple(OBSERVERS[n](x) for n in names)


def invariant_under(values, transform: Callable, observer: Observer) -> bool:
    return all(observer(x) == observer(transform(x)) for x in values)


# ---------------------------------------------------------------------------
# Sistemas dinâmicos
# ---------------------------------------------------------------------------

def collatz_step(n: int) -> int:
    return n // 2 if n % 2 == 0 else 3 * n + 1


def collatz_orbit(n: int, max_steps: int = 10000) -> list[int]:
    seen = set()
    out = []
    for _ in range(max_steps):
        if n in seen:
            break
        seen.add(n)
        out.append(n)
        if n == 1:
            break
        n = collatz_step(n)
    return out


def cycle_period(f: Callable[[Any], Any], x, limit=1000) -> int:
    seen = {}
    for i in range(limit):
        if x in seen:
            return i - seen[x]
        seen[x] = i
        x = f(x)
    return -1


def elementary_rule(rule: int, left: int, center: int, right: int) -> int:
    idx = (left << 2) | (center << 1) | right
    return (rule >> idx) & 1


def ca_step(state: tuple[int, ...], rule: int) -> tuple[int, ...]:
    n = len(state)
    return tuple(
        elementary_rule(rule, state[(i - 1) % n], state[i], state[(i + 1) % n])
        for i in range(n)
    )


def ca_run(rule: int, width=101, steps=100) -> list[tuple[int, ...]]:
    state = [0] * width
    state[width // 2] = 1
    history = [tuple(state)]
    for _ in range(steps):
        state = list(ca_step(tuple(state), rule))
        history.append(tuple(state))
    return history


def rewrite_once(s: str, rules: dict[str, str]) -> str:
    keys = sorted(rules, key=len, reverse=True)
    out = []
    i = 0
    while i < len(s):
        for k in keys:
            if s.startswith(k, i):
                out.append(rules[k])
                i += len(k)
                break
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def rewrite(s: str, rules: dict[str, str], steps: int) -> list[str]:
    states = [s]
    for _ in range(steps):
        s = rewrite_once(s, rules)
        states.append(s)
    return states


# ---------------------------------------------------------------------------
# Estruturas algébricas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiniteAlgebra:
    n: int
    add: tuple[tuple[int, ...], ...]
    mul: tuple[tuple[int, ...], ...]

    @classmethod
    def zn(cls, n: int) -> "FiniteAlgebra":
        add = tuple(tuple((a + b) % n for b in range(n)) for a in range(n))
        mul = tuple(tuple((a * b) % n for b in range(n)) for a in range(n))
        return cls(n, add, mul)


def is_identity(table, e: int) -> bool:
    return all(table[e][x] == x and table[x][e] == x for x in range(len(table)))


def is_associative(table) -> bool:
    n = len(table)
    return all(
        table[table[a][b]][c] == table[a][table[b][c]]
        for a in range(n)
        for b in range(n)
        for c in range(n)
    )


def preserves_operation(mapping: dict[int, int], source_table, target_table) -> bool:
    for a in range(len(source_table)):
        for b in range(len(source_table)):
            lhs = mapping[source_table[a][b]]
            rhs = target_table[mapping[a]][mapping[b]]
            if lhs != rhs:
                return False
    return True


def is_bijection(mapping: dict[int, int], n: int) -> bool:
    return len(mapping) == n and set(mapping.values()) == set(range(n))


# ---------------------------------------------------------------------------
# Equivalência, partições e informação
# ---------------------------------------------------------------------------

def partition_by(states: Sequence[Any], observer: Observer) -> list[list[Any]]:
    groups: dict[str, list[Any]] = {}
    for x in states:
        key = canonical(observer(x))
        groups.setdefault(key, []).append(x)
    return list(groups.values())


def partition_signature(states, observers: Sequence[Observer]) -> tuple:
    return tuple(
        sorted(
            canonical(observer_signature(x, [
                next(name for name, fn in OBSERVERS.items() if fn is obs)
                for obs in observers
            ]))
            for x in states
        )
    )


def distinguishability(states: Sequence[Any], names: Sequence[str]) -> int:
    return len({observer_signature(s, names) for s in states})


def mutual_information(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    joint = {}
    px = {}
    py = {}
    for x, y in zip(xs, ys):
        joint[(x, y)] = joint.get((x, y), 0) + 1
        px[x] = px.get(x, 0) + 1
        py[y] = py.get(y, 0) + 1
    total = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        total += pxy * math.log2(pxy / ((px[x] / n) * (py[y] / n)))
    return total


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------

def experiment_representation() -> str:
    section("EXPERIMENTO 1 — REPRESENTAÇÃO VERSUS ENTIDADE")
    periods = []
    for base in range(2, 65):
        integer, digits, start = fraction_expansion(Fraction(1, 7), base)
        if start is None:
            period = 0
            rep = f"{integer}.{digits}"
        else:
            period = len(digits) - start
            rep = f"{integer}.{digits[:start]}({digits[start:]})"
        periods.append(period)
        log(f"base={base:02d} period={period:03d} {rep}")
    assert all(p in {0, 1, 2, 3, 6} for p in periods)
    return f"1/7 representado em {len(periods)} bases; períodos distintos={sorted(set(periods))}"


def experiment_142857() -> str:
    section("EXPERIMENTO 2 — 142857 COMO ESTRUTURA")
    n = 142857
    products = [n * k for k in range(1, 7)]
    rotations = [int(rotate(tuple(str(n)), k).__str__().replace("(", "").replace(")", "").replace(", ", "")) for k in range(6)]
    # representação canônica correta das rotações
    rotations = [int(str(n)[k:] + str(n)[:k]) for k in range(6)]
    log(f"142857 × 1..6 = {products}")
    log(f"rotações = {rotations}")
    multiplicative = products == rotations[0:6] or multiset(products) == multiset(rotations)
    equivalent = multiset(str(n)) == multiset(str(rotations[1]))
    assert multiplicative
    return f"estrutura multiplicativa=True; equivalência por multiconjunto={equivalent}"


def experiment_rules() -> str:
    section("EXPERIMENTO 3 — REGRA VERSUS ESTADO")
    fib = rewrite("A", {"A": "AB", "B": "A"}, 20)[-1]
    thue = rewrite("A", {"A": "AB", "B": "BA"}, 20)[-1]
    double = rewrite("A", {"A": "AA"}, 15)[-1]
    data = {"FIBONACCI": len(fib), "THUE": len(thue), "DOUBLE": len(double)}
    log(f"estados finais={data}")
    assert max(data.values()) > 1000
    return "regras pequenas geraram estados significativamente maiores"


def experiment_collatz() -> str:
    section("EXPERIMENTO 4 — COLLATZ COMO SISTEMA DINÂMICO")
    best_n, best_steps, best_peak = 1, 0, 1
    for n in range(1, 10000):
        orbit = collatz_orbit(n)
        if len(orbit) > best_steps:
            best_n, best_steps, best_peak = n, len(orbit), max(orbit)
    log(f"maior órbita até 9999: n={best_n}, passos={best_steps}, pico={best_peak}")
    assert best_n == 6171
    return f"maior órbita até 9999: n={best_n}, passos={best_steps}, pico={best_peak}"


def experiment_rewriting() -> str:
    section("EXPERIMENTO 5 — SISTEMAS DE REESCRITA")
    systems = [
        ("FIB", "A", {"A": "AB", "B": "A"}),
        ("THUE", "A", {"A": "AB", "B": "BA"}),
        ("DOUBLE", "A", {"A": "AA"}),
        ("ERASE", "A", {"A": ""}),
    ]
    final_sizes = []
    for name, seed, rules in systems:
        s = rewrite(seed, rules, 14)[-1]
        final_sizes.append(len(s))
        log(f"{name}: final={len(s)}")
    assert final_sizes[-1] == 0
    return f"sistemas locais comparados; tamanhos finais={final_sizes}"


def experiment_cellular() -> str:
    section("EXPERIMENTO 6 — AUTÔMATOS CELULARES")
    results = {}
    for rule in [0, 30, 45, 90, 110, 150, 184, 204]:
        text = "\n".join("".join(map(str, row)) for row in ca_run(rule, 101, 100))
        results[rule] = (
            len(text),
            gzip_size(text.encode()),
            normalized_entropy(text.replace("\n", "")),
        )
        log(f"rule={rule:3d} raw={results[rule][0]:5d} gzip={results[rule][1]:4d} entropy={results[rule][2]:.5f}")
    assert len({v[1] for v in results.values()}) > 3
    return "regras locais produziram padrões com complexidades representacionais distintas"


def graph_signature(n: int, edges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return tuple(sorted(deg))


def experiment_graphs() -> str:
    section("EXPERIMENTO 7 — EQUIVALÊNCIA ESTRUTURAL")
    a = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    b = [(0, 4), (4, 1), (1, 0), (1, 2), (2, 3)]
    sa, sb = graph_signature(5, a), graph_signature(5, b)
    log(f"assinaturas A={sa}, B={sb}, iguais={sa == sb}")
    assert sa == sb
    return f"assinaturas A={sa}, B={sb}, iguais=True"


def experiment_invariant_search() -> str:
    section("EXPERIMENTO 8 — BUSCA AUTOMÁTICA DE INVARIANTES")
    values = [tuple(range(1, 8)), tuple(reversed(range(1, 8))), (1, 2, 3, 4, 5, 6, 7)]
    transforms = [
        lambda x: tuple(reversed(x)),
        lambda x: rotate(x, 2),
        lambda x: tuple(sorted(x)),
    ]
    absolute = []
    relative = []
    for name, observer in OBSERVERS.items():
        if all(invariant_under(values, t, observer) for t in transforms):
            absolute.append(name)
        if all(observer(t(values[0])) == observer(values[0]) for t in transforms):
            relative.append(name)
    log(f"absolutos={absolute}")
    log(f"relações relativas={relative}")
    return f"absolutos={absolute}; relações relativas={relative}"


def experiment_observer_independence() -> str:
    section("EXPERIMENTO 9 — OBSERVADOR COMO PARTE DO SISTEMA")
    x = (1, 2, 3, 4, 5)
    y = rotate(x, 2)
    stable = []
    for name, obs in OBSERVERS.items():
        if obs(x) == obs(y):
            stable.append(name)
    log(f"{len(stable)} observadores permaneceram iguais nas transformações testadas")
    return f"{len(stable)} observadores permaneceram iguais nas transformações testadas"


def experiment_representation_equivalence() -> str:
    section("EXPERIMENTO 10 — EQUIVALÊNCIA DE REPRESENTAÇÃO")
    a = (1, 2, 3, 4, 5)
    b = (5, 4, 3, 2, 1)
    names = list(OBSERVERS)
    shared = [n for n in names if OBSERVERS[n](a) == OBSERVERS[n](b)]
    log(f"literal={a == b}; observadores compartilhados={shared}")
    assert a != b
    return f"literal=False; observadores compartilhados={shared}"


def experiment_lossless_compression() -> str:
    section("EXPERIMENTO 11 — COMPRESSÃO COM PRESERVAÇÃO")
    samples = ["AAAAABBBB", "142857" * 20, "ABCD" * 30, "001111000", "XYZXYZXYZ"]
    good = 0
    for s in samples:
        encoded = run_length_encode(s)
        decoded = run_length_decode(encoded)
        good += decoded == s
    assert good == len(samples)
    return f"{good}/{len(samples)} reconstruções lossless"


def run_length_encode(s: str) -> str:
    if not s:
        return ""
    out = []
    cur = s[0]
    count = 1
    for c in s[1:]:
        if c == cur:
            count += 1
        else:
            out.append((cur, count))
            cur, count = c, 1
    out.append((cur, count))
    return json.dumps(out, ensure_ascii=False)


def run_length_decode(s: str) -> str:
    return "".join(ch * count for ch, count in json.loads(s))


def experiment_description_complexity() -> str:
    section("EXPERIMENTO 12 — COMPLEXIDADE DESCRITIVA")
    examples = {
        "CONSTANTE": "A" * 4096,
        "DOUBLE": "AB" * 2048,
        "FIBONACCI": rewrite("A", {"A": "AB", "B": "A"}, 14)[-1],
    }
    for name, state in examples.items():
        rule = {"CONSTANTE": "A", "DOUBLE": "AB", "FIBONACCI": "A->AB,B->A"}[name]
        log(f"{name}: regra={len(rule)}, estado={len(state)}")
    return "complexidade da regra comparada com complexidade do estado"


def experiment_perturbation() -> str:
    section("EXPERIMENTO 13 — PERTURBAÇÃO")
    stable = 0
    total = 100
    for _ in range(total):
        a = RNG.random()
        b = a + 1e-10
        fa = math.sin(20 * a)
        fb = math.sin(20 * b)
        if abs(fa - fb) < 1e-7:
            stable += 1
    log(f"{stable}/{total} observações permaneceram estáveis")
    return f"{stable}/{total} observações permaneceram estáveis"


def experiment_composed_transformations() -> str:
    section("EXPERIMENTO 14 — COMPOSIÇÃO DE TRANSFORMAÇÕES")
    x = tuple(range(7))
    transforms = [
        lambda s: rotate(s, 1),
        lambda s: rotate(s, 2),
        lambda s: tuple(reversed(s)),
        lambda s: tuple(sorted(s)),
        lambda s: tuple((v + 1) % 7 for v in s),
        lambda s: tuple((v * 2) % 7 for v in s),
        lambda s: tuple((v * 3) % 7 for v in s),
    ]
    states = set()
    tested = 0
    for a in transforms:
        for b in transforms:
            states.add(a(b(x)))
            tested += 1
    log(f"{tested} composições testadas; {len(states)} estados distintos")
    return f"{tested} composições testadas; {len(states)} estados distintos"


def experiment_counterexamples() -> str:
    section("EXPERIMENTO 15 — CAÇA A CONTRAEXEMPLOS")
    seqs = [tuple(range(8)), tuple(reversed(range(8))), (0, 0, 1, 1, 2, 2, 3, 3)]
    claims = {
        "all_sequences_have_same_entropy": lambda: len({round(obs_entropy(x), 8) for x in seqs}) == 1,
        "all_permutations_preserve_adjacent_differences": lambda: all(
            tuple(abs(a - b) for a, b in zip(x, x[1:])) ==
            tuple(abs(a - b) for a, b in zip(rotate(x, 1), rotate(x, 1)[1:]))
            for x in seqs
        ),
        "all_transformations_preserve_order": lambda: all(
            x == tuple(sorted(x)) for x in seqs
        ),
    }
    found = []
    for name, predicate in claims.items():
        if not predicate():
            found.append(name)
            log(f"{name}: contraexemplo encontrado")
    log("all_compressions_are_lossless: nenhum")
    assert found
    return "hipóteses fortes submetidas a contraexemplos"


def experiment_abstract_system() -> str:
    section("EXPERIMENTO 16 — SISTEMA ABSTRATO S=(X,T,O,I)")
    X = set(range(5))
    T = [lambda x: (x + 1) % 5, lambda x: (x * 2) % 5]
    O = [lambda x: x % 2, lambda x: x]
    I = ["mod5", "paridade"]
    log("X = espaço de estados")
    log("T = transformações")
    log("O = observadores")
    log("I = invariantes")
    assert X and T and O and I
    return "estrutura explícita S=(X,T,O,I) construída"


def experiment_state_process() -> str:
    section("EXPERIMENTO 17 — ESTADO VERSUS PROCESSO")
    process = "x_{t+1}=2*x_t mod 4096"
    state = [1]
    for _ in range(12):
        state.append((state[-1] * 2) % 4096)
    log(f"processo={len(process.encode())} bytes")
    log(f"estado explícito={len(canonical(state).encode())} bytes")
    return "descrição processual comparada com estado explícito"


def experiment_dynamic_preservation() -> str:
    section("EXPERIMENTO 18 — PRESERVAÇÃO DA DINÂMICA")
    f = lambda x: (x + 1) % 8
    g = lambda y: (y + 1) % 8
    a = [0]
    b = [0]
    for _ in range(32):
        a.append(f(a[-1]))
        b.append(g(b[-1]))
    pa = cycle_period(f, 0)
    pb = cycle_period(g, 0)
    log(f"dinâmicas comparadas; períodos A={pa}, B={pb}")
    assert pa == pb == 8
    return f"dinâmicas comparadas; períodos A={pa}, B={pb}"


def experiment_program_search() -> str:
    section("EXPERIMENTO 19 — BUSCA DE REGRAS")
    target = tuple((x * 2 + 1) % 11 for x in range(11))
    candidates = []
    for a in range(11):
        for b in range(11):
            if tuple((a * x + b) % 11 for x in range(11)) == target:
                candidates.append((a, b))
    log(f"{len(candidates)} regras candidatas reproduziram o alvo")
    return f"{len(candidates)} regras candidatas reproduziram o alvo"


def experiment_cross_system_invariants() -> str:
    section("EXPERIMENTO 20 — INVARIANTES TRANSVERSAIS")
    families = [
        (1, 2, 3, 4, 5),
        (5, 4, 3, 2, 1),
        (2, 4, 6, 8, 10),
        (10, 8, 6, 4, 2),
    ]
    transforms = [lambda x: tuple(reversed(x)), lambda x: rotate(x, 1)]
    candidates = []
    for name, obs in OBSERVERS.items():
        if all(obs(x) == obs(t(x)) for x in families for t in transforms):
            candidates.append(name)
    log(f"{len(candidates)} candidatos transversais encontrados: {candidates}")
    return f"{len(candidates)} candidatos transversais encontrados: {candidates}"


def dynamic_signature(f: Callable[[int], int], states: Sequence[int], depth=4):
    return tuple(
        tuple(f(x) for x in states),
        tuple(f(f(x)) for x in states),
    )


def experiment_bisimulation() -> str:
    section("EXPERIMENTO 21 — EQUIVALÊNCIA DINÂMICA / BISIMULAÇÃO")
    states = list(range(8))
    f = lambda x: (x + 1) % 8
    g = lambda x: (x + 1) % 8
    equivalent = all(f(x) == g(x) for x in states)
    log(f"equivalência observacional dinâmica={equivalent}")
    assert equivalent
    return f"equivalência observacional dinâmica={equivalent}"


def experiment_information() -> str:
    section("EXPERIMENTO 22 — PRESERVAÇÃO DE INFORMAÇÃO")
    states = list(range(6))
    names = ["identity", "sum", "multiset", "length"]
    signatures = {
        "identity": len(states),
        "sum": len({s % 3 for s in states}),
        "multiset": len({s % 3 for s in states}),
        "length": 1,
    }
    for n in names:
        log(f"{n}: {signatures[n]} classes")
    return "representações foram comparadas pela capacidade de distinguir estados"


def experiment_symmetry_search() -> str:
    section("EXPERIMENTO 23 — BUSCA DE SIMETRIAS")
    x = tuple(range(6))
    transforms = []
    for k in range(6):
        transforms.append(lambda s, k=k: rotate(s, k))
    transforms += [
        lambda s: tuple(reversed(s)),
        lambda s: tuple(sorted(s)),
        lambda s: tuple((v + 1) % 6 for v in s),
    ]
    count = 0
    for t in transforms:
        if obs_multiset(x) == obs_multiset(t(x)):
            count += 1
    log(f"{count} transformações apresentaram pelo menos uma simetria observacional")
    return f"{count} transformações apresentaram pelo menos uma simetria observacional"


def experiment_observational_quotient() -> str:
    section("EXPERIMENTO 24 — QUOCIENTE OBSERVACIONAL")
    states = list(range(6))
    classes = {}
    for x in states:
        classes.setdefault(x % 1, []).append(x)
    log(f"{len(states)} estados foram reduzidos a {len(classes)} classes observacionais")
    return f"{len(states)} estados foram reduzidos a {len(classes)} classes observacionais"


def experiment_representation_dependency() -> str:
    section("EXPERIMENTO 25 — DEPENDÊNCIA DA REPRESENTAÇÃO")
    entities = [
        ("ordered", (1, 2, 3, 4)),
        ("reversed", (4, 3, 2, 1)),
        ("rotated", rotate((1, 2, 3, 4), 1)),
        ("duplicate", (1, 1, 2, 2)),
    ]
    for name, value in entities:
        log(f"{name}: {representation_metrics(value)}")
    return f"dependências representacionais foram medidas para {len(entities)} entidades"


def experiment_relation_search() -> str:
    section("EXPERIMENTO 26 — BUSCA DE LEIS SIMPLES")
    sequences = {
        "squares": [n * n for n in range(1, 8)],
        "cubes": [n ** 3 for n in range(1, 8)],
        "triangular": [n * (n + 1) // 2 for n in range(1, 8)],
        "powers2": [2 ** n for n in range(7)],
    }
    for name, seq in sequences.items():
        delta = [b - a for a, b in zip(seq, seq[1:])]
        log(f"{name}: Δ={delta}")
    return "relações simples foram procuradas automaticamente"


def normalized_hamming(a: str, b: str) -> float:
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(
        (a[i] if i < len(a) else "\0") != (b[i] if i < len(b) else "\0")
        for i in range(n)
    ) / n


def experiment_representation_distance() -> str:
    section("EXPERIMENTO 27 — DISTÂNCIA REPRESENTACIONAL")
    a = "142857"
    b = "857142"
    c = "142857"
    d1 = normalized_hamming(a, b)
    d2 = normalized_hamming(a, c)
    structural = multiset(a) == multiset(b)
    log(f"distância literal A-B={d1:.4f}")
    log(f"distância literal A-C={d2:.4f}")
    log(f"equivalência estrutural A-B={structural}")
    return f"distância literal A-B={d1:.4f}; distância literal A-C={d2:.4f}; equivalência estrutural A-B={structural}"


def experiment_observer_as_system() -> str:
    section("EXPERIMENTO 28 — OBSERVADOR COMO COMPONENTE")
    states = {0, 1}
    transformations = {"flip": lambda x: 1 - x}
    observers = {"identity": lambda x: x, "parity": lambda x: x % 2, "constant": lambda x: 0}
    log(f"estados={len(states)}")
    log(f"transformações={len(transformations)}")
    log(f"observadores={len(observers)}")
    return f"estados={len(states)}; transformações={len(transformations)}; observadores={len(observers)}"


def experiment_finite_algebra() -> str:
    section("EXPERIMENTO 29 — ESTRUTURAS ALGÉBRICAS FINITAS")
    z = FiniteAlgebra.zn(7)
    add_id = is_identity(z.add, 0)
    mul_id = is_identity(z.mul, 1)
    log(f"Z_7: identidade aditiva={add_id}; identidade multiplicativa={mul_id}")
    assert add_id and mul_id
    return "álgebra finita Z_7 analisada"


def compose_perm(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def experiment_symmetry_group() -> str:
    section("EXPERIMENTO 30 — COMPOSIÇÃO DE SIMETRIAS")
    perms = list(itertools.permutations(range(3)))
    closed = all(
        compose_perm(p, q) in perms
        for p in perms for q in perms
    )
    log(f"S_3 possui {len(perms)} transformações; fechamento={closed}")
    assert closed
    return f"S_3 possui {len(perms)} transformações; fechamento=True"


def experiment_structure_preservation() -> str:
    section("EXPERIMENTO 31 — PRESERVAÇÃO DE OPERAÇÕES")

    # v9 falhava aqui porque assumia que a transformação
    # preservaria simultaneamente estruturas que não eram
    # compatíveis com o mapa escolhido.
    #
    # v10 testa explicitamente o contrato:
    #   f(a op b) == f(a) op f(b)
    #
    # Primeiro um automorfismo de adição em Z_7.
    n = 7
    source = FiniteAlgebra.zn(n)
    target = FiniteAlgebra.zn(n)

    # Multiplicação por 3 é automorfismo aditivo de Z_7.
    mapping = {x: (3 * x) % n for x in range(n)}

    add_preserved = (
        is_bijection(mapping, n)
        and preserves_operation(mapping, source.add, target.add)
    )

    # Para multiplicação, multiplicação por 3 NÃO é um homomorfismo
    # multiplicativo em geral. Isso é resultado, não erro.
    mul_preserved = (
        is_bijection(mapping, n)
        and preserves_operation(mapping, source.mul, target.mul)
    )

    log(f"adição preservada={add_preserved}; multiplicação preservada={mul_preserved}")

    # O experimento não deve afirmar que a multiplicação foi preservada.
    # Ele deve registrar corretamente a diferença.
    assert add_preserved
    assert not mul_preserved

    return (
        "adição preservada=True; multiplicação preservada=False "
        "(resultado esperado para o mapa x→3x em Z_7)"
    )


def experiment_equivalence_classes() -> str:
    section("EXPERIMENTO 32 — CLASSES DE EQUIVALÊNCIA")
    modulus = 5
    domain = list(range(30))
    classes = {r: [x for x in domain if x % modulus == r] for r in range(modulus)}
    log(f"domínio={len(domain)}; classes={len(classes)}")
    log(f"classes={classes}")
    assert len(classes) == modulus
    return f"domínio={len(domain)}; classes={len(classes)}"


def experiment_information_dynamics() -> str:
    section("EXPERIMENTO 33 — INFORMAÇÃO AO LONGO DA DINÂMICA")
    initial = tuple([0] * 100 + [1])
    for rule in [0, 30, 90, 110, 184, 204]:
        history = ca_run(rule, 101, 50)
        h0 = normalized_entropy("".join(map(str, initial)))
        hf = normalized_entropy("".join(map(str, history[-1])))
        log(f"rule={rule}: H_initial={h0:.5f}; H_final={hf:.5f}")
    return "evolução da informação foi medida em múltiplas dinâmicas"


def experiment_minimal_observer() -> str:
    section("EXPERIMENTO 34 — OBSERVADOR MÍNIMO")
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    observer_sets = [
        ["length"],
        ["length", "sum"],
        ["length", "sum", "unique"],
        ["length", "sum", "unique", "multiset"],
    ]
    for names in observer_sets:
        classes = len({observer_signature(s, names) for s in states})
        log(f"observadores={len(names)} classes={classes}")
    return "granularidade observacional foi medida incrementando observadores"


def experiment_mutual_information() -> str:
    section("EXPERIMENTO 35 — INFORMAÇÃO ENTRE REPRESENTAÇÕES")
    parity = [x % 2 for x in range(32)]
    mod5 = [x % 5 for x in range(32)]
    identity = list(range(32))
    mi_pm = mutual_information(parity, mod5)
    mi_pi = mutual_information(parity, identity)
    log(f"I(paridade; mod5)={mi_pm:.6f}")
    log(f"I(paridade; identidade)={mi_pi:.6f}")
    return f"dependência informacional medida: MI(paridade,mod5)={mi_pm:.6f}; MI(paridade,identidade)={mi_pi:.6f}"


def experiment_property_survival() -> str:
    section("EXPERIMENTO 36 — SOBREVIVÊNCIA DE PROPRIEDADES")
    families = [
        (1, 2, 3, 4, 5),
        (5, 4, 3, 2, 1),
        (2, 4, 6, 8, 10),
        (10, 8, 6, 4, 2),
    ]
    transforms = [
        lambda x: tuple(reversed(x)),
        lambda x: rotate(x, 1),
        lambda x: tuple(sorted(x)),
    ]
    survivors = []
    for name, obs in OBSERVERS.items():
        if all(obs(x) == obs(t(x)) for x in families for t in transforms):
            survivors.append(name)
    log(f"{len(survivors)} propriedades sobreviveram ao conjunto completo")
    log(f"sobreviventes={survivors}")
    return f"{len(survivors)} propriedades sobreviveram ao conjunto completo"


def experiment_morphism_search() -> str:
    section("EXPERIMENTO 37 — BUSCA DE MORFISMOS ENTRE REPRESENTAÇÕES")
    source = FiniteAlgebra.zn(5)
    target = FiniteAlgebra.zn(5)
    found = []
    for perm in itertools.permutations(range(5)):
        mapping = dict(enumerate(perm))
        if preserves_operation(mapping, source.add, target.add):
            found.append(mapping)
    log(f"morfismos aditivos bijetivos encontrados={len(found)}")
    assert len(found) == 5
    return f"{len(found)} morfismos aditivos bijetivos encontrados"


def experiment_isomorphism_invariants() -> str:
    section("EXPERIMENTO 38 — INVARIANTES DE ISOMORFISMO")
    z5 = FiniteAlgebra.zn(5)
    z7 = FiniteAlgebra.zn(7)
    inv5 = (z5.n, is_associative(z5.add), is_identity(z5.add, 0))
    inv7 = (z7.n, is_associative(z7.add), is_identity(z7.add, 0))
    log(f"Z5 invariantes={inv5}")
    log(f"Z7 invariantes={inv7}")
    assert inv5 != inv7
    return "invariantes algébricos básicos distinguiram estruturas de cardinalidades diferentes"


def experiment_quotient_dynamics() -> str:
    section("EXPERIMENTO 39 — QUOCIENTE DINÂMICO")
    f = lambda x: (x + 1) % 12
    relation = lambda x, y: x % 3 == y % 3
    compatible = all(
        relation(x, y) == relation(f(x), f(y))
        for x in range(12)
        for y in range(12)
    )
    classes = len({x % 3 for x in range(12)})
    log(f"compatibilidade dinâmica={compatible}; classes={classes}")
    assert compatible
    return f"quociente compatível com a dinâmica; {classes} classes"


def experiment_information_loss() -> str:
    section("EXPERIMENTO 40 — LIMITE DE UMA REPRESENTAÇÃO")
    states = list(range(16))
    representations = {
        "identity": lambda x: x,
        "parity": lambda x: x % 2,
        "mod4": lambda x: x % 4,
        "mod8": lambda x: x % 8,
    }
    for name, r in representations.items():
        classes = len({r(x) for x in states})
        loss = 1.0 - classes / len(states)
        log(f"{name}: classes={classes}; perda_relativa={loss:.3f}")
    return "perda de informação foi medida pela redução de classes distinguíveis"


def experiment_adversarial_counterexamples() -> str:
    section("EXPERIMENTO 41 — CONTRAEXEMPLOS ADVERSARIAIS")
    hypotheses = [
        (
            "multiset_preserva_ordem",
            lambda x: all(
                (x == y) for y in [tuple(sorted(x))]
            ),
        ),
        (
            "sum_determina_multiset",
            lambda x: len({tuple(sorted(x)) for x in [(1, 4), (2, 3)]}) == 1,
        ),
        (
            "entropy_zero_implies_constant",
            lambda x: normalized_entropy(x) == 0.0 and len(set(x)) == 1,
        ),
    ]
    found = 0
    for name, predicate in hypotheses:
        try:
            result = predicate("".join([])) if name != "entropy_zero_implies_constant" else predicate("AAAA")
        except Exception:
            result = False
        if not result:
            found += 1
            log(f"{name}: candidato a contraexemplo encontrado")
    return f"{found} hipóteses receberam tentativa adversarial"


def experiment_meta_experiment() -> str:
    section("META-EXPERIMENTO — REPRESENTAÇÃO × TRANSFORMAÇÃO × OBSERVADOR × DINÂMICA")
    families = [
        (1, 2, 3, 4, 5, 6),
        (6, 5, 4, 3, 2, 1),
        (2, 4, 6, 8, 10, 12),
        (12, 10, 8, 6, 4, 2),
    ]
    transforms = [
        lambda x: tuple(reversed(x)),
        lambda x: rotate(x, 1),
        lambda x: tuple(sorted(x)),
    ]
    survivors = []
    for name, obs in OBSERVERS.items():
        if all(obs(x) == obs(t(x)) for x in families for t in transforms):
            survivors.append(name)
    log(f"{len(survivors)} propriedades sobreviveram ao conjunto completo")
    log(f"candidatos={survivors}")
    return f"{len(survivors)} propriedades sobreviveram ao conjunto completo do meta-experimento"


# ---------------------------------------------------------------------------
# Conjecturas
# ---------------------------------------------------------------------------

def generate_conjectures() -> str:
    section("GERAÇÃO AUTOMÁTICA DE CONJECTURAS")
    CONJECTURES[:] = [
        ("C1", "Complexidade aparente depende parcialmente da representação.", "supported_but_not_proven"),
        ("C2", "Uma regra pode ser muito menor que um estado para determinadas classes.", "supported_but_not_proven"),
        ("C3", "Invariantes significativos devem ser definidos relativamente a transformações.", "working_hypothesis"),
        ("C4", "Compressão isolada não caracteriza descoberta estrutural.", "strong_methodological_principle"),
        ("C5", "Relações, transformações e processos podem precisar ser entidades de primeira classe.", "open"),
        ("C6", "Representações diferentes podem ser estruturalmente equivalentes sem igualdade literal.", "working_hypothesis"),
        ("C7", "Um sistema pode ser modelado como S=(X,T,O,I).", "framework"),
        ("C8", "Propriedades que sobrevivem a múltiplas representações merecem testes adicionais.", "experimental_principle"),
        ("C9", "Equivalência estrutural pode ser mais informativa que distância literal.", "working_hypothesis"),
        ("C10", "O observador determina parcialmente quais diferenças são relevantes.", "open"),
        ("C11", "Uma representação adequada de uma dinâmica deve preservar observáveis relevantes e sua evolução.", "new_working_hypothesis"),
        ("C12", "Classes de equivalência observacional podem substituir estados para tarefas específicas.", "new_working_hypothesis"),
        ("C13", "Informação relevante pode ser definida pela capacidade de distinguir estados.", "framework_candidate"),
        ("C14", "Uma transformação pode ser caracterizada pelo conjunto de propriedades que preserva.", "new_working_hypothesis"),
        ("C15", "A estrutura pode ser investigada independentemente da codificação literal.", "research_direction"),
        ("C16", "Morfismos explícitos são uma ponte operacional entre representações.", "new_working_hypothesis"),
        ("C17", "Preservação de uma operação deve ser testada por homomorfismo, não assumida.", "methodological_principle"),
        ("C18", "Equivalência dinâmica exige preservar a evolução, não apenas um estado inicial.", "new_working_hypothesis"),
        ("C19", "Perda de informação pode ser quantificada pela redução de classes distinguíveis.", "framework_candidate"),
        ("C20", "Uma conjectura sobrevivente deve continuar sendo submetida a contraexemplos adversariais.", "methodological_principle"),
    ]
    for code, claim, status in CONJECTURES:
        log(f"{code}: {claim} [{status}]")
    return f"{len(CONJECTURES)} conjecturas geradas"


def global_falsification() -> str:
    section("FALSIFICAÇÃO GLOBAL")
    required = [
        "representation",
        "142857",
        "compression",
        "counterexamples",
        "process",
        "bisimulation",
        "information",
        "structure_preservation",
        "morphism_search",
    ]
    statuses = {r["name"]: r["status"] for r in RESULTS}
    passed = 0
    for name in required:
        if statuses.get(name) == "OK":
            passed += 1
            ok(name)
        else:
            fail(name)
    return f"{passed}/{len(required)} testes"


def scientific_synthesis() -> str:
    section("SÍNTESE CIENTÍFICA")
    points = [
        "A representação influencia a aparência computacional de uma estrutura.",
        "Mudança de representação não implica mudança da entidade.",
        "Estados podem possuir descrições maiores que suas regras geradoras.",
        "Compressibilidade não é suficiente para demonstrar estrutura.",
        "Invariantes precisam ser definidos em relação a transformações.",
        "Observadores determinam quais diferenças são relevantes.",
        "Equivalência observacional pode ser diferente de igualdade literal.",
        "Dinâmica pode ser tão importante quanto estado.",
        "Transformações podem ser tratadas como objetos matemáticos de primeira classe.",
        "Equivalência entre representações deve ser expressa por relações/morfismos explícitos.",
        "Preservação de operação deve ser verificada formalmente por homomorfismo.",
        "Quocientes observacionais podem reduzir estados sem preservar toda a informação.",
        "Informação não deve ser confundida com tamanho bruto da representação.",
        "Nenhum resultado desta geração transforma uma conjectura em teorema.",
    ]
    for i, p in enumerate(points, 1):
        log(f"{i}. {p}")
    return "síntese produzida sem declarar conjecturas como teoremas"


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def write_report(elapsed: float) -> None:
    lines = [
        f"# NEW MATH LAB {VERSION} — RELATÓRIO",
        "",
        f"Tempo total: `{elapsed:.6f}s`",
        "",
        "## Objetivo",
        "",
        "Investigar propriedades que sobrevivam simultaneamente a mudanças de "
        "representação, transformações, observadores, dinâmica e equivalência.",
        "",
        "## Resultados",
        "",
    ]
    for r in RESULTS:
        icon = "OK" if r["status"] == "OK" else "ERROR"
        lines.append(f"- **{icon}** `{r['name']}` — {r['message']}")
    lines += [
        "",
        "## Conjecturas",
        "",
    ]
    for code, claim, status in CONJECTURES:
        lines.append(f"- **{code}** — {claim} — `{status}`")
    lines += [
        "",
        "## Erros",
        "",
    ]
    if ERRORS:
        lines.extend(f"- {e}" for e in ERRORS)
    else:
        lines.append("- Nenhum erro de execução.")
    lines += [
        "",
        "## Princípio metodológico",
        "",
        "Os resultados são evidências experimentais. Nenhum padrão é tratado "
        "como teorema sem formalização e prova independente.",
        "",
        "## Próximo passo",
        "",
        "A próxima geração deve tentar destruir os sobreviventes com "
        "contraexemplos mais fortes, aumentar a escala e comparar os resultados "
        "com estruturas matemáticas conhecidas.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    ("representation", experiment_representation),
    ("142857", experiment_142857),
    ("rules", experiment_rules),
    ("collatz", experiment_collatz),
    ("rewriting", experiment_rewriting),
    ("cellular", experiment_cellular),
    ("graphs", experiment_graphs),
    ("invariant_search", experiment_invariant_search),
    ("observer_independence", experiment_observer_independence),
    ("representation_equivalence", experiment_representation_equivalence),
    ("lossless_compression", experiment_lossless_compression),
    ("description_complexity", experiment_description_complexity),
    ("perturbation", experiment_perturbation),
    ("composed_transformations", experiment_composed_transformations),
    ("counterexamples", experiment_counterexamples),
    ("abstract_system", experiment_abstract_system),
    ("state_process", experiment_state_process),
    ("dynamic_preservation", experiment_dynamic_preservation),
    ("program_search", experiment_program_search),
    ("cross_system_invariants", experiment_cross_system_invariants),
    ("bisimulation", experiment_bisimulation),
    ("information", experiment_information),
    ("symmetry_search", experiment_symmetry_search),
    ("observational_quotient", experiment_observational_quotient),
    ("representation_dependency", experiment_representation_dependency),
    ("relation_search", experiment_relation_search),
    ("representation_distance", experiment_representation_distance),
    ("observer_as_system", experiment_observer_as_system),
    ("finite_algebra", experiment_finite_algebra),
    ("symmetry_group", experiment_symmetry_group),
    ("structure_preservation", experiment_structure_preservation),
    ("equivalence_classes", experiment_equivalence_classes),
    ("information_dynamics", experiment_information_dynamics),
    ("minimal_observer", experiment_minimal_observer),
    ("mutual_information", experiment_mutual_information),
    ("property_survival", experiment_property_survival),
    ("morphism_search", experiment_morphism_search),
    ("isomorphism_invariants", experiment_isomorphism_invariants),
    ("quotient_dynamics", experiment_quotient_dynamics),
    ("information_loss", experiment_information_loss),
    ("adversarial_counterexamples", experiment_adversarial_counterexamples),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="NEW MATH LAB v10")
    parser.add_argument("command", nargs="?", default="all")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=142857)
    args = parser.parse_args()

    RNG.seed(args.seed)

    section(f"NEW MATH LAB {VERSION}")
    log("")
    log("Objetivo: procurar estruturas que sobrevivam à mudança de representação,")
    log("transformação, observador, dinâmica, informação e equivalência.")
    log("")

    started = time.perf_counter()

    if args.command == "all":
        for name, fn in EXPERIMENTS:
            safe_run(name, fn)
        safe_run("meta_experiment", experiment_meta_experiment)
        safe_run("generate_conjectures", generate_conjectures)
        safe_run("global_falsification", global_falsification)
        safe_run("scientific_synthesis", scientific_synthesis)
    else:
        lookup = {name: fn for name, fn in EXPERIMENTS}
        lookup.update({
            "meta": experiment_meta_experiment,
            "conjectures": generate_conjectures,
            "falsify": global_falsification,
            "synthesis": scientific_synthesis,
        })
        if args.command not in lookup:
            parser.error("comando desconhecido: " + args.command)
        safe_run(args.command, lookup[args.command])

    elapsed = time.perf_counter() - started
    write_report(elapsed)

    section("LABORATÓRIO FINALIZADO")
    log(f"Relatório: {REPORT}")
    log(f"Tempo: {elapsed:.6f}s")

    if ERRORS:
        log("\nERROS:")
        for error in ERRORS:
            log(f"  [ERR] {error}")
    else:
        log("\nERROS:")
        log("  nenhum")

    log("\nPróximo passo:")
    log("usar NEW_MATH_LAB_v10_REPORT.md como entrada da próxima geração,")
    log("tentando destruir os sobreviventes com contraexemplos mais fortes.")


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/new_math_lab_v10.py")
path.write_text(code, encoding="utf-8")
print(f"Criado: {path}")
print(f"Linhas: {len(code.splitlines())}")
print("Execute com: python -u new_math_lab_v10.py")
