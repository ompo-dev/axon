"""Independent exact symbolic checker; never used by the AXON learner."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("predictions", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--dependency-path", type=Path)
args = parser.parse_args()
if args.dependency_path:
    sys.path.insert(0, str(args.dependency_path.resolve()))
import sympy as sp

symbols = {}


def expression(tree):
    op, *xs = tree
    if op == "num":
        return sp.Integer(xs[0])
    if op == "var":
        return symbols.setdefault(xs[0], sp.Symbol(xs[0], real=True))
    values = [expression(t) for t in xs]
    if op == "add":
        return values[0] + values[1]
    if op == "sub":
        return values[0] - values[1]
    if op == "mul":
        return values[0] * values[1]
    if op == "div":
        return values[0] / values[1]
    if op == "pow":
        return values[0] ** values[1]
    if op == "sin":
        return sp.sin(values[0])
    if op == "cos":
        return sp.cos(values[0])
    if op == "exp":
        return sp.exp(values[0])
    if op == "d":
        return sp.diff(values[0], values[1])
    if op == "lim":
        return sp.limit(values[0], values[1], values[2])
    raise ValueError(f"Unknown exact symbolic operation: {op}")


started = time.monotonic()
raw = args.predictions.read_bytes()
rows = json.loads(raw)
results = []
for row in rows:
    if row["family"] not in {
        "symbolic-derivative", "quotient-derivative", "continuous-limit",
        "removable-limit", "linear-equation",
        "advanced-probe",
        "derivative-order-2", "derivative-order-3", "discovered-program",
    }:
        continue
    error = None
    try:
        request = row["input"]
        actual = expression(row["actual"])
        if row["family"] == "discovered-program":
            a, b = map(sp.Integer, request[1:])
            expected = {"circuit-a": lambda: a*a+b,
                        "circuit-b": lambda: (a-b)**2,
                        "circuit-c": lambda: a*a+b*b,
                        "circuit-d": lambda: sp.sqrt(a*a+b*b)}[request[0]]()
        elif request[0] == "d":
            expected = sp.diff(expression(request[1]), expression(request[2]))
        elif request[0] == "lim":
            f, x, a = map(expression, request[1:])
            expected = (sp.limit(f, x, a) if row["family"] in {"removable-limit", "advanced-probe"}
                        else f.subs(x, a))
        elif request[0] == "solve":
            left, right, x = map(expression, request[1:])
            solutions = sp.solve(sp.Eq(left, right), x)
            if len(solutions) != 1:
                raise ValueError("Equation does not have one solution")
            expected = solutions[0]
        else:
            raise ValueError("Unknown request")
        valid = actual == expected or sp.simplify(actual - expected) == 0
        if actual.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            valid = False
    except Exception as exc:
        valid = False
        error = f"{type(exc).__name__}: {exc}"
    results.append({"id": row["id"], "family": row["family"], "correct": bool(valid), "error": error})
    if len(results) % 200 == 0:
        print(json.dumps({"checked": len(results), "correct": sum(r["correct"] for r in results)}), flush=True)

report = {
    "schema": "axon-independent-symbolic-audit/v1",
    "sympy_version": sp.__version__,
    "prediction_sha256": hashlib.sha256(raw).hexdigest(),
    "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "checked": len(results), "correct": sum(r["correct"] for r in results),
    "elapsed_seconds": time.monotonic() - started,
    "scope": "Taught algebra/calculus on declared expression families, not general theorem proving or autonomous discovery. Expected AST labels ignored by this checker.",
    "cases": results,
}
with args.output.open("x", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=True, indent=2)
print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
if report["correct"] != report["checked"]:
    sys.exit(1)
