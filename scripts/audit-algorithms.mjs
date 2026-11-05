import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { parse } from '../ui/node_modules/csv-parse/lib/sync.js';
import { m0Reference } from './m0-references.mjs';

const sha = raw => createHash('sha256').update(raw).digest('hex');
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const integer = n => { const x = Number(n); if (!Number.isSafeInteger(x)) throw Error('auditor integer bounds'); return x; };
const sum = v => integer(v.reduce((s, n) => s + BigInt(n), 0n));
const total = rows => sum(rows.map(r => integer(BigInt(r[1]) * BigInt(r[2]))));
const group = rows => {
  const state = new Map();
  for (const [key, n] of rows) {
    if (!['string', 'boolean'].includes(typeof key) && !Number.isSafeInteger(key)) throw Error('auditor scalar key bounds');
    const next = (state.get(key) ?? 0n) + BigInt(n);
    state.delete(key); state.set(key, next);
  }
  return [...state].map(([key, n]) => [key, integer(n)]);
};
export function reference(r) {
  const x = r.bindings.dados;
  const select = () => x[0].filter(v => equal(v, x[1]));
  const family = r.family;
  if (family.startsWith('composition-')) return reference({ ...r, family: family.slice(12), bindings: { dados: select() } });
  switch (family) {
    case 'state-source-wrap': case 'state-source-prepend': case 'state-source-initial':
    case 'state-source-history': case 'state-source-modular': case 'state-source-total': case 'state-source-remainder':
    case 'order-reversal': case 'cumulative-state-history': case 'modulus-changing-state': return m0Reference(family, x);
    case 'transition-source-0': case 'transition-source-1': case 'transition-source-2':
    case 'linear-recurrence': case 'quadratic-recurrence': case 'bilinear-recurrence': return m0Reference(family, x);
    case 'numeric-sum': return sum(x);
    case 'numeric-sum-squares': return sum(x.map(n => integer(BigInt(n) ** 2n)));
    case 'numeric-product': return integer(x.reduce((s, n) => s * BigInt(n), 1n));
    case 'numeric-maximum': return x.reduce((a, b) => a > b ? a : b);
    case 'numeric-minimum': return x.reduce((a, b) => a < b ? a : b);
    case 'numeric-equality-count': case 'text-equality-count': return select().length;
    case 'numeric-equality-filter': case 'text-equality-filter': return select();
    case 'numeric-conditional-sum': return sum(x[0].filter(n => n > x[1]));
    case 'numeric-conditional-count': return x[0].filter(n => n > x[1]).length;
    case 'record-weighted-reduction': return total(x);
    case 'record-conditional-filter': return x[0].filter(row => row[2] > x[1]);
    case 'record-learned-pipeline': return total(x[0].filter(row => row[2] > x[1]));
    case 'visual-count-via-learned-call': return x[0].flat().filter(n => n === x[1]).length;
    case 'visual-conditional-nested-loop': return x.map(row => row.map(n => n === 0 ? 4 : n));
    case 'longer-than-training': return r.id.startsWith('large-count') ? select().length : sum(x);
    case 'written-learned-composition': return sum(select());
    case 'unsignalled-change': return sum(x) + x.length;
    case 'retention-after-change': return sum(x.map(n => integer(BigInt(n) ** 2n)));
    case 'dependency-map': return x.map(n => integer(BigInt(n) ** 2n));
    case 'dependency-select-map': return select().map(n => integer(BigInt(n) ** 2n));
    case 'dependency-report-original': case 'dependency-report-independent': return sum(select().map(n => integer(BigInt(n) ** 2n)));
    case 'dependency-report-changed': return sum(select().map(n => integer(BigInt(n) ** 2n))) + select().length;
    case 'dependency-invalidated': return null;
    case 'abstraction-weighted-list': return x[0].filter(row => BigInt(row[0]) < BigInt(x[1])).map(row => integer(BigInt(row[1]) * BigInt(row[2])));
    case 'abstraction-weighted-total': return total(x[0].filter(row => BigInt(row[0]) < BigInt(x[1])));
    case 'stateful-query': return sum(x[0].filter(row => equal(row[0], x[1])).map(row => row[1]));
    case 'stateful-without': return x[0].filter(row => !equal(row[0], x[1]));
    case 'stateful-project': return [x[0], integer(BigInt(x[1]) * BigInt(x[2]))];
    case 'stateful-update': return [...x[0].filter(row => !equal(row[0], x[1][0])), [x[1][0], sum(x[0].filter(row => equal(row[0], x[1][0])).map(row => row[1])) + x[1][1]]];
    case 'stateful-key-total': return group(x);
    case 'stateful-project-map': return x.map(row => [row[0], integer(BigInt(row[1]) * BigInt(row[2]))]);
    case 'stateful-weighted-group': return group(x.map(row => [row[0], integer(BigInt(row[1]) * BigInt(row[2]))]));
    case 'stateful-unit': return [x, 1];
    case 'stateful-unit-map': return x.map(key => [key, 1]);
    case 'stateful-histogram': return group(x.map(key => [key, 1]));
    case 'unsupported-language': case 'unsupported-binding': return null;
    default: throw Error(`Unknown independent reference: ${family}`);
  }
}
export async function audit(file, output) {
  const raw = await fs.readFile(file), rows = JSON.parse(raw), families = {};
  let correct = 0;
  for (const row of rows) {
    const expected = reference(row);
    if (!equal(expected, row.expected)) throw Error(`Untrusted generated label disagrees: ${row.id}`);
    const match = equal(expected, row.actual);
    correct += Number(match);
    const f = families[row.family] ??= { correct: 0, total: 0 };
    f.total++; f.correct += Number(match);
    if (match !== row.correct) throw Error(`Receipt scoring mismatch: ${row.id}`);
  }
  const report = { correct, total: rows.length, families, predictions_sha256: sha(raw), scope: 'Independent BigInt/array reference; no AXON AST or expected labels used as computational oracle. Same task semantics, not a proof of general intelligence.' };
  const csv = path.join(path.dirname(file), 'algorithm-input.csv');
  try {
    const input = await fs.readFile(csv), records = parse(input, { columns: true });
    const actual = JSON.parse(await fs.readFile(path.join(path.dirname(file), 'algorithm-file-result.json')));
    const parameter = actual.parameter ?? 12;
    if (!Number.isSafeInteger(parameter)) throw Error('Missing numeric CSV parameter');
    const expected = total(records.filter(r => BigInt(r.valor) > BigInt(parameter)).map(r => [r.pedido, r.quantidade, r.valor]));
    report.file = { rows: records.length, expected, actual: actual.actual, correct: expected === actual.actual, input_sha256: sha(input), threshold: parameter };
  } catch (e) { if (e.code !== 'ENOENT') throw e; }
  await fs.writeFile(output, JSON.stringify(report, null, 2), { flag: 'wx' });
  if (correct !== rows.length || report.file?.correct === false) process.exitCode = 1;
  return report;
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const [file, output] = process.argv.slice(2);
  if (!output) throw Error('usage: bun scripts/audit-algorithms.mjs <predictions.json> <new-audit.json>');
  console.log(JSON.stringify(await audit(file, output)));
}
