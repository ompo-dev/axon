import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { reference } from './audit-algorithms.mjs';

const [numberPath, outputPath, executable] = process.argv.slice(2);
if (!executable) throw Error('usage: bun scripts/benchmark-algorithm-inquiry.mjs <source-only.axnum> <new-output> <candidate.exe>');
const output = path.resolve(outputPath), binary = path.resolve(executable);
await fs.mkdir(output);
const number = path.join(output, 'isolated.axnum');
await fs.copyFile(numberPath, number, fs.constants.COPYFILE_EXCL);
const sha = raw => createHash('sha256').update(raw).digest('hex');
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
let state = 0x63ac781b;
const random = n => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) % n; };
const families = ['numeric-sum', 'numeric-product', 'numeric-minimum', 'numeric-maximum', 'numeric-sum-squares'];
const tasks = [], answers = [];
for (const family of families) for (let replicate = 0; replicate < 10; replicate++) {
  const target = input => reference({ family, bindings: { dados: input } });
  const initial = family === 'numeric-sum-squares' ? [-1, 0, 1] : [-3, 0, 7];
  const pool = [
    [11 + replicate], [-24 - replicate], [38 + replicate],
    [2 + replicate, 5 + replicate, -3], [-9 - replicate, -4, -2],
    [3, 6 + replicate, 11], [0, -4, 7 + replicate, 2],
  ];
  const inputs = Array.from({ length: 16 }, (_, i) => Array.from({ length: 2 + random(7) }, () => i % 4 === 0 ? -1 - random(9) : i % 4 === 1 ? 1 + random(9) : random(19) - 9));
  for (const variant of ['active', 'passive-prefix', 'passive-shuffled']) {
    const ordered = structuredClone(pool);
    if (variant === 'passive-shuffled') for (let i = ordered.length - 1; i > 0; i--) {
      const j = random(i + 1); [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
    }
    const id = `trial-${tasks.length}`;
    tasks.push({ id, train: initial.map(n => ({ input: [n], output: target([n]) })), inputs,
      inquiry: { pool: ordered.map(input => ({ input, output: target(input) })), max_queries: 2, max_hypotheses: 16 }, passive: variant !== 'active' });
    answers.push({ id, family, variant, replicate, expected: inputs.map(target) });
  }
}
await write('tasks.json', tasks);
await write('answers.json', answers);
await write('freeze.json', {
  input_seed: '0x63ac781b', binary_sha256: sha(await fs.readFile(binary)), binary_bytes: (await fs.stat(binary)).size,
  number_sha256: sha(await fs.readFile(number)), tasks_sha256: sha(await fs.readFile(path.join(output, 'tasks.json'))), answers_sha256: sha(await fs.readFile(path.join(output, 'answers.json'))),
  paired_trials: 50, known_families: families, cases_per_variant: 800, max_authorized_queries: 2,
  scope: 'Development ambiguity stress test, not fresh independent families or full-M0 approval. Same compiled priors, initial demonstrations, query pool and total search caps; active selects inputs, passive prefix and pre-shuffled controls have the same two-query opportunity. Only authorized teaching labels enter the child. Future answers stay in answers.json. Logical isolation, not an OS sandbox. Internal work includes committee construction and selection, not physical CPU or oracle implementation cost.',
});
const started = performance.now();
const run = await promisify(execFile)(binary, ['algorithm-external', number, path.join(output, 'birth'), path.join(output, 'tasks.json'), path.join(output, 'predictions.json')], { cwd: output, windowsHide: true, timeout: 300000, maxBuffer: 1000000 });
await fs.writeFile(path.join(output, 'stdout.log'), run.stdout, { flag: 'wx' });
await fs.writeFile(path.join(output, 'stderr.log'), run.stderr, { flag: 'wx' });
const rows = JSON.parse(await fs.readFile(path.join(output, 'predictions.json'), 'utf8'));
if (rows.length !== tasks.length) throw Error('missing task receipts');
const variants = {};
for (let i = 0; i < rows.length; i++) {
  const row = rows[i], answer = answers[i], task = tasks[i];
  if (row.id !== answer.id || row.predictions.length !== answer.expected.length) throw Error('receipt identity/cardinality');
  if (row.work > 30000000 || row.candidates > 50000) throw Error('combined search/selection budget exceeded');
  const observed = row.questions.filter(e => e.phase === 'query_observed');
  if (observed.length > 2 || row.questions.some(e => e.phase === 'query_selected' && e.actual !== null)) throw Error('oracle/selection boundary');
  for (const event of observed) {
    const index = task.inquiry.pool.findIndex(e => JSON.stringify(e.input) === JSON.stringify(event.input));
    if (index < 0 || JSON.stringify(event.actual) !== JSON.stringify(task.inquiry.pool[index].output)) throw Error('unauthorized/misreported answer');
  }
  const correct = row.predictions.filter((p, j) => JSON.stringify(p) === JSON.stringify(answer.expected[j])).length;
  const result = variants[answer.variant] ??= { correct: 0, total: 0, solved: 0, trials: 0, queries: 0, work: 0, failed_acquisitions: 0, families: {} };
  const family = result.families[answer.family] ??= { correct: 0, total: 0, solved: 0, trials: 0 };
  for (const target of [result, family]) { target.correct += correct; target.total += answer.expected.length; target.solved += Number(correct === answer.expected.length); target.trials++; }
  result.queries += observed.length;
  result.work += row.work;
  result.failed_acquisitions += Number(row.program === null);
}
const report = { variants, whole_process_wall_ms: performance.now() - started, predictions_sha256: sha(await fs.readFile(path.join(output, 'predictions.json'))), full_milestone_approval: false,
  scope: 'All future cases and failed acquisitions counted; 50 paired trials of five known families. No promotion, no inference-speed ratio from work counters, no statistical superiority claim from pooled dependent cases. Physical CPU/RAM and ancestral teaching are not measured by this wrapper.' };
await write('comparison.json', report);
console.log(JSON.stringify(report));
