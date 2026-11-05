import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';

const [freezePath, numberPath, repositoryPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw Error('usage: bun scripts/benchmark-arc-algorithms.mjs <freeze.json> <number.axnum> <ARC-AGI-checkout> <new-output>');
const frozen = JSON.parse(await fs.readFile(freezePath));
const binary = path.resolve(frozen.binary), sha = v => createHash('sha256').update(v).digest('hex');
if (sha(await fs.readFile(binary)) !== frozen.binary_sha256.toLowerCase() || sha(await fs.readFile(numberPath)) !== frozen.number_sha256.toLowerCase()) throw Error('Frozen artifact changed');
const output = path.resolve(outputPath); await fs.mkdir(output);
const invoke = promisify(execFile), repository = path.resolve(repositoryPath);
const revision = (await invoke('git', ['rev-parse', 'HEAD'], { cwd: repository, windowsHide: true })).stdout.trim();
const folder = path.join(repository, 'data/evaluation');
const files = (await fs.readdir(folder)).filter(f => /^[0-9a-f]+\.json$/.test(f)).sort();
if (files.length !== 400) throw Error('Expected all 400 official evaluation tasks');
const inputs = [], manifest = [], labels = new Map(), baseline = new Map();
for (const file of files) {
  const raw = await fs.readFile(path.join(folder, file)), task = JSON.parse(raw), id = file.slice(0, -5);
  manifest.push({ file, sha256: sha(raw) });
  inputs.push({ id, train: task.train, inputs: task.test.map(p => p.input) });
  labels.set(id, task.test.map(p => p.output));
  const constant = task.train.every(p => JSON.stringify(p.output) === JSON.stringify(task.train[0].output));
  baseline.set(id, { identity: task.test.map(p => p.input), constant: task.test.map(() => constant ? task.train[0].output : null) });
}
await fs.writeFile(path.join(output, 'provenance.json'), JSON.stringify({ source: 'https://github.com/fchollet/ARC-AGI', revision, license: 'Apache-2.0', subset: 'evaluation', tasks: files.length, manifest, freeze: frozen, scope: 'All tasks; one prediction, exact all-test-grid match. No task-specific tuning or retries after scoring. Worker receives demonstrations and test inputs, never test outputs. Process separation is not an OS security sandbox.' }, null, 2), { flag: 'wx' });
const isolated = path.join(output, 'worker'); await fs.mkdir(isolated);
const number = path.join(isolated, 'isolated.axnum'), cases = path.join(isolated, 'inputs-only.json'), predictions = path.join(isolated, 'predictions.json');
await fs.copyFile(numberPath, number, fs.constants.COPYFILE_EXCL);
await fs.writeFile(cases, JSON.stringify(inputs), { flag: 'wx' });
const started = performance.now();
const execution = await invoke(binary, ['algorithm-external', number, path.join(isolated, 'birth'), cases, predictions], { cwd: isolated, windowsHide: true, timeout: 600000, maxBuffer: 1000000 });
const rows = JSON.parse(await fs.readFile(predictions)), details = [];
const scores = { axon: 0, identity: 0, constant: 0 };
if (rows.length !== files.length || new Set(rows.map(r => r.id)).size !== files.length) throw Error('Missing/duplicate predictions');
for (const row of rows) {
  const expected = labels.get(row.id), b = baseline.get(row.id);
  if (!expected) throw Error('Unknown task prediction');
  const match = actual => JSON.stringify(actual) === JSON.stringify(expected);
  const score = { axon: match(row.predictions), identity: match(b.identity), constant: match(b.constant) };
  for (const k of Object.keys(scores)) scores[k] += Number(score[k]);
  details.push({ id: row.id, ...score, error: row.error, work: row.work, candidates: row.candidates });
}
const report = { total: files.length, correct: scores, coverage: rows.filter(r => !r.error).length, work: rows.reduce((a, r) => a + r.work, 0), elapsed_ms: performance.now() - started, predictions_sha256: sha(await fs.readFile(predictions)), inputs_sha256: sha(await fs.readFile(cases)), details, worker_summary: JSON.parse(execution.stdout), scope: 'External diagnostic, not official leaderboard scoring; ARC-AGI-1 allows up to three trials, this harness uses one. No LLM/API teacher. Insufficient demonstrations and unsupported programs count as failures, not omitted tasks.' };
await fs.writeFile(path.join(output, 'arc-result.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ ...report, details: undefined }));
