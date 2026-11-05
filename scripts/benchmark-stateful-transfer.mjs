import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { reference } from './audit-algorithms.mjs';

const [numberPath, outputPath, candidatePath, baselinePath] = process.argv.slice(2);
if (!baselinePath) throw Error('usage: bun scripts/benchmark-stateful-transfer.mjs <source-only.axnum> <new-output> <candidate.exe> <baseline.exe>');
const output = path.resolve(outputPath); await fs.mkdir(output);
const binaries = { baseline: path.resolve(baselinePath), candidate: path.resolve(candidatePath) };
const number = path.join(output, 'isolated.axnum');
await fs.copyFile(numberPath, number, fs.constants.COPYFILE_EXCL);
const sha = raw => createHash('sha256').update(raw).digest('hex');
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
let state = 0x759c198b;
const random = n => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) % n; };
const tasks = [], answers = [];
function key(group, replicate, i) {
  return replicate % 3 === 0 ? `unseen-${group}-${i}` : replicate % 3 === 1 ? i - 7 : i % 3 === 0 ? false : i % 3 === 1 ? true : `other-${i}`;
}
function input(group, replicate, length, family) {
  return Array.from({ length }, () => {
    const row = [key(group, replicate, random(13)), random(61) - 30];
    if (family === 'stateful-project-map') row.push(random(41) - 20);
    return row;
  });
}
for (const family of ['stateful-key-total', 'stateful-project-map']) for (let replicate = 0; replicate < 12; replicate++) {
  const id = `task-${tasks.length}`;
  const make = (length, group) => input(group, replicate, length, family);
  const expected = x => reference({ family, bindings: { dados: x } });
  const trainInputs = [make(0, id), make(9, id), make(13, id)];
  const inputs = Array.from({ length: 16 }, (_, i) => make(i === 0 ? 0 : 1 + random(80), `${id}-test-${i}`));
  tasks.push({ id, train: trainInputs.map(input => ({ input, output: expected(input) })), inputs });
  answers.push({ id, family, expected: inputs.map(expected) });
}
await write('tasks.json', tasks); await write('answers.json', answers);
const prefix = JSON.parse(await fs.readFile('evals/curricula/stateful-prefix-v1-data.json', 'utf8'));
await write('source-requests.json', prefix.requests);
await write('freeze.json', {
  number_sha256: sha(await fs.readFile(number)), tasks_sha256: sha(await fs.readFile(path.join(output, 'tasks.json'))), answers_sha256: sha(await fs.readFile(path.join(output, 'answers.json'))),
  binaries: Object.fromEntries(await Promise.all(Object.entries(binaries).map(async ([name, binary]) => [name, { sha256: sha(await fs.readFile(binary)), bytes: (await fs.stat(binary)).size }]))),
  input_seed: '0x759c198b', tasks: tasks.length, cases: answers.reduce((s, t) => s + t.expected.length, 0),
  scope: 'Matched compiled priors and demonstrations; 24 trials of two declared development families, not 24 independent algorithms or a fresh full-M0 reserve. Test labels remain outside the child payload. Scalar-key types and lengths vary; no general-language claim. Logical directory isolation, not an OS sandbox.',
});
const results = [];
for (const [name, binary] of Object.entries(binaries)) {
  const run = async (args, log) => {
    const result = await promisify(execFile)(binary, args, { cwd: output, windowsHide: true, timeout: 300000, maxBuffer: 1000000 });
    await fs.writeFile(path.join(output, `${log}.stdout.log`), result.stdout, { flag: 'wx' });
    await fs.writeFile(path.join(output, `${log}.stderr.log`), result.stderr, { flag: 'wx' });
  };
  const compatibility = path.join(output, `${name}-source-predictions.json`);
  await run(['algorithm-batch', number, path.join(output, `${name}-source-birth`), path.join(output, 'source-requests.json'), compatibility], `${name}-source`);
  const checked = JSON.parse(await fs.readFile(compatibility, 'utf8'));
  if (checked.length !== prefix.requests.length || checked.some(r => !r.correct)) throw Error(`${name}: source model incompatibility, not a target-acquisition score`);
  const receipt = path.join(output, `${name}-predictions.json`);
  await run(['algorithm-external', number, path.join(output, `${name}-birth`), path.join(output, 'tasks.json'), receipt], name);
  const rows = JSON.parse(await fs.readFile(receipt, 'utf8'));
  if (rows.length !== tasks.length) throw Error('missing external task rows');
  let correct = 0, solved = 0;
  const families = {};
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].id !== answers[i].id || rows[i].predictions.length !== answers[i].expected.length) throw Error('external task identity/cardinality');
    const matches = rows[i].predictions.map((p, j) => JSON.stringify(p) === JSON.stringify(answers[i].expected[j]));
    correct += matches.filter(Boolean).length; solved += Number(matches.every(Boolean));
    const f = families[answers[i].family] ??= { correct: 0, total: 0, solved: 0, tasks: 0 };
    f.correct += matches.filter(Boolean).length; f.total += matches.length; f.solved += Number(matches.every(Boolean)); f.tasks++;
  }
  results.push({ variant: name, source_compatibility: true, correct, total: answers.reduce((s, r) => s + r.expected.length, 0), solved, tasks: rows.length, work: rows.reduce((s, r) => s + r.work, 0), proposed: rows.filter(r => r.program != null).length, families });
}
await write('comparison.json', { results, scope: 'All target cases, refusals and failed acquisitions counted. Provided compiled priors and study scope are in freeze.json; ancestry is not cost-free.' });
console.log(JSON.stringify({ results }));
if (results.find(r => r.variant === 'candidate').correct !== answers.reduce((s, r) => s + r.expected.length, 0)) process.exitCode = 1;
