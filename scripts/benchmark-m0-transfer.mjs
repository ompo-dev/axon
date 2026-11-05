import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { m0Reference } from './m0-references.mjs';

const [sourceNumber, outputPath, executable, dataPath = 'evals/curricula/m0-transfer-v1'] = process.argv.slice(2);
if (!executable) throw Error('usage: bun scripts/benchmark-m0-transfer.mjs <source-only.axnum> <new-output> <candidate.exe> [data-directory]');
const output = path.resolve(outputPath);
await fs.mkdir(output);
const binary = path.join(output,'candidate.exe');
await fs.copyFile(path.resolve(executable),binary,fs.constants.COPYFILE_EXCL);
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
const sha = raw => createHash('sha256').update(raw).digest('hex');
const number = path.join(output, 'source-only.axnum');
await fs.copyFile(sourceNumber, number, fs.constants.COPYFILE_EXCL);
const sourceCompiled = path.join(path.dirname(path.resolve(sourceNumber)), 'structural.axstruct');
const sourceRaw = await fs.readFile(sourceCompiled), sourceSeed = JSON.parse(sourceRaw);
const metadata = JSON.parse(await fs.readFile(path.join(dataPath,'data-freeze.json'),'utf8'));
const targetIds = metadata.reserve_target_ids ?? [0,1,2].map(i=>`algorithm-recurrence-${i}`);
const sourceIds = metadata.required_source_ids ?? [0,1,2].map(i=>`algorithm-transition-${i}`);
if (targetIds.length !== 3 || sourceIds.length < 3 || sourceSeed.rules.some(r => targetIds.includes(r.id)) || !sourceIds.every(id => sourceSeed.rules.some(r => r.id === id))) throw Error('source seed contains targets or lacks source procedures');
await fs.writeFile(path.join(output, 'source.axstruct'), sourceRaw, { flag: 'wx' });
for (const name of ['external-inputs.json', 'external-answers.json', 'data-freeze.json']) await fs.copyFile(path.join(dataPath, name), path.join(output, name), fs.constants.COPYFILE_EXCL);
const inputs = path.join(output, 'external-inputs.json'), answersFile = path.join(output, 'external-answers.json');
const tasks = JSON.parse(await fs.readFile(inputs, 'utf8')), answers = JSON.parse(await fs.readFile(answersFile, 'utf8'));
if (tasks.length !== 96 || answers.length !== tasks.length) throw Error('prospective task count');
const identities = {};
for (const [name, file] of Object.entries({ binary, source_number: number, compiled_sources: path.join(output, 'source.axstruct'), tasks: inputs, private_answers: answersFile,
  protocol: metadata.protocol ?? 'evals/protocols/m0-completion-v1.json', evaluator: 'scripts/benchmark-m0-transfer.mjs', reference: 'scripts/m0-references.mjs', generator: metadata.generator ?? 'scripts/create-m0-transfer-curriculum.mjs' })) {
  const raw = await fs.readFile(file); identities[name] = { sha256: sha(raw), bytes: raw.length };
  if (!['binary','source_number','compiled_sources','tasks','private_answers'].includes(name)) await fs.writeFile(path.join(output,`frozen-${name}${path.extname(file)}`),raw,{flag:'wx'});
}
await write('freeze.json', { identities, frozen_before_execution: true, expected_cases_per_variant: 768,
  scope: metadata.scope ?? 'Three prospectively withheld temporal recurrence classes, with 16 paired input-set replicates per class. Source-only seed contains source transitions but no recurrence target lessons. Same initial seed, examples and caps in inherited and no-library controls. Answers are not passed to the child process; no OS sandbox claim.' });
const started = performance.now();
const run = await promisify(execFile)(binary, ['algorithm-external', number, path.join(output, 'birth'), inputs, path.join(output, 'predictions.json')], { cwd: output, windowsHide: true, timeout: 300000, maxBuffer: 1000000 });
await fs.writeFile(path.join(output, 'stdout.log'), run.stdout, { flag: 'wx' });
await fs.writeFile(path.join(output, 'stderr.log'), run.stderr, { flag: 'wx' });
const rows = JSON.parse(await fs.readFile(path.join(output, 'predictions.json'), 'utf8'));
if (sha(await fs.readFile(path.join(output, 'birth', 'structural.axstruct'))) !== sha(sourceRaw)) throw Error('number did not reproduce the frozen source-only seed');
if (rows.length !== tasks.length) throw Error('missing task receipt');
const variants = {};
for (let i = 0; i < rows.length; i++) {
  const row = rows[i], task = tasks[i], answer = answers[i];
  if (row.id !== task.id || answer.id !== task.id || row.predictions.length !== task.inputs.length || row.inherit !== task.inherit || row.work > 30000000 || row.candidates > 50000) throw Error('identity, cardinality or cap mismatch');
  if (!task.inherit && row.library_programs !== 0) throw Error('no-library control contaminated');
  const expected = task.inputs.map(input => m0Reference(answer.family, input));
  if (JSON.stringify(expected) !== JSON.stringify(answer.expected)) throw Error('teacher label disagrees with independent reference');
  const correct = row.predictions.filter((value, j) => JSON.stringify(value) === JSON.stringify(expected[j])).length;
  const name = task.inherit ? 'inherited' : 'no-library';
  const result = variants[name] ??= { correct: 0, total: 0, solved: 0, trials: 0, work: 0, candidates: 0, failed_acquisitions: 0, families: {} };
  const family = result.families[answer.family] ??= { correct: 0, total: 0, solved: 0, trials: 0 };
  for (const r of [result, family]) { r.correct += correct; r.total += expected.length; r.solved += Number(correct === expected.length); r.trials++; }
  result.work += row.work; result.candidates += row.candidates; result.failed_acquisitions += Number(row.program === null);
}
const passed = variants.inherited.correct === 768 && variants.inherited.solved === 48 && variants.inherited.correct > variants['no-library'].correct;
await write('comparison.json', { variants, acquisition_transfer_gates_passed: passed, full_milestone_approval: false,
  whole_harness_wall_ms: performance.now() - started, predictions_sha256: sha(await fs.readFile(path.join(output, 'predictions.json'))),
  scope: 'All cases, refusals and failed acquisition counted. This evaluation tests a bounded declared family split, not universal transfer, free language, ARC improvement or historical mathematical novelty. Remaining G1--G10 need separate receipts.' });
console.log(JSON.stringify({ variants, acquisition_transfer_gates_passed: passed }));
if (!passed) process.exitCode = 1;
