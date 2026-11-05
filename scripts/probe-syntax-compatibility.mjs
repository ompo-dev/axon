import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
const [executable, destination] = process.argv.slice(2);
if (!destination) throw Error('usage: node scripts/probe-syntax-compatibility.mjs <candidate> <new-output>');
const output = path.resolve(destination); await fs.mkdir(output);
const binary = path.join(output, 'candidate.exe'); await fs.copyFile(executable, binary, fs.constants.COPYFILE_EXCL);
const sha = b => createHash('sha256').update(b).digest('hex');
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), {flag: 'wx'});
const cases = [
  ['legacy_btree', 'dependency-batch', '.axon-memory/m1-public-syntax-v2', 'policy-0', 903],
  ['legacy_contiguous', 'dependency-batch', '.axon-memory/m1-public-syntax-v3', 'policy-3', 903],
  ['modern', 'dependency-batch', '.axon-memory/m1-public-syntax-reproduction-v1', 'policy-0', 903],
  ['previous_factorized', 'factorized-batch', '.axon-memory/m1-public-factorized-v3', 'policy-1', 1504],
];
const identities = {}, sources = {};
for (const file of ['src/demand.rs', 'src/genome.rs', 'src/structural_dependency.rs', 'src/structural_dependency_lab.rs', 'src/structural_lab.rs', 'src/curriculum.rs', 'scripts/probe-syntax-compatibility.mjs']) {
  const bytes = await fs.readFile(file), dest = path.join(output, 'source', file); sources[file] = sha(bytes);
  await fs.mkdir(path.dirname(dest), {recursive: true}); await fs.writeFile(dest, bytes, {flag: 'wx'});
}
for (const [name, , run, policy] of cases) {
  for (const [source, suffix] of [[`${run}/${policy}/structural.axnum`, 'axnum'], [`${run}/development-inputs.json`, 'inputs.json'], [`${run}/${policy}-predictions.json`, 'expected.json']]) {
    const bytes = await fs.readFile(source); identities[`${name}.${suffix}`] = sha(bytes);
    await fs.writeFile(path.join(output, `${name}.${suffix}`), bytes, {flag: 'wx'});
  }
}
await write('freeze.json', {binary_sha256: sha(await fs.readFile(binary)), sources, identities, scope: 'Exact compatibility against frozen, already consumed predictions. Expected outputs remain in host; only id/text reach child. Not a new generalization test.'});
const results = [];
for (const [name, command, , , total] of cases) {
  const inputs = JSON.parse(await fs.readFile(path.join(output, `${name}.inputs.json`)));
  assert(inputs.every(row => Object.keys(row).sort().join(',') === 'id,text'));
  const run = await promisify(execFile)(binary, [command, path.join(output, `${name}.axnum`), path.join(output, `${name}-birth`), path.join(output, `${name}.inputs.json`), path.join(output, `${name}-predictions.json`)], {cwd: output, windowsHide: true, timeout: 300000, maxBuffer: 64000000});
  await fs.writeFile(path.join(output, `${name}.stdout.log`), run.stdout, {flag: 'wx'}); await fs.writeFile(path.join(output, `${name}.stderr.log`), run.stderr, {flag: 'wx'});
  const expected = JSON.parse(await fs.readFile(path.join(output, `${name}.expected.json`))), actual = JSON.parse(await fs.readFile(path.join(output, `${name}-predictions.json`)));
  assert.equal(actual.length, total); assert.deepEqual(actual, expected);
  const summary = JSON.parse(run.stdout.trim().split(/\r?\n/u).at(-1)); assert(summary.seed_unchanged && summary.teacher_queries === 0);
  results.push({name, total, identical: total, seed_unchanged: true, teacher_queries: 0});
}
await write('audit.json', {results, m1_complete: false, source_format_and_residency_compatible: true}); console.log(JSON.stringify({results}));
