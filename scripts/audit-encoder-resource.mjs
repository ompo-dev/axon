import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const [directory] = process.argv.slice(2);
if (!directory) throw Error('usage: node scripts/audit-encoder-resource.mjs <resource-directory>');
const read = async name => JSON.parse(await fs.readFile(path.join(directory, name)));
const sha = b => createHash('sha256').update(b).digest('hex');
const digest = async name => sha(await fs.readFile(name));
const protocol = await read('protocol.json'), inputs = await read('inputs.json');
assert.equal(protocol.schema, 'axon-matched-process-resource/v1');
assert.equal(inputs.length, protocol.requests);
assert.equal(new Set(inputs.map(r => r.id)).size, inputs.length);
const reference = JSON.parse(await fs.readFile(path.join(protocol.run, 'development-inputs.json'))).slice(0, 64);
assert.equal(protocol.unique_accuracy_cases, reference.length);
assert.deepEqual(inputs, Array.from({length: 4}, (_, n) => reference.map(r => ({id: `${n}-${r.id}`, text: r.text}))).flat());
assert.equal(await digest(path.join(protocol.run, 'candidate.exe')), protocol.binary_sha256);
const receipts = [], models = new Map(), conditions = {};
for (const [repeat, order] of protocol.orders.entries()) {
  assert.equal(new Set(order).size, Object.keys(protocol.conditions).length);
  for (const index of order) {
    assert(Object.hasOwn(protocol.conditions, index));
    const name = `repeat-${repeat}-policy-${index}`, receipt = await read(`${name}.json`);
    assert.equal(receipt.mode, 'curriculum-encoder-batch');
    assert.equal(receipt.exit_code, 0);
    assert.equal(receipt.seed, null); assert.equal(receipt.requested_lab_rows, null);
    assert.equal(receipt.binary_sha256.toLowerCase(), protocol.binary_sha256);
    const number = path.join(protocol.run, `policy-${index}/structural.axnum`);
    assert.equal(path.resolve(receipt.sources), path.resolve(number));
    assert.equal(path.resolve(receipt.evaluation), path.resolve(directory, 'inputs.json'));
    const before = models.get(index) ?? await digest(number);
    assert.equal(await digest(number), before); models.set(index, before);
    const summary = JSON.parse(await fs.readFile(path.join(directory, `${name}.stdout.log`), 'utf8'));
    assert(summary.seed_unchanged && summary.teacher_queries === 0 && summary.loads === 1 && summary.active_modules === 1);
    assert.equal(summary.total, inputs.length);
    const rows = await read(`${name}.predictions.json`);
    const expected = JSON.parse(await fs.readFile(path.join(protocol.run, `policy-${index}-predictions.json`))).slice(0, 64);
    assert.deepEqual(rows, Array.from({length: 4}, (_, n) => expected.map(r => ({...r, id: `${n}-${r.id}`}))).flat());
    for (const field of ['lifecycle_ms', 'samples']) assert(Number.isFinite(receipt[field]) && receipt[field] > 0);
    for (const field of ['max_sampled_working_set_bytes', 'max_sampled_private_bytes', 'max_observed_os_peak_working_set_bytes', 'cpu_ms_last_sample']) {
      assert(receipt[field] === null || Number.isFinite(receipt[field]) && receipt[field] >= 0);
    }
    (conditions[index] ??= []).push({...receipt, logical_work: summary.work});
    receipts.push({repeat, index, sha256: await digest(path.join(directory, `${name}.json`)),
      predictions_sha256: await digest(path.join(directory, `${name}.predictions.json`))});
  }
}
function range(values) {
  const available = values.filter(v => v !== null).sort((a, b) => a - b);
  return available.length ? {min: available[0], median: available[Math.floor(available.length / 2)], max: available.at(-1), samples: available.length} : null;
}
const results = Object.entries(conditions).map(([index, rows]) => ({index: Number(index), number_sha256: models.get(Number(index)),
  processes: rows.length, ...Object.fromEntries(['lifecycle_ms', 'max_sampled_working_set_bytes', 'max_sampled_private_bytes',
    'max_observed_os_peak_working_set_bytes', 'cpu_ms_last_sample', 'logical_work'].map(k => [k, range(rows.map(r => r[k]))]))}));
const result = {schema: 'axon-encoder-resource-audit/v1', protocol_sha256: await digest(path.join(directory, 'protocol.json')),
  sampler_sha256: await digest(path.join(directory, 'sampler.ps1')), inputs_sha256: await digest(path.join(directory, 'inputs.json')),
  auditor_sha256: await digest(process.argv[1]), receipts, results, predictions_identical_to_consumed_development: true,
  new_independent_accuracy_cases: 0, m1_complete: false, energy_joules: null,
  scope: protocol.scope, orchestration_note: protocol.orchestration_note ?? null};
await fs.copyFile(process.argv[1], path.join(directory, 'auditor.mjs'), fs.constants.COPYFILE_EXCL);
await fs.writeFile(path.join(directory, 'audit.json'), JSON.stringify(result, null, 2), {flag: 'wx'});
console.log(JSON.stringify({results, energy_joules: null}));
