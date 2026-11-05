import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {annotation} from './massive-grammar.mjs';

const [campaignPath, corpusPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw Error('usage: bun scripts/evaluate-m1-public-reserve.mjs <campaign> <corpus> <output>');

const campaign = path.resolve(campaignPath);
const corpus = path.resolve(corpusPath);
const output = path.resolve(outputPath);
await fs.mkdir(output, {recursive: false});
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const readJson = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const writeJson = async (name, value) => fs.writeFile(
  path.join(output, name), JSON.stringify(value, null, 2), {flag: 'wx'},
);

const audit = await readJson(path.join(campaign, 'audit.json'));
const freeze = await readJson(path.join(campaign, 'freeze.json'));
assert.equal(audit.schema, 'axon-public-grammar/v1');
const selected = audit.selected;
assert(Number.isSafeInteger(selected));
const binary = path.join(campaign, 'candidate.exe');
const model = path.join(campaign, `policy-${selected}`, 'structural.axnum');
const protocol = await readJson(path.join(campaign, 'source/evals/protocols/m1-completion-v1.json'));
const gate = protocol.gates.find(item => item.id === 'external-language');
assert(gate?.required);
const provenanceBytes = await fs.readFile(path.join(corpus, 'provenance.json'));
const provenance = JSON.parse(provenanceBytes);
const raw = await fs.readFile(path.join(corpus, '1.1/data/pt-PT.jsonl'));
assert.equal(sha(raw), provenance.data_sha256);
const rows = raw.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse);
const reservedWorkers = new Set(provenance.reserved_workers);
const groups = {
  reserved_worker: rows.filter(row => reservedWorkers.has(row.worker_id)),
  original_test_nonreserved: rows.filter(row => row.partition === 'test' && !reservedWorkers.has(row.worker_id)),
};
const allIds = new Set();
for (const [name, entries] of Object.entries(groups)) {
  assert(entries.length > 0, `empty reserve group: ${name}`);
  for (const row of entries) {
    assert(!allIds.has(row.id), `overlapping reserve row: ${row.id}`);
    allIds.add(row.id);
    assert.equal(typeof row.utt, 'string');
  }
}

const exec = promisify(execFile);
const invoke = async (name, args) => {
  const started = performance.now();
  const result = await exec(binary, args, {
    cwd: output,
    windowsHide: true,
    timeout: 600000,
    maxBuffer: 64000000,
    env: {...process.env, AXON_LAB_TELEMETRY: '1'},
  });
  await fs.writeFile(path.join(output, `${name}.stdout.jsonl`), result.stdout, {flag: 'wx'});
  await fs.writeFile(path.join(output, `${name}.stderr.log`), result.stderr, {flag: 'wx'});
  const events = result.stdout.trim().split(/\r?\n/u).filter(Boolean).map(JSON.parse);
  const summary = events.pop();
  assert(summary);
  events.forEach((event, index) => assert.equal(event.coreSeq, index + 1));
  assert.equal(summary.seed_unchanged, true);
  assert.equal(summary.teacher_queries, 0);
  assert.equal(summary.loads, 1);
  return {summary, elapsed_ms: performance.now() - started};
};

const metrics = (predictions, labelled) => {
  assert.equal(predictions.length, labelled.length);
  let answered = 0, exact = 0, intent = 0, slots = 0;
  const families = {};
  predictions.forEach((row, index) => {
    const target = labelled[index].target;
    assert.equal(String(row.id), String(labelled[index].id));
    const family = families[target[1]] ??= {total: 0, answered: 0, exact: 0};
    family.total++;
    if (row.prediction === null) return;
    answered++; family.answered++;
    const same = JSON.stringify(row.prediction) === JSON.stringify(target);
    exact += Number(same); family.exact += Number(same);
    intent += Number(Array.isArray(row.prediction) && row.prediction[1] === target[1]);
    slots += Number(Array.isArray(row.prediction) && JSON.stringify(row.prediction[2]) === JSON.stringify(target[2]));
  });
  return {
    total: labelled.length, answered, abstained: labelled.length - answered,
    exact, intent_correct: intent, slots_exact: slots,
    exact_accuracy: exact / labelled.length,
    intent_accuracy: intent / labelled.length,
    coverage: answered / labelled.length,
    minimum_family_accuracy: Math.min(...Object.values(families).map(f => f.exact / f.total)),
    families,
  };
};

const results = {};
for (const [name, sourceRows] of Object.entries(groups)) {
  const labelled = sourceRows.map(row => ({id: String(row.id), target: annotation(row).target}));
  const predictions = [];
  const calls = [];
  for (let start = 0; start < sourceRows.length; start += 128) {
    const suffix = `${name}-${start}`;
    const inputs = sourceRows.slice(start, start + 128).map(row => ({id: String(row.id), text: row.utt}));
    const inputFile = path.join(output, `${suffix}-inputs.json`);
    const receipt = path.join(output, `${suffix}-predictions.json`);
    await fs.writeFile(inputFile, JSON.stringify(inputs), {flag: 'wx'});
    const measured = await invoke(suffix, [
      'encoder-batch', model, path.join(output, `${suffix}-birth`), inputFile, receipt,
    ]);
    calls.push(measured);
    predictions.push(...await readJson(receipt));
  }
  await writeJson(`${name}-labels.json`, labelled);
  await writeJson(`${name}-predictions.json`, predictions);
  results[name] = {
    metrics: metrics(predictions, labelled),
    chunks: calls.length,
    elapsed_ms: calls.reduce((sum, call) => sum + call.elapsed_ms, 0),
    core_work: calls.reduce((sum, call) => sum + call.summary.work, 0),
  };
}

const combined = {
  total: Object.values(results).reduce((sum, result) => sum + result.metrics.total, 0),
  exact: Object.values(results).reduce((sum, result) => sum + result.metrics.exact, 0),
  intent_correct: Object.values(results).reduce((sum, result) => sum + result.metrics.intent_correct, 0),
  answered: Object.values(results).reduce((sum, result) => sum + result.metrics.answered, 0),
};
const checks = {
  minimum_intent_accuracy: combined.intent_correct / combined.total >= gate.minimum_intent_accuracy,
  minimum_exact_structured_accuracy: combined.exact / combined.total >= gate.minimum_exact_structured_accuracy,
  minimum_coverage: combined.answered / combined.total >= gate.minimum_coverage,
};
const report = {
  schema: 'axon-m1-public-reserve-audit/v1',
  campaign: campaignPath,
  selected_policy: selected,
  binary_sha256: sha(await fs.readFile(binary)),
  model_sha256: sha(await fs.readFile(model)),
  corpus_sha256: sha(raw),
  provenance_sha256: sha(provenanceBytes),
  frozen_corpus_sha256: freeze.corpus_sha256,
  groups: results,
  combined: {
    ...combined,
    exact_accuracy: combined.exact / combined.total,
    intent_accuracy: combined.intent_correct / combined.total,
    coverage: combined.answered / combined.total,
  },
  checks,
  reserved_evaluated: true,
  original_test_evaluated: true,
  labels_sent_to_core: false,
  teacher_queries: 0,
  seed_unchanged: true,
  m1_complete: false,
  m2_permitted: false,
  scope: 'Public MASSIVE pt-PT reserved workers and original test, trained model only. Inputs contain id/text; labels are parsed after process inference. This audit does not infer the other M1 gates.',
};
await writeJson('audit.json', report);
console.log(JSON.stringify({schema: report.schema, selected_policy: selected, combined: report.combined, checks}));
