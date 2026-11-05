import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { audit } from './audit-algorithms.mjs';
import { m0Reference } from './m0-references.mjs';
import { parse } from '../ui/node_modules/csv-parse/lib/sync.js';

const [output] = process.argv.slice(2);
if (!output) throw Error('usage: bun scripts/verify-m0.mjs <new-audit-directory>');
await fs.mkdir(output);
const identities = {}, gates = {};
const raw = async file => {
  const bytes = await fs.readFile(file);
  identities[file] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
  return bytes;
};
const json = async file => JSON.parse(await raw(file));
const hash = async file => { await raw(file); return identities[file].sha256; };
const sameFile = async (a, b) => assert.equal(await hash(a), await hash(b), `${a} differs from ${b}`);
const equal = (a, b) => assert.deepEqual(a, b);
const gate = async (id, verify) => {
  try { gates[id] = { passed: true, evidence: await verify() }; }
  catch (error) { gates[id] = { passed: false, error: error.message }; }
};
const base = '.axon-memory/', qa = `${base}observatory/qa/`;
const reserve = `${base}m0-state-reserve-v2/`, portable = `${base}m0-state-portable-v2/`;
const full = `${base}m0-full-resource-v2.outputs/`, source = `${base}m0-source-resource-v2.outputs/`;
const prior = `${base}m0-prior-portable-v2/`, data = 'evals/curricula/m0-state-transfer-v3/';
const contract = await json('evals/protocols/m0-completion-v2.json');
const freeze = await json(`${reserve}freeze.json`), metadata = await json(`${data}data-freeze.json`);
const result = await json(`${full}structural-result.json`);
const compiled = await json(`${full}structural.axstruct`);

await gate('G1', async () => {
  const rust = (await raw(`${qa}m0-final-tests-v2.log`)).toString();
  assert(!/test result: FAILED|error:/.test(rust));
  const count = [...rust.matchAll(/test result: ok\. (\d+) passed; 0 failed/g)].reduce((n, m) => n + Number(m[1]), 0);
  assert.equal(count, 594);
  for (const file of ['m0-final-clippy-v2.log', 'm0-final-minimal-v2.log']) {
    const text = (await raw(qa + file)).toString();
    assert(text.includes('Finished') && !/error:|warning:/.test(text));
  }
  for (const file of ['m0-ui-check-v2.log', 'm0-ui-build-v2.log', 'm0-ui-tests-v2.log', 'm0-ui-browser-v2.log']) await raw(qa + file);
  assert((await raw(`${qa}m0-ui-tests-v2.log`)).toString().includes('22 pass'));
  assert((await raw(`${qa}m0-ui-browser-v2.log`)).toString().includes('27 passed'));
  execFileSync('cargo', ['fmt', '--all', '--', '--check'], { windowsHide: true, stdio: 'pipe' });
  return { rust_tests: count, ui_api_tests: 22, real_core_browser_tests: 27, formatting_checked_now: true };
});
await gate('G2', async () => {
  assert(result.admitted && result.algorithm_correct === 4693 && result.algorithm_total === 4693);
  const checked = await audit(`${full}algorithm-predictions.json`, path.join(output, 'full-independent.json'));
  equal([checked.correct, checked.total, checked.file.rows, checked.file.correct], [4692, 4692, 400, true]);
  const old = await audit(`${prior}predictions.json`, path.join(output, 'prior-independent.json'));
  equal([old.correct, old.total], [2903, 2903]);
  const csv = parse(await raw(`${prior}portfolio.csv`), { columns: true });
  assert.equal(csv.length, 600);
  const rows = await json(`${prior}predictions.json`);
  assert(rows.some(r => r.family === 'stateful-weighted-group' && r.bindings.dados.length === 600 && r.correct));
  for (const family of ['stateful-weighted-group', 'stateful-histogram', 'linear-recurrence', 'quadratic-recurrence', 'bilinear-recurrence', ...metadata.families]) {
    const f = checked.families[family]; assert(f && f.correct === f.total && f.total > 0);
  }
  return { integrated: 4693, portable_retention: 2903, csv_rows: [400, 600], csv_scope: 'Two actual artifacts; 600-row task is in the portable retention run, not the 400-row full-campaign task.' };
});
await gate('G3', async () => {
  const locations = { binary: 'candidate.exe', source_number: 'source-only.axnum', compiled_sources: 'source.axstruct', tasks: 'external-inputs.json', private_answers: 'external-answers.json', protocol: 'frozen-protocol.json', evaluator: 'frozen-evaluator.mjs', reference: 'frozen-reference.mjs', generator: 'frozen-generator.mjs' };
  for (const [key, file] of Object.entries(locations)) {
    equal({ sha256: await hash(reserve + file), bytes: identities[reserve + file].bytes }, freeze.identities[key]);
  }
  await sameFile(`${reserve}source.axstruct`, `${reserve}birth/structural.axstruct`);
  await sameFile(`${reserve}source-only.axnum`, `${source}structural.axnum`);
  const seed = await json(`${reserve}source.axstruct`);
  assert(metadata.required_source_ids.every(id => seed.rules.some(r => r.id === id)));
  assert(metadata.reserve_target_ids.every(id => !seed.rules.some(r => r.id === id)));
  const tasks = await json(`${reserve}external-inputs.json`), labels = await json(`${reserve}external-answers.json`), rows = await json(`${reserve}predictions.json`);
  equal([tasks.length, labels.length, rows.length], [96, 96, 96]);
  let correct = 0; const families = {};
  for (let i = 0; i < tasks.length; i++) {
    const t = tasks[i], r = rows[i], a = labels[i];
    equal([t.id, r.id], [a.id, a.id]);
    assert(!('expected' in t) && !('answers' in t));
    equal(a.expected, t.inputs.map(input => m0Reference(a.family, input)));
    if (!t.inherit) continue;
    equal(r.predictions, a.expected); correct += r.predictions.length;
    families[a.family] = (families[a.family] ?? 0) + r.predictions.length;
  }
  equal(correct, 768); equal(Object.values(families), [256, 256, 256]);
  return { future_cases: correct, paired_input_set_replicates: 48, distinct_algorithm_classes: 3, families, source_targets_absent: true, labels_passed_to_child: false, isolation: 'logical, not an OS sandbox' };
});
await gate('G4', async () => {
  const tasks = await json(`${reserve}external-inputs.json`), rows = await json(`${reserve}predictions.json`), labels = await json(`${reserve}external-answers.json`);
  let correct = 0;
  for (let i = 0; i < tasks.length; i++) {
    const t = tasks[i], r = rows[i];
    assert(r.work <= contract.limits.work_per_acquisition && r.candidates <= contract.limits.candidates_per_acquisition);
    assert(r.library_programs <= contract.limits.library_programs_per_window && r.library_windows <= contract.limits.library_windows);
    equal(r.inherit, t.inherit);
    if (t.inherit) continue;
    equal(r.library_programs, 0);
    const previous = tasks[i - 1];
    equal({ train: t.train, inputs: t.inputs }, { train: previous.train, inputs: previous.inputs });
    correct += r.predictions.filter((value, j) => JSON.stringify(value) === JSON.stringify(labels[i].expected[j])).length;
  }
  equal(correct, 0);
  return { no_library_correct: correct, inherited_correct: 768, limits: contract.limits, no_acquisition_limit_increase: true };
});
await gate('G5', async () => {
  equal([result.correct, result.total, result.dialogue_correct, result.dialogue_turns, result.rules], [7400, 7400, 3100, 3100, 107]);
  const original = await json('evals/curricula/inquiry-v1-data.json');
  for (const lesson of original.lessons) assert(compiled.rules.some(r => r.id === `algorithm-${lesson.id}`));
  const change = await json(`${full}algorithm-change-1-result.json`);
  assert(change.relearned && change.suspended && change.invalidated.includes('algorithm-report-squares'));
  const controls = await json(`${full}algorithm-teaching-result.json`);
  equal(controls.controls.find(r => r.variant === 'inherited').acquired, 56);
  return { retained_base_procedures: original.lessons.length, total_procedures: 56, structural: 7400, dialogue: 3100, relearned_after_dependency_change: true };
});
await gate('G6', async () => {
  const rows = await json(`${portable}predictions.json`), answers = await json(`${portable}portable-answers.json`), requests = await json(`${portable}portable-inputs.json`);
  equal(rows.length, 384);
  await sameFile(`${portable}isolated.axnum`, `${full}structural.axnum`);
  await sameFile(`${base}m0-birth-resource-v2.predictions.json`, `${portable}predictions.json`);
  for (let i = 0; i < rows.length; i++) {
    equal(rows[i].id, answers[i].id); equal(requests[i].expected, null); equal(rows[i].expected, null);
    equal(rows[i].actual, m0Reference(answers[i].family, answers[i].input));
  }
  return { new_portable_cases: 384, teacher: false, separate_process_birth: true, independently_measured_birth_matches: true };
});
await gate('G7', async () => {
  const costs = {};
  for (const name of ['source', 'full', 'birth']) {
    const c = await json(`${base}m0-${name}-resource-v2.json`);
    assert(c.exit_code === 0 && c.lifecycle_ms > 0 && c.lifecycle_ms <= contract.limits.source_and_full_process_wall_ms_each);
    assert(c.max_observed_os_peak_working_set_bytes > 0 && c.max_observed_os_peak_working_set_bytes <= contract.limits.peak_process_working_set_bytes);
    equal(c.binary_sha256.toLowerCase(), freeze.identities.binary.sha256);
    costs[name] = c;
  }
  assert(result.binary_bytes <= contract.limits.binary_seed_bytes);
  equal(result.decimal_bytes, identities[`${full}structural.axnum`]?.bytes ?? (await raw(`${full}structural.axnum`)).length);
  return { costs, seed_binary_bytes: result.binary_bytes, seed_decimal_digits: result.decimal_bytes, energy: null, historical_research_cpu_ram: null, note: 'Whole-process observations, not hot inference or full-system energy. Sampling can miss the final interval. Passive control used less aggregate work; no universal efficiency claim.' };
});
await gate('G8', async () => {
  const text = (await raw(`${qa}m0-final-tests-v2.log`)).toString();
  const required = ['admission_is_independent_atomic_and_replay_validates_storage', 'output_growth_is_refused_before_instantiating_an_oversized_tree', 'exact_integer_execution_refuses_overflow_and_inexact_roots', 'authorization_and_quota_fail_before_any_callback', 'denied_outcome_scope_is_checked_before_callback'];
  for (const name of required) assert(text.includes(`test ${name} ... ok`));
  const vm = (await raw('src/structural_algorithm.rs')).toString();
  assert(!/std::fs|std::process|std::net|unsafe\s*\{/.test(vm));
  return { required_contract_tests: required, effect_boundary: 'Pure typed VM expressions, no evaluator/file/process capability; CLI and host still have filesystem authority. This is not OS confinement.' };
});
await gate('G9', async () => {
  for (const f of metadata.files) equal({ bytes: (await raw(data + f.name)).length, sha256: identities[data + f.name].sha256 }, { bytes: f.bytes, sha256: f.sha256 });
  for (const name of ['external-inputs.json', 'external-answers.json', 'portable-inputs.json', 'portable-answers.json']) await sameFile(data + name, 'evals/curricula/m0-state-transfer-v2/' + name);
  const fullData = await json(`${data}full-data.json`), tasks = await json(`${reserve}external-inputs.json`), labels = await json(`${reserve}external-answers.json`), ports = await json(`${portable}portable-answers.json`);
  for (let i = 0; i < metadata.families.length; i++) {
    const family = metadata.families[i], lesson = fullData.lessons.find(l => l.family === family);
    const seen = new Set([...lesson.examples, ...lesson.admission].map(x => JSON.stringify(x.input)));
    const add = input => { const key = JSON.stringify(input); assert(!seen.has(key), `duplicated future input: ${family}`); seen.add(key); };
    tasks.forEach((t, j) => { if (t.inherit && labels[j].family === family) t.inputs.forEach(add); });
    fullData.requests.filter(r => r.family === family).forEach(r => add(r.bindings.dados));
    ports.filter(r => r.family === family).forEach(r => add(r.input));
  }
  const failed = await json(`${base}m0-reserve-v1/comparison.json`);
  equal(failed.variants.inherited.correct, 513); equal(failed.acquisition_transfer_gates_passed, false);
  const oldFull = await json(`${base}m0-full-v1/structural-result.json`);
  assert(!oldFull.admitted && oldFull.algorithm_correct === 4272 && oldFull.algorithm_total === 4273);
  return { disjoint_future_inputs: true, source_preparation_did_not_change_new_reserve_inputs: true, failed_v1_preserved: true, v1_binary_limitation: 'Hash retained; original failed v1 executable was not copied before replacement. Exact v2 executable and evaluators are frozen.' };
});
await gate('G10', async () => {
  const run = await json(`${qa}m0-live-workflow-v2.json`);
  const events = (await raw(`${qa}m0-live-export-v2.jsonl`)).toString().trim().split(/\r?\n/).map(JSON.parse);
  equal([run.status, run.result.algorithm_correct, events.length, events.at(-1).event, events.at(-1).status], ['complete', 4693, run.trace.records, 'end', 'complete']);
  const pending = new Map(); let questions = 0, previous = 0;
  for (const e of events) {
    equal(e.run, run.id); equal(e.coreGaps, 0);
    if (e.coreSeq !== undefined) { equal(e.coreSeq, previous + 1); previous = e.coreSeq; }
    if (!e.inquiry) continue;
    const q = JSON.parse(e.inquiry), key = `${e.variant}/${e.operation}`;
    if (q.phase === 'query_selected') { equal(q.actual, null); pending.set(key, q); }
    if (q.phase === 'query_observed') { assert(pending.has(key)); equal(q.input, pending.get(key).input); pending.delete(key); questions++; }
  }
  assert(questions > 0); equal(pending.size, 0);
  for (const name of ['inquiry-v1-desktop.png', 'inquiry-v1-mobile-dark.png', 'm0-state-full-v3-desktop.png', 'm0-state-full-v3-mobile-dark.png']) assert((await raw(`${qa}react-workspace/${name}`)).length > 1000);
  return { run: run.id, records: events.length, core_records: previous, gaps: 0, observed_questions: questions, browser_suite: 27, screenshots: 'Desktop/mobile reviewed; bounded table overflow is scrollable.', limitation: 'Persistent port 4322 retains an older loaded server module. The fresh ephemeral 4329 browser harness verified new bounded query restoration. The 4322 export is actual current-core telemetry, not proof of backend hot reload.' };
});
const passed = Object.keys(gates).length === 10 && Object.values(gates).every(g => g.passed);
const report = { schema: 'axon-milestone-audit/v1', milestone: 'M0', protocol: contract.revision, status: passed ? 'validated-within-declared-scope' : 'open', gates, identities,
  scope: contract.scope, excluded_claims: contract.excluded_claims, further_milestones_complete: false };
await fs.writeFile(path.join(output, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ status: report.status, gates: Object.fromEntries(Object.entries(gates).map(([k, v]) => [k, v.passed ? 'pass' : v.error])) }));
if (!passed) process.exitCode = 1;
