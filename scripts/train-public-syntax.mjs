import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha, parseCorpus, reservedDocument} from './conllu-source.mjs';

const [parentFile, executable, corpus, outputPath, protocolPath = 'evals/protocols/m1-public-syntax-v1.json'] = process.argv.slice(2);
if (!outputPath) throw Error('usage: node scripts/train-public-syntax.mjs <parent> <candidate> <bosque-corpus> <new-output> [protocol]');
const output = path.resolve(outputPath); await fs.mkdir(output);
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), {flag: 'wx'});
const protocolBytes = await fs.readFile(protocolPath), protocol = JSON.parse(protocolBytes);
const provenanceBytes = await fs.readFile(path.join(corpus, 'provenance.json')), provenance = JSON.parse(provenanceBytes);
assert.equal(protocol.revision, provenance.revision); assert.equal(provenance.original_test_downloaded, false);
const splits = {}, hashes = {};
for (const partition of ['train', 'dev']) {
  const bytes = await fs.readFile(path.join(corpus, `pt_bosque-ud-${partition}.conllu`));
  hashes[partition] = sha(bytes); assert.equal(hashes[partition], protocol[`${partition}_sha256`]);
  splits[partition] = parseCorpus(bytes, partition, m => !reservedDocument(m.document)).accepted;
}
assert.equal(splits.train.length, 5568); assert.equal(splits.dev.length, 903);
const documents = new Set(splits.train.map(r => r.document));
assert(splits.dev.every(r => !documents.has(r.document)));
const binary = path.join(output, 'candidate.exe'), parent = path.join(output, 'parent.axnum');
await fs.copyFile(executable, binary, fs.constants.COPYFILE_EXCL); await fs.copyFile(parentFile, parent, fs.constants.COPYFILE_EXCL);
const sources = {};
for (const file of ['Cargo.toml', 'Cargo.lock', 'src/lib.rs', 'src/structural.rs', 'src/demand.rs', 'src/genome.rs', 'src/structural_sequence.rs', 'src/structural_sequence_factorized.rs', 'src/structural_dependency.rs', 'src/structural_dependency_lab.rs', 'src/structural_lab.rs', 'src/curriculum.rs', 'scripts/conllu-source.mjs', 'scripts/conllu-source.test.mjs', 'scripts/prepare-bosque-source.mjs', 'scripts/train-public-syntax.mjs', 'scripts/audit-public-syntax.mjs', 'evals/protocols/m1-completion-v1.json', protocolPath]) {
  const bytes = await fs.readFile(file); sources[file] = sha(bytes);
  const dest = path.join(output, 'source', file); await fs.mkdir(path.dirname(dest), {recursive: true}); await fs.writeFile(dest, bytes, {flag: 'wx'});
}
await write('freeze.json', {created_at: new Date().toISOString(), sources, binary_sha256: sha(await fs.readFile(binary)), parent_sha256: sha(await fs.readFile(parent)),
  protocol_sha256: sha(protocolBytes), provenance_sha256: sha(provenanceBytes), corpus_hashes: hashes,
  train_ids: splits.train.map(r => r.id), development_ids: splits.dev.map(r => r.id), reserved_evaluated: false, original_test_evaluated: false});
await write('development-inputs.json', splits.dev.map(({id, text}) => ({id, text})));
await write('development-truth.json', splits.dev.map(({id, text, document, projective, words}) => ({id, text, document, projective, words})));
function score(rows) {
  assert.equal(rows.length, splits.dev.length);
  const totals = {sentences: 0, gold_words: 0, predicted_words: 0, answered: 0, aligned_sentences: 0, aligned_words: 0, pos_correct: 0, unlabeled_correct: 0, labeled_correct: 0, complete_trees: 0, ambiguous_expansions: 0};
  const strata = {projective: {...totals}, nonprojective: {...totals}}, families = {};
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index], gold = splits.dev[index]; assert.equal(row.id, gold.id);
    const predicted = row.prediction?.words ?? [];
    const aligned = JSON.stringify(predicted.map(w => w.form)) === JSON.stringify(gold.words.map(w => w.form));
    const stat = {sentences: 1, gold_words: gold.words.length, predicted_words: predicted.length, answered: Number(row.prediction != null),
      aligned_sentences: Number(aligned), aligned_words: aligned ? gold.words.length : 0, pos_correct: 0, unlabeled_correct: 0, labeled_correct: 0, complete_trees: 0, ambiguous_expansions: row.prediction?.ambiguous_expansions ?? 0};
    for (let i = 0; i < gold.words.length; i++) {
      const g = gold.words[i], p = predicted[i];
      const pos = aligned && p.upos === g.upos, head = aligned && p.head === g.head, label = head && p.deprel === g.deprel;
      stat.pos_correct += Number(pos); stat.unlabeled_correct += Number(head); stat.labeled_correct += Number(label);
      const f = families[g.deprel] ??= {total: 0, correct: 0}; f.total++; f.correct += Number(label);
    }
    stat.complete_trees = Number(stat.labeled_correct === gold.words.length && stat.pos_correct === gold.words.length);
    for (const target of [totals, gold.projective ? strata.projective : strata.nonprojective]) for (const key of Object.keys(stat)) target[key] += stat[key];
  }
  return {totals, strata, families, strict_pos: totals.pos_correct / totals.gold_words, strict_uas: totals.unlabeled_correct / totals.gold_words,
    strict_las: totals.labeled_correct / totals.gold_words, conditional_aligned_las: totals.aligned_words ? totals.labeled_correct / totals.aligned_words : null};
}
const exact = new Map(), words = new Map(), patterns = new Map();
const key = text => text.toLocaleLowerCase('pt').replace(/\s+/gu, ' ').trim();
for (const row of splits.train) {
  const k = key(row.text), variants = exact.get(k) ?? new Map(), encoded = JSON.stringify(row.words);
  variants.set(encoded, (variants.get(encoded) ?? 0) + 1); exact.set(k, variants);
  for (let i = 0; i < row.words.length; i++) {
    const w = row.words[i], form = w.form.toLocaleLowerCase('pt'), counts = words.get(form) ?? new Map();
    const value = JSON.stringify([w.upos, w.head === 0 ? null : w.head - i - 1, w.deprel]);
    counts.set(value, (counts.get(value) ?? 0) + 1); words.set(form, counts); patterns.set(value, (patterns.get(value) ?? 0) + 1);
  }
}
const common = counts => [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
const baselines = {untrained: score(splits.dev.map(r => ({id: r.id, prediction: null}))),
  exact_text_retrieval: score(splits.dev.map(r => ({id: r.id, prediction: exact.get(key(r.text))?.size === 1 ? {words: JSON.parse([...exact.get(key(r.text)).keys()][0])} : null}))),
  gold_words_lexical_distance_oracle: score(splits.dev.map(r => ({id: r.id, prediction: {words: r.words.map((g, i) => {
    const [upos, distance, deprel] = JSON.parse(common(words.get(g.form.toLocaleLowerCase('pt')) ?? patterns));
    const head = distance === null ? 0 : Math.max(0, Math.min(r.words.length, i + 1 + distance)); return {form: g.form, upos, head, deprel};
  })}}))),
  scope: 'Gold-words lexical-distance control has extra oracle segmentation; may emit invalid trees and is not a deployable parser. No pretrained model. Exact retrieval uses only allowed train text.'};
await write('baselines.json', baselines);
const examples = splits.train.map(r => ({text: r.text, surfaces: r.surfaces.map(s => ({form: s.form, words: s.words.map(id => r.words[id - 1].form)})), words: r.words.map(({form, upos, head, deprel}) => ({form, upos, head, deprel}))}));
async function core(name, args) {
  const started = performance.now(); let child;
  try {child = await promisify(execFile)(binary, args, {cwd: output, windowsHide: true, timeout: 1800000, maxBuffer: 64000000, env: {...process.env, AXON_LAB_TELEMETRY: '1'}});}
  catch (error) {await fs.writeFile(path.join(output, `${name}-failed.stdout.log`), error.stdout ?? '', {flag: 'wx'}); await fs.writeFile(path.join(output, `${name}-failed.stderr.log`), error.stderr ?? '', {flag: 'wx'}); throw error;}
  await fs.writeFile(path.join(output, `${name}.stdout.jsonl`), child.stdout, {flag: 'wx'}); await fs.writeFile(path.join(output, `${name}.stderr.log`), child.stderr, {flag: 'wx'});
  const events = child.stdout.trim().split(/\r?\n/u).map(JSON.parse), summary = events.pop(); events.forEach((e, i) => assert.equal(e.coreSeq, i + 1));
  return {summary, events: events.length, whole_process_ms: performance.now() - started};
}
const policies = [];
for (let i = 0; i < protocol.policies.length; i++) {
  const name = `policy-${i}`; await write(`${name}-train.json`, {budget: protocol.policies[i], examples});
  let trained;
  try {trained = await core(`${name}-training`, ['dependency-train', parent, path.join(output, `${name}-parent-birth`), path.join(output, `${name}-train.json`), path.join(output, name)]);}
  catch (error) {
    if (!/structural work or size quota|seed budget exhausted/u.test(error.stderr ?? '')) throw error;
    const rejected = {index: i, status: 'rejected-quota', error: error.stderr};
    policies.push(rejected); await write(`${name}-score.json`, rejected); console.log(JSON.stringify(rejected)); continue;
  }
  assert(trained.summary.parent_unchanged);
  assert.deepEqual(await fs.readFile(path.join(output, `${name}-parent-birth`, 'structural.axstruct')), await fs.readFile(path.join(output, name, 'structural.axstruct')));
  const number = path.join(output, name, 'structural.axnum');
  const evaluation = await core(`${name}-development`, ['dependency-batch', number, path.join(output, `${name}-dev-birth`), path.join(output, 'development-inputs.json'), path.join(output, `${name}-predictions.json`)]);
  assert(evaluation.summary.seed_unchanged && evaluation.summary.teacher_queries === 0);
  const metrics = score(JSON.parse(await fs.readFile(path.join(output, `${name}-predictions.json`))));
  const report = {index: i, status: 'evaluated', number_sha256: sha(await fs.readFile(number)), training: trained, development: evaluation, score: metrics};
  policies.push(report); await write(`${name}-score.json`, report);
  console.log(JSON.stringify({policy: i, training: trained.summary, score: metrics.totals}));
}
const selected = policies.filter(p => p.status === 'evaluated').sort((a, b) => b.score.totals.labeled_correct - a.score.totals.labeled_correct || b.score.totals.complete_trees - a.score.totals.complete_trees || a.training.summary.page_bytes - b.training.summary.page_bytes || a.index - b.index)[0];
assert(selected, 'No candidate passed physical admission');
await write('audit.json', {schema: 'axon-public-syntax/v1', train: splits.train.length, development: splits.dev.length, policies, baselines, selected: selected.index,
  novelty: {unseen_text: splits.dev.filter(r => !exact.has(key(r.text))).length, unseen_word_types: new Set(splits.dev.flatMap(r => r.words.map(w => w.form.toLocaleLowerCase('pt'))).filter(w => !words.has(w))).size},
  m1_complete: false, m2_permitted: false, production_promoted: false, reserved_evaluated: false, original_test_evaluated: false, rss_bytes: null, energy_joules: null, scope: protocol.scope});
