import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const [run, corpus, output] = process.argv.slice(2);
if (!output) throw Error('usage: node scripts/audit-public-syntax.mjs <run> <corpus> <new-output>');
await fs.mkdir(output);
const json = async name => JSON.parse(await fs.readFile(name));
const sha = b => createHash('sha256').update(b).digest('hex');
const freeze = await json(path.join(run, 'freeze.json')), audit = await json(path.join(run, 'audit.json'));
for (const [file, digest] of Object.entries(freeze.sources)) assert.equal(sha(await fs.readFile(path.join(run, 'source', file))), digest);
assert.equal(sha(await fs.readFile(path.join(run, 'candidate.exe'))), freeze.binary_sha256);
assert.equal(sha(await fs.readFile(path.join(run, 'parent.axnum'))), freeze.parent_sha256);
const decoded = {};
// Independent basic-column decoder; never opens original test and decides reserve by metadata first.
for (const partition of ['train', 'dev']) {
  const bytes = await fs.readFile(path.join(corpus, `pt_bosque-ud-${partition}.conllu`));
  assert.equal(sha(bytes), freeze.corpus_hashes[partition]);
  let document; const rows = [];
  for (const block of bytes.toString('utf8').trim().split(/\r?\n\r?\n/u)) {
    const lines = block.split(/\r?\n/u);
    const newdoc = lines.find(s => s.startsWith('# newdoc_id = ') || s.startsWith('# newdoc id = '));
    if (newdoc) document = newdoc.slice(newdoc.indexOf(' = ') + 3);
    assert(document);
    if (parseInt(sha(document).slice(0, 8), 16) % 5 === 0) continue;
    const id = lines.find(s => s.startsWith('# sent_id = ')).slice(12);
    const text = lines.find(s => s.startsWith('# text = ')).slice(9);
    const words = lines.filter(s => /^[1-9][0-9]*\t/u.test(s)).map(line => {
      const col = line.split('\t'); assert.equal(col.length, 10);
      return {form: col[1], upos: col[3], head: Number(col[6]), deprel: col[7]};
    });
    rows.push({id, text, document, words});
  }
  decoded[partition] = rows;
}
assert.deepEqual(decoded.train.map(r => r.id), freeze.train_ids);
assert.deepEqual(decoded.dev.map(r => r.id), freeze.development_ids);
const documents = new Set(decoded.train.map(r => r.document)); assert(decoded.dev.every(r => !documents.has(r.document)));
const truth = await json(path.join(run, 'development-truth.json')), inputs = await json(path.join(run, 'development-inputs.json'));
for (let i = 0; i < truth.length; i++) {
  const g = decoded.dev[i];
  assert.equal(truth[i].id, g.id); assert.equal(truth[i].document, g.document);
  assert.deepEqual(truth[i].words.map(({form, upos, head, deprel}) => ({form, upos, head, deprel})), g.words);
  assert.deepEqual(inputs[i], {id: g.id, text: g.text});
}
const policies = [];
for (const policy of audit.policies) {
  const name = `policy-${policy.index}`;
  if (policy.status === 'rejected-quota') {
    const error = await fs.readFile(path.join(run, `${name}-training-failed.stderr.log`), 'utf8');
    assert.equal(error, policy.error); assert.match(error, /structural work or size quota|seed budget exhausted/u);
    policies.push({index: policy.index, status: policy.status}); continue;
  }
  const rows = await json(path.join(run, `${name}-predictions.json`)); assert.equal(rows.length, decoded.dev.length);
  const totals = {sentences: rows.length, gold_words: 0, predicted_words: 0, answered: 0, aligned_sentences: 0, aligned_words: 0, pos_correct: 0, unlabeled_correct: 0, labeled_correct: 0, complete_trees: 0, ambiguous_expansions: 0};
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i], gold = decoded.dev[i]; assert.equal(row.id, gold.id);
    const predicted = row.prediction?.words ?? [];
    totals.gold_words += gold.words.length; totals.predicted_words += predicted.length;
    totals.answered += Number(row.prediction != null); totals.ambiguous_expansions += row.prediction?.ambiguous_expansions ?? 0;
    const aligned = predicted.length === gold.words.length && predicted.every((w, j) => w.form === gold.words[j].form);
    if (row.prediction != null) assert.equal(predicted.filter(w => w.head === 0).length, 1);
    for (let j = 0; j < predicted.length; j++) {
      let node = j + 1; const visited = new Set();
      while (node !== 0) {assert(node >= 1 && node <= predicted.length && !visited.has(node)); visited.add(node); node = predicted[node - 1].head;}
    }
    if (!aligned) continue;
    totals.aligned_sentences++; totals.aligned_words += gold.words.length;
    let complete = true;
    for (let j = 0; j < predicted.length; j++) {
      const p = predicted[j], g = gold.words[j];
      totals.pos_correct += Number(p.upos === g.upos); totals.unlabeled_correct += Number(p.head === g.head);
      totals.labeled_correct += Number(p.head === g.head && p.deprel === g.deprel);
      complete &&= p.head === g.head && p.deprel === g.deprel && p.upos === g.upos;
    }
    totals.complete_trees += Number(complete);
  }
  assert.deepEqual(totals, policy.score.totals);
  assert.equal(sha(await fs.readFile(path.join(run, name, 'structural.axnum'))), policy.number_sha256);
  assert.deepEqual(await fs.readFile(path.join(run, name, 'structural.axstruct')), await fs.readFile(path.join(run, `${name}-parent-birth`, 'structural.axstruct')));
  const log = (await fs.readFile(path.join(run, `${name}-training.stdout.jsonl`), 'utf8')).trim().split(/\r?\n/u).map(JSON.parse);
  const training = log.pop(); assert.deepEqual(training, policy.training.summary);
  assert.equal(log.length, training.budget.epochs);
  assert(log.every((e, i) => e.coreSeq === i + 1));
  policies.push({index: policy.index, status: policy.status, totals, page_bytes: training.page_bytes, epochs: log.length, number_sha256: policy.number_sha256});
}
const selected = policies.filter(p => p.status === 'evaluated').sort((a, b) => b.totals.labeled_correct - a.totals.labeled_correct || b.totals.complete_trees - a.totals.complete_trees || a.page_bytes - b.page_bytes || a.index - b.index)[0];
assert.equal(selected.index, audit.selected);
assert.equal(audit.m1_complete, false); assert.equal(audit.m2_permitted, false); assert.equal(audit.production_promoted, false);
const report = {schema: 'axon-public-syntax-recount/v1', policies, selected: selected.index, reserved_evaluated: false, original_test_evaluated: false,
  train_document_disjoint_development: true, inference_gold_fields_absent: true, source_truth_recounted: true, m1_complete: false,
  scope: 'Independent decoder/scorer implementation by the same engineering author; not independent human semantic review or a fresh linguistic reserve.'};
await fs.writeFile(path.join(output, 'audit.json'), JSON.stringify(report, null, 2), {flag: 'wx'});
console.log(JSON.stringify(report));
