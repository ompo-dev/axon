import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha, parseCorpus, reservedDocument} from './conllu-source.mjs';
import {canonical} from './massive-grammar.mjs';
import {sourceCategory, checkContext, alignedWords} from './dictionary-context.mjs';

const [coveragePath, corpusPath, syntaxPath, output] = process.argv.slice(2);
if (!output) throw Error('usage: node scripts/probe-dictionary-context.mjs <frozen-dictionary-coverage> <bosque-corpus> <frozen-syntax-run> <new-output>');
await fs.mkdir(output);
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), {flag: 'wx'});
const sourceFiles = ['scripts/probe-dictionary-context.mjs', 'scripts/dictionary-context.mjs', 'scripts/dictionary-context.test.mjs', 'scripts/conllu-source.mjs', 'scripts/massive-grammar.mjs'];
const sources = {};
for (const file of sourceFiles) {
  const bytes = await fs.readFile(file), dest = path.join(output, 'source', file);
  sources[file] = sha(bytes); await fs.mkdir(path.dirname(dest), {recursive: true});
  await fs.writeFile(dest, bytes, {flag: 'wx'});
}
const coverageBytes = await fs.readFile(path.join(coveragePath, 'audit.json'));
const coverage = JSON.parse(coverageBytes);
assert.equal(coverage.augmentation_permitted, false);
assert.equal(coverage.development_text_evaluated, false);
assert.equal(coverage.reserved_evaluated, false);
const candidateBytes = await fs.readFile(path.join(coveragePath, coverage.links_file));
const candidates = JSON.parse(candidateBytes), links = new Map();
for (const c of candidates) {
  assert(c.word === canonical(c.word) && !links.has(c.word), 'Duplicate/noncanonical dictionary candidate');
  sourceCategory(c.sense); assert(c.aliases.length > 0);
  links.set(c.word, c);
}
const frozenBytes = await fs.readFile(path.join(syntaxPath, 'freeze.json'));
const frozen = JSON.parse(frozenBytes);
assert.equal(frozen.reserved_evaluated, false); assert.equal(frozen.original_test_evaluated, false);
const syntaxAudit = JSON.parse(await fs.readFile(path.join(syntaxPath, 'audit.json')));
const selected = syntaxAudit.policies.find(p => p.index === syntaxAudit.selected);
assert.equal(selected.status, 'evaluated');
const predictionFile = `policy-${selected.index}-predictions.json`;
const predictionBytes = await fs.readFile(path.join(syntaxPath, predictionFile));
const predictions = JSON.parse(predictionBytes), splits = {}, corpusHashes = {};
const provenanceBytes = await fs.readFile(path.join(corpusPath, 'provenance.json'));
const provenance = JSON.parse(provenanceBytes);
assert.equal(sha(provenanceBytes), frozen.provenance_sha256);
assert.equal(provenance.original_test_downloaded, false);
for (const split of ['train', 'dev']) {
  const name = `pt_bosque-ud-${split}.conllu`, bytes = await fs.readFile(path.join(corpusPath, name));
  corpusHashes[split] = sha(bytes);
  assert.equal(corpusHashes[split], frozen.corpus_hashes[split]);
  assert.equal(corpusHashes[split], provenance.sources[name].sha256);
  // The existing parser excludes reserved documents before text/annotation parsing.
  splits[split] = parseCorpus(bytes, split, m => !reservedDocument(m.document)).accepted;
}
assert.deepEqual(splits.train.map(r => r.id), frozen.train_ids);
assert.deepEqual(splits.dev.map(r => r.id), frozen.development_ids);
const documents = new Set(splits.train.map(r => r.document));
assert(splits.dev.every(r => !documents.has(r.document)));
assert.equal(predictions.length, splits.dev.length);
const totals = {occurrences: 0, gold_pos_compatible: 0, gold_pos_incompatible: 0, gold_pos_unavailable: 0,
  predicted_pos_compatible: 0, predicted_pos_incompatible: 0, predicted_pos_unavailable: 0,
  compatible_with_incompatible_gold: 0, incompatible_with_compatible_gold: 0, unaligned_occurrences: 0};
const occurrences = [], trainingCounts = new Map();
for (const sentence of splits.train) for (const w of sentence.words) {
  const word = canonical(w.form);
  if (!links.has(word)) continue;
  const counts = trainingCounts.get(word) ?? {};
  counts[w.upos] = (counts[w.upos] ?? 0) + 1; trainingCounts.set(word, counts);
}
let posCorrect = 0, goldWords = 0;
for (let i = 0; i < splits.dev.length; i++) {
  const sentence = splits.dev[i], row = predictions[i]; assert.equal(row.id, sentence.id);
  const predicted = row.prediction?.words, aligned = alignedWords(sentence.words, predicted);
  goldWords += sentence.words.length;
  for (let j = 0; j < sentence.words.length; j++) {
    const w = sentence.words[j];
    if (aligned && predicted[j].upos === w.upos) posCorrect++;
    const c = links.get(canonical(w.form)); if (!c) continue;
    const gold = checkContext(c.sense, w.upos), actual = checkContext(c.sense, aligned ? predicted[j].upos : null);
    totals.occurrences++;
    totals[`gold_pos_${gold.status === 'pos-compatible-only' ? 'compatible' : gold.status === 'incompatible-pos' ? 'incompatible' : 'unavailable'}`]++;
    totals[`predicted_pos_${actual.status === 'pos-compatible-only' ? 'compatible' : actual.status === 'incompatible-pos' ? 'incompatible' : 'unavailable'}`]++;
    if (!aligned) totals.unaligned_occurrences++;
    if (actual.status === 'pos-compatible-only' && gold.status === 'incompatible-pos') totals.compatible_with_incompatible_gold++;
    if (actual.status === 'incompatible-pos' && gold.status === 'pos-compatible-only') totals.incompatible_with_compatible_gold++;
    occurrences.push({id: sentence.id, document: sentence.document, word_index: j, form: w.form,
      sense: c.sense, aliases: c.aliases, gold, predicted: actual, aligned});
  }
}
assert.equal(posCorrect, selected.score.totals.pos_correct);
assert.equal(goldWords, selected.score.totals.gold_words);
await write('freeze.json', {sources, coverage_sha256: sha(coverageBytes), candidates_sha256: sha(candidateBytes),
  syntax_freeze_sha256: sha(frozenBytes), predictions_sha256: sha(predictionBytes),
  syntax_number_sha256: selected.number_sha256, corpus_hashes: corpusHashes,
  scope: 'Retrospective audit of already consumed syntax predictions and gold POS; not new inference, training, sense annotations or held-out semantic evidence.'});
await write('occurrences.json', occurrences);
await write('training-pos-counts.json', [...links.values()].map(c => ({word: c.word, sense: c.sense,
  expected_pos: sourceCategory(c.sense), training_pos_counts: trainingCounts.get(c.word) ?? {}})));
const audit = {schema: 'axon-dictionary-context-diagnostic/v1', totals, syntax_pos_correct: posCorrect, syntax_gold_words: goldWords,
  dictionary_candidates: links.size, training_words_observed: trainingCounts.size, development_sentences: splits.dev.length,
  replacement_permitted: false, sense_disambiguation_delivered: false, training_performed: false,
  new_core_predictions: 0, reserved_evaluated: false, original_test_evaluated: false, m1_complete: false,
  attribution: [provenance.attribution, coverage.authors], licenses: [provenance.license, coverage.license],
  references: ['https://wordnet.princeton.edu/documentation/wndb5wn', 'https://universaldependencies.org/u/pos/'],
  scope: 'POS is a necessary condition, not semantic acceptance. Proper nouns and auxiliaries are conservatively incompatible with common-noun/lexical-verb senses. No automatic synonyms, stemming, inflection, sense or register guarantees. Missing segmentation/POS remains unavailable.'};
await write('audit.json', audit); console.log(JSON.stringify(audit));
