import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha, parseCorpus, reservedDocument} from './conllu-source.mjs';

export function trainingFrames(sentences, limit = 64) {
  assert(Number.isInteger(limit) && limit >= 1 && limit <= 64);
  const examples = [], origins = [];
  for (const sentence of sentences) {
    assert(sentence.words.length > 0);
    const tokens = sentence.words.flatMap((w, sourceWord) => {
      assert(typeof w.form === 'string' && w.form.length > 0 && Buffer.byteLength(w.form) <= 256
        && !/\p{Cc}/u.test(w.form), 'Auxiliary word must fit bounded core tokens');
      // Bosque has a single annotated NUM "34 470"; retain its category on both parts.
      const groupedNumber = w.upos === 'NUM' && /^[0-9]{1,3}( [0-9]{3})+(?:,[0-9]+)?$/u.test(w.form);
      assert(groupedNumber || !/\s/u.test(w.form), 'Unsupported multi-token auxiliary word');
      return (groupedNumber ? w.form.split(' ') : [w.form]).map(form => ({form, upos: w.upos, sourceWord}));
    });
    for (let start = 0; start < tokens.length; start += limit) {
      const end = Math.min(start + limit, tokens.length), words = tokens.slice(start, end);
      examples.push({text: words.map(w => w.form).join(' '), labels: words.map(w => w.upos)});
      origins.push({id: sentence.id, document: sentence.document, start, end, source_words: sentence.words.length,
        core_words: tokens.length, source_word_indices: words.map(w => w.sourceWord)});
    }
  }
  assert(examples.length <= 10000, 'Auxiliary example quota');
  return {examples, origins};
}
export function shuffledLabels(examples, seed) {
  assert(Number.isSafeInteger(seed) && seed > 0 && seed <= 0xffffffff);
  let state = seed;
  const labels = examples.flatMap(e => e.labels);
  for (let i = labels.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1); [labels[i], labels[j]] = [labels[j], labels[i]];
  }
  let cursor = 0;
  return examples.map(e => {
    const row = {text: e.text, labels: labels.slice(cursor, cursor + e.labels.length)};
    cursor += e.labels.length; return row;
  });
}
export async function loadPublicPos(source) {
  const provenanceBytes = await fs.readFile(path.join(source.corpus, 'provenance.json'));
  assert.equal(sha(provenanceBytes), source.provenance_sha256);
  const provenance = JSON.parse(provenanceBytes);
  assert.equal(provenance.original_test_downloaded, false);
  assert.equal(provenance.reserved_annotations_parsed, false);
  const name = 'pt_bosque-ud-train.conllu', raw = await fs.readFile(path.join(source.corpus, name));
  assert.equal(sha(raw), source.train_sha256);
  assert.equal(sha(raw), provenance.sources[name].sha256);
  const parsed = parseCorpus(raw, 'train', m => !reservedDocument(m.document));
  assert.equal(parsed.accepted.length, provenance.statistics.train.sentences);
  const frames = trainingFrames(parsed.accepted);
  const words = frames.examples.reduce((n, e) => n + e.labels.length, 0);
  const sourceWords = parsed.accepted.reduce((n, s) => n + s.words.length, 0);
  assert.equal(sourceWords, provenance.statistics.train.words);
  assert(words >= sourceWords);
  const stats = {source: provenance.source, attribution: provenance.attribution, license: provenance.license,
    train_sha256: sha(raw), provenance_sha256: sha(provenanceBytes), source_sentences: parsed.accepted.length,
    frames: frames.examples.length, source_words: sourceWords, words, formatted_number_extra_tokens: words - sourceWords,
    reserved_metadata_rows: parsed.excluded.length,
    training_ids: parsed.accepted.map(r => r.id), reserved_annotations_parsed: false,
    development_file_read: false, original_test_evaluated: false,
    scope: 'Gold syntactic words/POS from allowed public train only. Sentences longer than 64 are split without dropping words; new boundaries lose outside context. Annotated grouped NUM forms are split with NUM on each part and explicit original indices. No lemma/head labels, dev annotations or semantic-task labels.'};
  return {...frames, stats};
}
