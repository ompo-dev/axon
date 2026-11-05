import {test} from 'node:test';
import assert from 'node:assert/strict';
import {trainingFrames, shuffledLabels} from './auxiliary-pos-source.mjs';

test('auxiliary chunks preserve every original syntactic word/POS once with explicit origins', () => {
  const words = ['De', 'o', 'gato', ',', 'correu'].map((form, i) => ({form, upos: ['ADP','DET','NOUN','PUNCT','VERB'][i]}));
  const {examples, origins} = trainingFrames([{id: 's', document: 'd', words}], 2);
  assert.deepEqual(examples.map(e => e.text), ['De o', 'gato ,', 'correu']);
  assert.deepEqual(examples.flatMap(e => e.labels), words.map(w => w.upos));
  assert.deepEqual(origins.map(o => [o.start, o.end]), [[0,2],[2,4],[4,5]]);
  assert(origins.every(o => o.id === 's' && o.document === 'd' && o.source_words === 5));
});
test('overlong, multi-token or control-bearing source words fail rather than disappearing', () => {
  for (const form of ['', 'two words', 'a\nb', 'x'.repeat(257)]) {
    assert.throws(() => trainingFrames([{id: 's', words: [{form, upos: 'NOUN'}]}]));
  }
  for (const limit of [0, 65, 1.5]) assert.throws(() => trainingFrames([], limit));
  assert.throws(() => trainingFrames([{id: 's', words: []}]));
  assert.throws(() => trainingFrames([{id: 's', words: [{form: '2 3', upos: 'NUM'}]}]));
});
test('one source grouped numeric unit retains NUM on each core part with original-index witnesses', () => {
  const {examples, origins} = trainingFrames([{id: 's', document: 'd', words: [{form: '34 470', upos: 'NUM'}]}]);
  assert.deepEqual(examples, [{text: '34 470', labels: ['NUM', 'NUM']}]);
  assert.deepEqual(origins[0].source_word_indices, [0, 0]);
  assert.equal(origins[0].source_words, 1); assert.equal(origins[0].core_words, 2);
});
test('shuffled-label control preserves inputs, label counts and the untouched original deterministically', () => {
  const examples = Array.from({length: 8}, (_, i) => ({text: `word-${i} other-${i}`, labels: ['NOUN', i % 2 ? 'VERB' : 'ADJ']}));
  const original = structuredClone(examples), changed = shuffledLabels(examples, 47);
  assert.deepEqual(changed, shuffledLabels(examples, 47));
  assert.deepEqual(examples, original);
  assert.deepEqual(changed.map(e => e.text), examples.map(e => e.text));
  assert.deepEqual(changed.flatMap(e => e.labels).sort(), examples.flatMap(e => e.labels).sort());
  assert.notDeepEqual(changed, examples);
  assert.throws(() => shuffledLabels(examples, 0));
});
