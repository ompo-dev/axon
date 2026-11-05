import assert from 'node:assert/strict';

// WordNet ss_type and UD UPOS are different schemas, not sense annotations.
// This conservative mapping checks only a necessary condition for substitution.
const categories = {n: 'NOUN', v: 'VERB', a: 'ADJ', s: 'ADJ', r: 'ADV'};
const upos = new Set('ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X'.split(' '));
export function sourceCategory(sense) {
  assert.equal(typeof sense, 'string');
  const parsed = /^own-pt-synset-[0-9]{8}-([nvasr])$/u.exec(sense);
  assert(parsed, 'Unsupported dictionary sense schema');
  return categories[parsed[1]];
}
export function checkContext(sense, predictedPos) {
  const expected = sourceCategory(sense);
  assert(predictedPos === null || upos.has(predictedPos), 'Invalid contextual POS');
  return {
    status: predictedPos === null || predictedPos === 'X' ? 'unavailable'
      : predictedPos === expected ? 'pos-compatible-only' : 'incompatible-pos',
    expected_pos: expected,
    observed_pos: predictedPos,
    replacement_permitted: false,
    limitation: 'POS compatibility does not establish sense, inflection, register or task preservation.',
  };
}
export function alignedWords(gold, predicted) {
  return Array.isArray(predicted) && predicted.length === gold.length
    && gold.every((word, i) => word.form === predicted[i].form);
}
