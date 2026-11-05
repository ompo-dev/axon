import {test} from 'node:test';
import assert from 'node:assert/strict';
import {sourceCategory, checkContext, alignedWords} from './dictionary-context.mjs';

test('dictionary POS schemas are explicit and malformed senses fail closed', () => {
  for (const [code, pos] of Object.entries({n: 'NOUN', v: 'VERB', a: 'ADJ', s: 'ADJ', r: 'ADV'})) {
    assert.equal(sourceCategory(`own-pt-synset-00000001-${code}`), pos);
  }
  for (const value of [null, 1, 'own-pt-synset-1-n', 'own-pt-synset-00000001-x', 'other-00000001-n']) {
    assert.throws(() => sourceCategory(value));
  }
  assert.throws(() => checkContext('own-pt-synset-00000001-n', 'arbitrary'));
});

test('articles, demonstratives and auxiliaries cannot inherit a lexical sense solely by spelling', () => {
  const noun = 'own-pt-synset-00000001-n', verb = 'own-pt-synset-00000001-v';
  for (const pos of ['DET', 'ADP', 'PRON', 'PROPN']) assert.equal(checkContext(noun, pos).status, 'incompatible-pos');
  assert.equal(checkContext(verb, 'AUX').status, 'incompatible-pos');
  assert.equal(checkContext(noun, 'NOUN').status, 'pos-compatible-only');
  assert.equal(checkContext(verb, 'VERB').status, 'pos-compatible-only');
});

test('missing contexts stay unavailable and even compatible contexts do not authorize replacement', () => {
  for (const pos of [null, 'X', 'NOUN', 'DET']) {
    const result = checkContext('own-pt-synset-00000001-n', pos);
    assert.equal(result.replacement_permitted, false);
    if (pos === null || pos === 'X') assert.equal(result.status, 'unavailable');
  }
});

test('word alignment requires the complete same segmentation, not matching offsets or counts alone', () => {
  const words = [{form: 'De'}, {form: 'o'}, {form: 'gato'}];
  assert(alignedWords(words, words));
  assert(!alignedWords(words, [{form: 'Do'}, {form: 'gato'}]));
  assert(!alignedWords(words, [{form: 'De'}, {form: 'a'}, {form: 'gato'}]));
  assert(!alignedWords(words, null));
});
