import test from 'node:test';
import assert from 'node:assert/strict';
import {inspectQuestion} from './math-actor-input.mjs';

test('actor input ceiling distinguishes question and expanded frame limits',()=>{
  const text=n=>['3',...Array(n-1).fill('word')].join(' ');
  assert.equal(inspectQuestion(text(61)).unconditionally_unsupported,false);
  assert.deepEqual(inspectQuestion(text(62)).reasons,['initial-frame-token-quota']);
  assert.deepEqual(inspectQuestion(text(65)).reasons,['question-token-quota','initial-frame-token-quota']);
});
test('actor ceiling never assumes word numerals, percentages or grouped values are available',()=>{
  for(const text of ['three bags','50%','1,000','$1,000','01'])assert(inspectQuestion(text).reasons.includes('no-copyable-scalar'));
  assert.equal(inspectQuestion('Has $2.5.').copyable_scalars,1);assert.equal(inspectQuestion('3 and -5.').copyable_scalars,2);
});
test('actor ceiling includes controls, UTF8 token bounds and empty inputs',()=>{
  for(const text of ['', ' ', '3\n4',`3 ${'x'.repeat(257)}`,`3 ${'á'.repeat(129)}`])assert(inspectQuestion(text).reasons.includes('question-token-quota'));
  assert.equal(inspectQuestion(`3 ${'á'.repeat(128)}`).unconditionally_unsupported,false);
  assert.throws(()=>inspectQuestion(null));
});
test('actor frame byte overhead is accounted without truncating a valid question',()=>{
  const text=['3',...Array(16).fill('x'.repeat(253))].join(' '),r=inspectQuestion(text);
  assert(Buffer.byteLength(text)<=4096);assert(r.initial_frame_bytes>4096);assert.deepEqual(r.reasons,['initial-frame-token-quota']);
});
