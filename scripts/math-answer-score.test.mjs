import test from 'node:test';
import assert from 'node:assert/strict';
import {finalAnswer} from './math-answer-score.mjs';
test('final scalar preserves every leading digit, sign and fractional part',()=>{
  for(const n of [0,1,9,10,12,123,-1,-123,.5,-.25,12345.67])assert.equal(finalAnswer(`Reasoning with 999\n#### ${n}`),n);
  assert.equal(finalAnswer('x\n#### 1,200.50'),1200.5);assert.equal(finalAnswer('x\n#### +12\n'),12);
  assert.equal(finalAnswer('<<3+2=5>>\n#### 5'),5);
});
test('scoring accepts no equation, executable code, malformed group or missing final marker',()=>{
  for(const s of ['#### 1','x\n#### ','x\n#### NaN','x\n#### 1,20','x\n#### 12 dollars','x\n#### 1+2','x\n#### 1e309','x\n#### -Infinity','x\n#### 2\nthen something'])assert.equal(finalAnswer(s),null);
});
