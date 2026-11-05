import test from 'node:test';
import assert from 'node:assert/strict';
import {teaching,scalar} from './math-program-source.mjs';
import {decimal,frame,episode,focusExample,canonicalExamples} from './math-actor-source.mjs';

test('actor frames contain only question, step, and past numeric results',()=>{
  const e=teaching('Has 3 bags of 8 and loses 5. How many?', '<<3*8=24>><<24-5=19>>\n#### 19');
  const full=episode(e),free=episode(e,true);
  assert.deepEqual(full.examples.map(x=>x.intent),['math-p034','math-p033','math-done']);
  assert(!full.examples[0].text.includes('24'));assert(!full.examples[0].text.includes('19'));
  assert(full.examples[1].text.endsWith('r0 24'));assert(full.examples[2].text.endsWith('r0 24 r1 19'));
  assert.equal(full.final,19);assert.equal(free.examples[0].text,'axon-question 3 8 5. axon-step-0 axon-memory');
  assert.deepEqual(free.examples[1].tags,['O','O','O','B:b','O','O','O','B:a']);
  assert.equal(free.examples[2].tags.at(-1),'B:a');
});
test('CSE and both operands preserve exact source graph without algebraic rewriting',()=>{
  const e=teaching('3 and 8', '<<3+8=11>><<11*11=121>>\n#### 121');
  const r=episode(e);assert.equal(r.actions.length,2);assert.equal(r.examples[1].tags.filter(x=>x==='B:both').length,1);
  assert.equal(r.examples[1].intent,'math-p034');assert.equal(r.final,121);
});
test('decimal register strings roundtrip without exponent or implicit rounding',()=>{
  for(const n of [0,-0,1,1e-7,-1e-18,1/3,999999999,1e9,Number.MIN_VALUE]){
    const s=decimal(n);assert(!s.includes('e'));assert.equal(scalar(s),n===0?0:n);
  }
  for(const n of [Infinity,NaN,1e10])assert.throws(()=>decimal(n));
});
test('oversized complete episodes refuse instead of dropping or truncating late steps',()=>{
  assert.throws(()=>frame(Array(63).fill('x'),[]));assert.throws(()=>frame(['q'],Array(9).fill(1)));
  const words=['3','8',...Array(57).fill('word')].join(' '),e=teaching(words,'<<3+8=11>><<11*11=121>>\n#### 121');
  assert.throws(()=>episode(e));
});
test('state-focused frames move only the existing state marker and its O label',()=>{
  const e=teaching('3 bags of 8 and 5 removed', '<<3*8=24>><<24-5=19>>\n#### 19'),trace=episode(e);
  for(const [step,original]of trace.examples.entries()){
    const focused=focusExample(original,step);assert.equal(focused.text.split(' ')[0],`axon-step-${step}`);assert.equal(focused.intent,original.intent);assert.equal(focused.tags.length,original.tags.length);
    const selected=r=>r.tags.flatMap((tag,i)=>tag==='O'?[]:[[tag,r.text.split(' ')[i]]]);assert.deepEqual(selected(focused),selected(original));
  }
  assert.throws(()=>focusExample(trace.examples[1],0));
});
test('canonical teacher questions match runtime terminal token normalization at every step',()=>{
  const e=episode(teaching('Has 4 tickets at $2.5. How much?','<<4*2.5=10>>\n#### 10'),true),before=JSON.stringify(e),rows=canonicalExamples(e.examples);
  assert(rows.every(r=>r.text.includes('4 $2.5 axon-step-')));
  assert.deepEqual(rows.map(r=>r.tags),e.examples.map(r=>r.tags));assert.equal(JSON.stringify(e),before);
  assert.deepEqual(canonicalExamples(rows),rows);assert.throws(()=>canonicalExamples([e.examples[1]]));
});
