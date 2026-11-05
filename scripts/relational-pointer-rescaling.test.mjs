import test from 'node:test';
import assert from 'node:assert/strict';
import {teaching} from './math-program-source.mjs';
import {episode} from './math-actor-source.mjs';
import {rescaleSource,rescaleProgram} from './relational-pointer-rescaling.mjs';
import {pointerExamples} from './relational-pointer-source.mjs';

test('rescaling recomputes every teacher register rather than doubling an old answer',()=>{
  const source=episode(teaching('4 tickets at $2.5. How much?','<<4*2.5=10>>\n#### 10')),before=JSON.stringify(source),changed=rescaleSource(source);
  assert(changed.examples[0].text.includes('8 tickets at $5.'));assert.equal(changed.final,40);assert.equal(changed.actions[0].value,40);assert.equal(JSON.stringify(source),before);
  assert.deepEqual(pointerExamples(changed).map(r=>r.targets),pointerExamples(source).map(r=>r.targets));
});
test('rescaling preserves repeated references and recomputes composed non-linear results',()=>{
  const source=episode(teaching('3 and 8','<<3+8=11>><<11*11=121>>\n#### 121')),changed=rescaleSource(source);
  assert.equal(changed.final,484);assert.deepEqual(changed.actions.map(r=>r.value),[22,484]);assert.equal(changed.actions[1].a.index,changed.actions[1].b.index);assert(!changed.examples[0].text.includes('22'));assert.throws(()=>rescaleSource(source,3));
});
test('training factor4 uses the same graph with recomputed registers and strict bounds',()=>{
  const source=episode(teaching('3 and 8','<<3+8=11>><<11*11=121>>\n#### 121')),changed=rescaleProgram(source,4);
  assert.equal(changed.final,1936);assert.deepEqual(changed.actions.map(r=>r.value),[44,1936]);
  assert.deepEqual(pointerExamples(changed).map(e=>e.targets),pointerExamples(source).map(e=>e.targets));
  assert.throws(()=>rescaleProgram(source,1));assert.throws(()=>rescaleProgram(source,8));assert.throws(()=>rescaleSource(source,4));
});
