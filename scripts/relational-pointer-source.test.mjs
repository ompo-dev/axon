import test from 'node:test';
import assert from 'node:assert/strict';
import {question,teaching} from './math-program-source.mjs';
import {frame,episode} from './math-actor-source.mjs';
import {pointerExamples} from './relational-pointer-source.mjs';

test('long interface is explicit and legacy text/frame limits never change',()=>{
  const text=`3 ${Array(80).fill('word').join(' ')} 8`;
  assert.throws(()=>question(text));assert.equal(question(text,512).words.length,82);assert.throws(()=>question(text,1000));assert.throws(()=>teaching(text,'<<3+8=11>>\n#### 11'));
  const e=teaching(text,'<<3+8=11>>\n#### 11',512);assert.throws(()=>episode(e));const rows=pointerExamples(episode(e,false,544));assert.equal(rows[0].targets[0].left,2);assert.equal(rows[0].targets[0].right,83);assert.equal(rows[1].targets[0].left,86);
  assert.throws(()=>frame(question(text,512).words,[]));assert(frame(question(text,512).words,[],544).split(' ').length>64);
});
test('source positions distinguish repeated values and generic both/stop without future injection',()=>{
  const raw=episode(teaching('3 bags of 8 and 5 removed','<<3*8=24>><<24-5=19>>\n#### 19')),rows=pointerExamples(raw),before=JSON.stringify(raw);
  assert.equal(rows[0].targets.length,2);assert.equal(rows[1].targets.length,1);assert(rows[1].input.text.endsWith('r0 24'));assert(!rows[0].input.text.includes('24'));assert(!rows[1].input.text.includes('19'));assert.equal(rows[2].targets[0].right,null);assert.equal(JSON.stringify(raw),before);
  const both=pointerExamples(episode(teaching('3','<<3*3=9>>\n#### 9')))[0];assert.equal(both.targets.length,1);assert.equal(both.targets[0].left,both.targets[0].right);
});
