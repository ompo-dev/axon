import test from 'node:test';
import assert from 'node:assert/strict';
import {teaching} from './math-program-source.mjs';
import {episode} from './math-actor-source.mjs';
import {workingExamples} from './working-pointer-source.mjs';
import {checkWorkingTrace} from './working-pointer-trace.mjs';

test('working frames retain only already executed operators/sources without future targets',()=>{
  const source=episode(teaching('3 and 8 and 2','<<3*8=24>><<24+2=26>>\n#### 26')),before=JSON.stringify(source),rows=workingExamples(source);
  assert.equal(rows[0].input.text,'3 and 8 and 2');assert(!rows[0].input.text.includes('math-'));assert(rows[1].input.text.endsWith('op:math-p034 a:q0 b:q2 24'));assert(!rows[1].input.text.includes('math-p032'));assert(!rows[1].input.text.includes('26'));assert.equal(rows[1].targets[0].left,8);assert.equal(rows[1].targets[0].right,4);assert.equal(rows[2].targets[0].left,12);
  const blind=workingExamples(source,true);assert(blind[1].input.text.endsWith('work left right 24'));assert.deepEqual(blind.map(r=>r.targets),rows.map(r=>r.targets));assert.deepEqual(blind.map(r=>r.input.candidates),rows.map(r=>r.input.candidates));assert.equal(JSON.stringify(source),before);
});
test('working trace verifies actual provenance and refuses forged operations/sources/values',()=>{
  const text='2 and 2',row={answer:4,error:null,arithmetic_evaluations:1,language_work:20,trace:[{step:0,frame:text,work:10,prediction:{label:'math-p032',left:0,right:2},left_origin:{kind:'question',index:0},right_origin:{kind:'question',index:2},computed:4},{step:1,frame:'2 and 2 op:math-p032 a:q0 b:q2 4',work:10,prediction:{label:'math-done',left:6,right:null},left_origin:{kind:'register',index:0},answer:4}]};
  assert.equal(checkWorkingTrace(row,text).calculations,1);
  for(const mutate of [r=>r.trace[1].frame='2 and 2 op:math-p034 a:q0 b:q2 4',r=>r.trace[1].frame='2 and 2 op:math-p032 a:r0 b:q2 4',r=>r.trace[1].frame='2 and 2 op:math-p032 a:q0 b:q2 9',r=>r.trace[0].right_origin.index=0]){const copy=structuredClone(row);mutate(copy);assert.throws(()=>checkWorkingTrace(copy,text));}
  const blinded=structuredClone(row);blinded.trace[1].frame='2 and 2 work left right 4';assert.equal(checkWorkingTrace(blinded,text,true).calculations,1);assert.throws(()=>checkWorkingTrace(blinded,text));
});
test('literal marker words in questions never truncate working context',()=>{
  const source=episode(teaching('3 axon-memory 8','<<3+8=11>>\n#### 11')),rows=workingExamples(source);
  assert.equal(rows[0].input.text,'3 axon-memory 8');assert(rows[1].input.text.startsWith('3 axon-memory 8 op:math-p032'));assert.deepEqual(rows[0].input.candidates,[0,2]);
});
