import test from 'node:test';
import assert from 'node:assert/strict';
import {teaching} from './math-program-source.mjs';
import {episode} from './math-actor-source.mjs';
import {repairExamples} from './working-pointer-repair.mjs';

test('repair uses actual wrong provenance, recomputes a missing expression and stops on its new register',()=>{
  const source=episode(teaching('2 and 3','<<2+3=5>>\n#### 5')),rollout={trace:[{prediction:{label:'math-p034'},left_origin:{kind:'question',index:0},right_origin:{kind:'question',index:2},computed:6}]},before=JSON.stringify(rollout),r=repairExamples(source,rollout);assert.equal(r.refusals.length,0);assert.equal(r.pairs.length,2);const first=r.pairs[0],last=r.pairs[1];assert(first.working.input.text.endsWith('op:math-p034 a:q0 b:q2 6'));assert(!first.working.input.text.includes('5'));assert.deepEqual(first.working.targets[0],{label:'math-p032',left:0,right:2});assert(first.blind.input.text.endsWith('work left right 6'));assert.equal(last.working.targets[0].left,10);assert.equal(last.working.targets[0].label,'math-done');assert.equal(JSON.stringify(rollout),before);
});
test('equal scalar answers do not turn an unrelated computation into the required expression',()=>{
  const source=episode(teaching('2','<<2+2=4>>\n#### 4')),r=repairExamples(source,{trace:[{prediction:{label:'math-p034'},left_origin:{kind:'question',index:0},right_origin:{kind:'question',index:0},computed:4}]});assert.equal(r.pairs[0].working.targets[0].label,'math-p032');assert.equal(r.pairs.at(-1).working.targets[0].left,8);
});
