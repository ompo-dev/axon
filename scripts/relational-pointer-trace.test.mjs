import test from 'node:test';
import assert from 'node:assert/strict';
import {checkPointerTrace} from './relational-pointer-trace.mjs';

function row(){const q=['3',...Array(80).fill('word'),'3'],first=`axon-step-0 axon-question ${q.join(' ')} axon-memory`,second=`axon-step-1 axon-question ${q.join(' ')} axon-memory r0 6`;return {text:q.join(' '),result:{answer:6,error:null,language_work:20,arithmetic_evaluations:1,trace:[
  {step:0,frame:first,work:10,prediction:{label:'math-p032',left:2,right:83},left_origin:{kind:'question',index:0},right_origin:{kind:'question',index:81},computed:6},
  {step:1,frame:second,work:10,prediction:{label:'math-done',left:86,right:null},left_origin:{kind:'register',index:0},answer:6}
]}};}
test('new trace verifier checks complete long text and distinct positional provenance',()=>{const r=row();assert.equal(checkPointerTrace(r.result,r.text).calculations,1);});
test('position verifier refuses false equal-value provenance, future state and computation',()=>{
  for(const mutate of [r=>r.trace[0].right_origin.index=0,r=>r.trace[1].left_origin.index=1,r=>r.trace[0].computed=99,r=>r.trace[0].prediction.left=0]){const r=row();mutate(r.result);assert.throws(()=>checkPointerTrace(r.result,r.text));}
});
test('pointer action quota remains an exact refusal, not a computation or generic error',()=>{
  const trace=Array.from({length:9},(_,step)=>({step,frame:`axon-step-${step} axon-question 3 axon-memory${Array.from({length:step},(_,i)=>` r${i} 6`).join('')}`,work:1,prediction:{label:'math-p032',left:2,right:2},left_origin:{kind:'question',index:0},...(step===8?{error:'pointer action quota'}:{right_origin:{kind:'question',index:0},computed:6})}));
  const r={answer:null,error:'pointer action quota',language_work:9,arithmetic_evaluations:8,trace};assert.equal(checkPointerTrace(r,'3').calculations,8);r.trace[8].error='actor action quota';assert.throws(()=>checkPointerTrace(r,'3'));
});
