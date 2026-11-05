import test from 'node:test';
import assert from 'node:assert/strict';
import {checkTrace} from './math-actor-trace.mjs';

function row(){return {answer:13,error:null,language_work:20,arithmetic_evaluations:1,trace:[
  {step:0,frame:'axon-question combine 2 with 11 axon-step-0 axon-memory',work:10,prediction:['intent','math-p032',['slots',['a','2'],['b','11']]],computed:13},
  {step:1,frame:'axon-question combine 2 with 11 axon-step-1 axon-memory r0 13',work:10,prediction:['intent','math-done',['slots',['a','13']]],answer:13}
]};}
test('shared trace checker verifies own intermediate and explicit stop in both layouts',()=>{
  for(const focused of [false,true]){const r=row();if(focused)for(const e of r.trace)e.frame=e.frame.replace(/^(axon-question .+) (axon-step-\d+) (axon-memory.*)$/u,'$2 $1 $3');assert.deepEqual(checkTrace(r,'combine 2 with 11',focused),{calculations:1,order:'math-p032',new_order:true});}
});
test('shared trace checker rejects injected future state, false computation and counters',()=>{
  for(const mutate of [r=>r.trace[0].frame+=' r0 13',r=>r.trace[0].computed=99,r=>r.trace[1].frame=r.trace[1].frame.replace('r0 13','r0 99'),r=>r.language_work=0,r=>r.arithmetic_evaluations=0,r=>r.trace[0].prediction[2][1][1]='99']){const r=row();mutate(r);assert.throws(()=>checkTrace(r,'combine 2 with 11'));}
});
