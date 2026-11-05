import test from 'node:test';
import assert from 'node:assert/strict';
import {target,selected,compare,summarize,canonicalInitialFrame} from './math-actor-diagnostic.mjs';

const t=(op,a,b)=>['intent',op,['slots',['a',String(a)],...(b===undefined?[]:[['b',String(b)]])]];
test('diagnostic preserves strict copy targets, punctuation and scalar roles',()=>{
  assert.deepEqual(target({text:'AXON-question 3 and 5.',intent:'math-p032',tags:['O','B:a','O','B:b']}),t('math-p032',3,5));
  assert.throws(()=>target({text:'3 5',intent:'math-p032',tags:['B:a']}));
  assert.deepEqual(selected(['intent','math-p034',['slots',['both','8']]]),{program:'math-p034',operands:[8,8]});
});
test('swapping is equivalent only for supplied commutative operations',()=>{
  for(const op of ['math-p032','math-p034'])assert.deepEqual(compare(t(op,5,3),t(op,3,5)),{exact:false,operation:true,semantic:true,value:true,refused:false});
  for(const op of ['math-p033','math-p035'])assert.equal(compare(t(op,5,3),t(op,3,5)).semantic,false);
});
test('coincidental numerical equality is never a semantic action certificate',()=>{
  const r=compare(t('math-p034',2,2),t('math-p032',2,2));assert(r.value&&!r.operation&&!r.semantic);
  assert.equal(compare(t('math-done',4),t('math-p032',2,2)).semantic,false);
});
test('invalid actions and undefined arithmetic remain failures',()=>{
  for(const term of [null,t('open-file',3),t('math-done',3,5),['intent','math-p032',['slots',['both','3'],['a','5']]]])assert.equal(selected(term),null);
  assert.equal(compare(t('math-p035',5,0),t('math-p032',3,5)).value,false);
  assert.deepEqual(summarize([compare(null,t('math-done',4)),compare(t('math-done',4),t('math-done',4))]),{total:2,exact:1,operation:1,semantic:1,value:1,refused:1});
});
test('initial state diagnostic isolates terminal punctuation in both layouts',()=>{
  for(const focused of [false,true]){
    const text=focused?'axon-step-0 axon-question 4 2 $2.5. axon-memory':'axon-question 4 2 $2.5. axon-step-0 axon-memory';
    assert.equal(canonicalInitialFrame(text,focused),text.replace('$2.5.','$2.5'));
    assert.equal(canonicalInitialFrame(text.replace('$2.5.','$2.5'),focused),text.replace('$2.5.','$2.5'));
    assert.throws(()=>canonicalInitialFrame(text.replace('step-0','step-1'),focused));
  }
});
