import test from 'node:test';
import assert from 'node:assert/strict';
import {annotation, sequenceAnnotation, metrics} from './massive-grammar.mjs';

test('structured annotations preserve order, duplicate roles and multiword values',()=>{
  const row={intent:'send',utt:'send Ada Lovelace from New York at 003',annot_utt:'send [person : Ada Lovelace] from [place : New York] at [time : 003]'};
  assert.deepEqual(annotation(row).target,['intent','send',['slots',['person','ada lovelace'],['place','new york'],['time','003']]]);
  assert.deepEqual(annotation(row).template,['send',['slot','person'],'from',['slot','place'],'at',['slot','time']]);
  assert.deepEqual(annotation({intent:'x',utt:'A to B',annot_utt:'[person : A] to [person : B]'}).target[2],['slots',['person','a'],['person','b']]);
});
test('annotation parser fails closed on malformed boundaries and mismatched raw text',()=>{
  for(const annot_utt of ['[person Ada]', '[person : ]', '[person : [name : Ada]]', 'Ada]', '[person : Ada', '[person : Bob]']) {
    assert.throws(()=>annotation({intent:'x',utt:'Ada',annot_utt}));
  }
});
test('exact score counts abstentions and distinguishes intention from arguments',()=>{
  const labels=[1,2,3].map(id=>({id:String(id),target:['intent','send',['slots',['person','ada']]]}));
  const result=metrics([{id:'1',prediction:labels[0].target},{id:'2',prediction:['intent','send',['slots',['person','bob']]]},{id:'3',prediction:null}],labels);
  assert.equal(result.exact,1);assert.equal(result.intent_correct,2);assert.equal(result.wrong,1);assert.equal(result.abstained,1);
});
test('BIO supervision covers raw text exactly and preserves duplicate ordered roles',()=>{
  assert.deepEqual(sequenceAnnotation({intent:'send',utt:'Send Ada Lovelace to New York.',annot_utt:'Send [person : Ada Lovelace] to [place : New York].'}).tags,['O','B:person','I:person','O','B:place','I:place']);
  assert.deepEqual(sequenceAnnotation({intent:'send',utt:'A to B',annot_utt:'[person : A] to [person : B]'}).tags,['B:person','O','B:person']);
  assert.throws(()=>sequenceAnnotation({intent:'x',utt:'preword',annot_utt:'pre[thing : word]'}));
});
