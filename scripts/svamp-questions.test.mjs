import test from 'node:test';
import assert from 'node:assert/strict';
import {sealedQuestions,answerScalar} from './svamp-questions.mjs';

const record={ID:'synthetic-only',Body:'Ada has 3 items.',Question:'How many items?',Answer:3,Equation:'untrusted never executed',Type:'test-fixture'};
test('SVAMP child receives only id/text while answer/equation accesses fail',()=>{
  const sealed=sealedQuestions(Buffer.from(JSON.stringify([record])),1);assert.deepEqual(sealed.inputs,[{id:record.ID,text:record.Body+' '+record.Question}]);assert.deepEqual(sealed.accesses,{answers:0,equations:0});
  assert.throws(()=>sealed.sealed_records[0].Answer,/Sealed/);assert.throws(()=>sealed.sealed_records[0].Equation,/Sealed/);assert.deepEqual(sealed.accesses,{answers:1,equations:1});
  assert.throws(()=>sealedQuestions(Buffer.from(JSON.stringify([record,record])),2));assert.throws(()=>sealedQuestions(Buffer.from(JSON.stringify([{...record,unexpected:3}])),1));
});
test('final scalar reader cannot evaluate target strings or silently accept booleans/null',()=>{
  for(const n of [0,3,-2.25,'3','-2.25'])assert.equal(answerScalar(n),Number(n));
  for(const n of [null,true,false,NaN,Infinity,'1+2','3;doSomething()','3e9','',{},[]])assert.throws(()=>answerScalar(n));
});
