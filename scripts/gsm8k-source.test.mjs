import test from 'node:test';
import assert from 'node:assert/strict';
import {partitionQuestions} from './gsm8k-source.mjs';

const hidden=question=>({question,get answer(){throw Error('Answer property must stay sealed');}});
test('math partition never accesses answers, including reserved and development records',()=>{
  const source=Array.from({length:80},(_,i)=>hidden(`Problem name-${String.fromCharCode(97+Math.floor(i/26),97+i%26)} has 23 coins. How many remain?`));
  const {rows,statistics}=partitionQuestions(source);
  assert.equal(rows.length,80);assert.equal(statistics.answer_properties_accessed,false);
  assert(statistics.counts.train>0&&statistics.counts.development>0&&statistics.counts.reserved>0);
});
test('duplicate numeric templates and README exposure stay grouped without reading labels',()=>{
  const source=[hidden('Mia has 12 coins. How many coins?'),hidden(' mia has 99 COINS.  How many coins? '),
    hidden('A carnival snack booth made $50 selling popcorn each day. What is left?'),hidden('A carnival snack booth made $75 selling popcorn each day. What is left?')];
  const result=partitionQuestions(source,['A carnival snack booth made $50 selling popcorn each day.']);
  assert.equal(result.rows[0].template_sha256,result.rows[1].template_sha256);assert.equal(result.rows[0].split,result.rows[1].split);
  assert.deepEqual(result.rows.slice(2).map(r=>r.split),['research_only','research_only']);assert.equal(result.statistics.counts.research_only,2);
});
test('invalid schemas, controls, bounds and empty exposure prefixes are refused',()=>{
  for(const source of [[],[hidden('')],[hidden('x\u0000y')],[hidden('x\u007fy')],[hidden('x\u0080y')],[hidden('x'.repeat(8193))],[{question:'q',answer:'a',expected:'forbidden'}]])assert.throws(()=>partitionQuestions(source));
  assert.throws(()=>partitionQuestions([hidden('q')],['']));
});
