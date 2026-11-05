import test from 'node:test';
import assert from 'node:assert/strict';
import {partitionContexts} from './pacific-source.mjs';

const context=(uid,cell)=>({table:{uid,table:[['year','value'],['2024',cell]]},get questions(){throw Error('Annotations accessed');},get paragraphs(){throw Error('Prose accessed');}});
test('table split never reads annotations or paragraphs, including reserved records',()=>{
  const rows=Array.from({length:100},(_,i)=>context(`table-${i}`,String(i)));
  const result=partitionContexts(rows);
  assert.equal(result.metadata.length,100);
  assert(result.statistics.reserved.contexts>0);assert(result.statistics.train.contexts>0);assert(result.statistics.development.contexts>0);
  for(const partition of ['train','development'])assert.equal(result.partitions[partition].length,result.statistics[partition].contexts);
  assert(result.metadata.every(r=>Object.keys(r).sort().join(',')==='fingerprint,index,partition,uid'));
});
test('duplicate cells stay together across different UIDs and formatting',()=>{
  const rows=[context('a',' a  b '),context('b','A B'),context('a','a b')];
  const result=partitionContexts(rows);
  assert.equal(new Set(result.metadata.map(r=>r.fingerprint)).size,1);
  assert.equal(new Set(result.metadata.map(r=>r.partition)).size,1);
  assert.equal(Object.values(result.statistics).reduce((n,s)=>n+s.tables,0),1);
});
test('identity conflicts, malformed cells and allocation bounds fail closed',()=>{
  assert.throws(()=>partitionContexts([context('a','1'),context('a','2')]),/Same table UID/);
  assert.throws(()=>partitionContexts([context('a',42)]),/Table cell/);
  assert.throws(()=>partitionContexts([]),/Context quota/);
  assert.throws(()=>partitionContexts(Array(3001).fill(context('a','1'))),/Context quota/);
  const wide=context('a','1');wide.table.table=[Array(129).fill('x')];assert.throws(()=>partitionContexts([wide]),/column quota/);
});
