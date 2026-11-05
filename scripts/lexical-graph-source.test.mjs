import test from 'node:test';import assert from 'node:assert/strict';
import {lexicalEdges,rewireEdges} from './lexical-graph-source.mjs';
const vocab=Array.from({length:32},(_,i)=>`word-${String(i).padStart(2,'0')}`);
const fact=(i,sense)=>({relation:'word',sense:`own-pt-synset-${String(sense).padStart(8,'0')}-n`,object:vocab[i]});
test('all senses contribute soft edges; no new word or definition is admitted',()=>{
  const rows=[fact(0,1),fact(1,1),fact(1,2),fact(2,2),fact(0,1),{...fact(3,1),object:'not-in-training'},{relation:'definition',sense:'ignored',object:'execute untrusted prose'}];
  const result=lexicalEdges(rows,vocab);assert.deepEqual(result.edges,[[vocab[0],vocab[1]],[vocab[1],vocab[2]]]);assert.equal(result.statistics.vertices,3);
  assert.throws(()=>lexicalEdges([{...fact(0,1),sense:'unrecognized'}],vocab),/Sense schema/);
  assert.throws(()=>lexicalEdges(rows,[...vocab].reverse()),/Vocabulary order/);
});
test('rewiring is deterministic, does not mutate input and preserves every vertex degree',()=>{
  const edges=Array.from({length:24},(_,i)=>[vocab[i],vocab[i+1]]),before=structuredClone(edges);
  const result=rewireEdges(edges,19374611);assert.deepEqual(edges,before);assert.deepEqual(result,rewireEdges(edges,19374611));
  const degrees=e=>Object.fromEntries([...new Set(e.flat())].sort().map(v=>[v,e.flat().filter(x=>x===v).length]));
  assert.deepEqual(degrees(edges),degrees(result.edges));assert.equal(new Set(result.edges.map(JSON.stringify)).size,edges.length);assert(result.edges.every(e=>e[0]!==e[1]));assert(result.statistics.accepted_swaps>0);assert(result.statistics.original_edges_retained<edges.length);
  assert.throws(()=>rewireEdges(edges,0));assert.throws(()=>rewireEdges([[vocab[0],vocab[0]],[vocab[1],vocab[2]]],1),/Self edge/);
});
