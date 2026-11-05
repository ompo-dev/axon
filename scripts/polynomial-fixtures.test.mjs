import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {fixtures,exact} from './probe-polynomial.mjs';
test('native teaching matches independent BigInt reference and future inputs are disjoint',async()=>{
  const p=JSON.parse(await fs.readFile('evals/protocols/polynomial-proposal-v1.json')),tasks=fixtures(p);
  assert.equal(tasks.length,24);
  for(const t of tasks) {
    const exposed=new Set([...t.train,...t.admission].map(e=>JSON.stringify(e.input)));
    for(const e of [...t.train,...t.admission])assert.equal(e.output,exact(t.terms,e.input));
    assert.equal(t.inputs.length,16);assert(t.inputs.every(v=>!exposed.has(JSON.stringify(v))));
    assert(t.inputs.every(v=>Number.isSafeInteger(exact(t.terms,v))));
  }
});
