import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';
import {inspectQuestion} from './math-actor-input.mjs';

const [study,out]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/audit-math-actor-input-ceiling.mjs <frozen-study> <new-output.json>');
const frozen=JSON.parse(await fs.readFile(path.join(study,'freeze.json'))),report=JSON.parse(await fs.readFile(path.join(study,'predictions-ready.json'))),inputBytes=await fs.readFile(path.join(study,'development-inputs.json'));
assert.equal(sha(inputBytes),frozen.inputs_sha256);const inputs=JSON.parse(inputBytes);assert.equal(inputs.length,1481);
const inspections=inputs.map(row=>({id:row.id,...inspectQuestion(row.text)}));
const results=[];
for(const p of report.results){
  if(p.policy.numbers_only)continue;
  const bytes=await fs.readFile(path.join(study,`policy-${p.index}-predictions.json`));assert.equal(sha(bytes),p.predictions_sha256);const rows=JSON.parse(bytes);assert.deepEqual(rows.map(r=>r.id),inputs.map(r=>r.id));
  let observed=0;for(const [i,r]of rows.entries())if(inspections[i].unconditionally_unsupported){assert.equal(r.answer,null,'Input representation ceiling must predict refusal independently of weights');assert.equal(r.arithmetic_evaluations,0);observed++;}
  results.push({index:p.index,forced_refusals_verified:observed});
}
const unavailable=inspections.filter(r=>r.unconditionally_unsupported).length,maximumCorrect=inputs.length-unavailable;
const audit={schema:'axon-math-actor-input-ceiling/v1',inputs_sha256:sha(inputBytes),auditor_sha256:sha(await fs.readFile(process.argv[1])),inspector_sha256:sha(await fs.readFile('scripts/math-actor-input.mjs')),total:inputs.length,forced_refusals:unavailable,maximum_possible_correct_with_current_input_contract:maximumCorrect,maximum_possible_fraction:maximumCorrect/inputs.length,target_fraction:.95,target_mathematically_unattainable_under_current_input_contract:maximumCorrect/inputs.length<.95,reasons:Object.fromEntries(['question-token-quota','initial-frame-token-quota','no-copyable-scalar'].map(reason=>[reason,inspections.filter(r=>r.reasons.includes(reason)).length])),results,rows:inspections,source_answers_read:0,scope:'Deterministic question-only upper bound, independently checked against every full-text frozen policy. No answer, derivation or reserve read. Unsupported predicates overlap; this is an upper bound on attainable full-denominator scalar accuracy, not actual accuracy, reasoning or an M1 gate. Per-frame quotas must remain bounded; any future longer-context design needs an explicit new interface, no silent truncation/cap raise.',m1_complete:false,m2_permitted:false};
await fs.writeFile(out,JSON.stringify(audit,null,2),{flag:'wx'});console.log(JSON.stringify({total:audit.total,forced_refusals:unavailable,maximum_possible_correct:maximumCorrect,reasons:audit.reasons,target_unattainable:audit.target_mathematically_unattainable_under_current_input_contract}));
