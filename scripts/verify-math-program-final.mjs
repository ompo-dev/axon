import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const output=process.argv[2];if(!output)throw Error('usage: node scripts/verify-math-program-final.mjs <new-verification.json>');
const qa='.axon-memory/observatory/qa',run='.axon-memory/m1-math-constrained-decoding-v1';
const sha=b=>createHash('sha256').update(b).digest('hex'),read=async f=>JSON.parse(await fs.readFile(f));
const frozen=await read(`${run}/freeze.json`),study=await read(`${run}/predictions-ready.json`),score=await read('.axon-memory/m1-math-constrained-decoding-audit-v1/audit.json');
for(const [file,hash]of Object.entries(frozen.sources))assert.equal(sha(await fs.readFile(`${run}/source/${file}`)),hash);
assert.equal(sha(await fs.readFile(`${run}/lab.exe`)),frozen.binaries.lab.sha256);
for(const file of ['src/structural_math_lab.rs','src/structural_lexical_encoder.rs','src/structural_sequence_search.rs'])assert.equal(sha(await fs.readFile(file)),frozen.sources[file],'Frozen engine source changed');
const full=await fs.readFile(`${qa}/m1-math-program-full-v1.log`,'utf8'),groups=[...full.matchAll(/^test result: ok\. (\d+) passed; 0 failed/gm)].map(m=>Number(m[1]));
assert.equal(groups.reduce((a,b)=>a+b,0),735);assert.equal(groups.length,105);assert(!/^error:|^test result: FAILED/m.test(full));
for(const f of ['m1-math-program-clippy-v3.log','m1-math-program-minimal-v1.log']){const log=await fs.readFile(`${qa}/${f}`,'utf8');assert(log.includes('Finished')&&!/^error:/m.test(log));}
const js=await fs.readFile(`${qa}/m1-math-program-js-v1.log`,'utf8');assert(/pass 37/u.test(js)&&/fail 0/u.test(js));
const source=await read('.axon-memory/m1-math-program-training-source-v1/audit.json');assert.deepEqual(source.answer_accesses,{train:4429,development:0,reserved:0,research_only:0});assert.equal(source.eligible,663);
const numerical=await read('.axon-memory/m1-math-program-library-probe-v1/audit.json');assert.equal(numerical.passed,27);assert.equal(numerical.finite_cases,3456);assert.equal(numerical.domain_cases_not_executed,0);
const transfer=await read('.axon-memory/m1-math-program-transfer-v1/audit.json');assert.equal(transfer.results.length,8);assert(transfer.results.every(r=>!r.admitted));
assert.equal(score.results.length,4);assert.deepEqual(score.results.map(r=>r.correct),[2,11,12,14]);assert.equal(score.baselines.constant.correct,35);assert.equal(score.unscorable_gold,0);assert.equal(score.answer_accesses.reserved+score.answer_accesses.research_only,0);
for(const p of study.results){assert.equal(p.predictions,1481);assert(p.pack.encoded_bytes<=262144);assert.equal(p.peak_active_modules,1);assert.equal(sha(await fs.readFile(`${run}/policy-${p.index}-combined/structural.axnum`)),p.number_sha256);assert.equal(sha(await fs.readFile(`${run}/policy-${p.index}-predictions.json`)),p.predictions_sha256);}
const search=await read(`${qa}/m1-math-program-old-search-recount-v1.json`);assert.equal(search.rows_identical,27072);
const lexical=await read('.axon-memory/m1-math-program-legacy-replay-v1/audit.json');assert.equal(lexical.results.reduce((n,r)=>n+r.identical,0),9024);
const data=await read('.axon-memory/m1-math-program-data-retention-v1/audit.json');assert.equal(data.results[1].correct,896);assert.equal(data.results[1].files_correct,64);assert.equal(data.negative_payloads_refused,8);assert.equal(data.durable_private_mutations_from_negative_payloads,0);
for(const [kind,count]of [['context',170],['state',384]]){const log=JSON.parse(await fs.readFile(`${qa}/m1-math-program-${kind}-retention-v1.log`,'utf8'));assert.equal(log.correct,count);assert.equal(log.total,count);}
assert.equal(sha(await fs.readFile('target/release/axon-uic-curriculum-lab.exe')),'f0cf11fcf95c04935548231cfd6d80d4d6e0ce93eb2afd1f45838c2c95da3660');
assert.equal(sha(await fs.readFile('evals/protocols/m1-completion-v1.json')),'0e99ea97dcc7eb4cae95ba8a1d099edbb8497b3f65e290ba4778f9c43fc47e79');
const archives=[];
for(const file of ['m1-math-program-source-archive-v1.json','m1-math-program-study-archive-v1.json',...[0,1,2,3].map(i=>`m1-math-decoding-policy-${i}-archive-v1.json`),'m1-math-decoding-common-archive-v1.json','m1-math-program-compatibility-archive-v1.json','m1-math-program-retention-archive-v1.json']){
  const receipt=await read(path.join(qa,file));assert(receipt.verified&&receipt.files<=4096&&receipt.source_bytes<=536870912);assert.equal(sha(await fs.readFile(receipt.archive)),receipt.sha256);archives.push(receipt);
}
const result={schema:'axon-math-program-final-verification/v1',verified:true,canonical_document_sha256:sha(await fs.readFile('docs/AXON.md')),lab_binary_sha256:frozen.binaries.lab.sha256,rust_passed:735,rust_groups:104,empty_doc_groups:1,js_passed:37,numeric_program_entries:27,signed_fractional_cases:3456,public_correct:score.results.map(r=>r.correct),public_total_per_policy:1481,constant_correct:35,public_math_passed:false,lexical_rows_unchanged:9024,search_rows_unchanged:27072,retention:{data:896,winners:64,refusals:8,negative_private_mutations:0,context:170,state:384},archives,production_unchanged:true,m1_contract_unchanged:true,m1_complete:false,m2_permitted:false,scope:'Frozen phase45 study integrity and engineering only. Mathematical text generalization failed, below constant; no source annotations during core inference, no reserved/original-test answers, no M1/AGI/fluency/speed/energy approval. Future architectural work excluded.'};
await fs.writeFile(output,JSON.stringify(result,null,2),{flag:'wx'});console.log(JSON.stringify({verified:true,rust:735,js:37,archives:archives.length,public_correct:result.public_correct,m1_complete:false}));
