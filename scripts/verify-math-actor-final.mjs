import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';

const output=process.argv[2];if(!output)throw Error('usage: node scripts/verify-math-actor-final.mjs <new-verification.json>');
const qa='.axon-memory/observatory/qa',read=async file=>JSON.parse(await fs.readFile(file));
const first=await read('.axon-memory/m1-math-actor-v1/predictions-ready.json'),state=await read('.axon-memory/m1-math-actor-state-v1/predictions-ready.json'),frozen=await read('.axon-memory/m1-math-actor-state-v1/freeze.json');
for(const [file,hash]of Object.entries(frozen.sources))assert.equal(sha(await fs.readFile(path.join('.axon-memory/m1-math-actor-state-v1/source',file))),hash);
for(const file of ['src/structural_math_actor_lab.rs','src/structural_math_lab.rs','src/structural_lexical_encoder.rs','src/structural_sequence_search.rs','src/demand.rs','src/genome.rs','src/curriculum.rs'])assert.equal(sha(await fs.readFile(file)),frozen.sources[file]);
assert.equal(sha(await fs.readFile('.axon-memory/m1-math-actor-state-v1/lab.exe')),frozen.binaries.lab.sha256);
const source=await read('.axon-memory/m1-math-actor-source-v1/audit.json');assert.equal(source.episodes,629);assert.equal(source.frames_per_policy,2251);assert.equal(source.shapes,253);assert.deepEqual(source.answer_accesses,{train:4429,development:0,reserved:0,research_only:0});
const full=await fs.readFile(`${qa}/m1-math-actor-full-v2.log`,'utf8'),groups=[...full.matchAll(/^test result: ok\. (\d+) passed; 0 failed/gm)].map(m=>Number(m[1]));assert.equal(groups.reduce((a,b)=>a+b,0),739);assert.equal(groups.length,105);assert(!/^error:|^test result: FAILED/m.test(full));
for(const file of ['m1-math-actor-clippy-v2.log','m1-math-actor-minimal-v1.log']){const log=await fs.readFile(path.join(qa,file),'utf8');assert(log.includes('Finished')&&!/^error:/m.test(log));}
const js=await fs.readFile(`${qa}/m1-math-actor-js-v1.log`,'utf8');assert(/pass 45/u.test(js)&&/fail 0/u.test(js)&&/skipped 0/u.test(js));
const scores=[await read('.axon-memory/m1-math-actor-audit-v3/audit.json'),await read('.axon-memory/m1-math-actor-state-audit-v4/audit.json')];assert.deepEqual(scores[0].results.map(r=>r.correct),[24,12]);assert.deepEqual(scores[1].results.map(r=>r.correct),[21,27,21,13]);
for(const score of scores){assert.equal(score.constant.correct,35);assert.equal(score.copy_controls.first_quantity.correct,27);assert.equal(score.copy_controls.last_quantity.correct,29);assert.equal(score.copy_controls.first_quantity.refused,58);assert.equal(score.answer_accesses.reserved+score.answer_accesses.research_only,0);assert(score.results.every(r=>r.total===1481&&!r.passed));}
const genomes=new Set();let presentations=0;
for(const [directory,study]of [['m1-math-actor-v1',first],['m1-math-actor-state-v1',state]]){
  assert.equal(study.development_answers_accessed,false);assert.equal(study.reserved_answers_accessed,false);assert.equal(study.future_teacher_states_used,false);
  for(const p of study.results){const base=path.join('.axon-memory',directory);assert.equal(sha(await fs.readFile(path.join(base,`policy-${p.index}-combined/structural.axnum`))),p.number_sha256);assert.equal(sha(await fs.readFile(path.join(base,`policy-${p.index}-predictions.json`))),p.predictions_sha256);assert.equal(p.predictions,1481);assert.equal(p.pack.modules,35);assert.equal(p.pack.old_modules_preserved,30);assert.equal(p.pack.actor_active_module_quota,2);assert(p.pack.encoded_bytes<=262144&&p.peak_active_modules<=2);assert.equal(p.number_digits,1+3*p.pack.encoded_bytes);assert(!genomes.has(p.number_sha256));genomes.add(p.number_sha256);presentations+=p.training.total*p.training.budget.epochs;}
}
assert.equal(presentations,216096);
const cartesian=await read('.axon-memory/m1-math-actor-cartesian-v1/audit.json');assert.equal(cartesian.correct,138);assert.equal(cartesian.domain_refused,6);assert.equal(cartesian.total,144);assert(cartesian.number_unchanged&&cartesian.teacher_queries===0);
const replay=await read('.axon-memory/m1-math-actor-legacy-replay-v1/audit.json');assert.equal(replay.results.reduce((n,r)=>n+r.rows_identical,0),2962);
const language=await read('.axon-memory/m1-math-actor-language-retention-v1/audit.json');assert.equal(language.results[0].identical,1504);
const data=await read('.axon-memory/m1-math-actor-data-retention-v1/audit.json');assert.equal(data.results[1].correct,896);assert.equal(data.results[1].files_correct,64);assert.equal(data.negative_payloads_refused,8);assert.equal(data.durable_private_mutations_from_negative_payloads,0);
for(const [name,count]of [['context',170],['state',384]]){const r=JSON.parse(await fs.readFile(`${qa}/m1-math-actor-${name}-retention-v1.log`,'utf8'));assert.equal(r.correct,count);assert.equal(r.total,count);}
assert.equal(sha(await fs.readFile('target/release/axon-uic-curriculum-lab.exe')),'f0cf11fcf95c04935548231cfd6d80d4d6e0ce93eb2afd1f45838c2c95da3660');assert.equal(sha(await fs.readFile('evals/protocols/m1-completion-v1.json')),'0e99ea97dcc7eb4cae95ba8a1d099edbb8497b3f65e290ba4778f9c43fc47e79');
const archives=[],indexes=[];
for(const name of ['study-archive-v2','state-archive-v2','replay-archive-v1','cartesian-archive-v1']){
  const r=await read(`${qa}/m1-math-actor-${name}.json`);assert(r.verified&&r.raw_files<=100000&&r.raw_bytes<=536870912);assert.equal(sha(await fs.readFile(r.index)),r.index_sha256);indexes.push({index:r.index,raw_files:r.raw_files,raw_bytes:r.raw_bytes,unique_files:r.unique_files});archives.push(...r.archives);
}
for(const name of ['source-audit','retention','final-audits'])archives.push(await read(`${qa}/m1-math-actor-${name}-archive-v1.json`));
for(const r of archives){assert(r.verified&&r.files<=4096&&r.source_bytes<=536870912);assert.equal(sha(await fs.readFile(r.archive)),r.sha256);}
const verification={schema:'axon-math-actor-final-verification/v1',verified:true,canonical_document_sha256:sha(await fs.readFile('docs/AXON.md')),binary_sha256:frozen.binaries.lab.sha256,rust_passed:739,rust_groups:104,empty_doc_groups:1,js_passed:45,trained_episodes:629,source_program_shapes:253,frames_per_policy:2251,models:6,frame_presentations:presentations,public_correct:scores.flatMap(s=>s.results.map(r=>r.correct)),public_total_per_model:1481,constant_correct:35,copy_last_correct:29,all_public_variants_failed:true,primitive_birth_cases:144,legacy_actor_rows_identical:2962,integrated_language_rows_identical:1504,retention:{data:896,winners:64,negative_refusals:8,negative_private_mutations:0,context:170,state:384},indexes,archives,production_unchanged:true,m1_contract_unchanged:true,m1_complete:false,m2_permitted:false,
  scope:'Frozen phase46 engineering and complete failed-development record, not full M1 approval. Learned bounded own-result composition exists; all variants remain below simple controls. No learned scientific theorem, fluency, physical-resource/energy or broad intelligence claim. Stored byte deduplication affects evidence only; future changes excluded.'};
await fs.writeFile(output,JSON.stringify(verification,null,2),{flag:'wx'});console.log(JSON.stringify({verified:true,rust:739,js:45,models:6,public_correct:verification.public_correct,archives:archives.length,m1_complete:false}));
