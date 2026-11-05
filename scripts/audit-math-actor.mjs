import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';
import {finalAnswer} from './math-answer-score.mjs';
import {close,decimal,checkTrace} from './math-actor-trace.mjs';

const [run,output]=process.argv.slice(2);if(!output)throw Error('usage: node scripts/audit-math-actor.mjs <frozen-study> <new-audit>');
const read=async f=>JSON.parse(await fs.readFile(path.join(run,f)));
const frozen=await read('freeze.json'),study=await read('predictions-ready.json');assert.equal(study.schema,'axon-math-actor-study/v1');
for(const [file,hash]of Object.entries(frozen.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',file))),hash);
for(const [name,info]of Object.entries(frozen.binaries))assert.equal(sha(await fs.readFile(path.join(run,`${name}.exe`))),info.sha256);
const protocol=JSON.parse(await fs.readFile(path.join(run,'source/evals/protocols/m1-math-actor-v1.json'))),inputs=await read('development-inputs.json');
assert.equal(sha(await fs.readFile(path.join(run,'development-inputs.json'))),frozen.inputs_sha256);assert.equal(inputs.length,1481);
assert.equal(sha(await fs.readFile(path.join(run,'episodes.json'))),frozen.episodes_sha256);
const episodes=await read('episodes.json'),orders=new Set(episodes.map(e=>e.full.actions.map(a=>a.program).join(','))),operations=new Map(protocol.primitive_library.map(p=>[p.id,p.operator]));
const checked=[];
for(const result of study.results){
  const bytes=await fs.readFile(path.join(run,`policy-${result.index}-predictions.json`));assert.equal(sha(bytes),result.predictions_sha256);
  const rows=JSON.parse(bytes),policyInputs=await read(`policy-${result.index}-inputs.json`);assert.equal(rows.length,1481);assert.deepEqual(rows.map(r=>r.id),inputs.map(r=>r.id));
  assert.equal(sha(await fs.readFile(path.join(run,`policy-${result.index}-combined/structural.axnum`))),result.number_sha256);
  checked.push({result,rows,traces:rows.map((r,i)=>checkTrace(r,policyInputs[i].text,result.policy.layout==='focused',{operations,orders}))});
}
for(const [index,call]of study.calls.entries()){assert.equal(index,call.order);assert.deepEqual(call,await read(`call-${index}.json`));}
await fs.mkdir(output);await fs.writeFile(path.join(output,'freeze-before-gold.json'),JSON.stringify({predictions:study.results.map(p=>({index:p.index,sha256:p.predictions_sha256})),auditor_sha256:sha(await fs.readFile(process.argv[1])),trace_checker_sha256:sha(await fs.readFile('scripts/math-actor-trace.mjs')),scalar_reader_sha256:sha(await fs.readFile('scripts/math-answer-score.mjs')),scope:'All own-result traces verified before reading development final scalars in this recount; already consumed development, not a fresh reserve.'},null,2),{flag:'wx'});
const partitionBytes=await fs.readFile(protocol.partition);assert.equal(sha(partitionBytes),protocol.partition_sha256);const partition=JSON.parse(partitionBytes);
const raw=await fs.readFile('.axon-memory/corpora/gsm8k-3101c7d50724/train.jsonl');assert.equal(sha(raw),protocol.source_train_sha256);
const records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse),accesses={train:0,development:0,reserved:0,research_only:0};
for(const [i,r]of records.entries())if(['reserved','research_only'].includes(partition[i].split))Object.defineProperty(r,'answer',{get(){accesses[partition[i].split]++;throw Error('sealed answer');}});
const counts=new Map();for(const p of partition)if(p.split==='train'){accesses.train++;const n=finalAnswer(records[p.ordinal].answer);if(n!==null)counts.set(n,(counts.get(n)??0)+1);}
const constant=[...counts].sort((a,b)=>b[1]-a[1]||a[0]-b[0])[0][0],gold=new Map();
for(const p of partition)if(p.split==='development'){accesses.development++;gold.set(String(p.ordinal),finalAnswer(records[p.ordinal].answer));}
assert.equal(gold.size,1481);assert([...gold.values()].every(n=>n!==null));assert.equal(accesses.reserved+accesses.research_only,0);
const quantities=inputs.map(input=>input.text.toLowerCase().trim().replace(/[.?!]+$/u,'').split(/\s+/u).flatMap(word=>{try{return[decimal(word)];}catch{return[];}}));
const copyControls={first_quantity:{correct:inputs.filter((r,i)=>quantities[i].length&&close(quantities[i][0],gold.get(r.id))).length,total:1481},last_quantity:{correct:inputs.filter((r,i)=>quantities[i].length&&close(quantities[i].at(-1),gold.get(r.id))).length,total:1481},gold_matches_any_input_quantity:inputs.filter((r,i)=>quantities[i].some(n=>close(n,gold.get(r.id)))).length,scope:'Post-hoc no-training copy controls on already consumed development; whole question, no64-token model cap. Any-quantity statistic is an oracle overlap diagnostic, not an inferred prediction or evidence of understanding.'};
for(const baseline of [copyControls.first_quantity,copyControls.last_quantity]){baseline.covered=quantities.filter(q=>q.length>0).length;baseline.incorrect=baseline.covered-baseline.correct;baseline.refused=1481-baseline.covered;}
const results=[];
for(const {result,rows,traces}of checked){let correct=0,incorrect=0,refused=0,calculatedCorrect=0,multistepCorrect=0,newOrderCorrect=0,inputValueCorrect=0;const errors={},depths={};
  const scored=rows.map((r,i)=>{const ok=r.answer!==null&&close(r.answer,gold.get(r.id)),t=traces[i],alreadyInQuestion=quantities[i].some(n=>r.answer!==null&&close(n,r.answer));if(ok){correct++;calculatedCorrect+=Number(t.calculations>0);multistepCorrect+=Number(t.calculations>1);newOrderCorrect+=Number(t.new_order);inputValueCorrect+=Number(alreadyInQuestion);}else if(r.answer===null)refused++;else incorrect++;
    const depth=String(t.calculations);if(!depths[depth])depths[depth]={correct:0,incorrect:0,refused:0};depths[depth][ok?'correct':r.answer===null?'refused':'incorrect']++;
    if(r.error)errors[r.error]=(errors[r.error]??0)+1;return {id:r.id,expected:gold.get(r.id),actual:r.answer,correct:ok,answer_already_in_question:alreadyInQuestion,...t};});
  await fs.writeFile(path.join(output,`policy-${result.index}-scored.json`),JSON.stringify(scored,null,2),{flag:'wx'});
  results.push({index:result.index,name:result.policy.name,correct,incorrect,refused,total:1481,accuracy:correct/1481,calculated_correct:calculatedCorrect,multistep_correct:multistepCorrect,correct_answer_already_in_question:inputValueCorrect,correct_answer_not_in_question:correct-inputValueCorrect,unseen_operation_order_correct:newOrderCorrect,depths,error_reasons:errors,passed:correct/1481>=.95});
}
const report={schema:'axon-math-actor-recount/v1',results,constant:{value:constant,correct:[...gold.values()].filter(n=>close(n,constant)).length,total:1481},copy_controls:copyControls,answer_accesses:accesses,development_calculation_traces_interpreted:false,own_result_traces_verified:checked.reduce((n,p)=>n+p.rows.length,0),future_teacher_states_used:false,m1_complete:false,m2_permitted:false,scope:'End-to-end scalar correctness over full denominator. Every action operand copied, arithmetic/result history independently recounted. Multiple calculations may be algebraically redundant; new operation order is not proof of a new mathematical algorithm. Engineering author of scorer is not independent; no understanding, physical-unit or discovery certificate.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
