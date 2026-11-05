import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {finalAnswer} from './math-answer-score.mjs';

const [run,output]=process.argv.slice(2);
if(!output)throw Error('usage: node scripts/audit-math-programs.mjs <predictions-ready-study> <new-audit-directory>');
const sha=b=>createHash('sha256').update(b).digest('hex');
const read=async file=>JSON.parse(await fs.readFile(path.join(run,file)));
const frozen=await read('freeze.json'),study=await read('predictions-ready.json');
assert.equal(study.schema,'axon-math-program-study/v1');assert.equal(study.development_answers_accessed,false);
for(const [file,hash]of Object.entries(frozen.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',file))),hash);
for(const [name,info]of Object.entries(frozen.binaries))assert.equal(sha(await fs.readFile(path.join(run,`${name}.exe`))),info.sha256);
const protocol=JSON.parse(await fs.readFile(path.join(run,'source/evals/protocols/m1-public-math-programs-v1.json')));
const partitionBytes=await fs.readFile(protocol.partition);assert.equal(sha(partitionBytes),protocol.partition_sha256);
const partition=JSON.parse(partitionBytes),inputBytes=await fs.readFile(path.join(run,'development-inputs.json'));assert.equal(sha(inputBytes),frozen.inputs_sha256);
const inputs=JSON.parse(inputBytes),development=partition.filter(r=>r.split==='development');assert.equal(inputs.length,1481);assert.deepEqual(inputs.map(r=>r.id),development.map(r=>String(r.ordinal)));
const policies=[];
for(const result of study.results){
  const bytes=await fs.readFile(path.join(run,`policy-${result.index}-predictions.json`));assert.equal(sha(bytes),result.predictions_sha256);
  const rows=JSON.parse(bytes);assert.deepEqual(rows.map(r=>r.id),inputs.map(r=>r.id));assert.equal(rows.length,1481);
  assert.equal(sha(await fs.readFile(path.join(run,`policy-${result.index}-combined/structural.axnum`))),result.number_sha256);
  for(const row of rows)assert(row.answer===null||typeof row.answer==='number'&&Number.isFinite(row.answer));
  policies.push({result,rows,sha256:sha(bytes)});
}
for(const [order,call]of study.calls.entries())assert.deepEqual(call,await read(`call-${order}.json`));
await fs.mkdir(output);
await fs.writeFile(path.join(output,'freeze-before-gold.json'),JSON.stringify({predictions:policies.map(p=>({index:p.result.index,sha256:p.sha256})),auditor_sha256:sha(await fs.readFile(process.argv[1])),scalar_reader_sha256:sha(await fs.readFile('scripts/math-answer-score.mjs')),scope:'All predictions/identities verified before reading development answer properties in this scorer. Recount of already consumed development, not a new reserve. Original scorer offset error retained.'},null,2),{flag:'wx'});
const raw=await fs.readFile('.axon-memory/corpora/gsm8k-3101c7d50724/train.jsonl');assert.equal(sha(raw),protocol.source_train_sha256);
const records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse);
const accesses={train:0,development:0,reserved:0,research_only:0};
for(const [index,record]of records.entries()){
  const split=partition[index].split;
  if(['reserved','research_only'].includes(split))Object.defineProperty(record,'answer',{get(){accesses[split]++;throw Error('Sealed answer');}});
}
function gold(index){
  const split=partition[index].split;assert(['train','development'].includes(split));accesses[split]++;
  return finalAnswer(records[index].answer);
}
const counts=new Map(),literal=new Map();
for(const p of partition)if(p.split==='train'){
  const n=gold(p.ordinal);if(n!==null)counts.set(n,(counts.get(n)??0)+1);
  const key=records[p.ordinal].question.toLowerCase().trim().replace(/\s+/gu,' ');
  if(literal.has(key)&&literal.get(key)!==n)literal.set(key,null);else if(!literal.has(key))literal.set(key,n);
}
const mostCommon=[...counts].sort((a,b)=>b[1]-a[1]||a[0]-b[0])[0][0];
const golds=development.map(p=>gold(p.ordinal));
assert.equal(accesses.reserved+accesses.research_only,0);
const agrees=(a,b)=>a!==null&&b!==null&&Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-9+1e-9*Math.abs(b);
const results=[];
for(const {result,rows}of policies){
  let correct=0,incorrect=0,refused=0;const errors={};
  const scored=rows.map((row,i)=>{const ok=agrees(row.answer,golds[i]);if(ok)correct++;else if(row.answer===null)refused++;else incorrect++;if(row.error)errors[row.error]=(errors[row.error]??0)+1;return {id:row.id,expected:golds[i],actual:row.answer,program:row.program,values:row.values,correct:ok,error:row.error};});
  const summary={index:result.index,name:result.policy.name,correct,incorrect,refused,total:rows.length,accuracy:correct/rows.length,coverage:(correct+incorrect)/rows.length,error_reasons:errors,passed:correct/rows.length>=0.95};results.push(summary);
  await fs.writeFile(path.join(output,`policy-${result.index}-scored.json`),JSON.stringify(scored,null,2),{flag:'wx'});
}
const literalPredictions=inputs.map(r=>literal.get(r.text.toLowerCase().trim().replace(/\s+/gu,' '))??null);
const baselines={constant:{value:mostCommon,correct:golds.filter(n=>agrees(mostCommon,n)).length,total:1481},literal:{correct:literalPredictions.filter((n,i)=>agrees(n,golds[i])).length,coverage:literalPredictions.filter(n=>n!==null).length,total:1481}};
const audit={schema:'axon-math-program-development-recount/v1',results,baselines,answer_accesses:accesses,unscorable_gold:golds.filter(n=>n===null).length,development_calculation_annotations_parsed:false,reserved_answers_opened:false,original_test_downloaded:false,m1_complete:false,m2_permitted:false,scope:'Independent scalar/log recount, not independent authorship of the evaluator. Full1481 denominator, including domain refusals and wrong answers. Question-free success is a shortcut diagnostic, not mathematical language understanding. No source annotation/equation during core inference.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(audit,null,2),{flag:'wx'});console.log(JSON.stringify(audit));
