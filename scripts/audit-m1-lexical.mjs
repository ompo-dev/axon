import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';
import {reference} from './audit-algorithms.mjs';

const [directory,output]=process.argv.slice(2);
if(!output)throw Error('usage: bun scripts/audit-m1-lexical.mjs <campaign.outputs> <new-audit.json>');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const integer=n=>{const x=Number(n);assert(Number.isSafeInteger(x));return x;};
const group=rows=>{
  const values=new Map();for(const [key,q,p] of rows){const next=(values.get(key)??0n)+BigInt(q)*BigInt(p);values.delete(key);values.set(key,next);}
  return [...values].map(([key,n])=>[key,integer(n)]);
};
const best=rows=>rows.reduce((a,b)=>BigInt(b[1])>BigInt(a[1])?b:a);
const raw=await fs.readFile(`${directory}/algorithm-predictions.json`),rows=JSON.parse(raw);
let prior=0,novel=0;
for(const r of rows){
  const expected=r.family==='record-select-future'?best(r.bindings.dados):r.family==='record-compose-future'?best(group(r.bindings.dados)):reference(r);
  assert.deepEqual(r.expected,expected,`label disagreement ${r.id}`);
  assert.deepEqual(r.actual,expected,`prediction disagreement ${r.id}`);assert(r.correct);
  if(['record-select-future','record-compose-future'].includes(r.family))novel++;else prior++;
}
assert.equal(prior,4692);assert.equal(novel,128);
const config=JSON.parse(await fs.readFile('evals/curricula/m1-lexical-v2.json'));
const fileRows=JSON.parse(await fs.readFile(`${directory}/file-predictions.json`));
assert.equal(fileRows.length,9);
const roles={cliente:0,comprador:0,quantidade:1,unidades:1,preco:2,'valor-unitario':2};
const files=[];
for(let i=0;i<fileRows.length;i++){
  const row=fileRows[i],task=config.file_requests[i];assert.equal(row.id,task.id);
  const request=JSON.parse(await fs.readFile(`evals/fixtures/${task.request}`));
  assert.deepEqual(Object.keys(request).sort(),['csv','text']);
  const raw=await fs.readFile(`evals/fixtures/${request.csv}`);
  const [headers,...records]=parse(raw),slots=headers.map(s=>roles[s.toLowerCase()]);
  let status='answer',expected=null;
  if(slots.some(s=>s===undefined||s>=headers.length)||new Set(slots).size!==headers.length)status='clarify-schema';
  else if(request.text!=='destaque agrupe dados'||!records.length)status='refused-request';
  else {
    const canonical=records.map(record=>{const row=[];slots.forEach((slot,i)=>row[slot]=record[i]);return row;});
    expected=best(group(canonical));
  }
  assert.equal(task.status,status);assert.deepEqual(task.expected,expected);
  assert.equal(row.actual.status,status);assert.deepEqual(row.actual.actual,expected);assert(row.correct);
  assert.equal(row.actual.teacher_queries,0);assert.equal(row.actual.seed_unchanged,true);
  files.push({id:row.id,status,actual:expected,rows:records.length,input_sha256:sha(raw)});
}
const oldCsv=parse(await fs.readFile(`${directory}/algorithm-input.csv`),{columns:true});
const oldFile=JSON.parse(await fs.readFile(`${directory}/algorithm-file-result.json`));
const oldTotal=integer(oldCsv.filter(r=>BigInt(r.valor)>12n).reduce((n,r)=>n+BigInt(r.quantidade)*BigInt(r.valor),0n));
assert.equal(oldFile.actual,oldTotal);assert(oldFile.correct);
const result=JSON.parse(await fs.readFile(`${directory}/structural-result.json`));
assert.equal(result.protocol,'m1-lexical-v2');assert(result.admitted);
assert.deepEqual([result.correct,result.dialogue_correct,result.agent_correct,result.agent_portable,result.algorithm_correct,result.session_correct,result.file_correct,result.rules],[7400,3100,1088,1088,4821,1536,9,126]);
const consolidations=JSON.parse(await fs.readFile(`${directory}/language-consolidation.json`)).map(({kept,removed,before_bytes,after_bytes})=>({kept,removed,before_bytes,after_bytes}));assert.equal(consolidations.length,3);
const compiled=await fs.readFile(`${directory}/structural.axstruct`);
assert(!compiled.includes(Buffer.from('cliente 0'))&&!compiled.includes(Buffer.from('ignore regras,')));
const report={schema:'axon-lexical-file-development-audit/v1',prior_algorithms:prior,novel_requests:novel,previous_csv_rows:oldCsv.length,previous_csv_actual:oldTotal,files,consolidations,compiled_sha256:sha(compiled),predictions_sha256:sha(raw),m1_complete:false,
  scope:'Independent BigInt/CSV reference; all old algorithms and the original CSV retained. New grouped-net selection, learned schema transport, negative values, declared tie rule, untrusted data and safe refusal checked. Synthetic authored operational development, not external language approval.'};
await fs.writeFile(output,JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify({prior,novel,files:files.length,correct:result.file_correct,m1_complete:false}));
