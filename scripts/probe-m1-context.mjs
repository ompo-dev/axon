import fs from 'node:fs/promises';
import path from 'node:path';
import {execFile} from 'node:child_process';
import {promisify,isDeepStrictEqual} from 'node:util';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';

const [numberFile, outputPath, executable] = process.argv.slice(2);
if(!executable)throw Error('usage: bun scripts/probe-m1-context.mjs <full-number> <new-output> <candidate.exe>');
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const sha=raw=>createHash('sha256').update(raw).digest('hex');
const number=path.join(output,'isolated.axnum'),binary=path.join(output,'candidate.exe');
await fs.copyFile(numberFile,number,fs.constants.COPYFILE_EXCL);
await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
const config=JSON.parse(await fs.readFile('evals/curricula/m1-context-v1.json'));
const interfaces=Object.fromEntries(config.agent.episodes[0].interfaces.map(t=>[t.name,{arity:2,epoch:t.epoch,probes:t.observations.map(o=>o.input)}]));
const turns=[],answers=[];
const add=(text,status,value=null,unit='BRL')=>{
  const id=`portable-${turns.length}`;turns.push({id,text,source:'child'});answers.push({id,status,value:value===null?null:{value,unit}});
};
add('saldo de ana maria-0','unknown');
add('o saldo dele','clarify');
for(let i=0;i<24;i++){
  const name=`pessoa nova-${i}`,n=1001+i*11;
  add(`defina saldo de ${name} como ${n} reais`,'recorded',n);
  add(`duplique saldo de ${name}`,'answer',Number(2n*BigInt(n)));
  add(`considere saldo de ${name}`,'context');
  add('some o saldo dele com 7 reais','answer',Number(BigInt(n)+7n));
  add(`defina saldo de ${name} como ${n+5} reais`,'recorded',n+5);
  add(`desconte 3 reais de saldo de ${name}`,'answer',Number(BigInt(n)+2n));
  add(`some saldo de ${name} com 2 quilos`,'unit-mismatch');
}
await write('inputs.json',{interfaces,turns});await write('private-answers.json',answers);
await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),number_sha256:sha(await fs.readFile(number)),inputs_sha256:sha(await fs.readFile(path.join(output,'inputs.json'))),private_answers_sha256:sha(await fs.readFile(path.join(output,'private-answers.json'))),
  scope:'Separate child process receives only the compiled number and statements/requests with interface signatures/probes, no teacher outputs or future labels. Parent private facts/focus must be absent. New personal observations are input experience, not inherited training. Logical isolation, not an OS sandbox.'});
const started=performance.now();
const run=await promisify(execFile)(binary,['session-batch',number,path.join(output,'birth'),path.join(output,'inputs.json'),path.join(output,'predictions.json')],{cwd:output,windowsHide:true,timeout:300000,maxBuffer:1000000});
await fs.writeFile(path.join(output,'stdout.log'),run.stdout,{flag:'wx'});await fs.writeFile(path.join(output,'stderr.log'),run.stderr,{flag:'wx'});
const summary=JSON.parse(run.stdout.trim()),rows=JSON.parse(await fs.readFile(path.join(output,'predictions.json')));
assert.equal(rows.length,170);assert.equal(summary.denied_queries,0);assert.equal(summary.seed_unchanged,true);
let correct=0;
for(let i=0;i<rows.length;i++){
  const r=rows[i],a=answers[i];assert.equal(r.id,a.id);
  if(r.actual?.status===a.status && isDeepStrictEqual(r.actual?.value,a.value))correct++;
  if(['unknown','clarify','unit-mismatch'].includes(r.actual?.status))assert.equal(r.episodes_before,r.episodes_after);
}
assert.equal(correct,170);
await write('audit.json',{correct,total:170,parent_facts_absent:rows[0].actual.status==='unknown',parent_focus_absent:rows[1].actual.status==='clarify',teacher_queries:summary.denied_queries,seed_unchanged:summary.seed_unchanged,whole_harness_wall_ms:performance.now()-started,predictions_sha256:sha(await fs.readFile(path.join(output,'predictions.json'))),
  scope:'New personal names/values in declared same-author grammar. Actual separate-process inheritance without parent personal memory; not independent-authorship language or M1 closure. RSS/energy unavailable in this harness.'});
console.log(JSON.stringify({correct,total:170,teacher_queries:summary.denied_queries,seed_unchanged:summary.seed_unchanged}));
