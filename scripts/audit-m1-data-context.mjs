import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify,isDeepStrictEqual} from 'node:util';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';

const [runPath,outputPath,candidatePath,replacementNumber]=process.argv.slice(2);
if(!outputPath||replacementNumber&&!candidatePath)throw Error('usage: node scripts/audit-m1-data-context.mjs <run> <new-output> [candidate] [replacement-number]');
const run=path.resolve(runPath),output=path.resolve(outputPath);await fs.mkdir(output);
const sha=raw=>createHash('sha256').update(raw).digest('hex');
const read=async file=>JSON.parse(await fs.readFile(path.join(run,file)));
const write=(file,value)=>fs.writeFile(path.join(output,file),JSON.stringify(value,null,2),{flag:'wx'});
await fs.copyFile(process.argv[1],path.join(output,'auditor.mjs'),fs.constants.COPYFILE_EXCL);
const frozen=await read('freeze.json');
for(const [file,h]of Object.entries(frozen.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',file))),h);
for(const [file,h]of Object.entries(frozen.files))assert.equal(sha(await fs.readFile(path.join(run,file))),h);
const binary=path.join(run,'candidate.exe');assert.equal(sha(await fs.readFile(binary)),frozen.binary_sha256);
let validationBinary=binary;
if(candidatePath){validationBinary=path.join(output,'candidate.exe');await fs.copyFile(candidatePath,validationBinary,fs.constants.COPYFILE_EXCL);}
let validationNumber=path.join(run,'trained','structural.axnum');
if(replacementNumber){validationNumber=path.join(output,'replacement.axnum');await fs.copyFile(replacementNumber,validationNumber,fs.constants.COPYFILE_EXCL);}
const validationNumberHash=sha(await fs.readFile(validationNumber));
assert.equal(sha(await fs.readFile(path.join(run,'parent.axnum'))),frozen.parent_sha256);
assert.equal(sha(await fs.readFile(path.join(run,'trained','structural.axnum'))),(await read('model-freeze.json')).number_sha256);
const cases=await read('cases.json'),results=[];
for(const variant of ['old-number','integrated']) {
  let correct=0,total=0,filesCorrect=0;
  for(let i=0;i<cases.length;i++) {
    const c=cases[i],input=await read(c.input),answers=await read(c.answer),rows=await read(`${variant}-${i}-predictions.json`);
    assert.deepEqual(Object.keys(input).sort(),['files','interfaces','turns']);
    for(const t of input.turns)assert.deepEqual(Object.keys(t).sort(),['id','source','text']);
    const fields=parse(await fs.readFile(path.join(run,c.csv)),{columns:true});
    const records=fields.map(r=>({name:r.cliente??r.comprador,q:BigInt(r.quantidade??r.unidades),price:BigInt(r.preco??r['valor-unitario'])}));
    const groups=[];let net=0n;
    for(let j=0;j<records.length;j++) {
      const r=records[j];net+=r.q*r.price;
      let group=groups.find(g=>g.name===r.name);
      if(!group){group={name:r.name,total:0n,last:0};groups.push(group);}
      group.total+=r.q*r.price;group.last=j;
    }
    groups.sort((a,b)=>a.last-b.last);let best=groups[0];for(const g of groups)if(g.total>best.total)best=g;
    assert.deepEqual(c.winner,[best.name,Number(best.total)]);assert.equal(c.net,Number(net));
    for(const j of [1,3,4,6,13]) {
      const measurement=c.measurement??'net-total';assert(['net-total','winner-amount'].includes(measurement));
      const measured=measurement==='winner-amount'?best.total:net;
      const balance=BigInt(1001+13*i),expected=j===1?measured:j===4?measured-balance:measured+balance;
      assert.equal(answers[j].value.value,Number(expected));
    }
    assert.equal(rows.length,answers.length);
    for(let j=0;j<rows.length;j++) {
      const r=rows[j],a=answers[j];assert.equal(r.id,a.id);assert.equal(r.text,input.turns[j].text);total++;
      correct+=Number(r.actual?.status===a.status&&isDeepStrictEqual(r.actual?.value,a.value)&&isDeepStrictEqual(r.actual?.inputs??[],a.inputs)&&r.actual.evidence.length===a.evidence);
      if(!['recorded','retracted','context'].includes(r.actual?.status))assert.equal(r.episodes_before,r.episodes_after);
    }
    const summary=JSON.parse((await fs.readFile(path.join(run,`${variant}-${i}.stdout.log`),'utf8')).trim());
    assert(summary.seed_unchanged&&summary.denied_queries===0&&summary.training_required===false);
    assert.equal(summary.files[0].bytes,(await fs.stat(path.join(run,c.csv))).size);
    const file=await read(`${variant}-${i}-file-result.json`);
    filesCorrect+=Number(file.status==='answer'&&isDeepStrictEqual(file.actual,c.winner));
  }
  const original=await read(`${variant}-audit.json`);assert.deepEqual([correct,total,filesCorrect],[original.correct,original.total,original.files_correct]);
  results.push({variant,correct,total,files_correct:filesCorrect,files_total:cases.length});
}
const first=await read(cases[0].input);
const malicious=[
  {...first,expected:7},
  {...first,turns:[{...first.turns[0],expected:7}]},
  {...first,files:{dados:{csv:'../parent.axnum',unit:'BRL'}}},
  {...first,files:{dados:{csv:'C:/Windows/win.ini',unit:'BRL'}}},
  {...first,files:{dados:{csv:path.join(run,cases[0].csv),unit:'BRL;delete'}}},
  {...first,files:{dados:{csv:'data.csv',unit:'BRL',expected:7}}},
  {...first,files:{dados:{csv:'data.csv',unit:'BRL;delete'}}},
  {...first,files:Object.fromEntries(Array.from({length:17},(_,i)=>[`data-${i}`,{csv:'data.csv',unit:'BRL'}]))},
];
await fs.copyFile(path.join(run,cases[0].csv),path.join(output,'data.csv'),fs.constants.COPYFILE_EXCL);
await write('negative-inputs.json',malicious);
await write('freeze.json',{binary_sha256:frozen.binary_sha256,validation_binary_sha256:sha(await fs.readFile(validationBinary)),validation_number_sha256:validationNumberHash,replacement_number:replacementNumber??null,run_freeze_sha256:sha(await fs.readFile(path.join(run,'freeze.json'))),auditor_sha256:sha(await fs.readFile(process.argv[1])),negative_inputs_sha256:sha(await fs.readFile(path.join(output,'negative-inputs.json')))});
let replayed=0,behaviorIdentical=0;
if(candidatePath)for(let i=0;i<cases.length;i++) {
  const receipt=path.join(output,`replay-${i}.json`);
  const child=await promisify(execFile)(validationBinary,['session-batch',validationNumber,path.join(output,`replay-${i}-birth`),path.join(run,cases[i].input),receipt],{windowsHide:true,timeout:30000,maxBuffer:1000000});
  await fs.writeFile(path.join(output,`replay-${i}.stdout.log`),child.stdout,{flag:'wx'});
  const rows=JSON.parse(await fs.readFile(receipt)),previous=await read(`integrated-${i}-predictions.json`);
  if(replacementNumber) {
    const behavior=rows=>rows.map(r=>({...r,actual:r.actual===null?null:{...r.actual,work:undefined}}));
    assert.deepEqual(behavior(rows),behavior(previous));behaviorIdentical+=rows.length;
  } else {assert.deepEqual(rows,previous);replayed+=rows.length;}
  const summary=JSON.parse(child.stdout.trim());assert(summary.seed_unchanged&&summary.denied_queries===0);
}
let refused=0;
for(let i=0;i<malicious.length;i++) {
  const input=path.join(output,`invalid-${i}.json`),receipt=path.join(output,`invalid-${i}-receipt.json`),birth=path.join(output,`invalid-${i}-birth`);
  await fs.writeFile(input,JSON.stringify(malicious[i]),{flag:'wx'});
  let failure;
  try {await promisify(execFile)(validationBinary,['session-batch',validationNumber,birth,input,receipt],{windowsHide:true,timeout:30000,maxBuffer:1000000});}
  catch(e){failure=e;}
  assert(failure&&failure.code===1,`invalid payload ${i} must be rejected`);
  await fs.writeFile(path.join(output,`invalid-${i}.stderr.log`),failure.stderr,{flag:'wx'});
  assert.equal(await fs.stat(receipt).catch(()=>null),null);
  assert.equal(await fs.stat(path.join(birth,'personal.axep')).catch(()=>null),null);
  refused++;
}
const brain=JSON.parse(await fs.readFile(path.join(run,'trained','structural.axstruct')));
assert(brain.rules.length<=128);
assert.equal(sha(await fs.readFile(validationNumber)),validationNumberHash);
const result={schema:'axon-m1-data-context-independent-recount/v1',results,final_candidate_replay_rows_identical:replayed,replacement_number_replay_rows_behavior_identical:behaviorIdentical,replacement_comparison:replacementNumber?'All output/status/evidence/source/mutation fields identical; work is independently reported, not required equal.':null,negative_payloads_refused:refused,negative_payloads_total:malicious.length,durable_private_mutations_from_negative_payloads:0,seed_rules:brain.rules.length,seed_rule_quota:128,validation_number_unchanged:true,unverified_metrics:{rss_bytes:null,energy_joules:null},scope:'Separate CSV parser and aggregation algorithm, same engineering author. Recount is not independent linguistic authorship, public reserve, general accounting or full M1 approval.',m1_complete:false};
await write('audit.json',result);console.log(JSON.stringify(result));
