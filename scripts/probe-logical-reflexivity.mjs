import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';
const [candidate,previousNumber,augmentedNumber,corpus,outputPath,hypothesisNumber]=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/probe-logical-reflexivity.mjs <engine> <previous-number> <augmented-number> <corpus> <new-output>');
const output=path.resolve(outputPath);await fs.mkdir(output);const write=(n,x)=>fs.writeFile(path.join(output,n),JSON.stringify(x,null,2),{flag:'wx'});const sha=b=>createHash('sha256').update(b).digest('hex');
const provenance=JSON.parse(await fs.readFile(path.join(corpus,'provenance.json')));const source={};
for(const split of ['train','test']){const b=await fs.readFile(path.join(corpus,`${split}.csv`));assert.equal(sha(b),provenance.files.find(r=>r.name===`${split}.csv`).sha256);source[split]=parse(b,{columns:true,bom:true});}
const canonical=s=>s.toLowerCase().trim().replace(/\s+/gu,' ');const taught=new Set(source.train.flatMap(r=>[canonical(r.premise),canonical(r.hypothesis)]));
const novel=new Map(source.test.flatMap(r=>[r.premise,r.hypothesis]).filter(s=>!taught.has(canonical(s))).map(s=>[canonical(s),s]));
const statements=[...novel].sort((a,b)=>sha(a[0]).localeCompare(sha(b[0]))).map(([,s])=>s);assert(statements.length>0&&statements.length<=10000);
const inputs=statements.map((s,i)=>({id:`reflexivity-${i}`,premise:s,hypothesis:s}));await write('inputs.json',inputs);
const reviewed={Contradiction:0,Entailment:1,Neutral:2};
const contrasts=source.test.map((r,i)=>({...r,index:i})).filter(r=>r.label!=='1'&&r.low_quality==='N'&&reviewed[r.reviwers_output]===Number(r.label)&&!taught.has(canonical(r.premise))&&!taught.has(canonical(r.hypothesis)));
const contrastInputs=contrasts.flatMap((r,i)=>[{id:`contrast-${i}-original`,premise:r.premise,hypothesis:r.hypothesis},{id:`contrast-${i}-identity`,premise:r.hypothesis,hypothesis:r.hypothesis}]);
assert(contrastInputs.length>0&&contrastInputs.length<=10000);await write('contrast-inputs.json',contrastInputs);await write('contrast-source.json',contrasts.map(r=>({original_test_index:r.index,original_label:Number(r.label),identity_label:1})));
const binary=path.join(output,'candidate.exe');await fs.copyFile(candidate,binary,fs.constants.COPYFILE_EXCL);
const numbers={previous:previousNumber,augmented:augmentedNumber};const identities={};
if(hypothesisNumber)numbers.hypothesis_only=hypothesisNumber;
for(const[name,file]of Object.entries(numbers)){const b=await fs.readFile(file);identities[name]=sha(b);await fs.writeFile(path.join(output,`${name}.axnum`),b,{flag:'wx'});}
await fs.mkdir(path.join(output,'source'),{recursive:true});await fs.copyFile(process.argv[1],path.join(output,'source/probe-logical-reflexivity.mjs'),fs.constants.COPYFILE_EXCL);
await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),identities,driver_sha256:sha(await fs.readFile(process.argv[1])),inputs_sha256:sha(await fs.readFile(path.join(output,'inputs.json'))),contrast_inputs_sha256:sha(await fs.readFile(path.join(output,'contrast-inputs.json'))),pairs:inputs.length,contrasts:contrasts.length,scope:'Derived identity-property and same-hypothesis controls on public test statements absent from train, but original source test was previously consumed. Self-entailment target is a supplied logical axiom, not new human annotation or full semantic benchmark. No target passed to inference.'});
const results=[];
for(const name of Object.keys(numbers))for(const batch of (name==='hypothesis_only'?['contrast']:['identity','contrast'])){
  const prefix=`${name}-${batch}`,data=batch==='identity'?inputs:contrastInputs;
  const child=await promisify(execFile)(binary,['relation-network-batch',path.join(output,`${name}.axnum`),path.join(output,`${prefix}-birth`),path.join(output,batch==='identity'?'inputs.json':'contrast-inputs.json'),path.join(output,`${prefix}-predictions.json`)],{cwd:output,windowsHide:true,timeout:600000,maxBuffer:64000000});
  await fs.writeFile(path.join(output,`${prefix}.stdout.log`),child.stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${prefix}.stderr.log`),child.stderr,{flag:'wx'});
  const rows=JSON.parse(await fs.readFile(path.join(output,`${prefix}-predictions.json`)));assert.equal(rows.length,data.length);rows.forEach((r,i)=>assert.equal(r.id,data[i].id));
  const summary=JSON.parse(child.stdout.trim().split(/\r?\n/u).at(-1));assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.peak_active_modules===1);
  if(batch==='identity')results.push({name,batch,total:rows.length,correct:rows.filter(r=>r.prediction===1).length,constant_entailment_correct:rows.length,summary});
  else {
    let original=0,identity=0,both=0;
    contrasts.forEach((r,i)=>{const a=rows[2*i].prediction===Number(r.label),b=rows[2*i+1].prediction===1;original+=Number(a);identity+=Number(b);both+=Number(a&&b);if(name==='hypothesis_only')assert.equal(rows[2*i].prediction,rows[2*i+1].prediction);});
    if(name==='hypothesis_only')assert.equal(both,0);
    results.push({name,batch,total_rows:rows.length,pairs:contrasts.length,original_correct:original,identity_correct:identity,both_correct:both,constant_entailment_both_correct:0,summary});
  }
}
await write('audit.json',{results,m1_complete:false,scope:'One supplied logical property only. Do not substitute this score for independent negation, roles, condition, ambiguity, math/CSV or conversation gates.'});console.log(JSON.stringify({results}));
