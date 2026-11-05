import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const [candidate,runPath,outputPath,replacementNumber,selectedPolicy]=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/verify-encoder-replay.mjs <candidate> <frozen-encoder-run> <new-output>');
if(replacementNumber)assert(typeof selectedPolicy==='string'&&/^\d+$/u.test(selectedPolicy),'A replacement must select exactly one frozen policy');
else assert(selectedPolicy===undefined);
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const sha=b=>createHash('sha256').update(b).digest('hex');
const read=async f=>JSON.parse(await fs.readFile(path.join(runPath,f)));
const audit=await read('audit.json'),inputs=await read('development-inputs.json');
assert(['lexical-encoder-v1','segment-sequence-v1'].includes(audit.learner));
const segment=audit.learner==='segment-sequence-v1',command=segment?'segment-batch':'encoder-batch';
const binary=path.join(output,'candidate.exe');await fs.copyFile(candidate,binary,fs.constants.COPYFILE_EXCL);
await fs.copyFile(process.argv[1],path.join(output,'driver.mjs'),fs.constants.COPYFILE_EXCL);
const policies=[];
const selected=audit.policies.filter(p=>!replacementNumber||p.index===Number(selectedPolicy));assert(selected.length>0);if(replacementNumber)assert.equal(selected.length,1);
for(const p of selected){const name=`policy-${p.index}.axnum`,original=await fs.readFile(path.join(runPath,`policy-${p.index}`,'structural.axnum'));assert.equal(sha(original),p.number_sha256);const bytes=replacementNumber?await fs.readFile(replacementNumber):original;await fs.writeFile(path.join(output,name),bytes,{flag:'wx'});policies.push({index:p.index,number:name,sha256:sha(bytes),original_genome_sha256:p.number_sha256,replacement:!!replacementNumber,byte_convolution:p.training.summary.budget.byte_convolution??false,ordered_context:p.training.summary.budget.ordered_context??false});}
await write('inputs.json',inputs);await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),policies,inputs_sha256:sha(await fs.readFile(path.join(output,'inputs.json'))),scope:'Final engine replay of frozen models, no training or held-out test. Decoder and catalog identity must remain unchanged.'});
const results=[];
for(const p of policies){const rows=[];
  const chunkSize=segment?16:(p.byte_convolution||p.ordered_context)?32:128;
  for(let start=0;start<inputs.length;start+=chunkSize){
    const name=`policy-${p.index}-${start}`,input=path.join(output,`${name}-inputs.json`),receipt=path.join(output,`${name}-predictions.json`);
    await write(`${name}-inputs.json`,inputs.slice(start,start+chunkSize));
    const child=await promisify(execFile)(binary,[command,path.join(output,p.number),path.join(output,`${name}-birth`),input,receipt],{windowsHide:true,timeout:300000,maxBuffer:1000000});
    await fs.writeFile(path.join(output,`${name}.stdout.log`),child.stdout,{flag:'wx'});
    const summary=JSON.parse(child.stdout.trim());assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.loads===1&&summary.active_modules===1);rows.push(...JSON.parse(await fs.readFile(receipt)));
  }
  const previous=await read(`policy-${p.index}-predictions.json`),corrections=previous.filter(r=>r.tag_margin===2147483647).length;
  const expected=previous.map(r=>r.tag_margin===2147483647?{...r,tag_margin:null}:r);
  assert.deepEqual(rows,expected);assert.equal(sha(await fs.readFile(path.join(output,p.number))),p.sha256);
  await write(`policy-${p.index}-predictions.json`,rows);results.push({index:p.index,identical:corrections?0:rows.length,behavior_identical:rows.length,unavailable_margin_corrections:corrections,number_unchanged:true,no_teacher:true});
}
await write('audit.json',{results,m1_complete:false});console.log(JSON.stringify({results}));
