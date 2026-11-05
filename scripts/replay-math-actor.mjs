import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha} from './gsm8k-source.mjs';

const [candidate,source,output]=process.argv.slice(2);if(!output)throw Error('usage: node scripts/replay-math-actor.mjs <lab.exe> <frozen-study> <new-output>');
await fs.mkdir(output);const binary=path.resolve(output,'candidate.exe');await fs.copyFile(candidate,binary,fs.constants.COPYFILE_EXCL);
const study=JSON.parse(await fs.readFile(path.join(source,'predictions-ready.json'))),results=[];
await fs.writeFile(path.join(output,'freeze.json'),JSON.stringify({binary_sha256:sha(await fs.readFile(binary)),original_freeze_sha256:sha(await fs.readFile(path.join(source,'freeze.json'))),script_sha256:sha(await fs.readFile(process.argv[1])),scope:'Every original actor rollout replayed with frozen weights/inputs, including all frame strings, operands, results, errors and logical work. No learning or source/development answers.'},null,2),{flag:'wx'});
for(const p of study.results){
  const bytes=await fs.readFile(path.join(source,`policy-${p.index}-combined/structural.axnum`));assert.equal(sha(bytes),p.number_sha256);const number=path.resolve(output,`policy-${p.index}.axnum`);await fs.writeFile(number,bytes,{flag:'wx'});
  const inputs=JSON.parse(await fs.readFile(path.join(source,`policy-${p.index}-inputs.json`))),expectedBytes=await fs.readFile(path.join(source,`policy-${p.index}-predictions.json`));assert.equal(sha(expectedBytes),p.predictions_sha256);const expected=JSON.parse(expectedBytes),rows=[];
  for(let start=0;start<inputs.length;start+=4){
    const name=`policy-${p.index}-${start}`,input=path.resolve(output,`${name}-inputs.json`),receipt=path.resolve(output,`${name}-predictions.json`);
    await fs.writeFile(input,JSON.stringify(inputs.slice(start,start+4)),{flag:'wx'});
    const child=await promisify(execFile)(binary,[p.policy.layout==='focused'?'math-actor-focused-batch':'math-actor-batch',number,path.resolve(output,`${name}-birth`),input,receipt],{windowsHide:true,timeout:300000,maxBuffer:1000000});
    await fs.writeFile(path.join(output,`${name}.stdout.log`),child.stdout,{flag:'wx'});const summary=JSON.parse(child.stdout);assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.peak_active_modules<=2);
    const chunk=JSON.parse(await fs.readFile(receipt));assert.deepEqual(chunk,expected.slice(start,start+4));rows.push(...chunk);
  }
  assert.equal(sha(await fs.readFile(number)),p.number_sha256);assert.equal(rows.length,1481);results.push({index:p.index,rows_identical:rows.length});
}
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify({schema:'axon-math-actor-replay/v1',results,no_training:true,no_answers_read:true,m1_complete:false,scope:'Exact backward compatibility for frozen actor behavior; not new generalization or M1 approval.'},null,2),{flag:'wx'});console.log(JSON.stringify({results}));
