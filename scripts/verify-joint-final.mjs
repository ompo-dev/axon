import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const [candidate,joint,control,destination]=process.argv.slice(2);
if(!destination)throw Error('usage: node scripts/verify-joint-final.mjs <candidate> <joint-run> <control-run> <new-output>');
const output=path.resolve(destination);await fs.mkdir(output);
const write=(n,v)=>fs.writeFile(path.join(output,n),JSON.stringify(v,null,2),{flag:'wx'});
const sha=b=>createHash('sha256').update(b).digest('hex');
const binary=path.join(output,'candidate.exe');await fs.copyFile(candidate,binary,fs.constants.COPYFILE_EXCL);
await fs.copyFile(process.argv[1],path.join(output,'driver.mjs'),fs.constants.COPYFILE_EXCL);
const jobs=[];
for(const [source,command,count]of [[joint,'joint-batch',4],[control,'enhanced-batch',2]])for(let index=0;index<count;index++) {
  const name=`${command}-${index}`,number=path.join(output,`${name}.axnum`),inputs=path.join(output,`${name}-inputs.json`),expected=path.join(output,`${name}-previous.json`);
  await fs.copyFile(path.join(source,`policy-${index}`,'structural.axnum'),number,fs.constants.COPYFILE_EXCL);
  await fs.copyFile(path.join(source,'development-inputs.json'),inputs,fs.constants.COPYFILE_EXCL);
  await fs.copyFile(path.join(source,`policy-${index}-predictions.json`),expected,fs.constants.COPYFILE_EXCL);
  jobs.push({name,command,number,inputs,expected,number_sha256:sha(await fs.readFile(number))});
}
await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),jobs,scope:'Fixed final binary replay of all six trained models. Stored previous predictions are comparison evidence, never inference inputs.'});
const results=[];
for(const j of jobs) {
  const receipt=path.join(output,`${j.name}-predictions.json`);
  const child=await promisify(execFile)(binary,[j.command,j.number,path.join(output,`${j.name}-birth`),j.inputs,receipt],{windowsHide:true,timeout:300000,maxBuffer:2000000});
  await fs.writeFile(path.join(output,`${j.name}.stdout.log`),child.stdout,{flag:'wx'});
  const summary=JSON.parse(child.stdout.trim());assert(summary.seed_unchanged&&summary.teacher_queries===0);
  const rows=JSON.parse(await fs.readFile(receipt));assert.deepEqual(rows,JSON.parse(await fs.readFile(j.expected)));
  assert.equal(sha(await fs.readFile(j.number)),j.number_sha256);
  results.push({name:j.name,identical:rows.length,teacher_queries:0,number_unchanged:true});
}
await write('audit.json',{results,m1_complete:false});console.log(JSON.stringify({results}));
