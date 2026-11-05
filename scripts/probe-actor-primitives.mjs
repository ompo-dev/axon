import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha} from './gsm8k-source.mjs';

const [source,outputPath]=process.argv.slice(2);if(!outputPath)throw Error('usage: node scripts/probe-actor-primitives.mjs <frozen-actor-study> <new-output>');
const output=path.resolve(outputPath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const study=JSON.parse(await fs.readFile(path.join(source,'predictions-ready.json'))),binary=path.join(output,'candidate.exe'),number=path.join(output,'brain.axnum');
await fs.copyFile(path.join(source,'main.exe'),binary,fs.constants.COPYFILE_EXCL);await fs.copyFile(path.join(source,'policy-0-combined/structural.axnum'),number,fs.constants.COPYFILE_EXCL);assert.equal(sha(await fs.readFile(number)),study.results[0].number_sha256);
const pool=[-17,-3,0,.125,5,23],operations=[['math-p032','+'],['math-p033','-'],['math-p034','*'],['math-p035','/']];
await fs.writeFile(path.join(output,'freeze.json'),JSON.stringify({binary_sha256:sha(await fs.readFile(binary)),number_sha256:study.results[0].number_sha256,script_sha256:sha(await fs.readFile(process.argv[1])),pool,operations,scope:'Cartesian numerical probe after original primitive admission; all144 cases through separate-process birth from frozen mixed genome. Targets are supplied arithmetic semantics, not new text/math discoveries. Original pseudo-random probe did not guarantee every zero-divisor combination.'},null,2),{flag:'wx'});
let correct=0,refused=0;const rows=[];
for(const [id,op]of operations)for(const a of pool)for(const b of pool){
  const index=rows.length,name=`case-${index}`,birth=path.join(output,`${name}-birth`);let expected;
  switch(op){case '+':expected=a+b;break;case '-':expected=a-b;break;case '*':expected=a*b;break;case '/':expected=b===0?null:a/b;break;default:assert.fail();}
  let stdout='',stderr='',code=0;
  try{({stdout,stderr}=await promisify(execFile)(binary,['genome-number',number,birth,id,'axon/constructive-v1',`${a},${b}`],{windowsHide:true,timeout:300000,maxBuffer:1000000}));}
  catch(e){stdout=e.stdout??'';stderr=e.stderr??'';code=e.code;assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal);}
  await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
  const fields=Object.fromEntries(stdout.trim().split(/\r?\n/u).filter(s=>s.includes('=')).map(s=>{const i=s.indexOf('=');return[s.slice(0,i),s.slice(i+1)];}));
  if(expected===null){assert.equal(code,2);assert(stderr.includes('division by zero'));refused++;}
  else{assert.equal(code,0);const actual=Number(fields.value);assert(Number.isFinite(actual)&&Math.abs(actual-expected)<=1e-9*(1+Math.abs(expected)));assert.equal(fields.module_loads,'1');assert.equal(fields.teacher_queries,'0');correct++;}
  rows.push({index,program:id,values:[a,b],expected,expected_refusal:expected===null,exit_code:code,actual:fields.value===undefined?null:Number(fields.value)});
}
assert.equal(correct,138);assert.equal(refused,6);assert.equal(sha(await fs.readFile(number)),study.results[0].number_sha256);
const result={schema:'axon-actor-primitive-cartesian/v1',rows,correct,domain_refused:refused,total:rows.length,number_unchanged:true,teacher_queries:0,m1_complete:false,scope:'All supplied arithmetic cases and zero-divisor refusals verified from the mixed numeric genome; no new linguistic, physical-unit or scientific-discovery claim. Birth eagerly installs the genome; positive queries load only the requested numerical module.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(result,null,2),{flag:'wx'});console.log(JSON.stringify({correct,domain_refused:refused,total:rows.length,m1_complete:false}));
