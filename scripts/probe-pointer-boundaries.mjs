import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha} from './gsm8k-source.mjs';

const [run,out,candidate]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/probe-pointer-boundaries.mjs <pointer-study> <new-output> [candidate.exe]');
const output=path.resolve(out);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const freeze=JSON.parse(await fs.readFile(path.join(run,'freeze.json'))),study=JSON.parse(await fs.readFile(path.join(run,'predictions-ready.json'))),model=study.results[0],number=path.resolve(run,'policy-0/combined/structural.axnum'),binary=path.join(output,'candidate.exe');
await fs.copyFile(candidate??path.resolve(run,'lab.exe'),binary,fs.constants.COPYFILE_EXCL);const binaryHash=sha(await fs.readFile(binary));if(!candidate)assert.equal(binaryHash,freeze.binaries.lab.sha256);assert.equal(sha(await fs.readFile(number)),model.number_sha256);
const rows=n=>Array.from({length:n},(_,i)=>({id:String(i),text:'3'})),calls=[];
const cases=[
  ['frames32','pointer-frames-batch',rows(32),'frames'],
  ['frame544','pointer-frames-batch',[{id:'long',text:[...Array(543).fill('x'),'3'].join(' ')}],'frames'],
  ['candidates32','pointer-frames-batch',[{id:'positions',text:Array.from({length:32},(_,i)=>String(i+1)).join(' ')}],'frames'],
  ['frames33','pointer-frames-batch',rows(33),'prebirth'],
  ['frame-duplicate','pointer-frames-batch',[{id:'same',text:'3'},{id:'same',text:'4'}],'prebirth'],
  ['frame-gold','pointer-frames-batch',[{id:'gold',text:'3',expected:3}],'prebirth'],
  ['frame-empty','pointer-frames-batch',[{id:'empty',text:''}],'prebirth'],
  ['frame545','pointer-frames-batch',[{id:'long',text:[...Array(544).fill('x'),'3'].join(' ')}],'prebirth'],
  ['word257','pointer-frames-batch',[{id:'word',text:'x'.repeat(257)+' 3'}],'prebirth'],
  ['bytes-overflow','pointer-frames-batch',[{id:'bytes',text:[...Array(40).fill('x'.repeat(220)),'3'].join(' ')}],'prebirth'],
  ['candidates33','pointer-frames-batch',[{id:'positions',text:Array.from({length:33},(_,i)=>String(i+1)).join(' ')}],'prebirth'],
  ['own4','pointer-batch',rows(4),'own'],
  ['question512','pointer-batch',[{id:'long',text:[...Array(511).fill('x'),'3'].join(' ')}],'own'],
  ['question513','pointer-batch',[{id:'long',text:[...Array(512).fill('x'),'3'].join(' ')}],'row-refusal'],
  ['own5','pointer-batch',rows(5),'prebirth'],
  ['own-gold','pointer-batch',[{id:'gold',text:'3',answer:3}],'prebirth'],
  ['own-candidates','pointer-batch',[{id:'gold',text:'3',candidates:[0]}],'prebirth'],
  ['own-duplicate','pointer-batch',[{id:'same',text:'3'},{id:'same',text:'4'}],'prebirth']
];
for(const [name,command,requests,expect]of cases){
  const input=path.join(output,`${name}-inputs.json`),birth=path.join(output,`${name}-birth`),receipt=path.join(output,`${name}-predictions.json`);await fs.writeFile(input,JSON.stringify(requests),{flag:'wx'});
  let stdout='',stderr='',code=0;try{({stdout,stderr}=await promisify(execFile)(binary,[command,number,birth,input,receipt],{windowsHide:true,timeout:300000,maxBuffer:1000000}));}catch(e){code=e.code;stdout=e.stdout??'';stderr=e.stderr??'';assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal);}
  await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
  if(expect==='prebirth'){assert.notEqual(code,0,stderr);for(const file of [birth,receipt])await assert.rejects(fs.access(file));}
  else{assert.equal(code,0,stderr);const summary=JSON.parse(stdout),result=JSON.parse(await fs.readFile(receipt));assert(summary.seed_unchanged);assert.deepEqual(result.map(r=>r.id),requests.map(r=>r.id));
    if(expect==='frames'){assert(summary.supplied_states&&summary.additional_teacher_queries===0&&summary.arithmetic_evaluations===0);for(const row of result){assert(row.prediction&&!row.error,JSON.stringify(row));assert.equal(row.frame,requests.find(r=>r.id===row.id).text);}}
    else if(expect==='own'){assert.equal(summary.teacher_queries,0);for(const row of result){assert.equal(row.answer,3);assert.equal(row.arithmetic_evaluations,0);}}
    else{assert.equal(result[0].answer,null);assert(result[0].error);assert.equal(result[0].arithmetic_evaluations,0);}
  }
  assert.equal(sha(await fs.readFile(number)),model.number_sha256);calls.push({name,command,expect,exit_code:code});
}
const audit={schema:'axon-pointer-boundaries/v1',binary_sha256:binaryHash,original_binary_sha256:freeze.binaries.lab.sha256,script_sha256:sha(await fs.readFile(process.argv[1])),calls,valid_supplied_frames:34,valid_own_queries:5,invalid_batches_refused_before_birth:12,overlength_question_refused_without_calculation:1,seed_unchanged:true,m1_complete:false};
assert.equal(calls.filter(r=>r.expect==='prebirth').length,audit.invalid_batches_refused_before_birth);await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(audit,null,2),{flag:'wx'});console.log(JSON.stringify(audit));
