import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha} from './gsm8k-source.mjs';

const [binary,diagnostic,out]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/probe-math-actor-diagnostic-boundaries.mjs <lab.exe> <diagnostic-run> <new-output>');
const output=path.resolve(out);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const frozen=JSON.parse(await fs.readFile(path.join(diagnostic,'freeze.json'))),hash=sha(await fs.readFile(binary));assert.equal(hash,frozen.binary_sha256);
const calls=[];
for(const index of [0,3]){
  const model=frozen.models[index],focused=model.policy.layout==='focused',frame=JSON.parse(await fs.readFile(path.join(diagnostic,`model-${index}-teacher.json`)))[0].frame,command=focused?'math-actor-focused-frames-batch':'math-actor-frames-batch';
  const cases=[
    {name:'max32',rows:Array.from({length:32},(_,i)=>({id:String(i),text:frame})),valid:true},
    {name:'overflow33',rows:Array.from({length:33},(_,i)=>({id:String(i),text:frame}))},
    {name:'duplicate',rows:[{id:'same',text:frame},{id:'same',text:frame}]},
    {name:'teacher-field',rows:[{id:'wrong',text:frame,expected:13}]},
    {name:'empty',rows:[{id:'empty',text:''}]},
    {name:'tokens65',rows:[{id:'long',text:Array(65).fill('word').join(' ')}]}
  ];
  for(const c of cases){
    const name=`model-${index}-${c.name}`,input=path.join(output,`${name}-inputs.json`),birth=path.join(output,`${name}-birth`),receipt=path.join(output,`${name}-predictions.json`);
    await fs.writeFile(input,JSON.stringify(c.rows),{flag:'wx'});let code=0,stdout='',stderr='';
    try{({stdout,stderr}=await promisify(execFile)(path.resolve(binary),[command,model.number,birth,input,receipt],{windowsHide:true,timeout:300000,maxBuffer:1000000}));}
    catch(e){code=e.code;stdout=e.stdout??'';stderr=e.stderr??'';assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal,'Spawn failures are not native refusals');}
    await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
    if(c.valid){
      assert.equal(code,0,stderr);const summary=JSON.parse(stdout),rows=JSON.parse(await fs.readFile(receipt));assert.equal(summary.total,32);assert(summary.seed_unchanged&&summary.supplied_states&&summary.arithmetic_evaluations===0&&summary.peak_active_modules<=1);assert.equal(rows.length,32);
      for(const [i,row]of rows.entries()){assert.equal(row.id,String(i));assert.equal(row.frame,frame);assert.deepEqual(row.prediction,rows[0].prediction);assert.equal(row.work,rows[0].work);}
    }else{
      assert.notEqual(code,0);for(const absent of [birth,receipt])await assert.rejects(fs.access(absent));
    }
    assert.equal(sha(await fs.readFile(model.number)),model.number_sha256);calls.push({index,name,accepted:!!c.valid,exit_code:code,refused_before_birth:!c.valid});
  }
}
const audit={schema:'axon-math-actor-diagnostic-boundaries/v1',binary_sha256:hash,script_sha256:sha(await fs.readFile(process.argv[1])),calls,valid_batches:2,valid_supplied_frames:64,invalid_batches_refused_before_birth:10,teacher_state_success_is_not_autonomous_success:true,m1_complete:false};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(audit,null,2),{flag:'wx'});console.log(JSON.stringify({valid_frames:64,negative_refused_before_birth:10,verified:true}));
