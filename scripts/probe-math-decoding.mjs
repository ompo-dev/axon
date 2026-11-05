import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const [executable,outputPath,protocolFile='evals/protocols/m1-math-constrained-decoding-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/probe-math-decoding.mjs <lab.exe> <new-output> [protocol]');
const protocol=JSON.parse(await fs.readFile(protocolFile)),source=path.resolve(protocol.source_study),output=path.resolve(outputPath);
assert(output.startsWith(path.resolve()+path.sep));assert.equal(protocol.schema,'axon-math-constrained-decoding/v1');await fs.mkdir(output);
const sha=b=>createHash('sha256').update(b).digest('hex');
const read=async f=>JSON.parse(await fs.readFile(path.join(source,f)));
const write=(file,data)=>fs.writeFile(path.join(output,file),JSON.stringify(data,null,2),{flag:'wx'});
const old=await read('predictions-ready.json'),oldFreeze=await read('freeze.json'),library=await read('library.json');
const binary=path.join(output,'lab.exe');await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
const sources={};for(const file of [protocolFile,'evals/protocols/m1-public-math-programs-v1.json','evals/protocols/m1-completion-v1.json','src/structural_math_lab.rs','src/structural_lexical_encoder.rs','src/structural_sequence_search.rs','src/structural_lab.rs','src/curriculum.rs','scripts/probe-math-decoding.mjs','scripts/audit-math-programs.mjs','scripts/math-answer-score.mjs','scripts/math-answer-score.test.mjs']){
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);const dest=path.join(output,'source',file);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});
}
for(const f of ['parent.axnum','development-inputs.json'])await fs.copyFile(path.join(source,f),path.join(output,f),fs.constants.COPYFILE_EXCL);
assert.equal(sha(await fs.readFile(path.join(output,'parent.axnum'))),oldFreeze.parent_sha256);
const inputs=await read('development-inputs.json');assert.equal(inputs.length,1481);
const files={};for(const lib of library.filter(p=>p.admitted))files[lib.id]=sha(await fs.readFile(lib.file));
await write('freeze.json',{binaries:{lab:{sha256:sha(await fs.readFile(binary))}},sources,inputs_sha256:sha(await fs.readFile(path.join(output,'development-inputs.json'))),parent_sha256:oldFreeze.parent_sha256,original_study_freeze_sha256:sha(await fs.readFile(path.join(source,'freeze.json'))),library_sha256:files,scope:'Fixed decoding correction on previously consumed development, frozen weights/library. No new learning or reserved-label access.'});
const calls=[];
async function core(name,args){
  const t=performance.now(),child=await promisify(execFile)(binary,args,{windowsHide:true,timeout:300000,maxBuffer:8000000});
  await fs.writeFile(path.join(output,`${name}.stdout.log`),child.stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),child.stderr,{flag:'wx'});
  const call={order:calls.length,name,binary:'lab',arguments:args,exit_code:0,whole_process_ms:performance.now()-t};calls.push(call);await write(`call-${call.order}.json`,call);return {call,summary:JSON.parse(child.stdout)};
}
const results=[];
for(const p of old.results){
  const manifest={encoder:path.join(source,`policy-${p.index}-model/semantic-lexical-encoder-v1.axseq`),circuits:library.filter(q=>q.admitted).map(({id,file})=>({id,file}))};
  assert.equal(sha(await fs.readFile(manifest.encoder)),sha(await fs.readFile(path.join(source,`policy-${p.index}-combined/math-language.axseq`))));
  await write(`model-${p.index}-manifest.json`,manifest);
  const packed=await core(`model-${p.index}-pack`,['math-pack',path.join(output,'parent.axnum'),path.join(output,`model-${p.index}-pack-birth`),path.join(output,`model-${p.index}-manifest.json`),path.join(output,`model-${p.index}-combined`)]);
  const number=path.join(output,`model-${p.index}-combined/structural.axnum`),numberBytes=await fs.readFile(number),policyInputs=await read(`policy-${p.index}-inputs.json`);
  for(const plan of protocol.plans){
    const index=results.length,name=`policy-${index}`;await fs.mkdir(path.join(output,`${name}-combined`));await fs.writeFile(path.join(output,`${name}-combined/structural.axnum`),numberBytes,{flag:'wx'});
    const rows=[],summaries=[];
    for(let start=0;start<inputs.length;start+=protocol.batch_size){
      const chunk=`${name}-dev-${start}`;await write(`${chunk}-inputs.json`,policyInputs.slice(start,start+protocol.batch_size));
      const call=await core(chunk,[plan.command,number,path.join(output,`${chunk}-birth`),path.join(output,`${chunk}-inputs.json`),path.join(output,`${chunk}-predictions.json`)]);
      assert(call.summary.seed_unchanged&&call.summary.teacher_queries===0&&call.summary.peak_active_modules<=1);summaries.push(call.summary);
      rows.push(...JSON.parse(await fs.readFile(path.join(output,`${chunk}-predictions.json`))));
    }
    assert.deepEqual(rows.map(r=>r.id),inputs.map(r=>r.id));assert.deepEqual(await fs.readFile(number),numberBytes);
    let punctuation_corrections=0;
    if(plan.command==='math-batch'){
      const previous=await read(`policy-${p.index}-predictions.json`);
      for(const [i,row]of rows.entries()){
        if(previous[i].error!=='math operand source/role')assert.deepEqual(row,previous[i],'No unannounced greedy changes');
        else{assert.equal(row.language_work,previous[i].language_work);if(JSON.stringify(row)!==JSON.stringify(previous[i]))punctuation_corrections++;}
      }
    }
    await write(`${name}-predictions.json`,rows);
    results.push({index,source_policy:p.index,policy:{name:`${p.policy.name}/${plan.name}`},punctuation_corrections,number_sha256:sha(numberBytes),predictions_sha256:sha(await fs.readFile(path.join(output,`${name}-predictions.json`))),predictions:rows.length,answers:rows.filter(r=>r.answer!==null).length,loads:summaries.reduce((n,s)=>n+s.loads,0),language_work:summaries.reduce((n,s)=>n+s.language_work,0),peak_active_payload_bytes:Math.max(...summaries.map(s=>s.peak_active_payload_bytes)),peak_active_modules:Math.max(...summaries.map(s=>s.peak_active_modules)),pack:packed.summary});
  }
}
await write('predictions-ready.json',{schema:'axon-math-program-study/v1',results,calls,development_answers_accessed:false,reserved_answers_accessed:false,m1_complete:false,m2_permitted:false,scope:'No answer reads during this probe. This is consumed development, not a new reserve. Greedy exact replay plus shape-constrained beam; no learning or semantic certificate.'});
console.log(JSON.stringify({results:results.map(r=>({index:r.index,name:r.policy.name,answers:r.answers,work:r.language_work,peak:r.peak_active_payload_bytes})),calls:calls.length,m1_complete:false}));
