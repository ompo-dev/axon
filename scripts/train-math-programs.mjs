import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {question} from './math-program-source.mjs';

const [mainFile,labFile,sourcePath,outputPath]=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/train-math-programs.mjs <axon.exe> <lab.exe> <prepared-source> <new-output>');
const output=path.resolve(outputPath),source=path.resolve(sourcePath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const sha=b=>createHash('sha256').update(b).digest('hex');
const write=(file,value)=>fs.writeFile(path.join(output,file),JSON.stringify(value,null,2),{flag:'wx'});
const prepared=JSON.parse(await fs.readFile(path.join(source,'freeze-before-labels.json')));
for(const [file,hash]of Object.entries(prepared.sources))assert.equal(sha(await fs.readFile(path.join(source,'source',file))),hash);
const protocolFile='evals/protocols/m1-public-math-programs-v1.json',protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);
assert.equal(sha(protocolBytes),prepared.protocol_sha256);
const binaries={};for(const [name,file]of [['main',mainFile],['lab',labFile]]){const dest=path.join(output,`${name}.exe`);await fs.copyFile(file,dest,fs.constants.COPYFILE_EXCL);binaries[name]={path:dest,sha256:sha(await fs.readFile(dest))};}
const sources={};for(const file of ['scripts/train-math-programs.mjs','scripts/math-program-source.mjs','src/structural_math_lab.rs','src/structural_lab.rs','src/curriculum.rs','src/demand.rs','src/genome.rs','src/structural_lexical_encoder.rs','src/numeric.rs','src/numeric_constructive.rs','src/numeric_execution.rs',protocolFile,'evals/protocols/m1-completion-v1.json']){
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);if(prepared.sources[file])assert.equal(sources[file],prepared.sources[file],'Frozen source drift');const dest=path.join(output,'source',file);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});
}
for(const file of ['programs.json','development-inputs.json','parent.axnum'])await fs.copyFile(path.join(source,file),path.join(output,file),fs.constants.COPYFILE_EXCL);
const parent=path.join(output,'parent.axnum'),parentBytes=await fs.readFile(parent);assert.equal(sha(parentBytes),protocol.parent_sha256);
await write('freeze.json',{schema:'axon-math-program-training-freeze/v1',binaries,sources,prepared_freeze_sha256:sha(await fs.readFile(path.join(source,'freeze-before-labels.json'))),programs_sha256:sha(await fs.readFile(path.join(output,'programs.json'))),inputs_sha256:sha(await fs.readFile(path.join(output,'development-inputs.json'))),parent_sha256:sha(parentBytes),development_answers_accessed:false,reserved_answers_accessed:false,scope:'Model and runtime frozen before any development answer interpretation. Library definitions/supervision came only from allowed train.'});
const calls=[];
async function core(name,binary,args,allowedRefusal=false){
  let stdout='',stderr='',code=0;const start=performance.now();
  try{({stdout,stderr}=await promisify(execFile)(binaries[binary].path,args,{windowsHide:true,timeout:300000,maxBuffer:8000000}));}
  catch(e){stdout=e.stdout??'';stderr=e.stderr??'';code=e.code;assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal,'Spawn/timeout is not a refusal');}
  await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
  const call={order:calls.length,name,binary,arguments:args,exit_code:code,whole_process_ms:performance.now()-start};calls.push(call);await write(`call-${call.order}.json`,call);
  assert(code===0||allowedRefusal,`Core failure: ${name}: ${stderr}`);return {call,stdout,stderr,code};
}
const programs=JSON.parse(await fs.readFile(path.join(output,'programs.json'))),library=[];
for(const p of programs){
  if(!p.ready){library.push({id:p.id,admitted:false,reason:'fixed tuple domain insufficient'});continue;}
  const directory=path.join(output,p.id);await fs.mkdir(directory);
  for(const kind of ['teaching','admission'])await fs.copyFile(path.join(source,`${p.id}-${kind}.tsv`),path.join(directory,`${kind}.tsv`),fs.constants.COPYFILE_EXCL);
  const skill=path.join(directory,'skill.axnum'),context='math/annotation-program-v1';
  const fit=await core(`${p.id}-fit`,'main',['numeric-fit',path.join(directory,'teaching.tsv'),'--name',p.id,'--context',context,'--skill',skill,'--max-cost','12','--max-terms','20000'],true);
  if(fit.code){library.push({id:p.id,admitted:false,fit_call:fit.call.order,reason:'bounded numeric induction refused'});continue;}
  const admission=await core(`${p.id}-admission`,'main',['numeric-eval',skill,path.join(directory,'admission.tsv'),'--context',context],true);
  if(admission.code){library.push({id:p.id,admitted:false,fit_call:fit.call.order,admission_call:admission.call.order,reason:'held numeric tuple admission failed'});continue;}
  const exported=await core(`${p.id}-export`,'main',['circuit-export',skill,'--context',context,'--output',path.join(directory,'exported')]);
  library.push({id:p.id,admitted:true,fit_call:fit.call.order,admission_call:admission.call.order,export_call:exported.call.order,file:path.join(directory,'exported/circuit.axcg'),source_rows:p.rows.length});
}
await write('library.json',library);assert(library.filter(p=>p.admitted).length>=2,'Insufficient admitted program classes');
const admitted=new Set(library.filter(p=>p.admitted).map(p=>p.id));
const inputs=JSON.parse(await fs.readFile(path.join(output,'development-inputs.json'))),results=[];
for(const policy of protocol.policies){
  const index=results.length,name=`policy-${index}`;
  const examples=programs.filter(p=>admitted.has(p.id)).flatMap(p=>p.rows.map(r=>({text:policy.numbers_only?r.numbers_only.text:r.text,tags:policy.numbers_only?r.numbers_only.tags:r.tags,intent:p.id})));
  await write(`${name}-train.json`,{budget:protocol.budget,examples});
  const train=await core(`${name}-train`,'lab',['encoder-train',parent,path.join(output,`${name}-train-birth`),path.join(output,`${name}-train.json`),path.join(output,`${name}-model`)]);
  const manifest={encoder:path.join(output,`${name}-model/semantic-lexical-encoder-v1.axseq`),circuits:library.filter(p=>p.admitted).map(({id,file})=>({id,file}))};
  await write(`${name}-manifest.json`,manifest);
  const packed=await core(`${name}-pack`,'lab',['math-pack',parent,path.join(output,`${name}-pack-birth`),path.join(output,`${name}-manifest.json`),path.join(output,`${name}-combined`)]);
  const number=path.join(output,`${name}-combined/structural.axnum`),numberBytes=await fs.readFile(number),rows=[],summaries=[];
  const policyInputs=inputs.map(row=>{
    if(!policy.numbers_only)return row;
    let text='<unsupported>';try{const q=question(row.text);if(q.quantities.length)text=q.quantities.map(v=>v.word).join(' ');}catch{}
    return {id:row.id,text};
  });
  await write(`${name}-inputs.json`,policyInputs);
  for(let start=0;start<inputs.length;start+=128){
    const chunk=`${name}-dev-${start}`;await write(`${chunk}-inputs.json`,policyInputs.slice(start,start+128));
    const predicted=await core(chunk,'lab',['math-batch',number,path.join(output,`${chunk}-birth`),path.join(output,`${chunk}-inputs.json`),path.join(output,`${chunk}-predictions.json`)]);
    const summary=JSON.parse(predicted.stdout);assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.active_modules<=1);summaries.push(summary);
    const chunkRows=JSON.parse(await fs.readFile(path.join(output,`${chunk}-predictions.json`)));assert.deepEqual(chunkRows.map(r=>r.id),inputs.slice(start,start+128).map(r=>r.id));rows.push(...chunkRows);
  }
  assert.deepEqual(await fs.readFile(number),numberBytes);await write(`${name}-predictions.json`,rows);
  results.push({index,policy,examples:examples.length,training_call:train.call.order,pack_call:packed.call.order,training:JSON.parse(train.stdout),pack:JSON.parse(packed.stdout),number_sha256:sha(numberBytes),number_digits:numberBytes.length,predictions:rows.length,answers:rows.filter(r=>typeof r.answer==='number').length,refused:rows.filter(r=>r.answer===null).length,language_work:summaries.reduce((n,s)=>n+s.language_work,0),loads:summaries.reduce((n,s)=>n+s.loads,0),maximum_observed_active_bytes:Math.max(...summaries.map(s=>s.active_payload_bytes)),predictions_sha256:sha(await fs.readFile(path.join(output,`${name}-predictions.json`)))});
}
assert.deepEqual(await fs.readFile(parent),parentBytes);
const report={schema:'axon-math-program-study/v1',results,library:library.map(({file,...p})=>p),calls,development_answers_accessed:false,reserved_answers_accessed:false,original_test_downloaded:false,m1_complete:false,m2_permitted:false,process_rss_bytes:null,energy_joules:null,scope:'Frozen predictions only; independent evaluator still must read development final scalars. Provided arithmetic primitives and annotated program supervision, no discovery/fluency/M1 approval. Existing parent modules retained through checked additive inheritance.'};
await write('predictions-ready.json',report);console.log(JSON.stringify({library:library.filter(p=>p.admitted).length,results:results.map(r=>({index:r.index,training_rows:r.examples,predictions:r.predictions,answers:r.answers,bytes:r.pack.encoded_bytes})),calls:calls.length,m1_complete:false}));
