import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify,isDeepStrictEqual} from 'node:util';
import {sha} from './gsm8k-source.mjs';
import {question,evaluate} from './math-program-source.mjs';
import {focusExample,canonicalExamples} from './math-actor-source.mjs';

const [mainFile,labFile,sourcePath,outputPath,experimentFile]=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/train-math-actor.mjs <axon.exe> <lab.exe> <prepared-source> <new-output>');
const output=path.resolve(outputPath),source=path.resolve(sourcePath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const write=(file,value)=>fs.writeFile(path.join(output,file),JSON.stringify(value,null,2),{flag:'wx'});
const protocolFile='evals/protocols/m1-math-actor-v1.json',protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);
const experiment=experimentFile?JSON.parse(await fs.readFile(experimentFile)):null;
if(experiment){assert(['axon-math-actor-state/v1','axon-math-actor-acquisition/v1'].includes(experiment.schema));assert.equal(experiment.base_protocol,protocolFile);protocol.policies=experiment.policies;}
const acquisition=experiment?.schema==='axon-math-actor-acquisition/v1';
const prepared=JSON.parse(await fs.readFile(path.join(source,'freeze-before-labels.json')));assert.equal(sha(protocolBytes),prepared.protocol_sha256);
for(const [file,hash]of Object.entries(prepared.sources))assert.equal(sha(await fs.readFile(path.join(source,'source',file))),hash);
const binaries={};for(const [name,file]of [['main',mainFile],['lab',labFile]]){const dest=path.join(output,`${name}.exe`);await fs.copyFile(file,dest,fs.constants.COPYFILE_EXCL);binaries[name]={path:dest,sha256:sha(await fs.readFile(dest))};}
const sources={},sourceChanges={};for(const file of ['scripts/train-math-actor.mjs',protocolFile,...(experiment?[experimentFile]:[]),'scripts/math-actor-source.mjs','scripts/math-program-source.mjs','src/structural_math_actor_lab.rs','src/structural_math_lab.rs','src/structural_lexical_encoder.rs','src/structural_sequence_search.rs','src/curriculum.rs','src/numeric.rs','src/numeric_constructive.rs','src/numeric_cli.rs','src/numeric_execution.rs','src/genome.rs','src/demand.rs','evals/protocols/m1-completion-v1.json']){
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);if(prepared.sources[file]&&sources[file]!==prepared.sources[file]){assert(experiment,'Original experiment source drift');assert(['scripts/math-actor-source.mjs','src/structural_math_actor_lab.rs','src/structural_math_lab.rs','src/curriculum.rs'].includes(file),'Undeclared state-experiment source change');sourceChanges[file]={ancestor:prepared.sources[file],current:sources[file]};}const dest=path.join(output,'source',file);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});
}
for(const file of ['development-inputs.json','episodes.json'])await fs.copyFile(path.join(source,file),path.join(output,file),fs.constants.COPYFILE_EXCL);
await fs.copyFile(acquisition?experiment.parent:path.join(source,'parent.axnum'),path.join(output,'parent.axnum'),fs.constants.COPYFILE_EXCL);
const episodes=JSON.parse(await fs.readFile(path.join(output,'episodes.json')));
for(const [i,policy]of protocol.policies.entries()){
  const examples=episodes.flatMap(e=>{const rows=(policy.numbers_only?e.free:e.full).examples;return (acquisition?canonicalExamples(rows):rows).map((r,step)=>policy.layout==='focused'?focusExample(r,step):r);});
  const budget={...protocol.budget,...(policy.relative_regions!==undefined?{relative_regions:policy.relative_regions}:{}),...(acquisition?policy.budget:{})};
  const training={budget,examples};if(!experiment)assert.deepEqual(training,JSON.parse(await fs.readFile(path.join(source,`policy-${i}-train.json`))));await write(`policy-${i}-train.json`,training);
}
const parent=path.join(output,'parent.axnum'),parentBytes=await fs.readFile(parent);assert.equal(sha(parentBytes),acquisition?experiment.parent_sha256:protocol.parent_sha256);
if(experiment){const ancestor=JSON.parse(await fs.readFile(path.join(experiment.source_study,'freeze.json')));assert.equal(sha(await fs.readFile(path.join(output,'episodes.json'))),ancestor.episodes_sha256);assert.equal(sha(await fs.readFile(path.join(output,'development-inputs.json'))),ancestor.inputs_sha256);if(!acquisition)assert.equal(sha(parentBytes),ancestor.parent_sha256);else assert.equal(sha(await fs.readFile(path.join(source,'parent.axnum'))),ancestor.parent_sha256);}
await write('freeze.json',{binaries,sources,source_changes_from_prepared_ancestor:sourceChanges,experiment:experimentFile??null,alternate_parent:acquisition?{path:experiment.parent,sha256:experiment.parent_sha256,preserved_modules:experiment.parent_modules,scope:'Explicit new branch from the pre-math two-module ancestor, NOT a descendant of the failed35-module actors. No modules removed from existing genomes. Primitive ancestry/cost retained separately.'}:null,prepared_freeze_sha256:sha(await fs.readFile(path.join(source,'freeze-before-labels.json'))),inputs_sha256:sha(await fs.readFile(path.join(output,'development-inputs.json'))),episodes_sha256:sha(await fs.readFile(path.join(output,'episodes.json'))),parent_sha256:sha(parentBytes),development_answers_accessed:false,scope:'Learned actor experiment frozen before fitting. Previous development already consumed, not a new reserve. Optional state/acquisition experiment explicitly versions changed engine/source against immutable ancestor; no new source labels. Primitive semantics and state grammar provided, operational trace supervision from allowed train only.'});
const calls=[];
async function core(name,kind,args,refusal=false){let stdout='',stderr='',code=0;const started=performance.now();
  try{({stdout,stderr}=await promisify(execFile)(binaries[kind].path,args,{windowsHide:true,timeout:300000,maxBuffer:16000000}));}
  catch(e){stdout=e.stdout??'';stderr=e.stderr??'';code=e.code;assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal,'Spawn/timeout not an AXON refusal');}
  await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
  const call={order:calls.length,name,binary:kind,arguments:args,exit_code:code,whole_process_ms:performance.now()-started};calls.push(call);await write(`call-${call.order}.json`,call);assert(code===0||refusal,`${name}: ${stderr}`);return {call,code,stdout,stderr};
}
const library=[];let state=20261025;const values=[-7,-1,0,.25,2,9];
if(experiment){
  const ancestor=JSON.parse(await fs.readFile(path.join(experiment.source_study,'library.json')));assert.equal(ancestor.length,4);assert(ancestor.every(p=>p.admitted));
  const report=JSON.parse(await fs.readFile(path.join(experiment.source_study,'predictions-ready.json'))),number=path.resolve(experiment.source_study,'policy-0-combined/structural.axnum'),birth=path.join(output,'primitive-ancestor-birth');assert.equal(sha(await fs.readFile(number)),report.results[0].number_sha256);
  await core('reuse-primitive-birth','main',['genome-birth',number,birth]);
  for(const p of ancestor){assert(protocol.primitive_library.some(q=>q.id===p.id&&q.operator===p.operator));const dir=path.join(output,p.id,'exported');await fs.mkdir(dir,{recursive:true});const file=path.join(dir,'circuit.axcg'),bytes=await fs.readFile(p.file);assert.deepEqual(bytes,await fs.readFile(path.join(birth,`${p.id}.axcg`)));await fs.writeFile(file,bytes,{flag:'wx'});library.push({id:p.id,operator:p.operator,admitted:true,file,reused:true,source_study:experiment.source_study,source_circuit_sha256:sha(bytes),ancestral_entry:p});}
}
for(const primitive of experiment?[]:protocol.primitive_library){
  const id=primitive.id,dir=path.join(output,id);await fs.mkdir(dir);for(const kind of ['teaching','admission'])await fs.copyFile(path.join(source,`${id}-${kind}.tsv`),path.join(dir,`${kind}.tsv`),fs.constants.COPYFILE_EXCL);
  const skill=path.join(dir,'skill.axnum'),context='math/actor-primitive-v1';
  const fit=await core(`${id}-fit`,'main',['numeric-fit',path.join(dir,'teaching.tsv'),'--name',id,'--context',context,'--skill',skill,'--max-cost','3','--max-terms','20000'],true);
  if(fit.code){library.push({id,admitted:false,reason:'primitive induction refused'});continue;}
  const admission=await core(`${id}-admission`,'main',['numeric-eval',skill,path.join(dir,'admission.tsv'),'--context',context],true);
  if(admission.code){library.push({id,admitted:false,reason:'primitive admission failed'});continue;}
  const finite=[],domain=[];
  for(let j=0;j<128;j++){const input=Array.from({length:2},()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return values[state%values.length];});try{finite.push({input,output:evaluate([primitive.operator,'x0','x1'],input)});}catch(e){domain.push({input,error:e.message});}}
  await fs.writeFile(path.join(dir,'probe.tsv'),'x0\tx1\ty\n'+finite.map(r=>[...r.input,r.output].join('\t')).join('\n')+'\n',{flag:'wx'});
  const probe=await core(`${id}-probe`,'main',['numeric-eval',skill,path.join(dir,'probe.tsv'),'--context',context],true);
  if(probe.code){library.push({id,admitted:false,reason:'primitive signed/fractional probe failed'});continue;}
  const exported=await core(`${id}-export`,'main',['circuit-export',skill,'--context',context,'--output',path.join(dir,'exported')]);
  let domainCall=null;if(domain.length){const d=await core(`${id}-domain`,'main',['circuit-predict',path.join(dir,'exported/circuit-number.txt'),domain[0].input.join(',')],true);assert(d.code>0);domainCall=d.call.order;}
  library.push({id,operator:primitive.operator,admitted:true,file:path.join(dir,'exported/circuit.axcg'),fit_call:fit.call.order,admission_call:admission.call.order,probe_call:probe.call.order,export_call:exported.call.order,finite_probe_cases:finite.length,domain_targets:domain.length,domain_cases_executed:domain.length?1:0,domain_call:domainCall});
}
await write('library.json',library);assert.equal(library.filter(p=>p.admitted).length,4,'Do not import ready primitives to replace failed learning');
const inputs=JSON.parse(await fs.readFile(path.join(output,'development-inputs.json')));assert.equal(inputs.length,1481);
await write('untrained-inputs.json',inputs.slice(0,4));await core('untrained-probe','lab',['math-actor-batch',parent,path.join(output,'untrained-birth'),path.join(output,'untrained-inputs.json'),path.join(output,'untrained-predictions.json')]);
assert(JSON.parse(await fs.readFile(path.join(output,'untrained-predictions.json'))).every(r=>r.answer===null));
const results=[];
function target(e){
  const words=e.text.toLowerCase().trim().replace(/[.?!]+$/u,'').split(/\s+/u),slots=[];
  for(const [i,tag]of e.tags.entries())if(tag.startsWith('B:'))slots.push([tag.slice(2),words[i].replace(/[.?!]+$/u,'')]);
  return ['intent',e.intent,slots.length?['slots',...slots]:'none'];
}
for(const [index,policy]of protocol.policies.entries()){
  const name=`policy-${index}`,training=JSON.parse(await fs.readFile(path.join(output,`${name}-train.json`))),focused=policy.layout==='focused';
  const trained=await core(`${name}-train`,'lab',['encoder-train',parent,path.join(output,`${name}-training-birth`),path.join(output,`${name}-train.json`),path.join(output,`${name}-model`)]);
  const trainingReport=JSON.parse(trained.stdout);assert(trainingReport.parent_unchanged);
  if(acquisition&&policy.expected_encoder)assert.deepEqual(await fs.readFile(path.join(output,`${name}-model/semantic-lexical-encoder-v1.axseq`)),await fs.readFile(policy.expected_encoder),'Frozen canonical full control must retain identical neural bytes');
  let fitCorrect=0,fitWork=0;
  for(let start=0;start<training.examples.length;start+=128){const chunk=`${name}-fit-${start}`;await write(`${chunk}-inputs.json`,training.examples.slice(start,start+128).map((r,i)=>({id:String(start+i),text:r.text})));const child=await core(chunk,'lab',['encoder-batch',path.join(output,`${name}-model/structural.axnum`),path.join(output,`${chunk}-birth`),path.join(output,`${chunk}-inputs.json`),path.join(output,`${chunk}-predictions.json`)]);fitWork+=JSON.parse(child.stdout).work;const rows=JSON.parse(await fs.readFile(path.join(output,`${chunk}-predictions.json`)));rows.forEach((r,i)=>{fitCorrect+=Number(isDeepStrictEqual(r.prediction,target(training.examples[start+i])));});}
  await write(`${name}-manifest.json`,{encoder:path.join(output,`${name}-model/semantic-lexical-encoder-v1.axseq`),circuits:library.map(({id,file})=>({id,file}))});
  const packed=await core(`${name}-pack`,'lab',[focused?'math-actor-focused-pack':'math-actor-pack',parent,path.join(output,`${name}-pack-birth`),path.join(output,`${name}-manifest.json`),path.join(output,`${name}-combined`)]);
  const packReport=JSON.parse(packed.stdout);assert.equal(packReport.actor_active_module_quota,2);assert.equal(packReport.old_modules_preserved,acquisition?experiment.parent_modules:30);
  const number=path.join(output,`${name}-combined/structural.axnum`),numberBytes=await fs.readFile(number);
  const policyInputs=inputs.map(r=>{if(!policy.numbers_only)return r;let text='<unsupported>';try{const q=question(r.text);if(q.quantities.length)text=q.quantities.map(v=>v.word).join(' ');}catch{}return {id:r.id,text};});
  await write(`${name}-inputs.json`,policyInputs);const rows=[],summaries=[];
  for(let start=0;start<inputs.length;start+=4){const chunk=`${name}-dev-${start}`;await write(`${chunk}-inputs.json`,policyInputs.slice(start,start+4));const child=await core(chunk,'lab',[focused?'math-actor-focused-batch':'math-actor-batch',number,path.join(output,`${chunk}-birth`),path.join(output,`${chunk}-inputs.json`),path.join(output,`${chunk}-predictions.json`)]);const summary=JSON.parse(child.stdout);assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.peak_active_modules<=2);summaries.push(summary);const part=JSON.parse(await fs.readFile(path.join(output,`${chunk}-predictions.json`)));assert.deepEqual(part.map(r=>r.id),inputs.slice(start,start+4).map(r=>r.id));rows.push(...part);}
  assert.deepEqual(await fs.readFile(number),numberBytes);await write(`${name}-predictions.json`,rows);
  results.push({index,policy,training_call:trained.call.order,training:trainingReport,teacher_frame_fit:{correct:fitCorrect,total:training.examples.length,work:fitWork,scope:'Greedy on taught frames, NOT own-state rollout success'},pack_call:packed.call.order,pack:packReport,predictions:rows.length,answers:rows.filter(r=>r.answer!==null).length,arithmetic_evaluations:rows.reduce((n,r)=>n+r.arithmetic_evaluations,0),language_work:summaries.reduce((n,r)=>n+r.language_work,0),loads:summaries.reduce((n,r)=>n+r.loads,0),peak_active_modules:Math.max(...summaries.map(r=>r.peak_active_modules)),peak_active_payload_bytes:Math.max(...summaries.map(r=>r.peak_active_payload_bytes)),number_sha256:sha(numberBytes),number_digits:numberBytes.length,predictions_sha256:sha(await fs.readFile(path.join(output,`${name}-predictions.json`)))});
}
assert.deepEqual(await fs.readFile(parent),parentBytes);
await write('predictions-ready.json',{schema:'axon-math-actor-study/v1',results,library:library.map(({file,...r})=>r),calls,development_answers_accessed:false,reserved_answers_accessed:false,future_teacher_states_used:false,untrained_probe_cases:4,untrained_full_development_evaluated:false,m1_complete:false,m2_permitted:false,process_rss_bytes:null,energy_joules:null,scope:'Autoregressive own-result controller on previously consumed full development. State/action shape and four elementary semantics provided; trace supervision allowed train only. No semantic, unit, discovery or full M1 certificate.'});
console.log(JSON.stringify({results:results.map(r=>({index:r.index,taught_frames:r.teacher_frame_fit.total,frame_fit:r.teacher_frame_fit.correct,answers:r.answers,calculations:r.arithmetic_evaluations,bytes:r.pack.encoded_bytes})),calls:calls.length,m1_complete:false}));
