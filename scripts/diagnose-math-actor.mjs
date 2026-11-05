import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha} from './gsm8k-source.mjs';
import {focusExample} from './math-actor-source.mjs';
import {target,compare,summarize,canonicalInitialFrame} from './math-actor-diagnostic.mjs';
import {agrees} from './math-program-source.mjs';

const [candidate,out,reuse]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/diagnose-math-actor.mjs <lab.exe> <new-output> [previous-diagnostic]');
const output=path.resolve(out);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const write=(file,value)=>fs.writeFile(path.join(output,file),JSON.stringify(value,null,2),{flag:'wx'});
const protocolFile='evals/protocols/m1-math-actor-diagnostic-v1.json',protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);
const binary=path.join(output,'candidate.exe');await fs.copyFile(candidate,binary,fs.constants.COPYFILE_EXCL);
const episodeBytes=await fs.readFile(protocol.episodes),episodes=JSON.parse(episodeBytes);assert.equal(episodes.length,629);
const sources={};for(const file of [protocolFile,'scripts/diagnose-math-actor.mjs','scripts/math-actor-diagnostic.mjs','scripts/math-actor-source.mjs','scripts/math-program-source.mjs','scripts/gsm8k-source.mjs','src/structural_math_actor_lab.rs','src/structural_math_lab.rs','src/curriculum.rs','src/structural_lexical_encoder.rs','src/structural_sequence_search.rs']){
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);const dest=path.join(output,'source',file);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});
}
await fs.writeFile(path.join(output,'episodes.json'),episodeBytes,{flag:'wx'});
const models=[];
for(const study of protocol.studies){
  const report=JSON.parse(await fs.readFile(path.join(study,'predictions-ready.json')));
  for(const r of report.results){
    const number=path.resolve(study,`policy-${r.index}-combined/structural.axnum`),bytes=await fs.readFile(number);assert.equal(sha(bytes),r.number_sha256);
    const index=models.length,copy=path.join(output,`model-${index}.axnum`);await fs.writeFile(copy,bytes,{flag:'wx'});
    models.push({index,study,policy_index:r.index,policy:r.policy,number:copy,number_sha256:r.number_sha256,greedy_fit:r.teacher_frame_fit,study_sha256:sha(await fs.readFile(path.join(study,'predictions-ready.json')))});
  }
}
assert.equal(models.length,6);
const previousCalls=new Map();
if(reuse){
  const previous=JSON.parse(await fs.readFile(path.join(reuse,'freeze.json')));
  assert.equal(previous.protocol_sha256,sha(protocolBytes));assert.equal(previous.binary_sha256,sha(await fs.readFile(binary)));assert.equal(previous.episodes_sha256,sha(episodeBytes));assert.deepEqual(previous.models.map(m=>m.number_sha256),models.map(m=>m.number_sha256));
  for(const file of (await fs.readdir(reuse)).filter(f=>/^call-\d+\.json$/u.test(f))){const bytes=await fs.readFile(path.join(reuse,file)),call=JSON.parse(bytes);assert.equal(call.exit_code,0);assert(!previousCalls.has(call.name));previousCalls.set(call.name,{...call,record_sha256:sha(bytes)});}
}
await write('freeze.json',{schema:protocol.schema,protocol_sha256:sha(protocolBytes),binary_sha256:sha(await fs.readFile(binary)),episodes_sha256:sha(episodeBytes),sources,models,reused_run:reuse??null,corpus_answers_read:false,no_fitting:true,m1_complete:false});
const calls=[],results=[];
async function core(name,args){
  const started=performance.now();let stdout='',stderr='',code=0;
  const previous=previousCalls.get(name);
  if(previous){
    assert.equal(previous.args[0],args[0]);assert.equal(sha(await fs.readFile(previous.args[1])),sha(await fs.readFile(args[1])));assert.deepEqual(JSON.parse(await fs.readFile(previous.args[3])),JSON.parse(await fs.readFile(args[3])));
    stdout=await fs.readFile(path.join(reuse,`${name}.stdout.log`),'utf8');stderr=await fs.readFile(path.join(reuse,`${name}.stderr.log`),'utf8');assert.equal(stderr,'');await fs.copyFile(previous.args[4],args[4],fs.constants.COPYFILE_EXCL);
  }else try{({stdout,stderr}=await promisify(execFile)(binary,args,{windowsHide:true,timeout:300000,maxBuffer:4000000}));}
  catch(e){stdout=e.stdout??'';stderr=e.stderr??'';code=e.code??'spawn';}
  await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
  const call={order:calls.length,name,args,exit_code:code,...(previous?{reused_run:reuse,original_call_sha256:previous.record_sha256,original_whole_process_ms:previous.original_whole_process_ms??previous.whole_process_ms}:{}),whole_process_ms:previous?null:performance.now()-started};calls.push(call);await write(`call-${call.order}.json`,call);assert.equal(code,0,stderr);return JSON.parse(stdout);
}
for(const model of models){
  const name=`model-${model.index}`,focused=model.policy.layout==='focused',page=model.policy.numbers_only?'free':'full';
  const taught=episodes.flatMap(e=>e[page].examples.map((original,step)=>({id:`${e.id}:${step}`,episode:e.id,step,example:focused?focusExample(original,step):original})));
  assert.equal(taught.length,2251);
  const teacher=[];
  for(let start=0;start<taught.length;start+=32){
    const chunk=`${name}-frames-${start}`,input=path.join(output,`${chunk}-inputs.json`),receipt=path.join(output,`${chunk}-predictions.json`);
    await write(`${chunk}-inputs.json`,taught.slice(start,start+32).map(r=>({id:r.id,text:r.example.text})));
    const summary=await core(chunk,[focused?'math-actor-focused-frames-batch':'math-actor-frames-batch',model.number,path.join(output,`${chunk}-birth`),input,receipt]);
    assert(summary.seed_unchanged&&summary.supplied_states&&summary.arithmetic_evaluations===0&&summary.additional_teacher_queries===0&&summary.peak_active_modules<=1);
    const rows=JSON.parse(await fs.readFile(receipt));assert.deepEqual(rows.map(r=>r.id),taught.slice(start,start+32).map(r=>r.id));
    for(const [i,row]of rows.entries()){
      const item=taught[start+i];assert.equal(row.frame,item.example.text);
      teacher.push({...row,episode:item.episode,step:item.step,expected:target(item.example),comparison:compare(row.prediction,target(item.example))});
    }
  }
  await write(`${name}-teacher.json`,teacher);
  const inputs=episodes.map(e=>({id:e.id,text:e[page].examples[0].text.split(' ').slice(1,-2).join(' ')})),own=[];
  for(let start=0;start<inputs.length;start+=4){
    const chunk=`${name}-own-${start}`,input=path.join(output,`${chunk}-inputs.json`),receipt=path.join(output,`${chunk}-predictions.json`);
    await write(`${chunk}-inputs.json`,inputs.slice(start,start+4));
    const summary=await core(chunk,[focused?'math-actor-focused-batch':'math-actor-batch',model.number,path.join(output,`${chunk}-birth`),input,receipt]);
    assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.peak_active_modules<=2);
    const rows=JSON.parse(await fs.readFile(receipt));assert.deepEqual(rows.map(r=>r.id),inputs.slice(start,start+4).map(r=>r.id));own.push(...rows);
  }
  const episodeRows=episodes.map((e,i)=>{
    const oracle=teacher.filter(r=>r.episode===e.id),row=own[i],first=oracle[0];
    const firstMatches=row.trace[0].frame===first.frame;
    if(firstMatches){assert.deepEqual(row.trace[0].prediction??null,first.prediction);assert.equal(row.trace[0].work,first.work);}else{assert(model.policy.numbers_only,'Unexpected full question frame mismatch');assert.equal(row.trace[0].frame,canonicalInitialFrame(first.frame,focused));}
    return {id:e.id,first_teacher_frame_matches:firstMatches,teacher_steps:oracle.length,teacher_all_semantic:oracle.every(r=>r.comparison.semantic),teacher_all_exact:oracle.every(r=>r.comparison.exact),first_teacher_disagreement:oracle.find(r=>!r.comparison.semantic)?.step??null,expected_final:e[page].final,own_answer:row.answer,own_correct:agrees(row.answer,e[page].final),own_error:row.error,own_calculations:row.arithmetic_evaluations};
  });
  assert.equal(sha(await fs.readFile(model.number)),model.number_sha256);
  await write(`${name}-own.json`,own);await write(`${name}-episode-audit.json`,episodeRows);
  const group=key=>Object.fromEntries([...new Set(teacher.map(key))].map(k=>[k,summarize(teacher.filter(r=>key(r)===k).map(r=>r.comparison))]));
  const result={index:model.index,policy:model.policy,number_sha256:model.number_sha256,teacher:summarize(teacher.map(r=>r.comparison)),by_step:group(r=>r.step),by_operation:group(r=>r.expected[1]),teacher_complete_semantic:episodeRows.filter(r=>r.teacher_all_semantic).length,teacher_complete_exact:episodeRows.filter(r=>r.teacher_all_exact).length,first_frame_mismatches:episodeRows.filter(r=>!r.first_teacher_frame_matches).length,own_correct:episodeRows.filter(r=>r.own_correct).length,own_refused:own.filter(r=>r.answer===null).length,own_total:own.length,own_failed_despite_teacher_complete:episodeRows.filter(r=>r.teacher_all_semantic&&!r.own_correct).length,own_failed_despite_teacher_complete_and_same_first_frame:episodeRows.filter(r=>r.first_teacher_frame_matches&&r.teacher_all_semantic&&!r.own_correct).length,teacher_work:teacher.reduce((n,r)=>n+r.work,0),own_work:own.reduce((n,r)=>n+r.language_work,0)};
  results.push(result);await write(`${name}-audit.json`,result);console.log(JSON.stringify(result));
}
await write('audit.json',{schema:protocol.schema,results,calls,corpus_answers_read:false,no_fitting:true,teacher_states_are_not_autonomous_success:true,m1_complete:false,m2_permitted:false,scope:'Matched-decoder diagnosis on already-taught train episodes only, not generalization or causal isolation of exposure bias. Numerical value equality and semantic action agreement separate. Duplicate-value token provenance remains unavailable.'});
console.log(JSON.stringify({models:results.length,calls:calls.length,teacher_decisions:results.reduce((n,r)=>n+r.teacher.total,0),own_episodes:results.reduce((n,r)=>n+r.own_total,0),m1_complete:false}));
