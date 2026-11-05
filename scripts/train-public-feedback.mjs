import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {sequenceAnnotation,metrics} from './massive-grammar.mjs';

const [parentFile,executable,corpus,outputPath,protocolPath='evals/protocols/m1-public-feedback-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/train-public-feedback.mjs <parent-number> <candidate> <corpus> <new-output> [protocol]');
const output=path.resolve(outputPath);await fs.mkdir(output);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const json=async file=>JSON.parse(await fs.readFile(file));
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const protocolBytes=await fs.readFile(protocolPath),protocol=JSON.parse(protocolBytes);
assert.equal(protocol.schema,'axon-public-feedback-protocol/v1');
assert(['enhanced-sequence-v1','factorized-sequence-v1','lexical-encoder-v1'].includes(protocol.learner));
const lexical=protocol.learner==='lexical-encoder-v1';
const command=lexical?'encoder':protocol.learner==='factorized-sequence-v1'?'factorized':'enhanced';
if(lexical)assert.equal(protocol.chunk_requests,128);
assert.deepEqual(protocol.prefixes,[2500,4000,5500,7000,8526]);
const provenanceBytes=await fs.readFile(path.join(corpus,'provenance.json')),provenance=JSON.parse(provenanceBytes);
const bytes=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
assert.equal(sha(bytes),protocol.source_sha256);assert.equal(sha(bytes),provenance.data_sha256);
const reserved=new Set(provenance.reserved_workers);
const allowed=bytes.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse)
  .filter(r=>!reserved.has(r.worker_id)&&['train','dev'].includes(r.partition));
const train=allowed.filter(r=>r.partition==='train'),dev=allowed.filter(r=>r.partition==='dev');
assert.equal(train.length,8526);assert.equal(dev.length,1504);
const binary=path.join(output,'candidate.exe'),parent=path.join(output,'parent.axnum');
await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
await fs.copyFile(parentFile,parent,fs.constants.COPYFILE_EXCL);
const sources={};
for(const file of ['src/structural_sequence.rs','src/structural_sequence_compact.rs','src/structural_sequence_factorized.rs','src/structural_sequence_lab.rs','src/structural_lab.rs','src/curriculum.rs','src/demand.rs','src/genome.rs','scripts/train-public-feedback.mjs','scripts/massive-grammar.mjs','scripts/audit-public-feedback.mjs','evals/protocols/m1-completion-v1.json',protocolPath]) {
  const raw=await fs.readFile(file);sources[file]=sha(raw);
  const destination=path.join(output,'source',file);await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,raw,{flag:'wx'});
}
if(lexical)for(const file of ['src/structural_lexical_encoder.rs','src/structural_encoder_lab.rs','src/structural_byte_convolution.rs','src/structural_ordered_context.rs','src/structural_optimizer.rs','src/structural_lexical_graph.rs','src/structural_sequence_search.rs','Cargo.toml','Cargo.lock']) {
  const raw=await fs.readFile(file);sources[file]=sha(raw);const destination=path.join(output,'source',file);
  await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,raw,{flag:'wx'});
}
const freeze={schema:'axon-public-feedback-freeze/v1',created_at:new Date().toISOString(),sources,
  protocol_path:protocolPath,protocol_sha256:sha(protocolBytes),binary_sha256:sha(await fs.readFile(binary)),parent_sha256:sha(await fs.readFile(parent)),
  corpus_sha256:sha(bytes),provenance_sha256:sha(provenanceBytes),train_ids:train.map(r=>String(r.id)),development_ids:dev.map(r=>String(r.id)),
  scope:'Frozen before core calls. Host can read public labels; the core sees future-window labels only after both pre-feedback prediction calls. Logical separation, not OS isolation or a new corpus.'};
await write('freeze.json',freeze);
const calls=[];
async function core(name,args) {
  const started=performance.now();let child;
  try {child=await promisify(execFile)(binary,args,{cwd:output,windowsHide:true,timeout:600000,maxBuffer:64000000,env:{...process.env,AXON_LAB_TELEMETRY:'1'}});}
  catch(error) {await fs.writeFile(path.join(output,`${name}-failed.stdout.log`),error.stdout??'',{flag:'wx'});await fs.writeFile(path.join(output,`${name}-failed.stderr.log`),error.stderr??'',{flag:'wx'});throw error;}
  await fs.writeFile(path.join(output,`${name}.stdout.jsonl`),child.stdout,{flag:'wx'});
  await fs.writeFile(path.join(output,`${name}.stderr.log`),child.stderr,{flag:'wx'});
  const events=child.stdout.trim().split(/\r?\n/u).map(JSON.parse),summary=events.pop();
  events.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
  const call={order:calls.length,name,command:args[0],arguments:args.slice(1).map(p=>path.relative(output,p)),summary,
    process_ms:performance.now()-started,core_events:events.length};calls.push(call);await write(`call-${call.order}.json`,call);return call;
}
const input=rows=>rows.map(r=>({id:String(r.id),text:r.utt}));
const truth=rows=>rows.map(r=>({id:String(r.id),target:sequenceAnnotation(r).target}));
async function evaluate(name,number,rows) {
  await write(`${name}-inputs.json`,input(rows));
  const before=sha(await fs.readFile(number));
  if(lexical) {
    const calls=[],predictions=[];
    for(let start=0;start<rows.length;start+=protocol.chunk_requests) {
      const part=`${name}-${start}`;await write(`${part}-inputs.json`,input(rows.slice(start,start+protocol.chunk_requests)));
      const call=await core(part,[`${command}-batch`,number,path.join(output,`${part}-birth`),path.join(output,`${part}-inputs.json`),path.join(output,`${part}-predictions.json`)]);
      assert(call.summary.seed_unchanged&&call.summary.teacher_queries===0&&call.summary.loads===1&&call.summary.active_modules===1);
      calls.push(call.order);predictions.push(...await json(path.join(output,`${part}-predictions.json`)));
    }
    assert.equal(sha(await fs.readFile(number)),before);await write(`${name}-predictions.json`,predictions);
    return {number_sha256:before,calls,score:metrics(predictions,truth(rows))};
  }
  const call=await core(name,[`${command}-batch`,number,path.join(output,`${name}-birth`),path.join(output,`${name}-inputs.json`),path.join(output,`${name}-predictions.json`)]);
  assert(call.summary.seed_unchanged&&call.summary.teacher_queries===0);
  assert.equal(sha(await fs.readFile(number)),before);
  return {number_sha256:before,call:call.order,score:metrics(await json(path.join(output,`${name}-predictions.json`)),truth(rows))};
}
let current=parent,ancestor;const stages=[],numbers=new Set();
for(const [index,prefix] of protocol.prefixes.entries()) {
  const before=index?protocol.prefixes[index-1]:0;
  const stage={index,prefix,parent_sha256:sha(await fs.readFile(current))};
  if(index) {
    const window=train.slice(before,prefix);
    stage.before=await evaluate(`stage-${index}-before`,current,window);
    stage.ancestor=await evaluate(`stage-${index}-ancestor`,ancestor,window);
  }
  const examples=train.slice(0,prefix).map(r=>({text:r.utt,intent:r.intent,tags:sequenceAnnotation(r).tags}));
  await write(`stage-${index}-train.json`,{budget:protocol.budget,examples});
  const trained=await core(`stage-${index}-training`,[`${command}-${index?'refine':'train'}`,current,path.join(output,`stage-${index}-parent-birth`),path.join(output,`stage-${index}-train.json`),path.join(output,`stage-${index}`)]);
  assert(trained.summary.parent_unchanged);assert.equal(lexical?!!trained.summary.refinement:trained.summary.refined,index>0);
  assert.equal(sha(await fs.readFile(current)),stage.parent_sha256);
  assert.deepEqual(await fs.readFile(path.join(output,`stage-${index}-parent-birth/structural.axstruct`)),await fs.readFile(path.join(output,`stage-${index}/structural.axstruct`)));
  stage.training=trained.order;current=path.join(output,`stage-${index}/structural.axnum`);
  stage.number_sha256=sha(await fs.readFile(current));assert(!numbers.has(stage.number_sha256));numbers.add(stage.number_sha256);
  ancestor??=current;
  stage.retention=await evaluate(`stage-${index}-retention`,current,train.slice(0,protocol.prefixes[0]));
  stage.development=await evaluate(`stage-${index}-development`,current,dev);
  if(index)stage.after_feedback=await evaluate(`stage-${index}-after`,current,train.slice(before,prefix));
  stages.push(stage);await write(`stage-${index}-score.json`,stage);
  console.log(JSON.stringify({stage:index,prefix,number:stage.number_sha256,before:stage.before?.score.exact,after:stage.after_feedback?.score.exact,retention:stage.retention.score.exact,dev:stage.development.score.exact}));
}
const finalFit=await evaluate('final-training-fit',current,train);
const presentations=protocol.prefixes.reduce((a,b)=>a+b,0)*protocol.budget.epochs;
const report={schema:'axon-public-feedback/v1',learner:protocol.learner,stages,calls,final_training_fit:finalFit,sentence_presentations:presentations,
  reserved_evaluated:false,original_test_evaluated:false,m1_complete:false,m2_permitted:false,production_promoted:false,
  rss_bytes:null,energy_joules:null,scope:protocol.promotion};
await write('audit.json',report);
console.log(JSON.stringify({stages:stages.length,sentence_presentations:presentations,final_fit:finalFit.score.exact,m1_complete:false}));
