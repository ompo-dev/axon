import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify,isDeepStrictEqual} from 'node:util';
import assert from 'node:assert/strict';
import {sequenceAnnotation,metrics} from './massive-grammar.mjs';

const [reference,executable,corpus,outputPath,protocolPath='evals/protocols/m1-compression-probe-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/probe-sequence-compression.mjs <frozen-enhanced-run> <candidate> <corpus> <new-output> [protocol]');
const output=path.resolve(outputPath);await fs.mkdir(output);
const sha=b=>createHash('sha256').update(b).digest('hex');
const json=async p=>JSON.parse(await fs.readFile(p));
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const protocolBytes=await fs.readFile(protocolPath),protocol=JSON.parse(protocolBytes);
assert.equal(path.resolve(reference),path.resolve(protocol.reference_run));
const old=await json(path.join(reference,'audit.json')),oldFreeze=await json(path.join(reference,'freeze.json'));
assert.equal(old.selected,protocol.reference_policy);
const trainingBytes=await fs.readFile(path.join(reference,`policy-${old.selected}-train.json`));
const training=JSON.parse(trainingBytes);
const raw=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
assert.equal(sha(raw),protocol.source_sha256);assert.equal(sha(raw),oldFreeze.corpus_sha256);
const provenance=await json(path.join(corpus,'provenance.json')),reserved=new Set(provenance.reserved_workers);
const allowed=raw.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse)
  .filter(r=>!reserved.has(r.worker_id)&&['train','dev'].includes(r.partition));
const train=allowed.filter(r=>r.partition==='train'),dev=allowed.filter(r=>r.partition==='dev');
assert.deepEqual(train.map(r=>String(r.id)),oldFreeze.train_ids);assert.equal(dev.length,1504);
assert.deepEqual(training.examples,train.map(r=>({text:r.utt,intent:r.intent,tags:sequenceAnnotation(r).tags})));
const sampled=new Set(train.map(r=>({id:r.id,hash:sha('m1-compression-probe:'+r.id)}))
  .sort((a,b)=>a.hash.localeCompare(b.hash)).slice(0,1000).map(r=>r.id));
const groups={development:dev,training_fit_sample:train.filter(r=>sampled.has(r.id))};
const labels=[],inputs=[];
for(const [group,rows] of Object.entries(groups))for(const r of rows) {
  const id=`${group}:${r.id}`;inputs.push({id,text:r.utt});labels.push({id,group,target:sequenceAnnotation(r).target});
}
const binary=path.join(output,'candidate.exe');await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
await write('probe.json',{training,inputs});await write('host-labels.json',labels);
const sources={};
for(const file of ['src/structural_sequence.rs','src/structural_sequence_compact.rs','src/structural_sequence_lab.rs','src/curriculum.rs','scripts/probe-sequence-compression.mjs','scripts/massive-grammar.mjs','evals/protocols/m1-completion-v1.json',protocolPath]) {
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);const dest=path.join(output,'source',file);
  await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});
}
await write('freeze.json',{created_at:new Date().toISOString(),sources,binary_sha256:sha(await fs.readFile(binary)),corpus_sha256:sha(raw),protocol_sha256:sha(protocolBytes),
  reference,reference_training_sha256:sha(trainingBytes),probe_sha256:sha(await fs.readFile(path.join(output,'probe.json'))),host_labels_sha256:sha(await fs.readFile(path.join(output,'host-labels.json'))),
  scope:'Frozen before the core call. Diagnostic compares a nonportable quantized scratch model to the exact deployment-pruned control. Host labels never enter the prediction inputs. Logical isolation only.'});
const started=performance.now();let child;
try {child=await promisify(execFile)(binary,['sequence-compression-probe',path.join(output,'probe.json'),path.join(output,'results')],
  {cwd:output,windowsHide:true,timeout:600000,maxBuffer:64000000,env:{...process.env,AXON_LAB_TELEMETRY:'1'}});}
catch(error){await fs.writeFile(path.join(output,'failed-stdout.log'),error.stdout??'',{flag:'wx'});await fs.writeFile(path.join(output,'failed-stderr.log'),error.stderr??'',{flag:'wx'});throw error;}
await fs.writeFile(path.join(output,'stdout.jsonl'),child.stdout,{flag:'wx'});await fs.writeFile(path.join(output,'stderr.log'),child.stderr,{flag:'wx'});
const events=child.stdout.trim().split(/\r?\n/u).map(JSON.parse),summary=events.pop();events.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
assert.equal(summary.genome_published,false);assert.equal(summary.teacher_queries_at_prediction,0);
const predictions=await json(path.join(output,'results/predictions.json'));assert.equal(predictions.length,labels.length);
const previous=await json(path.join(reference,`policy-${old.selected}-predictions.json`));
const report={schema:'axon-sequence-compression-diagnostic/v1',summary,process_ms:performance.now()-started,groups:{},m1_complete:false,m2_started:false,production_promoted:false};
for(const [group] of Object.entries(groups)) {
  const indices=labels.flatMap((r,i)=>r.group===group?[i]:[]),truth=indices.map(i=>labels[i]);
  const before=indices.map(i=>predictions[i].unpruned),after=indices.map(i=>predictions[i].pruned);
  for(const i of indices)assert.equal(predictions[i].id,labels[i].id);
  let lost=0,gained=0;
  for(let i=0;i<truth.length;i++) {
    const a=isDeepStrictEqual(before[i].prediction,truth[i].target),b=isDeepStrictEqual(after[i].prediction,truth[i].target);
    lost+=Number(a&&!b);gained+=Number(!a&&b);
  }
  report.groups[group]={unpruned:metrics(before,truth),pruned:metrics(after,truth),pruning_lost:lost,pruning_gained:gained};
  if(group==='development')for(let i=0;i<after.length;i++) {
    assert.deepEqual(after[i].prediction,previous[i].prediction);assert.equal(after[i].error,previous[i].error);
  }
}
await write('audit.json',report);
console.log(JSON.stringify({...report,groups:Object.fromEntries(Object.entries(report.groups).map(([k,v])=>[k,{before:v.unpruned.exact,after:v.pruned.exact,total:v.pruned.total,lost:v.pruning_lost,gained:v.pruning_gained}]))}));
