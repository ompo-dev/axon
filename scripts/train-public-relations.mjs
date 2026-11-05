import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';

const [parentFile,executable,corpus,outputPath,protocolPath='evals/protocols/m1-public-relations-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/train-public-relations.mjs <parent-number> <candidate> <corpus> <new-output>');
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,value)=>fs.writeFile(path.join(output,name),JSON.stringify(value,null,2),{flag:'wx'});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const protocolBytes=await fs.readFile(protocolPath);
const protocol=JSON.parse(protocolBytes);
const provenance=JSON.parse(await fs.readFile(path.join(corpus,'provenance.json'),'utf8'));
assert.equal(provenance.revision,protocol.revision);
const splits={};const hashes={};
for(const [split,count] of [['train',8399],['val',646],['test',1758]]) {
  const raw=await fs.readFile(path.join(corpus,`${split}.csv`));hashes[split]=sha(raw);
  assert.equal(hashes[split],provenance.files.find(f=>f.name===`${split}.csv`).sha256);
  splits[split]=parse(raw,{columns:true,bom:true});assert.equal(splits[split].length,count);
  assert(splits[split].every(r=>['0','1','2'].includes(r.label)));
}
const binary=path.join(output,'candidate.exe'),parent=path.join(output,'parent.axnum');
await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
await fs.copyFile(parentFile,parent,fs.constants.COPYFILE_EXCL);
const sources={};
if(protocol.training_command==='relation-network-train')for(const file of ['Cargo.toml','Cargo.lock','src/structural_sequence.rs','src/structural_sequence_factorized.rs','src/structural_relation_network.rs','src/structural_relation_network_lab.rs','src/structural_dependency.rs','src/demand.rs','src/genome.rs','src/structural_lab.rs','src/curriculum.rs','scripts/train-public-relations.mjs','evals/protocols/m1-completion-v1.json',protocolPath]) {
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);const target=path.join(output,'source',file);
  await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,bytes,{flag:'wx'});
}
await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),parent_sha256:sha(await fs.readFile(parent)),
  protocol_sha256:sha(protocolBytes),driver_sha256:sha(await fs.readFile(process.argv[1])),corpus:hashes,policies:protocol.policies,sources,
  scope:'Frozen before execution. Only train labels passed to learner; inference payloads contain no labels. Logical separation, not OS sandbox.'});
async function core(name,args) {
  const start=performance.now();let child;
  try {child=await promisify(execFile)(binary,args,{cwd:output,windowsHide:true,timeout:300000,maxBuffer:32000000,env:{...process.env,AXON_LAB_TELEMETRY:'1'}});}
  catch(e) {await fs.writeFile(path.join(output,`${name}-failed.stdout.log`),e.stdout??'',{flag:'wx'});await fs.writeFile(path.join(output,`${name}-failed.stderr.log`),e.stderr??'',{flag:'wx'});await write(`${name}-failure.json`,{message:e.message,code:e.code});throw e;}
  await fs.writeFile(path.join(output,`${name}.stdout.jsonl`),child.stdout,{flag:'wx'});
  await fs.writeFile(path.join(output,`${name}.stderr.log`),child.stderr,{flag:'wx'});
  const lines=child.stdout.trim().split(/\r?\n/).map(s=>JSON.parse(s));const summary=lines.pop();
  for(let i=0;i<lines.length;i++)assert.equal(lines[i].coreSeq,i+1);
  return {summary,whole_process_ms:performance.now()-start,core_events:lines.length};
}
function metrics(rows,labels) {
  assert.equal(rows.length,labels.length);
  let correct=0,answered=0;
  const matrix=Array.from({length:3},()=>[0,0,0,0]);
  rows.forEach((row,i)=>{const prediction=row.prediction;assert(prediction===null||[0,1,2].includes(prediction));
    matrix[labels[i]][prediction??3]++;answered+=Number(prediction!==null);correct+=Number(prediction!==null&&prediction===labels[i]);});
  const f1=[0,1,2].map(k=>{const tp=matrix[k][k],fp=matrix.reduce((n,r)=>n+r[k],0)-tp,fn=matrix[k].reduce((a,b)=>a+b,0)-tp;return 2*tp/(2*tp+fp+fn||1);});
  return {total:rows.length,answered,correct,wrong:answered-correct,abstained:rows.length-answered,accuracy:correct/(rows.length||1),answered_accuracy:answered?correct/answered:null,macro_f1:f1.reduce((a,b)=>a+b,0)/3,matrix,columns:['Contradiction','Entailment','Neutral','Abstain']};
}
async function evaluate(name,number,split) {
  const inputs=splits[split].map((r,i)=>({id:`${split}-${i}`,premise:r.premise,hypothesis:r.hypothesis}));
  await write(`${name}-inputs.json`,inputs);
  const run=await core(name,[protocol.batch_command??'relation-batch',number,path.join(output,`${name}-birth`),path.join(output,`${name}-inputs.json`),path.join(output,`${name}-predictions.json`)]);
  assert(run.summary.seed_unchanged&&run.summary.teacher_queries===0);
  const rows=JSON.parse(await fs.readFile(path.join(output,`${name}-predictions.json`)));
  rows.forEach((r,i)=>assert.equal(r.id,inputs[i].id));
  const report={...run,metrics:metrics(rows,splits[split].map(r=>Number(r.label)))};
  await write(`${name}-score.json`,report);return {report,rows};
}
assert([undefined,'all','reviewer-clean'].includes(protocol.training_quality),'Unknown training quality policy');
const reviewLabels={Contradiction:0,Entailment:1,Neutral:2};
const trainingSource=protocol.training_quality==='reviewer-clean'?splits.train.filter(r=>r.low_quality==='N'&&reviewLabels[r.reviwers_output]===Number(r.label)):splits.train;
const training=trainingSource.map(r=>({premise:r.premise,hypothesis:r.hypothesis,label:Number(r.label)}));
await write('training-source.json',{original_train:splits.train.length,selected:trainingSource.length,excluded_quality:splits.train.length-trainingSource.length,policy:protocol.training_quality??'all',scope:'Training selection only; every original validation/test pair remains in evaluation denominators.'});
const axioms=[];
assert([undefined,'statement-hash'].includes(protocol.identity_sampling),'Unknown identity sampling policy');
if(protocol.identity_examples&&protocol.identity_sampling==='statement-hash') {
  assert.equal(protocol.training_command,'relation-network-train');
  const statements=new Map(trainingSource.flatMap(r=>[r.premise,r.hypothesis]).map(s=>[normalize(s),s]));
  const existing=new Set(trainingSource.filter(r=>normalize(r.premise)===normalize(r.hypothesis)).map(r=>normalize(r.premise)));
  const candidates=[...statements].filter(([k])=>!existing.has(k)).sort((a,b)=>sha(a[0]).localeCompare(sha(b[0])));
  for(const [key,statement]of candidates.slice(0,protocol.identity_examples)) {
    training.push({premise:statement,hypothesis:statement,label:1});
    axioms.push({source_statement_sha256:sha(statement),canonical_statement_sha256:sha(key),source_partition:'train',label:1,kind:'explicit supervised reflexivity axiom; not an external human annotation or autonomous discovery'});
  }
  assert.equal(axioms.length,protocol.identity_examples);assert(training.length<=10000);
} else if(protocol.identity_examples) {
  const lengths=new Set();
  for(const statement of [...new Set(trainingSource.flatMap(r=>[r.premise,r.hypothesis]))].sort()) {
    const length=statement.trim().split(/\s+/).length;
    if(length<2||length>31||lengths.has(length))continue;
    lengths.add(length);training.push({premise:statement,hypothesis:statement,label:1});
    axioms.push({source_statement_sha256:sha(statement),length,label:1,kind:'supervised NLI reflexivity axiom over public train text'});
    if(axioms.length>=protocol.identity_examples)break;
  }
}
await write('supervised-axioms.json',axioms);
const policies=[];
for(let i=0;i<protocol.policies.length;i++) {
  const name=`policy-${i}`;await write(`${name}-train.json`,{budget:protocol.policies[i],examples:training});
  const trained=await core(`${name}-training`,[protocol.training_command??'relation-train',parent,path.join(output,`${name}-parent-birth`),path.join(output,`${name}-train.json`),path.join(output,name)]);
  assert(trained.summary.parent_unchanged);
  const number=path.join(output,name,'structural.axnum');
  const validation=await evaluate(`${name}-validation`,number,'val');
  policies.push({index:i,number,training:trained,validation:validation.report,number_sha256:sha(await fs.readFile(number))});
  console.log(JSON.stringify({policy:i,training:trained.summary.statistics??trained.summary,validation:validation.report.metrics}));
}
const complexity=p=>p.training.summary.statistics?.selected??p.training.summary.page_bytes;
const selected=[...policies].sort((a,b)=>b.validation.metrics.correct-a.validation.metrics.correct||complexity(a)-complexity(b)||a.index-b.index)[0];
await write('selection.json',{selected:selected.index,policies,scope:'Validation only. Test score has not been computed by this driver at selection time.'});
const evaluated=await evaluate('selected-test',selected.number,'test');
function normalize(s){return s.toLowerCase().trim().replace(/\s+/g,' ');}
const trainStatements=new Set(splits.train.flatMap(r=>[normalize(r.premise),normalize(r.hypothesis)]));
const reviewed={Contradiction:0,Entailment:1,Neutral:2};
const strata={};
for(const [name,include,label] of [
  ['reviewed',r=>Object.hasOwn(reviewed,r.reviwers_output),r=>reviewed[r.reviwers_output]],
  ['reviewed_clean',r=>Object.hasOwn(reviewed,r.reviwers_output)&&r.low_quality==='N',r=>reviewed[r.reviwers_output]],
  ['statement_disjoint',r=>!trainStatements.has(normalize(r.premise))&&!trainStatements.has(normalize(r.hypothesis)),r=>Number(r.label)],
]) {
  const indices=splits.test.map((r,i)=>include(r)?i:null).filter(i=>i!==null);
  strata[name]=metrics(indices.map(i=>evaluated.rows[i]),indices.map(i=>label(splits.test[i])));
}
const report={schema:'axon-public-relations/v1',selected:selected.index,test:evaluated.report,strata,supervised_axioms:axioms.length,representation:protocol.representation??'raw paired utterance trees',batch_command:protocol.batch_command??'relation-batch',
  parent_retained:policies.every(p=>p.training.summary.parent_unchanged),m1_complete:false,
  limits:`Supervised finite structural patterns, not free semantic understanding. ${policies.length} policies; whole process times include birth and I/O. Test consumed previously; statement-disjoint subset is not independent author metadata. RSS, energy and LLM comparison unavailable.`};
await write('audit.json',report);console.log(JSON.stringify(report));
if(protocol.training_command==='relation-network-train') {
  const labelCounts=[0,0,0],memory=new Map();
  const key=r=>JSON.stringify([normalize(r.premise),normalize(r.hypothesis)]);
  for(const r of training){labelCounts[Number(r.label)]++;const values=memory.get(key(r))??new Set();values.add(Number(r.label));memory.set(key(r),values);}
  const common=[0,1,2].sort((a,b)=>labelCounts[b]-labelCounts[a]||a-b)[0];
  const baselines={};
  for(const split of ['val','test'])baselines[split]={
    constant:metrics(splits[split].map(()=>({prediction:common})),splits[split].map(r=>Number(r.label))),
    exact_pair_retrieval:metrics(splits[split].map(r=>({prediction:memory.get(key(r))?.size===1?[...memory.get(key(r))][0]:null})),splits[split].map(r=>Number(r.label))),
  };
  await write('baselines.json',{...baselines,scope:'Constant and literal memory from allowed train only. Computed after policy selection; test remains previously consumed diagnostic. Hypothesis-only neural ablations are separate frozen policies.'});
}
