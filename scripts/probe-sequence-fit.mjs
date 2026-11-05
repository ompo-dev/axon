import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {sequenceAnnotation,metrics} from './massive-grammar.mjs';

const [run,corpus,outputPath,policyIndex]=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/probe-sequence-fit.mjs <frozen-sequence-run> <massive-corpus> <new-output> [diagnostic-policy-index]');
const output=path.resolve(outputPath);await fs.mkdir(output);
const json=async file=>JSON.parse(await fs.readFile(file));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const audit=await json(path.join(run,'audit.json')),freeze=await json(path.join(run,'freeze.json'));
assert(['sequence-v1','compact-sequence-v1','enhanced-sequence-v1','factorized-sequence-v1','joint-sequence-v1','lexical-encoder-v1','segment-sequence-v1'].includes(audit.learner));
if(policyIndex!==undefined)assert(/^(0|[1-9][0-9]*)$/u.test(policyIndex),'Invalid diagnostic policy index');
const policy=audit.policies.find(p=>p.index===(policyIndex===undefined?audit.selected:Number(policyIndex)));assert(policy);
const raw=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
assert.equal(sha(raw),freeze.corpus_sha256);
const provenance=await json(path.join(corpus,'provenance.json')),reserved=new Set(provenance.reserved_workers);
const rows=raw.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse)
  .filter(r=>r.partition==='train'&&!reserved.has(r.worker_id));
assert.deepEqual(rows.map(r=>String(r.id)),freeze.train_ids);
const truth=rows.map(r=>({id:String(r.id),target:sequenceAnnotation(r).target}));
await write('inputs.json',rows.map(r=>({id:String(r.id),text:r.utt})));
await write('host-truth.json',truth);
const binary=path.join(output,'candidate.exe'),number=path.join(output,'isolated.axnum');
await fs.copyFile(path.join(run,'candidate.exe'),binary,fs.constants.COPYFILE_EXCL);
await fs.copyFile(path.join(run,`policy-${policy.index}/structural.axnum`),number,fs.constants.COPYFILE_EXCL);
assert.equal(sha(await fs.readFile(binary)),freeze.binary_sha256);
assert.equal(sha(await fs.readFile(number)),policy.number_sha256);
await write('freeze.json',{run,selected:policy.index,study_selected:audit.selected,diagnostic_override:policyIndex!==undefined,binary_sha256:freeze.binary_sha256,number_sha256:policy.number_sha256,
  probe_sha256:sha(await fs.readFile(process.argv[1])),parser_sha256:sha(await fs.readFile(new URL('./massive-grammar.mjs',import.meta.url))),
  inputs_sha256:sha(await fs.readFile(path.join(output,'inputs.json'))),truth_sha256:sha(await fs.readFile(path.join(output,'host-truth.json'))),
  scope:'Training-set fit diagnostic only, not validation, new learning, an independent reserve or proof of generalization. No reserved/test rows or future labels enter inference.'});
const segment=audit.learner==='segment-sequence-v1',encoder=audit.learner==='lexical-encoder-v1',joint=audit.learner==='joint-sequence-v1';
const command=segment?'segment-batch':encoder?'encoder-batch':joint?'joint-batch':audit.learner==='factorized-sequence-v1'?'factorized-batch':audit.learner==='enhanced-sequence-v1'?'enhanced-batch':audit.learner==='compact-sequence-v1'?'compact-batch':'sequence-batch';
let child;
if(joint||encoder||segment) {
  const allInputs=await json(path.join(output,'inputs.json')),allPredictions=[],summaries=[];
  const chunkSize=segment?16:encoder?((policy.training.summary.budget.byte_convolution||policy.training.summary.budget.ordered_context)?32:128):512;
  for(let start=0;start<allInputs.length;start+=chunkSize) {
    const name=`chunk-${start}`,input=path.join(output,`${name}-inputs.json`),receipt=path.join(output,`${name}-predictions.json`);
    await write(`${name}-inputs.json`,allInputs.slice(start,start+chunkSize));
    const part=await promisify(execFile)(binary,[command,number,path.join(output,`${name}-birth`),input,receipt],{cwd:output,windowsHide:true,timeout:300000,maxBuffer:1000000});
    await fs.writeFile(path.join(output,`${name}.stdout.log`),part.stdout,{flag:'wx'});
    const summary=JSON.parse(part.stdout.trim());assert(summary.seed_unchanged&&summary.teacher_queries===0);summaries.push(summary);
    allPredictions.push(...await json(receipt));
  }
  await write('predictions.json',allPredictions);await write('chunk-summaries.json',summaries);
  child={stdout:JSON.stringify({seed_unchanged:true,teacher_queries:0,total:allPredictions.length,chunks:summaries.length,work:summaries.reduce((n,s)=>n+s.work,0)}),stderr:''};
} else child=await promisify(execFile)(binary,[command,number,path.join(output,'birth'),path.join(output,'inputs.json'),path.join(output,'predictions.json')],
  {cwd:output,windowsHide:true,timeout:300000,maxBuffer:64000000,env:{...process.env,AXON_LAB_TELEMETRY:'1'}});
await fs.writeFile(path.join(output,'stdout.jsonl'),child.stdout,{flag:'wx'});
await fs.writeFile(path.join(output,'stderr.log'),child.stderr,{flag:'wx'});
const events=child.stdout.trim().split(/\r?\n/u).map(JSON.parse),summary=events.pop();
events.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
assert(summary.seed_unchanged&&summary.teacher_queries===0);
const predictions=await json(path.join(output,'predictions.json'));
const report={schema:'axon-sequence-training-fit/v1',run,selected:policy.index,training_fit:metrics(predictions,truth),development:policy.score,
  summary,core_events:events.length,m1_complete:false,production_promoted:false,
  scope:'Scores recorded real-core outputs on previously taught rows. A weak fit diagnoses acquisition/compaction limits; even perfect training fit would not establish semantic generalization or M1 approval.'};
await write('audit.json',report);
console.log(JSON.stringify({...report,training_fit:{...report.training_fit,families:undefined},development:{...report.development,families:undefined}}));
