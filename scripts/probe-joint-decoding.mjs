import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify,isDeepStrictEqual} from 'node:util';
import {metrics} from './massive-grammar.mjs';
const [executable,controlPath,jointPath,outputPath,protocolPath]=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/probe-joint-decoding.mjs <candidate> <matched-control-run> <joint-run> <new-output> [decoder-protocol]');
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const sha=b=>createHash('sha256').update(b).digest('hex');
const binary=path.join(output,'candidate.exe');await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
await fs.copyFile(process.argv[1],path.join(output,'driver.mjs'),fs.constants.COPYFILE_EXCL);
const read=async(p,f)=>JSON.parse(await fs.readFile(path.join(p,f)));
const protocol=protocolPath?JSON.parse(await fs.readFile(protocolPath)):null;
if(protocol) {
  assert.equal(protocol.schema,'axon-lexical-normalized-decoding/v1');
  assert.equal(path.resolve(protocol.source),path.resolve(controlPath));
  assert.equal(path.resolve(jointPath),path.resolve(controlPath));
  assert.equal(protocol.chunk_requests,16);
  assert.equal(protocol.models.length,2);assert.equal(new Set(protocol.models).size,2);
  assert.equal(protocol.plans.length,7);
  await fs.copyFile(protocolPath,path.join(output,'protocol.json'),fs.constants.COPYFILE_EXCL);
}
const inputs=await read(controlPath,'development-inputs.json'),truth=await read(controlPath,'development-truth.json');
assert.deepEqual(inputs,await read(jointPath,'development-inputs.json'));assert.deepEqual(truth,await read(jointPath,'development-truth.json'));
const rows=truth.map(r=>({...r,family:r.target[1]}));
const plans=[];
if(protocol) {
  const source=await read(controlPath,'audit.json');
  for(const index of protocol.models) {
    const p=source.policies.find(p=>p.index===index);assert(p);
    const number=`lexical-${index}.axnum`;
    await fs.copyFile(path.join(controlPath,`policy-${index}/structural.axnum`),path.join(output,number),fs.constants.COPYFILE_EXCL);
    assert.equal(sha(await fs.readFile(path.join(output,number))),p.number_sha256);
    for(const plan of protocol.plans) {
      assert(['lexical-sum','lexical-mean'].includes(plan.format));
      assert(Number.isSafeInteger(plan.budget.beam)&&plan.budget.beam>=1&&plan.budget.beam<=16);
      assert(Number.isSafeInteger(plan.budget.intents)&&plan.budget.intents>=1&&plan.budget.intents<=8);
      assert.equal(plan.budget.max_work,5_000_000);
      assert(/^[a-z0-9-]+$/u.test(plan.name));
      plans.push({...plan,name:`lexical-${index}-${plan.name}`,number,source_index:index});
    }
  }
} else {
for(const [run,format,ids]of [[controlPath,'enhanced',[0,1]],[jointPath,'joint',[0,1,2,3]]])for(const i of ids) {
  const number=`${format}-${i}.axnum`;await fs.copyFile(path.join(run,`policy-${i}`,'structural.axnum'),path.join(output,number),fs.constants.COPYFILE_EXCL);
  for(const [beam,intents]of [[1,1],[2,1],[4,2]])plans.push({name:`${format}-${i}-${beam}-${intents}`,format,number,budget:{beam,intents,max_work:5000000}});
}
}
await write('inputs.json',inputs);await write('private-truth.json',truth);await write('plans.json',plans);
const sources={};
if(protocol)for(const file of ['src/structural_lexical_encoder.rs','src/structural_sequence_search.rs','src/structural_joint_lab.rs','src/structural_encoder_lab.rs','src/structural_ordered_context.rs','src/structural_byte_convolution.rs','src/structural_sequence.rs','src/structural_sequence_factorized.rs','src/structural.rs','src/demand.rs','src/genome.rs','Cargo.toml','Cargo.lock','scripts/massive-grammar.mjs','evals/protocols/m1-completion-v1.json']) {
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);
  const destination=path.join(output,'source',file);await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,bytes,{flag:'wx'});
}
const hashes={};for(const file of await fs.readdir(output,{withFileTypes:true}))if(file.isFile())hashes[file.name]=sha(await fs.readFile(path.join(output,file.name)));
await write('freeze.json',{hashes,sources,scope:'Decoder-only ablation on already consumed development, all fixed plans before scoring. No training, original test or reserved authors. Each numeric model unchanged.'});
const results=[];
for(const plan of plans) {
  const predictions=[];let chargedWork=0;
  const chunkSize=protocol?.chunk_requests??128;
  for(let start=0;start<inputs.length;start+=chunkSize) {
    const name=`${plan.name}-${start}`,file=path.join(output,`${name}-inputs.json`),receipt=path.join(output,`${name}-predictions.json`);
    await fs.writeFile(file,JSON.stringify({format:plan.format,budget:plan.budget,inputs:inputs.slice(start,start+chunkSize)}),{flag:'wx'});
    const run=await promisify(execFile)(binary,['search-batch',path.join(output,plan.number),path.join(output,`${name}-birth`),file,receipt],{windowsHide:true,timeout:300000,maxBuffer:2000000});
    await fs.writeFile(path.join(output,`${name}.stdout.log`),run.stdout,{flag:'wx'});
    const summary=JSON.parse(run.stdout.trim());assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.loads===1&&summary.active_modules===1);
    chargedWork+=summary.work;predictions.push(...JSON.parse(await fs.readFile(receipt)));
  }
  const source=protocol||plan.format==='enhanced'?controlPath:jointPath,index=Number(plan.number.match(/-(\d+)\.axnum$/)[1]);
  const baseline=await read(source,`policy-${index}-predictions.json`);
  const transitions={both:0,gained:0,lost:0,neither:0,identical:0};
  predictions.forEach((p,i)=>{assert.equal(p.id,truth[i].id);const old=isDeepStrictEqual(baseline[i].prediction,truth[i].target),now=isDeepStrictEqual(p.prediction,truth[i].target);
    transitions[old?(now?'both':'lost'):(now?'gained':'neither')]++;transitions.identical+=Number(isDeepStrictEqual(p.prediction,baseline[i].prediction));});
  if(protocol&&plan.budget.beam===1&&plan.budget.intents===1)predictions.forEach((p,i)=>{
    assert.deepEqual(p.prediction,baseline[i].prediction);assert.equal(p.error,baseline[i].error);assert.equal(p.work,baseline[i].work);
  });
  assert.equal(sha(await fs.readFile(path.join(output,plan.number))),hashes[plan.number]);
  const result={plan,score:metrics(predictions,rows),transitions,charged_work:chargedWork};
  await write(`${plan.name}-predictions.json`,predictions);await write(`${plan.name}-score.json`,result);results.push(result);
  console.log(JSON.stringify({name:plan.name,exact:result.score.exact,abstained:result.score.abstained,transitions}));
}
const selected=protocol?[...results].sort((a,b)=>b.score.exact-a.score.exact||a.charged_work-b.charged_work||a.plan.source_index-b.plan.source_index||plans.indexOf(a.plan)-plans.indexOf(b.plan))[0].plan.name:null;
await write('audit.json',{results,selected,m1_complete:false,reserved_evaluated:false,original_test_evaluated:false,production_promoted:false,scope:'All declared decoder-only controls reported, not an independent public reserve or a promotion based on new test data.'});
