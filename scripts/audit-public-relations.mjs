import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';

const [runPath,corpus,outputPath]=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/audit-public-relations.mjs <completed-run> <public-corpus> <new-output>');
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,value)=>fs.writeFile(path.join(output,name),JSON.stringify(value,null,2),{flag:'wx'});
const sha=raw=>createHash('sha256').update(raw).digest('hex');
const audit=JSON.parse(await fs.readFile(path.join(runPath,'audit.json')));
const selection=JSON.parse(await fs.readFile(path.join(runPath,'selection.json')));
const expected=[...selection.policies].sort((a,b)=>b.validation.metrics.correct-a.validation.metrics.correct||a.training.summary.statistics.selected-b.training.summary.statistics.selected||a.index-b.index)[0];
assert.equal(expected.index,audit.selected);
const test=parse(await fs.readFile(path.join(corpus,'test.csv')),{columns:true,bom:true});
const rows=JSON.parse(await fs.readFile(path.join(runPath,'selected-test-predictions.json')));
assert.equal(rows.length,test.length);
const counted={answered:0,correct:0,wrong:0,abstained:0};
rows.forEach((r,i)=>{assert.equal(r.id,`test-${i}`);if(r.prediction===null)counted.abstained++;else {counted.answered++;if(r.prediction===Number(test[i].label))counted.correct++;else counted.wrong++;}});
for(const [key,value] of Object.entries(counted))assert.equal(audit.test.metrics[key],value);
const parentBytes=await fs.readFile(path.join(runPath,'policy-0','structural.axstruct'));
for(const policy of selection.policies)assert.deepEqual(await fs.readFile(path.join(runPath,`policy-${policy.index}`,'structural.axstruct')),parentBytes);
const binary=path.resolve(runPath,'candidate.exe'),number=path.resolve(runPath,`policy-${audit.selected}`,'structural.axnum');
const trainRaw=await fs.readFile(path.join(corpus,'train.csv'));
const training=parse(trainRaw,{columns:true,bom:true});
const statements=[...new Set(training.flatMap(r=>[r.premise,r.hypothesis]))];
await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),number_sha256:sha(await fs.readFile(number)),source_sha256:sha(trainRaw),scorer_sha256:sha(await fs.readFile(process.argv[1])),
  rule:'Every distinct statement in public train is paired with itself. The logical NLI reflexivity expectation is Entailment (1). This is a generated engineering diagnostic using public source text, not independent-authorship M1 acceptance or retraining. Frozen before predictions.'});
let correct=0,wrong=0,abstained=0;
for(let start=0;start<statements.length;start+=4096) {
  const name=`identity-${start}`;
  const input=statements.slice(start,start+4096).map((s,i)=>({id:`identity-${start+i}`,premise:s,hypothesis:s}));
  await write(`${name}-inputs.json`,input);
  const receipt=path.join(output,`${name}-predictions.json`);
  const child=await promisify(execFile)(binary,[audit.batch_command??'relation-batch',number,path.join(output,`${name}-birth`),path.join(output,`${name}-inputs.json`),receipt],{windowsHide:true,timeout:300000,maxBuffer:16000000,env:{...process.env,AXON_LAB_TELEMETRY:'1'}});
  await fs.writeFile(path.join(output,`${name}.stdout.jsonl`),child.stdout,{flag:'wx'});
  await fs.writeFile(path.join(output,`${name}.stderr.log`),child.stderr,{flag:'wx'});
  const predictions=JSON.parse(await fs.readFile(receipt));assert.equal(predictions.length,input.length);
  predictions.forEach((r,i)=>{assert.equal(r.id,input[i].id);if(r.prediction===null)abstained++;else if(r.prediction===1)correct++;else wrong++;});
}
const report={schema:'axon-relation-independent-recount/v1',original_test:counted,parent_pages_byte_identical:true,
  reflexivity:{total:statements.length,correct,wrong,abstained},m1_complete:false,
  scope:'Recount of existing consumed test, plus separately frozen metamorphic control. Never count a correct answer subset as broad language understanding. Reflexivity failures are preserved, not used to alter this candidate.'};
await write('audit.json',report);console.log(JSON.stringify(report));
