import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';
import {rescaleProgram} from './relational-pointer-rescaling.mjs';
import {pointerExamples} from './relational-pointer-source.mjs';
import {workingExamples} from './working-pointer-source.mjs';
import {repairExamples} from './working-pointer-repair.mjs';
import {checkPointerTrace} from './relational-pointer-trace.mjs';

const out=process.argv[2],root=path.resolve();if(!out||!path.resolve(out).startsWith(root+path.sep))throw Error('usage: node scripts/prepare-pointer-augmentation.mjs <new-workspace-output>');
await fs.mkdir(out);const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2),{flag:'wx'});
const ancestor='.axon-memory/m1-relational-pointer-source-v2',frozen=JSON.parse(await fs.readFile(path.join(ancestor,'freeze-before-labels.json'))),protocolFile=process.argv[3]??'evals/protocols/m1-pointer-augmentation-v1.json',protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes),sources={};
async function capture(file){const relative=path.relative(root,path.resolve(file));assert(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative));let current=root;for(const part of relative.split(path.sep)){current=path.join(current,part);assert(!(await fs.lstat(current)).isSymbolicLink(),'Source link');}const bytes=await fs.readFile(file),name=relative.replaceAll('\\','/'),destination=path.join(out,'source',relative);assert(bytes.length<=67108864);if(sources[name]){assert.equal(sources[name],sha(bytes));return;}await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,bytes,{flag:'wx'});sources[name]=sha(bytes);}
for(const file of [...Object.keys(frozen.sources),protocolFile,'scripts/prepare-pointer-augmentation.mjs','scripts/relational-pointer-rescaling.mjs','scripts/relational-pointer-rescaling.test.mjs'])await capture(file);
for(const dir of ['scripts','evals/protocols'])for(const e of await fs.readdir(dir,{withFileTypes:true})){assert(!e.isSymbolicLink());if(e.isFile()&&/\.(mjs|ps1|json|txt)$/u.test(e.name))await capture(path.join(dir,e.name));}
for(const dir of ['examples','src']){async function walk(directory){for(const e of await fs.readdir(directory,{withFileTypes:true})){assert(!e.isSymbolicLink());const file=path.join(directory,e.name);if(e.isDirectory())await walk(file);else if(/\.(md|rs)$/u.test(file))await capture(file);}}await walk(dir);}
for(const file of ['dependencies-v1-data.json','agent-v1.json','structural-v3.json','m1-context-v1.json'])await capture(`evals/curricula/${file}`);
if(protocol.repair_donor){assert.equal(sha(await fs.readFile(protocol.repair_donor.file)),protocol.repair_donor.sha256);await capture(protocol.repair_donor.file);}
const sourceBytes=await fs.readFile(path.join(ancestor,'cohorts.json')),ancestralStudy=JSON.parse(await fs.readFile('.axon-memory/m1-relational-pointer-v2/freeze.json'));assert.equal(sha(sourceBytes),ancestralStudy.cohorts_sha256);
const parent=await fs.readFile(protocol.parent);assert.equal(sha(parent),protocol.parent_sha256);await fs.writeFile(path.join(out,'parent.axnum'),parent,{flag:'wx'});
await write('freeze-before-labels.json',{protocol_file:protocolFile,protocol_sha256:sha(protocolBytes),sources,parent_sha256:sha(parent),source_cohorts_sha256:sha(sourceBytes),scope:'All current sources and native contracts fixed before deriving frozen train cohorts. No raw public corpus or new answers opened; already consumed train source programs only. Function graphs, not English truth certificates.'});
const matched=JSON.parse(sourceBytes).matched;assert.equal(matched.length,629);const working=protocol.intervention==='working-provenance',anonymity=protocol.intervention==='candidate-vocabulary',frozenCohort=working||anonymity,cohorts=frozenCohort?{matched}:{practice:[],augmented:[]},refusals=[];
if(working)for(const e of matched){e.working={source:e.full.source,examples:workingExamples(e.full.source)};e.blind={source:e.full.source,examples:workingExamples(e.full.source,true)};assert.deepEqual(e.working.examples.map(x=>x.targets),e.blind.examples.map(x=>x.targets));}
let repairAudit;
if(working&&protocol.repair_donor){
  const donor=JSON.parse(await fs.readFile(protocol.repair_donor.file));assert.deepEqual(donor.map(r=>r.id),matched.map(e=>e.id));assert(Number.isSafeInteger(protocol.repair_limit)&&protocol.repair_limit>0&&protocol.repair_limit<=3000);
  cohorts.practice=structuredClone(matched);cohorts.repair=structuredClone(matched);let added=0,duplicates=0,skipped=0;const refused=[];
  for(const [i,e]of matched.entries()){
    const text=e.full.source.examples[0].text.split(' ').slice(1,-2).join(' ');checkPointerTrace(donor[i],text);
    const r=repairExamples(e.full.source,donor[i]),seen=new Set(e.working.examples.map(x=>JSON.stringify(x)));refused.push(...r.refusals.map(x=>({id:e.id,...x})));
    for(const pair of r.pairs){const key=JSON.stringify(pair.working);if(seen.has(key)){duplicates++;continue;}seen.add(key);if(added>=protocol.repair_limit){skipped++;continue;}
      for(const page of ['working','blind']){cohorts.practice[i][page].examples.push(e[page].examples[pair.teacher_index]);cohorts.repair[i][page].examples.push(pair[page]);}added++;
    }
  }
  for(const page of ['working','blind'])assert.deepEqual(cohorts.practice.map(e=>e[page].examples.length),cohorts.repair.map(e=>e[page].examples.length));
  repairAudit={donor:protocol.repair_donor,added,duplicates,skipped,refusals:refused,independent_new_questions:0,scope:'Actual frozen reference-model train rollouts; structural repair teacher on already provided programs, not autonomous discovery, online RL or development-derived supervision.'};await write('repair-audit.json',repairAudit);
}
for(const e of frozenCohort?[]:matched){
  let scaled;try{scaled=[2,4].map(factor=>Object.fromEntries(['full','free'].map(page=>{const source=rescaleProgram(e[page].source,factor),examples=pointerExamples(source);assert.deepEqual(examples.map(x=>x.targets),e[page].examples.map(x=>x.targets));return[page,{source,examples}];})));}catch(error){refusals.push({id:e.id,error:error.message});continue;}
  for(let replica=0;replica<3;replica++){const metadata={id:`${e.id}-replica-${replica}`,source_id:e.id,replica,question_sha256:e.question_sha256,shape:e.shape};cohorts.practice.push({...metadata,full:e.full,free:e.free});cohorts.augmented.push({...metadata,...(replica===0?{full:e.full,free:e.free}:scaled[replica-1])});}
}
assert(Object.values(cohorts).every(rows=>rows.length>0));await write('cohorts.json',cohorts);
for(const [i,policy]of protocol.policies.entries()){const examples=cohorts[policy.cohort].flatMap(e=>e[policy.page??(policy.numbers_only?'free':'full')].examples);assert(examples.length<=10000);await write(`policy-${i}-train.json`,{budget:{...protocol.budget,...policy.budget},examples});}
if(protocol.default_reproduction)await write('default-reproduction-train.json',{budget:protocol.default_reproduction.budget,examples:matched.flatMap(e=>e.full.examples)});
for(const file of ['development-inputs.json','library.json']){const bytes=await fs.readFile(path.join(ancestor,file));if(file==='development-inputs.json')assert.equal(sha(bytes),ancestralStudy.inputs_sha256);await fs.writeFile(path.join(out,file),bytes,{flag:'wx'});}
const library=JSON.parse(await fs.readFile(path.join(out,'library.json')));for(const p of library){const bytes=await fs.readFile(path.join(ancestor,p.file));assert.equal(sha(bytes),p.sha256);await fs.writeFile(path.join(out,p.file),bytes,{flag:'wx'});}
const audit={schema:working?'axon-working-pointer-source/v1':anonymity?'axon-pointer-anonymity-source/v1':'axon-pointer-augmentation-source/v1',cohorts:Object.fromEntries(Object.entries(cohorts).map(([name,rows])=>[name,{episodes:rows.length,independent_originals:frozenCohort?rows.length:rows.length/3,frames:rows.reduce((n,r)=>n+r[working?'working':'full'].examples.length,0)}])),repair:repairAudit,refusals,source_programs:629,new_corpus_answers_read:0,development_inputs:1481,m1_complete:false,m2_permitted:false};await write('audit.json',audit);console.log(JSON.stringify({...audit,repair:repairAudit?{...repairAudit,refusals:repairAudit.refusals.length}:undefined}));
