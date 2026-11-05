import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { randomUUID } from 'node:crypto';
import { saveRun, restoreRuns } from '../run-ledger.mjs';
import { workflowCommand } from '../laboratory.mjs';
import { validateRun, artifactFiles } from '../server.mjs';
import { mergeSeedCandidates } from '../trace.js';
import { prepareCurriculum, oracle } from '../curriculum.mjs';

async function temporary(fn) {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-ledger-'));
  try{await fn(root);}finally{
    const resolved=await fs.realpath(root),relative=path.relative(await fs.realpath(os.tmpdir()),resolved);
    assert(!relative.startsWith('..')&&!path.isAbsolute(relative)&&path.basename(resolved).startsWith('axon-ui-ledger-'));
    await fs.rm(resolved,{recursive:true,force:true});
  }
}
test('workflow commands are fixed and options are bounded',()=>{
  assert.equal(validateRun({mode:'evolution'}).strategy,'balanced');
  for(const v of [{mode:'evolution',count:99},{mode:'continuous',count:999},{mode:'reactive',strategy:'balanced'},{mode:'tests',suite:'shell'},{mode:'tests',count:100},{mode:'evolution',command:'x'},{mode:'evolution',strategy:'x;echo'},{mode:'evolution',curriculum:'outside'},{mode:'tests',paused:true}])assert.throws(()=>validateRun(v));
  const root=path.resolve('.'),output=path.join(root,'owned');
  const command=workflowCommand(root,output,validateRun({mode:'evolution',seed:51}));
  assert.deepEqual(command.args,[output,'51','1000','balanced','transfer']);
  assert(command.program.endsWith(process.platform==='win32'?'axon-uic-evolution-lab.exe':'axon-uic-evolution-lab'));
  assert.equal(workflowCommand(root,output,validateRun({mode:'tests',suite:'ui'})).program,'bun');
  assert(workflowCommand(root,output,validateRun({mode:'tests',suite:'rust'})).args.includes('curriculum'));
  assert(!workflowCommand(root,output,validateRun({mode:'tests',suite:'rust-minimal'})).args.includes('--features'));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'seed',seed:51,validation:'development'})).args,[output,'51','1000','development']);
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'controller',seed:72918463,validation:'development'})).args,[output,'72918463','1000','development']);
  assert.throws(()=>validateRun({mode:'controller',strategy:'balanced'}));
  assert.throws(()=>validateRun({mode:'controller',validation:'execute'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'genome',seed:84371921,validation:'development'})).args,[output,'84371921','1000','development']);
  assert.throws(()=>validateRun({mode:'genome',strategy:'balanced'}));
  assert.throws(()=>validateRun({mode:'genome',validation:'shell'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'language',seed:71582347,validation:'development'})).args,[output,'71582347','1000','development']);
  assert.throws(()=>validateRun({mode:'language',strategy:'balanced'}));
  assert.throws(()=>validateRun({mode:'language',validation:'shell'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'cumulative',seed:39172841,validation:'development'})).args,[output,'39172841','1000','development']);
  assert.throws(()=>validateRun({mode:'cumulative',strategy:'balanced'}));
  assert.throws(()=>validateRun({mode:'cumulative',validation:'shell'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'semantic',seed:61423917,validation:'development'})).args,[output,'61423917','1000','development']);
  assert.throws(()=>validateRun({mode:'semantic',strategy:'balanced'}));
  assert.throws(()=>validateRun({mode:'semantic',validation:'shell'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'nursery',seed:61423917,count:1000})).args,[output,'61423917','1000']);
  assert.throws(()=>validateRun({mode:'nursery',validation:'reserved'}));
  assert.throws(()=>validateRun({mode:'nursery',count:10001}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'constructive',seed:75218341,count:5000})).args,[output,'75218341','5000']);
  assert.throws(()=>validateRun({mode:'constructive',validation:'reserved'}));
  assert.throws(()=>validateRun({mode:'constructive',strategy:'balanced'}));
  assert.throws(()=>validateRun({mode:'constructive',count:10001}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'constructive',seed:75218341,count:1000,strategy:'reinforced',curriculum:'arithmetic'})).args,[output,'75218341','1000','reinforced','arithmetic']);
  assert.throws(()=>validateRun({mode:'constructive',strategy:'shell'}));
  assert.throws(()=>validateRun({mode:'constructive',curriculum:'transfer'}));
  assert.throws(()=>validateRun({mode:'evolution',strategy:'reinforced'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'curriculum',seed:51,count:100})).args,[path.join(root,'ui/curriculum.mjs'),path.join(root,'evals/curricula/mixed-128-v1.json'),output,'51','100']);
  for(const curriculum of ['../private','a.json','a;echo','', 'A'])assert.throws(()=>validateRun({mode:'curriculum',curriculum}));
  assert.throws(()=>validateRun({mode:'curriculum',strategy:'diverse'}));
  assert.deepEqual(workflowCommand(root,output,validateRun({mode:'integration'})).args,['exercise',output]);
  assert.throws(()=>validateRun({mode:'integration',curriculum:'transfer-v2'}));
  const structural=validateRun({mode:'structural'});
  assert.equal(structural.seed,undefined);
  assert.deepEqual(workflowCommand(root,output,structural).args,['structure',path.join(root,'evals/curricula/compositional-v1.json'),output]);
  for(const extra of [{seed:51},{count:100},{curriculum:'../private'},{strategy:'balanced'}])assert.throws(()=>validateRun({mode:'structural',...extra}));
  for(const v of [{mode:'seed',strategy:'balanced'},{mode:'seed',validation:'shell'},{mode:'evolution',validation:'reserved'},{mode:'tests',validation:'reserved'}])assert.throws(()=>validateRun(v));
});
test('seed candidates aggregate only recorded evaluations and selection, without contaminating scores with search checkpoints',()=>{
  let records=mergeSeedCandidates([],{event:'seed_candidate',experiment:'seed',candidate:0,generation:0,genotype:'a',parent_a:'parent'});
  records=mergeSeedCandidates(records,{event:'seed_candidate',experiment:'seed',candidate:1,generation:0,genotype:'b'});
  records=mergeSeedCandidates(records,{event:'seed_evaluation',experiment:'seed',candidate:0,generation:0,work:500,solved:5});
  records=mergeSeedCandidates(records,{event:'seed_progress',experiment:'seed',candidate:0,work:12});assert.equal(records[0].work,500);assert.equal(records[0].parent_a,'parent');
  records=mergeSeedCandidates(records,{event:'seed_selection',experiment:'seed',generation:0,selected:[0]});assert.equal(records[0].frontier,true);assert.equal(records[1].frontier,false);
  assert.equal(mergeSeedCandidates(records,{event:'seed_evaluation',experiment:'seed',candidate:999}).length,2);
  let genomes=mergeSeedCandidates([],{event:'seed_evaluation',experiment:'common-genome-v1',candidate:255,generation:7,artifact:'candidate-255.axnum'});
  assert.equal(genomes[0].artifact,'candidate-255.axnum');
  genomes=mergeSeedCandidates(genomes,{event:'seed_selection',experiment:'common-genome-v1',generation:7,selected:[255]});assert.equal(genomes[0].frontier,true);
  assert.equal(mergeSeedCandidates(genomes,{event:'seed_evaluation',experiment:'common-genome-v1',candidate:256}).length,1);
  let cumulative=mergeSeedCandidates([],{event:'seed_candidate',experiment:'cumulative-v1',candidate:15,generation:1,admitted:5});
  cumulative=mergeSeedCandidates(cumulative,{event:'seed_selection',experiment:'cumulative-v1',generation:1,selected:[15]});assert.equal(cumulative[0].frontier,true);
  assert.equal(mergeSeedCandidates(cumulative,{event:'seed_candidate',experiment:'cumulative-v1',candidate:16}).length,1);
  let relational=mergeSeedCandidates([],{event:'seed_evaluation',experiment:'relational-v1',candidate:767,generation:11,genotype:'brain',encoded_bytes:6000,artifact:'population/candidate-767.axsem'});
  relational=mergeSeedCandidates(relational,{event:'seed_selection',experiment:'relational-v1',generation:11,selected:[767]});
  assert.equal(relational[0].frontier,true);
  assert.equal(mergeSeedCandidates(relational,{event:'seed_evaluation',experiment:'relational-v1',candidate:768}).length,1);
});
test('ledger replays recorded evidence instead of trusting stale manifest totals',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'));
  await saveRun(dir,{id,mode:'evolution',status:'complete',started:1,result:{fake:true},latest:{queries:999},output:'fake',cases:[{case:99}],trace:{mode:'search-checkpoints',methods:{}}});
  const records=[{event:'lab_start',experiment:'evolution',run:id,time:1,coreSeq:1},{event:'skill',experiment:'evolution',run:id,time:2,coreSeq:2,case:0,expression:'(x0 * x0)'},{event:'lab_result',experiment:'evolution',run:id,time:3,coreSeq:3,correct:100},{event:'test_output',run:id,time:4,text:'real output'},{event:'end',run:id,time:5,status:'complete',exitCode:0}];
  await fs.writeFile(path.join(root,'events',`${id}.jsonl`),records.map(JSON.stringify).join('\n')+'\n');
  const r=await restoreRuns(root);assert.equal(r.current.status,'complete');assert.equal(r.current.trace.records,5);assert.equal(r.current.trace.gaps,0);assert.equal(r.current.result.correct,100);assert.equal(r.current.cases.length,1);assert.equal(r.current.output,'real output\n');assert.deepEqual(r.current.latest,{});
  for(const content of [records.slice(0,-1).map(JSON.stringify).join('\n')+'\n',records.map(JSON.stringify).join('\n')+'\n{"partial":',records.map(e=>JSON.stringify(e.event==='end'?{...e,status:'Admitted'}:e)).join('\n')+'\n']){
    await fs.writeFile(path.join(root,'events',`${id}.jsonl`),content);const interrupted=await restoreRuns(root);assert.equal(interrupted.current.status,'interrupted');assert.equal(interrupted.current.finished,null);
  }
}));

test('semantic, nursery and constructive manifests survive restart with real progress fields',()=>temporary(async root=>{
  for(const mode of ['semantic','nursery','constructive','curriculum','integration']){
    const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'),{recursive:true});
    await saveRun(dir,{id,mode,status:'complete',started:mode==='nursery'?2:mode==='constructive'?3:1,trace:{mode:'test-checkpoints',methods:{}}});
    const records=[{event:'nursery_progress',experiment:'nursery-v1',run:id,time:1,coreSeq:1,candidates:100,unique:103,genes:3,imported:3},{event:'lab_result',experiment:'nursery-v1',run:id,time:2,coreSeq:2,candidates:100,unique:103,admitted:false},{event:'end',run:id,time:3,status:'complete',exitCode:0}];
    await fs.writeFile(path.join(root,'events',`${id}.jsonl`),records.map(JSON.stringify).join('\n')+'\n');
  }
  const restored=await restoreRuns(root);assert.equal(restored.runs.length,5);assert(restored.runs.some(r=>r.mode==='curriculum'));assert(restored.runs.some(r=>r.mode==='integration'));assert.equal(restored.current.samples[0].imported,3);assert.equal(restored.current.result.admitted,false);assert.equal(restored.current.trace.gaps,0);
}));

test('JSON curriculum is deterministic, editable, disjoint and never sends oracles to the learner',async()=>{
  const source=JSON.parse(await fs.readFile(new URL('../../evals/curricula/mixed-128-v1.json',import.meta.url),'utf8'));
  const a=prepareCurriculum(source,71),b=prepareCurriculum(source,71);
  assert.deepEqual(a,b);assert.equal(a.stages.length,128);
  assert.deepEqual(Object.fromEntries(['numeric','text','word_math'].map(k=>[k,a.stages.filter(s=>s.kind===k).length])),{numeric:64,text:40,word_math:24});
  for(const stage of a.stages){const inputs=Object.values(Object.fromEntries(['train','admission','test','transfer'].map(k=>[k,stage[k]]))).flat().map(r=>JSON.stringify(r.input));assert.equal(new Set(inputs).size,inputs.length);assert(!Object.hasOwn(stage,'oracle'));}
  const extended=structuredClone(source);extended.stages.push({...source.stages[0],id:'sum-new'});
  assert.deepEqual(prepareCurriculum(extended,71).stages.slice(0,128),a.stages);
  assert.notDeepEqual(prepareCurriculum(source,72).stages[0].test,a.stages[0].test);
  const written=a.stages.find(s=>s.id==='written-product');assert(written.transfer[0].input.startsWith('produto entre'));
  assert.equal(oracle(['x0','x0','mul','x1','x1','mul','add','sqrt'],[3,4]),5);
  assert.throws(()=>oracle(['x0','process.exit()'],[1]));
  for(const modify of [s=>s.stages.push(s.stages[0]),s=>s.stages[0].parents=['missing'],s=>s.stages[0].oracle=['x0','eval'],s=>s.budget.total_candidates=0,s=>s.splits.test=513,s=>s.stages[0].command='shell',s=>s.stages[64].output='{missing}']){
    const invalid=structuredClone(source);modify(invalid);assert.throws(()=>prepareCurriculum(invalid,71));
  }
});

test('curriculum artifacts are downloadable but unrelated JSON stays private',()=>temporary(async root=>{
  const id=randomUUID(),folder=path.join(root,'runs',id,'artifacts');await fs.mkdir(folder,{recursive:true});
  const names=['curriculum.json','prepared.json','curriculum-results.json','model-index.json','sum-result.json','sum-predictions.json'];
  for(const name of [...names,'private.json'])await fs.writeFile(path.join(folder,name),'{}');
  assert.deepEqual((await artifactFiles(root,id)).map(f=>f.name).sort(),names.sort());
}));
test('recovered mixed genome cases have the same numeric order and latest evidence as live cases',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'));
  await saveRun(dir,{id,mode:'genome',status:'complete',started:1,trace:{mode:'birth-checkpoints',methods:{}}});
  const events=[{event:'skill',case:2000,steps:['real-module']},{event:'validation',case:2,correct:32},{event:'validation',case:2000,correct:64},{event:'end',status:'complete',exitCode:0}].map((e,i)=>({...e,experiment:'common-genome-v1',run:id,time:i+1,coreSeq:i+1}));
  await fs.writeFile(path.join(root,'events',`${id}.jsonl`),events.map(JSON.stringify).join('\n')+'\n');
  const {current}=await restoreRuns(root);assert.deepEqual(current.cases.map(c=>c.case),[2,2000]);assert.equal(current.cases[1].correct,64);assert.deepEqual(current.cases[1].steps,['real-module']);assert.equal(current.trace.gaps,0);
}));
test('artifact catalog does not expose arbitrary files or traversal',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id,'artifacts');await fs.mkdir(dir,{recursive:true});await fs.writeFile(path.join(dir,'results.csv'),'case,status\n0,Admitted\n');await fs.writeFile(path.join(dir,'private.env'),'SECRET');await fs.mkdir(path.join(dir,'directory.csv'));
  assert.deepEqual((await artifactFiles(root,id)).map(f=>f.name),['results.csv']);
  assert.deepEqual(await artifactFiles(root,randomUUID()),[]);
}));
test('agent investigations and all policy summaries survive ledger restoration',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'));
  await saveRun(dir,{id,mode:'structural',status:'complete',started:1,trace:{mode:'example-induced-tree-rules-dialogue-and-symbolic-transfer',methods:{}}});
  const events=[{event:'investigation',method:'Agent::investigate',phase:'refutation',input_json:'[3,8]',actual_text:'11'},...['active','passive','no-library'].map((variant,i)=>({event:'agent_result',method:'Agent::campaign',case:200000+i,variant,correct:10,total:10})),{event:'lab_result',agent_correct:10,agent_total:10,agent_portable:10},{event:'end',status:'complete',exitCode:0}].map((e,i)=>({...e,experiment:'structural-v1',run:id,time:i+1,coreSeq:i+1}));
  await fs.writeFile(path.join(root,'events',`${id}.jsonl`),events.map(JSON.stringify).join('\n')+'\n');
  const {current}=await restoreRuns(root);assert.equal(current.result.agent_correct,10);assert.equal(current.cases.length,3);assert.equal(current.samples.filter(e=>e.event==='agent_result').length,3);assert.equal(current.samples[0].phase,'refutation');assert.equal(current.trace.gaps,0);
}));

test('algorithm hypotheses, drift and control measurements survive ledger restoration',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'));
  await saveRun(dir,{id,mode:'structural',status:'complete',started:1,trace:{mode:'example-induced-tree-rules-dialogue-and-symbolic-transfer',methods:{}}});
  const events=[{event:'algorithm_investigation',method:'Agent::learn_algorithm',operation:'somar',phase:'hypothesis',work:17,templates:3,mining_work:7},{event:'algorithm_drift',method:'Agent::algorithm_feedback',suspended:true,invalidated:2},...['inherited','no-library'].map((variant,i)=>({event:'algorithm_result',method:'Algorithm::control',case:399001+i,variant,correct:10,total:10,work:20+i})),...Array.from({length:300},()=>({event:'validation',phase:'later-stage'})),{event:'lab_result',algorithm_correct:12,algorithm_total:12},{event:'end',status:'complete',exitCode:0}].map((e,i)=>({...e,experiment:'structural-v1',run:id,time:i+1,coreSeq:i+1}));
  await fs.writeFile(path.join(root,'events',`${id}.jsonl`),events.map(JSON.stringify).join('\n')+'\n');
  const {current}=await restoreRuns(root);assert.equal(current.result.algorithm_correct,12);assert.equal(current.cases.length,2);assert.equal(current.samples.length,256);assert(current.samples.every(e=>e.event==='validation'));assert.equal(current.algorithmDrift.invalidated,2);assert.equal(current.algorithmInvestigation.work,17);assert.equal(current.algorithmInvestigation.templates,3);assert.equal(current.algorithmInvestigation.mining_work,7);assert.equal(current.trace.gaps,0);
}));

test('actual algorithm questions survive the bounded sample window and later unrelated investigations',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'));
  await saveRun(dir,{id,mode:'structural',status:'complete',started:1,trace:{methods:{}}});
  const question=JSON.stringify({phase:'query_observed',input:[2,3],actual:5,hypotheses:8,survivors:1,queries:1});
  const events=[{event:'algorithm_investigation',phase:'query_observed',operation:'investigate',last_inquiry:question,query_count:1},
    {event:'algorithm_investigation',phase:'consolidation',operation:'other'},...Array.from({length:300},()=>({event:'validation'})),
    {event:'end',status:'complete',exitCode:0}].map((e,i)=>({...e,run:id,time:i+1,coreSeq:i+1}));
  await fs.writeFile(path.join(root,'events',`${id}.jsonl`),events.map(JSON.stringify).join('\n')+'\n');
  const {current}=await restoreRuns(root);
  assert.equal(current.algorithmInvestigation.operation,'other');assert.equal(current.algorithmInquiry.operation,'investigate');
  assert.deepEqual(JSON.parse(current.algorithmInquiry.last_inquiry).actual,5);assert.equal(current.algorithmInquiry.query_count,1);
  assert(current.samples.every(e=>e.event==='validation'));assert.equal(current.trace.gaps,0);
}));

test('context decisions and replies survive bounded replay without control contamination',()=>temporary(async root=>{
  const id=randomUUID(),dir=path.join(root,'runs',id);await fs.mkdir(dir,{recursive:true});await fs.mkdir(path.join(root,'events'));
  await saveRun(dir,{id,mode:'structural',status:'complete',started:1,trace:{methods:{}}});
  const events=[{event:'session_interpretation',method:'Session::ground',case:810000,variant:'context',meanings:2,grounded:1,plans_json:'[]'},
    {event:'session_turn',method:'Session::turn',variant:'context',case:800000,actual_json:'{"status":"answer","value":{"value":17,"unit":"BRL"}}'},
    {event:'session_interpretation',method:'Session::ground',case:810001,variant:'no-memory',grounded:0},...Array.from({length:300},()=>({event:'validation'})),
    {event:'end',status:'complete',exitCode:0}].map((e,i)=>({...e,experiment:'structural-v1',run:id,time:i+1,coreSeq:i+1}));
  await fs.writeFile(path.join(root,'events',`${id}.jsonl`),events.map(JSON.stringify).join('\n')+'\n');
  const {current}=await restoreRuns(root);
  assert.equal(current.cases.find(e=>e.method==='Session::ground'&&e.variant==='context').grounded,1);
  assert.equal(JSON.parse(current.cases.find(e=>e.method==='Session::turn'&&e.variant==='context').actual_json).value.value,17);
  assert.equal(current.samples.length,256);assert.equal(current.trace.gaps,0);
}));

test('nested learned seeds and snapshots are downloadable without exposing arbitrary JSON',()=>temporary(async root=>{
  const id=randomUUID(),folder=path.join(root,'runs',id,'artifacts');await fs.mkdir(path.join(folder,'best-candidate'),{recursive:true});
  for(const [name,data] of [['best-candidate/seed.axnum','1001'],['library.axsem','capsule'],['registry-snapshot.axep','journal'],['audit-nursery-v2.json','{}'],['credentials.json','secret'],['unexpected.js','source']])await fs.writeFile(path.join(folder,name),data);
  assert.deepEqual((await artifactFiles(root,id)).map(f=>f.name),['audit-nursery-v2.json','best-candidate/seed.axnum','library.axsem','registry-snapshot.axep']);
}));

test('nested birth artifacts cannot starve root seeds when the catalog reaches its quota',()=>temporary(async root=>{
  const id=randomUUID(),folder=path.join(root,'runs',id,'artifacts'),birth=path.join(folder,'birth-0');
  await fs.mkdir(birth,{recursive:true});
  for(let i=0;i<1024;i++)await fs.writeFile(path.join(birth,`module-${i}.axseed`),'module');
  await fs.writeFile(path.join(folder,'founder.axnum'),'1001');
  await fs.writeFile(path.join(folder,'candidate-255.axnum'),'1002');
  const files=await artifactFiles(root,id);
  assert.equal(files.length,1024);
  assert(files.some(f=>f.name==='founder.axnum'));
  assert(files.some(f=>f.name==='candidate-255.axnum'));
}));
