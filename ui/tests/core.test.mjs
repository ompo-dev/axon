import {test} from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import fs from 'node:fs/promises';
import os from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {createObservatory} from '../server.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const real=process.env.AXON_UI_REAL_CORE==='1';
test('real core: checkpoint controls, audited transfer, factor, exact task and local knowledge', {skip:!real,timeout:180000},async t=>{
  const storageRoot=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-core-'));
  const app=await createObservatory({root,storageRoot,port:0,autoSnapshot:false});
  const headers={'Content-Type':'application/json','X-Axon-Session':app.state().token};
  const post=async(route,body)=>{const response=await fetch(`${app.url}/api/${route}`,{method:'POST',headers,body:JSON.stringify(body)});const value=await response.json();assert.equal(response.status,200,JSON.stringify(value));return value;};
  const until=async(predicate,timeout=60000)=>{const end=Date.now()+timeout;while(Date.now()<end){const state=app.state();if(predicate(state))return state;await delay(40);}assert.fail(JSON.stringify(app.state().events.slice(-4)));};
  const complete=()=>until(s=>['complete','error','cancelled'].includes(s.current?.status));
  try{
    await t.test('initial pause, one query, pause stability, resume and exact result',async()=>{
      const job=await post('run',{mode:'transfer',paused:true});
      await until(s=>s.events.some(e=>e.event==='start'&&e.run===job.id));
      assert(!app.state().events.some(e=>e.event==='query'));
      await post('control',{command:'step'});
      await until(s=>s.events.some(e=>e.event==='query'&&e.run===job.id));
      const count=app.state().events.filter(e=>e.event==='query').length;
      await delay(180);assert.equal(app.state().events.filter(e=>e.event==='query').length,count);
      assert.equal(app.state().current.status,'paused');
      assert.equal((await fetch(`${app.url}/api/run`,{method:'POST',headers,body:'{"mode":"factor"}'})).status,400);
      await post('control',{command:'resume'});
      const state=await complete();assert.equal(state.current.status,'complete');
      const result=state.events.findLast(e=>e.event==='result'&&e.run===job.id);assert(result.exact);assert.equal(result.residual,0);assert(result.checkedEdges>0);assert.equal(result.fallback,false);
      const response=await fetch(`${app.url}/api/export?id=${job.id}`);assert.equal(response.status,200);const records=(await response.text()).trim().split('\n').map(JSON.parse);assert(records.every(e=>e.run===job.id));assert(records.some(e=>e.event==='graph'));assert(records.some(e=>e.event==='result'));
      const core=records.filter(e=>e.coreSeq!==undefined);core.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
      assert(records.every(e=>e.session===state.session));
      assert(core.every((e,i)=>Number.isFinite(e.unixMs)&&e.coreNs>=0&&(!i||e.coreNs>=core[i-1].coreNs)));
      const methods={};for(const e of records)if(e.method)methods[e.method]=(methods[e.method]??0)+1;
      assert.deepEqual(Object.fromEntries(Object.entries(state.current.trace.methods)),methods);
      const contexts=new Set(records.filter(e=>e.event==='query').map(e=>e.context));
      for(const context of contexts){const queries=records.filter(e=>e.event==='query'&&e.context===context);assert.equal(queries.length,queries.at(-1).calls);assert(queries.some(e=>e.cacheHit));}
      assert.equal(state.current.trace.gaps,0);assert.equal(records.length,state.current.trace.records);
    });
    await t.test('cancel releases paused process',async()=>{await post('run',{mode:'factor',dimensions:2,paused:true});await post('control',{command:'cancel'});const state=await complete();assert.equal(state.current.status,'cancelled');});
    await t.test('unknown mechanism refutes prior and learns a new model',async()=>{const job=await post('run',{mode:'transfer',novel:true});const state=await complete();assert.equal(state.current.status,'complete');const result=state.events.findLast(e=>e.event==='result'&&e.run===job.id);assert.equal(result.fallback,true);assert.equal(result.exact,true);});
    await t.test('composed graph contains actual complete transitions',async()=>{const job=await post('run',{mode:'factor',dimensions:2});const state=await complete();assert.equal(state.current.status,'complete');const result=state.events.findLast(e=>e.event==='result'&&e.run===job.id);assert(result.exact);assert.equal(state.graph.models.length,2);const ids=new Set(state.graph.nodes.map(n=>n.id));assert(state.graph.edges.every(e=>ids.has(e.source)&&ids.has(e.target)));assert(result.checkedEdges>0);
      const records=(await(await fetch(`${app.url}/api/export?id=${job.id}`)).text()).trim().split('\n').map(JSON.parse);
      const queries=records.filter(e=>e.event==='query');assert.equal(queries.length,queries.at(-1).calls);const core=records.filter(e=>e.coreSeq!==undefined);core.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
      const memoryIds=new Set(state.memory.map(n=>n.id));assert(state.memory.every(n=>!n.parent||memoryIds.has(n.parent)));
      for(const step of records.filter(e=>e.event==='step'))for(let i=0;i<2;i++){const context=`${step.context}/factor-${i}`;const edge=state.graph.edges.find(e=>e.source===`${context}:s${step.fromStates[i]}`&&e.action===step.action);assert.equal(edge.target,`${context}:s${step.toStates[i]}`);assert.equal(edge.output,step.reply[i]);}
      assert(state.memory.filter(n=>n.root).every(n=>n.active===false));
    });
    await t.test('typed exact task really invokes Rust solver',async()=>{const job=await post('run',{mode:'math',task:'task average_stream {\n data numbers: Vec<u64> = [2, 4, 9, 12, 18]\n goal derive IncrementalArtifact<AverageExactU64>\n}'});const state=await complete();assert.equal(state.current.status,'complete');assert(state.events.some(e=>e.run===job.id&&e.event==='output'&&e.text.length>20));});
    await t.test('index and investigate use separate local memory',async()=>{await post('run',{mode:'index'});assert.equal((await complete()).current.status,'complete');assert(app.state().indexed);const job=await post('run',{mode:'knowledge',query:'BehaviorMemory drift context'});const state=await complete();assert.equal(state.current.status,'complete');assert(state.events.some(e=>e.event==='output'&&e.run===job.id&&e.text.includes('behavior')));});
  }finally{await app.close();await delay(100);const resolved=await fs.realpath(storageRoot),relative=path.relative(await fs.realpath(os.tmpdir()),resolved);assert(!relative.startsWith('..')&&!path.isAbsolute(relative)&&path.basename(resolved).startsWith('axon-ui-core-'));await fs.rm(resolved,{recursive:true,force:true});}
});
test('real labs: telemetry, artifacts, test runner and durable interrupted recovery', {skip:!real,timeout:120000}, async()=>{
  const storageRoot=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-core-'));
  let app=await createObservatory({root,storageRoot,port:0,autoSnapshot:false});
  const post=async(route,body)=>{const response=await fetch(`${app.url}/api/${route}`,{method:'POST',headers:{'Content-Type':'application/json','X-Axon-Session':app.state().token},body:JSON.stringify(body)});const value=await response.json();assert.equal(response.status,200,JSON.stringify(value));return value;};
  const complete=async()=>{const limit=Date.now()+60000;while(Date.now()<limit){const job=app.state().current;if(job.finished)return job;await delay(40);}assert.fail('Workflow timeout');};
  try {
    for(const options of [{mode:'evolution',strategy:'balanced',curriculum:'transfer',count:1000},{mode:'reactive',count:1000},{mode:'continuous',count:3000},{mode:'tests',suite:'ui'},{mode:'seed',validation:'development',count:100},{mode:'controller',validation:'development',count:100},{mode:'genome',validation:'development',count:100},{mode:'language',validation:'development',count:100},{mode:'cumulative',validation:'development',count:100}]){
      await post('run',{...options,seed:options.mode==='seed'?70143863:options.mode==='controller'?72918463:options.mode==='genome'?84371921:82641973});const job=await complete();assert.equal(job.status,'complete',job.output);assert.equal(job.exitCode,0);assert.equal(job.trace.gaps,0);
      const records=(await(await fetch(`${app.url}/api/export?id=${job.id}`)).text()).trim().split('\n').map(JSON.parse);
      assert.equal(records.length,job.trace.records);assert.equal(records.at(-1).event,'end');
      const core=records.filter(e=>e.coreSeq!==undefined);core.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
      if(options.mode==='evolution'){assert.equal(job.result.passed,8);assert.equal(job.result.correct,8000);assert(job.cases.every(c=>c.wiring.length===224));assert(records.some(e=>e.event==='curriculum_frozen'));}
      if(options.mode==='reactive')assert.equal(job.result.correct,12000);
      if(options.mode==='continuous'){assert.equal(job.result.passed,true);assert(job.result.promotions>=9);assert.equal(job.result.errors,job.result.scored-job.result.correct);}
      if(options.mode==='tests'){assert.equal(core.length,0);assert(records.some(e=>e.event==='process_stderr'||e.event==='test_output'));}
      if(options.mode==='seed'){assert.equal(job.candidates.length,24);assert.equal(job.result.solved,5);assert.equal(job.result.correct,job.result.heldout);assert(job.candidates.some(c=>c.frontier));assert(records.some(e=>e.event==='curriculum_frozen'));}
      if(options.mode==='controller'){assert.equal(job.result.correct,200);assert.equal(job.result.physical_correct,199);assert.equal(job.result.drifts,1);assert(job.result.baseline_correct<200);assert(job.cases.every(c=>c.steps.length>=5));assert(records.some(e=>e.event==='controller_drift'));assert(records.some(e=>e.event==='controller_repair'));assert(job.samples.length>0);}
      if(options.mode==='genome'){assert.equal(job.candidates.length,256);assert.equal(job.result.attempts,1536);assert(job.candidates.some(c=>c.frontier));assert(job.cases.some(c=>c.steps?.length===6));assert(records.some(e=>e.event==='stage'&&e.protocol==='common-genome-v3'));assert(records.some(e=>e.method==='GenomeTransfer::audit'));}
      if(options.mode==='language'){assert.equal(job.result.correct,388);assert.equal(job.result.refused,12);assert.equal(job.result.wrong,0);assert.equal(job.result.negative_pass,4);assert.equal(job.cases.length,4);assert(records.some(e=>e.method==='LanguageFrame::learn'));}
      if(options.mode==='cumulative'){assert.equal(job.candidates.length,16);assert.equal(job.result.checked,3200);assert.equal(job.result.wrong,0);assert.equal(job.result.file_plans,3);assert.equal(job.cases.filter(c=>c.arm).length,32);assert.equal(job.samples.filter(e=>e.event==='plan').length,3);assert(records.some(e=>e.event==='study_query'));assert(job.candidates.some(c=>c.frontier));}
      const files=(await(await fetch(`${app.url}/api/artifacts?id=${job.id}`)).json()).files;
      if(options.mode!=='tests'){assert(files.some(f=>f.name==='results.csv'));assert.equal((await fetch(`${app.url}/api/artifact?id=${job.id}&name=results.csv`)).status,200);}
      assert.equal((await fetch(`${app.url}/api/artifact?id=${job.id}&name=../private.env`)).status,404);
    }
    const finished=app.state().current;await delay(100);await app.close();app=await createObservatory({root,storageRoot,port:0,autoSnapshot:false});
    assert.equal(app.state().current.id,finished.id);assert.equal(app.state().current.status,'complete');assert.equal(app.state().current.trace.records,finished.trace.records);assert.equal(app.state().current.output,finished.output);assert.deepEqual(app.state().current.candidates,finished.candidates);
    assert.deepEqual(app.state().current.cases,finished.cases);assert.deepEqual(app.state().current.samples,finished.samples);
    await post('run',{mode:'evolution',seed:51,count:10000,strategy:'mutation-only',curriculum:'transfer'});await post('control',{command:'cancel'});assert.equal((await complete()).status,'cancelled');
    await post('run',{mode:'transfer',paused:true});const interrupted=app.state().current.id;await app.close();await delay(100);app=await createObservatory({root,storageRoot,port:0,autoSnapshot:false});
    assert.equal(app.state().current.id,interrupted);assert.equal(app.state().current.status,'interrupted');assert.equal((await fetch(`${app.url}/api/export?id=${interrupted}`)).status,200);
  } finally {
    await app.close();const resolved=await fs.realpath(storageRoot),relative=path.relative(await fs.realpath(os.tmpdir()),resolved);assert(!relative.startsWith('..')&&!path.isAbsolute(relative)&&path.basename(resolved).startsWith('axon-ui-core-'));await fs.rm(resolved,{recursive:true,force:true});
  }
});
