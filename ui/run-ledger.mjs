import fs from 'node:fs/promises';
import path from 'node:path';
import { createReadStream } from 'node:fs';
import readline from 'node:readline';
import { QueryMemory, mergeSeedCandidates } from './trace.js';

const idPattern=/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const terminal=['complete','error','cancelled'];
const modes=['factor','transfer','math','knowledge','index','evolution','reactive','continuous','seed','controller','genome','language','cumulative','semantic','nursery','constructive','curriculum','integration','structural','tests'];
export async function saveRun(directory,job) {
  const temporary=path.join(directory,'run.next.json'),destination=path.join(directory,'run.json');
  const handle=await fs.open(temporary,'w');
  try {await handle.writeFile(JSON.stringify({schema:'axon-run-v1',job}));await handle.sync();}finally{await handle.close();}
  await fs.rename(temporary,destination);
}
async function allowed(file,root,limit) {
  const stat=await fs.lstat(file);
  if(!stat.isFile()||stat.isSymbolicLink()||stat.size>limit)throw new Error('Invalid ledger file');
  const relative=path.relative(await fs.realpath(root),await fs.realpath(file));
  if(relative.startsWith('..')||path.isAbsolute(relative))throw new Error('Ledger outside scope');
}
export async function restoreRuns(dataRoot) {
  const runs=[];let names;
  try{names=await fs.readdir(path.join(dataRoot,'runs'));}catch{return {runs,current:null};}
  for(const id of names.filter(n=>idPattern.test(n)).slice(-256)){
    try{
      const file=path.join(dataRoot,'runs',id,'run.json');await allowed(file,dataRoot,4000000);
      const saved=JSON.parse(await fs.readFile(file,'utf8'));
      if(saved.schema!=='axon-run-v1'||saved.job?.id!==id||!Number.isFinite(saved.job.started)||!modes.includes(saved.job.mode)||!saved.job.trace||typeof saved.job.trace.methods!=='object')continue;
      if(!terminal.includes(saved.job.status))saved.job.status='interrupted';
      runs.push({...saved.job,recovered:true});
    }catch{/* An invalid manifest does not grant authority to resume a process. */}
  }
  runs.sort((a,b)=>a.started-b.started);const retained=runs.slice(-32),current=retained.at(-1);
  if(!current)return {runs:retained,current:null};
  const memory=new QueryMemory();let graph={nodes:[],edges:[],models:[],phase:'stored-snapshot'};const events=[];
  const counters={physical:0,cached:0,queryCalls:0,steps:0,errors:0};
  const trace={mode:current.trace?.mode??'entry-exit-only',records:0,lastCoreSeq:0,gaps:0,methods:{},logError:null};
  let incomplete=false,sawEnd=false;
  current.latest={};current.output='';current.result=null;current.progress=null;current.cases=[];current.samples=[];current.candidates=[];current.finished=null;current.exitCode=null;
  try{
    const file=path.join(dataRoot,'events',`${current.id}.jsonl`);await allowed(file,dataRoot,256000000);
    const stream=createReadStream(file);const lines=readline.createInterface({input:stream,crlfDelay:Infinity});
    for await(const line of lines){
      if(Buffer.byteLength(line)>512000){incomplete=true;break;}
      let e;try{e=JSON.parse(line);}catch{incomplete=true;break;}
      if(!e||e.run!==current.id||typeof e.event!=='string'||sawEnd){incomplete=true;break;}
      trace.records++;if(e.method)trace.methods[e.method]=(trace.methods[e.method]??0)+1;
      if(e.coreSeq!==undefined){if(e.coreSeq!==trace.lastCoreSeq+1)trace.gaps++;trace.lastCoreSeq=e.coreSeq;}
      if(e.telemetry)trace.mode=e.telemetry;
      if(e.event==='query'){counters.queryCalls++;counters[e.cacheHit?'cached':'physical']++;}
      if(e.event==='step')counters.steps++;
      if(e.event==='error'||e.event==='query_error')counters.errors++;
      memory.accept(e);
      current.candidates=mergeSeedCandidates(current.candidates,e);
      if(e.event==='graph')graph={nodes:e.nodes,edges:e.edges,models:e.models,phase:e.phase};
      if(e.event==='result'||e.event==='lab_result')current.result=e;
      if(e.event==='algorithm_investigation')current.algorithmInvestigation=e;
      if(e.event==='algorithm_investigation'&&['query_selected','query_observed'].includes(e.phase))current.algorithmInquiry=e;
      if(e.event==='algorithm_drift')current.algorithmDrift=e;
      if(e.experiment){current.progress=e;const old=current.cases.find(c=>c.case===e.case);if(Number.isInteger(e.case)){current.cases=current.cases.filter(c=>c.case!==e.case);current.cases.push({...old,...e});}}
      if(['evolution_progress','continuous_progress','controller_feedback','study_query','validation','plan','prediction','seed_selection','genome','nursery_progress','dna_growth','investigation','agent_result','algorithm_investigation','algorithm_drift','algorithm_result'].includes(e.event)){current.samples.push(e);if(current.samples.length>256)current.samples.shift();}
      if(['output','process_output','process_stderr','test_output'].includes(e.event))current.output=`${current.output}${e.text??''}\n`.slice(-128000);
      for(const key of ['queries','latencyNs','cacheNodes','hypotheses','residual'])if(typeof e[key]==='number')current.latest[key]=e[key];
      if(e.event==='end'){if(!terminal.includes(e.status)||!Number.isFinite(e.time)){incomplete=true;break;}sawEnd=true;current.status=e.status;current.finished=e.time;current.exitCode=e.exitCode??null;}
      events.push(e.event==='graph'?{...e,nodes:undefined,edges:undefined,models:undefined,detail:`${e.nodes.length} states / ${e.edges.length} transitions`}:e);if(events.length>160)events.shift();
    }
    lines.close();stream.destroy();
  }catch(error){incomplete=true;trace.logError=error.message;}
  if(incomplete||!sawEnd){current.status='interrupted';current.finished=null;trace.logError??='Incomplete trace or missing terminal record';}
  if(current.status==='interrupted')memory.accept({event:'end'});
  current.cases.sort((a,b)=>a.case-b.case);
  current.trace=trace;current.counters=counters;
  return {runs:retained,current,graph,memory:[...memory.nodes.values()],outputs:[...memory.outputs.values()],events};
}
