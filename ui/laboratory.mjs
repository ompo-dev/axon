import path from 'node:path';
import { spawn } from 'node:child_process';

export const workflows = [
  {id:'evolution',label:'Evolucao de programas',kind:'training',binary:'axon-uic-evolution-lab',telemetry:'search-checkpoints'},
  {id:'reactive',label:'Aprendizagem reativa',kind:'training',binary:'axon-uic-reactive-lab',telemetry:'acquisition-and-validation-checkpoints'},
  {id:'continuous',label:'Aprendizagem continua',kind:'training',binary:'axon-uic-continuous-lab',telemetry:'prequential-checkpoints'},
  {id:'seed',label:'Seeds herdaveis',kind:'training',binary:'axon-uic-seed-lab',telemetry:'seed-selection-and-search-checkpoints'},
  {id:'controller',label:'Objetivos e procedimentos',kind:'training',binary:'axon-uic-controller-lab',telemetry:'typed-composition-and-64-frame-checkpoints'},
  {id:'genome',label:'Genoma e nascimento',kind:'training',binary:'axon-uic-genome-lab',telemetry:'birth-acquisition-and-selection-checkpoints'},
  {id:'language',label:'Linguagem e matematica',kind:'training',binary:'axon-uic-language-lab',telemetry:'dictionary-and-grounded-validation-checkpoints'},
  {id:'cumulative',label:'Aprendizagem cumulativa',kind:'training',binary:'axon-uic-cumulative-lab',telemetry:'durable-investigation-inheritance-and-file-plan-checkpoints'},
  {id:'semantic',label:'Texto, mundo e matematica / R1',kind:'training',binary:'axon-uic-semantic-lab',telemetry:'relational-induction-structural-selection-and-heldout-checkpoints'},
  {id:'nursery',label:'Viveiro de seeds / busca sem repeticao',kind:'training',binary:'axon-uic-nursery-lab',telemetry:'durable-reservation-mutation-growth-and-selection'},
  {id:'constructive',label:'DNA construtivo / novos circuitos',kind:'training',binary:'axon-uic-constructive-lab',telemetry:'durable-graph-mutation-growth-and-prospective-validation'},
  {id:'curriculum',label:'Curriculo JSON / matematica e texto',kind:'training',binary:'axon-uic-curriculum-lab',telemetry:'typed-stages-training-admission-transfer-and-paged-birth'},
  {id:'integration',label:'Heranca e fluxo continuo / integrado',kind:'training',binary:'axon-uic-curriculum-lab',telemetry:'bounded-policy-inheritance-and-prequential-sensor-feedback'},
  {id:'structural',label:'Linguagem e calculo simbolico / estrutural',kind:'training',binary:'axon-uic-curriculum-lab',telemetry:'example-induced-tree-rules-dialogue-and-symbolic-transfer'},
  {id:'tests',label:'Testes e verificacao',kind:'tests',binary:null,telemetry:'compiler-json-and-harness-output'},
];
export function validateLabOptions(input) {
  if (input.mode==='tests') {
    if (['count','strategy','curriculum','validation'].some(key=>input[key]!==undefined)) throw new Error('Training options do not apply to tests');
    const suite=input.suite??'rust';
    if(!['rust','rust-minimal','ui','check'].includes(suite))throw new Error('Unknown test suite');
    return {suite};
  }
  if (input.suite!==undefined)throw new Error('Test suite does not apply to training');
  if(input.mode==='structural'){
    if(['seed','count','strategy','validation'].some(k=>input[k]!==undefined))throw new Error('Structural induction uses a declared JSON protocol, not an RNG or candidate count');
    const curriculum=input.curriculum??'compositional-v1';
    if(typeof curriculum!=='string'||!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(curriculum))throw new Error('Invalid structural curriculum id');
    return {curriculum};
  }
  if(!['seed','controller','genome','language','cumulative','semantic'].includes(input.mode)&&input.validation!==undefined)throw new Error('Validation protocol does not apply to this learner');
  const count=input.count??1000;
  const minimum=input.mode==='continuous'?1000:100;
  if(!Number.isInteger(count)||count<minimum||count>10000)throw new Error(`Count must be ${minimum}..10000`);
  if(input.mode==='integration'){
    if(['strategy','curriculum','validation'].some(k=>input[k]!==undefined))throw new Error('Integration has a fixed experimental protocol');
    return {count};
  }
  if(input.mode==='curriculum'){
    if(input.strategy!==undefined)throw new Error('Strategy is declared by the JSON curriculum');
    const curriculum=input.curriculum??'mixed-128-v1';
    if(typeof curriculum!=='string'||!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(curriculum))throw new Error('Invalid curriculum id');
    return {count,curriculum};
  }
  if(input.mode==='constructive'){
    if(input.strategy===undefined&&input.curriculum===undefined)return {count};
    const strategy=input.strategy??'elitist',curriculum=input.curriculum??'legacy';
    if(!['elitist','diverse','reinforced'].includes(strategy))throw new Error('Unknown constructive strategy');
    if(!['legacy','arithmetic'].includes(curriculum))throw new Error('Unknown constructive curriculum');
    return {count,strategy,curriculum};
  }
  if(input.mode!=='evolution'&&['strategy','curriculum'].some(key=>input[key]!==undefined))throw new Error('Evolution options do not apply to this learner');
  if(['seed','controller','genome','language','cumulative','semantic'].includes(input.mode)){
    const validation=input.validation??'development';if(!['development','reserved'].includes(validation))throw new Error('Unknown seed protocol');return {count,validation};
  }
  const strategy=input.strategy??'balanced',curriculum=input.curriculum??'transfer';
  if(!['balanced','hybrid','mutation-only','no-inheritance'].includes(strategy))throw new Error('Unknown evolutionary strategy');
  if(!['core','transfer'].includes(curriculum))throw new Error('Unknown curriculum');
  return {count,...(input.mode==='evolution'?{strategy,curriculum}:{})};
}
export function stopWorker(child) {
  if(!child?.pid||child.exitCode!==null)return;
  if(process.platform==='win32'){
    const killer=spawn('taskkill',['/PID',String(child.pid),'/T','/F'],{windowsHide:true,shell:false,stdio:'ignore'});
    killer.on('error',()=>child.kill());
  } else {try {process.kill(-child.pid,'SIGTERM');}catch{child.kill();}}
}
export function workflowCommand(root,output,options) {
  const workflow=workflows.find(w=>w.id===options.mode);
  if(!workflow)throw new Error('Unknown workflow');
  if(options.mode==='curriculum')return {program:'bun',args:[path.join(root,'ui/curriculum.mjs'),path.join(root,'evals/curricula',`${options.curriculum}.json`),output,String(options.seed),String(options.count)],cwd:root};
  if(options.mode==='integration')return {program:path.join(root,'target','release',`axon-uic-curriculum-lab${process.platform==='win32'?'.exe':''}`),args:['exercise',output],cwd:root};
  if(options.mode==='structural')return {program:path.join(root,'target','release',`axon-uic-curriculum-lab${process.platform==='win32'?'.exe':''}`),args:['structure',path.join(root,'evals/curricula',`${options.curriculum}.json`),output],cwd:root};
  if(workflow.binary)return {program:path.join(root,'target','release',`${workflow.binary}${process.platform==='win32'?'.exe':''}`),args:[output,String(options.seed),String(options.count),...(options.mode==='evolution'||options.mode==='constructive'&&options.strategy?[options.strategy,options.curriculum]:['seed','controller','genome','language','cumulative','semantic'].includes(options.mode)?[options.validation]:[])],cwd:root};
  if(options.suite==='ui')return {program:'bun',args:['run','test'],cwd:path.join(root,'ui')};
  return {program:'cargo',args:options.suite==='check'?['clippy','--features','curriculum','--all-targets','--message-format=json','--','-D','warnings']:['test','--all-targets','--message-format=json','--color','never',...(options.suite==='rust-minimal'?['--no-default-features']:['--features','curriculum']),'--','--color','never'],cwd:root};
}
export async function executeWorkflow({root,output,job,options,publish}) {
  const command=workflowCommand(root,output,options);
  publish({event:'stage',method:'Workflow::spawn',detail:`${options.mode} / ${options.strategy??options.suite??'default'}`,telemetry:job.workflow.telemetry});
  const child=spawn(command.program,command.args,{cwd:command.cwd,env:{...process.env,AXON_LAB_TELEMETRY:'1',...(options.mode==='nursery'?{AXON_NURSERY_ROOT:path.join(path.dirname(path.dirname(path.dirname(output))),'nursery')}:{}) ,...(options.mode==='constructive'?{AXON_CONSTRUCTIVE_ROOT:path.join(path.dirname(path.dirname(path.dirname(output))),'constructive')}:{}) ,...(options.mode==='curriculum'?{AXON_CURRICULUM_ROOT:path.join(path.dirname(path.dirname(path.dirname(output))),'curricula')}:{})},windowsHide:true,shell:false,detached:process.platform!=='win32',stdio:['ignore','pipe','pipe']});
  job.process=child;
  const timeout=setTimeout(()=>{job.timedOut=true;publish({event:'error',detail:'Workflow exceeded the 10 minute wall-time quota'});stopWorker(child);},600000);timeout.unref();
  return new Promise((resolve,reject)=>{
    let pending='',recordLimit=false;
    const line=text=>{
      if(!text)return;
      let record;try{record=JSON.parse(text);}catch{}
      if(record?.experiment&&typeof record.event==='string'){
        if(record.status!==undefined){record.outcome=record.status;delete record.status;}
        publish(record);
      }else if(record?.reason){publish({event:'build',method:'Cargo',detail:`${record.reason} / ${record.target?.name??''}`,record});if(record.reason==='compiler-message'&&record.message?.rendered)publish({event:'process_stderr',text:record.message.rendered,detail:record.message.message});}
      else publish({event:options.mode==='tests'?'test_output':'process_output',text,detail:text.slice(0,180)});
    };
    child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
    child.stdout.on('data',chunk=>{
      pending+=chunk;
      let index;while((index=pending.indexOf('\n'))!==-1){const row=pending.slice(0,index).replace(/\r$/,'');pending=pending.slice(index+1);if(Buffer.byteLength(row)>512000){recordLimit=true;pending='';stopWorker(child);publish({event:'error',detail:'Workflow record limit'});return;}line(row);}
      if(Buffer.byteLength(pending)>512000){recordLimit=true;pending='';stopWorker(child);publish({event:'error',detail:'Workflow record limit'});}
    });
    child.stderr.on('data',text=>publish({event:'process_stderr',text:text.slice(0,32000),detail:text.slice(0,180)}));
    child.on('error',error=>{clearTimeout(timeout);reject(error);});
    child.on('close',code=>{clearTimeout(timeout);if(pending&&!recordLimit)line(pending);resolve(job.timedOut?124:recordLimit?1:code??1);});
  });
}
