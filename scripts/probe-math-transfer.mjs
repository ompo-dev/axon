import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {evaluate} from './math-program-source.mjs';

const [outputPath,protocolFile='evals/protocols/m1-math-library-transfer-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/probe-math-transfer.mjs <new-output> [protocol]');
const output=path.resolve(outputPath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const protocol=JSON.parse(await fs.readFile(protocolFile));assert.equal(protocol.schema,'axon-math-library-transfer/v1');
const run=path.resolve(protocol.source_study),sha=b=>createHash('sha256').update(b).digest('hex');
const write=(file,data)=>fs.writeFile(path.join(output,file),JSON.stringify(data,null,2),{flag:'wx'});
const library=JSON.parse(await fs.readFile(path.join(run,'library.json'))),programs=JSON.parse(await fs.readFile(path.join(run,'programs.json')));
assert.equal(library.filter(p=>p.admitted).length,27);
for(const id of protocol.targets)assert(library.some(p=>p.id===id&&!p.admitted&&p.reason==='bounded numeric induction refused'));
const binary=path.join(output,'candidate.exe');await fs.copyFile(path.join(run,'main.exe'),binary,fs.constants.COPYFILE_EXCL);
const sources={};for(const file of [protocolFile,'scripts/probe-math-transfer.mjs','scripts/math-program-source.mjs','src/numeric.rs','src/numeric_library.rs','src/numeric_cli.rs','src/numeric_constructive.rs']){const bytes=await fs.readFile(file);sources[file]=sha(bytes);const target=path.join(output,'source',file);await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,bytes,{flag:'wx'});}
for(const id of protocol.targets){for(const kind of ['teaching','admission'])await fs.copyFile(path.join(run,id,`${kind}.tsv`),path.join(output,`${id}-${kind}.tsv`),fs.constants.COPYFILE_EXCL);}
const manifest=['skill\tcontext'],ancestral={};
for(const p of library.filter(p=>p.admitted)){const bytes=await fs.readFile(path.join(run,p.id,'skill.axnum'));ancestral[p.id]=sha(bytes);const file=`ancestor-${p.id}.axnum`;await fs.writeFile(path.join(output,file),bytes,{flag:'wx'});manifest.push(`${file}\tmath/annotation-program-v1`);}
await fs.writeFile(path.join(output,'sources.tsv'),manifest.join('\n')+'\n',{flag:'wx'});
await write('freeze.json',{sources,ancestral,binary_sha256:sha(await fs.readFile(binary)),programs_sha256:sha(await fs.readFile(path.join(run,'programs.json'))),seed:20261023,scope:'Source-only program recomposition. No target expressions, questions or expected development values given to AXON search; annotated formulas used by the separate numeric probe evaluator only.'});
const calls=[];
async function core(name,args,refusal=false){let stdout='',stderr='',code=0;const t=performance.now();try{({stdout,stderr}=await promisify(execFile)(binary,args,{windowsHide:true,timeout:300000,maxBuffer:8000000}));}catch(e){stdout=e.stdout??'';stderr=e.stderr??'';code=e.code;assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal);}await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});const call={order:calls.length,name,arguments:args,exit_code:code,whole_process_ms:performance.now()-t};calls.push(call);await write(`call-${call.order}.json`,call);assert(code===0||refusal,stderr);return {code,stdout,call};}
const mined=path.join(output,'library.axlib'),mining=await core('mine',['numeric-mine',path.join(output,'sources.tsv'),'--context','math/source-library-v1','--library',mined]);
const results=[];let state=20261023;const pool=[-31,-13,-3,-.5,.25,2,7,19,37];
for(const id of protocol.targets){
  const teacher=programs.find(p=>p.id===id),valid=[],domain=[];
  for(let j=0;j<128;j++){const input=Array.from({length:teacher.arity},()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return pool[state%pool.length];});try{valid.push({input,output:evaluate(teacher.tree,input)});}catch(e){domain.push({input,error:e.message});}}
  const probe=path.join(output,`${id}-probe.tsv`);await fs.writeFile(probe,[...Array.from({length:teacher.arity},(_,i)=>`x${i}`),'y'].join('\t')+'\n'+valid.map(r=>[...r.input,r.output].join('\t')).join('\n')+'\n',{flag:'wx'});
  for(const attempt of protocol.attempts){
    const name=`${id}-${attempt.name}`,skill=path.join(output,`${name}.axnum`),context='math/annotation-program-v1';
    const args=[attempt.command,path.join(output,`${id}-teaching.tsv`),'--name',id,'--context',context,'--skill',skill,'--mined-library',mined,'--library-context','math/source-library-v1','--max-cost',String(attempt.max_cost),'--max-terms',String(attempt.max_terms)];
    if(attempt.command==='numeric-develop')args.push('--max-attempts',String(attempt.max_attempts),'--total-work',String(attempt.total_work));
    const learned=await core(`${name}-learn`,args,true);const result={id,attempt:attempt.name,learn_call:learned.call.order,learned:learned.code===0,admitted:false,finite_probe_cases:valid.length,unexecuted_target_domain_cases:domain.length};
    if(!learned.code){
      const admission=await core(`${name}-admission`,['numeric-eval',skill,path.join(output,`${id}-admission.tsv`),'--context',context],true);
      const observed=await core(`${name}-probe`,['numeric-eval',skill,probe,'--context',context],true);result.admission_call=admission.call.order;result.probe_call=observed.call.order;result.admitted=!admission.code&&!observed.code;
      if(result.admitted){const exported=await core(`${name}-export`,['circuit-export',skill,'--context',context,'--output',path.join(output,`${name}-exported`)]);const born=await core(`${name}-dna`,['circuit-predict',path.join(output,`${name}-exported/circuit-number.txt`),valid[0].input.join(',')]);const fields=Object.fromEntries(born.stdout.trim().split(/\r?\n/u).filter(s=>s.includes('=')).map(s=>{const at=s.indexOf('=');return [s.slice(0,at),s.slice(at+1)];}));assert(Math.abs(Number(fields.value)-valid[0].output)<=1e-9*(1+Math.abs(valid[0].output)));result.export_call=exported.call.order;result.portability_call=born.call.order;}
    }
    results.push(result);
  }
}
await write('audit.json',{schema:'axon-math-library-transfer-study/v1',results,calls,mining_call:mining.call.order,ancestral_programs:27,development_labels_read:false,source_study_changed:false,m1_complete:false,m2_permitted:false,scope:'Original four primitive-only failures, existing learned abstraction miner and bounded I/O synthesis, same fixed numerical admission plus signed/fractional checks. Teacher formulas from allowed train annotations, not scientific discoveries; cost/information not equivalent to primitive-only search.'});
console.log(JSON.stringify({results,calls:calls.length,m1_complete:false}));
