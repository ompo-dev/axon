import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {evaluate} from './math-program-source.mjs';

const [run,output]=process.argv.slice(2);if(!output)throw Error('usage: node scripts/probe-math-library.mjs <study> <new-output>');
await fs.mkdir(output);const sha=b=>createHash('sha256').update(b).digest('hex');
const library=JSON.parse(await fs.readFile(path.join(run,'library.json'))),programs=JSON.parse(await fs.readFile(path.join(run,'programs.json')));
const binary=path.resolve(output,'candidate.exe');await fs.copyFile(path.join(run,'main.exe'),binary,fs.constants.COPYFILE_EXCL);
const source={};for(const file of ['scripts/probe-math-library.mjs','scripts/math-program-source.mjs',path.join(run,'library.json'),path.join(run,'programs.json')])source[file]=sha(await fs.readFile(file));
await fs.writeFile(path.join(output,'freeze.json'),JSON.stringify({source,binary_sha256:sha(await fs.readFile(binary)),seed:20261022,values:[-31,-13,-3,-.5,.25,2,7,19,37],attempts_per_program:128,scope:'Independent future signed/fractional numerical tuples, before tests. Targets come from train-annotated formulas, not new mathematical discovery or development labels. Domain-failing target tuples recorded, not submitted to numeric-eval which requires finite labels.'},null,2),{flag:'wx'});
const results=[];let state=20261022;const pool=[-31,-13,-3,-.5,.25,2,7,19,37];
for(const p of library.filter(p=>p.admitted)){
  const teacher=programs.find(t=>t.id===p.id),rows=[],domain=[];
  for(let j=0;j<128;j++){
    const input=Array.from({length:teacher.arity},()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return pool[state%pool.length];});
    try{rows.push({input,expected:evaluate(teacher.tree,input)});}catch(error){domain.push({input,expected_refusal:error.message});}
  }
  const skill=path.join(output,`${p.id}.axnum`);await fs.copyFile(path.join(run,p.id,'skill.axnum'),skill,fs.constants.COPYFILE_EXCL);
  const file=path.join(output,`${p.id}.tsv`);await fs.writeFile(file,[...Array.from({length:teacher.arity},(_,i)=>`x${i}`),'y'].join('\t')+'\n'+rows.map(r=>[...r.input,r.expected].join('\t')).join('\n')+'\n',{flag:'wx'});
  let stdout='',stderr='',code=0;try{({stdout,stderr}=await promisify(execFile)(binary,['numeric-eval',path.resolve(skill),path.resolve(file),'--context','math/annotation-program-v1'],{windowsHide:true,timeout:300000,maxBuffer:1000000}));}
  catch(e){stdout=e.stdout??'';stderr=e.stderr??'';code=e.code;assert(Number.isInteger(code)&&code>0&&!e.killed&&!e.signal);}
  await fs.writeFile(path.join(output,`${p.id}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${p.id}.stderr.log`),stderr,{flag:'wx'});
  const result={id:p.id,finite_target_cases:rows.length,domain_target_cases:domain.length,domain_targets_executed:false,domain,exit_code:code,passed:code===0,source_skill_sha256:sha(await fs.readFile(skill))};results.push(result);
}
const audit={schema:'axon-math-library-signed-recount/v1',programs:results.length,passed:results.filter(r=>r.passed).length,results,finite_cases:results.reduce((n,r)=>n+r.finite_target_cases,0),domain_cases_not_executed:results.reduce((n,r)=>n+r.domain_target_cases,0),original_admission_or_models_changed:false,m1_complete:false,m2_permitted:false,scope:'Numerical formula fidelity after the original8 teaching+8 admission tuples. Negative findings do not retroactively remove programs from the frozen public study. No public linguistic claim or domain-refusal success for unexecuted target-domain cases.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(audit,null,2),{flag:'wx'});console.log(JSON.stringify({programs:audit.programs,passed:audit.passed,finite_cases:audit.finite_cases,domain_cases_not_executed:audit.domain_cases_not_executed,failed:results.filter(r=>!r.passed).map(r=>r.id)}));
