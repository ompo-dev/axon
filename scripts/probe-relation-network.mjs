import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';
const [run,corpus,outputPath]=process.argv.slice(2);if(!outputPath)throw Error('usage: node scripts/probe-relation-network.mjs <run> <corpus> <new-output>');
const output=path.resolve(outputPath);await fs.mkdir(output);const write=(n,x)=>fs.writeFile(path.join(output,n),JSON.stringify(x,null,2),{flag:'wx'});
const json=async p=>JSON.parse(await fs.readFile(p));const sha=b=>createHash('sha256').update(b).digest('hex');const freeze=await json(path.join(run,'freeze.json')),selection=await json(path.join(run,'selection.json'));
const binary=path.join(output,'candidate.exe');await fs.copyFile(path.join(run,'candidate.exe'),binary,fs.constants.COPYFILE_EXCL);assert.equal(sha(await fs.readFile(binary)),freeze.binary_sha256);
const source={};for(const s of ['train','test']){const bytes=await fs.readFile(path.join(corpus,`${s}.csv`));assert.equal(sha(bytes),freeze.corpus[s]);source[s]=parse(bytes,{columns:true,bom:true});await write(`${s}-inputs.json`,source[s].map((r,i)=>({id:`${s}-${i}`,premise:r.premise,hypothesis:r.hypothesis})));}
for(const p of selection.policies){const bytes=await fs.readFile(path.join(run,`policy-${p.index}/structural.axnum`));assert.equal(sha(bytes),p.number_sha256);await fs.writeFile(path.join(output,`policy-${p.index}.axnum`),bytes,{flag:'wx'});}
await fs.mkdir(path.join(output,'source/scripts'),{recursive:true});await fs.copyFile(process.argv[1],path.join(output,'source/scripts/probe-relation-network.mjs'),fs.constants.COPYFILE_EXCL);
await write('freeze.json',{binary_sha256:freeze.binary_sha256,source_freeze_sha256:sha(await fs.readFile(path.join(run,'freeze.json'))),driver_sha256:sha(await fs.readFile(process.argv[1])),driver_snapshot:'source/scripts/probe-relation-network.mjs',selected:selection.selected,tests:selection.policies.map(p=>p.index),fit:[selection.selected],scope:'Post-selection diagnostic ablations on already consumed test. No reselection, no new training, no fresh-reserve claim. Fit uses only already taught training examples.'});
async function evaluate(name,index,split){
  const started=performance.now();const result=await promisify(execFile)(binary,['relation-network-batch',path.join(output,`policy-${index}.axnum`),path.join(output,`${name}-birth`),path.join(output,`${split}-inputs.json`),path.join(output,`${name}-predictions.json`)],{cwd:output,windowsHide:true,timeout:600000,maxBuffer:64000000});
  await fs.writeFile(path.join(output,`${name}.stdout.log`),result.stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),result.stderr,{flag:'wx'});
  const rows=await json(path.join(output,`${name}-predictions.json`));assert.equal(rows.length,source[split].length);const summary=JSON.parse(result.stdout.trim().split(/\r?\n/u).at(-1));assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.peak_active_modules===1);
  const correct=rows.filter((r,i)=>r.prediction===Number(source[split][i].label)).length;const report={policy:index,total:rows.length,correct,summary,whole_process_ms:performance.now()-started};await write(`${name}-score.json`,report);return {report,rows};
}
const outputs=new Map(),reports=[];
for(const p of selection.policies){const {report,rows}=await evaluate(`test-${p.index}`,p.index,'test');outputs.set(p.index,rows);reports.push(report);console.log(JSON.stringify({policy:p.index,total:report.total,correct:report.correct}));}
const fit=await evaluate('fit-selected',selection.selected,'train');
const paired=[];
const pairs=[];
const sameBudget=(a,b)=>['seed','buckets','dimensions','epochs','learning_rate','minimum_margin'].every(k=>a[k]===b[k]);
for(const p of selection.policies){const a=p.training.summary.budget;const modes=a.mode==='bound-graph'?['graph']:a.mode==='graph'?['pair','hypothesis']:[];
  for(const control of selection.policies){const b=control.training.summary.budget;if(modes.includes(b.mode)&&sameBudget(a,b))pairs.push([p.index,control.index]);}}
for(const[graph,control]of pairs){
  const counts={both_correct:0,graph_only:0,control_only:0,both_wrong:0};const a=outputs.get(graph),b=outputs.get(control);
  a.forEach((r,i)=>{const truth=Number(source.test[i].label),x=r.prediction===truth,y=b[i].prediction===truth;counts[x?(y?'both_correct':'graph_only'):(y?'control_only':'both_wrong')]++;});paired.push({graph,control,...counts});
}
const chosen=outputs.get(selection.selected),ranked=chosen.map((r,i)=>({margin:r.details?.margin??-Infinity,correct:r.prediction===Number(source.test[i].label),i})).sort((a,b)=>b.margin-a.margin||a.i-b.i);
const risk=[1,0.95,0.9,0.75,0.5,0.25,0.1].map(coverage=>{const n=Math.ceil(coverage*ranked.length);return {requested_coverage:coverage,answered:n,total:ranked.length,correct:ranked.slice(0,n).filter(r=>r.correct).length,scope:'Uncalibrated margin diagnostic, not a deployed abstention policy or approval.'};});
await write('audit.json',{schema:'axon-relation-network-ablation/v1',tests:reports,fit:fit.report,paired,risk,selected_unchanged:selection.selected,m1_complete:false,scope:'Consumed-test adaptive diagnostic. Strong hypothesis-only performance is evidence of possible annotation artifacts. Differences need independently authored semantic contrasts before any M1 approval.'});console.log(JSON.stringify({fit:fit.report,paired,risk}));
