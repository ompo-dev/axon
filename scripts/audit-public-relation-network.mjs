import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';
const [run,corpus,output]=process.argv.slice(2);
if(!output)throw Error('usage: node scripts/audit-public-relation-network.mjs <run> <corpus> <new-output>');
await fs.mkdir(output);
const json=async p=>JSON.parse(await fs.readFile(p));const sha=b=>createHash('sha256').update(b).digest('hex');
const freeze=await json(path.join(run,'freeze.json')),selection=await json(path.join(run,'selection.json')),audit=await json(path.join(run,'audit.json'));
assert.equal(sha(await fs.readFile(path.join(run,'candidate.exe'))),freeze.binary_sha256);
for(const[p,h]of Object.entries(freeze.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',p))),h);
const source={};for(const name of ['train','val','test']){const raw=await fs.readFile(path.join(corpus,`${name}.csv`));assert.equal(sha(raw),freeze.corpus[name]);source[name]=parse(raw,{columns:true,bom:true});}
function recount(rows,split){
  assert.equal(rows.length,source[split].length);let correct=0,answered=0;const matrix=Array.from({length:3},()=>[0,0,0,0]);
  rows.forEach((r,i)=>{assert.equal(r.id,`${split}-${i}`);assert(r.prediction===null||[0,1,2].includes(r.prediction));matrix[Number(source[split][i].label)][r.prediction??3]++;answered+=Number(r.prediction!==null);correct+=Number(r.prediction===Number(source[split][i].label));});
  return {correct,answered,total:rows.length,matrix};
}
const policies=[];
const axiomSource=await json(path.join(run,'supervised-axioms.json'));
const trainStatements=new Set(source.train.flatMap(r=>[r.premise,r.hypothesis]));
for(const p of selection.policies){
  const prefix=`policy-${p.index}`;const inputs=await json(path.join(run,`${prefix}-validation-inputs.json`));
  const teaching=await json(path.join(run,`${prefix}-train.json`));
  assert.equal(teaching.examples.length,source.train.length+axiomSource.length);
  assert.deepEqual(teaching.examples.slice(0,source.train.length),source.train.map(r=>({premise:r.premise,hypothesis:r.hypothesis,label:Number(r.label)})));
  const seen=new Set();
  teaching.examples.slice(source.train.length).forEach((r,i)=>{assert.equal(r.premise,r.hypothesis);assert.equal(r.label,1);assert(trainStatements.has(r.premise));assert(!seen.has(r.premise));seen.add(r.premise);assert.equal(sha(r.premise),axiomSource[i].source_statement_sha256);assert.equal(axiomSource[i].source_partition,'train');});
  assert.deepEqual(inputs,source.val.map((r,i)=>({id:`val-${i}`,premise:r.premise,hypothesis:r.hypothesis})));
  const rows=await json(path.join(run,`${prefix}-validation-predictions.json`));const result=recount(rows,'val');
  for(const key of Object.keys(result))assert.deepEqual(result[key],p.validation.metrics[key]);
  assert.equal(sha(await fs.readFile(path.join(run,prefix,'structural.axnum'))),p.number_sha256);
  for(const name of ['structural.axstruct','learned-dependency-v1.axseq'])assert.deepEqual(await fs.readFile(path.join(run,prefix,name)),await fs.readFile(path.join(run,`${prefix}-parent-birth`,name)));
  const summary=p.validation.summary;assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.peak_active_modules===1);
  const graph=['graph','bound-graph'].includes(p.training.summary.budget.mode);
  assert.equal(summary.loads,graph?1+2*rows.length:1);
  assert.equal(summary.evictions,graph?2*rows.length:0);
  assert(rows.every(r=>r.syntax_used===graph&&(graph||r.syntax_available===null)));
  policies.push({index:p.index,...result,page_bytes:p.training.summary.page_bytes,mode:p.training.summary.budget.mode,loads:summary.loads,one_active_page:true});
}
const best=[...policies].sort((a,b)=>b.correct-a.correct||a.page_bytes-b.page_bytes||a.index-b.index)[0];assert.equal(best.index,selection.selected);assert.equal(best.index,audit.selected);
const testInputs=await json(path.join(run,'selected-test-inputs.json'));assert.deepEqual(testInputs,source.test.map((r,i)=>({id:`test-${i}`,premise:r.premise,hypothesis:r.hypothesis})));
const result=recount(await json(path.join(run,'selected-test-predictions.json')),'test');for(const key of Object.keys(result))assert.deepEqual(result[key],audit.test.metrics[key]);
const report={schema:'axon-relation-network-recount/v1',policies,selected:best.index,test:result,source_truth_recounted:true,no_inference_target_fields:true,one_active_page_verified:true,m1_complete:false,scope:'Separate scorer, same engineering author. Previously consumed InferBR test, not independent M1 certification.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
