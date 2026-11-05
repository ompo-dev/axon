import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { prepareCurriculum } from '../ui/curriculum.mjs';

const root=path.resolve(process.argv[2]);
const json=async name=>JSON.parse(await fs.readFile(path.join(root,name),'utf8'));
const source=await json('curriculum.json'),prepared=await json('prepared.json'),result=await json('curriculum-results.json');
const rng=Number(process.argv[3]),cap=Number(process.argv[4]??5000);
// JSON canonicalizes -0 to 0 at the actual teacher/worker boundary.
assert.deepEqual(prepared,JSON.parse(JSON.stringify(prepareCurriculum(source,rng,cap))));
assert.equal(result.stages.length,prepared.stages.length);
const hashes={};
const hash=async name=>hashes[name]=createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex');
const matches=(actual,expected)=>typeof expected==='string'?actual===expected:typeof actual==='number'&&Number.isFinite(actual)&&Math.abs(actual-expected)<=1e-9+1e-9*Math.abs(expected);
const domains={};
for(const [i,stage] of prepared.stages.entries()){
  const outcome=result.stages[i],copy=await json(`${stage.id}-result.json`),predictions=await json(`${stage.id}-predictions.json`);
  assert.deepEqual(copy,outcome);assert.equal(outcome.id,stage.id);assert.equal(outcome.kind,stage.kind);
  const domain=domains[stage.kind]??={stages:0,admitted:0,complete:0,correct:0,wrong:0,unavailable:0,test_correct:0,test_total:0,transfer_correct:0,transfer_total:0};
  domain.stages++;domain.admitted+=Number(outcome.admitted);
  domain.complete+=Number(outcome.admitted&&outcome.test_correct===outcome.test_total&&outcome.transfer_correct===outcome.transfer_total);
  assert.equal(predictions.length,stage.test.length+stage.transfer.length);
  let at=0;
  for(const phase of ['test','transfer']){
    let correct=0;
    for(const [row,example] of stage[phase].entries()){
      const p=predictions[at++];assert.equal(p.phase,phase);assert.equal(p.row,row);
      assert.equal(p.correct,matches(p.actual,example.output));correct+=Number(p.correct);
      domain[p.correct?'correct':p.actual===null?'unavailable':'wrong']++;
      if(!outcome.admitted)assert.equal(p.actual,null);
    }
    assert.equal(outcome[`${phase}_correct`],correct);assert.equal(outcome[`${phase}_total`],stage[phase].length);
    domain[`${phase}_correct`]+=correct;domain[`${phase}_total`]+=stage[phase].length;
  }
  if(outcome.admitted){assert(outcome.model_bytes>0);const bytes=await fs.readFile(path.join(root,outcome.artifact));assert.equal(bytes.length,outcome.model_bytes);await hash(outcome.artifact);}
  await hash(`${stage.id}-result.json`);await hash(`${stage.id}-predictions.json`);
}
const index=await json('model-index.json');assert.deepEqual(index.pages,result.pages);
const modules=new Set();let decimalBytes=0;
for(const page of result.pages){
  assert.equal(page.initial_loads,0);assert.equal(page.active_modules,1);
  const decimal=await fs.readFile(path.join(root,page.file),'utf8');assert(/^1(?:[0-9]{3})+$/.test(decimal));
  assert.equal(decimal.length,page.encoded_bytes*3+1);decimalBytes+=decimal.length;
  for(let i=1;i<decimal.length;i+=3)assert(Number(decimal.slice(i,i+3))<=255);
  for(const id of page.modules){assert(!modules.has(id));modules.add(id);assert(result.stages.some(s=>s.id===id&&s.admitted));}
  assert.equal(page.retention_checks,page.modules.reduce((n,id)=>{const s=prepared.stages.find(s=>s.id===id);return n+s.train.length+s.admission.length;},0));
  if(page.post_selection_rechecks!==undefined)assert.equal(page.post_selection_rechecks,page.modules.reduce((n,id)=>{const s=prepared.stages.find(s=>s.id===id);return n+s.test.length+s.transfer.length;},0));
  await hash(page.file);
}
assert.equal(modules.size,result.stages.filter(s=>s.admitted).length);
assert.equal(result.new_candidates,result.stages.reduce((n,s)=>n+s.new_candidates,0));
assert(result.new_candidates<=prepared.budget.total_candidates);assert(result.numeric_work<=prepared.budget.total_numeric_work);
for(const name of ['curriculum.json','prepared.json','curriculum-results.json','model-index.json'])await hash(name);
const audit={schema:'axon-curriculum-audit/v1',verified:true,domains,new_candidates:result.new_candidates,numeric_work:result.numeric_work,process_elapsed_ns:result.process_elapsed_ns,pages:result.pages.length,binary_seed_bytes:result.pages.reduce((n,p)=>n+p.encoded_bytes,0),decimal_seed_bytes:decimalBytes,failures:result.stages.filter(s=>!s.admitted||s.test_correct!==s.test_total||s.transfer_correct!==s.transfer_total).map(s=>({id:s.id,reason:s.reason,admitted:s.admitted,test_correct:s.test_correct,transfer_correct:s.transfer_correct})),hashes};
await fs.writeFile(path.join(root,'audit-curriculum-v1.json'),JSON.stringify(audit,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({...audit,hashes:undefined,failures:undefined}));
