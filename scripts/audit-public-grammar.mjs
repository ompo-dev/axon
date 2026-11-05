import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import assert from 'node:assert/strict';

const [run,corpus,output]=process.argv.slice(2);
if(!output)throw Error('usage: node scripts/audit-public-grammar.mjs <run> <corpus> <new-output>');
await fs.mkdir(output);
const json=async name=>JSON.parse(await fs.readFile(name,'utf8'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const freeze=await json(path.join(run,'freeze.json')),audit=await json(path.join(run,'audit.json'));
for(const [name,expected] of Object.entries(freeze.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',name))),expected);
const decoding=Array.isArray(audit.results);
let base=freeze;
if(decoding) {
  const protocol=await json(path.join(run,'protocol.json'));
  assert.equal(protocol.schema,'axon-lexical-normalized-decoding/v1');
  base=await json(path.join(protocol.source,'freeze.json'));
  for(const [name,expected] of Object.entries(freeze.hashes))assert.equal(sha(await fs.readFile(path.join(run,name))),expected);
} else {
  assert.equal(sha(await fs.readFile(path.join(run,'candidate.exe'))),freeze.binary_sha256);
  assert.equal(sha(await fs.readFile(path.join(run,'parent.axnum'))),freeze.parent_sha256);
}
const bytes=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
assert.equal(sha(bytes),base.corpus_sha256);
const provenance=await json(path.join(corpus,'provenance.json')),reserved=new Set(provenance.reserved_workers);
assert.equal(sha(await fs.readFile(path.join(corpus,'provenance.json'))),base.provenance_sha256);
const rows=bytes.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse);
const allowed=rows.filter(row=>!reserved.has(row.worker_id)&&['train','dev'].includes(row.partition));
const norm=s=>s.toLowerCase().trim().split(/\s+/u).join(' ').replace(/[.?!]+$/u,'');

// Independent source decoder: read annotations as a stream of bracket fields.
function target(row) {
  const slots=[];let body='',inside=false;
  for(const char of row.annot_utt) {
    if(char==='[') {assert(!inside);inside=true;body='';}
    else if(char===']') {
      assert(inside);const colon=body.indexOf(':');assert(colon>0);
      slots.push([body.slice(0,colon).trim(),norm(body.slice(colon+1))]);inside=false;
    } else if(inside)body+=char;
  }
  assert(!inside);
  return ['intent',row.intent,slots.length?['slots',...slots]:'none'];
}
const train=allowed.filter(r=>r.partition==='train'),dev=allowed.filter(r=>r.partition==='dev');
assert.deepEqual(dev.map(r=>String(r.id)),base.development_ids);
assert.deepEqual(train.map(r=>String(r.id)),base.train_ids);
const truth=await json(path.join(run,decoding?'private-truth.json':'development-truth.json')),inputs=await json(path.join(run,decoding?'inputs.json':'development-inputs.json'));
for(let i=0;i<dev.length;i++) {
  assert.equal(truth[i].id,String(dev[i].id));assert.deepEqual(truth[i].target,target(dev[i]));
  assert.deepEqual(inputs[i],{id:String(dev[i].id),text:dev[i].utt});
}
function score(predictions) {
  assert.equal(predictions.length,dev.length);let exact=0,wrong=0,abstained=0;
  predictions.forEach((r,i)=>{assert.equal(r.id,String(dev[i].id));
    if(r.prediction===null)abstained++;else if(isDeepStrictEqual(r.prediction,target(dev[i])))exact++;else wrong++;});
  return {total:dev.length,exact,wrong,abstained};
}
if(decoding) {
  const protocol=await json(path.join(run,'protocol.json')),plans=await json(path.join(run,'plans.json'));
  assert.equal(plans.length,protocol.models.length*protocol.plans.length);assert.equal(audit.results.length,plans.length);
  const results=[];
  for(const [index,plan] of plans.entries()) {
    const recorded=audit.results[index];assert.deepEqual(recorded.plan,plan);
    const predictions=await json(path.join(run,`${plan.name}-predictions.json`)),raw=[];let work=0;
    for(let start=0;start<inputs.length;start+=protocol.chunk_requests) {
      const name=`${plan.name}-${start}`;
      assert.deepEqual(await json(path.join(run,`${name}-inputs.json`)),{format:plan.format,budget:plan.budget,inputs:inputs.slice(start,start+protocol.chunk_requests)});
      const chunk=await json(path.join(run,`${name}-predictions.json`));raw.push(...chunk);
      const summary=await json(path.join(run,`${name}.stdout.log`));
      assert(summary.seed_unchanged&&summary.teacher_queries===0&&summary.loads===1&&summary.active_modules===1);
      assert.equal(summary.work,chunk.reduce((n,r)=>n+r.work,0));work+=summary.work;
    }
    assert.deepEqual(raw,predictions);assert.equal(recorded.charged_work,work);
    const recount=score(predictions);for(const [key,value]of Object.entries(recount))assert.equal(recorded.score[key],value);
    let intent=0,slots=0;const families={};
    for(const [i,row]of predictions.entries()) {
      const expected=target(dev[i]),family=families[expected[1]]??={total:0,exact:0};family.total++;
      if(row.prediction===null)continue;
      intent+=Number(row.prediction[1]===expected[1]);slots+=Number(isDeepStrictEqual(row.prediction[2],expected[2]));family.exact+=Number(isDeepStrictEqual(row.prediction,expected));
    }
    assert.equal(recorded.score.intent_correct,intent);assert.equal(recorded.score.slots_exact,slots);
    for(const [name,value]of Object.entries(families))for(const key of ['total','exact'])assert.equal(recorded.score.families[name][key],value[key]);
    assert.equal(sha(await fs.readFile(path.join(run,plan.number))),freeze.hashes[plan.number]);
    results.push({plan:plan.name,...recount,intent_correct:intent,slots_exact:slots,minimum_family_accuracy:Math.min(...Object.values(families).map(f=>f.exact/f.total)),work});
  }
  const winner=[...audit.results].sort((a,b)=>b.score.exact-a.score.exact||a.charged_work-b.charged_work||a.plan.source_index-b.plan.source_index||plans.findIndex(p=>p.name===a.plan.name)-plans.findIndex(p=>p.name===b.plan.name))[0];
  assert.equal(audit.selected,winner.plan.name);
  for(const key of ['m1_complete','reserved_evaluated','original_test_evaluated','production_promoted'])assert.equal(audit[key],false);
  const report={schema:'axon-decoder-independent-recount/v1',verified:true,results,selected:audit.selected,source_derived_truth:true,training_presentations:0,m1_complete:false,reserved_evaluated:false,original_test_evaluated:false,production_promoted:false,auditor_sha256:sha(await fs.readFile(process.argv[1])),scope:'Independent annotation decoder, same engineering author. Recounts every recorded chunk and charged work; no new inference, fresh reserve or language certification.'};
  await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));process.exit(0);
}
const policies=[];
for(const p of audit.policies) {
  const name=`policy-${p.index}`,rows=await json(path.join(run,`${name}-predictions.json`));
  const recount=score(rows);for(const [key,value] of Object.entries(recount))assert.equal(p.score[key],value);
  const parentPage=await fs.readFile(path.join(run,`${name}-parent-birth/structural.axstruct`));
  assert.deepEqual(parentPage,await fs.readFile(path.join(run,name,'structural.axstruct')));
  assert.equal(sha(await fs.readFile(path.join(run,name,'structural.axnum'))),p.number_sha256);
  policies.push({index:p.index,...recount,rules:p.training.summary.statistics?.selected??p.training.summary.weights,source_derived_truth:true,parent_page_unchanged:true});
}
const winner=[...policies].sort((a,b)=>b.exact-a.exact||a.rules-b.rules||a.index-b.index)[0];
assert.equal(winner.index,audit.selected);
const memory=new Map(),frequencies=new Map();
for(const row of train) {
  const key=norm(row.utt),label=JSON.stringify(target(row)),values=memory.get(key)??new Set();
  values.add(label);memory.set(key,values);frequencies.set(label,(frequencies.get(label)??0)+1);
}
const common=[...frequencies].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))[0][0];
const baselines={};
for(const [name,predict] of [['untrained',()=>null],['constant',()=>JSON.parse(common)],['exact_retrieval',r=>{const values=memory.get(norm(r.utt));return values?.size===1?JSON.parse([...values][0]):null;}]]) {
  baselines[name]=score(dev.map(r=>({id:String(r.id),prediction:predict(r)})));
  for(const [key,value] of Object.entries(baselines[name]))assert.equal(audit.baselines[name][key],value);
}
const report={schema:'axon-grammar-independent-recount/v1',run,verified:true,selected:winner.index,policies,baselines,
  training:train.length,development:dev.length,reserved_evaluated:false,original_test_evaluated:false,m1_complete:false,production_promoted:false,
  scope:'Separate source decoder and scorer, same engineering author. Recounts recorded real-core predictions; not another independent corpus, a fresh reserve, an execution rerun, or full M1 approval.'};
report.auditor_sha256=sha(await fs.readFile(process.argv[1]));
for(const name of ['grammar','sequence','compact'])if(audit.baselines[`previous_${name}`]) {
  const previous=await json(path.join(run,`previous-${name}-predictions.json`));
  const provenance=await json(path.join(run,`previous-${name}-source.json`));
  assert.equal(sha(await fs.readFile(path.join(run,`previous-${name}-predictions.json`))),provenance.sha256);
  report.baselines[`previous_${name}`]=score(previous);
  for(const [key,value] of Object.entries(report.baselines[`previous_${name}`]))assert.equal(audit.baselines[`previous_${name}`][key],value);
}
let gates;
try {gates=await json(path.join(run,'milestone-gates.json'));}
catch(error) {if(error.code!=='ENOENT')throw error;}
if(gates) {
  const contractBytes=await fs.readFile(path.join(run,'source/evals/protocols/m1-completion-v1.json'));
  assert.equal(sha(contractBytes),gates.completion_contract_sha256);
  const contract=JSON.parse(contractBytes),language=contract.gates.find(g=>g.id==='external-language');
  assert(language?.required);
  const predictions=await json(path.join(run,`policy-${winner.index}-predictions.json`));
  let intent=0,answered=0;const families={};
  predictions.forEach((row,i)=>{
    const expected=target(dev[i]),family=families[expected[1]]??={total:0,exact:0};family.total++;
    if(row.prediction===null)return;
    answered++;intent+=Number(Array.isArray(row.prediction)&&row.prediction[0]==='intent'&&row.prediction[1]===expected[1]);
    family.exact+=Number(isDeepStrictEqual(row.prediction,expected));
  });
  const observed={intent_accuracy:intent/dev.length,exact_structured_accuracy:winner.exact/dev.length,
    coverage:answered/dev.length,per_family_accuracy:Math.min(...Object.values(families).map(f=>f.exact/f.total))};
  const checks=Object.fromEntries(Object.entries(observed).map(([key,value])=>[key,{observed:value,minimum:language[`minimum_${key}`],passed:value>=language[`minimum_${key}`]}]));
  assert(Object.values(checks).every(c=>Number.isFinite(c.minimum)));
  const expectedGates=contract.gates.map(g=>({id:g.id,required:g.required,status:g.id==='external-language'?
    (Object.values(checks).every(c=>c.passed)?'awaiting-frozen-reserved-audit':'failed-development'):'not-demonstrated-by-this-experiment',
    ...(g.id==='external-language'?{checks}:{})}));
  assert.deepEqual(gates.gates,expectedGates);
  assert.equal(gates.selected,winner.index);assert.equal(gates.binary_sha256,freeze.binary_sha256);
  assert.equal(gates.number_sha256,audit.policies.find(p=>p.index===winner.index).number_sha256);
  for(const key of ['reserved_evaluated','original_test_evaluated','m1_complete','m2_permitted','production_promoted'])assert.equal(gates[key],false);
  report.milestone_gates_recounted=true;report.development_checks=checks;
}
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify(report));
