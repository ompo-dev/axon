import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import assert from 'node:assert/strict';

const [run,corpus,output]=process.argv.slice(2);
if(!output)throw Error('usage: bun scripts/audit-public-feedback.mjs <run> <corpus> <new-output>');
await fs.mkdir(output);
const json=async file=>JSON.parse(await fs.readFile(path.join(run,file)));
const sha=b=>createHash('sha256').update(b).digest('hex');
const freeze=await json('freeze.json'),audit=await json('audit.json');
for(const [file,hash] of Object.entries(freeze.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',file))),hash);
assert.equal(sha(await fs.readFile(path.join(run,'candidate.exe'))),freeze.binary_sha256);
assert.equal(sha(await fs.readFile(path.join(run,'parent.axnum'))),freeze.parent_sha256);
const protocolBytes=await fs.readFile(path.join(run,'source',freeze.protocol_path));
assert.equal(sha(protocolBytes),freeze.protocol_sha256);const protocol=JSON.parse(protocolBytes);
assert(['enhanced-sequence-v1','factorized-sequence-v1','lexical-encoder-v1'].includes(protocol.learner));
const lexical=protocol.learner==='lexical-encoder-v1';
const command=lexical?'encoder':protocol.learner==='factorized-sequence-v1'?'factorized':'enhanced';
if(lexical)assert.equal(protocol.chunk_requests,128);
const raw=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
assert.equal(sha(raw),freeze.corpus_sha256);assert.equal(sha(raw),protocol.source_sha256);
const provenanceBytes=await fs.readFile(path.join(corpus,'provenance.json'));
assert.equal(sha(provenanceBytes),freeze.provenance_sha256);
const reserved=new Set(JSON.parse(provenanceBytes).reserved_workers);
const rows=raw.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse)
  .filter(r=>!reserved.has(r.worker_id)&&['train','dev'].includes(r.partition));
const train=rows.filter(r=>r.partition==='train'),dev=rows.filter(r=>r.partition==='dev');
assert.deepEqual(train.map(r=>String(r.id)),freeze.train_ids);
assert.deepEqual(dev.map(r=>String(r.id)),freeze.development_ids);
const norm=s=>s.toLowerCase().trim().split(/\s+/u).join(' ').replace(/[.?!]+$/u,'');
// Separate annotation parser/scorer, without the driver's annotation helpers.
function target(row) {
  const slots=[];let body='',inside=false;
  for(const c of row.annot_utt) {
    if(c==='['){assert(!inside);inside=true;body='';}
    else if(c===']'){assert(inside);const at=body.indexOf(':');assert(at>0);slots.push([body.slice(0,at).trim(),norm(body.slice(at+1))]);inside=false;}
    else if(inside)body+=c;
  }
  assert(!inside);return ['intent',row.intent,slots.length?['slots',...slots]:'none'];
}
function tags(row) {
  const words=[],labels=[];
  for(const match of row.annot_utt.matchAll(/\[([^:[\]]+):([^\[\]]+)\]|([^\[\]]+)/gu)) {
    const tokens=(match[2]??match[3]).toLowerCase().trim().split(/\s+/u).filter(Boolean);
    words.push(...tokens);
    labels.push(...tokens.map((_,i)=>match[1]?`${i?'I':'B'}:${match[1].trim()}`:'O'));
  }
  if(words.length)words[words.length-1]=words.at(-1).replace(/[.?!]+$/u,'');
  if(words.at(-1)===''){words.pop();labels.pop();}
  assert.equal(words.join(' '),norm(row.utt));return labels;
}
function score(predictions,source) {
  assert.equal(predictions.length,source.length);
  let answered=0,exact=0,intent=0,slots=0;const families={};
  for(let i=0;i<source.length;i++) {
    const actual=predictions[i],expected=target(source[i]);assert.equal(actual.id,String(source[i].id));
    const family=families[expected[1]]??={total:0,answered:0,exact:0,wrong:0};family.total++;
    if(actual.prediction===null)continue;
    answered++;family.answered++;const yes=isDeepStrictEqual(actual.prediction,expected);
    exact+=Number(yes);family.exact+=Number(yes);family.wrong+=Number(!yes);
    intent+=Number(actual.prediction[0]==='intent'&&actual.prediction[1]===expected[1]);
    slots+=Number(isDeepStrictEqual(actual.prediction[2],expected[2]));
  }
  return {total:source.length,answered,exact,wrong:answered-exact,abstained:source.length-answered,intent_correct:intent,slots_exact:slots,exact_accuracy:exact/(source.length||1),families};
}
let cursor=0;
async function call(name,command) {
  const stored=await json(`call-${cursor}.json`);assert.deepEqual(stored,audit.calls[cursor]);
  assert.equal(stored.order,cursor++);assert.equal(stored.name,name);assert.equal(stored.command,command);
  const events=(await fs.readFile(path.join(run,`${name}.stdout.jsonl`),'utf8')).trim().split(/\r?\n/u).map(JSON.parse);
  assert.deepEqual(events.pop(),stored.summary);events.forEach((e,i)=>assert.equal(e.coreSeq,i+1));assert.equal(events.length,stored.core_events);
  return {...stored,arguments:stored.arguments.map(p=>p.split(path.sep).join('/'))};
}
async function evaluation(name,source,expected,number) {
  if(lexical) {
    const merged=[],indices=[];
    for(let start=0;start<source.length;start+=protocol.chunk_requests) {
      const part=`${name}-${start}`,rows=source.slice(start,start+protocol.chunk_requests),c=await call(part,`${command}-batch`);
      indices.push(c.order);assert.deepEqual(c.arguments,[number,`${part}-birth`,`${part}-inputs.json`,`${part}-predictions.json`]);
      assert.deepEqual(await json(`${part}-inputs.json`),rows.map(r=>({id:String(r.id),text:r.utt})));
      assert(c.summary.seed_unchanged&&c.summary.teacher_queries===0&&c.summary.loads===1&&c.summary.active_modules===1);
      const predictions=await json(`${part}-predictions.json`);assert.equal(predictions.length,rows.length);merged.push(...predictions);
    }
    assert.deepEqual(indices,expected.calls);assert.deepEqual(merged,await json(`${name}-predictions.json`));
    assert.deepEqual(await json(`${name}-inputs.json`),source.map(r=>({id:String(r.id),text:r.utt})));
    assert.equal(sha(await fs.readFile(path.join(run,number))),expected.number_sha256);
    const result=score(merged,source);assert.deepEqual(result,expected.score);return result;
  }
  const c=await call(name,`${command}-batch`);assert.equal(c.order,expected.call);
  assert.deepEqual(c.arguments,[number,`${name}-birth`,`${name}-inputs.json`,`${name}-predictions.json`]);
  assert.deepEqual(await json(`${name}-inputs.json`),source.map(r=>({id:String(r.id),text:r.utt})));
  assert(c.summary.seed_unchanged&&c.summary.teacher_queries===0);
  assert.equal(sha(await fs.readFile(path.join(run,number))),expected.number_sha256);
  const result=score(await json(`${name}-predictions.json`),source);assert.deepEqual(result,expected.score);return result;
}
let parent='parent.axnum',previousRetention;const hashes=new Set(),recount=[];
for(const [index,prefix] of protocol.prefixes.entries()) {
  const stage=await json(`stage-${index}-score.json`);assert.deepEqual(stage,audit.stages[index]);
  assert.equal(stage.prefix,prefix);assert.equal(stage.index,index);
  assert.equal(stage.parent_sha256,sha(await fs.readFile(path.join(run,parent))));
  let novelty;
  if(index) {
    const window=train.slice(protocol.prefixes[index-1],prefix);
    await evaluation(`stage-${index}-before`,window,stage.before,parent);
    await evaluation(`stage-${index}-ancestor`,window,stage.ancestor,'stage-0/structural.axnum');
    const seen=new Set(train.slice(0,protocol.prefixes[index-1]).map(r=>norm(r.utt)));
    const predictions=await json(`stage-${index}-before-predictions.json`);
    const indices=window.flatMap((r,i)=>seen.has(norm(r.utt))?[]:[i]);
    novelty=score(indices.map(i=>predictions[i]),indices.map(i=>window[i]));
  }
  const c=await call(`stage-${index}-training`,`${command}-${index?'refine':'train'}`);
  assert.equal(c.order,stage.training);
  assert.deepEqual(c.arguments,[parent,`stage-${index}-parent-birth`,`stage-${index}-train.json`,`stage-${index}`]);
  assert(c.summary.parent_unchanged);assert.equal(lexical?!!c.summary.refinement:c.summary.refined,index>0);assert.equal(c.summary.total,prefix);
  if(lexical&&index) {
    const previous=await json(`stage-${index-1}/training.json`);
    assert.equal(c.summary.refinement.inherited_words,previous.words);
    assert.equal(c.summary.refinement.inherited_weights,previous.weights);
    assert.equal(c.summary.refinement.parent_number_unchanged,true);
    assert.equal(c.summary.refinement.optimizer_state_inherited,false);
  }
  const teaching=await json(`stage-${index}-train.json`);assert.deepEqual(teaching.budget,protocol.budget);
  assert.equal(teaching.examples.length,prefix);
  teaching.examples.forEach((e,i)=>{assert.equal(e.text,train[i].utt);assert.equal(e.intent,train[i].intent);assert.deepEqual(e.tags,tags(train[i]));});
  assert.deepEqual(await fs.readFile(path.join(run,`stage-${index}-parent-birth/structural.axstruct`)),await fs.readFile(path.join(run,`stage-${index}/structural.axstruct`)));
  parent=`stage-${index}/structural.axnum`;
  assert.equal(sha(await fs.readFile(path.join(run,parent))),stage.number_sha256);assert(!hashes.has(stage.number_sha256));hashes.add(stage.number_sha256);
  const retention=await evaluation(`stage-${index}-retention`,train.slice(0,protocol.prefixes[0]),stage.retention,parent);
  const retentionRows=await json(`stage-${index}-retention-predictions.json`);
  const transitions={lost:0,gained:0};
  if(previousRetention)retentionRows.forEach((r,i)=>{
    const expected=target(train[i]),before=isDeepStrictEqual(previousRetention[i].prediction,expected),after=isDeepStrictEqual(r.prediction,expected);
    transitions.lost+=Number(before&&!after);transitions.gained+=Number(!before&&after);
  });
  previousRetention=retentionRows;
  const development=await evaluation(`stage-${index}-development`,dev,stage.development,parent);
  if(index)await evaluation(`stage-${index}-after`,train.slice(protocol.prefixes[index-1],prefix),stage.after_feedback,parent);
  recount.push({index,retention:retention.exact,development:development.exact,retention_transitions:transitions,
    ...(novelty?{pre_feedback_unseen_text:{total:novelty.total,exact:novelty.exact,wrong:novelty.wrong,abstained:novelty.abstained}}:{})});
}
await evaluation('final-training-fit',train,audit.final_training_fit,parent);
assert.equal(cursor,audit.calls.length);
assert.equal(audit.sentence_presentations,protocol.prefixes.reduce((a,b)=>a+b,0)*protocol.budget.epochs);
for(const key of ['reserved_evaluated','original_test_evaluated','m1_complete','m2_permitted','production_promoted'])assert.equal(audit[key],false);
assert.equal(audit.rss_bytes,null);assert.equal(audit.energy_joules,null);
const completion=await json('source/evals/protocols/m1-completion-v1.json');
const language=completion.gates.find(g=>g.id==='external-language');assert(language.required);
const final=audit.stages.at(-1).development.score;
const observed={intent_accuracy:final.intent_correct/final.total,exact_structured_accuracy:final.exact/final.total,
  coverage:final.answered/final.total,per_family_accuracy:Math.min(...Object.values(final.families).map(f=>f.exact/f.total))};
const checks=Object.fromEntries(Object.entries(observed).map(([key,value])=>[key,{observed:value,minimum:language[`minimum_${key}`],passed:value>=language[`minimum_${key}`]}]));
assert(Object.values(checks).every(c=>Number.isFinite(c.minimum)));
const seenTexts=new Set(train.map(r=>norm(r.utt)));
const roleValues=new Set(train.flatMap(r=>{const t=target(r);return Array.isArray(t[2])?t[2].slice(1).map(v=>JSON.stringify(v)):[];}));
const signature=r=>{const t=target(r);return JSON.stringify([t[1],Array.isArray(t[2])?t[2].slice(1).map(v=>v[0]):[]]);};
const signatures=new Set(train.map(signature)),strata={};
const finalPredictions=await json(`stage-${audit.stages.length-1}-development-predictions.json`);
for(const [name,include] of [
  ['unseen_text',r=>!seenTexts.has(norm(r.utt))],
  ['unseen_slot_values',r=>{const t=target(r);return Array.isArray(t[2])&&t[2].slice(1).some(v=>!roleValues.has(JSON.stringify(v)));}],
  ['multiword_slots',r=>{const t=target(r);return Array.isArray(t[2])&&t[2].slice(1).some(v=>v[1].includes(' '));}],
  ['unseen_role_signature',r=>!signatures.has(signature(r))],
]) {
  const indices=dev.flatMap((r,i)=>include(r)?[i]:[]),s=score(indices.map(i=>finalPredictions[i]),indices.map(i=>dev[i]));
  strata[name]={total:s.total,exact:s.exact,wrong:s.wrong,abstained:s.abstained};
}
const report={schema:'axon-public-feedback-recount/v1',verified:true,calls:cursor,stages:recount,
  sentence_presentations:audit.sentence_presentations,auditor_sha256:sha(await fs.readFile(process.argv[1])),
  development_checks:checks,m1_complete:false,m2_permitted:false,production_promoted:false,
  source_derived_development_strata:strata,
  scope:'Independent decoder and call-order recount by the same engineering author. Verifies recorded calls, not OS isolation, an independent corpus, execution rerun, or full M1 certification.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
