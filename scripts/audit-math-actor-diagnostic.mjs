import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';
import {focusExample} from './math-actor-source.mjs';
import {target,selected,compare,summarize,canonicalInitialFrame} from './math-actor-diagnostic.mjs';
import {checkTrace,close} from './math-actor-trace.mjs';

const [run,out]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/audit-math-actor-diagnostic.mjs <frozen-diagnostic> <new-audit.json>');
const read=async file=>JSON.parse(await fs.readFile(path.join(run,file)));
const frozen=await read('freeze.json'),audit=await read('audit.json'),episodes=await read('episodes.json');
assert.equal(sha(await fs.readFile(path.join(run,'candidate.exe'))),frozen.binary_sha256);assert.equal(sha(await fs.readFile(path.join(run,'episodes.json'))),frozen.episodes_sha256);
for(const [file,hash]of Object.entries(frozen.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',file))),hash);
assert.equal(frozen.models.length,6);assert.equal(audit.results.length,6);assert.equal(episodes.length,629);
assert.equal(audit.corpus_answers_read,false);assert.equal(audit.no_fitting,true);
const results=[];
for(const model of frozen.models){
  assert.equal(sha(await fs.readFile(model.number)),model.number_sha256);
  const focused=model.policy.layout==='focused',page=model.policy.numbers_only?'free':'full',teacher=await read(`model-${model.index}-teacher.json`),own=await read(`model-${model.index}-own.json`),episodeRows=await read(`model-${model.index}-episode-audit.json`);
  assert.equal(teacher.length,2251);assert.equal(own.length,629);assert.equal(episodeRows.length,629);
  let cursor=0;
  for(const [index,e]of episodes.entries()){
    const oracle=teacher.slice(cursor,cursor+e[page].examples.length);cursor+=oracle.length;
    for(const [step,r]of oracle.entries()){
      const original=e[page].examples[step],example=focused?focusExample(original,step):original;
      assert.equal(r.id,`${e.id}:${step}`);assert.equal(r.episode,e.id);assert.equal(r.step,step);assert.equal(r.frame,example.text);assert.deepEqual(r.expected,target(example));assert.deepEqual(r.comparison,compare(r.prediction,r.expected));
      const taught=step<e[page].actions.length?e[page].actions[step]:null,p=selected(r.prediction),program=taught?taught.program:'math-done',values=taught?[taught.a.value,taught.b.value]:[e[page].final];
      const semantic=p!==null&&p.program===program&&(p.operands.every((n,i)=>close(n,values[i]))||(['math-p032','math-p034'].includes(program)&&close(p.operands[0],values[1])&&close(p.operands[1],values[0])));
      assert.equal(semantic,r.comparison.semantic,'Source action records and tagged-frame expectation must agree');
    }
    const row=own[index],scored=episodeRows[index];assert.equal(row.id,e.id);assert.equal(scored.id,e.id);
    const text=e[page].examples[0].text.split(' ').slice(1,-2).join(' ');checkTrace(row,text,focused);
    assert.equal(scored.own_answer,row.answer);assert.equal(scored.own_correct,close(row.answer,e[page].final));assert.equal(scored.expected_final,e[page].final);assert.equal(scored.own_error,row.error);
    const matches=row.trace[0].frame===oracle[0].frame;assert.equal(scored.first_teacher_frame_matches,matches);
    if(matches){assert.deepEqual(row.trace[0].prediction??null,oracle[0].prediction);assert.equal(row.trace[0].work,oracle[0].work);}else{assert(model.policy.numbers_only);assert.equal(row.trace[0].frame,canonicalInitialFrame(oracle[0].frame,focused));}
    assert.equal(scored.teacher_all_semantic,oracle.every(r=>r.comparison.semantic));assert.equal(scored.teacher_all_exact,oracle.every(r=>r.comparison.exact));assert.equal(scored.first_teacher_disagreement,oracle.find(r=>!r.comparison.semantic)?.step??null);
  }
  assert.equal(cursor,2251);
  const recorded=audit.results[model.index];assert.deepEqual(recorded.teacher,summarize(teacher.map(r=>r.comparison)));
  for(const [field,predicate]of [
    ['teacher_complete_semantic',r=>r.teacher_all_semantic],['teacher_complete_exact',r=>r.teacher_all_exact],['first_frame_mismatches',r=>!r.first_teacher_frame_matches],['own_correct',r=>r.own_correct],['own_refused',r=>r.own_answer===null],
    ['own_failed_despite_teacher_complete',r=>r.teacher_all_semantic&&!r.own_correct],['own_failed_despite_teacher_complete_and_same_first_frame',r=>r.first_teacher_frame_matches&&r.teacher_all_semantic&&!r.own_correct]
  ])assert.equal(recorded[field],episodeRows.filter(predicate).length);
  for(const [field,key]of [['by_step',r=>r.step],['by_operation',r=>r.expected[1]]])for(const group of new Set(teacher.map(key)))assert.deepEqual(recorded[field][group],summarize(teacher.filter(r=>key(r)===group).map(r=>r.comparison)));
  assert.equal(recorded.teacher_work,teacher.reduce((n,r)=>n+r.work,0));assert.equal(recorded.own_work,own.reduce((n,r)=>n+r.language_work,0));results.push(recorded);
}
let reused=0;
const provenanceIndexes=new Map();
async function provenance(directory){
  if(!provenanceIndexes.has(directory)){
    const index=new Map();for(const file of (await fs.readdir(directory)).filter(f=>/^call-\d+\.json$/u.test(f))){const bytes=await fs.readFile(path.join(directory,file));index.set(sha(bytes),JSON.parse(bytes));}
    provenanceIndexes.set(directory,index);
  }
  return provenanceIndexes.get(directory);
}
for(const [index,call]of audit.calls.entries()){
  assert.equal(call.order,index);assert.deepEqual(call,await read(`call-${index}.json`));assert.equal(call.exit_code,0);
  const summary=JSON.parse(await fs.readFile(path.join(run,`${call.name}.stdout.log`),'utf8')),inputs=JSON.parse(await fs.readFile(call.args[3])),rows=JSON.parse(await fs.readFile(call.args[4]));
  assert(summary.seed_unchanged);assert.deepEqual(rows.map(r=>r.id),inputs.map(r=>r.id));assert(inputs.every(r=>Object.keys(r).sort().join(',')==='id,text'));
  if(call.args[0].endsWith('-frames-batch')){assert(summary.supplied_states&&summary.additional_teacher_queries===0&&summary.arithmetic_evaluations===0&&summary.peak_active_modules<=1);assert(inputs.length<=32);}else{assert.equal(summary.teacher_queries,0);assert(summary.peak_active_modules<=2);assert(inputs.length<=4);}
  if(call.reused_run){
    let link=call;const visited=new Set();
    while(link.reused_run){
      const key=`${path.resolve(link.reused_run)}:${link.original_call_sha256}`;assert(!visited.has(key)&&visited.size<8,'Reused provenance cycle/depth');visited.add(key);
      const previous=(await provenance(link.reused_run)).get(link.original_call_sha256);assert(previous,'Reused call provenance missing');assert.equal(previous.name,call.name);assert.equal(previous.exit_code,0);assert.equal(previous.args[0],call.args[0]);assert.deepEqual(JSON.parse(await fs.readFile(previous.args[3])),inputs);assert.deepEqual(JSON.parse(await fs.readFile(previous.args[4])),rows);
      const ancestor=JSON.parse(await fs.readFile(path.join(link.reused_run,'freeze.json')));assert.equal(ancestor.binary_sha256,frozen.binary_sha256);assert.equal(ancestor.episodes_sha256,frozen.episodes_sha256);link=previous;
    }
    assert(Number.isFinite(link.whole_process_ms)&&link.whole_process_ms>=0);assert((await fs.stat(link.args[2])).isDirectory());reused++;
  }
}
assert.equal(audit.calls.length,1374);
const report={schema:'axon-math-actor-diagnostic-recount/v1',verified:true,binary_sha256:frozen.binary_sha256,auditor_sha256:sha(await fs.readFile(process.argv[1])),trace_checker_sha256:sha(await fs.readFile('scripts/math-actor-trace.mjs')),diagnostic_helpers_sha256:sha(await fs.readFile('scripts/math-actor-diagnostic.mjs')),results,teacher_decisions:13506,own_episodes:3774,calls:audit.calls.length,reused_calls:reused,new_corpus_answers_read:0,new_fits:0,m1_complete:false,m2_permitted:false,scope:'All own-state arithmetic/history and teacher tagged actions checked against source action/value records. Training-only diagnosis, no new public capability or causal proof. Reused native results preserve their source call hashes and ancestral costs; post-processing failures were retained, not relearning.'};
await fs.writeFile(out,JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify({verified:true,models:results.length,teacher:13506,own:3774,calls:1374,reused,m1_complete:false}));
