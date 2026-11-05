import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual as equal} from 'node:util';
import {canonical,sequenceAnnotation,metrics} from './massive-grammar.mjs';

const [corpus,output,...runs]=process.argv.slice(2);
if(!output||!runs.length)throw Error('usage: node scripts/analyze-public-errors.mjs <corpus> <new-output> <frozen-run>...');
await fs.mkdir(output);
const sha=b=>createHash('sha256').update(b).digest('hex');
const json=async f=>JSON.parse(await fs.readFile(f));
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const bytes=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
const provenance=await json(path.join(corpus,'provenance.json')),reserved=new Set(provenance.reserved_workers);
assert.equal(sha(bytes),provenance.data_sha256);
const allowed=bytes.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse)
  .filter(r=>!reserved.has(r.worker_id)&&['train','dev'].includes(r.partition));
const train=allowed.filter(r=>r.partition==='train'),dev=allowed.filter(r=>r.partition==='dev');
assert.equal(train.length,8526);assert.equal(dev.length,1504);
const truth=dev.map(r=>({id:String(r.id),...sequenceAnnotation(r)}));
const words=text=>canonical(text).split(' '),vocabulary=new Set(train.flatMap(r=>words(r.utt)));
const counts=new Map();for(const r of train)counts.set(r.intent,(counts.get(r.intent)??0)+1);
const models=[],sources={};
for(const run of runs) {
  const audit=await json(path.join(run,'audit.json')),freeze=await json(path.join(run,'freeze.json'));
  assert.equal(freeze.corpus_sha256,sha(bytes));
  assert.deepEqual(freeze.train_ids,train.map(r=>String(r.id)));assert.deepEqual(freeze.development_ids,dev.map(r=>String(r.id)));
  for(const p of audit.policies) {
    assert(models.length<64,'Diagnostic model quota');
    const file=path.join(run,`policy-${p.index}-predictions.json`),raw=await fs.readFile(file),rows=JSON.parse(raw);
    sources[file]=sha(raw);const score=metrics(rows,truth);assert.deepEqual(score,p.score);
    models.push({name:`${run}:${p.index}`,rows,score});
  }
}
await write('freeze.json',{created_at:new Date().toISOString(),corpus_sha256:sha(bytes),sources,
  analyzer_sha256:sha(await fs.readFile(process.argv[1])),parser_sha256:sha(await fs.readFile(new URL('./massive-grammar.mjs',import.meta.url))),
  scope:'Retrospective analysis of recorded consumed-development outputs only. No new predictions, training, test/reserved text, deployed ensemble or prospective accuracy claim.'});
const summary=[],failures=[];
for(const m of models) {
  const errors={intent_only:0,slots_only:0,both:0,abstained:0},strata={},confusions=new Map();
  m.rows.forEach((row,i)=>{
    const target=truth[i].target,p=row.prediction;
    const intent=p?.[1]===target[1],slots=p!==null&&equal(p[2],target[2]),exact=equal(p,target);
    if(!exact) {if(p===null)errors.abstained++;else errors[intent?'slots_only':slots?'intent_only':'both']++;}
    const ws=words(dev[i].utt),unknown=ws.filter(w=>!vocabulary.has(w));
    for(const label of [unknown.length?'unknown_words':'known_words',ws.length<=5?'short':'long',counts.get(target[1])<50?'rare_intent':'frequent_intent']) {
      const bucket=strata[label]??={total:0,exact:0,intent:0,slots:0};bucket.total++;bucket.exact+=Number(exact);bucket.intent+=Number(intent);bucket.slots+=Number(slots);
    }
    if(!intent&&p) {const key=JSON.stringify([target[1],p[1]]);confusions.set(key,(confusions.get(key)??0)+1);}
    if(!exact)failures.push({model:m.name,id:row.id,text:dev[i].utt,target,prediction:p,unknown_words:unknown,intent_training_rows:counts.get(target[1])??0});
  });
  summary.push({name:m.name,score:m.score,errors,strata,top_confusions:[...confusions].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,12)});
}
const oracle={exact:0,intent:0,slots:0,total:dev.length,scope:'Unattainable answer-informed upper bound, NOT an evaluated learner or deployable selection policy.'};
const never=[];
for(let i=0;i<dev.length;i++) {
  const target=truth[i].target,exact=models.some(m=>equal(m.rows[i].prediction,target));
  oracle.exact+=Number(exact);oracle.intent+=Number(models.some(m=>m.rows[i].prediction?.[1]===target[1]));
  oracle.slots+=Number(models.some(m=>equal(m.rows[i].prediction?.[2],target[2])));
  if(!exact)never.push({id:truth[i].id,text:dev[i].utt,target});
}
const pairs=[];
for(let a=0;a<models.length;a++)for(let b=a+1;b<models.length;b++) {
  let both=0,onlyA=0,onlyB=0;
  for(let i=0;i<dev.length;i++) {const x=equal(models[a].rows[i].prediction,truth[i].target),y=equal(models[b].rows[i].prediction,truth[i].target);both+=Number(x&&y);onlyA+=Number(x&&!y);onlyB+=Number(y&&!x);}
  pairs.push({a:models[a].name,b:models[b].name,both,onlyA,onlyB,oracle_union:both+onlyA+onlyB});
}
await write('errors.json',failures);await write('never-correct.json',never);await write('pair-complementarity.json',pairs);
const report={schema:'axon-public-error-diagnostic/v1',models:summary,oracle,reserved_evaluated:false,original_test_evaluated:false,m1_complete:false,production_promoted:false};
await write('audit.json',report);
console.log(JSON.stringify({models:models.length,oracle,best_pairs:pairs.sort((a,b)=>b.oracle_union-a.oracle_union).slice(0,4),selected:summary.filter(m=>[826,842,851,859].includes(m.score.exact)).map(m=>({name:m.name,exact:m.score.exact,errors:m.errors,strata:m.strata}))}));
