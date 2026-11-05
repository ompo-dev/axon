import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import assert from 'node:assert/strict';
import {annotation, sequenceAnnotation, canonical, metrics} from './massive-grammar.mjs';
import {loadPublicPos, shuffledLabels} from './auxiliary-pos-source.mjs';
import {loadLexicalGraph,rewireEdges} from './lexical-graph-source.mjs';

const [parentFile,executable,corpus,outputPath,protocolPath='evals/protocols/m1-public-grammar-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/train-public-grammar.mjs <parent-number> <candidate> <massive-corpus> <new-output> [protocol]');
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,value)=>fs.writeFile(path.join(output,name),JSON.stringify(value,null,2),{flag:'wx'});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const protocolBytes=await fs.readFile(protocolPath),protocol=JSON.parse(protocolBytes);
const trainingTimeout=protocol.training_timeout_ms??600000;
assert(Number.isSafeInteger(trainingTimeout)&&trainingTimeout>=600000&&trainingTimeout<=3600000,'Training timeout bound');
const segment=protocol.learner==='segment-sequence-v1',encoder=protocol.learner==='lexical-encoder-v1',joint=protocol.learner==='joint-sequence-v1',factorized=protocol.learner==='factorized-sequence-v1',enhanced=protocol.learner==='enhanced-sequence-v1',compact=protocol.learner==='compact-sequence-v1',sequence=segment||encoder||joint||factorized||enhanced||compact||protocol.learner==='sequence-v1';
assert([undefined,'intent-only'].includes(protocol.training_projection));
const intentOnly=protocol.training_projection==='intent-only';assert(!intentOnly||encoder,'Intent component uses the explicit lexical encoder');
const trainCommand=segment?'segment-train':encoder?'encoder-train':joint?'joint-train':factorized?'factorized-train':enhanced?'enhanced-train':compact?'compact-train':sequence?'sequence-train':'grammar-train';
const batchCommand=segment?'segment-batch':encoder?'encoder-batch':joint?'joint-batch':factorized?'factorized-batch':enhanced?'enhanced-batch':compact?'compact-batch':sequence?'sequence-batch':'grammar-batch';
const provenanceBytes=await fs.readFile(path.join(corpus,'provenance.json')),provenance=JSON.parse(provenanceBytes);
const raw=await fs.readFile(path.join(corpus,'1.1/data/pt-PT.jsonl'));
assert.equal(sha(raw),provenance.data_sha256);
assert.equal(sha(raw),protocol.source_sha256);
const reserved=new Set(provenance.reserved_workers),rows=raw.toString('utf8').trim().split(/\r?\n/u).map(JSON.parse);
// Filter metadata BEFORE accessing annotations or text from any reserved worker/test.
const allowed=rows.filter(row=>!reserved.has(row.worker_id)&&['train','dev'].includes(row.partition));
const splits={train:[],dev:[]},invalid=[];
for(const row of allowed) {
  try {splits[row.partition].push({id:String(row.id),text:row.utt,...(sequence?sequenceAnnotation(row):annotation(row))});}
  catch(error) {invalid.push({id:String(row.id),partition:row.partition,error:error.message});}
}
assert.equal(invalid.length,0,'Malformed source annotation: inspect development only, never silently drop evaluation cases');
assert.equal(splits.train.length,8526);
let auxiliary;
if(protocol.auxiliary_source) {
  assert(encoder && protocol.auxiliary_policies?.length===protocol.policies.length,'Auxiliary policy cardinality');
  auxiliary=await loadPublicPos(protocol.auxiliary_source);
  await write('auxiliary-source.json',auxiliary.stats);await write('auxiliary-origins.json',auxiliary.origins);
  await write('auxiliary-examples.json',auxiliary.examples);
  for(const name of ['pt_bosque-ud-train.conllu','provenance.json','LICENSE.txt','README.md']) {
    const destination=path.join(output,'auxiliary-source',name);await fs.mkdir(path.dirname(destination),{recursive:true});
    await fs.copyFile(path.join(protocol.auxiliary_source.corpus,name),destination,fs.constants.COPYFILE_EXCL);
  }
} else assert(protocol.auxiliary_policies===undefined,'Auxiliary source required');
let lexicalGraph,rewiredGraph;
if(protocol.lexical_source) {
  assert(encoder&&!auxiliary&&protocol.lexical_policies?.length===protocol.policies.length,'Lexical policy cardinality/combination');
  lexicalGraph=await loadLexicalGraph(protocol.lexical_source);rewiredGraph=rewireEdges(lexicalGraph.edges,19374611);
  await write('lexical-source.json',lexicalGraph.statistics);await write('lexical-edges.json',lexicalGraph.edges);
  await write('lexical-rewired-edges.json',rewiredGraph.edges);await write('lexical-rewiring.json',rewiredGraph.statistics);
  const copies=[['facts.tsv',protocol.lexical_source.facts],['facts.tsv.provenance.json',`${protocol.lexical_source.facts}.provenance.json`],['vocabulary.axseq',protocol.lexical_source.vocabulary],['LICENSE',protocol.lexical_source.license_file]];
  await fs.mkdir(path.join(output,'lexical-source'));
  for(const [name,file]of copies)await fs.copyFile(file,path.join(output,'lexical-source',name),fs.constants.COPYFILE_EXCL);
} else assert(protocol.lexical_policies===undefined,'Lexical source required');
const binary=path.join(output,'candidate.exe'),parent=path.join(output,'parent.axnum');
await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);await fs.copyFile(parentFile,parent,fs.constants.COPYFILE_EXCL);
const sources={};
const freezeSource=async file=>{
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);
  const destination=path.join(output,'source',file);await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,bytes,{flag:'wx'});
};
for(const file of ['src/structural.rs','src/demand.rs','src/genome.rs','src/structural_phrase.rs','src/structural_induction.rs','src/structural_grammar_lab.rs','src/structural_sequence.rs','src/structural_sequence_compact.rs','src/structural_sequence_factorized.rs','src/structural_sequence_search.rs','src/structural_sequence_joint.rs','src/structural_joint_lab.rs','src/structural_lexical_encoder.rs','src/structural_byte_convolution.rs','src/structural_ordered_context.rs','src/structural_optimizer.rs','src/structural_encoder_lab.rs','src/structural_dependency.rs','src/structural_sequence_segment.rs','src/structural_segment_lab.rs','src/structural_sequence_lab.rs','src/structural_lab.rs','src/curriculum.rs','scripts/train-public-grammar.mjs','scripts/auxiliary-pos-source.mjs','scripts/auxiliary-pos-source.test.mjs','scripts/conllu-source.mjs','scripts/massive-grammar.mjs','scripts/massive-grammar.test.mjs','scripts/audit-public-grammar.mjs','tests/structural_induction_contracts.rs','tests/structural_sequence_contracts.rs','evals/protocols/m1-completion-v1.json',protocolPath]) {
  await freezeSource(file);
}
for(const file of ['src/structural_lexical_graph.rs','scripts/lexical-graph-source.mjs','scripts/lexical-graph-source.test.mjs'])await freezeSource(file);
await write('freeze.json',{created_at:new Date().toISOString(),sources,binary_sha256:sha(await fs.readFile(binary)),parent_sha256:sha(await fs.readFile(parent)),
  corpus_sha256:sha(raw),provenance_sha256:sha(provenanceBytes),protocol_sha256:sha(protocolBytes),
  driver_sha256:sha(await fs.readFile(process.argv[1])),parser_sha256:sha(await fs.readFile(new URL('./massive-grammar.mjs',import.meta.url))),
  train_ids:splits.train.map(r=>r.id),development_ids:splits.dev.map(r=>r.id),reserved_worker_count:reserved.size,
  auxiliary:auxiliary?{...auxiliary.stats,examples_sha256:sha(await fs.readFile(path.join(output,'auxiliary-examples.json'))),origins_sha256:sha(await fs.readFile(path.join(output,'auxiliary-origins.json')))}:null,
  lexical_graph:lexicalGraph?{...lexicalGraph.statistics,edges_sha256:sha(await fs.readFile(path.join(output,'lexical-edges.json'))),rewired_edges_sha256:sha(await fs.readFile(path.join(output,'lexical-rewired-edges.json')))}:null,
  scope:'Frozen before any core training or predictions. Reserved worker and original test text not inspected by this driver.'});
async function core(name,args) {
  const started=performance.now();let child;
  try {child=await promisify(execFile)(binary,args,{cwd:output,windowsHide:true,timeout:args[0]===trainCommand?trainingTimeout:600000,maxBuffer:64000000,env:{...process.env,AXON_LAB_TELEMETRY:'1'}});}
  catch(error) {await fs.writeFile(path.join(output,`${name}-failed.stdout.log`),error.stdout??'',{flag:'wx'});await fs.writeFile(path.join(output,`${name}-failed.stderr.log`),error.stderr??'',{flag:'wx'});throw error;}
  await fs.writeFile(path.join(output,`${name}.stdout.jsonl`),child.stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),child.stderr,{flag:'wx'});
  const events=child.stdout.trim().split(/\r?\n/u).map(JSON.parse),summary=events.pop();events.forEach((e,i)=>assert.equal(e.coreSeq,i+1));
  return {summary,whole_process_ms:performance.now()-started,core_events:events.length};
}
const families=new Map(),memory=new Map(),counts=new Map();
for(const row of splits.train) {
  const keyFamily=protocol.grouping==='typed-template'&&Array.isArray(row.target[2])?JSON.stringify([row.family,row.template]):row.family;
  const list=families.get(keyFamily)??[];list.push({text:row.text,target:row.target});families.set(keyFamily,list);
  const key=canonical(row.text),target=JSON.stringify(row.target);
  const candidates=memory.get(key)??new Set();candidates.add(target);memory.set(key,candidates);counts.set(target,(counts.get(target)??0)+1);
}
const common=[...counts.entries()].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))[0][0];
const baselines={};
for(const [name,predict] of [
  ['untrained',()=>null],['exact_retrieval',row=>{const values=memory.get(canonical(row.text));return values?.size===1?JSON.parse([...values][0]):null;}],
  ['constant',()=>JSON.parse(common)],
])baselines[name]=metrics(splits.dev.map(row=>({id:row.id,prediction:predict(row)})),splits.dev);
for(const [field,name] of [['previous_predictions','grammar'],['previous_sequence_predictions','sequence'],['previous_compact_predictions','compact']])if(protocol[field]) {
  const bytes=await fs.readFile(protocol[field]),predictions=JSON.parse(bytes);
  baselines[`previous_${name}`]=metrics(predictions,splits.dev);
  await fs.writeFile(path.join(output,`previous-${name}-predictions.json`),bytes,{flag:'wx'});
  await write(`previous-${name}-source.json`,{path:protocol[field],sha256:sha(bytes),scope:'Stored predictions from the previous consumed development experiment, not a live rerun.'});
}
await write('baselines.json',baselines);
await write('development-inputs.json',splits.dev.map(({id,text})=>({id,text})));
await write('development-truth.json',splits.dev.map(({id,target})=>({id,target})));
const policies=[];
for(let i=0;i<protocol.policies.length;i++) {
  const name=`policy-${i}`,training=sequence?{budget:protocol.policies[i],examples:splits.train.map(r=>({text:r.text,intent:r.target[1],tags:intentOnly?r.tags.map(()=>'O'):r.tags}))}:{budget:protocol.policies[i],lexical_domains:protocol.lexical_domains??false,families:[...families.values()]};
  const extra=protocol.auxiliary_policies?.[i];
  if(extra) {
    assert(auxiliary && Object.keys(extra).sort().join(',')==='interval,shuffle_labels,weight','Auxiliary policy fields');
    assert(typeof extra.shuffle_labels==='boolean');
    training.auxiliary={weight:extra.weight,interval:extra.interval,examples:extra.shuffle_labels?shuffledLabels(auxiliary.examples,19374611):auxiliary.examples};
  }
  const lexicalPolicy=protocol.lexical_policies?.[i];
  if(lexicalPolicy) {
    assert(Object.keys(lexicalPolicy).sort().join(',')==='interval,rewire,weight');assert(typeof lexicalPolicy.rewire==='boolean');
    training.lexical_graph={weight:lexicalPolicy.weight,interval:lexicalPolicy.interval,edges:lexicalPolicy.rewire?rewiredGraph.edges:lexicalGraph.edges};
  }
  await write(`${name}-train.json`,training);
  const trained=await core(`${name}-training`,[trainCommand,parent,path.join(output,`${name}-parent-birth`),path.join(output,`${name}-train.json`),path.join(output,name)]);
  assert(trained.summary.parent_unchanged);
  assert.deepEqual(await fs.readFile(path.join(output,`${name}-parent-birth`,'structural.axstruct')),await fs.readFile(path.join(output,name,'structural.axstruct')),'Operational page must survive byte-for-byte');
  const number=path.join(output,name,'structural.axnum');
  if(i===0 && protocol.control_number_sha256)assert.equal(sha(await fs.readFile(number)),protocol.control_number_sha256,'Primary-only control must reproduce the frozen number');
  if(protocol.expected_numbers?.[i])assert.equal(sha(await fs.readFile(number)),protocol.expected_numbers[i],'Frozen comparison number must reproduce');
  let evaluated;
  if(encoder||segment) {
    const predictions=[],chunks=[];
    const chunkSize=segment?16:(protocol.policies[i].byte_convolution||protocol.policies[i].ordered_context)?32:128;
    for(let start=0;start<splits.dev.length;start+=chunkSize) {
      const chunk=`${name}-dev-${start}`,input=path.join(output,`${chunk}-inputs.json`),receipt=path.join(output,`${chunk}-predictions.json`);
      await write(`${chunk}-inputs.json`,splits.dev.slice(start,start+chunkSize).map(({id,text})=>({id,text})));
      const measured=await core(chunk,[batchCommand,number,path.join(output,`${chunk}-birth`),input,receipt]);
      assert(measured.summary.loads===1&&measured.summary.active_modules===1&&measured.summary.seed_unchanged&&measured.summary.teacher_queries===0);chunks.push(measured);
      predictions.push(...JSON.parse(await fs.readFile(receipt)));
    }
    await write(`${name}-predictions.json`,predictions);await write(`${name}-chunks.json`,chunks);
    evaluated={summary:{seed_unchanged:true,teacher_queries:0,total:predictions.length,chunks:chunks.length,work:chunks.reduce((n,c)=>n+c.summary.work,0)},whole_process_ms:chunks.reduce((n,c)=>n+c.whole_process_ms,0),core_events:chunks.reduce((n,c)=>n+c.core_events,0)};
  } else evaluated=await core(`${name}-development`,[batchCommand,number,path.join(output,`${name}-dev-birth`),path.join(output,'development-inputs.json'),path.join(output,`${name}-predictions.json`)]);
  assert(evaluated.summary.seed_unchanged&&evaluated.summary.teacher_queries===0);
  const predictions=JSON.parse(await fs.readFile(path.join(output,`${name}-predictions.json`))),score=metrics(predictions,splits.dev);
  const trainValues=new Set(splits.train.flatMap(row=>Array.isArray(row.target[2])?row.target[2].slice(1).map(([role,value])=>JSON.stringify([role,value])):[]));
  const strata={};
  for(const [label,include] of [
    ['unseen_text',row=>!memory.has(canonical(row.text))],
    ['unseen_slot_values',row=>Array.isArray(row.target[2])&&row.target[2].slice(1).some(([role,value])=>!trainValues.has(JSON.stringify([role,value])))],
    ['multiword_slots',row=>Array.isArray(row.target[2])&&row.target[2].slice(1).some(([,value])=>value.includes(' '))],
    ['unseen_role_signature',row=>!splits.train.some(r=>r.family===row.family)],
  ]) {
    const indices=splits.dev.map((row,index)=>include(row)?index:null).filter(index=>index!==null);
    strata[label]=metrics(indices.map(index=>predictions[index]),indices.map(index=>splits.dev[index]));
  }
  const report={index:i,number_sha256:sha(await fs.readFile(number)),training:trained,development:evaluated,score,strata};
  policies.push(report);await write(`${name}-score.json`,report);console.log(JSON.stringify({policy:i,statistics:trained.summary.statistics,score}));
}
const complexity=p=>p.training.summary.statistics?.selected??p.training.summary.weights;
const selected=[...policies].sort((a,b)=>b.score.exact-a.score.exact||complexity(a)-complexity(b)||a.index-b.index)[0];
await write('audit.json',{schema:'axon-public-grammar/v1',learner:protocol.learner??'rules',train:splits.train.length,development:splits.dev.length,families:sequence?null:families.size,
  selected:selected.index,policies,baselines,reserved_evaluated:false,original_test_evaluated:false,m1_complete:false,production_promoted:false,scope:protocol.scope});
const completionBytes=await fs.readFile(path.join(output,'source/evals/protocols/m1-completion-v1.json'));
const completion=JSON.parse(completionBytes),language=completion.gates.find(g=>g.id==='external-language');
assert(language?.required,'Full M1 contract must include the external-language gate');
const observed={intent_accuracy:selected.score.intent_correct/selected.score.total,
  exact_structured_accuracy:selected.score.exact/selected.score.total,coverage:selected.score.answered/selected.score.total,
  per_family_accuracy:Math.min(...Object.values(selected.score.families).map(f=>f.exact/f.total))};
const checks=Object.fromEntries(Object.entries(observed).map(([key,value])=>[key,{observed:value,minimum:language[`minimum_${key}`],passed:value>=language[`minimum_${key}`]}]));
assert(Object.values(checks).every(c=>Number.isFinite(c.minimum)),'Do not silently omit an acceptance threshold');
await write('milestone-gates.json',{schema:'axon-m1-development-gates/v1',completion_contract_sha256:sha(completionBytes),
  binary_sha256:sha(await fs.readFile(binary)),number_sha256:selected.number_sha256,selected:selected.index,
  gates:completion.gates.map(g=>({id:g.id,required:g.required,status:g.id==='external-language'?
    (Object.values(checks).every(c=>c.passed)?'awaiting-frozen-reserved-audit':'failed-development'):'not-demonstrated-by-this-experiment',
    ...(g.id==='external-language'?{checks}:{})})),
  reserved_evaluated:false,original_test_evaluated:false,m1_complete:false,m2_permitted:false,production_promoted:false,
  scope:'Mechanical development gate check, not full M1 certification. Missing evidence fails promotion; no other gate is inferred from an intent/slot score.'});
