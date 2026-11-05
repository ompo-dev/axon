import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {partitionQuestions,sha} from './gsm8k-source.mjs';
import {teaching,tuples,evaluate} from './math-program-source.mjs';
import {episode} from './math-actor-source.mjs';

const [outputPath,protocolFile='evals/protocols/m1-math-actor-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/prepare-math-actor.mjs <new-output> [protocol]');
const output=path.resolve(outputPath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const write=(file,value)=>fs.writeFile(path.join(output,file),JSON.stringify(value,null,2),{flag:'wx'});
const protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);assert.equal(protocol.schema,'axon-math-actor/v1');
const raw=await fs.readFile('.axon-memory/corpora/gsm8k-3101c7d50724/train.jsonl');assert.equal(sha(raw),protocol.source_train_sha256);
const partitionBytes=await fs.readFile(protocol.partition);assert.equal(sha(partitionBytes),protocol.partition_sha256);
const partition=JSON.parse(partitionBytes),records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse),accesses={train:0,development:0,reserved:0,research_only:0};
for(const [i,record]of records.entries()){
  const split=partition[i].split;
  if(split==='train'){const value=Object.getOwnPropertyDescriptor(record,'answer').value;Object.defineProperty(record,'answer',{enumerable:true,get(){accesses.train++;return value;}});}
  else Object.defineProperty(record,'answer',{enumerable:true,get(){accesses[split]++;throw Error('Sealed nontraining actor answer');}});
}
const sourceProtocol=JSON.parse(await fs.readFile(protocol.source_protocol));assert.deepEqual(partitionQuestions(records,sourceProtocol.exposed_question_prefixes).rows,partition);
const sources={};
for(const file of [protocolFile,protocol.source_protocol,process.argv[1],'scripts/math-actor-source.mjs','scripts/math-actor-source.test.mjs','scripts/math-program-source.mjs','scripts/gsm8k-source.mjs','src/structural_math_actor_lab.rs','src/structural_math_lab.rs','src/structural_lexical_encoder.rs','src/structural_sequence_search.rs','src/curriculum.rs','src/genome.rs','src/demand.rs','Cargo.toml','Cargo.lock','evals/protocols/m1-completion-v1.json']){
  const relative=path.relative(process.cwd(),path.resolve(file));assert(!relative.startsWith('..')&&!path.isAbsolute(relative));
  const bytes=await fs.readFile(file);sources[relative.replaceAll('\\','/')]=sha(bytes);const dest=path.join(output,'source',relative);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});
}
const parent=await fs.readFile(protocol.parent);assert.equal(sha(parent),protocol.parent_sha256);await fs.writeFile(path.join(output,'parent.axnum'),parent,{flag:'wx'});
await write('freeze-before-labels.json',{sources,protocol_sha256:sha(protocolBytes),parent_sha256:sha(parent),source_sha256:sha(raw),partition_sha256:sha(partitionBytes),answer_accesses:{...accesses},scope:'Source and parser frozen before training answer interpretation. Raw JSON bytes parsed in host RAM, not OS label isolation. This is previously consumed development, not a new reserve.'});
const episodes=[],refusals={},shapes=new Set();
for(const p of partition){if(p.split!=='train')continue;
  try{const e=teaching(records[p.ordinal].question,records[p.ordinal].answer),full=episode(e),free=episode(e,true);assert.equal(full.examples.length,free.examples.length);episodes.push({id:String(p.ordinal),question_sha256:p.question_sha256,shape:e.key,full,free});shapes.add(e.key);}
  catch(error){refusals[error.message]=(refusals[error.message]??0)+1;}
}
assert.equal(accesses.train,4429);assert.equal(accesses.development+accesses.reserved+accesses.research_only,0);
const frames=episodes.reduce((n,e)=>n+e.full.examples.length,0);assert(frames>=4&&frames<=10000);
await write('episodes.json',episodes);
for(const [i,policy]of protocol.policies.entries())await write(`policy-${i}-train.json`,{budget:protocol.budget,examples:episodes.flatMap(e=>(policy.numbers_only?e.free:e.full).examples)});
for(const primitive of protocol.primitive_library){
  assert(/^math-p03[2-5]$/u.test(primitive.id));
  const rows=tuples(2,0,16).map(input=>({input,output:evaluate([primitive.operator,'x0','x1'],input)}));
  for(const [kind,values]of [['teaching',rows.slice(0,8)],['admission',rows.slice(8)]])await fs.writeFile(path.join(output,`${primitive.id}-${kind}.tsv`),'x0\tx1\ty\n'+values.map(r=>[...r.input,r.output].join('\t')).join('\n')+'\n',{flag:'wx'});
}
const inputs=partition.filter(p=>p.split==='development').map(p=>({id:String(p.ordinal),text:records[p.ordinal].question.replace(/\s+/gu,' ').trim()}));assert.equal(inputs.length,1481);await write('development-inputs.json',inputs);
const audit={schema:'axon-math-actor-source/v1',train:4429,episodes:episodes.length,shapes:shapes.size,frames_per_policy:frames,source_refused:4429-episodes.length,refusal_reasons:refusals,answer_accesses:accesses,maximum_actions:Math.max(...episodes.map(e=>e.full.actions.length)),maximum_frame_tokens:Math.max(...episodes.flatMap(e=>e.full.examples.map(r=>r.tags.length))),development_inputs:1481,supervision:'Teacher states only for allowed train. Core inference must generate every intermediate result itself; no independent or mathematical discovery claim.',m1_complete:false,m2_permitted:false};
await write('audit.json',audit);console.log(JSON.stringify(audit));
