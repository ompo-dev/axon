import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {partitionQuestions} from './gsm8k-source.mjs';
import {teaching,numbersOnly,tuples,evaluate} from './math-program-source.mjs';

const [outputPath,protocolFile='evals/protocols/m1-public-math-programs-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/prepare-math-program-training.mjs <new-output> [protocol]');
const output=path.resolve(outputPath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const sha=b=>createHash('sha256').update(b).digest('hex');
const write=(file,data)=>fs.writeFile(path.join(output,file),JSON.stringify(data,null,2),{flag:'wx'});
const protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);
assert.equal(protocol.schema,'axon-public-math-programs/v1');
const sourceProtocol=JSON.parse(await fs.readFile(protocol.source_protocol));
const raw=await fs.readFile('.axon-memory/corpora/gsm8k-3101c7d50724/train.jsonl');assert.equal(sha(raw),protocol.source_train_sha256);
const partitionBytes=await fs.readFile(protocol.partition);assert.equal(sha(partitionBytes),protocol.partition_sha256);
const partition=JSON.parse(partitionBytes),records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse);
const counts={train:0,development:0,reserved:0,research_only:0},accesses={...counts};
for(const [i,record]of records.entries()){
  const split=partition[i].split;counts[split]++;
  if(split==='train'){
    const value=Object.getOwnPropertyDescriptor(record,'answer').value;
    Object.defineProperty(record,'answer',{enumerable:true,get(){accesses.train++;return value;}});
  }else Object.defineProperty(record,'answer',{enumerable:true,get(){accesses[split]++;throw Error('Non-training answer access prohibited');}});
}
assert.deepEqual(partitionQuestions(records,sourceProtocol.exposed_question_prefixes).rows,partition);
const sources={};
for(const file of [protocolFile,protocol.source_protocol,'scripts/math-program-source.mjs','scripts/math-program-source.test.mjs','scripts/gsm8k-source.mjs',process.argv[1],'src/structural_math_lab.rs','src/structural_lab.rs','src/curriculum.rs','src/demand.rs','src/genome.rs','src/numeric.rs','src/numeric_execution.rs','src/numeric_constructive.rs','src/structural_lexical_encoder.rs','src/structural_sequence.rs','Cargo.toml','Cargo.lock','evals/protocols/m1-completion-v1.json','ui/node_modules/typescript/package.json']){
  const bytes=await fs.readFile(file),relative=path.relative(process.cwd(),path.resolve(file));assert(!relative.startsWith('..')&&!path.isAbsolute(relative));sources[relative.replaceAll('\\','/')]=sha(bytes);
  const destination=path.join(output,'source',relative);await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,bytes,{flag:'wx'});
}
const parent=await fs.readFile(protocol.parent);assert.equal(sha(parent),protocol.parent_sha256);await fs.writeFile(path.join(output,'parent.axnum'),parent,{flag:'wx'});
await write('freeze-before-labels.json',{sources,protocol_sha256:sha(protocolBytes),partition_sha256:sha(partitionBytes),parent_sha256:sha(parent),source_sha256:sha(raw),answer_accesses:{...accesses},counts,scope:'Frozen before interpreting train answer properties. Raw source JSON parsed in host RAM; no OS isolation. All nontraining answer getters throw.'});
const groups=new Map(),refusals={},examples=[];
for(const p of partition){
  if(p.split!=='train')continue;
  try{
    const e=teaching(records[p.ordinal].question,records[p.ordinal].answer);
    if(!groups.has(e.key))groups.set(e.key,{key:e.key,tree:e.tree,arity:e.arity,ops:e.ops,rows:[]});
    const row={id:String(p.ordinal),text:e.text,tags:e.tags,numbers_only:numbersOnly(e),source_question_sha256:p.question_sha256};groups.get(e.key).rows.push(row);examples.push(row);
  }catch(error){refusals[error.message]=(refusals[error.message]??0)+1;}
}
const ranked=[...groups.values()].filter(g=>g.rows.length>=protocol.domain.minimum_program_examples).sort((a,b)=>b.rows.length-a.rows.length||(a.key<b.key?-1:a.key>b.key?1:0)).slice(0,protocol.domain.programs);
const programs=[];
for(const [index,g]of ranked.entries()){
  const id=`math-p${String(index).padStart(3,'0')}`;
  const values=tuples(g.arity,0,16).flatMap(input=>{try{return [{input,output:evaluate(g.tree,input)}];}catch{return [];}});
  const train=values.slice(0,8),admission=values.slice(8,16);const ready=train.length===8&&admission.length===8;
  programs.push({...g,id,ready,teaching:train,admission});
  if(ready){for(const [kind,rows]of [['teaching',train],['admission',admission]])await fs.writeFile(path.join(output,`${id}-${kind}.tsv`),[...Array.from({length:g.arity},(_,i)=>`x${i}`),'y'].join('\t')+'\n'+rows.map(r=>[...r.input,r.output].join('\t')).join('\n')+'\n',{flag:'wx'});}
}
assert.equal(accesses.train,4429);assert.equal(accesses.development+accesses.reserved+accesses.research_only,0);
await write('programs.json',programs);
const development=partition.filter(p=>p.split==='development').map(p=>({id:String(p.ordinal),text:records[p.ordinal].question.replace(/\s+/gu,' ').trim()}));
await write('development-inputs.json',development);
const audit={schema:'axon-math-program-training-source/v1',source_train:4429,eligible:examples.length,refused:4429-examples.length,refusal_reasons:refusals,unique_signatures:groups.size,selected:programs.length,selected_rows:programs.reduce((n,g)=>n+g.rows.length,0),ready_for_numeric_admission:programs.filter(g=>g.ready).length,answer_accesses:accesses,development_inputs:development.length,learning_started:false,m1_complete:false,m2_permitted:false,scope:'Conservative annotation-grounded program supervision from allowed train only; all evaluation questions retained. No autonomous formula discovery or public accuracy yet.'};
await write('audit.json',audit);console.log(JSON.stringify(audit));
