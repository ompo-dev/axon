import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha,partitionQuestions} from './gsm8k-source.mjs';
import {teaching} from './math-program-source.mjs';
import {episode} from './math-actor-source.mjs';
import {pointerExamples} from './relational-pointer-source.mjs';

const output=path.resolve(process.argv[2]??'');if(!process.argv[2]||!output.startsWith(path.resolve()+path.sep))throw Error('usage: node scripts/prepare-relational-pointer.mjs <new-workspace-output>');
await fs.mkdir(output);const write=(name,value)=>fs.writeFile(path.join(output,name),JSON.stringify(value,null,2),{flag:'wx'});
const protocolFile='evals/protocols/m1-relational-pointer-v1.json',protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes),base=JSON.parse(await fs.readFile(protocol.source_protocol));
const raw=await fs.readFile('.axon-memory/corpora/gsm8k-3101c7d50724/train.jsonl');assert.equal(sha(raw),base.source_train_sha256);
const partitionBytes=await fs.readFile(base.partition);assert.equal(sha(partitionBytes),base.partition_sha256);const partition=JSON.parse(partitionBytes),records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse),accesses={train:0,development:0,reserved:0,research_only:0};
for(const [i,r]of records.entries()){const split=partition[i].split,value=Object.getOwnPropertyDescriptor(r,'answer').value;Object.defineProperty(r,'answer',{enumerable:true,get(){accesses[split]++;if(split!=='train')throw Error('Sealed nontraining pointer source');return value;}});}
const sourceProtocol=JSON.parse(await fs.readFile(base.source_protocol));assert.deepEqual(partitionQuestions(records,sourceProtocol.exposed_question_prefixes).rows,partition);
const sources={};
async function capture(file){const bytes=await fs.readFile(file);assert(bytes.length<=67108864);const name=file.replaceAll('\\','/');if(sources[name]){assert.equal(sources[name],sha(bytes));return;}sources[name]=sha(bytes);const dest=path.join(output,'source',file);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,bytes,{flag:'wx'});}
async function walk(directory,pattern=/\.(rs|mjs|ps1)$/u){for(const e of await fs.readdir(directory,{withFileTypes:true})){assert(!e.isSymbolicLink(),'Source links refused');const file=path.join(directory,e.name);if(e.isDirectory())await walk(file,pattern);else if(pattern.test(e.name))await capture(file);}}
for(const dir of ['src','tests','scripts'])await walk(dir);
await walk('evals/protocols',/\.(json|txt)$/u);
for(const file of [protocolFile,protocol.source_protocol,base.source_protocol,'evals/protocols/m1-completion-v1.json','Cargo.toml','Cargo.lock','docs/AXON.md','ui/package.json','ui/bun.lock','ui/node_modules/typescript/package.json','ui/node_modules/typescript/lib/typescript.js'])await capture(file);
const parent=await fs.readFile(protocol.parent);assert.equal(sha(parent),protocol.parent_sha256);await fs.writeFile(path.join(output,'parent.axnum'),parent,{flag:'wx'});
const matchedBytes=await fs.readFile('.axon-memory/m1-math-actor-source-v1/episodes.json'),ancestor=JSON.parse(await fs.readFile('.axon-memory/m1-math-actor-v1/freeze.json'));assert.equal(sha(matchedBytes),ancestor.episodes_sha256);
const matched=JSON.parse(matchedBytes);assert.equal(matched.length,629);
await write('freeze-before-labels.json',{protocol_sha256:sha(protocolBytes),sources,parent_sha256:sha(parent),source_sha256:sha(raw),partition_sha256:sha(partitionBytes),matched_episodes_sha256:sha(matchedBytes),answer_accesses:{...accesses},scope:'New source/runtime/protocol fixed before reinterpreting train annotations with512-word interface. Raw JSON parsed in host RAM, not OS label isolation. Matched cohort and development already consumed, reserve not opened.'});
function convert(e){return {id:e.id,question_sha256:e.question_sha256,shape:e.shape,full:{source:e.full,examples:pointerExamples(e.full)},free:{source:e.free,examples:pointerExamples(e.free)}};}
const cohorts={matched:matched.map(convert),extended:[]},refusals={};
for(const row of partition){if(row.split!=='train')continue;try{const e=teaching(records[row.ordinal].question,records[row.ordinal].answer,512);cohorts.extended.push(convert({id:String(row.ordinal),question_sha256:row.question_sha256,shape:e.key,full:episode(e,false,544),free:episode(e,true,544)}));}catch(e){refusals[e.message]=(refusals[e.message]??0)+1;}}
assert.equal(accesses.train,4429);assert.equal(accesses.development+accesses.reserved+accesses.research_only,0);
const extendedIds=new Map(cohorts.extended.map(e=>[e.id,e]));for(const e of cohorts.matched){assert(extendedIds.has(e.id),'Extended eligibility must include every matched episode');assert.deepEqual(extendedIds.get(e.id).full.examples,e.full.examples);assert.deepEqual(extendedIds.get(e.id).free.examples,e.free.examples);}
await write('cohorts.json',cohorts);
const plans=[protocol.untrained_control,...protocol.policies];
for(const [i,policy]of plans.entries()){const examples=cohorts[policy.cohort].flatMap(e=>e[policy.numbers_only?'free':'full'].examples);assert(examples.length<=10000);await write(`policy-${i}-train.json`,{budget:{...protocol.budget,...(policy.epochs===0?{epochs:0}:{})},examples});}
const inputBytes=await fs.readFile('.axon-memory/m1-math-actor-source-v1/development-inputs.json');assert.equal(sha(inputBytes),ancestor.inputs_sha256);await fs.writeFile(path.join(output,'development-inputs.json'),inputBytes,{flag:'wx'});
const library=JSON.parse(await fs.readFile(path.join(protocol.primitive_source,'library.json')));assert.equal(library.length,4);const primitives=[];for(const p of library){assert(p.admitted);const bytes=await fs.readFile(p.file),file=`${p.id}.axcg`;await fs.writeFile(path.join(output,file),bytes,{flag:'wx'});primitives.push({...p,file,sha256:sha(bytes),ancestor_file:p.file,source_study:protocol.primitive_source});}
await write('library.json',primitives);
const audit={schema:'axon-relational-pointer-source/v1',cohorts:Object.fromEntries(Object.entries(cohorts).map(([name,rows])=>[name,{episodes:rows.length,frames:rows.reduce((n,r)=>n+r.full.examples.length,0),maximum_words:Math.max(...rows.flatMap(r=>r.full.examples.map(e=>e.input.text.split(' ').length)))}])),training_source:4429,extended_refusals:refusals,answer_accesses:accesses,development_inputs:1481,primitive_ancestry:'Four supplied meanings, learned ancestral circuit bytes copied unchanged; not new arithmetic discovery.',m1_complete:false,m2_permitted:false};
await write('audit.json',audit);console.log(JSON.stringify(audit));
