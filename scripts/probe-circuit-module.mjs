import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const [executable,outputPath,protocolFile='evals/protocols/m1-circuit-module-v1.json']=process.argv.slice(2);
if(!outputPath)throw Error('usage: node scripts/probe-circuit-module.mjs <candidate.exe> <new-output> [protocol]');
const output=path.resolve(outputPath);assert(output.startsWith(path.resolve()+path.sep));await fs.mkdir(output);
const write=(name,data)=>fs.writeFile(path.join(output,name),JSON.stringify(data,null,2),{flag:'wx'});
const sha=b=>createHash('sha256').update(b).digest('hex');
const protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);assert.equal(protocol.schema,'axon-compiled-circuit-module/v1');assert.equal(protocol.demo_cases.length,2);
const binary=path.join(output,'candidate.exe');await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
const sources={};
for(const file of ['src/demand.rs','src/demand_cli.rs','src/genome.rs','src/genome_cli.rs','src/cli.rs','src/numeric.rs','src/numeric_cli.rs','src/numeric_execution.rs','src/numeric_constructive.rs','src/numeric_evolution.rs','src/numeric_library.rs','tests/numeric_contracts.rs','tests/genome_contracts.rs','Cargo.toml','Cargo.lock','scripts/probe-circuit-module.mjs','evals/protocols/m1-completion-v1.json',protocolFile]) {
  const bytes=await fs.readFile(file);sources[file]=sha(bytes);const destination=path.join(output,'source',file);await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,bytes,{flag:'wx'});
}
const teaching={};
for(const c of protocol.demo_cases){assert(/^[a-z0-9_-]{1,32}$/u.test(c.name));const bytes=await fs.readFile(c.training);teaching[c.name]={source:c.training,sha256:sha(bytes)};await fs.writeFile(path.join(output,`${c.name}-teaching.tsv`),bytes,{flag:'wx'});}
await write('freeze.json',{created_at:new Date().toISOString(),binary_sha256:sha(await fs.readFile(binary)),protocol_sha256:sha(protocolBytes),sources,teaching,scope:'Fixed numerical regression cases frozen before fitting. No public word-problem answers or independent language benchmark.'});
const calls=[];
async function core(name,args,success=true){
  const started=performance.now();let stdout='',stderr='',code=0;
  try{const child=await promisify(execFile)(binary,args,{cwd:output,windowsHide:true,timeout:300000,maxBuffer:8000000});stdout=child.stdout;stderr=child.stderr;}
  catch(error){stdout=error.stdout??'';stderr=error.stderr??'';code=error.code;assert(Number.isInteger(code)&&code>0&&!error.killed&&!error.signal,'A spawn/timeout failure is not a numerical refusal');}
  await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});
  const fields=Object.create(null);for(const line of stdout.trim().split(/\r?\n/u)){const at=line.indexOf('=');if(at>0){const key=line.slice(0,at);assert(!Object.hasOwn(fields,key),'Ambiguous report key');fields[key]=line.slice(at+1);}}
  const call={order:calls.length,name,arguments:args,exit_code:code,whole_process_ms:performance.now()-started,fields};calls.push(call);await write(`call-${call.order}.json`,call);
  assert.equal(code===0,success,`Unexpected core status for ${name}: ${stderr}`);return call;
}
const results=[],registry=path.join(output,'registry');await fs.mkdir(registry);const manifest=[['id','kind','file','input','output','cost']];
for(const c of protocol.demo_cases){
  const directory=path.join(output,c.name);await fs.mkdir(directory);const skill=path.join(directory,'source.axnum');
  const trained=await core(`${c.name}-fit`,['numeric-fit',path.join(output,`${c.name}-teaching.tsv`),'--name',`source-only-${c.name}`,'--context',c.context,'--skill',skill,'--max-cost',String(c.max_cost),'--max-terms',String(c.max_terms)]);
  const exported=await core(`${c.name}-export`,['circuit-export',skill,'--context',c.context,'--output',path.join(directory,'exported')]);
  const page=await fs.readFile(path.join(directory,'exported/circuit.axcg'));assert(!page.includes(Buffer.from(c.context)));
  await fs.writeFile(path.join(registry,`${c.name}.axcg`),page,{flag:'wx'});const cost=Number(exported.fields.program_instructions);assert(Number.isSafeInteger(cost)&&cost>0&&cost<=256);
  manifest.push([c.name,'circuit',`${c.name}.axcg`,'numbers','number',String(cost)]);
  const queries=[];
  for(const [index,q]of c.queries.entries()){
    assert(q.values.length>0&&q.values.length<=4&&q.values.every(Number.isFinite)&&Number.isFinite(q.expected));
    const values=q.values.join(','),old=await core(`${c.name}-${index}-old`,['numeric-predict',skill,'--context',c.context,'--values',values]);
    const dna=await core(`${c.name}-${index}-dna`,['circuit-predict',path.join(directory,'exported/circuit-number.txt'),values]);
    const a=Number(old.fields.value),b=Number(dna.fields.value);assert(Number.isFinite(a)&&Number.isFinite(b));assert.equal(a,b);assert(Math.abs(b-q.expected)<=1e-9*(1+Math.abs(q.expected)));
    queries.push({values:q.values,expected:q.expected,old_call:old.order,dna_call:dna.order,value:b});
  }
  results.push({name:c.name,fit_call:trained.order,export_call:exported.order,source_bytes:(await fs.stat(skill)).size,circuit_bytes:page.length,circuit_digits:(await fs.stat(path.join(directory,'exported/circuit-number.txt'))).size,source_examples_omitted:Number(exported.fields.source_examples_omitted),queries});
}
await fs.writeFile(path.join(registry,'modules.tsv'),manifest.map(row=>row.join('\t')).join('\n')+'\n',{flag:'wx'});
await core('catalog',['seed-build',path.join(registry,'modules.tsv'),path.join(registry,'catalog.axseed')]);
const number=path.join(output,'brain.axnum');await core('pack',['genome-pack',path.join(registry,'catalog.axseed'),number]);const originalNumber=await fs.readFile(number);
const evidence=path.join(output,'original-registry-evidence');assert(path.dirname(registry)===output&&path.dirname(evidence)===output);assert(!(await fs.lstat(registry)).isSymbolicLink());
await fs.rename(registry,evidence);assert(!(await fs.readdir(output)).includes('registry'));
for(const c of protocol.demo_cases){const old=path.join(output,c.name,'source.axnum'),next=path.join(output,c.name,'source-evidence.axnum');assert(old.startsWith(output+path.sep)&&next.startsWith(output+path.sep));await fs.rename(old,next);}
for(const result of results)for(const [index,q]of result.queries.entries()){
  const call=await core(`${result.name}-${index}-born`,['genome-number',number,path.join(output,`born-${result.name}-${index}`),result.name,'axon/constructive-v1',q.values.join(',')]);
  assert.equal(Number(call.fields.value),q.value);assert.equal(call.fields.module_loads,'1');assert.equal(call.fields.active_modules,'1');assert.equal(call.fields.teacher_queries,'0');assert.equal(call.fields.training_required,'false');
  q.born_call=call.order;q.active_payload_bytes=Number(call.fields.active_payload_bytes);assert(q.active_payload_bytes>0);
}
const negatives=[];
for(const [name,id,context,values]of [['nan','calibration','axon/constructive-v1','NaN'],['many','calibration','axon/constructive-v1','1,2,3,4,5'],['context','calibration','wrong/context','1'],['id','absent','axon/constructive-v1','1'],['arity','calibration','axon/constructive-v1','1,2']]){
  const birth=path.join(output,`refused-${name}`),call=await core(`negative-${name}`,['genome-number',number,birth,id,context,values],false);
  let created=true;try{await fs.stat(birth);}catch(e){assert.equal(e.code,'ENOENT');created=false;}
  if(['nan','many'].includes(name))assert(!created);negatives.push({name,call:call.order,local_birth_created:created,scope:'Any local birth writes were explicitly authorized; no external action or private journal requested.'});
}
const bad=path.join(output,'bad-kind');await fs.mkdir(bad);await fs.copyFile(path.join(evidence,'calibration.axcg'),path.join(bad,'program.axcg'),fs.constants.COPYFILE_EXCL);
await fs.writeFile(path.join(bad,'modules.tsv'),'id\tkind\tfile\tinput\toutput\tcost\nbad\tnumeric\tprogram.axcg\tnumbers\tnumber\t1\n',{flag:'wx'});
await core('negative-kind',['seed-build',path.join(bad,'modules.tsv'),path.join(bad,'catalog.axseed')],false);
assert.deepEqual(await fs.readFile(number),originalNumber);const report={schema:'axon-circuit-module-study/v1',results,calls,negatives,numeric_queries:results.reduce((n,r)=>n+r.queries.length,0),number_sha256:sha(originalNumber),number_digits:originalNumber.length,original_registry_moved:true,source_skill_paths_removed:true,
  process_rss_bytes:null,energy_joules:null,public_math_training_started:false,m1_complete:false,m2_permitted:false,promotion:false,
  scope:'Learned programs transported without foundry witnesses through existing DNA, catalog and demand runtime. Same-author regression, no new theorem, word-problem solver or independent M1 approval. Active payload bytes exclude eager genome, process, decode workspace and stack. Files moved inside this experiment remain immutable evidence, not OS-isolated secrets.'};
await write('audit.json',report);console.log(JSON.stringify({queries:report.numeric_queries,results:results.map(r=>({name:r.name,source_bytes:r.source_bytes,circuit_bytes:r.circuit_bytes,omitted:r.source_examples_omitted})),number_digits:report.number_digits,calls:calls.length,m1_complete:false}));
