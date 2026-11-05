import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const [run,output]=process.argv.slice(2);
if(!output)throw Error('usage: node scripts/audit-circuit-module.mjs <frozen-study> <new-audit.json>');
const sha=b=>createHash('sha256').update(b).digest('hex');
const read=async file=>JSON.parse(await fs.readFile(path.join(run,file)));
const freeze=await read('freeze.json'),report=await read('audit.json');
assert.equal(sha(await fs.readFile(path.join(run,'candidate.exe'))),freeze.binary_sha256);
for(const [file,hash]of Object.entries(freeze.sources))assert.equal(sha(await fs.readFile(path.join(run,'source',file))),hash);
assert.equal(sha(await fs.readFile('evals/protocols/m1-completion-v1.json')),freeze.sources['evals/protocols/m1-completion-v1.json']);
assert.equal(report.schema,'axon-circuit-module-study/v1');assert.equal(report.calls.length,36);
for(const [order,call]of report.calls.entries()){
  assert.equal(order,call.order);assert.deepEqual(call,await read(`call-${order}.json`));
  const stdout=await fs.readFile(path.join(run,`${call.name}.stdout.log`),'utf8'),fields=Object.create(null);
  for(const line of stdout.split(/\r?\n/u)){const match=/^([^=]+)=(.*)$/u.exec(line);if(match){assert(!Object.hasOwn(fields,match[1]));fields[match[1]]=match[2];}}
  assert.deepEqual({...fields},call.fields);assert.equal(call.exit_code,order<30?0:2);
}
assert.equal(report.results.length,2);let queries=0;
for(const result of report.results){
  const page=await fs.readFile(path.join(run,result.name,'exported/circuit.axcg'));
  const decimal=await fs.readFile(path.join(run,result.name,'exported/circuit-number.txt'),'utf8');
  assert.equal(decimal,'1'+[...page].map(v=>String(v).padStart(3,'0')).join(''));
  assert.equal(page.subarray(0,8).toString(),'AXCG0001');assert.equal(page.length,result.circuit_bytes);
  assert.equal((await fs.stat(path.join(run,result.name,'source-evidence.axnum'))).size,result.source_bytes);
  assert(page.length<result.source_bytes);assert(result.source_examples_omitted>0);
  await assert.rejects(fs.stat(path.join(run,result.name,'source.axnum')),{code:'ENOENT'});
  for(const q of result.queries){
    const calls=[q.old_call,q.dna_call,q.born_call].map(i=>report.calls[i]);
    assert.deepEqual(calls.map(c=>c.arguments.at(-1)),[q.values.join(','),q.values.join(','),q.values.join(',')]);
    const expected=result.name==='calibration'?2*q.values[0]+1:Math.hypot(...q.values);
    for(const call of calls){const value=Number(call.fields.value);assert(Number.isFinite(value));assert(Math.abs(value-expected)<=1e-9*(1+Math.abs(expected)));}
    assert.equal(Number(calls[2].fields.active_payload_bytes),q.active_payload_bytes);
    for(const [key,value]of Object.entries({teacher_queries:'0',module_loads:'1',active_modules:'1',training_required:'false'}))assert.equal(calls[2].fields[key],value);
    queries++;
  }
}
assert.equal(queries,8);const number=await fs.readFile(path.join(run,'brain.axnum'));
assert.equal(sha(number),report.number_sha256);assert.equal(number.length,report.number_digits);
await assert.rejects(fs.stat(path.join(run,'registry')),{code:'ENOENT'});
for(const negative of report.negatives){
  const call=report.calls[negative.call];assert.equal(call.exit_code,2);
  const exists=await fs.stat(call.arguments[2]).then(()=>true,e=>{assert.equal(e.code,'ENOENT');return false;});
  assert.equal(exists,negative.local_birth_created);
}
assert.equal(report.public_math_training_started,false);assert.equal(report.m1_complete,false);assert.equal(report.m2_permitted,false);
const audit={schema:'axon-circuit-module-recount/v1',verified:true,binary_sha256:freeze.binary_sha256,queries,calls:report.calls.length,number_sha256:sha(number),number_digits:number.length,active_payload_bytes:report.results.map(r=>({name:r.name,bytes:r.queries[0].active_payload_bytes})),m1_complete:false,m2_permitted:false,scope:'Independent log/formula recount, not new held-out language evidence. Arithmetic and serialization checks are not certificates of semantic context or private-constant removal.'};
await fs.writeFile(output,JSON.stringify(audit,null,2),{flag:'wx'});console.log(JSON.stringify(audit));
