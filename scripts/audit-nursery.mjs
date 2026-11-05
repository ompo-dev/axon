import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '../ui/node_modules/csv-parse/lib/sync.js';
const base=path.resolve(process.argv[2]??'.axon-memory/observatory');
const readTable=async file=>parse(await fs.readFile(file,'utf8'),{columns:true,delimiter:'\t'});
const protocol=async file=>Object.fromEntries((await fs.readFile(file,'utf8')).trim().split('\n').map(line=>{const i=line.indexOf('=');return [line.slice(0,i),line.slice(i+1)];}));
const runs=[];
for(const entry of await fs.readdir(path.join(base,'runs'),{withFileTypes:true})){
  if(!entry.isDirectory())continue;
  const root=path.join(base,'runs',entry.name);
  try{const {job}=JSON.parse(await fs.readFile(path.join(root,'run.json'),'utf8'));if(job.mode==='nursery')runs.push({job,root,artifacts:path.join(root,'artifacts'),protocol:await protocol(path.join(root,'artifacts','protocol.txt'))});}
  catch(error){if(error.code!=='ENOENT')throw error;}
}
runs.sort((a,b)=>a.job.started-b.job.started);assert(runs.length);
const last=runs.at(-1);const scope=last.protocol.scope;const campaign=runs.filter(r=>r.protocol.scope===scope);
const order=new Map();const imported=new Set();const counts=[];let digits=1;
for(const run of campaign){
  const rows=await readTable(path.join(run.artifacts,'new-candidates.tsv'));
  for(const row of rows){
    assert(/^(0|[1-9][0-9]*)$/.test(row.dna));assert(!order.has(row.dna),'DNA was evaluated twice');
    assert(Number(row.digits)>=digits);digits=Number(row.digits);assert(row.dna.length<=digits);
    assert(BigInt(row.dna)<=BigInt(run.protocol.max_interpretable_dna));
    if(row.parent!=='_'){
      assert(order.has(row.parent),'parent not previously evaluated');
      const a=row.parent.padStart(digits,'0'),b=row.dna.padStart(digits,'0');assert.equal([...a].filter((c,i)=>c!==b[i]).length,1,'mutation is not one decimal position');
    }
    for(const key of ['checked','correct','wrong','bytes','elapsed_ns'])assert(/^[0-9]+$/.test(row[key]));
    assert(Number(row.correct)<=Number(row.checked));assert(Number(row.wrong)<=Number(row.checked)-Number(row.correct));
    const {digits:_,...entry}=row;order.set(row.dna,entry);
  }
  counts.push({id:run.job.id,status:run.job.status,new_candidates:rows.length});
}
assert.equal(last.job.status,'complete');const final=await readTable(path.join(last.artifacts,'population.tsv'));
assert.equal(final.length,order.size);assert.equal(new Set(final.map(r=>r.dna)).size,final.length);
for(const row of final)assert.deepEqual(row,order.get(row.dna));
const maximum=Number(last.protocol.max_interpretable_dna);assert(maximum<=1_000_000);
for(let i=0;i<=maximum;i++)if(!order.has(String(i)))imported.add(String(i));
assert.equal(imported.size,last.job.result.imported);
assert.equal(last.job.result.unique,final.length+imported.size+last.job.result.unfinished);
assert.equal(last.job.result.candidates,final.length);
// Match omitted DNA against the old pilot, not invented current-protocol fitness.
const historical=new Set();
for(const entry of await fs.readdir(path.join(base,'runs'),{withFileTypes:true})){
  if(!entry.isDirectory())continue;const root=path.join(base,'runs',entry.name);
  try{
    const {job}=JSON.parse(await fs.readFile(path.join(root,'run.json'),'utf8'));if(job.mode!=='semantic')continue;
    const artifacts=path.join(root,'artifacts');const rows=await readTable(path.join(artifacts,'population.tsv'));
    for(const row of rows)if(row.bits!==undefined)historical.add(row.bits);
    const consolidation=await protocol(path.join(artifacts,'consolidation.txt')).catch(e=>{if(e.code==='ENOENT')return null;throw e;});
    if(consolidation&&Number(consolidation.rules)===15)historical.add('32767');
  }catch(error){if(error.code!=='ENOENT')throw error;}
}
for(const dna of imported)assert(historical.has(dna),'excluded DNA has no recorded legacy source');
for(const dna of order.keys())assert(!historical.has(dna),'a historical genotype was reevaluated');
const best=[...order.values()].filter(r=>Number(r.bytes)<=131072&&Number(r.elapsed_ns)<=1e9).sort((a,b)=>Number(b.correct)-Number(a.correct)||Number(a.wrong)-Number(b.wrong)||Number(a.bytes)-Number(b.bytes)||Number(a.elapsed_ns)-Number(b.elapsed_ns)||Number(a.dna)-Number(b.dna))[0];
assert.equal(Number(best.correct),last.job.result.correct);assert.equal(Number(best.checked),last.job.result.checked);
assert.equal(last.job.result.admitted,Number(best.correct)===Number(best.checked)&&Number(best.wrong)===0);
const hashes={};for(const name of ['population.tsv','selection.tsv','protocol.txt','result.txt','registry-snapshot.axep','best-candidate/seed.axnum'])hashes[name]=createHash('sha256').update(await fs.readFile(path.join(last.artifacts,name))).digest('hex');
// Independently check a structural obstruction: a subset cannot introduce a
// matching first-clause template absent from its entire acquired library.
const bytes=await fs.readFile(path.join(last.artifacts,'library.axsem'));assert.equal(bytes.subarray(0,8).toString(),'AXSM0001');
let at=8;
const take=n=>{assert(n>=0&&at+n<=bytes.length-8);const b=bytes.subarray(at,at+n);at+=n;return b;};
const byte=()=>take(1)[0];
const source=()=>{const tag=byte();if(tag===0)take(1);else assert(tag===1||tag===2);};
const cell=()=>{source();source();};
const program=()=>{const n=take(4).readUInt32LE();assert(n<=32768);take(n);};
const patterns=[];const ruleCount=byte();assert(ruleCount<=64);
for(let i=0;i<ruleCount;i++){
  const pattern=[];const n=byte();assert(n<=32);
  for(let j=0;j<n;j++){const size=take(2).readUInt16LE();assert(size<=128);pattern.push(size?take(size).toString('utf8'):null);}
  patterns.push(pattern);const action=byte();
  if(action===0){const effects=byte();assert(effects<=4);for(let j=0;j<effects;j++){cell();source();assert(byte()<=1);program();}}
  else if(action===1){const cells=byte();assert(cells<=4);for(let j=0;j<cells;j++)cell();program();}
  else if(action===2){const args=byte();assert(args<=4);for(let j=0;j<args;j++)source();program();}
  else assert.equal(action,3);
}
assert.equal(at,bytes.length-8);
const probes=(await readTable(path.join(last.artifacts,'selection.tsv'))).filter(c=>c.family==='natural-probe');
const unsupported=probes.filter(c=>{const words=c.text.split(/[.!?]/)[0].trim().toLowerCase().split(/\s+/);return !patterns.some(p=>p.length===words.length&&p.every((v,i)=>v===null||v===words[i]));}).map(c=>c.id);
assert.equal(unsupported.length,5);
const result={schema:'nursery-audit-v2',verified:true,scope,unique:final.length+imported.size,new_evaluations:order.size,imported_history:imported.size,duplicates:0,unconfirmed:last.job.result.unfinished,digits,growths:last.job.result.growths,predictions:order.size*Number(best.checked),best_dna:best.dna,best_correct:Number(best.correct),checked:Number(best.checked),all_tests_passed:last.job.result.admitted,unsupported_first_clauses:unsupported,runs:counts,hashes};
const output=path.join(last.artifacts,'audit-nursery-v2.json');
try{assert.deepEqual(JSON.parse(await fs.readFile(output,'utf8')),result);}catch(error){if(error.code!=='ENOENT')throw error;await fs.writeFile(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});}
console.log(JSON.stringify(result,null,2));
