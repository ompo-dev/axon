// Independently recompute the recorded verdicts, selection and costs without loading Rust.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '../ui/node_modules/csv-parse/lib/sync.js';
const root = path.resolve(process.argv[2] ?? '');
const table = async name => parse(await fs.readFile(path.join(root,name),'utf8'),{columns:true,delimiter:'\t'});
const validation = await table('validation.tsv');
assert.equal(new Set(validation.map(c=>c.id)).size,validation.length);
const cases = new Map(validation.map(c=>[c.id,c]));
const predictions = await table('predictions.tsv');
const arms = {};
const near = (a,b) => Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-9+1e-9*Math.max(Math.abs(a),Math.abs(b));
for (const row of predictions) {
  const c = cases.get(row.id); assert(c);
  assert.equal(row.expected,c.expected); assert.equal(row.expected_refusal,c.refusal);assert.equal(row.text,c.text);assert.equal(row.family,c.family);
  const answer = row.prediction === '' ? null : Number(row.prediction);
  if(answer!==null)assert(Number.isFinite(answer));
  const ok = answer===null ? c.expected===''&&row.status===c.refusal : c.expected!==''&&row.status==='answered'&&near(answer,Number(c.expected));
  assert.equal(row.correct,String(ok));
  const arm = arms[row.arm] ??= {checked:0,correct:0,wrong:0,refused:0,families:{},latencies:[],ids:new Set()};
  assert(!arm.ids.has(row.id));arm.ids.add(row.id);arm.checked++;arm.correct+=Number(ok);arm.wrong+=Number(!ok&&answer!==null);arm.refused+=Number(answer===null);
  const family=arm.families[c.family]??={checked:0,correct:0};family.checked++;family.correct+=Number(ok);arm.latencies.push(Number(row.latency_ns));
}
for(const arm of Object.values(arms)){
  assert.equal(arm.checked,validation.length);assert.equal(arm.ids.size,validation.length);delete arm.ids;
  arm.latencies.sort((a,b)=>a-b);arm.narrative_p50_ns=arm.latencies[Math.floor(arm.latencies.length*.5)];arm.narrative_p95_ns=arm.latencies[Math.floor(arm.latencies.length*.95)];delete arm.latencies;
}
const summary=Object.fromEntries((await fs.readFile(path.join(root,'result.txt'),'utf8')).trim().split('\n').map(line=>{const at=line.indexOf('=');return [line.slice(0,at),line.slice(at+1)];}));
const winners=Object.entries(arms).filter(([name])=>name.startsWith('winner-')).map(([,arm])=>arm);
for(const key of ['checked','correct','wrong','refused'])assert.equal(Number(summary[key]),winners.reduce((n,arm)=>n+arm[key],0));
for(const [key,family,metric] of [['natural_correct','natural-probe','correct'],['natural_checked','natural-probe','checked'],['negative_correct','negative','correct'],['negative_checked','negative','checked']])assert.equal(Number(summary[key]),winners.reduce((n,arm)=>n+arm.families[family][metric],0));
const population=await table('population.tsv');assert.equal(population.length,768);
const control=await table('random-control.tsv');assert.equal(control.length,population.length);
assert(control.every(r=>Number(r.checked)===96&&Number(r.correct)<=96));
assert.equal(new Set(population.map(c=>c.candidate)).size,768);
const bitcount = n => n.toString(2).replaceAll('0','').length;
const phenotypes = new Map(), generations = new Map();
for (const row of population){
  const bytes=await fs.readFile(path.join(root,`population/candidate-${row.candidate}.axsem`));
  assert.equal(bytes.length,Number(row.encoded_bytes));assert.equal(bytes.subarray(0,8).toString(),'AXSM0001');assert.equal(bytes[8],Number(row.rules));assert.equal(Number(row.rules),bitcount(BigInt(row.bits)));
  const sha=createHash('sha256').update(bytes).digest('hex');
  if(phenotypes.has(row.genotype))assert.equal(phenotypes.get(row.genotype),sha);else phenotypes.set(row.genotype,sha);
  const g=`${row.campaign}:${row.generation}`;if(!generations.has(g))generations.set(g,[]);generations.get(g).push(row);
}
for (const rows of generations.values()){
  assert.equal(rows.length,64);assert.equal(rows.filter(r=>r.selected==='true').length,8);
  for(let i=1;i<rows.length;i++){
    const a=rows[i-1],b=rows[i];assert(Number(a.correct)>Number(b.correct)||Number(a.correct)===Number(b.correct)&&(Number(a.wrong)<Number(b.wrong)||Number(a.wrong)===Number(b.wrong)&&Number(a.encoded_bytes)<=Number(b.encoded_bytes)));
  }
}
const checksums = {};
for(const name of ['lessons.tsv','selection.tsv','validation.tsv','independent.tsv','population.tsv','random-control.tsv','predictions.tsv','protocol.txt','result.txt'])checksums[name]=createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex');
const seeds=[];
const winnerCount=Object.keys(arms).filter(a=>a.startsWith('winner-')).length;
for(let i=0;i<winnerCount;i++){
  const digits=await fs.readFile(path.join(root,`winner-${i}/seed.axnum`),'utf8');assert(/^1(?:[0-9]{3})+$/.test(digits));
  const bytes=Buffer.from(digits.slice(1).match(/.../g).map(n=>Number(n)));assert.equal(bytes.subarray(0,8).toString(),'AXGN0001');
  for(const birth of [`birth-${i}`,`restart-${i}`])assert.deepEqual(await fs.readFile(path.join(root,birth,'relational.axsem')),await fs.readFile(path.join(root,`winner-${i}`,'relational.axsem')));
  seeds.push({binary_bytes:bytes.length,decimal_digits:digits.length,sha256:createHash('sha256').update(bytes).digest('hex')});
}
const result={verified:true,candidates:population.length,unique_structures:phenotypes.size,selection_predictions:768*96,random_control_predictions:768*96,arms,seeds,checksums,rss_bytes:null,external_llm_comparisons:null};
try { assert.deepEqual(JSON.parse(await fs.readFile(path.join(root,'audit.json'),'utf8')),result); }
catch(error) { if(error.code!=='ENOENT')throw error; await fs.writeFile(path.join(root,'audit.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'}); }
console.log(JSON.stringify(result,null,2));
