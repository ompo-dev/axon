import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createReadStream,existsSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {parse} from '../ui/node_modules/csv-parse/lib/index.js';

const workspace=await fs.realpath(fileURLToPath(new URL('../',import.meta.url)));
const root=await fs.realpath(path.resolve(process.argv[2]??''));
const relative=path.relative(workspace,root);
assert(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative),'audit requires an in-workspace campaign');
const modes=['cold','founder-random','learned-random','learned-active'];
const tasks=[0,1,2,4,8,9,10,11];
async function rows(name,delimiter=','){
  const file=path.join(root,name),stat=await fs.lstat(file);
  assert(stat.isFile()&&!stat.isSymbolicLink()&&stat.size<=64000000,'bounded regular artifact');
  return createReadStream(file).pipe(parse({columns:true,delimiter,max_record_size:2048}));
}
function oracle(task,x){
  const norm=Math.sqrt(x[0]*x[0]+x[1]*x[1]+(task>=8?x[2]*x[2]:0));
  const y=({0:()=>x[0]+x[1],1:()=>x[0]*x[1],2:()=>x[0]-x[1],4:()=>norm,
    8:()=>norm/x[3],9:()=>norm*x[3],10:()=>norm-x[3],11:()=>Math.sqrt((x[0]-x[2])**2+(x[1]-x[3])**2)})[task]();
  return Number.isFinite(y)?y:null;
}
function scalar(s){if(s==='domain')return null;assert(s.length&&Number.isFinite(Number(s)),'finite scalar');return Number(s);}
function near(a,b){return a===null||b===null?a===b:Math.abs(a-b)<=1e-9+1e-9*Math.abs(b);}
function fnv(bytes){let hash=0xcbf29ce484222325n;for(const byte of bytes)hash=BigInt.asUintN(64,(hash^BigInt(byte))*0x100000001b3n);return hash.toString(16).toUpperCase().padStart(16,'0');}
const summaries=new Map();
for await(const row of await rows('results.csv')){
  assert(modes.includes(row.mode)&&tasks.includes(Number(row.task)),'known arm and task');
  const key=`${row.mode}:${row.task}`;assert(!summaries.has(key),'one summary per paired task');summaries.set(key,row);
}
assert.equal(summaries.size,32);
const results=[];let checked=0;
for(let arm=0;arm<modes.length;arm++)for(const task of tasks){
  const mode=modes[arm],summary=summaries.get(`${mode}:${task}`),counts={correct:0,domain_refusals:0,unavailable:0,wrong:0};let index=0;
  for await(const row of await rows(`predictions-${arm}-${task}.tsv`,'\t')){
    assert.equal(row.mode,mode);assert.equal(Number(row.task),task);assert.equal(Number(row.row),index++);
    const input=row.input.split(',').map(Number);assert.equal(input.length,task<5?2:4);assert(input.every(v=>Number.isFinite(v)&&Math.abs(v)<=1000000));
    const expected=oracle(task,input),prediction=scalar(row.prediction);
    assert(near(scalar(row.expected),expected),'stored oracle must match independent JS squared-sum oracle');
    if(row.status==='unavailable'){assert.equal(summary.admitted,'false');assert.equal(prediction,null);counts.unavailable++;}
    else if(row.status==='domain'){assert.equal(summary.admitted,'true');assert.equal(expected,null);assert.equal(prediction,null);counts.domain_refusals++;}
    else if(row.status==='correct'){assert.equal(summary.admitted,'true');assert.notEqual(expected,null);assert(near(prediction,expected),'prediction disagrees with independent oracle');counts.correct++;}
    else {assert.equal(row.status,'wrong');assert(!near(prediction,expected),'wrong record must be an actual mismatch');counts.wrong++;}
  }
  assert.equal(index,Number(summary.heldout));assert(index>=100&&index<=10000);
  for(const [key,value] of Object.entries(counts))assert.equal(value,Number(summary[key]),`summary ${key}`);
  assert.equal(counts.wrong,0);checked+=index;results.push({mode,task,heldout:index,...counts});
}
const offerRows=[];for await(const row of await rows('file-demo/offers.csv'))offerRows.push(row);
assert.equal(offerRows.length,2);assert(offerRows.every(r=>r.item==='alpha'&&r.unit==='pecas'&&Number(r.available)<=16));
const offersHash=fnv(await fs.readFile(path.join(root,'file-demo/offers.csv')));let plans=0;
for await(const p of await rows('plans.csv')){
  const snapshot=await fs.readFile(path.join(root,`file-demo/stock-${Number(p.stock)}.csv`));assert.equal(fnv(snapshot),p.stock_hash);assert.equal(offersHash,p.offers_hash);
  const needed=20*3-Number(p.stock);assert.equal(Number(p.needed),needed);
  let best=Infinity;
  for(let a=0;a<=Number(offerRows[0].available);a++)for(let b=0;b<=Number(offerRows[1].available);b++){
    const units=a*Number(offerRows[0].units)+b*Number(offerRows[1].units),cost=a*Number(offerRows[0].price_cents)+b*Number(offerRows[1].price_cents);
    if(units>=needed&&cost<=30000)best=Math.min(best,cost);
  }
  assert.equal(Number(p.cost_cents),best);assert(Number(p.units)>=needed);plans++;
}
assert.equal(plans,3);
const archived=path.join(root,'evaluated-cumulative.exe');
const binary=await fs.readFile(existsSync(archived)?archived:path.join(workspace,'target/release/axon-uic-cumulative-lab'+(process.platform==='win32'?'.exe':'')));
const receipt={schema:'axon-cumulative-independent-audit-v1',checked,plans,wrong:0,binary_sha256:createHash('sha256').update(binary).digest('hex'),results};
const receiptPath=path.join(root,'independent-audit.json');
if(existsSync(receiptPath))assert.deepEqual(JSON.parse(await fs.readFile(receiptPath,'utf8')),receipt,'existing audit must agree; never overwrite it');
else await fs.writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({checked,plans,wrong:0,binary_sha256:receipt.binary_sha256}));
