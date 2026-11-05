import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from '../ui/node_modules/csv-parse/lib/sync.js';

const root=path.resolve(process.argv[2]);
const table=async name=>parse(await fs.readFile(path.join(root,name),'utf8'),{columns:true,delimiter:'\t'});
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
function fnv(bytes){let h=0xcbf29ce484222325n;for(const b of bytes)h=BigInt.asUintN(64,(h^BigInt(b))*0x100000001b3n);return h;}
function reader(bytes){let at=0;return {take(n){assert(Number.isInteger(n)&&n>=0&&at+n<=bytes.length);const b=bytes.subarray(at,at+n);at+=n;return b;},u8(){return this.take(1)[0];},u32(){return this.take(4).readUInt32LE();},u64(){return this.take(8).readBigUInt64LE();},f64(){return this.take(8).readDoubleLE();},text(){return new TextDecoder('utf-8',{fatal:true}).decode(this.take(this.u32()));},done(){assert.equal(at,bytes.length);}};}
function decimal(s){assert(/^1[0-9]+$/.test(s)&&s.length%3===1);const bytes=[];for(let i=1;i<s.length;i+=3){const b=Number(s.slice(i,i+3));assert(b<=255);bytes.push(b);}return Buffer.from(bytes);}
function program(r){const arity=r.u8(),n=r.u32(),ops=[];assert(arity>=1&&arity<=4&&n<=256);for(let i=0;i<n;i++){const op=r.u8();assert(op<=6);ops.push([op,op===0?r.u8():op===1?r.f64():null]);}return {arity,ops};}
function graph(r,tools){const arity=r.u8(),n=r.u8(),genes=[];assert(arity>=1&&arity<=4&&n>=1&&n<=64);for(let i=0;i<n;i++){const op=r.u8(),args=[r.u8(),r.u8(),r.u8(),r.u8()];assert(op<5+tools.length);const k=op===4?1:op<4?2:tools[op-5].arity;assert(args.slice(0,k).every(a=>a<arity+4+i));assert(args.slice(k).every(a=>a===0));if(op===0||op===2)assert(args[0]<=args[1]);genes.push({op,args,k});}return {arity,genes};}
function primitive(op,a,b){let v;switch(op){case 0:v=a+b;break;case 1:v=a-b;break;case 2:v=a*b;break;case 3:v=b===0?NaN:a/b;break;case 4:v=a<0?NaN:Math.sqrt(a);break;default:throw Error('opcode');}if(v===0&&a!==0&&b!==0&&(op===2||op===3))return NaN;return Number.isFinite(v)?v:NaN;}
function runProgram(p,x){const stack=[];for(const [op,arg] of p.ops){if(op===0)stack.push(x[arg]);else if(op===1)stack.push(arg);else if(op===6)stack.push(primitive(4,stack.pop()));else{const b=stack.pop(),a=stack.pop();stack.push(primitive(op-2,a,b));}}assert.equal(stack.length,1);return stack[0];}
function signals(g,tools,x){assert.equal(x.length,g.arity);const s=[...x,-1,0,1,2];for(const gene of g.genes){const a=gene.args.map(i=>s[i]);s.push(gene.op<5?primitive(gene.op,a[0],a[1]):runProgram(tools[gene.op-5],a.slice(0,gene.k)));}return s;}
const agrees=(a,b)=>Number.isFinite(a)&&Math.abs(a-b)<=1e-9+1e-9*Math.abs(b);
const inputs=row=>Object.entries(row).filter(([k])=>/^x[0-3]$/.test(k)).map(([,v])=>Number(v));
const truth=(i,x)=>[()=>x[0]*x[0],()=>Math.hypot(x[0],x[1]),()=>Math.hypot(x[0]-x[2],x[1]-x[3]),()=>Math.hypot(x[0],x[1])/Math.hypot(x[2],x[3]),()=>(x[0]+x[1])*(x[2]-x[3]),()=>Math.sin(x[0]),()=>x[0]+x[1],()=>x[0]-x[1],()=>x[0]*x[1],()=>x[0]/x[1]][i]();
const protocol=Object.fromEntries((await fs.readFile(path.join(root,'protocol.txt'),'utf8')).trim().split('\n').filter(s=>s.includes('=')).map(s=>[s.slice(0,s.indexOf('=')),s.slice(s.indexOf('=')+1)]));
const results=await table('results.tsv'),audited=[],hashes={};assert.equal(results.length,protocol.curriculum==='arithmetic'?10:6);
let attempts=0,totalGraphs=0;
for(const row of results){
  const i=Number(row.case),encoded=await fs.readFile(path.join(root,`task-${i}-circuit.txt`),'utf8'),bytes=decimal(encoded);
  assert.equal(bytes.subarray(0,8).toString(),'AXCG0001');assert.equal(fnv(bytes.subarray(0,-8)),bytes.readBigUInt64LE(bytes.length-8));
  assert.equal(bytes.length,Number(row.dna_bytes));assert.equal(encoded.length,Number(row.dna_digits));
  const r=reader(bytes.subarray(8,-8)),tools=[];const n=r.u8();assert(n<=8);for(let j=0;j<n;j++)tools.push(program(r));
  const g=graph(r,tools),output=r.u8();r.done();assert(output<g.arity+4+g.genes.length);
  const graphKey=Buffer.from([g.arity,g.genes.length,...g.genes.flatMap(gene=>[gene.op,...gene.args])]);
  const train=await table(`task-${i}-training.tsv`),admission=await table(`task-${i}-admission.tsv`),final=await table(`task-${i}-final.tsv`),predictions=await table(`task-${i}-predictions.tsv`);
  assert.equal(train.length,16);assert.equal(admission.length,16);assert.equal(final.length,1024);assert.equal(predictions.length,1024);
  for(const set of [train,admission,final])for(const example of set)assert(agrees(truth(Number(protocol[`task-${i}-family`]??i),inputs(example)),Number(example.output)));
  assert(admission.every(a=>!train.some(t=>JSON.stringify(inputs(a))===JSON.stringify(inputs(t)))));
  const counters={correct:0,wrong:0,refused:0};
  for(let j=0;j<final.length;j++){
    const expected=Number(final[j].output),actual=signals(g,tools,inputs(final[j]))[output],verdict=!Number.isFinite(actual)?'refused':agrees(actual,expected)?'correct':'wrong';
    assert.equal(predictions[j].row,String(j));assert(agrees(Number(predictions[j].expected),expected));assert.equal(predictions[j].verdict,verdict);
    if(Number.isFinite(actual))assert(agrees(Number(predictions[j].actual),actual));else assert.equal(predictions[j].actual,'');counters[verdict]++;
  }
  for(const key of Object.keys(counters))assert.equal(counters[key],Number(row[`final_${key}`]));
  const trainCorrect=train.filter(e=>agrees(signals(g,tools,inputs(e))[output],Number(e.output))).length;
  assert.equal(trainCorrect,Number(row.training_correct));
  if(row.status==='admitted'){assert.equal(trainCorrect,16);assert(admission.every(e=>agrees(signals(g,tools,inputs(e))[output],Number(e.output))));}
  // Verify every durable reservation, hash link, result and selected fitness independently.
  const journal=await fs.readFile(path.join(root,`task-${i}-journal.axep`));assert.equal(journal.subarray(0,8).toString(),'AXEP0001');assert.equal(journal.readUInt32LE(8),1);assert.equal(journal.readUInt32LE(12),0xfffffffe);
  let at=16,previousHash=0n,id=0,genes=Number(protocol.initial_genes??2),growths=0,work=0n;const reserved=new Set(),completed=new Set();
  while(at<journal.length){
    const size=journal.readUInt32LE(at);assert(size<=32768);assert.equal(journal.readUInt32LE(at+4),(~size)>>>0);assert.equal(journal.readBigUInt64LE(at+8),previousHash);
    const end=at+16+size;assert(end+8<=journal.length);previousHash=fnv(journal.subarray(at,end));assert.equal(journal.readBigUInt64LE(end),previousHash);
    const payload=reader(journal.subarray(at+16,end));assert.equal(payload.u64(),BigInt(++id));payload.u64();payload.text();payload.text();assert.equal(payload.u8(),3);assert.equal(payload.text(),'axon/constructive-v1');const channel=payload.text(),fields={};const count=payload.u8();assert(count>0&&count<=16);
    for(let k=0;k<count;k++){const name=payload.text();assert.equal(payload.u8(),0);assert(!Object.hasOwn(fields,name));fields[name]=payload.text();}payload.done();at=end+8;
    if(channel==='metadata'){assert.equal(id,1);continue;}
    if(channel==='growth'){assert.equal(Number(fields.genes),++genes);assert(genes<=16);growths++;continue;}
    const key=fields.dna;const raw=decimal(key),gr=reader(raw),candidate=graph(gr,tools);gr.done();assert.equal(candidate.arity,g.arity);
    if(channel==='reservation'){assert.equal(candidate.genes.length,genes);assert(!reserved.has(key),'duplicate genotype reservation');reserved.add(key);continue;}
    assert.equal(channel,'result');assert(reserved.has(key)&&!completed.has(key));completed.add(key);
    const f=reader(decimal(fields.fitness)),errors=[];for(let k=0;k<32;k++)errors.push(f.f64());const loss=f.f64(),correct=f.u8(),out=f.u8(),active=f.u8();f.u64();f.u64();work+=f.u64();f.done();assert(active<=candidate.genes.length);
    let ok=0,sum=0;for(let k=0;k<train.length;k++){const expected=Number(train[k].output),v=signals(candidate,tools,inputs(train[k]))[out],error=Number.isFinite(v)?Math.min(Math.abs(v-expected)/(1+Math.abs(expected)),1e6):1e6;assert(Math.abs(errors[k]-error)<=1e-10);sum+=error;ok+=Number(agrees(v,expected));}
    assert.equal(correct,ok);assert(Math.abs(loss-sum/train.length)<=1e-10);assert(errors.slice(train.length).every(e=>e===0));
  }
  assert.equal(completed.size,Number(row.total_graphs));assert.equal(reserved.size,completed.size);assert.equal(work,BigInt(row.work));assert.equal(growths,Number(row.growths));
  assert([...completed].some(key=>decimal(key).equals(graphKey)),'exported circuit was not evaluated');
  attempts+=final.length;totalGraphs+=completed.size;audited.push({case:i,status:row.status,graphs:completed.size,growths,training_correct:trainCorrect,...counters,bytes:bytes.length,digits:encoded.length});
  for(const name of [`task-${i}-circuit.txt`,`task-${i}-journal.axep`,`task-${i}-final.tsv`,`task-${i}-predictions.tsv`])hashes[name]=hash(await fs.readFile(path.join(root,name)));
}
for(const name of ['protocol.txt','results.tsv','result.txt'])hashes[name]=hash(await fs.readFile(path.join(root,name)));
const result={schema:'constructive-audit-v1',verified:true,total_graphs:totalGraphs,duplicate_reservations:0,unfinished_reservations:0,independent_predictions:attempts,admitted_tasks:audited.filter(r=>r.status==='admitted').length,tasks:audited,hashes};
await fs.writeFile(path.join(root,'audit-constructive-v1.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({...result,hashes:undefined},null,2));
