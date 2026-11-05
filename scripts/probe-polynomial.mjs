import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {pathToFileURL} from 'node:url';
import {sha} from './gsm8k-source.mjs';

export function fixtures(protocol) {
  const tasks=[];
  for(let k=0;k<12;k++) {
    const c=[19+2*k,-13-k,7+k%5,(k%2?-1:1)*(3+k)];
    const f=x=>((c[3]*x+c[2])*x+c[1])*x+c[0];
    tasks.push({id:`s${k}`,terms:c.map((n,x)=>[n,x,0]),train:protocol.teaching.univariate_points.map(x=>({input:x,output:f(x)})),inputs:protocol.future.univariate_points,admission:[-4,3,8].map(input=>({input,output:f(input)}))});
  }
  for(let k=0;k<12;k++) {
    const terms=[];
    for(let d=0;d<=3;d++) for(let y=0;y<=d;y++) {
      const i=terms.length,c=(i%2?-1:1)*(3+((k+1)*(i+3)*7)%29);
      terms.push([c,d-y,y]);
    }
    const f=([x,y])=>terms.reduce((s,[c,a,b])=>s+c*x**a*y**b,0);
    tasks.push({id:`b${k}`,terms,train:protocol.teaching.bivariate_points.map(input=>({input,output:f(input)})),inputs:protocol.future.bivariate_points,admission:[[-1,2],[4,1],[3,-2]].map(input=>({input,output:f(input)}))});
  }
  return tasks;
}
export function exact(terms,input) {
  const [x,y]=Array.isArray(input)?input:[input,0];
  const n=terms.reduce((s,[c,a,b])=>s+BigInt(c)*BigInt(x)**BigInt(a)*BigInt(y)**BigInt(b),0n);
  const value=Number(n);assert(Number.isSafeInteger(value)&&BigInt(value)===n);return value;
}

async function main() {
  const [lab,out,previous]=process.argv.slice(2);assert(lab&&out&&previous);
  assert(path.resolve(out).startsWith(path.resolve()+path.sep));await fs.mkdir(out);
  const write=async(name,data)=>fs.writeFile(path.join(out,name),typeof data==='string'?data:JSON.stringify(data,null,2),{flag:'wx'});
  const read=async file=>JSON.parse(await fs.readFile(file));
  const protocolPath='evals/protocols/polynomial-proposal-v1.json',protocol=await read(protocolPath),tasks=fixtures(protocol);
  const sources={};for(const file of [protocolPath,'scripts/probe-polynomial.mjs','src/algorithm_polynomial.rs','src/algorithm_agent.rs','src/algorithm_lab.rs','src/structural_algorithm.rs','evals/curricula/algorithms-v3.json'])sources[file]=sha(await fs.readFile(file));
  const binary=sha(await fs.readFile(lab)),production=sha(await fs.readFile('target/release/axon-uic-curriculum-lab.exe'));
  const cases=[];
  for(const t of tasks) {
    for(const e of [...t.train,...t.admission])assert.equal(e.output,exact(t.terms,e.input));
    const exposed=new Set([...t.train,...t.admission].map(e=>JSON.stringify(e.input)));
    assert(t.inputs.every(v=>!exposed.has(JSON.stringify(v))));
    cases.push({id:t.id,expected:t.inputs.map(v=>exact(t.terms,v))});
  }
  const controls=[
    {id:'quartic',xs:[-3,-2,-1,0,1,2,3],f:x=>x**4},
    {id:'parity',xs:[0,1,2,3,4,5,6,7],f:x=>x%2},
    {id:'piecewise',xs:[-3,-2,-1,0,1,2,3],f:Math.abs},
    {id:'rational',xs:[-4,-2,0,2,4,6],f:x=>x/2},
    {id:'too-few',xs:[0,1],f:x=>x},
    {id:'coordinate',xs:[0,1,4097],f:()=>1},
    {id:'ambiguous',xs:[-1,0,1],f:x=>x**3},
    {id:'hidden-quartic',xs:[-1,0,1],f:x=>x**3+x*x*(x-1)*(x+1)},
  ].map(({id,xs,f})=>({id,train:xs.map(input=>({input,output:f(input)})),inputs:[2,7,11,19],inherit:false,representation:'polynomial'}));
  controls.push({id:'conflict',train:[{input:0,output:0},{input:1,output:1},{input:2,output:2},{input:1,output:9}],inputs:[7],inherit:false,representation:'polynomial'});
  controls.push({id:'shape',train:[{input:[0,1,2],output:1},{input:[1,2,3],output:1},{input:[2,3,4],output:1}],inputs:[[7,8,9]],inherit:false,representation:'polynomial'});
  const nativeInput=representation=>tasks.map(({id,train,inputs})=>({id,train,inputs,inherit:false,...(representation?{representation}:{} )}));
  await write('host-only-gold.json',cases);
  await write('enumerate-input.json',nativeInput());await write('polynomial-input.json',nativeInput('polynomial'));await write('controls-input.json',controls);
  const old=await read('evals/curricula/algorithms-v3.json'),grammar=old.lessons.find(t=>t.id==='algorithm-goal-0');assert(grammar);
  const base={version:1,id:'polynomial-proposal-v1',lessons:[grammar],cases:[{id:'grammar',family:'invoke',input:['goal','aplique','nova','em','lote'],expected:['invoke','nova','lote']}],dialogues:[]};
  await write('base.json',base);
  const lessons=tasks.map(t=>({id:t.id,family:'polynomial',examples:t.train,admission:t.admission,representation:'polynomial'}));
  lessons.push({id:'ambiguous',family:'counterexample',examples:controls.find(t=>t.id==='ambiguous').train,admission:[2,3,4,5].map(input=>({input,output:input**3})),representation:'polynomial'});
  lessons.push({id:'quartic',family:'negative-control',examples:controls[0].train,admission:[4,5,6].map(input=>({input,output:input**4})),representation:'polynomial'});
  await write('campaign.json',{version:1,lessons,requests:[{id:'observed',family:'observed-check',text:'aplique s0 em lote',bindings:{lote:0},expected:19}],changes:[],policy_trials:0,file_demo:null});
  await write('train.json',{...base,algorithms:'campaign.json'});
  const future=tasks.flatMap(t=>t.inputs.map((input,i)=>({id:`${t.id}-${i}`,family:'unseen',text:`aplique ${t.id} em lote`,bindings:{lote:input},expected:null})));
  future.push({id:'repaired',family:'refuted',text:'aplique ambiguous em lote',bindings:{lote:19},expected:null});
  await write('future-input.json',future);
  const frozen={};for(const file of await fs.readdir(out))frozen[file]=sha(await fs.readFile(path.join(out,file)));
  await write('freeze-before-execution.json',{binary_sha256:binary,previous_sha256:sha(await fs.readFile(previous)),source_hashes:sources,inputs:frozen,scope:'Labels/formulas only in host gold and source; future native payload contains values or null expected only. Fixed degree prior, developer authored tasks, no external authorship or global semantic proof.'});
  const calls=[];
  async function core(name,args,{exe=lab,telemetry=false}={}) {
    const start=performance.now();
    const {stdout,stderr}=await promisify(execFile)(path.resolve(exe),args,{env:{...process.env,AXON_LAB_TELEMETRY:telemetry?'1':'0'},timeout:240000,maxBuffer:32*1024*1024});
    const elapsed_ms=performance.now()-start;
    await write(`${name}-stdout.jsonl`,stdout);await write(`${name}-stderr.txt`,stderr);
    const lines=stdout.trim().split(/\r?\n/u).map(line=>JSON.parse(line));calls.push({name,args,elapsed_ms,binary_sha256:sha(await fs.readFile(exe))});
    return lines;
  }
  const file=n=>path.join(out,n);
  await core('cold',['structure',file('base.json'),file('cold')]);
  const number=file('cold/structural.axnum'),results={};
  for(const mode of ['enumerate','polynomial','controls']) {
    await core(mode,['algorithm-external',number,file(`${mode}-birth`),file(`${mode}-input.json`),file(`${mode}-predictions.json`)]);
    results[mode]=await read(file(`${mode}-predictions.json`));
  }
  await core('previous',['algorithm-external',number,file('previous-birth'),file('enumerate-input.json'),file('previous-predictions.json')],{exe:previous});
  const prior=await read(file('previous-predictions.json'));assert.deepEqual(results.enumerate,prior);
  await core('training',['structure',file('train.json'),file('trained')],{telemetry:true});
  const trained=file('trained/structural.axnum');
  await core('newborn',['algorithm-batch',trained,file('newborn-birth'),file('future-input.json'),file('newborn-predictions.json')]);
  await core('newborn-again',['algorithm-batch',trained,file('newborn-again-birth'),file('future-input.json'),file('newborn-again-predictions.json')]);
  const born=await read(file('newborn-predictions.json'));assert.deepEqual(born,await read(file('newborn-again-predictions.json')));
  const events=(await fs.readFile(file('training-stdout.jsonl'),'utf8')).trim().split(/\r?\n/u).map(JSON.parse).filter(r=>r.event==='algorithm_investigation');
  const scores={};for(const mode of ['enumerate','polynomial']) {
    const rows=results[mode].map((r,i)=>({...r,correct:r.predictions.filter((v,j)=>v===cases[i].expected[j]).length,total:cases[i].expected.length}));
    scores[mode]={correct:rows.reduce((n,r)=>n+r.correct,0),total:384,fully_correct_tasks:rows.filter(r=>r.correct===r.total).length,learned:rows.filter(r=>r.program).length,work:rows.reduce((n,r)=>n+r.work,0),candidates:rows.reduce((n,r)=>n+r.candidates,0),rows};
  }
  const gold=cases.flatMap(t=>t.expected);gold.push(6859);
  const newbornCorrect=born.filter((r,i)=>r.actual?.value===gold[i]||r.actual===gold[i]).length;
  const literal=tasks.reduce((n,t)=>n+t.inputs.filter(input=>t.train.some(e=>JSON.stringify(e.input)===JSON.stringify(input))).length,0);
  for(const [name,hash]of Object.entries(frozen))assert.equal(sha(await fs.readFile(file(name))),hash);
  for(const [name,hash]of Object.entries(sources))assert.equal(sha(await fs.readFile(name)),hash);
  assert.equal(production,sha(await fs.readFile('target/release/axon-uic-curriculum-lab.exe')));
  const audit={schema:'axon-polynomial-proposal/v1',binary_sha256:binary,protocol_sha256:sources[protocolPath],source_hashes:sources,scores,literal:{correct:literal,total:384,abstained:384-literal},controls:results.controls,newborn:{correct:newbornCorrect,total:gold.length,identical_repeated_birth:true,number_bytes:(await fs.stat(trained)).size,number_sha256:sha(await fs.readFile(trained))},events:{total:events.length,refutations:events.filter(e=>e.phase==='refutation').length,consolidations:events.filter(e=>e.phase==='consolidation').length},training:await read(file('trained/algorithm-teaching-result.json')),default_matches_previous:true,calls,production_unchanged:true,m1_complete:false,m2_permitted:false,limits:'Host-provided polynomial prior and exact solver, no self-invention of interpolation, no universal understanding or ETH performance claim. Fixed-width work counter is not wall time, RSS or energy. Experimental alternate proposal ignores inherited program libraries. Finite admission cannot prove extrapolation or exclude hidden terms.'};
  await write('audit.json',audit);
  console.log(JSON.stringify({scores:Object.fromEntries(Object.entries(scores).map(([k,v])=>[k,{...v,rows:undefined}])),literal:audit.literal,newborn:audit.newborn,events:audit.events,controls:audit.controls.map(r=>({id:r.id,error:r.error,predictions:r.predictions})),calls:calls.map(c=>({name:c.name,elapsed_ms:c.elapsed_ms}))}));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)await main();
