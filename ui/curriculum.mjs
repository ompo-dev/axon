import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const operations = new Set(['add','sub','mul','div','sqrt','sin','exp']);
const id = value => typeof value==='string' && /^[a-z0-9][a-z0-9_-]{0,63}$/.test(value);
const strict = (object, keys, label) => {
  if(!object || typeof object!=='object' || Array.isArray(object) || Object.keys(object).some(k=>!keys.includes(k)))throw Error(`Invalid ${label} fields`);
};
const integer = (n,min,max) => Number.isSafeInteger(n)&&n>=min&&n<=max;
const sha = value => createHash('sha256').update(value).digest('hex');
function random(seed){let x=(seed>>>0)||1;return n=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)%n;};}
function hashSeed(value){return parseInt(sha(value).slice(0,8),16);}
export function oracle(program,input){
  const stack=[];
  for(const op of program){
    if(typeof op==='number')stack.push(op);
    else if(/^x[0-3]$/.test(op))stack.push(input[Number(op[1])]);
    else if(['sqrt','sin','exp'].includes(op)){
      if(!stack.length)throw Error('Oracle stack underflow');
      const x=stack.pop();stack.push(op==='sqrt'?Math.sqrt(x):op==='sin'?Math.sin(x):Math.exp(x));
    }else{
      if(!operations.has(op))throw Error('Unknown oracle operation');
      if(stack.length<2)throw Error('Oracle stack underflow');
      const b=stack.pop(),a=stack.pop();stack.push(op==='add'?a+b:op==='sub'?a-b:op==='mul'?a*b:b===0?NaN:a/b);
    }
  }
  if(stack.length!==1)throw Error('Oracle must return one value');
  return stack[0];
}
const placeholders = template => [...template.matchAll(/\{([a-z][a-z0-9_]*)(?:\|(upper|lower))?\}/g)];
function render(template,values){return template.replace(/\{([a-z][a-z0-9_]*)(?:\|(upper|lower))?\}/g,(_,key,mode)=>{
  if(!Object.hasOwn(values,key))throw Error(`Unknown template field ${key}`);
  const value=String(values[key]);return mode==='upper'?value.toUpperCase():mode==='lower'?value.toLowerCase():value;
});}
export function prepareCurriculum(source,rng=75218341,cap=5000){
  if(!integer(rng,1,2147483647)||!integer(cap,1,20000))throw Error('Invalid RNG or candidate ceiling');
  strict(source,['schema','id','title','splits','budget','fields','stages'],'curriculum');
  if(source.schema!=='axon-curriculum/v1'||!id(source.id)||!Array.isArray(source.stages)||!integer(source.stages.length,1,1000)||!source.stages.some(s=>s.required===true))throw Error('Invalid curriculum schema or count');
  strict(source.splits,['train','admission','test','transfer'],'splits');
  for(const [key,max] of [['train',32],['admission',32],['test',512],['transfer',128]])if(!integer(source.splits[key],key==='transfer'?0:key==='test'?1:4,max))throw Error(`Invalid split ${key}`);
  strict(source.budget,['candidates_per_stage','total_candidates','work_per_stage','total_numeric_work','text_work_per_stage','semantic_work_per_stage'],'budget');
  const budget={...source.budget,candidates_per_stage:Math.min(source.budget.candidates_per_stage,cap)};
  for(const [key,min,max] of [['candidates_per_stage',1,20000],['total_candidates',1,1000000],['work_per_stage',1000,100000000],['total_numeric_work',1000,2000000000],['text_work_per_stage',1000,2000000],['semantic_work_per_stage',1000,20000000]])if(!integer(budget[key],min,max))throw Error(`Invalid budget ${key}`);
  if(budget.total_candidates<budget.candidates_per_stage||budget.total_numeric_work<budget.work_per_stage)throw Error('Inconsistent global budget');
  const seen=new Set(),numericIds=new Set(),relationalIds=new Set();
  const stages=source.stages.map(stage=>{
    strict(stage,['id','family','kind','required','parents','arity','oracle','input','output','transfer_input','examples'],'stage');
    if(!id(stage.id)||!id(stage.family)||seen.has(stage.id)||!['numeric','composed','text','word_math','relational'].includes(stage.kind)||typeof stage.required!=='boolean')throw Error(`Invalid stage ${stage.id}`);
    const parents=stage.parents??[];
    if(!Array.isArray(parents)||parents.length>8||new Set(parents).size!==parents.length||parents.some(p=>!(stage.kind==='relational'?relationalIds:numericIds).has(p))||!['numeric','composed','relational'].includes(stage.kind)&&parents.length)throw Error(`Invalid/forward parent in ${stage.id}`);
    seen.add(stage.id);
    if(['numeric','composed'].includes(stage.kind))numericIds.add(stage.id);
    if(stage.kind==='relational')relationalIds.add(stage.id);
    if(stage.kind!=='text'){
      if(!integer(stage.arity,1,4)||!Array.isArray(stage.oracle)||!integer(stage.oracle.length,1,96)||stage.oracle.some(v=>!(typeof v==='number'&&Number.isFinite(v))&&!(typeof v==='string'&&(operations.has(v)||/^x[0-3]$/.test(v)&&Number(v[1])<stage.arity))))throw Error(`Invalid oracle ${stage.id}`);
      oracle(stage.oracle,Array(stage.arity).fill(1));
    }
    if(!['numeric','composed'].includes(stage.kind)&&!stage.examples){
      if(typeof stage.input!=='string'||typeof stage.output!=='string'&&stage.kind==='text'||stage.input.length>1024||(stage.transfer_input&&typeof stage.transfer_input!=='string'))throw Error('Invalid text template');
      for(const template of [stage.input,stage.output,stage.transfer_input].filter(v=>typeof v==='string')){
        if(/[{}]/.test(template.replace(/\{([a-z][a-z0-9_]*)(?:\|(upper|lower))?\}/g,'')))throw Error('Invalid placeholder');
        for(const match of placeholders(template))if(stage.kind==='text'?!Object.hasOwn(source.fields??{},match[1]):!/^x[0-3]$/.test(match[1])||Number(match[1][1])>=stage.arity)throw Error(`Unknown field ${match[1]}`);
      }
    }
    const result={id:stage.id,family:stage.family,kind:stage.kind,required:stage.required,parents};
    const allInputs=new Set();
    for(const split of ['train','admission','test','transfer']){
      const rows=[]; const pick=random(hashSeed(`${source.id}/${rng}/${stage.id}/${split}`));
      const count=source.splits[split];
      if(stage.examples){
        strict(stage.examples,['train','admission','test','transfer'],'explicit examples');
        if(!Array.isArray(stage.examples[split])||stage.examples[split].length>512)throw Error('Invalid explicit examples');
        for(const row of stage.examples[split]){strict(row,['input','output','arguments'],'example');rows.push({...row});}
      }else for(let attempt=0;rows.length<count&&attempt<200000;attempt++){
        let row;
        if(stage.kind==='text'){
          const variables={};
          for(const key of new Set([...placeholders(stage.input),...placeholders(stage.output),...placeholders(stage.transfer_input??'')].map(m=>m[1]))){
            const field=source.fields[key];strict(field,['train','admission','test','transfer'],`field ${key}`);
            const pool=field[split];if(!Array.isArray(pool)||pool.length<2||pool.some(s=>typeof s!=='string'||s.length>128))throw Error(`Invalid pool ${key}/${split}`);
            variables[key]=pool[pick(pool.length)];
          }
          row={input:render(split==='transfer'?(stage.transfer_input??stage.input):stage.input,variables),output:render(stage.output,variables)};
        }else{
          const x=Array.from({length:stage.arity},()=>split==='train'||split==='admission'?(pick(20001)-10000)/1000:(pick(200001)-100000)/(split==='transfer'?37:317));
          if(split==='train'&&attempt<stage.arity+1){x.fill(0);if(attempt)x[attempt-1]=1;}
          const output=oracle(stage.oracle,x);if(!Number.isFinite(output)||Math.abs(output)>1e12)continue;
          if(['numeric','composed'].includes(stage.kind))row={input:x,output};
          else {const vars=Object.fromEntries(x.map((n,i)=>[`x${i}`,String(n)]));row={input:render(split==='transfer'?(stage.transfer_input??stage.input):stage.input,vars),output,arguments:x};}
        }
        const key=JSON.stringify(row.input);if(allInputs.has(key))continue;allInputs.add(key);rows.push(row);
      }
      if(!stage.examples&&rows.length!==count)throw Error(`Insufficient distinct cases for ${stage.id}/${split}`);
      if(stage.examples){
        if(!integer(rows.length,split==='transfer'?0:split==='test'?1:4,split==='transfer'?128:split==='test'?512:32))throw Error('Explicit split count');
        for(const row of rows){const key=JSON.stringify(row.input);if(allInputs.has(key))throw Error('Duplicate/leaked explicit input');allInputs.add(key);}
      }
      for(const row of rows){
        if(typeof row.input==='string'&&Buffer.byteLength(row.input)>1024||typeof row.output==='string'&&Buffer.byteLength(row.output)>256)throw Error('Example byte ceiling');
        if(['numeric','composed'].includes(stage.kind)? !Array.isArray(row.input)||row.input.length!==stage.arity||row.input.some(n=>!Number.isFinite(n))||!Number.isFinite(row.output)||row.arguments!==undefined
          :stage.kind==='text'?typeof row.input!=='string'||typeof row.output!=='string'||row.arguments!==undefined
          :typeof row.input!=='string'||!Number.isFinite(row.output)||!Array.isArray(row.arguments)||row.arguments.length!==stage.arity||row.arguments.some(n=>!Number.isFinite(n)))throw Error('Invalid example types');
      }
      result[split]=rows;
    }
    return result;
  });
  return {version:1,id:source.id,source_sha256:sha(JSON.stringify(source)),budget,stages};
}
export async function launch(sourcePath,output,rng,cap){
  const real=await fs.realpath(sourcePath),stat=await fs.stat(real);
  if(path.relative(real,path.resolve(sourcePath)))throw Error('Linked curriculum sources are not allowed');
  if(stat.size>2000000)throw Error('Curriculum source exceeds 2 MB');
  const source=JSON.parse(await fs.readFile(real,'utf8'));
  const prepared=prepareCurriculum(source,rng,cap),encoded=JSON.stringify(prepared,null,2);
  if(Buffer.byteLength(encoded)>16000000)throw Error('Prepared curriculum exceeds 16 MB');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output,'curriculum.json'),JSON.stringify(source,null,2),{flag:'wx'});
  const preparedPath=path.join(output,'prepared.json');await fs.writeFile(preparedPath,encoded,{flag:'wx'});
  const root=fileURLToPath(new URL('../',import.meta.url));
  const binary=path.join(root,'target/release',`axon-uic-curriculum-lab${process.platform==='win32'?'.exe':''}`);
  const child=spawn(binary,['run',preparedPath,output,String(rng)],{cwd:root,env:process.env,stdio:'inherit',shell:false,windowsHide:true});
  return await new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',code=>resolve(code??1));});
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [source,output,rng='75218341',cap='5000']=process.argv.slice(2);
  if(!source||!output)throw Error('usage: bun ui/curriculum.mjs <curriculum.json> <new-output> [rng] [candidate-cap]');
  process.exitCode=await launch(path.resolve(source),path.resolve(output),Number(rng),Number(cap));
}
