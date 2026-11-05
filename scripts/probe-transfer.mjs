import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const folder=path.resolve(process.argv[2]),output=path.join(folder,'portable-probes');await fs.mkdir(output);
const index=JSON.parse(await fs.readFile(path.join(folder,'model-index.json'),'utf8'));assert.equal(index.pages.length,1);
const seed=path.join(output,'number.axnum');await fs.copyFile(path.join(folder,index.pages[0].file),seed,fs.constants.COPYFILE_EXCL);
const prepared=JSON.parse(await fs.readFile(path.join(folder,'prepared.json'),'utf8'));
const cases=[];
// Vieta supplies integer roots, independently of the quadratic formula teacher.
for(let i=0;i<1000;i++){
  const a=(i%7+1)*(i%2?-1:1),r1=i%97-48,r2=i*13%89-44,b=-a*(r1+r2),c=a*r1*r2;
  for(const id of ['quadratic-plus','quadratic-minus']){const high=id==='quadratic-plus'?a>0:a<0;cases.push({module:id,input:[a,b,c],expected:high?Math.max(r1,r2):Math.min(r1,r2)});}
}
for(const id of ['quadratic-plus','quadratic-minus'])for(const input of [[0,5,-4],[1,0,1]])cases.push({module:id,input,expected:null});
for(const stage of prepared.stages.filter(s=>s.kind==='relational'))for(const row of [...stage.test,...stage.transfer])cases.push({module:null,input:row.input,expected:row.output});
for(const input of ['invoice preco 7 quantidade 13','quantidade 13 invoice preco 7'])cases.push({module:null,input,expected:91});
for(const input of ['nao invoice preco 7 quantidade 13','invoice preco 7 quantidade 13 quantidade 4','invoice preco 7','compre qualquer coisa desconhecida'])cases.push({module:null,input,expected:null});
const fixture=path.join(output,'cases.json');await fs.writeFile(fixture,JSON.stringify(cases),{flag:'wx'});
const binary=fileURLToPath(new URL(`../target/release/axon-uic-curriculum-lab${process.platform==='win32'?'.exe':''}`,import.meta.url));
const receipt=path.join(folder,'audit-portable-probes-v1.json');
const p=spawnSync(binary,['batch',seed,path.join(output,'newborn'),fixture,receipt],{encoding:'utf8',shell:false,windowsHide:true,timeout:120000});
assert.equal(p.status,0,p.stderr);const r=JSON.parse(await fs.readFile(receipt));console.log(JSON.stringify({correct:r.correct,checked:r.checked,loads:r.loads,failures:r.records.filter(r=>!r.correct).slice(0,10)}));
