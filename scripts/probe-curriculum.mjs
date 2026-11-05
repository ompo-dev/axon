import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const root=path.resolve(process.argv[2]);
const index=JSON.parse(await fs.readFile(path.join(root,'model-index.json'),'utf8'));
const output=path.join(root,'probes');await fs.mkdir(output);
const binary=fileURLToPath(new URL(`../target/release/axon-uic-curriculum-lab${process.platform==='win32'?'.exe':''}`,import.meta.url));
const checks=[
  {id:'hypotenuse',input:[5,12],expected:13},
  {id:'distance-two',input:[0,0,3,4],expected:5},
  {id:'invoice-total',input:'preco 7 quantidade 13',expected:91},
  {id:'person-reverse',input:'Maria Souza',expected:'Souza, Maria'},
  {id:'email-domain',input:'contato@exemplo.org',expected:'exemplo.org'},
  {id:'written-product',input:'produto entre 7 e 13',expected:91},
];
const proofs=[];
for(const [i,check] of checks.entries()){
  const page=index.pages.find(p=>p.modules.includes(check.id));assert(page,`Missing module ${check.id}`);
  const seed=path.join(output,page.file);
  try{await fs.copyFile(path.join(root,page.file),seed,fs.constants.COPYFILE_EXCL);}catch(e){if(e.code!=='EEXIST')throw e;}
  // The independent process receives only the copied number, module id and input.
  const child=spawnSync(binary,['predict',seed,path.join(output,`birth-${i}`),check.id,JSON.stringify(check.input)],{encoding:'utf8',shell:false,windowsHide:true,timeout:10000});
  assert.equal(child.status,0,child.stderr);const reply=JSON.parse(child.stdout);
  proofs.push({...check,...reply,correct:reply.answer===check.expected});
}
await fs.writeFile(path.join(root,'audit-curriculum-probes.json'),JSON.stringify({schema:'axon-curriculum-probes/v1',no_teacher_in_child:true,proofs},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(proofs));
