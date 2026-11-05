import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {inventory} from './evidence-inventory.mjs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

test('byte deduplication preserves every original name and recoverable representative',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-evidence-index-')),source=path.join(root,'source');await fs.mkdir(source);await fs.mkdir(path.join(source,'born'));
  await fs.writeFile(path.join(source,'seed'),'same bytes');await fs.writeFile(path.join(source,'born/copy'),'same bytes');await fs.writeFile(path.join(source,'empty'),'');await fs.writeFile(path.join(source,'binary'),Buffer.from([0,255,17]));
  const result=await inventory(root,'source','index');assert.equal(result.raw_files,4);assert.equal(result.unique_files,3);
  const index=JSON.parse(await fs.readFile(path.join(root,'index/inventory.json'))),plan=JSON.parse(await fs.readFile(path.join(root,'index/plan.json')));
  for(const e of index.files)assert.deepEqual(await fs.readFile(path.join(root,e.path)),await fs.readFile(path.join(root,e.content_path)));
  assert.equal(new Set(plan.shards.flatMap(s=>s.paths).filter(p=>p.startsWith('source/'))).size,3);
  await assert.rejects(inventory(root,'source','index'),{code:'EEXIST'});assert.equal(await fs.readFile(path.join(source,'seed'),'utf8'),'same bytes');
});
test('inventory refuses traversal, recursive self-capture and linked sources',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-evidence-guard-'));await fs.mkdir(path.join(root,'source'));await fs.writeFile(path.join(root,'source/a'),'x');
  await assert.rejects(inventory(root,'source','../outside'),/outside root/);await assert.rejects(inventory(root,'source','source/index'),/inside evidence/);
  const elsewhere=await fs.mkdtemp(path.join(os.tmpdir(),'axon-evidence-link-'));await fs.symlink(elsewhere,path.join(root,'source/link'),process.platform==='win32'?'junction':'dir');
  await assert.rejects(inventory(root,'source','refused-index'),/link/);
});
test('direct inventory declares omitted descendants for separately bounded archives',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-evidence-direct-'));await fs.mkdir(path.join(root,'source/sub'),{recursive:true});await fs.writeFile(path.join(root,'source/a'),'one');await fs.writeFile(path.join(root,'source/sub/b'),'two');
  const direct=await inventory(root,'source','direct',false),nested=await inventory(root,'source/sub','nested');
  assert.equal(direct.raw_files,1);assert.equal(nested.raw_files,1);const index=JSON.parse(await fs.readFile(path.join(root,'direct/inventory.json')));assert.equal(index.recursive,false);assert.deepEqual(index.files.map(x=>x.path),['source/a']);
});
test('native ZIP wrapper stores aliases recoverably and refuses omitted index metadata',{skip:process.platform!=='win32'},async()=>{
  const relative=`.axon-memory/evidence-wrapper-test-${process.pid}-${Date.now()}`,root=path.resolve(relative);await fs.mkdir(root);await fs.mkdir(path.join(root,'source'));
  await fs.writeFile(path.join(root,'source/a'),'same');await fs.writeFile(path.join(root,'source/b'),'same');await fs.writeFile(path.join(root,'source/empty'),'');
  await inventory(process.cwd(),`${relative}/source`,`${relative}/index`);
  const call=promisify(execFile),wrapper=path.resolve('scripts/archive-inventory.ps1'),report=path.join(root,'verified.json');
  await call('pwsh',['-NoProfile','-File',wrapper,'-IndexDirectory',path.join(root,'index'),'-DestinationPrefix',path.join(root,'snapshot'),'-Verification',report],{windowsHide:true,timeout:120000,maxBuffer:1000000});
  const proof=JSON.parse(await fs.readFile(report));assert(proof.verified);assert.equal(proof.raw_files,3);assert.equal(proof.unique_files,2);assert.equal(proof.archives.length,1);
  const second=path.join(root,'rechecked.json');await call('pwsh',['-NoProfile','-File',wrapper,'-IndexDirectory',path.join(root,'index'),'-DestinationPrefix',path.join(root,'snapshot'),'-Verification',second,'-VerifyOnly'],{windowsHide:true,timeout:120000,maxBuffer:1000000});assert(JSON.parse(await fs.readFile(second)).verified);
  const planPath=path.join(root,'index/plan.json'),plan=JSON.parse(await fs.readFile(planPath));plan.shards[0].paths=plan.shards[0].paths.filter(p=>!p.endsWith('/inventory.json'));await fs.writeFile(planPath,JSON.stringify(plan));
  await assert.rejects(call('pwsh',['-NoProfile','-File',wrapper,'-IndexDirectory',path.join(root,'index'),'-DestinationPrefix',path.join(root,'missing-index'),'-Verification',path.join(root,'must-not-verify.json')],{windowsHide:true,timeout:120000,maxBuffer:1000000}));
  await assert.rejects(fs.stat(path.join(root,'must-not-verify.json')),{code:'ENOENT'});assert.equal(await fs.readFile(path.join(root,'source/a'),'utf8'),'same');
});
