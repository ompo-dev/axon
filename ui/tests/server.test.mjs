import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { catalogFiles, validateRun, createObservatory, evolutionCatalog } from '../server.mjs';
import {QueryMemory} from '../trace.js';

test('query history contains only received prefixes and exact output counts',()=>{
  const memory=new QueryMemory();
  const event={event:'query',context:'test',actions:[1,2],reply:[101,203],actionsCount:4,cacheHit:false};
  const path=memory.accept(event);assert.equal(memory.nodes.size,3);
  assert.deepEqual(memory.nodes.get(path[1]).word,[1,2]);
  assert.deepEqual(memory.nodes.get(path[1]).output,[203]);
  memory.accept({...event,cacheHit:true});assert.equal(memory.nodes.size,3);
  assert.equal(memory.outputs.get('output:test:101').physical,1);
  assert.equal(memory.outputs.get('output:test:101').cached,1);
  memory.accept({event:'cache_release',context:'test'});assert.equal(memory.nodes.get('cache:test:root').active,false);
  const restored=new QueryMemory();restored.restore([...memory.nodes.values()]);
  assert.equal(restored.nodes.size,memory.nodes.size);assert.equal(restored.nodes.get('cache:test:root').active,false);
  restored.accept({...event,context:'new-context'});restored.accept({event:'end'});
  assert([...restored.nodes.values()].filter(n=>n.root).every(n=>n.active===false));
});

async function removeTestRoot(root) {
  const resolved = await fs.realpath(root);
  const relative = path.relative(await fs.realpath(os.tmpdir()), resolved);
  assert(!relative.startsWith('..') && !path.isAbsolute(relative));
  assert(path.basename(resolved).startsWith('axon-ui-'));
  await fs.rm(resolved, {recursive:true,force:true});
}

test('run inputs are typed and cannot inject executable arguments',()=>{
  assert.equal(validateRun({mode:'transfer'}).seed,719319);
  for(const input of [null,[],{mode:'shell'},{mode:'transfer',seed:'1;rm'},{mode:'transfer',seed:-1},{mode:'factor',dimensions:9},{mode:'factor',command:'powershell'},{mode:'factor',paused:'true'},{mode:'knowledge',query:''},{mode:'math',task:'x'.repeat(33000)},{mode:'math',task:'x',paused:true}])assert.throws(()=>validateRun(input));
});
test('catalog is scoped and excludes unsupported files',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-'));
  try{await fs.mkdir(path.join(root,'src'));await fs.mkdir(path.join(root,'docs'));await fs.writeFile(path.join(root,'src','a.rs'),'pub fn a() {}');await fs.writeFile(path.join(root,'docs','a.md'),'# Source');await fs.writeFile(path.join(root,'docs','secret.env'),'TOKEN=hidden');
    const files=await catalogFiles(root);assert.equal(files.length,2);assert(files.every(f=>!f.path.includes('secret')));assert.equal(files.find(f=>f.path.endsWith('.rs')).kind,'code');
  }finally{await removeTestRoot(root);}
});
test('loopback API protects writes, rejects traversal and handles malformed requests',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-http-'));await fs.mkdir(path.join(root,'docs'));await fs.writeFile(path.join(root,'docs','safe.md'),'# Safe source\n');
  const app=await createObservatory({root,port:0,autoSnapshot:false});
  try{
    const state=await(await fetch(`${app.url}/api/state`)).json();assert.equal(state.sources.length,1);assert.equal(state.coreReady,false);
    assert.match(state.session,/^[a-f0-9-]{36}$/);assert.notEqual(state.session,state.token);
    const controller=new AbortController();
    try{const response=await fetch(`${app.url}/api/events`,{signal:controller.signal});const reader=response.body.getReader();const packet=new TextDecoder().decode((await reader.read()).value).split('data: ')[1].trim();assert.equal(JSON.parse(packet).session,state.session);}finally{controller.abort();}
    assert.equal((await fetch(`${app.url}/api/source?id=../../secret`)).status,400);
    const source=await(await fetch(`${app.url}/api/source?id=0`)).json();assert.equal(source.text,'# Safe source\n');
    assert.equal((await fetch(`${app.url}/api/run`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'transfer'})})).status,403);
    const headers={'Content-Type':'application/json','X-Axon-Session':state.token};
    assert.equal((await fetch(`${app.url}/api/run`,{method:'POST',headers:{...headers,Origin:'https://foreign.invalid'},body:'{}'})).status,403);
    assert.equal((await fetch(`${app.url}/api/run`,{method:'POST',headers,body:'invalid'})).status,400);
    assert.equal((await fetch(`${app.url}/api/run`,{method:'POST',headers,body:JSON.stringify({mode:'transfer'})})).status,400);
    assert.equal((await fetch(`${app.url}/api/control`,{method:'POST',headers,body:JSON.stringify({command:'cancel'})})).status,409);
    assert.equal((await fetch(`${app.url}/server.mjs`)).status,404);
    assert.equal((await fetch(`${app.url}/api/export?id=../x`)).status,404);
    const html=await fetch(app.url);assert.equal(html.status,200);assert(html.headers.get('content-security-policy').includes("frame-ancestors 'none'"));assert((await html.text()).includes('AXON | Workspace'));
    assert.equal((await fetch(`${app.url}/app.js`)).status,404);
    assert.deepEqual((await(await fetch(`${app.url}/api/evolution`)).json()).campaigns,[]);
  }finally{await app.close();await removeTestRoot(root);}
});

test('canonical document inspection is bounded without raising other source limits',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-document-'));
  await fs.mkdir(path.join(root,'docs'));
  const master=path.join(root,'docs','AXON.md');
  const text='# AXON: Documento Mestre\n'+'x'.repeat(200000);
  await fs.writeFile(master,text);
  await fs.writeFile(path.join(root,'docs','other.md'),'x'.repeat(128001));
  const app=await createObservatory({root,port:0,autoSnapshot:false});
  try{
    const state=await(await fetch(`${app.url}/api/state`)).json();
    const source=state.sources.find(s=>s.path==='docs/AXON.md');
    const other=state.sources.find(s=>s.path==='docs/other.md');
    const response=await fetch(`${app.url}/api/source?id=${source.id}`);
    assert.equal(response.status,200);
    assert.equal((await response.json()).text,text);
    assert.equal((await fetch(`${app.url}/api/source?id=${other.id}`)).status,413);
    await fs.writeFile(master,'x'.repeat(1048577));
    assert.equal((await fetch(`${app.url}/api/source?id=${source.id}`)).status,413);
  }finally{await app.close();await removeTestRoot(root);}
});

test('stored evolution uses CSV parsing and only allowlisted in-workspace reports',async()=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-evolution-'));
  try {
    const dir=path.join(root,'.axon-memory','evolution');await fs.mkdir(dir,{recursive:true});
    const name='reserved-42-hybrid-v1';await fs.mkdir(path.join(dir,`${name}.outputs`));
    await fs.writeFile(path.join(dir,`${name}.json`),JSON.stringify({mode:'evolution',seed:42,evolution_mode:'hybrid',output:'C:/outside',lifecycle_ms:10,max_sampled_working_set_bytes:100,binary_bytes:200,exit_code:0}));
    await fs.writeFile(path.join(dir,`${name}.outputs`,'results.csv'),'case,status,generation,evaluations,work,active_genes,instructions,dependencies,capsule_bytes,correct,heldout,training_ns,local_proposals,residual_errors,expression\n0,Admitted,1,2,3,4,5,0,6,7,7,8,9,0,"add(x0,x1)"\n');
    await fs.writeFile(path.join(dir,'secret.json'),'{}');
    const data=await evolutionCatalog(root);assert.equal(data.length,1);assert.equal(data[0].cases[0].expression,'add(x0,x1)');assert.equal(data[0].cases[0].correct,7);assert.equal(data[0].seed,42);
    await fs.appendFile(path.join(dir,`${name}.outputs`,'results.csv'),'1,WorkLimit,1,2,3,,,,,0,7,8,9,,\n');
    const partial=await evolutionCatalog(root);assert.equal(partial.length,1);assert.equal(partial[0].cases.length,2);assert.equal(partial[0].cases[1].capsule_bytes,null);assert.equal(partial[0].cases[1].residual_errors,null);
    await fs.writeFile(path.join(dir,`${name}.outputs`,'results.csv'),'invalid');assert.deepEqual(await evolutionCatalog(root),[]);
  } finally {await removeTestRoot(root);}
});
