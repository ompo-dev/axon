import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {downloadPinnedSource} from './pinned-github-source.mjs';

const revision='1'.repeat(40),body=Buffer.from('{"a":1}\n');
const blob=b=>createHash('sha1').update(`blob ${b.length}\0`).update(b).digest('hex');
const descriptor={repository:'example/source',revision,files:[['data/train.jsonl',body.length,blob(body)]]};
async function workspace(t) {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'axon-pinned-'));
  t.after(async()=>{assert.equal(path.dirname(path.resolve(root)),path.resolve(os.tmpdir()));assert(path.basename(root).startsWith('axon-pinned-'));await fs.rm(root,{recursive:true,force:true});});
  return root;
}
test('pinned source verifies exact bytes and never replaces an existing destination',async t=>{
  const root=await workspace(t),urls=[];
  const fetchImpl=async url=>{urls.push(url);return new Response(body);};
  const result=await downloadPinnedSource(descriptor,'source',{workspace:root,fetchImpl});
  assert.deepEqual(await fs.readFile(path.join(root,'source/train.jsonl')),body);
  assert.deepEqual(urls,[`https://raw.githubusercontent.com/example/source/${revision}/data/train.jsonl`]);
  assert.equal(result.sources['train.jsonl'].git_blob_sha1,blob(body));
  assert.equal(result.sources['train.jsonl'].sha256,createHash('sha256').update(body).digest('hex'));
  await assert.rejects(downloadPinnedSource(descriptor,'source',{workspace:root,fetchImpl}),{code:'EEXIST'});assert.equal(urls.length,1);
});
test('invalid identities, paths, aliases and quotas fail before any network or output',async t=>{
  const root=await workspace(t);let calls=0;const options={workspace:root,fetchImpl:async()=>{calls++;throw Error('Unexpected network');}};
  for(const config of [{...descriptor,repository:'../source'},{...descriptor,revision:'master'},
    {...descriptor,files:[['../test.jsonl',8,blob(body)]]},{...descriptor,files:[['data/..',8,blob(body)]]},
    {...descriptor,files:[['provenance.json',8,blob(body)]]},{...descriptor,files:[['a/train.jsonl',8,blob(body)],['b/TRAIN.jsonl',8,blob(body)]]},
    {...descriptor,files:[['train.jsonl',8_000_001,blob(body)]]},
    {...descriptor,files:[['a',8_000_000,blob(body)],['b',8_000_000,blob(body)],['c',1,blob(body)]]}]) {
    await assert.rejects(downloadPinnedSource(config,'source',options));
  }
  await assert.rejects(downloadPinnedSource(descriptor,'../outside',options));
  assert.equal(calls,0);assert.deepEqual(await fs.readdir(root),[]);
});
test('HTTP, length, overflow and repaired metadata cannot export unverified source bytes',async t=>{
  const root=await workspace(t);
  const responses=[new Response('no',{status:403}),new Response(body.subarray(1)),new Response(Buffer.concat([body,body])),new Response(Buffer.alloc(body.length,120))];
  for(let i=0;i<responses.length;i++) {
    const destination=`case-${i}`;
    await assert.rejects(downloadPinnedSource(descriptor,destination,{workspace:root,fetchImpl:async()=>responses[i]}));
    assert.deepEqual(await fs.readdir(path.join(root,destination)),[]);
  }
});
test('a linked source parent is refused before fetching or writing',async t=>{
  const root=await workspace(t);await fs.mkdir(path.join(root,'actual'));
  await fs.symlink(path.join(root,'actual'),path.join(root,'linked'),'junction');
  let calls=0;
  await assert.rejects(downloadPinnedSource(descriptor,'linked/source',{workspace:root,fetchImpl:async()=>{calls++;return new Response(body);}}));
  assert.equal(calls,0);assert.deepEqual(await fs.readdir(path.join(root,'actual')),[]);
});
test('valid unusual Git filenames cannot mutate the provenance object prototype',async t=>{
  const root=await workspace(t),source={...descriptor,files:[['__proto__',body.length,blob(body)]]};
  const {sources}=await downloadPinnedSource(source,'source',{workspace:root,fetchImpl:async()=>new Response(body)});
  assert.equal(Object.getPrototypeOf(sources),null);assert(Object.hasOwn(sources,'__proto__'));
  assert.equal(JSON.parse(JSON.stringify(sources)).__proto__.git_blob_sha1,blob(body));
});
