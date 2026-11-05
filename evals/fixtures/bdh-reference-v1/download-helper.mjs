import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';

export async function downloadPinnedSource({repository,revision,files},destination,{workspace=process.cwd(),fetchImpl=fetch}={}) {
  assert(/^[a-zA-Z0-9][a-zA-Z0-9_.-]*\/[a-zA-Z0-9][a-zA-Z0-9_.-]*$/u.test(repository),'Repository identity');
  assert(/^[a-f0-9]{40}$/u.test(revision),'Pinned commit required');
  assert(Array.isArray(files)&&files.length>0&&files.length<=8,'Source file-count quota');
  const names=new Set();let total=0;
  for(const row of files) {
    assert(Array.isArray(row)&&row.length===3,'Pinned file tuple');
    const [file,length,blob]=row;
    assert(typeof file==='string'&&/^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_.-]+$/u.test(file),'Repository file path');
    const name=path.posix.basename(file).toLowerCase();
    assert(!['.','..','provenance.json','preparer.mjs','download-helper.mjs'].includes(name)&&!names.has(name),'Unique flat output name');names.add(name);
    assert(Number.isSafeInteger(length)&&length>0&&length<=8_000_000,'Pinned file-byte quota');total+=length;
    assert(total<=16_000_000,'Pinned total-byte quota');assert(typeof blob==='string'&&/^[a-f0-9]{40}$/u.test(blob),'Pinned Git blob required');
  }
  const root=path.resolve(workspace),output=path.resolve(root,destination);
  assert(output.startsWith(root+path.sep),'Source destination must stay inside workspace');
  for(let cursor=path.dirname(output);cursor!==root;cursor=path.dirname(cursor)) {
    try {assert(!(await fs.lstat(cursor)).isSymbolicLink(),'Source parent cannot be a link');}
    catch(error){if(error.code!=='ENOENT')throw error;}
  }
  await fs.mkdir(output);
  const sources=Object.create(null);
  for(const [file,length,blob]of files) {
    const name=path.posix.basename(file),url=`https://raw.githubusercontent.com/${repository}/${revision}/${file}`;
    const response=await fetchImpl(url,{signal:AbortSignal.timeout(120000)});assert(response.ok,`Download ${name}: ${response.status}`);assert(response.body,'Missing source body');
    const chunks=[];let size=0;
    for await(const chunk of response.body){size+=chunk.length;assert(size<=length,'Pinned byte quota');chunks.push(chunk);}
    const bytes=Buffer.concat(chunks);assert.equal(bytes.length,length,'Pinned byte length');
    assert.equal(createHash('sha1').update(`blob ${length}\0`).update(bytes).digest('hex'),blob,'Pinned Git blob identity');
    await fs.writeFile(path.join(output,name),bytes,{flag:'wx'});
    sources[name]={url,bytes:length,git_blob_sha1:blob,sha256:createHash('sha256').update(bytes).digest('hex')};
  }
  return {directory:output,sources};
}
