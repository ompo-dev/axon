import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const sha=b=>createHash('sha256').update(b).digest('hex');
async function guarded(root,target){
  const absolute=path.resolve(root,target),relative=path.relative(root,absolute);
  assert(relative&&!relative.startsWith('..'+path.sep)&&relative!=='..'&&!path.isAbsolute(relative),'Evidence outside root');
  let current=root;
  for(const part of relative.split(path.sep)){current=path.join(current,part);try{assert(!(await fs.lstat(current)).isSymbolicLink(),'Evidence link');}catch(e){if(e.code!=='ENOENT')throw e;}}
  return absolute;
}
export async function inventory(root,sourcePath,outputPath,recursive=true){
  assert.equal(typeof recursive,'boolean');
  root=path.resolve(root);const source=await guarded(root,sourcePath),output=await guarded(root,outputPath);
  assert(!output.startsWith(source+path.sep)&&source!==output,'Output inside evidence source');
  assert((await fs.stat(source)).isDirectory());await fs.mkdir(output);
  const relative=p=>path.relative(root,p).split(path.sep).join('/');
  const entries=[],unique=new Map(),names=new Set();let rawBytes=0;
  async function walk(directory){
    const children=await fs.readdir(directory,{withFileTypes:true});children.sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0);
    for(const child of children){
      const file=await guarded(root,path.join(directory,child.name));
      if(child.isDirectory()){if(recursive)await walk(file);continue;}assert(child.isFile(),'Evidence special file');
      assert(entries.length<100000,'Evidence index count');const before=await fs.stat(file);assert(before.size<=67108864,'Evidence single file quota');
      rawBytes+=before.size;assert(rawBytes<=536870912,'Evidence total raw bytes');
      const bytes=await fs.readFile(file),after=await fs.stat(file);assert(before.size===bytes.length&&after.size===before.size&&after.mtimeMs===before.mtimeMs,'Evidence changed during inventory');
      const name=relative(file),identity=name.toLowerCase();assert(!names.has(identity),'Case-aliased evidence path');names.add(identity);
      const hash=sha(bytes),canonical=unique.get(hash);if(canonical)assert.equal(canonical.bytes,bytes.length);else unique.set(hash,{path:name,bytes:bytes.length,sha256:hash});
      entries.push({path:name,bytes:bytes.length,sha256:hash,content_path:unique.get(hash).path});
    }
  }
  await walk(source);assert(entries.length>0,'Empty evidence');
  const data={schema:'axon-byte-deduplicated-evidence/v1',source:relative(source),recursive,raw_files:entries.length,raw_bytes:rawBytes,unique_files:unique.size,unique_bytes:[...unique.values()].reduce((n,r)=>n+r.bytes,0),files:entries,
    scope:'Every pathname in the declared scope and byte/hash identity retained; recursive=false explicitly excludes descendant directories, which require separate inventories. Equal byte content points to one representative included in bounded verified ZIP shards. No source file deleted or changed. Does not retain OS ACL/timestamps or certify a milestone.'};
  const index=path.join(output,'inventory.json'),indexBytes=Buffer.from(JSON.stringify(data,null,2));assert(indexBytes.length<=67108864);await fs.writeFile(index,indexBytes,{flag:'wx'});
  const planFile=path.join(output,'plan.json'),representatives=[...unique.values()],shards=[];
  for(let i=0;i<representatives.length;i+=2048)shards.push({index:shards.length,paths:[relative(index),relative(planFile),...representatives.slice(i,i+2048).map(r=>r.path)]});
  const plan={schema:'axon-evidence-shard-plan/v1',index:relative(index),index_sha256:sha(indexBytes),shards,limits:{archive_files:4096,archive_bytes:536870912,single_file_bytes:67108864},scope:'Reuse archive-evidence.ps1 unchanged. Every representative plus index/plan must be stored; verify all archive manifests against this index before treating this as recoverable evidence.'};
  await fs.writeFile(planFile,JSON.stringify(plan,null,2),{flag:'wx'});return {raw_files:data.raw_files,raw_bytes:rawBytes,unique_files:unique.size,unique_bytes:data.unique_bytes,shards:shards.length,index:relative(index)};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [source,output]=process.argv.slice(2);if(!output)throw Error('usage: node scripts/evidence-inventory.mjs <source-directory> <new-index-directory>');
  console.log(JSON.stringify(await inventory(process.cwd(),source,output)));
}
