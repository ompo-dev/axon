import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {inventory} from './evidence-inventory.mjs';
import {sha} from './gsm8k-source.mjs';

const [sourcePath,out,prefix]=process.argv.slice(2);if(!prefix)throw Error('usage: node scripts/package-evidence-tree.mjs <frozen-source-directory> <new-index-root> <new-zip-prefix>');
const root=process.cwd(),source=path.resolve(sourcePath),output=path.resolve(out);for(const target of [source,output,path.resolve(prefix)])assert(target.startsWith(root+path.sep));assert(!output.startsWith(source+path.sep));await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);
const relative=p=>path.relative(root,p).replaceAll('\\','/'),allFiles=[];
async function measure(directory){assert(!(await fs.lstat(directory)).isSymbolicLink());const children=[],files=[];let bytes=0,count=0;for(const e of await fs.readdir(directory,{withFileTypes:true})){assert(!e.isSymbolicLink(),'Evidence link');const name=path.join(directory,e.name);if(e.isDirectory()){const child=await measure(name);children.push(child);bytes+=child.bytes;count+=child.count;}else{assert(e.isFile());const size=(await fs.stat(name)).size;assert(size<=67108864);files.push(name);allFiles.push(relative(name));bytes+=size;count++;}}return {directory,children,files,bytes,count};}
const tree=await measure(source),scopes=[];
function split(node){if(node.bytes<=536870912&&node.count<=100000){scopes.push({source:relative(node.directory),recursive:true});}else{if(node.files.length)scopes.push({source:relative(node.directory),recursive:false});for(const child of node.children)split(child);}}
split(tree);const covered=new Set(),receipts=[];
for(const [i,scope]of scopes.entries()){
  const index=path.join(output,`index-${i}`);await inventory(root,scope.source,index,scope.recursive);
  const data=JSON.parse(await fs.readFile(path.join(index,'inventory.json')));for(const r of data.files){assert(!covered.has(r.path),'Duplicate evidence scope');covered.add(r.path);}
  const verification=path.join(output,`verified-${i}.json`);await promisify(execFile)('pwsh',['-NoProfile','-File',path.resolve('scripts/archive-inventory.ps1'),'-IndexDirectory',index,'-DestinationPrefix',`${prefix}-scope-${i}`,'-Verification',verification],{windowsHide:true,timeout:300000,maxBuffer:2000000});
  const proof=JSON.parse(await fs.readFile(verification));assert(proof.verified);for(const r of proof.archives)assert.equal(sha(await fs.readFile(r.archive)),r.sha256);receipts.push({scope,verification:relative(verification),sha256:sha(await fs.readFile(verification)),proof});console.log(JSON.stringify({scope:i,files:proof.raw_files,bytes:proof.raw_bytes,archives:proof.archives.length}));
}
assert.deepEqual([...covered].sort(),allFiles.sort());
const report={schema:'axon-evidence-tree/v1',verified:true,source:relative(source),raw_files:tree.count,raw_bytes:tree.bytes,coverage:'Every original file exactly once across explicitly declared recursive/direct scopes; no source deletion or milestone promotion.',receipts};await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify({verified:true,files:tree.count,scopes:scopes.length}));
