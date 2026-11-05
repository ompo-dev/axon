import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');

const [destination='.axon-memory/corpora/pacific-66a52ad4fd9e']=process.argv.slice(2);
const revision='66a52ad4fd9eb7ad71e82a5c0bfb1de226366827';
const files=[
  ['LICENSE',1067,'b7a1ebac91fead13950a77016c95d1d004be3379'],
  ['README.md',1501,'13956d69a5f706eb42cdfe2c3d648cc64fd0791e'],
  ['data/pacific/train.json',16342570,'1fd0a1c33b982688564fdcde49f7f3101e74c282'],
];
await fs.mkdir(destination);
const sources={};
for(const [file,length,blob]of files) {
  const url=`https://raw.githubusercontent.com/dengyang17/PACIFIC/${revision}/${file}`;
  const response=await fetch(url,{signal:AbortSignal.timeout(120000)});assert(response.ok,`Download ${file}: ${response.status}`);
  const chunks=[];let bytesRead=0;
  for await(const chunk of response.body) {bytesRead+=chunk.length;assert(bytesRead<=length,'Pinned file quota');chunks.push(chunk);}
  const bytes=Buffer.concat(chunks);assert.equal(bytes.length,length);
  const actual=createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');assert.equal(actual,blob,'Pinned Git blob identity');
  const name=path.basename(file);await fs.writeFile(path.join(destination,name),bytes,{flag:'wx'});
  sources[name]={url,sha256:sha(bytes),git_blob_sha1:blob,bytes:length};
}
const provenance={schema:'axon-public-pacific-source/v1',revision,source:'https://github.com/dengyang17/PACIFIC',sources,
  attribution:'Yang Deng, Wenqiang Lei, Wenxuan Zhang, Wai Lam, Tat-Seng Chua. PACIFIC, EMNLP 2022. Built from TAT-QA financial reports.',
  license:'Pinned repository MIT notice retained. Source report and TAT-QA rights must be audited before redistribution or commercial claims; repository code license is not asserted to extinguish third-party rights.',
  language:'English, not Portuguese; any later automatic translation must be identified as such, not newly human-authored Portuguese.',
  original_test_downloaded:false,original_validation_downloaded:false,annotations_parsed:false,
  scope:'Pinned original train bytes only, not yet adapted, partitioned, trained or evaluated. No model capability, fresh reserve, independent-worker split or M1 gate is claimed. Future splits must group source reports/tables/dialogues and audit overlap with other corpora before exposing reserved text.'};
await fs.copyFile(process.argv[1],path.join(destination,'preparer.mjs'),fs.constants.COPYFILE_EXCL);
await fs.writeFile(path.join(destination,'provenance.json'),JSON.stringify(provenance,null,2),{flag:'wx'});
console.log(JSON.stringify({revision,files:Object.keys(sources),bytes:Object.values(sources).reduce((n,s)=>n+s.bytes,0),original_test_downloaded:false,annotations_parsed:false}));
