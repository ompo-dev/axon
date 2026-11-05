import fs from 'node:fs/promises';
import path from 'node:path';
import {downloadPinnedSource} from './pinned-github-source.mjs';

const [destination='.axon-memory/corpora/svamp-78e727689e1c']=process.argv.slice(2);
const revision='78e727689e1c1bebfc4be39c446898e8e10b0518';
const files=[['LICENSE',1068,'fff1c244016305bf90d8c24f66d80a8cbaaa910d'],['README.md',8186,'d8b7b48bde47ab03ba8cfe7bd7f2816627ff5611'],['SVAMP.json',339970,'aad093525105c6331581d7082250b9c22578271e']];
const {sources}=await downloadPinnedSource({repository:'arkilpatel/SVAMP',revision,files},destination);
await fs.copyFile(process.argv[1],path.join(destination,'preparer.mjs'),fs.constants.COPYFILE_EXCL);
await fs.copyFile(new URL('./pinned-github-source.mjs',import.meta.url),path.join(destination,'download-helper.mjs'),fs.constants.COPYFILE_EXCL);
const report={schema:'axon-public-svamp-source/v1',revision,source:'https://github.com/arkilpatel/SVAMP',paper:'https://aclanthology.org/2021.naacl-main.168/',authors:'Arkil Patel; Satwik Bhattamishra; Navin Goyal',license:'Pinned author repository MIT notice retained; source dataset provenance/derivative overlap still requires audit.',language:'English, not Portuguese',sources,
  dataset_parsed:false,readme_interpreted:false,training_started:false,inference_started:false,
  reservation:'Sealed external arithmetic challenge bytes. No split admitted yet and no examples used for AXON training. Declare provenance, grouping, overlap and prospective evaluation before reading target annotations; public availability is not independent-author/novel-problem proof.',
  scope:'Immutable verified download only, not mathematical discovery, accuracy, a fresh independent M1 reserve or completion of any gate. Equations/program strings are untrusted data and must never be eval/exec input.'};
await fs.writeFile(path.join(destination,'provenance.json'),JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify({revision,files:Object.keys(sources),bytes:Object.values(sources).reduce((n,s)=>n+s.bytes,0),dataset_parsed:false}));
