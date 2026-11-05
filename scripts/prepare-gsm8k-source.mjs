import fs from 'node:fs/promises';
import path from 'node:path';
import {downloadPinnedSource} from './pinned-github-source.mjs';

const [destination='.axon-memory/corpora/gsm8k-3101c7d50724']=process.argv.slice(2);
const revision='3101c7d5072418e28b9008a6636bde82a006892c';
const files=[['LICENSE',1062,'9e84fcbc4d81a1f433c90caf9f1cef373c12edae'],['README.md',6112,'f5d12ef45cc7b78dbc148cbd060994decbc3159d'],['grade_school_math/data/train.jsonl',4166206,'7d97154b91aef28d01c3301741c81ce90039a4b1']];
const {sources}=await downloadPinnedSource({repository:'openai/grade-school-math',revision,files},destination);
await fs.copyFile(process.argv[1],path.join(destination,'preparer.mjs'),fs.constants.COPYFILE_EXCL);
await fs.copyFile(new URL('./pinned-github-source.mjs',import.meta.url),path.join(destination,'download-helper.mjs'),fs.constants.COPYFILE_EXCL);
const report={schema:'axon-public-gsm8k-source/v1',revision,source:'https://github.com/openai/grade-school-math',license:'Pinned author repository MIT notice retained. Dataset provenance, authorship, overlap and applicable terms must be checked before admission/distribution.',language:'English, not Portuguese',sources,
  original_test_downloaded:false,original_test_read:false,model_solutions_downloaded:false,dataset_parsed:false,readme_interpreted:false,training_started:false,inference_started:false,
  reservation:'Only original train bytes downloaded. Freeze train/development grouping and allowed supervision before interpreting target answers. No benchmark reserve or M1 gate is admitted by this download. Raw calculator annotations/answers are untrusted data, never eval/exec inputs or inference fields.',
  scope:'Pinned bounded source acquisition only. Not a learned capability, a new solver or any M1 approval.'};
await fs.writeFile(path.join(destination,'provenance.json'),JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify({revision,files:Object.keys(sources),bytes:Object.values(sources).reduce((n,s)=>n+s.bytes,0),dataset_parsed:false,original_test_downloaded:false}));
