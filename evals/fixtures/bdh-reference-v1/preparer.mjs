import fs from 'node:fs/promises';
import path from 'node:path';
import {downloadPinnedSource} from './pinned-github-source.mjs';

const out=process.argv[2];if(!out)throw Error('usage: node scripts/prepare-bdh-reference.mjs <new-workspace-directory>');
const request={repository:'pathwaycom/bdh',revision:'2b0d7a45b058d4309c84a10e0768d541fe18bdc2',files:[['LICENSE.md',1072,'0167e9c56a249efa2b15b9c0b96ee56a06127c83'],['README.md',6832,'d9193f7d5b0780c0f83c5b322fff5d5e6acfa419'],['bdh.py',5051,'4cfff79df97fa6faafb33bf20739a027edd8f360'],['train.py',3670,'6b982d8746a4455eb71f86ee54ddd40c1793d7b0']]};
const result=await downloadPinnedSource(request,out);
await fs.copyFile(process.argv[1],path.join(out,'preparer.mjs'),fs.constants.COPYFILE_EXCL);
await fs.copyFile('scripts/pinned-github-source.mjs',path.join(out,'download-helper.mjs'),fs.constants.COPYFILE_EXCL);
await fs.writeFile(path.join(out,'provenance.json'),JSON.stringify({request,...result,license:'MIT; copyright2025 Pathway Technology, Inc.',scope:'Pinned public baseline inference/reference only. Not BDH-CQ or the private Sudoku model; no downloaded code executed or weights/training data downloaded.'},null,2),{flag:'wx'});
console.log(JSON.stringify({files:request.files.length,revision:request.revision,code_executed:false}));
