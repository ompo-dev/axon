import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';
const qa='.axon-memory/observatory/qa',root='.axon-memory/polynomial-proposal-v1';
const read=async p=>JSON.parse(await fs.readFile(p));
const a=await read(`${root}/audit.json`),f=await read(`${root}/freeze-before-execution.json`);
for(const [file,hash]of Object.entries(f.inputs))assert.equal(sha(await fs.readFile(`${root}/${file}`)),hash);
for(const [file,hash]of Object.entries(f.source_hashes))assert.equal(sha(await fs.readFile(file)),hash);
assert.equal(a.binary_sha256,sha(await fs.readFile(`${root}/lab.exe`)));
assert.equal(a.binary_sha256,sha(await fs.readFile('target/m1-compact-20261018/release/axon-uic-curriculum-lab.exe')));
const modes=['enumerate','polynomial'];
const gold=await read(`${root}/host-only-gold.json`);
for(const mode of modes){
  const input=await read(`${root}/${mode}-input.json`),rows=await read(`${root}/${mode}-predictions.json`);
  assert.deepEqual(rows.map(r=>r.id),gold.map(r=>r.id));
  for(const task of input){assert.deepEqual(Object.keys(task).sort(),['id','inherit','inputs','train',...(mode==='polynomial'?['representation']:[])].sort());assert(task.inherit===false);}
  const correct=rows.reduce((n,r,i)=>n+r.predictions.filter((v,j)=>v===gold[i].expected[j]).length,0);
  assert.equal(correct,a.scores[mode].correct);assert.equal(correct,mode==='polynomial'?384:0);
}
assert.deepEqual(await read(`${root}/enumerate-predictions.json`),await read(`${root}/previous-predictions.json`));
assert.equal(a.literal.correct,0);assert.equal(a.literal.abstained,384);
assert.equal(a.controls.filter(r=>r.error!==null).length,8);
for(const id of ['ambiguous','hidden-quartic']){const r=a.controls.find(r=>r.id===id);assert.equal(r.error,null);assert.deepEqual(r.predictions,[2,7,11,19]);}
const born=await read(`${root}/newborn-predictions.json`);assert.deepEqual(born,await read(`${root}/newborn-again-predictions.json`));
const expected=gold.flatMap(r=>r.expected);expected.push(6859);
assert.equal(born.length,385);assert(born.every((r,i)=>(r.actual?.value??r.actual)===expected[i]));
assert.equal(a.newborn.correct,385);assert.equal(sha(await fs.readFile(`${root}/trained/structural.axnum`)),a.newborn.number_sha256);
assert.equal(a.events.refutations,3);assert.equal(a.events.consolidations,75);
assert(a.training.controls.every(r=>r.acquired===25&&r.lessons===26));assert.equal(a.training.failures.length,3);
const pointer=await read('.axon-memory/polynomial-pointer-replay-v1/audit.json');assert.equal(pointer.total_rows,17444);assert.equal(pointer.boundaries.length,11);assert.equal(pointer.binary_sha256,a.binary_sha256);
const language=await read('.axon-memory/polynomial-language-replay-v1/audit.json');assert.equal(language.results[0].behavior_identical,1504);
const rust=await fs.readFile(`${qa}/polynomial-rust-v2.log`,'utf8');
assert(!/test result: FAILED|^error:/mu.test(rust));assert(rust.includes('Doc-tests axon_uic'));
const groups=[...rust.matchAll(/^test result: ok\. (\d+) passed; 0 failed; (\d+) ignored/gmu)];assert(groups.length>100);assert(groups.every(g=>Number(g[2])===0));
const js=await fs.readFile(`${qa}/polynomial-js-v1.log`,'utf8');assert(/pass 74/u.test(js)&&/fail 0/u.test(js)&&/skipped 0/u.test(js));
for(const name of ['clippy','minimal']){const log=await fs.readFile(`${qa}/polynomial-${name}-v1.log`,'utf8');assert(log.includes('Finished')&&!/^error:/mu.test(log));}
assert.equal((await fs.readFile(`${qa}/polynomial-fmt-v1.log`,'utf8')).trim(),'');
const diff=(await fs.readFile(`${qa}/polynomial-diff-v1.log`,'utf8')).trim();
if(diff)for(const line of diff.split(/\r?\n/u))assert(/^warning: in the working copy of '.+', LF will be replaced by CRLF the next time Git touches it$/u.test(line),'Whitespace error beyond environment CRLF warning');const production=sha(await fs.readFile('target/release/axon-uic-curriculum-lab.exe'));assert.equal(production,'f0cf11fcf95c04935548231cfd6d80d4d6e0ce93eb2afd1f45838c2c95da3660');
const contract=sha(await fs.readFile('evals/protocols/m1-completion-v1.json'));assert.equal(contract,'0e99ea97dcc7eb4cae95ba8a1d099edbb8497b3f65e290ba4778f9c43fc47e79');
const graph=await read('.axon-memory/polynomial-graph-v1/extraction.json');assert.equal(graph.nodes.length,170);assert.equal(graph.edges.length,562);
const document=await fs.readFile('docs/AXON.md','utf8');assert(document.includes('Versao documental: 54.'));assert(document.includes('### 14.54 Proposta Polinomial No Motor Real'));
for(const name of ['polynomial-proposal-v1','polynomial-pointer-replay-v1','polynomial-language-replay-v1','polynomial-source-qa-v1','polynomial-graph-v1']){const proof=await read(`.axon-memory/polynomial-evidence-v1/${name}/audit.json`);assert(proof.verified);for(const scope of proof.receipts)for(const archive of scope.proof.archives)assert.equal(sha(await fs.readFile(archive.archive)),archive.sha256);}
const result={schema:'axon-polynomial-phase54-verification/v1',verified:true,runtime_sha256:a.binary_sha256,source_hashes:f.source_hashes,canonical_document_sha256:sha(Buffer.from(document)),rust_passed:groups.reduce((n,g)=>n+Number(g[1]),0),rust_groups:groups.length,js_passed:74,transfer:{polynomial:384,enumeration:0,literal:0,total:384},newborn:{correct:385,total:385,repeated_identically:true},refutations:3,refused_controls:8,ambiguous_false_generalizations:2,legacy_pointer_rows:17444,legacy_language_rows:1504,default_matches_previous:true,production_unchanged:true,completion_contract_unchanged:true,m1_complete:false,m2_permitted:false,scope:a.limits};
await fs.writeFile(`${qa}/polynomial-phase54-verification-v1.json`,JSON.stringify(result,null,2),{flag:'wx'});console.log(JSON.stringify(result));