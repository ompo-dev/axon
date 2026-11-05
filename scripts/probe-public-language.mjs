import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify,isDeepStrictEqual} from 'node:util';
import assert from 'node:assert/strict';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';

const [numberFile,executable,corpus,outputPath]=process.argv.slice(2);
if(!outputPath)throw Error('usage: bun scripts/probe-public-language.mjs <number> <candidate> <pinned-corpus> <new-output>');
const output=path.resolve(outputPath);await fs.mkdir(output);
const write=(name,value)=>fs.writeFile(path.join(output,name),JSON.stringify(value,null,2),{flag:'wx'});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const provenance=JSON.parse(await fs.readFile(path.join(corpus,'provenance.json'),'utf8'));
assert.equal(provenance.revision,'b6c5e0b27e52477bbc5433d130b0a0e825cf74dc');
const raw=await fs.readFile(path.join(corpus,'test.csv'));
assert.equal(sha(raw),provenance.files.find(f=>f.name==='test.csv').sha256);
const dataset=parse(raw,{columns:true,bom:true});assert.equal(dataset.length,1758);
const inputs=dataset.flatMap((r,i)=>[{id:`external-${i}-p`,text:r.premise},{id:`external-${i}-h`,text:r.hypothesis}]);
const answers=dataset.map((r,i)=>({id:i,label:Number(r.label),reviewer:r.reviwers_output,quality:r.low_quality}));
assert(answers.every(r=>[0,1,2].includes(r.label)));
assert(answers.every(r=>['Contradiction','Entailment','Neutral','Disagreement','Confusing'].includes(r.reviewer)),'Unknown original review annotation');
await write('inputs.json',inputs);await write('host-labels.json',answers);
const binary=path.join(output,'candidate.exe'),number=path.join(output,'isolated.axnum');
await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);await fs.copyFile(numberFile,number,fs.constants.COPYFILE_EXCL);
await write('freeze.json',{binary_sha256:sha(await fs.readFile(binary)),number_sha256:sha(await fs.readFile(number)),corpus_revision:provenance.revision,test_sha256:sha(raw),inputs_sha256:sha(await fs.readFile(path.join(output,'inputs.json'))),scorer_sha256:sha(await fs.readFile(process.argv[1])),protocol_sha256:sha(await fs.readFile('evals/protocols/m1-public-language-diagnostic-v1.json')),
  scope:'Frozen before first execution. Learner receives only compiled number and raw texts. Logical label isolation, not an OS sandbox; the host scorer parses labels but never generates meanings.'});
const started=performance.now();
let child;
try {
  child=await promisify(execFile)(binary,['interpret-batch',number,path.join(output,'birth'),path.join(output,'inputs.json'),path.join(output,'predictions.json')],{cwd:output,windowsHide:true,timeout:300000,maxBuffer:16000000});
} catch(error) {
  await fs.writeFile(path.join(output,'failed-stdout.log'),error.stdout??'',{flag:'wx'});
  await fs.writeFile(path.join(output,'failed-stderr.log'),error.stderr??'',{flag:'wx'});
  await write('failure.json',{code:error.code,message:error.message,scope:'Probe did not complete. No incomplete or unavailable prediction is counted as an evaluated success.'});
  throw error;
}
await fs.writeFile(path.join(output,'stdout.log'),child.stdout,{flag:'wx'});await fs.writeFile(path.join(output,'stderr.log'),child.stderr,{flag:'wx'});
const summary=JSON.parse(child.stdout.trim());assert(summary.seed_unchanged&&summary.teacher_queries===0);
const predictions=JSON.parse(await fs.readFile(path.join(output,'predictions.json')));assert.equal(predictions.length,3516);
const reviewer={Contradiction:0,Entailment:1,Neutral:2};
const pairs=answers.map((a,i)=>{
  const p=predictions[i*2],h=predictions[i*2+1];assert.equal(p.id,`external-${i}-p`);assert.equal(h.id,`external-${i}-h`);
  const covered=p.meanings.length===1&&h.meanings.length===1;
  const prediction=covered&&isDeepStrictEqual(p.meanings[0],h.meanings[0])?1:null;
  return {...a,reviewed_label:reviewer[a.reviewer]??null,covered,prediction};
});
function metrics(rows,label){
  const answered=rows.filter(r=>r.prediction!==null),correct=rows.filter(r=>r.prediction!==null&&r.prediction===label(r)).length;
  const labels=[0,1,2].map(key=>{
    const tp=rows.filter(r=>r.prediction===key&&label(r)===key).length;
    const fp=rows.filter(r=>r.prediction===key&&label(r)!==key).length;
    const fn=rows.filter(r=>r.prediction!==key&&label(r)===key).length;
    return {label:key,support:tp+fn,tp,fp,fn,f1:2*tp/(2*tp+fp+fn||1)};
  });
  return {total:rows.length,covered_pairs:rows.filter(r=>r.covered).length,answered:answered.length,correct,abstained:rows.length-answered.length,coverage:answered.length/(rows.length||1),accuracy:correct/(rows.length||1),answered_accuracy:answered.length?correct/answered.length:null,macro_f1:labels.reduce((n,r)=>n+r.f1,0)/3,labels};
}
await write('pair-receipts.json',pairs);
const report={schema:'axon-public-language-diagnostic/v1',source:provenance.repository,revision:provenance.revision,split:'original test, all pairs',sentences:predictions.length,sentence_coverage:predictions.filter(r=>r.meanings.length>0).length,
  original_labels:metrics(pairs,r=>r.label),reviewer_agreed:metrics(pairs.filter(r=>r.reviewed_label!==null),r=>r.reviewed_label),reviewer_clean:metrics(pairs.filter(r=>r.reviewed_label!==null&&r.quality==='N'),r=>r.reviewed_label),
  core:summary,whole_harness_wall_ms:performance.now()-started,predictions_sha256:sha(await fs.readFile(path.join(output,'predictions.json'))),m1_complete:false,
  scope:'Equivalence-only semantic interpretation diagnostic, not a full NLI classifier. External corpus, original quality flags; unknowns count as abstentions, never as correct Neutral. Public test consumed by this run; no fresh-reserve claim after adaptation. RSS/energy unavailable.'};
await write('audit.json',report);console.log(JSON.stringify({sentences:report.sentences,covered_sentences:report.sentence_coverage,pairs:pairs.length,answered:report.original_labels.answered,correct:report.original_labels.correct,m1_complete:false}));
