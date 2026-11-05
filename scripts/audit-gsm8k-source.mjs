import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {partitionQuestions} from './gsm8k-source.mjs';

const [corpus,partition,output]=process.argv.slice(2);
if(!output)throw Error('usage: node scripts/audit-gsm8k-source.mjs <corpus> <partition-run> <new-output>');
const sha=b=>createHash('sha256').update(b).digest('hex');
const freezeBytes=await fs.readFile(path.join(partition,'freeze.json')),freeze=JSON.parse(freezeBytes);
for(const [name,hash]of Object.entries(freeze.sources))assert.equal(sha(await fs.readFile(path.join(partition,'source',name))),hash);
assert.equal(sha(await fs.readFile(new URL('./gsm8k-source.mjs',import.meta.url))),freeze.sources['scripts/gsm8k-source.mjs']);
const protocol=JSON.parse(await fs.readFile(path.join(partition,'source/evals/protocols/m1-public-math-source-v1.json')));
const provenanceBytes=await fs.readFile(path.join(corpus,'provenance.json')),provenance=JSON.parse(provenanceBytes);
assert.equal(sha(provenanceBytes),freeze.provenance_sha256);assert.equal(provenance.revision,protocol.revision);
assert.deepEqual(Object.keys(provenance.sources).sort(),['LICENSE','README.md','train.jsonl']);
assert(!(await fs.readdir(corpus)).some(name=>name.startsWith('test')||name.startsWith('example_model')));
for(const [name,s]of Object.entries(provenance.sources))assert.equal(sha(await fs.readFile(path.join(corpus,name))),s.sha256);
const raw=await fs.readFile(path.join(corpus,'train.jsonl'));assert.equal(sha(raw),freeze.data_sha256);assert.equal(sha(raw),protocol.train_sha256);
const records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse);assert.equal(records.length,protocol.original_train_rows);
let answerReads=0;
for(const record of records)Object.defineProperty(record,'answer',{get(){answerReads++;throw Error('Answer access during metadata audit');}});
const actual=partitionQuestions(records,protocol.exposed_question_prefixes);
const stored=JSON.parse(await fs.readFile(path.join(partition,'partition.json')));assert.deepEqual(actual.rows,stored);
// Independent spelling of normalization and grouping; no annotation decoder or answer fields.
const normalize=q=>q.toLowerCase().trim().split(/\s+/u).join(' '),template=q=>normalize(q).replace(/\p{Number}+/gu,'<n>');
const exposed=protocol.exposed_question_prefixes.map(template),counts={train:0,development:0,reserved:0,research_only:0};
const expected=records.map((r,ordinal)=>{
  const t=template(r.question),hash=sha(t),remainder=Buffer.from(hash,'hex').readUInt32BE(0)%5;
  const split=exposed.some(p=>t.startsWith(p))?'research_only':remainder===0?'reserved':remainder===1?'development':'train';
  counts[split]++;return {ordinal,split,question_sha256:sha(normalize(r.question)),template_sha256:hash};
});
assert.deepEqual(expected,stored);assert.equal(answerReads,0);assert.deepEqual(counts,actual.statistics.counts);
await fs.mkdir(output);await fs.copyFile(process.argv[1],path.join(output,'auditor.mjs'),fs.constants.COPYFILE_EXCL);
const report={schema:'axon-gsm8k-source-audit/v1',verified:true,records:records.length,counts,actual_answer_getter_accesses:answerReads,
  source_sha256:sha(raw),partition_freeze_sha256:sha(freezeBytes),auditor_sha256:sha(await fs.readFile(process.argv[1])),
  original_test_downloaded:false,labels_used_for_partition:false,training_started:false,inference_started:false,m1_complete:false,m2_permitted:false,
  scope:'Actual question records tested with throwing answer getters and separately recomputed fingerprints. Host JSON parsing is not OS isolation or untouched raw answer bytes. This proves source partition integrity, not authorship separation, semantic generalization or math ability.'};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
