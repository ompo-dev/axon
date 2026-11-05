import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {partitionQuestions,sha} from './gsm8k-source.mjs';

const [corpus='.axon-memory/corpora/gsm8k-3101c7d50724',output='.axon-memory/m1-gsm8k-partition-v1',protocolFile='evals/protocols/m1-public-math-source-v1.json']=process.argv.slice(2);
const protocolBytes=await fs.readFile(protocolFile),protocol=JSON.parse(protocolBytes);
assert.equal(protocol.schema,'axon-public-math-source-admission/v1');
const provenanceBytes=await fs.readFile(path.join(corpus,'provenance.json')),provenance=JSON.parse(provenanceBytes);
assert.equal(provenance.revision,protocol.revision);assert.equal(provenance.original_test_downloaded,false);
assert.equal((await fs.stat(path.join(corpus,'train.jsonl'))).size,4166206);
const raw=await fs.readFile(path.join(corpus,'train.jsonl'));assert.equal(sha(raw),protocol.train_sha256);assert.equal(sha(raw),provenance.sources['train.jsonl'].sha256);
const records=raw.toString('utf8').trimEnd().split(/\r?\n/u).map(JSON.parse);assert.equal(records.length,protocol.original_train_rows);
const result=partitionQuestions(records,protocol.exposed_question_prefixes);
await fs.mkdir(output);const sources={};
for(const file of [protocolFile,process.argv[1],'scripts/gsm8k-source.mjs','scripts/gsm8k-source.test.mjs','scripts/pinned-github-source.mjs','scripts/pinned-github-source.test.mjs','scripts/prepare-gsm8k-source.mjs']) {
  const relative=path.relative(process.cwd(),path.resolve(file));assert(!relative.startsWith('..')&&!path.isAbsolute(relative));
  const bytes=await fs.readFile(file);sources[relative.split(path.sep).join('/')]=sha(bytes);const destination=path.join(output,'source',relative);
  await fs.mkdir(path.dirname(destination),{recursive:true});await fs.writeFile(destination,bytes,{flag:'wx'});
}
await fs.writeFile(path.join(output,'freeze.json'),JSON.stringify({sources,protocol_sha256:sha(protocolBytes),data_sha256:sha(raw),provenance_sha256:sha(provenanceBytes),scope:'Question-only partition before answer-property access or learning. JSON bytes parsed into host objects, not OS-isolated labels. Original test not downloaded; README exposure declared.'},null,2),{flag:'wx'});
await fs.writeFile(path.join(output,'partition.json'),JSON.stringify(result.rows,null,2),{flag:'wx'});
const report={schema:'axon-gsm8k-question-partition/v1',...result.statistics,readme_interpreted:true,exposed_readme_example:true,original_test_downloaded:false,answer_properties_accessed:false,training_started:false,inference_started:false,m1_complete:false,m2_permitted:false};
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
