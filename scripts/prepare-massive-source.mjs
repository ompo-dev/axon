import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';

const root='.axon-memory/corpora/massive-1.1';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytes=await fs.readFile(`${root}/1.1/data/pt-PT.jsonl`);
const rows=bytes.toString('utf8').trim().split(/\r?\n/).map(JSON.parse);
const partitions={},workers=new Map(),intents=new Map();
for(const row of rows) {
  assert.equal(row.locale,'pt-PT');assert(['train','dev','test'].includes(row.partition));
  assert.equal(typeof row.worker_id,'string');
  partitions[row.partition]=(partitions[row.partition]??0)+1;
  workers.set(row.worker_id,(workers.get(row.worker_id)??0)+1);
  intents.set(row.intent,(intents.get(row.intent)??0)+1);
}
const reservedWorkers=[...workers.keys()].filter(worker=>parseInt(sha(worker).slice(0,8),16)%5===0).sort();
const reservedIds=rows.filter(row=>reservedWorkers.includes(row.worker_id)).map(row=>row.id);
const report={schema:'axon-public-language-source/v1',source:'https://github.com/alexa/massive',
  download:'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz',
  attribution:'MASSIVE, Amazon.com Inc. or its affiliates; derived from SLURP. Dataset CC BY 4.0. Use original Portuguese pt-PT; no automatic pt-BR localization.',
  archive_sha256:sha(await fs.readFile(`${root}/original.tar.gz`)),data_sha256:sha(bytes),license_sha256:sha(await fs.readFile(`${root}/1.1/LICENSE`)),
  total:rows.length,partitions,workers:workers.size,intents:intents.size,reserved_workers:reservedWorkers,reserved_ids:reservedIds,
  reservation_rule:'SHA256 of public localization worker id modulo 5 equals zero. Do not train on any reserved worker. Localization-worker split, not necessarily original SLURP author split.',
  scope:'Metadata-only preparation; no utterances displayed, no model training or predictions on this source. Corpus alone does not provide every M1 acceptance family. Freeze appropriate contract before evaluating.'};
await fs.writeFile(`${root}/provenance.json`,JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify({total:rows.length,partitions,workers:workers.size,intents:intents.size,reserved_workers:reservedWorkers.length,reserved_utterances:reservedIds.length}));
