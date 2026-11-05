import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
export function partitionContexts(contexts) {
  assert(Array.isArray(contexts)&&contexts.length>0&&contexts.length<=3000,'Context quota');
  const metadata=[],uids=new Map(),groups=new Map();
  for(const [index,context]of contexts.entries()) {
    const {uid,table}=context.table;
    assert(typeof uid==='string'&&uid.length>0&&uid.length<=256,'Table identity');
    assert(Array.isArray(table)&&table.length>0&&table.length<=1024,'Table row quota');
    let cells=0;
    const normalized=table.map(row=>{
      assert(Array.isArray(row)&&row.length>0&&row.length<=128,'Table column quota');cells+=row.length;
      return row.map(cell=>{assert(typeof cell==='string'&&cell.length<=4096,'Table cell');return cell.toLowerCase().trim().replace(/\s+/gu,' ');});
    });
    assert(cells<=65536,'Table cell quota');
    const fingerprint=sha(JSON.stringify(normalized));
    if(uids.has(uid))assert.equal(uids.get(uid),fingerprint,'Same table UID with different cells');
    uids.set(uid,fingerprint);
    const bucket=parseInt(fingerprint.slice(0,8),16)%5;
    const partition=bucket===0?'reserved':bucket===1?'development':'train';
    groups.set(fingerprint,partition);
    metadata.push({index,uid,fingerprint,partition});
  }
  const partitions={train:[],development:[]};
  for(const row of metadata)if(row.partition!=='reserved')partitions[row.partition].push(contexts[row.index]);
  return {partitions,metadata,statistics:Object.fromEntries(['train','development','reserved'].map(partition=>[partition,{contexts:metadata.filter(r=>r.partition===partition).length,tables:[...groups.values()].filter(p=>p===partition).length}]))};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const [corpus,output]=process.argv.slice(2);assert(output,'usage: node scripts/pacific-source.mjs <pinned-corpus> <new-output>');
  await fs.mkdir(output);
  const provenance=JSON.parse(await fs.readFile(path.join(corpus,'provenance.json')));
  assert.equal(provenance.revision,'66a52ad4fd9eb7ad71e82a5c0bfb1de226366827');
  assert.equal(provenance.original_test_downloaded,false);assert.equal(provenance.original_validation_downloaded,false);
  const bytes=await fs.readFile(path.join(corpus,'train.json'));assert(bytes.length<=32000000);assert.equal(sha(bytes),provenance.sources['train.json'].sha256);
  const {metadata,statistics}=partitionContexts(JSON.parse(bytes));
  await fs.copyFile(process.argv[1],path.join(output,'partitioner.mjs'),fs.constants.COPYFILE_EXCL);
  const report={schema:'axon-pacific-metadata-partition/v1',revision:provenance.revision,source_sha256:sha(bytes),partitioner_sha256:sha(await fs.readFile(process.argv[1])),statistics,metadata,
    question_annotations_parsed:false,original_test_downloaded:false,original_validation_downloaded:false,
    rule:'SHA256 of case/whitespace-normalized complete table cells, first8hex modulo5: 0 reserved, 1 development, 2..4 train. Duplicate table contents stay together; UID conflicts fail.',
    m1_complete:false,scope:'Prospective table grouping over original train only. No question, answer, derivation, paragraphs or other label fields accessed. Does not establish disjoint source financial reports or annotation workers, independent Portuguese authorship, model accuracy or a passed M1 gate.'};
  await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify({statistics,question_annotations_parsed:false}));
}
