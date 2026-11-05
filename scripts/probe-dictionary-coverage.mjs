import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';
import {canonical} from './massive-grammar.mjs';

const [factsPath,trainingPath,output]=process.argv.slice(2);
if(!output)throw Error('usage: node scripts/probe-dictionary-coverage.mjs <own-pt-facts.tsv> <frozen-training.json> <new-output>');
await fs.mkdir(output);
await fs.copyFile(process.argv[1],path.join(output,'probe.mjs'),fs.constants.COPYFILE_EXCL);
const sha=b=>createHash('sha256').update(b).digest('hex');
const raw=await fs.readFile(factsPath),trainingBytes=await fs.readFile(trainingPath);
const provenance=JSON.parse(await fs.readFile(`${factsPath}.provenance.json`));
assert.equal(sha(raw),provenance.factsSha256.toLowerCase());
const rows=parse(raw,{columns:true,delimiter:'\t'}),words=new Map(),senses=new Map(),lowercase=new Set();
for(const r of rows.filter(r=>r.relation==='word')) {
  const word=canonical(r.object),list=words.get(word)??new Set();list.add(r.sense);words.set(word,list);
  const aliases=senses.get(r.sense)??new Set();aliases.add(word);senses.set(r.sense,aliases);
  if(r.object===r.object.toLowerCase())lowercase.add(word);
}
const eligible=w=>lowercase.has(w)&&/^[\p{L}][\p{L}-]*$/u.test(w)&&words.get(w)?.size===1;
const examples=JSON.parse(trainingBytes).examples,frequency=new Map();let contextTokens=0;
for(const e of examples) {
  const tokens=canonical(e.text).split(' ');assert.equal(tokens.length,e.tags.length);
  tokens.forEach((w,i)=>{if(e.tags[i]==='O'){contextTokens++;frequency.set(w,(frequency.get(w)??0)+1);}});
}
const links=[];
for(const [word,count] of frequency) {
  if(!eligible(word))continue;
  const sense=[...words.get(word)][0],aliases=[...senses.get(sense)].filter(w=>w!==word&&eligible(w)).sort();
  if(aliases.length)links.push({word,training_context_tokens:count,sense,aliases});
}
links.sort((a,b)=>b.training_context_tokens-a.training_context_tokens||a.word.localeCompare(b.word));
const report={schema:'axon-dictionary-coverage-diagnostic/v1',facts_sha256:sha(raw),training_sha256:sha(trainingBytes),probe_sha256:sha(await fs.readFile(process.argv[1])),
  source:provenance.source,authors:provenance.authors,license:provenance.license,
  training_rows:examples.length,context_tokens:contextTokens,context_word_types:frequency.size,
  eligible_context_word_types:links.length,eligible_context_tokens:links.reduce((n,r)=>n+r.training_context_tokens,0),
  aliases:new Set(links.flatMap(r=>r.aliases)).size,links_file:'unvalidated-context-candidates.json',augmentation_permitted:false,requires_contextual_admission:true,
  development_text_evaluated:false,reserved_evaluated:false,original_test_evaluated:false,m1_complete:false,
  scope:'Coverage only, no training or inference. All senses count toward ambiguity; require a single source synset and lowercase single-word aliases on both sides. Dictionary monosemy does not guarantee context, morphology, sentence meaning or safe task-preserving augmentation.'};
await fs.writeFile(path.join(output,'unvalidated-context-candidates.json'),JSON.stringify(links,null,2),{flag:'wx'});
await fs.writeFile(path.join(output,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify({...report,examples:links.slice(0,12)}));
