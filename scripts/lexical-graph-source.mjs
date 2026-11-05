import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const compare=(a,b)=>Buffer.compare(Buffer.from(a),Buffer.from(b));
const pair=(a,b)=>compare(a,b)<0?[a,b]:[b,a];
const key=edge=>JSON.stringify(edge);
const sort=edges=>edges.sort((a,b)=>compare(a[0],b[0])||compare(a[1],b[1]));

export function lexicalEdges(facts,vocabulary) {
  assert(Array.isArray(vocabulary)&&vocabulary.length>=16&&vocabulary.length<=4096);
  assert(vocabulary.every((w,i)=>typeof w==='string'&&w.length>0&&(!i||compare(vocabulary[i-1],w)<0)),'Vocabulary order');
  const known=new Set(vocabulary),senses=new Map();let wordFacts=0;
  for(const row of facts)if(row.relation==='word') {
    wordFacts++;assert(/^own-pt-synset-[0-9]{8}-[nvasr]$/u.test(row.sense),'Sense schema');
    const word=row.object.toLowerCase().trim();if(!known.has(word))continue;
    const words=senses.get(row.sense)??new Set();words.add(word);senses.set(row.sense,words);
  }
  const unique=new Map();
  for(const words of senses.values()) {
    const list=[...words].sort(compare);
    for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++) {const edge=[list[i],list[j]];unique.set(key(edge),edge);assert(unique.size<=50000,'Lexical edge quota');}
  }
  const edges=sort([...unique.values()]);assert(edges.length>0,'No lexical overlap');
  return {edges,statistics:{source_word_facts:wordFacts,covered_senses:senses.size,vocabulary:vocabulary.length,vertices:new Set(edges.flat()).size,edges:edges.length}};
}

export function rewireEdges(original,seed) {
  assert(Number.isSafeInteger(seed)&&seed>0&&seed<=0xffffffff);
  assert(original.length>=2&&original.length<=50000);
  const edges=original.map(e=>pair(...e)),keys=new Set(edges.map(key));assert.equal(keys.size,edges.length);
  assert(edges.every(e=>e[0]!==e[1]),'Self edge');
  let state=seed>>>0;const random=bound=>{state^=state<<13;state^=state>>>17;state^=state<<5;state>>>=0;return state%bound;};
  let accepted=0,attempts=0;
  for(;attempts<original.length*20&&accepted<original.length*10;attempts++) {
    const i=random(edges.length),j=random(edges.length);if(i===j)continue;
    let [a,b]=edges[i];const [c,d]=edges[j];if(random(2))[a,b]=[b,a];
    if(a===d||c===b)continue;
    const x=pair(a,d),y=pair(c,b),kx=key(x),ky=key(y);if(kx===ky||keys.has(kx)||keys.has(ky))continue;
    keys.delete(key(edges[i]));keys.delete(key(edges[j]));keys.add(kx);keys.add(ky);edges[i]=x;edges[j]=y;accepted++;
  }
  const old=new Set(original.map(key));
  return {edges:sort(edges),statistics:{seed,attempts,accepted_swaps:accepted,original_edges_retained:edges.filter(e=>old.has(key(e))).length,degree_preserved:true,scope:'Bounded double-edge swaps preserve every word degree and total edges. Not a uniform random-graph sampler or a linguistic corpus.'}};
}

export async function loadLexicalGraph(source) {
  const raw=await fs.readFile(source.facts),page=await fs.readFile(source.vocabulary);
  assert(raw.length<=32000000&&page.length<=196608);assert.equal(sha(raw),source.facts_sha256);assert.equal(sha(page),source.vocabulary_sha256);
  const provenanceBytes=await fs.readFile(`${source.facts}.provenance.json`),provenance=JSON.parse(provenanceBytes);assert.equal(sha(raw),provenance.factsSha256.toLowerCase());
  assert.equal(page.subarray(0,8).toString(),'AXSQ0009');
  const end=12+page.readUInt32LE(8),header=JSON.parse(page.subarray(12,end));
  assert(Number.isSafeInteger(header.word_bytes)&&header.word_bytes>0&&header.word_bytes<=65536&&end+header.word_bytes<page.length);
  const words=page.subarray(end,end+header.word_bytes).toString('utf8').split('\0');assert.equal(words.pop(),'');assert.equal(words.length,header.words);
  const graph=lexicalEdges(parse(raw,{columns:true,delimiter:'\t'}),words);
  return {...graph,statistics:{...graph.statistics,facts_sha256:sha(raw),vocabulary_sha256:sha(page),provenance_sha256:sha(provenanceBytes),license_notice_sha256:source.license_file?sha(await fs.readFile(source.license_file)):null,source:provenance.source,license:provenance.license,authors:provenance.authors,scope:'Train-vocabulary-only lexical adjacency. Word-level sense ambiguity is retained, not resolved. A soft similarity prior is not task-preserving substitution, gold inference data, or new semantic proof.'}};
}
