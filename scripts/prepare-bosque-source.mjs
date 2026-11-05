import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha, parseCorpus, reservedDocument} from './conllu-source.mjs';

const [destination = '.axon-memory/corpora/ud-bosque-r2.16', mode] = process.argv.slice(2);
assert(mode === undefined || mode === '--local', 'Unknown preparation option');
const revision = '4f136acb02344e4f2f5e3d3e9ecc70aab61db62d';
const base = `https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/${revision}/`;
const files = ['LICENSE.txt', 'README.md', 'pt_bosque-ud-train.conllu', 'pt_bosque-ud-dev.conllu'];
if (mode === undefined) await fs.mkdir(destination);
const sources = mode ? JSON.parse(await fs.readFile(path.join(destination, 'download.json'))).sources : {};
for (const name of files) if (mode === undefined) {
  const response = await fetch(base + name, {signal: AbortSignal.timeout(120000)});
  assert(response.ok, `Download failed: ${name} ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  assert(bytes.length > 0 && bytes.length <= 32000000, 'Download quota');
  await fs.writeFile(path.join(destination, name), bytes, {flag: 'wx'});
  sources[name] = {url: base + name, sha256: sha(bytes), bytes: bytes.length};
}
if (mode === undefined) await fs.writeFile(path.join(destination, 'download.json'), JSON.stringify({revision, sources}, null, 2), {flag: 'wx'});
for (const name of files) {
  const bytes = await fs.readFile(path.join(destination, name));
  assert.equal(sources[name].url, base + name);
  assert.equal(sha(bytes), sources[name].sha256); assert.equal(bytes.length, sources[name].bytes);
}
const splits = {}, reserved = [];
for (const partition of ['train', 'dev']) {
  const bytes = await fs.readFile(path.join(destination, `pt_bosque-ud-${partition}.conllu`));
  const parsed = parseCorpus(bytes, partition, m => !reservedDocument(m.document));
  reserved.push(...parsed.excluded);
  splits[partition] = parsed.accepted;
}
const docs = partition => new Set(splits[partition].map(r => r.document));
assert([...docs('dev')].every(d => !docs('train').has(d)), 'Document leakage across train/dev');
const stats = rows => ({sentences: rows.length, documents: new Set(rows.map(r => r.document)).size,
  words: rows.reduce((n, r) => n + r.words.length, 0), projective: rows.filter(r => r.projective).length,
  multiword_sentences: rows.filter(r => r.surfaces.some(s => s.words.length > 1)).length,
  surface_mismatch: rows.filter(r => !r.surface_matches_text).length,
  max_words: Math.max(...rows.map(r => r.words.length))});
const provenance = {schema: 'axon-public-syntax-source/v1', revision, source: 'https://github.com/UniversalDependencies/UD_Portuguese-Bosque',
  license: 'CC BY-SA 4.0; preserve attribution and share-alike obligations for redistributed derivatives. Not a pretrained model.',
  attribution: 'UD Portuguese Bosque / Floresta Sintatica, CETEMPublico and CETENFolha; original contributors and citations in the pinned README.',
  sources, reservation: 'SHA256(document id), first eight hex digits modulo five equals zero; filter before parsing text/annotations. Document groups do not prove distinct human authors.',
  reserved_metadata: reserved, statistics: Object.fromEntries(Object.entries(splits).map(([k, v]) => [k, stats(v)])),
  original_test_downloaded: false, reserved_annotations_parsed: false,
  scope: 'Development syntax supervision. Dependency trees are not semantics, conversational ability, or completion of M1. Raw gold annotations must not reach end-to-end inference.'};
await fs.writeFile(path.join(destination, 'provenance.json'), JSON.stringify(provenance, null, 2), {flag: 'wx'});
console.log(JSON.stringify({statistics: provenance.statistics, reserved_sentences: reserved.length, original_test_downloaded: false}));
