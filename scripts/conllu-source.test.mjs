import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseCorpus, parseSentence, projective, reservedDocument} from './conllu-source.mjs';

const rows = [
  '# newdoc id = doc1', '# sent_id = s1', '# text = Do gato.',
  '1-2\tDo\t_\t_\t_\t_\t_\t_\t_\t_',
  '1\tDe\tde\tADP\t_\t_\t3\tcase\t3:case\t_',
  '2\to\to\tDET\t_\t_\t3\tdet\t3:det\t_',
  '3\tgato\tgato\tNOUN\t_\tNumber=Sing\t0\troot\t0:root\tSpaceAfter=No',
  '4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t3:punct\t_',
];
const read = lines => parseSentence(lines, {id: 's1', document: 'doc1', partition: 'train'});
test('multiword surfaces retain distinct syntactic words and spacing', () => {
  const r = read(rows);
  assert.equal(r.text, 'Do gato.'); assert.equal(r.words.length, 4);
  assert.deepEqual(r.surfaces.map(s => s.words), [[1, 2], [3], [4]]);
  assert(r.surface_matches_text && r.projective);
  assert.deepEqual(r.words[2].features, {Number: 'Sing'});
  assert.equal(read(rows.map(r => r.replace('SpaceAfter=No', 'Flag|SpaceAfter=No'))).words[2].misc.Flag, true);
});
test('reservation filters metadata before parsing malformed held-out annotations', () => {
  const raw = '# newdoc id = held\n# sent_id = hidden\n# text = untrusted\ninvalid\n\n' + rows.join('\n') + '\n\n';
  const p = parseCorpus(Buffer.from(raw), 'train', m => m.document !== 'held');
  assert.equal(p.accepted.length, 1); assert.equal(p.excluded.length, 1);
  assert(!Object.hasOwn(p.excluded[0], 'text'));
  assert.throws(() => parseCorpus(Buffer.from(raw), 'train'));
  assert.throws(() => parseCorpus(Buffer.from(raw), 'test'));
  assert.equal(reservedDocument('stable'), reservedDocument('stable'));
});
test('cycles, roots, duplicate IDs, invalid ranges and columns fail closed', () => {
  for (const replacement of [
    ['\t3\tcase\t', '\t1\tcase\t'], ['\t0\troot\t', '\t2\troot\t'],
    ['1-2\tDo', '1-8\tDo'], ['2\to\to', '1\to\to'],
    ['Number=Sing', 'Number=Sing|Number=Plur'],
  ]) assert.throws(() => read(rows.map(r => r.replace(...replacement))));
  assert.throws(() => read([...rows, '5\tbad']));
  const cyclic = rows.map(r => r.replace('\t3\tcase\t', '\t2\tcase\t').replace('\t3\tdet\t', '\t1\tdet\t'));
  assert.throws(() => read(cyclic));
});
test('enhanced empty nodes are retained but not confused with basic tokens', () => {
  const copy = [...rows]; copy.splice(7, 0, '3.1\tfaz\tfazer\tVERB\t_\t_\t_\t_\t0:root\t_');
  const r = read(copy); assert.equal(r.words.length, 4); assert.equal(r.empty.length, 1);
  assert.throws(() => read(copy.map(s => s.replace('3.1\tfaz', '3.2\tfaz'))));
});
test('document identity carries over, comments are data, invalid UTF8 rejected', () => {
  const next = rows.slice(1).filter(r => !r.startsWith('1-2')).map(r => r.replace('s1', 's2').replace('# text = Do gato.', '# text = De o gato.'));
  const p = parseCorpus(Buffer.from(rows.join('\n') + '\n\n' + next.join('\n')), 'dev');
  assert.equal(p.accepted[1].document, 'doc1');
  assert.equal(parseCorpus(Buffer.from(rows.join('\n').replace('newdoc id', 'newdoc_id')), 'train').accepted[0].document, 'doc1');
  assert.throws(() => parseCorpus(Buffer.from([255]), 'train'));
  assert.throws(() => parseCorpus(Buffer.from(rows.join('\n') + '\n\n' + rows.join('\n')), 'train'));
  assert(!read(rows.map(r => r.replace('# text = Do gato.', '# text = Different'))).surface_matches_text);
});
test('projectivity detects a crossing while accepting a single rooted word', () => {
  assert(projective([{id: 1, head: 0}]));
  assert(!projective([{id: 1, head: 3}, {id: 2, head: 4}, {id: 3, head: 0}, {id: 4, head: 3}]));
});
