import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {parse} from '../ui/node_modules/csv-parse/lib/sync.js';

export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export const reservedDocument = id => parseInt(sha(id).slice(0, 8), 16) % 5 === 0;
const upos = new Set('ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X'.split(' '));
const integer = s => /^[1-9][0-9]*$/u.test(s);
const relation = s => /^[a-z]+(?::[a-z]+)?$/u.test(s);
function attributes(raw, strict = true) {
  if (raw === '_') return {};
  const result = {};
  for (const entry of raw.split('|')) {
    const p = entry.indexOf('=');
    if (!strict && p === -1) {
      assert(entry.length > 0 && !Object.hasOwn(result, entry), 'Duplicate/empty miscellaneous field');
      Object.defineProperty(result, entry, {value: true, enumerable: true});
      continue;
    }
    assert(p > 0 && p < entry.length - 1, `Malformed attribute: ${entry}`);
    const key = entry.slice(0, p);
    assert(!Object.hasOwn(result, key), 'Duplicate attribute');
    Object.defineProperty(result, key, {value: entry.slice(p + 1), enumerable: true});
  }
  return result;
}
export function projective(words) {
  // A projective dependency has every intervening word below its head.
  return words.every(w => {
    const lo = Math.min(w.id, w.head), hi = Math.max(w.id, w.head);
    for (let i = lo + 1; i < hi; i++) {
      let node = i;
      while (node !== 0 && node !== w.head) node = words[node - 1].head;
      if (node !== w.head) return false;
    }
    return true;
  });
}
export function parseSentence(lines, metadata) {
  const comments = lines.filter(line => line.startsWith('#'));
  const textLines = comments.filter(line => line.startsWith('# text = '));
  assert.equal(textLines.length, 1, `${metadata.id}: expected one text`);
  const text = textLines[0].slice(9);
  assert(text.length > 0 && text.length <= 32768, 'Sentence text quota');
  const rows = parse(lines.filter(line => !line.startsWith('#')).join('\n'), {
    delimiter: '\t', quote: false, relax_column_count: false,
    skip_empty_lines: false, max_record_size: 65536,
  });
  assert(rows.length > 0 && rows.length <= 2048, 'Sentence row quota');
  const words = [], ranges = [], empty = [];
  let lastWord = 0;
  const ids = new Set();
  for (const row of rows) {
    assert.equal(row.length, 10, 'CoNLL-U requires ten columns');
    assert(row.every(field => field.length > 0), 'Empty column');
    const [id, form, lemma, pos, xpos, feats, head, deprel, deps, misc] = row;
    assert(!ids.has(id), 'Duplicate token ID'); ids.add(id);
    assert(form.length <= 4096 && lemma.length <= 4096, 'Token quota');
    const features = attributes(feats), spacing = attributes(misc, false);
    if (integer(id)) {
      assert.equal(Number(id), ++lastWord, 'Noncontiguous word IDs');
      assert(upos.has(pos) && /^(0|[1-9][0-9]*)$/u.test(head) && relation(deprel), 'Invalid basic dependency');
      assert.equal(head === '0', deprel === 'root', 'Root/head disagreement');
      words.push({id: Number(id), form, lemma, upos: pos, xpos, features, head: Number(head), deprel, deps, misc: spacing});
    } else if (/^[1-9][0-9]*-[1-9][0-9]*$/u.test(id)) {
      const [start, end] = id.split('-').map(Number);
      assert(start === lastWord + 1 && end > start, 'Invalid multiword range');
      assert([lemma, pos, xpos, head, deprel, deps].every(x => x === '_'), 'Annotated multiword row');
      assert(feats === '_' || feats === 'Typo=Yes', 'Invalid multiword features');
      assert(!ranges.some(r => start <= r.end), 'Overlapping multiword ranges');
      ranges.push({start, end, form, misc: spacing});
    } else {
      assert(/^[0-9]+\.[1-9][0-9]*$/u.test(id), 'Invalid token ID');
      const [base, sequence] = id.split('.').map(Number);
      assert(base === lastWord && sequence === 1 + empty.filter(e => e.base === base).length, 'Invalid empty-node order');
      assert(head === '_' && deprel === '_' && deps !== '_', 'Invalid empty-node dependency');
      empty.push({id, base, form, lemma, upos: pos, xpos, features, deps, misc: spacing});
    }
  }
  assert(words.length > 0 && words.length <= 512, 'Word quota');
  assert(ranges.every(r => r.end <= words.length), 'Multiword range outside sentence');
  assert.equal(words.filter(w => w.head === 0).length, 1, 'Expected one root');
  for (const w of words) {
    assert(w.head <= words.length && w.head !== w.id, 'Head outside sentence/self-loop');
    const visited = new Set([w.id]);
    let cursor = w.head;
    while (cursor !== 0) {
      assert(!visited.has(cursor), 'Dependency cycle');
      visited.add(cursor); cursor = words[cursor - 1].head;
    }
  }
  // Enhanced annotations are retained, not used to teach the basic parser.
  for (const node of [...words, ...empty]) if (node.deps !== '_') {
    for (const arc of node.deps.split('|')) {
      const p = arc.indexOf(':');
      const target = arc.slice(0, p), label = arc.slice(p + 1);
      assert(p > 0 && (target === '0' || ids.has(target)) && /^[a-z]+(?::[a-z]+)*$/u.test(label), 'Invalid enhanced arc');
    }
  }
  const surfaces = [];
  for (let i = 1; i <= words.length;) {
    const range = ranges.find(r => r.start === i);
    const word = words[i - 1];
    surfaces.push({form: range?.form ?? word.form, misc: range?.misc ?? word.misc,
      words: range ? words.slice(range.start - 1, range.end).map(w => w.id) : [i]});
    i = range ? range.end + 1 : i + 1;
  }
  const reconstructed = surfaces.map((s, i) => s.form + (i + 1 === surfaces.length || s.misc.SpaceAfter === 'No' ? '' : ' ')).join('');
  return {...metadata, text, words, surfaces, empty, comments, projective: projective(words), surface_matches_text: reconstructed === text};
}

export function parseCorpus(raw, partition, accept = () => true) {
  assert(['train', 'dev'].includes(partition), 'Original test is not admitted');
  const text = new TextDecoder('utf-8', {fatal: true}).decode(raw);
  assert(text.length <= 32000000, 'Source quota');
  let document = null;
  const accepted = [], excluded = [], ids = new Set();
  for (const block of text.trim().split(/\r?\n\r?\n/u)) {
    const lines = block.split(/\r?\n/u);
    // Only these metadata fields are inspected before the reservation decision.
    const field = name => lines.filter(line => line.startsWith(`# ${name} = `)).map(line => line.slice(name.length + 5));
    // Bosque r2.16 uses the historical newdoc_id spelling.
    const newdoc = [...field('newdoc id'), ...field('newdoc_id')];
    assert(newdoc.length <= 1, 'Duplicate document metadata');
    if (newdoc.length) document = newdoc[0];
    const sent = field('sent_id');
    assert.equal(sent.length, 1, 'Missing/duplicate sentence identity');
    assert(document && sent[0] && !ids.has(sent[0]), 'Missing document or duplicate sentence ID');
    ids.add(sent[0]);
    const metadata = {id: sent[0], document, partition};
    if (!accept(metadata)) {excluded.push(metadata); continue;}
    accepted.push(parseSentence(lines, metadata));
  }
  return {accepted, excluded};
}
