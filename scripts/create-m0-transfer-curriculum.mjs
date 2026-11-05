import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const [directory] = process.argv.slice(2);
if (!directory) throw Error('usage: bun scripts/create-m0-transfer-curriculum.mjs <new-directory-under-evals/curricula>');
fs.mkdirSync(directory);
const write = (file, value) => fs.writeFileSync(path.join(directory, file), JSON.stringify(value, null, 2), { flag: 'wx' });
const root = JSON.parse(fs.readFileSync('evals/curricula/inquiry-v1.json', 'utf8'));
const parents = JSON.parse(fs.readFileSync('evals/curricula/inquiry-v1-data.json', 'utf8'));
const integer = x => { const n = Number(x); if (!Number.isSafeInteger(n)) throw Error('reference integer overflow'); return n; };
const step = (index, a, b) => index === 0 ? 2n * a + b : index === 1 ? a * a + b : a * b + b * b;
const future = (index, input) => integer(input.reduce((a, b) => step(index, a, BigInt(b)), 0n));
const names = ['linear-recurrence', 'quadratic-recurrence', 'bilinear-recurrence'];
for (let index = 0; index < names.length; index++) {
  const pairs = [[-2, 3], [4, -3], [0, 5], [3, 7], [-5, -2], [8, 0]];
  const examples = pairs.map(input => ({ input, output: integer(step(index, ...input.map(BigInt))) }));
  parents.lessons.push({ id: `transition-${index}`, family: `transition-source-${index}`, examples: examples.slice(0, 3), admission: examples.slice(3) });
  for (let i = 0; i < 8; i++) {
    const input = [i * 3 - 11, 13 - i * 2];
    parents.requests.push({ id: `transition-${index}-${i}`, family: `transition-source-${index}`, text: `aplique transition-${index} em dados`, bindings: { dados: input }, expected: integer(step(index, ...input.map(BigInt))) });
  }
}
const full = structuredClone(parents), external = [], answers = [], portable = [], portableAnswers = [];
let state = 0x5b41cd79;
const random = n => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) % n; };
const seen = names.map(() => new Set([[], [1,-1,2], [-1,2,0,1], [2,-1], [-2,0,1], [1,0,-1,1,0]].map(JSON.stringify)));
const make = index => {
  for (let attempt = 0; attempt < 10000; attempt++) {
    const input = Array.from({ length: index === 1 ? 2 + random(5) : 2 + random(12) }, () => index === 1 ? random(3) - 1 : random(7) - 3);
    const key = JSON.stringify(input);
    if (!seen[index].has(key)) { seen[index].add(key); return input; }
  }
  throw Error('unique future input space exhausted');
};
for (let index = 0; index < names.length; index++) {
  const id = `recurrence-${index}`;
  const examples = [[], [1, -1, 2], [-1, 2, 0, 1]].map(input => ({ input, output: future(index, input) }));
  const admission = [[2, -1], [-2, 0, 1], [1, 0, -1, 1, 0]].map(input => ({ input, output: future(index, input) }));
  full.lessons.push({ id, family: names[index], examples, admission });
  for (let replicate = 0; replicate < 16; replicate++) {
    const inputs = Array.from({ length: 16 }, () => make(index));
    for (const inherit of [true, false]) {
      const taskId = `reserved-${index}-${replicate}-${Number(inherit)}`;
      external.push({ id: taskId, train: examples, inputs, inherit });
      answers.push({ id: taskId, family: names[index], replicate, inherit, expected: inputs.map(input => future(index, input)) });
    }
  }
  for (let i = 0; i < 128; i++) {
    const input = make(index);
    full.requests.push({ id: `recurrence-${index}-${i}`, family: names[index], text: `aplique ${id} em dados`, bindings: { dados: input }, expected: future(index, input) });
    const another = make(index);
    const caseId = `portable-${index}-${i}`;
    portable.push({ id: caseId, family: names[index], text: `aplique ${id} em dados`, bindings: { dados: another }, expected: null });
    portableAnswers.push({ id: caseId, family: names[index], input: another, expected: future(index, another) });
  }
}
write('source-data.json', parents);
write('full-data.json', full);
for (const [name, data] of [['source', parents], ['full', full]]) {
  const source = structuredClone(root);
  source.id = `m0-transfer-${name}-v1`;
  // Curriculum dependencies are resolved relative to this file, not the process cwd.
  source.algorithms = `${name}-data.json`;
  write(`${name}.json`, source);
}
write('external-inputs.json', external);
write('external-answers.json', answers);
write('portable-inputs.json', portable);
write('portable-answers.json', portableAnswers);
const files = fs.readdirSync(directory).sort().map(name => {
  const bytes = fs.readFileSync(path.join(directory, name));
  return { name, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
});
write('data-freeze.json', { input_seed: '0x5b41cd79', files, families: names, independent_algorithm_classes: 3, paired_replicates_per_class: 16, cases_per_variant: 768,
  source_lessons: parents.lessons.length, full_lessons: full.lessons.length, full_requests: full.requests.length,
  scope: 'Prospective local transfer from learned two-argument state transitions to variable-length temporal recurrences. Three structural/mathematical classes, not 48 independent algorithms. Full target lessons are absent from the source-only seed. Typed I/O, grammar/lifting recipes and numeric domains are provided. Test answers are in separate files. No claim about ARC, free language or arbitrary new families.' });
console.log(JSON.stringify({ directory, source_lessons: parents.lessons.length, full_lessons: full.lessons.length, requests: full.requests.length }));
