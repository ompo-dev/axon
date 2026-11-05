import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const [directory] = process.argv.slice(2);
if (!directory) throw Error('usage: bun scripts/create-m0-state-transfer-curriculum.mjs <new-directory>');
fs.mkdirSync(directory);
const write = (name, data) => fs.writeFileSync(path.join(directory, name), JSON.stringify(data, null, 2), { flag: 'wx' });
const root = JSON.parse(fs.readFileSync('evals/curricula/m0-transfer-v1/full.json', 'utf8'));
const source = JSON.parse(fs.readFileSync('evals/curricula/m0-transfer-v1/full-data.json', 'utf8'));
const scalar = x => { const n = Number(x); if (!Number.isSafeInteger(n)) throw Error('exact state reference'); return n; };
const histories = input => { let total = 0n; const values = []; for (const x of input) { total += BigInt(x); values.push(scalar(total)); } return [scalar(total), values]; };
const target = (index, input) => index === 0 ? [...input].reverse() : index === 1 ? histories(input) : scalar(input.reduce((state, [add, modulus]) => (state + BigInt(add)) % BigInt(modulus), 0n));
const sources = [
  ['wrap-state', 'state-source-wrap', x => [x], [-4,0,7,12,-9]],
  ['prepend-state', 'state-source-prepend', ([items, x]) => [x,...items], [[[],3],[[1,2],7],[[-4,5,1],0],[[2],9],[[-1,8],-3]]],
  ['initial-state', 'state-source-initial', x => [x,[]], [-4,0,7,12,-9]],
  ['history-state', 'state-source-history', ([[total, items], x]) => { const n = scalar(BigInt(total) + BigInt(x)); return [n,[...items,n]]; }, [[[0,[]],3],[[5,[2,5]],-2],[[-3,[9]],7],[[4,[1,-8]],0],[[7,[]],-5]]],
  ['state-total', 'state-source-total', ([state,[add,modulus]]) => [scalar(BigInt(state)+BigInt(add)),modulus], [[1,[2,5]],[4,[3,5]],[0,[9,7]],[3,[5,4]],[8,[1,6]]]],
  ['state-remainder', 'state-source-remainder', ([value,modulus]) => scalar(BigInt(value)%BigInt(modulus)), [[7,3],[12,5],[3,7],[-8,3],[17,4]]],
  ['modular-state', 'state-source-modular', ([state,[add,modulus]]) => scalar((BigInt(state)+BigInt(add))%BigInt(modulus)), [[1,[2,5]],[4,[3,5]],[0,[9,7]],[3,[5,4]],[8,[1,6]],[6,[4,9]]]],
];
for (const [id, family, oracle, inputs] of sources) {
  const rows = inputs.map(input => ({ input, output: oracle(input) }));
  source.lessons.push({ id, family, examples: rows.slice(0,3), admission: rows.slice(3) });
  for (const [i, row] of rows.entries()) source.requests.push({ id: `state-source-${id}-${i}`, family, text: `aplique ${id} em dados`, bindings: { dados: row.input }, expected: row.output });
}
const full = structuredClone(source), external = [], answers = [], portable = [], portableAnswers = [];
const families = ['order-reversal', 'cumulative-state-history', 'modulus-changing-state'];
const teaching = [
  [[], ['a','b'], [3,-4,5]],
  [[], [3,-2,7], [-4,1,-3,2]],
  [[], [[2,5],[4,7]], [[9,4],[3,5],[6,8]]],
];
const admission = [
  [['x','y','z'],[8,1],[false,true,true,false]],
  [[7,-9], [2,0,-4,1], [-1]],
  [[[7,3],[9,5]], [[4,6],[8,7],[3,4]], [[5,2]]],
];
let state = 0x4c09b8a3;
const random = n => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) % n; };
const seen = families.map((_, i) => new Set([...teaching[i],...admission[i]].map(JSON.stringify)));
const make = index => {
  for (let attempt = 0; attempt < 10000; attempt++) {
    const input = Array.from({ length: 2 + random(index === 1 ? 30 : 18) }, () => index === 0 ? [random(23)-11, `unseen-${random(11)}`, random(2) === 0][random(3)] : index === 1 ? random(23)-11 : [random(32),2+random(16)]);
    const key = JSON.stringify(input);
    if (!seen[index].has(key)) { seen[index].add(key); return input; }
  }
  throw Error('state future space exhausted');
};
for (let index = 0; index < families.length; index++) {
  const id = `state-reserve-${index}`;
  const examples = teaching[index].map(input => ({ input, output: target(index,input) }));
  full.lessons.push({ id, family: families[index], examples, admission: admission[index].map(input => ({ input, output: target(index,input) })) });
  for (let replicate = 0; replicate < 16; replicate++) {
    const inputs = Array.from({ length: 16 }, () => make(index));
    for (const inherit of [true,false]) {
      const taskId = `state-reserved-${index}-${replicate}-${Number(inherit)}`;
      external.push({ id: taskId, train: examples, inputs, inherit });
      answers.push({ id: taskId, family: families[index], replicate, inherit, expected: inputs.map(input => target(index,input)) });
    }
  }
  for (let i = 0; i < 128; i++) {
    const input = make(index);
    full.requests.push({ id: `state-reserve-${index}-${i}`, family: families[index], text: `aplique ${id} em dados`, bindings: { dados: input }, expected: target(index,input) });
    const another = make(index), caseId = `state-portable-${index}-${i}`;
    portable.push({ id: caseId, family: families[index], text: `aplique ${id} em dados`, bindings: { dados: another }, expected: null });
    portableAnswers.push({ id: caseId, family: families[index], input: another, expected: target(index,another) });
  }
}
for (const [name, data] of [['source',source],['full',full]]) {
  const config = structuredClone(root);
  config.id = `m0-state-${name}-${path.basename(directory).split('-').at(-1)}`;
  config.algorithms = `${name}-data.json`;
  write(`${name}.json`,config); write(`${name}-data.json`,data);
  const alias = structuredClone(config);
  alias.algorithms = `${config.id}-data.json`;
  for (const [file, value] of [[`${config.id}.json`,alias],[alias.algorithms,data]]) fs.writeFileSync(path.join('evals/curricula',file),JSON.stringify(value,null,2),{flag:'wx'});
}
write('external-inputs.json',external); write('external-answers.json',answers);
write('portable-inputs.json',portable); write('portable-answers.json',portableAnswers);
const files = fs.readdirSync(directory).sort().map(name => { const raw = fs.readFileSync(path.join(directory,name)); return {name,bytes:raw.length,sha256:createHash('sha256').update(raw).digest('hex')}; });
write('data-freeze.json',{input_seed:'0x4c09b8a3',files,families,independent_algorithm_classes:3,paired_replicates_per_class:16,cases_per_variant:768,protocol:'evals/protocols/m0-completion-v2.json',generator:'scripts/create-m0-state-transfer-curriculum.mjs',
  reserve_target_ids:[0,1,2].map(i=>`algorithm-state-reserve-${i}`),required_source_ids:sources.map(([id])=>`algorithm-${id}`),
  source_lessons:source.lessons.length,full_lessons:full.lessons.length,full_requests:full.requests.length,
  scope:'Prospective transfer to order transformation, augmented state/history and changing-modulus dynamics, not reparameterizations of the consumed v1 recurrence reserve. The failed v1 remains development and all its integrated tasks remain in the curriculum. Source lessons teach only local constructors/updates; the three new sequential targets are withheld from the source seed. Future labels are separate; no claim of arbitrary family transfer, raw perception or free language.'});
console.log(JSON.stringify({directory,source_lessons:source.lessons.length,full_lessons:full.lessons.length,requests:full.requests.length}));
