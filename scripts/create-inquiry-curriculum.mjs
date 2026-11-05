import fs from 'node:fs';
import path from 'node:path';
import { reference } from './audit-algorithms.mjs';

const [rootPath, dataPath] = process.argv.slice(2);
if (!dataPath) throw Error('usage: bun scripts/create-inquiry-curriculum.mjs <new-root.json> <new-data.json>');
const root = JSON.parse(fs.readFileSync('evals/curricula/stateful-v1.json', 'utf8'));
const data = JSON.parse(fs.readFileSync('evals/curricula/stateful-v1-data.json', 'utf8'));
root.id = path.basename(rootPath, '.json');
root.algorithms = path.basename(dataPath);
const families = ['numeric-sum', 'numeric-product', 'numeric-minimum', 'numeric-maximum', 'numeric-sum-squares'];
for (const [index, family] of families.entries()) {
  const id = `investigate-${index}`;
  const target = input => reference({ family, bindings: { dados: input } });
  const initial = family === 'numeric-sum-squares' ? [-1, 0, 1] : [-3, 0, 7];
  data.lessons.push({ id, family, examples: initial.map(n => ({ input: [n], output: target([n]) })),
    admission: [[-17, -8, -4], [3, 9, 14], [2, -5, 8, 1], [-7, 0, 3]].map(input => ({ input, output: target(input) })),
    inquiry: { pool: [[11], [-24], [38], [2, 5, -3], [-9, -4, -2], [3, 6, 11], [0, -4, 7, 2]].map(input => ({ input, output: target(input) })), max_queries: 2, max_hypotheses: 16 } });
  for (let n = 1; n <= 100; n++) {
    const input = Array.from({ length: 2 + n % 7 }, (_, i) => n % 4 === 0 ? -1 - (n + i) % 9 : n % 4 === 1 ? 1 + (n * 3 + i) % 9 : (n * 7 + i * 3) % 19 - 9);
    data.requests.push({ id: `inquiry-${index}-${n}`, family, text: `aplique ${id} em dados`, bindings: { dados: input }, expected: target(input) });
  }
}
for (const [file, value] of [[rootPath, root], [dataPath, data]]) fs.writeFileSync(file, JSON.stringify(value, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ lessons: data.lessons.length, requests: data.requests.length, policy_trials: data.policy_trials, scope: 'Development integration of query selection in five known families; not general language or full-M0 approval.' }));
