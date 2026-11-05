import fs from 'node:fs';
import path from 'node:path';

const [protocolPath, dataPath, revision = 'v1'] = process.argv.slice(2);
if (!dataPath) throw Error('usage: bun scripts/create-abstraction-curriculum.mjs <new-protocol.json> <new-data.json>');
if (!['v1', 'v2'].includes(revision)) throw Error('unknown development revision');
const root = JSON.parse(fs.readFileSync('evals/curricula/dependencies-v1.json', 'utf8'));
const campaign = JSON.parse(fs.readFileSync('evals/curricula/dependencies-v1-data.json', 'utf8'));
root.id = path.basename(protocolPath, '.json');
root.algorithms = path.basename(dataPath);
if (revision === 'v2') {
  const request = campaign.changes[0].requests.find(r => r.id === 'invalidated-report-refuses');
  if (!request) throw Error('missing previous implementation-dependent assertion');
  request.id = 'independently-admitted-report-retained';
  request.family = 'dependency-report-independent';
  request.expected = 8;
}
const teaching = [
  [[[0, 12, 13], [2, 14, 12], [0, 13, 17]], 1],
  [[[1, -12, 14], [0, 15, 12], [1, 17, -11]], 1],
  [[[1, 16, 12], [2, 11, 19], [0, -13, -12]], 2],
  [[[0, 14, -11], [1, 12, 15], [2, 17, 12]], 1],
  [[[1, -14, 13], [1, 12, -12], [2, 11, 11]], 1],
];
function answer([rows, limit], reduce, column, equal, weighted) {
  const selected = rows.filter(r => equal ? r[0] === limit : r[0] < limit);
  const mapped = selected.map(r => weighted ? r[1] * r[2] : r[column]);
  return reduce ? mapped.reduce((sum, n) => sum + n, 0) : mapped;
}
for (const [id, reduce, column, equal, weighted] of [
  ['project-before', false, 1, false, false], ['project-equal', false, 2, true, false],
  ['total-before', true, 1, false, false], ['total-equal', true, 2, true, false],
  ['weighted-list', false, 0, false, true], ['weighted-total', true, 0, false, true],
]) {
  const examples = teaching.map(input => ({ input, output: answer(input, reduce, column, equal, weighted) }));
  const family = weighted ? `abstraction-weighted-${reduce ? 'total' : 'list'}` : `abstraction-source-${id}`;
  campaign.lessons.push({ id, family, examples: examples.slice(0, 3), admission: examples.slice(3) });
  if (!weighted) continue;
  for (let length = 0; length < 40; length++) {
    for (let limit = -3; limit < 5; limit++) {
      const rows = Array.from({ length }, (_, i) => [i * 17 % 13 - 6, i * 3 % 11 - 5, i * 7 % 17 - 8]);
      const input = [rows, limit];
      campaign.requests.push({ id: `${id}-${length}-${limit + 3}`, family, text: `aplique ${id} em dados`, bindings: { dados: input }, expected: answer(input, reduce, 0, false, true) });
    }
  }
}
for (const [file, value] of [[protocolPath, root], [dataPath, campaign]]) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ id: root.id, revision, lessons: campaign.lessons.length, requests: campaign.requests.length, new_target_cases: 640, scope: 'Declared development families and reused regression data; not an independent M0 transfer reserve. v2 asserts retention of the independently admitted inline report rather than assuming a particular search result.' }));
