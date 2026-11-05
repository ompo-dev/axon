import fs from 'node:fs';
import path from 'node:path';

const [protocolPath, dataPath] = process.argv.slice(2);
if (!dataPath) throw Error('usage: bun scripts/create-dependency-curriculum.mjs <new-protocol.json> <new-data.json>');
const root = JSON.parse(fs.readFileSync('evals/curricula/algorithms-v3.json', 'utf8'));
const campaign = JSON.parse(fs.readFileSync('evals/curricula/algorithms-v3-data.json', 'utf8'));
root.id = path.basename(protocolPath, '.json');
root.algorithms = path.basename(dataPath);
campaign.policy_trials = 0;
const inputs = [[[], 2], [[2, 3, 2], 2], [[4, 1, 4], 4], [[7, 2, 7], 7], [[-3, 5, -3, 8], -3]];
const selected = ([values, q]) => values.filter(n => n === q).map(n => n * n);
const sum = values => values.reduce((a, b) => a + b, 0);
function lesson(id, family, rows) {
  const examples = rows.map(([input, output]) => ({ input, output }));
  return { id, family, examples: examples.slice(0, -2), admission: examples.slice(-2) };
}
campaign.lessons.push(
  lesson('map-square', 'dependency-map', [
    [[], []], [[2, -3], [4, 9]], [[1, 4, 5], [1, 16, 25]], [[7, 2], [49, 4]], [[-2, 6, -5], [4, 36, 25]],
  ]),
  lesson('selected-squares', 'dependency-select-map', inputs.map(x => [x, selected(x)])),
  lesson('report-squares', 'dependency-report-original', inputs.map(x => [x, sum(selected(x))])),
);
function request(id, family, operation, x, expected) {
  return { id, family, text: `aplique ${operation} em dados`, bindings: { dados: x }, expected };
}
campaign.requests.push(
  request('before-change-report', 'dependency-report-original', 'report-squares', [[2, 7, 2], 2], 8),
  request('before-change-map', 'dependency-map', 'map-square', [3, -4], [9, 16]),
);
campaign.changes[0].requests.push(
  request('invalidated-report-refuses', 'dependency-invalidated', 'report-squares', [[2, 7, 2], 2], null),
  request('independent-composition-retained', 'dependency-select-map', 'selected-squares', [[2, 7, 2], 2], [4, 4]),
);
campaign.changes.push({
  operation: 'report-squares', input: [[2, 7, 2], 2], actual: 10,
  teaching: lesson('report-squares', 'dependency-report-changed', inputs.map(x => [x, sum(selected(x)) + selected(x).length])),
  requests: [
    request('revalidated-report', 'dependency-report-changed', 'report-squares', [[2, 7, 2], 2], 10),
    request('independent-map-retained', 'dependency-map', 'map-square', [3, -4], [9, 16]),
  ],
});
for (const [file, value] of [[protocolPath, root], [dataPath, campaign]]) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ id: root.id, lessons: campaign.lessons.length, requests: campaign.requests.length, changes: campaign.changes.length, scope: 'Engineering regression; reused development data, not a fresh M0 transfer reserve.' }));
