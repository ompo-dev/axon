import fs from 'node:fs';
import path from 'node:path';

const [protocolPath, dataPath, profile = 'complete'] = process.argv.slice(2);
if (!dataPath || !['complete', 'prefix'].includes(profile)) throw Error('usage: bun scripts/create-stateful-curriculum.mjs <new-protocol.json> <new-data.json> [complete|prefix]');
const root = JSON.parse(fs.readFileSync('evals/curricula/abstractions-v2.json', 'utf8'));
const campaign = profile === 'complete'
  ? JSON.parse(fs.readFileSync('evals/curricula/abstractions-v2-data.json', 'utf8'))
  : { version: 1, lessons: [], requests: [], changes: [], policy_trials: 0, file_demo: null };
if (profile === 'complete') campaign.version = 2;
root.id = path.basename(protocolPath, '.json');
root.algorithms = path.basename(dataPath);
const lesson = (id, family, rows) => {
  const examples = rows.map(([input, output]) => ({ input, output }));
  return { id, family, examples: examples.slice(0, 3), admission: examples.slice(3) };
};
const sources = [
  lesson('key-query', 'stateful-query', [
    [[[], 'a'], 0], [[ [['a', 12], ['b', -4]], 'a'], 12], [[ [['c', 6], ['d', 7]], 'd'], 7],
    [[ [['x', -9], ['y', 4]], 'x'], -9], [[ [['a', 3], ['b', 8]], 'z'], 0],
  ]),
  lesson('without-key', 'stateful-without', [
    [[[], 'a'], []], [[ [['a', 2], ['b', 3], ['c', 4]], 'b'], [['a', 2], ['c', 4]]],
    [[ [['z', 4], ['b', -2], ['a', 7]], 'z'], [['b', -2], ['a', 7]]],
    [[ [['x', 5], ['y', -3], ['z', 7]], 'z'], [['x', 5], ['y', -3]]], [[ [['k', 4]], 'other'], [['k', 4]]],
  ]),
  lesson('project-record', 'stateful-project', [
    [['a', 2, 3], ['a', 6]], [['b', -4, 5], ['b', -20]], [['c', 7, -2], ['c', -14]],
    [['new', 9, 8], ['new', 72]], [['other', 0, -3], ['other', 0]],
  ]),
  lesson('state-update', 'stateful-update', [
    [[[], ['azul', 2]], [['azul', 2]]],
    [[ [['azul', 7], ['verde', -2]], ['azul', 3]], [['verde', -2], ['azul', 10]]],
    [[ [['amarelo', 5], ['verde', 9]], ['novo', -4]], [['amarelo', 5], ['verde', 9], ['novo', -4]]],
    [[ [['u', -7], ['v', 12]], ['u', 2]], [['v', 12], ['u', -5]]], [[ [['u', 9]], ['u', -9]], [['u', 0]]],
  ]),
];
campaign.lessons.push(...sources);
function totalByKey(rows) {
  const state = new Map();
  for (const [key, amount] of rows) {
    const next = (state.get(key) ?? 0) + amount;
    state.delete(key); state.set(key, next);
  }
  return [...state];
}
if (profile === 'complete') campaign.lessons.push(
  lesson('group-total', 'stateful-key-total', [
    [[], []], [[['a', 2], ['b', 3], ['a', 5]], [['b', 3], ['a', 7]]],
    [[['c', -4], ['c', 3], ['b', 0], ['c', 2]], [['b', 0], ['c', 1]]],
    [[['q', -3], ['q', 7], ['r', 12]], [['q', 4], ['r', 12]]],
    [[['u', 0], ['v', 9], ['u', -2], ['v', -9]], [['u', -2], ['v', 0]]],
  ]),
  lesson('project-records', 'stateful-project-map', [
    [[], []], [[['a', 2, 3], ['b', 4, 5]], [['a', 6], ['b', 20]]],
    [[['c', -3, 4], ['d', 5, -2], ['c', 7, 3]], [['c', -12], ['d', -10], ['c', 21]]],
    [[['new', 3, -5]], [['new', -15]]], [[['u', 0, 4], ['v', 6, 8]], [['u', 0], ['v', 48]]],
  ]),
  lesson('weighted-group', 'stateful-weighted-group', [
    [[], []], [[['a', 2, 3], ['b', 3, 4], ['a', 5, 2]], [['b', 12], ['a', 16]]],
    [[['c', -4, 2], ['c', 3, 2], ['b', 0, 9], ['c', 2, -3]], [['b', 0], ['c', -8]]],
    [[['u', 2, 5], ['v', -3, 4], ['u', -2, 5]], [['v', -12], ['u', 0]]],
    [[['x', 7, -2], ['y', 3, 9]], [['x', -14], ['y', 27]]],
  ]),
  lesson('symbol-unit', 'stateful-unit', [['a', ['a', 1]], ['b', ['b', 1]], ['c', ['c', 1]], ['new', ['new', 1]], ['other', ['other', 1]]]),
  lesson('symbol-records', 'stateful-unit-map', [
    [[], []], [['a', 'b', 'c'], [['a', 1], ['b', 1], ['c', 1]]], [['a', 'c', 'a'], [['a', 1], ['c', 1], ['a', 1]]],
    [['u'], [['u', 1]]], [['v', 'u'], [['v', 1], ['u', 1]]],
  ]),
  lesson('histogram', 'stateful-histogram', [
    [[], []], [['a', 'b', 'a'], [['b', 1], ['a', 2]]], [['c', 'c', 'b', 'c'], [['b', 1], ['c', 3]]],
    [['x', 'y', 'x', 'z'], [['y', 1], ['x', 2], ['z', 1]]], [['u', 'u', 'u'], [['u', 3]]],
  ]),
);
const request = (id, family, operation, input, expected) => ({ id, family, text: `aplique ${operation} em dados`, bindings: { dados: input }, expected });
campaign.requests.push(
  request('new-query', 'stateful-query', 'key-query', [[['novo', -7], ['outro', 5]], 'novo'], -7),
  request('new-without', 'stateful-without', 'without-key', [[['novo', -7], ['outro', 5]], 'novo'], [['outro', 5]]),
  request('new-project', 'stateful-project', 'project-record', ['novo', -8, 7], ['novo', -56]),
  request('new-update', 'stateful-update', 'state-update', [[['novo', 8], ['outro', 3]], ['novo', -2]], [['outro', 3], ['novo', 6]]),
);
if (profile === 'complete') {
  for (let length = 0; length < 50; length++) for (let shift = 0; shift < 5; shift++) {
    const key = i => shift % 3 === 0 ? `nova-${i}` : shift % 3 === 1 ? i : i === 0 ? false : i > 6 ? true : `chave-${i}`;
    const rows = Array.from({ length }, (_, i) => [key((i * 7 + shift) % 13), (i * 3 + shift) % 19 - 9]);
    const records = rows.map(([k, n], i) => [k, n, (i * 5 + shift) % 11 - 5]);
    const projected = records.map(([k, n, price]) => [k, n * price]);
    const symbols = Array.from({ length }, (_, i) => `simbolo-${(i * 11 + shift) % 9}`);
    const suffix = `${length}-${shift}`;
    campaign.requests.push(
      request(`group-${suffix}`, 'stateful-key-total', 'group-total', rows, totalByKey(rows)),
      request(`project-${suffix}`, 'stateful-project-map', 'project-records', records, projected),
      request(`weighted-${suffix}`, 'stateful-weighted-group', 'weighted-group', records, totalByKey(projected)),
      request(`histogram-${suffix}`, 'stateful-histogram', 'histogram', symbols, totalByKey(symbols.map(k => [k, 1]))),
    );
  }
  campaign.requests.push(request('new-symbol', 'stateful-unit', 'symbol-unit', 'palavra-inteiramente-nova', ['palavra-inteiramente-nova', 1]));
}
for (const [file, value] of [[protocolPath, root], [dataPath, campaign]]) fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ id: root.id, profile, lessons: campaign.lessons.length, requests: campaign.requests.length, scope: 'Declared development and regression tasks, not an independent M0 reserve; generic constructors and transition lifting are provided priors, source procedures are induced.' }));
