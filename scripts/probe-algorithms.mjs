import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { reference, audit } from './audit-algorithms.mjs';
import { parse } from '../ui/node_modules/csv-parse/lib/sync.js';

const [numberPath, outputPath, binaryPath, profile = 'base'] = process.argv.slice(2);
if (!outputPath) throw Error('usage: bun scripts/probe-algorithms.mjs <number.axnum> <new-output> [binary]');
if (!['base', 'abstractions-v1', 'stateful-v1'].includes(profile)) throw Error('unknown portability profile');
const output = path.resolve(outputPath); await fs.mkdir(output);
const binary = path.resolve(binaryPath ?? 'target/release/axon-uic-curriculum-lab.exe');
const number = path.join(output, 'isolated.axnum'); await fs.copyFile(numberPath, number, fs.constants.COPYFILE_EXCL);
const sha = raw => createHash('sha256').update(raw).digest('hex');
const inputSeed = profile === 'base' ? 0x243f6a88 : profile === 'abstractions-v1' ? 0x5ce3187b : 0x63ab7157;
let state = inputSeed;
const random = n => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) % n; };
const names = ['somar', 'quadrados', 'produto', 'maior', 'menor', 'contar', 'selecionar', 'somarmaiores', 'contarmaiores', 'palavras', 'selecionarpalavras', 'total', 'escolherregistros', 'relatorio', 'cores', 'recolorir'];
const families = ['numeric-sum', 'numeric-sum-squares', 'numeric-product', 'numeric-maximum', 'numeric-minimum', 'numeric-equality-count', 'numeric-equality-filter', 'numeric-conditional-sum', 'numeric-conditional-count', 'text-equality-count', 'text-equality-filter', 'record-weighted-reduction', 'record-conditional-filter', 'record-learned-pipeline', 'visual-count-via-learned-call', 'visual-conditional-nested-loop'];
const frames = [(op, arg) => `aplique ${op} em ${arg}`, (op, arg) => `processe ${arg} usando ${op}`, (op, arg) => `execute ${op} sobre ${arg}`, (op, arg) => `quero ${op} para ${arg}`];
const requests = [];
for (let f = 0; f < 16; f++) for (let j = 0; j < 100; j++) {
  let x = Array.from({ length: 1 + random(f === 2 ? 10 : 80) }, () => random(f === 2 ? 5 : 2001) - (f === 2 ? 2 : 1000));
  if (f >= 5 && f <= 8) { x[0] = j % 11; x = [x, j % 11]; }
  if (f === 9 || f === 10) x = [Array.from({ length: 1 + random(80) }, () => `novo${random(13)}`), `novo${random(13)}`];
  if (f >= 11 && f <= 13) {
    x = Array.from({ length: 1 + random(120) }, (_, i) => [`registro${j}-${i}`, 1 + random(33), 1 + random(500)]);
    if (f >= 12) x = [x, 1 + random(500)];
  }
  if (f >= 14) {
    x = Array.from({ length: 1 + random(24) }, () => Array.from({ length: 1 + random(24) }, () => random(4)));
    if (f === 14) x = [x, random(4)];
  }
  const r = { id: `reserve-${f}-${j}`, family: families[f], text: frames[j % 4](names[f], 'dados'), bindings: { dados: x } };
  requests.push({ ...r, expected: reference(r) });
}
for (let f = 0; f < 5; f++) for (let j = 0; j < 100; j++) {
  const q = 1 + random(4), x = [Array.from({ length: 3 + random(18) }, () => random(7)), q]; x[0][0] = q;
  const r = { id: `reserve-compose-${f}-${j}`, family: `composition-${families[f]}`, text: frames[j % 4](names[f], `(${frames[(j + 1) % 4]('selecionar', 'dados')})`), bindings: { dados: x } };
  requests.push({ ...r, expected: reference(r) });
}
let csvHash = null;
if (profile !== 'base') {
  for (const reduce of [false, true]) for (let j = 0; j < 200; j++) {
    const rows = Array.from({ length: random(170) }, () => [random(101) - 50, random(401) - 200, random(2001) - 1000]);
    const x = [rows, random(101) - 50];
    const r = { id: `portable-weighted-${reduce}-${j}`, family: `abstraction-weighted-${reduce ? 'total' : 'list'}`, text: frames[j % 4](`weighted-${reduce ? 'total' : 'list'}`, 'dados'), bindings: { dados: x } };
    requests.push({ ...r, expected: reference(r) });
  }
  const csv = path.join(output, 'portfolio.csv');
  const rows = Array.from({ length: 600 }, () => [random(41) - 20, random(61) - 30, random(2001) - 1000]);
  await fs.writeFile(csv, ['regiao,quantidade,valor', ...rows.map(r => r.join(','))].join('\n') + '\n', { flag: 'wx' });
  const raw = await fs.readFile(csv);
  csvHash = sha(raw);
  const parsed = parse(raw, { columns: true }).map(r => [r.regiao, r.quantidade, r.valor].map(Number));
  if (parsed.some(row => row.some(n => !Number.isSafeInteger(n)))) throw Error('invalid numeric CSV');
  const r = { id: 'portable-actual-csv', family: 'abstraction-weighted-total', text: 'aplique weighted-total em dados', bindings: { dados: [parsed, 5] } };
  requests.push({ ...r, expected: reference(r) });
  if (profile === 'stateful-v1') {
    const csvRequest = { id: 'portable-grouped-csv', family: 'stateful-weighted-group', text: 'aplique weighted-group em dados', bindings: { dados: parsed } };
    requests.push({ ...csvRequest, expected: reference(csvRequest) });
    const scalar = { id: 'portable-scalar-symbol', family: 'stateful-unit', text: 'aplique symbol-unit em dados', bindings: { dados: 'simbolo-escalar-inedito' } };
    requests.push({ ...scalar, expected: reference(scalar) });
    for (const [family, operation] of [['stateful-key-total', 'group-total'], ['stateful-project-map', 'project-records'], ['stateful-weighted-group', 'weighted-group'], ['stateful-histogram', 'histogram']]) {
      for (let j = 0; j < 100; j++) {
        let input = Array.from({ length: random(180) }, () => [`chave-nova-${random(17)}`, random(301) - 150]);
        if (family === 'stateful-project-map' || family === 'stateful-weighted-group') input = input.map(row => [...row, random(601) - 300]);
        if (family === 'stateful-histogram') input = input.map(row => row[0]);
        const r = { id: `portable-${family}-${j}`, family, text: frames[j % 4](operation, 'dados'), bindings: { dados: input } };
        requests.push({ ...r, expected: reference(r) });
      }
    }
  }
}
const cases = path.join(output, 'cases.json'), receipt = path.join(output, 'predictions.json');
await fs.writeFile(cases, JSON.stringify(requests), { flag: 'wx' });
await fs.writeFile(path.join(output, 'freeze.json'), JSON.stringify({ binary_sha256: sha(await fs.readFile(binary)), number_sha256: sha(await fs.readFile(number)), cases_sha256: sha(await fs.readFile(cases)), tasks: requests.length, seed: `0x${inputSeed.toString(16)}`, profile, csv_sha256: csvHash, scope: 'Deterministic portability/development inputs and compositions; same supplied DSL and taught wording, not a fresh family-transfer reserve. Optional CSV is synthetic and parsed by the declared host adapter; only the numerical workflow is learned. Child working directory contains the number and test inputs, not ancestral teaching data; this is logical isolation, not an OS sandbox.' }, null, 2), { flag: 'wx' });
const result = await promisify(execFile)(binary, ['algorithm-batch', number, path.join(output, 'birth'), cases, receipt], { cwd: output, windowsHide: true, timeout: 300000, maxBuffer: 1000000 });
console.log(result.stdout.trim());
console.log(JSON.stringify(await audit(receipt, path.join(output, 'audit.json'))));
