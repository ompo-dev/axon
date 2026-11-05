import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { m0Reference } from './m0-references.mjs';

const [numberPath, outputPath, executable, dataPath = 'evals/curricula/m0-state-transfer-v3'] = process.argv.slice(2);
if (!executable) throw Error('usage: bun scripts/probe-m0-state.mjs <full-number.axnum> <new-output> <candidate.exe> [data-directory]');
const output = path.resolve(outputPath); await fs.mkdir(output);
const sha = raw => createHash('sha256').update(raw).digest('hex');
const write = (name, value) => fs.writeFile(path.join(output,name),JSON.stringify(value,null,2),{flag:'wx'});
const number = path.join(output,'isolated.axnum'), binary = path.join(output,'candidate.exe');
await fs.copyFile(numberPath,number,fs.constants.COPYFILE_EXCL);
await fs.copyFile(executable,binary,fs.constants.COPYFILE_EXCL);
for (const name of ['portable-inputs.json','portable-answers.json']) await fs.copyFile(path.join(dataPath,name),path.join(output,name),fs.constants.COPYFILE_EXCL);
const inputs = path.join(output,'portable-inputs.json');
const answers = JSON.parse(await fs.readFile(path.join(output,'portable-answers.json'),'utf8'));
await write('freeze.json',{number_sha256:sha(await fs.readFile(number)),binary_sha256:sha(await fs.readFile(binary)),inputs_sha256:sha(await fs.readFile(inputs)),private_answers_sha256:sha(await fs.readFile(path.join(output,'portable-answers.json'))),cases:384,training_required:false,
  scope:'Only compiled number and requests with expected=null reach the child. Future labels remain in a separate host file; logical isolation, not an OS sandbox. Same declared three families, new inputs disjoint from source/full/external cases.'});
const started = performance.now();
const run = await promisify(execFile)(binary,['algorithm-batch',number,path.join(output,'birth'),inputs,path.join(output,'predictions.json')],{cwd:output,windowsHide:true,timeout:300000,maxBuffer:1000000});
await fs.writeFile(path.join(output,'stdout.log'),run.stdout,{flag:'wx'});
await fs.writeFile(path.join(output,'stderr.log'),run.stderr,{flag:'wx'});
const rows = JSON.parse(await fs.readFile(path.join(output,'predictions.json'),'utf8'));
if (rows.length !== answers.length || rows.length !== 384) throw Error('portable cardinality');
let correct = 0; const families = {};
for (let i = 0; i < rows.length; i++) {
  const row = rows[i], answer = answers[i], expected = m0Reference(answer.family,answer.input);
  if (row.id !== answer.id || JSON.stringify(expected) !== JSON.stringify(answer.expected) || row.expected !== null) throw Error('portable identity/label boundary');
  const match = JSON.stringify(row.actual) === JSON.stringify(expected);
  const family = families[answer.family] ??= {correct:0,total:0};
  family.correct += Number(match); family.total++; correct += Number(match);
}
await write('audit.json',{correct,total:rows.length,families,training_required:false,whole_harness_wall_ms:performance.now()-started,predictions_sha256:sha(await fs.readFile(path.join(output,'predictions.json'))),scope:'Independent array/BigInt audit of real separate-process birth, not general language or universal new-family transfer.'});
console.log(JSON.stringify({correct,total:rows.length,training_required:false}));
if (correct !== rows.length) process.exitCode = 1;
