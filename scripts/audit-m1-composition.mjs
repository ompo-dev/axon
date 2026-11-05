import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const [baseline, candidate, output] = process.argv.slice(2);
if (!output) throw Error('usage: bun scripts/audit-m1-composition.mjs <baseline.outputs> <candidate.outputs> <new-report.json>');
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const annotations = await read('evals/curricula/m1-composition-v1-annotations.json');
// Independent host reference: parse the declared surface language, not the
// candidate's inferred tree or the generator's annotated plans/answers.
function reference(text) {
  const words = text.split(' '); let pos = 0;
  const take = word => { if (words[pos++] !== word) throw Error('syntax'); };
  const expression = () => {
    const token = words[pos++];
    if (/^-?\d+$/.test(token ?? '')) return BigInt(token);
    switch (token) {
      case 'oposto': take('de'); return -expression();
      case 'dobro': take('de'); return 2n * expression();
      case 'soma': { take('de'); const a = expression(); take('e'); return a + expression(); }
      case 'retire': { const a = expression(); take('de'); return expression() - a; }
      default: throw Error('unknown');
    }
  };
  try {
    let value = expression();
    if (words[pos] === 'mais' || words[pos] === 'menos') {
      const op = words[pos++], right = expression();
      value = op === 'mais' ? value + right : value - right;
    }
    if (pos !== words.length) return null;
    const result = Number(value); assert(Number.isSafeInteger(result)); return result;
  } catch { return null; }
}
for (const annotation of annotations) assert.equal(reference(annotation.text), annotation.expected);
const reports = {};
for (const [name, directory] of [['baseline', baseline], ['candidate', candidate]]) {
  const raw = await fs.readFile(`${directory}/agent-predictions.json`), rows = JSON.parse(raw), variants = {};
  assert.equal(rows.length, annotations.length * 3);
  for (const row of rows) {
    const index = Number(row.episode.split('-').at(-1)) * 64 + row.goal, a = annotations[index];
    assert.equal(a.text, row.input);
    const expected = reference(row.input), correct = row.actual === expected;
    assert.equal(row.expected, expected); assert.equal(row.correct, correct);
    if (expected === null) {
      assert.equal(row.queries, 0);
      assert(row.error && !row.error.includes('quota'));
      if (a.family === 'ambiguous-association' && name === 'candidate') assert.equal(row.error, 'conflicting structural interpretations');
    }
    const group = variants[row.variant] ??= { correct: 0, total: 0, positive_correct: 0, positive_total: 0, correct_refusals: 0, refusal_total: 0, portable_correct: 0, families: {} };
    const f = group.families[a.family] ??= { correct: 0, total: 0 };
    group.correct += Number(correct); group.total++; f.correct += Number(correct); f.total++;
    group.portable_correct += Number(row.child_actual === expected && row.child_correct);
    if (expected !== null) { group.positive_correct += Number(correct); group.positive_total++; }
    else { group.correct_refusals += Number(correct); group.refusal_total++; }
  }
  const result = await read(`${directory}/structural-result.json`);
  assert.equal(result.correct, 7400); assert.equal(result.dialogue_correct, 3100); assert.equal(result.algorithm_correct, 4693);
  const teaching = await read(`${directory}/agent-result.json`);
  reports[name] = { variants, integrated_result: result, learning_controls: teaching, predictions_sha256: createHash('sha256').update(raw).digest('hex') };
}
for (const variant of ['active','passive','no-library']) {
  assert.equal(reports.candidate.variants[variant].positive_correct, 1024);
  assert.equal(reports.candidate.variants[variant].correct_refusals, 64);
  assert.equal(reports.candidate.variants[variant].portable_correct, 1088);
  assert(reports.candidate.variants[variant].correct > reports.baseline.variants[variant].correct);
}
const report = { schema:'axon-m1-development-audit/v1', reports, development_passed:true, m1_complete:false,
  scope:'Same synthetic authored surface fragments, new combinations and numerical values; not independent-authorship language. Serialized per-request children share process, so this alone is not a new cross-process portability proof. Active/passive/no-library vary numeric acquisition, not the language composer; the frozen M0 executable is the composition ablation. Models receive local meaning demonstrations and existing typed interfaces. Research and grammar priors are provided.' };
await fs.writeFile(output, JSON.stringify(report, null, 2), {flag:'wx'});
console.log(JSON.stringify({baseline:reports.baseline.variants.active,candidate:reports.candidate.variants.active,m1_complete:false}));
