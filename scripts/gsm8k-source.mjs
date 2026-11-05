import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

export const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
export const questionKey=text=>text.toLowerCase().trim().replace(/\s+/gu,' ');
// Fingerprint only: this is not a number parser or an arithmetic interpretation.
export const questionTemplate=text=>questionKey(text).replace(/\p{N}+/gu,'<n>');
export function partitionQuestions(records,exposedPrefixes=[]) {
  assert(Array.isArray(records)&&records.length>0&&records.length<=10000,'Math source record quota');
  assert(Array.isArray(exposedPrefixes)&&exposedPrefixes.length<=64&&exposedPrefixes.every(p=>typeof p==='string'&&p.trim().length>0&&p.length<=8192),'Exposure prefixes');
  const exposed=exposedPrefixes.map(questionTemplate),counts={train:0,development:0,reserved:0,research_only:0},templates=new Map(),questions=new Set();
  const rows=records.map((record,ordinal)=>{
    assert(record&&typeof record==='object'&&!Array.isArray(record),'Math source record');
    assert.deepEqual(Object.keys(record).sort(),['answer','question'],'Math source fields');
    const question=record.question;
    assert(typeof question==='string'&&question.trim().length>0&&Buffer.byteLength(question)<=8192&&!/\p{Cc}/u.test(question.replace(/\s/gu,'')),'Question bounds');
    const template=questionTemplate(question),fingerprint=sha(template),questionHash=sha(questionKey(question));
    const split=exposed.some(prefix=>template.startsWith(prefix))?'research_only':['reserved','development','train','train','train'][Number.parseInt(fingerprint.slice(0,8),16)%5];
    assert(!templates.has(fingerprint)||templates.get(fingerprint)===split,'Template split conflict');templates.set(fingerprint,split);questions.add(questionHash);counts[split]++;
    return {ordinal,split,question_sha256:questionHash,template_sha256:fingerprint};
  });
  return {rows,statistics:{records:rows.length,unique_questions:questions.size,unique_numeric_templates:templates.size,counts,
    answer_properties_accessed:false,scope:'Question-only duplicate grouping. Numeric markers are not semantic parsing, answer consistency, author separation or a new benchmark approval.'}};
}
