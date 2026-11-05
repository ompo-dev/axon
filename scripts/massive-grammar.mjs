import assert from 'node:assert/strict';

export const normalize = text => text.toLowerCase().trim().replace(/\s+/gu, ' ');
export const canonical = text => normalize(text).replace(/[.?!]+$/u, '');

// MASSIVE annotations are a flat bracket language, not executable markup.
export function annotation(row) {
  const text = row.annot_utt;
  assert.equal(typeof text, 'string');
  assert.equal(typeof row.utt, 'string');
  assert.equal(typeof row.intent, 'string');
  const parts = [], slots = [], template = [], segments = [];
  let position = 0;
  while (position < text.length) {
    const start = text.indexOf('[', position);
    const endPlain = start < 0 ? text.length : start;
    const plain = text.slice(position, endPlain);
    assert(!plain.includes(']'), 'Unmatched annotation closing bracket');
    parts.push(plain);
    segments.push({text:plain,role:null});
    template.push(...normalize(plain).split(' ').filter(Boolean));
    if (start < 0) break;
    const end = text.indexOf(']', start + 1);
    assert(end > start && !text.slice(start + 1, end).includes('['), 'Nested or unterminated annotation');
    const body = text.slice(start + 1, end), colon = body.indexOf(':');
    assert(colon > 0, 'Missing slot separator');
    const role = body.slice(0, colon).trim(), value = body.slice(colon + 1).trim();
    assert(/^[a-z_]+$/u.test(role) && value.length > 0, 'Invalid slot');
    parts.push(value);
    segments.push({text:value,role});
    slots.push([role, canonical(value)]);
    template.push(['slot',role]);
    position = end + 1;
  }
  assert.equal(normalize(parts.join('')), normalize(row.utt), 'Annotation does not reconstruct utterance');
  assert(slots.every(([, value]) => value.length > 0), 'Empty normalized slot');
  return {target:['intent', row.intent, slots.length ? ['slots', ...slots] : 'none'],
    family:JSON.stringify([row.intent, slots.map(([role]) => role)]),template,segments};
}

export function sequenceAnnotation(row) {
  const result=annotation(row),words=[],tags=[];
  for(const segment of result.segments) {
    const tokens=normalize(segment.text).split(' ').filter(Boolean);
    words.push(...tokens);tags.push(...tokens.map((_,i)=>segment.role?`${i?'I':'B'}:${segment.role}`:'O'));
  }
  if(words.length)words[words.length-1]=words.at(-1).replace(/[.?!]+$/u,'');
  if(words.at(-1)===''){words.pop();tags.pop();}
  assert.equal(words.join(' '),canonical(row.utt),'Annotation requires within-token segmentation');
  assert.equal(words.length,tags.length);
  return {...result,tags};
}

export function metrics(predictions, labels) {
  assert.equal(predictions.length, labels.length);
  let answered=0, exact=0, intent=0, slotExact=0;
  const families={};
  predictions.forEach((row,i) => {
    const target=labels[i].target, prediction=row.prediction;
    assert.equal(row.id, labels[i].id);
    const bucket=families[target[1]]??={total:0,answered:0,exact:0,wrong:0};
    bucket.total++;
    if(prediction===null)return;
    answered++;bucket.answered++;
    const correct=JSON.stringify(prediction)===JSON.stringify(target);
    exact+=Number(correct);bucket.exact+=Number(correct);bucket.wrong+=Number(!correct);
    intent+=Number(Array.isArray(prediction)&&prediction[0]==='intent'&&prediction[1]===target[1]);
    slotExact+=Number(Array.isArray(prediction)&&JSON.stringify(prediction[2])===JSON.stringify(target[2]));
  });
  return {total:labels.length,answered,exact,wrong:answered-exact,abstained:labels.length-answered,
    intent_correct:intent,slots_exact:slotExact,exact_accuracy:exact/(labels.length||1),families};
}
