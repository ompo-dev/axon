import assert from 'node:assert/strict';
import {scalar} from './math-program-source.mjs';

export function inspectQuestion(text){
  assert.equal(typeof text,'string');
  const normalized=text.toLowerCase().trim().replace(/[.?!]+$/u,''),words=normalized?normalized.split(/\s+/u):[],bytes=Buffer.byteLength(text),invalidToken=words.some(w=>Buffer.byteLength(w)>256),quantities=words.filter(word=>{try{scalar(word);return true;}catch{return false;}}).length;
  const frameBytes=Buffer.byteLength(['axon-question',...words,'axon-step-0','axon-memory'].join(' ')),reasons=[];
  if(!text.length||bytes>4096||/\p{Cc}/u.test(text)||invalidToken||!words.length||words.length>64)reasons.push('question-token-quota');
  if(words.length+3>64||frameBytes>4096)reasons.push('initial-frame-token-quota');
  if(quantities===0)reasons.push('no-copyable-scalar');
  return {question_tokens:words.length,initial_frame_tokens:words.length+3,initial_frame_bytes:frameBytes,copyable_scalars:quantities,unconditionally_unsupported:reasons.length>0,reasons};
}
