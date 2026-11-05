import assert from 'node:assert/strict';
import {scalar} from './math-program-source.mjs';
import {close} from './math-actor-trace.mjs';
import {checkPointerTrace} from './relational-pointer-trace.mjs';

export function checkWorkingTrace(row,text,blind=false){
  const question=text.toLowerCase().trim().replace(/[.?!]+$/u,'').split(/\s+/u),history=[],trace=[];
  const tag=(side,ref)=>`${side}:${ref.kind==='question'?'q':'r'}${ref.index}`;
  for(const event of row.trace){
    assert.equal(event.step,history.length);const words=event.frame.split(' ');assert(words.length<=544);assert.deepEqual(words.slice(0,question.length),question);assert.equal(words.length,question.length+4*history.length);
    const old=[`axon-step-${history.length}`,'axon-question',...question,'axon-memory'];
    for(const [i,node]of history.entries()){const start=question.length+4*i;assert.deepEqual(words.slice(start,start+3),blind?['work','left','right']:[`op:${node.program}`,tag('a',node.left),tag('b',node.right)]);assert(close(scalar(words[start+3]),node.value));old.push(`r${i}`,words[start+3]);}
    const origin=p=>{assert(Number.isSafeInteger(p)&&p>=0&&p<words.length);scalar(words[p]);if(p<question.length)return {kind:'question',index:p};const n=p-question.length-3;assert(n>=0&&n%4===0&&n/4<history.length);return {kind:'register',index:n/4};};
    const position=ref=>ref.kind==='question'?ref.index+2:question.length+4+2*ref.index;
    const converted={...event,frame:old.join(' ')};
    if(event.prediction){const {label,left,right}=event.prediction;assert.deepEqual(Object.keys(event.prediction).sort(),['label','left','right']);const a=origin(left);assert.deepEqual(event.left_origin,a);let b;
      if(right!==null){b=origin(right);if(event.step<8)assert.deepEqual(event.right_origin,b);}else assert.equal(label,'math-done');
      converted.prediction={label,left:position(a),right:b?position(b):null};
      if(Object.hasOwn(event,'computed')){assert(b);history.push({program:label,left:a,right:b,value:event.computed});}
    }
    trace.push(converted);
  }
  return checkPointerTrace({...row,trace},text);
}
