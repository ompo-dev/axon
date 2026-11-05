import assert from 'node:assert/strict';
import {checkTrace} from './math-actor-trace.mjs';
import {scalar} from './math-program-source.mjs';

export function checkPointerTrace(row,text){
  const question=text.toLowerCase().trim().replace(/[.?!]+$/u,'').split(/\s+/u);
  const origin=position=>{
    assert(Number.isSafeInteger(position));
    if(position>=2&&position<question.length+2)return {kind:'question',index:position-2};
    const n=position-question.length-4;assert(n>=0&&n%2===0,'Selected position must have explicit numeric origin');return {kind:'register',index:n/2};
  };
  const trace=row.trace.map(event=>{
    if(!event.prediction)return event;
    const {label,left,right}=event.prediction;assert.deepEqual(Object.keys(event.prediction).sort(),['label','left','right']);
    const words=event.frame.split(' '),value=i=>{assert(Number.isSafeInteger(i)&&i>=0&&i<words.length);scalar(words[i]);return words[i].replace(/[.?!]+$/u,'');};
    const a=value(left);assert.deepEqual(event.left_origin,origin(left));if(event.left_origin.kind==='register')assert(event.left_origin.index<event.step,'No future register may be selected');
    let slots;
    if(right===null){slots=[['a',a]];assert.equal(label,'math-done');}
    else {const b=value(right);if(event.step<8){assert.deepEqual(event.right_origin,origin(right));if(event.right_origin.kind==='register')assert(event.right_origin.index<event.step);}slots=left===right?[['both',a]]:[['a',a],['b',b]];}
    return {...event,prediction:['intent',label,['slots',...slots]]};
  });
  const checked=checkTrace({...row,trace},text,true,{maximumTokens:544,actionQuotaError:'pointer action quota'});
  return {calculations:checked.calculations,order:checked.order};
}
