import assert from 'node:assert/strict';
import {isDeepStrictEqual} from 'node:util';
import {scalar,agrees,evaluate} from './math-program-source.mjs';

const operators=new Map([['math-p032','+'],['math-p033','-'],['math-p034','*'],['math-p035','/']]);
export function canonicalInitialFrame(frame,focused){
  const words=frame.split(' '),step=focused?0:words.length-2,index=focused?words.length-2:words.length-3;
  assert.equal(words[focused?1:0],'axon-question');assert.equal(words[step],'axon-step-0');assert.equal(words.at(-1),'axon-memory');assert(index>=(focused?2:1));
  words[index]=words[index].replace(/[.?!]+$/u,'');assert(words[index].length>0);return words.join(' ');
}
export function target(example){
  const words=example.text.toLowerCase().trim().replace(/[.?!]+$/u,'').split(/\s+/u);
  assert.equal(words.length,example.tags.length);
  const slots=example.tags.flatMap((tag,i)=>tag.startsWith('B:')?[[tag.slice(2),words[i].replace(/[.?!]+$/u,'')]]:[]);
  return ['intent',example.intent,slots.length?['slots',...slots]:'none'];
}
export function selected(term){
  if(!Array.isArray(term)||term.length!==3||term[0]!=='intent'||!(operators.has(term[1])||term[1]==='math-done'))return null;
  const slots=term[2];if(!Array.isArray(slots)||slots[0]!=='slots'||slots.length<2||slots.length>3)return null;
  let a,b;
  try{
    for(const slot of slots.slice(1)){
      if(!Array.isArray(slot)||slot.length!==2||typeof slot[1]!=='string')return null;
      const [role,value]=slot,n=scalar(value);
      if(role==='a'&&a===undefined)a=n;
      else if(role==='b'&&b===undefined)b=n;
      else if(role==='both'&&a===undefined&&b===undefined){a=n;b=n;}
      else return null;
    }
    if(a===undefined||(term[1]==='math-done'?b!==undefined:b===undefined))return null;
    return {program:term[1],operands:b===undefined?[a]:[a,b]};
  }catch{return null;}
}
export function compare(prediction,expected){
  const p=selected(prediction),e=selected(expected);assert(e,'Diagnostic expected action invalid');
  const exact=isDeepStrictEqual(prediction,expected),operation=p!==null&&p.program===e.program;
  const ordered=operation&&p.operands.every((n,i)=>agrees(n,e.operands[i]));
  const swapped=operation&&['math-p032','math-p034'].includes(e.program)&&agrees(p.operands[0],e.operands[1])&&agrees(p.operands[1],e.operands[0]);
  const result=action=>action.program==='math-done'?action.operands[0]:evaluate([operators.get(action.program),'x0','x1'],action.operands);
  let value=false;try{value=p!==null&&agrees(result(p),result(e));}catch{}
  return {exact,operation,semantic:ordered||swapped,value,refused:p===null};
}
export function summarize(rows){
  const result={total:rows.length,exact:0,operation:0,semantic:0,value:0,refused:0};
  for(const row of rows)for(const key of ['exact','operation','semantic','value','refused'])result[key]+=Number(row[key]);
  return result;
}
