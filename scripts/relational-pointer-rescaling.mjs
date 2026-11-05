import assert from 'node:assert/strict';
import {scalar,evaluate} from './math-program-source.mjs';
import {decimal,frame} from './math-actor-source.mjs';

const operators=new Map([['math-p032','+'],['math-p033','-'],['math-p034','*'],['math-p035','/']]);
export function rescaleSource(source,factor=2){
  assert.equal(factor,2,'Prospective diagnostic has one fixed intervention');
  return rescaleProgram(source,factor);
}
export function rescaleProgram(source,factor){
  assert([2,4].includes(factor),'Only prospective training factors2/4');
  const before=source.examples[0].text.split(' ');assert.equal(before[0],'axon-question');assert.equal(before.at(-2),'axon-step-0');
  const original=before.slice(1,-2),words=original.map(word=>{
    let value;try{value=scalar(word);}catch{return word;}
    const prefix=word.startsWith('$')?'$':'',suffix=/[.,?!;:]$/u.test(word)?word.at(-1):'';
    return prefix+decimal(value*factor)+suffix;
  }),registers=[],actions=[];
  const value=ref=>ref.kind==='question'?scalar(words[ref.index-1]):registers[ref.index];
  for(const action of source.actions){const a=value(action.a),b=value(action.b),n=evaluate([operators.get(action.program),'x0','x1'],[a,b]);decimal(n);actions.push({...action,a:{...action.a,value:a},b:{...action.b,value:b},value:n});registers.push(n);}
  assert.equal(source.examples.length,actions.length+1);
  const examples=source.examples.map((e,step)=>({...e,text:frame(words,registers.slice(0,step),544),tags:[...e.tags]})),last=examples.at(-1),root=last.tags.indexOf('B:a');assert(root>=0);const final=scalar(last.text.split(' ')[root]);
  return {examples,actions,final};
}
