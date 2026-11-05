import assert from 'node:assert/strict';
import {question,scalar,evaluate,agrees} from './math-program-source.mjs';

export const OPERATIONS=new Map([['+','math-p032'],['-','math-p033'],['*','math-p034'],['/','math-p035']]);
export function decimal(value){
  if(!Number.isFinite(value)||Math.abs(value)>1e9)throw Error('actor scalar bounds');
  const s=String(value===0?0:value);if(!/[eE]/u.test(s))return s;
  const [mantissa,exponent]=s.toLowerCase().split('e'),negative=mantissa.startsWith('-'),absolute=negative?mantissa.slice(1):mantissa;
  const parts=absolute.split('.'),digits=parts.join(''),point=parts[0].length+Number(exponent);
  const expanded=point<=0?'0.'+'0'.repeat(-point)+digits:point>=digits.length?digits+'0'.repeat(point-digits.length):digits.slice(0,point)+'.'+digits.slice(point);
  return (negative?'-':'')+expanded;
}
export function frame(words,registers,maximumTokens=64){
  assert([64,544].includes(maximumTokens));
  assert(Array.isArray(words)&&words.length>0&&Array.isArray(registers)&&registers.length<=8);
  const result=['axon-question',...words,`axon-step-${registers.length}`,'axon-memory'];
  for(const [index,value]of registers.entries())result.push(`r${index}`,decimal(value));
  if(result.length>maximumTokens||result.some(w=>Buffer.byteLength(w)>256)||Buffer.byteLength(result.join(' '))>(maximumTokens===64?4096:8192))throw Error('actor frame quota');
  return result.join(' ');
}
export function focusExample(example,step){
  const words=example.text.split(' '),tags=[...example.tags],position=words.length-2*step-2;
  assert.equal(words[0],'axon-question');assert.equal(words[position],`axon-step-${step}`);assert.equal(words[position+1],'axon-memory');assert.equal(tags[position],'O');
  words.unshift(...words.splice(position,1));tags.unshift(...tags.splice(position,1));
  return {...example,text:words.join(' '),tags};
}
export function canonicalExamples(examples){
  const first=examples[0].text.split(' '),index=first.length-3;
  assert.equal(first[0],'axon-question');assert.equal(first.at(-2),'axon-step-0');assert.equal(first.at(-1),'axon-memory');assert(index>=1);
  const original=first[index],canonical=original.replace(/[.?!]+$/u,'');assert(canonical.length>0);
  return examples.map((example,step)=>{
    const words=example.text.split(' ');assert.equal(words[index],original);assert.equal(words[index+1],`axon-step-${step}`);assert.equal(words[index+2],'axon-memory');
    words[index]=canonical;return {...example,text:words.join(' '),tags:[...example.tags]};
  });
}
export function episode(example,numbersOnly=false,maximumTokens=64){
  assert([64,544].includes(maximumTokens));
  const original=question(example.text,maximumTokens===64?64:512),positions=[];
  for(let i=0;i<example.arity;i++){
    const index=example.tags.indexOf(`B:x${i}`);assert(index>=0);positions.push(index);
  }
  const included=numbersOnly?original.quantities.map(q=>q.index):original.words.map((_,i)=>i);
  const words=included.map(i=>original.words[i]);if(!words.length)throw Error('actor no quantities');
  const leaves=positions.map(index=>included.indexOf(index)+1);assert(leaves.every(i=>i>=1));
  const actions=[],memo=new Map();
  function walk(tree){
    if(typeof tree==='string'){assert(/^x[0-3]$/u.test(tree));return {kind:'question',index:leaves[Number(tree[1])],value:example.values[Number(tree[1])]};}
    const key=JSON.stringify(tree);if(memo.has(key))return memo.get(key);
    const a=walk(tree[1]),b=walk(tree[2]),value=evaluate([tree[0],'x0','x1'],[a.value,b.value]);decimal(value);
    const result={kind:'register',index:actions.length,value};actions.push({program:OPERATIONS.get(tree[0]),a,b,value});
    if(actions.length>8||!actions.at(-1).program)throw Error('actor action quota');memo.set(key,result);return result;
  }
  const root=walk(example.tree),registers=[],examples=[];
  const location=ref=>ref.kind==='question'?ref.index:words.length+4+2*ref.index;
  function label(program,a,b){
    const text=frame(words,registers,maximumTokens),tags=text.split(' ').map(()=>'O'),i=location(a),j=b?location(b):-1;
    assert(i<tags.length&&i>=0);assert.equal(scalar(text.split(' ')[i]),a.value);
    if(b){assert(j<tags.length&&j>=0);assert.equal(scalar(text.split(' ')[j]),b.value);}
    if(i===j)tags[i]='B:both';else {tags[i]='B:a';if(b)tags[j]='B:b';}
    examples.push({text,intent:program,tags});
  }
  for(const action of actions){label(action.program,action.a,action.b);registers.push(action.value);}
  label('math-done',root);
  assert(agrees(root.value,example.final));
  return {examples,actions:actions.map(({program,a,b,value})=>({program,a,b,value})),final:root.value};
}
