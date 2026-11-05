import assert from 'node:assert/strict';
import {pointerExamples} from './relational-pointer-source.mjs';
import {scalar} from './math-program-source.mjs';
import {decimal} from './math-actor-source.mjs';

export function workingFrame(question,nodes,blind=false){
  assert(question.length>0&&question.length<=512&&Buffer.byteLength(question.join(' '))<=4096&&nodes.length<=8);const words=[...question];
  for(const [i,node]of nodes.entries()){
    assert(['math-p032','math-p033','math-p034','math-p035'].includes(node.program));
    const tag=(side,ref)=>{assert(Number.isSafeInteger(ref.index)&&ref.index>=0);if(ref.kind==='question'){assert(ref.index<question.length);scalar(question[ref.index]);}else{assert.equal(ref.kind,'register');assert(ref.index<i);}return `${side}:${ref.kind==='question'?'q':'r'}${ref.index}`;};
    const a=tag('a',node.left),b=tag('b',node.right);words.push(...(blind?['work','left','right']:[`op:${node.program}`,a,b]),decimal(node.value));
  }
  const candidates=[];for(const [i,w]of words.entries()){try{scalar(w);candidates.push(i);}catch{}}
  const text=words.join(' ');assert(words.length<=544&&Buffer.byteLength(text)<=8192&&words.every(w=>Buffer.byteLength(w)<=256)&&candidates.length>0&&candidates.length<=32);return {text,candidates};
}

export function workingExamples(source,blind=false){
  // Re-index existing targets; only already executed operations enter each frame.
  const legacy=pointerExamples(source);return legacy.map((example,step)=>{
    const old=example.input.text.split(' '),count=old.length-3-step*2,question=old.slice(2,count+2);
    assert.equal(old[0],`axon-step-${step}`);assert.equal(old[1],'axon-question');assert.equal(old[count+2],'axon-memory');assert.equal(old.length,count+3+step*2);
    const origin=position=>{if(position>=2&&position<count+2)return {kind:'question',index:position-2};const n=position-count-4;assert(n>=0&&n%2===0&&n/2<step);return {kind:'register',index:n/2};};
    const position=ref=>ref.kind==='question'?ref.index:count+3+4*ref.index;
    const nodes=[];
    for(let i=0;i<step;i++){
      const prior=legacy[i].targets[0],before=legacy[i].input.text.split(' '),priorCount=before.length-3-i*2;
      const priorOrigin=p=>{if(p>=2&&p<priorCount+2)return {kind:'question',index:p-2};const n=p-priorCount-4;assert(n>=0&&n%2===0&&n/2<i);return {kind:'register',index:n/2};};
      assert(prior.right!==null&&prior.label!=='math-done');const a=priorOrigin(prior.left),b=priorOrigin(prior.right);
      nodes.push({program:prior.label,left:a,right:b,value:scalar(old[count+4+2*i])});
    }
    const input=workingFrame(question,nodes,blind),targets=example.targets.map(t=>({label:t.label,left:position(origin(t.left)),right:t.right===null?null:position(origin(t.right))}));for(const t of targets){assert(input.candidates.includes(t.left));assert(t.right===null||input.candidates.includes(t.right));}
    return {input,targets};
  });
}
