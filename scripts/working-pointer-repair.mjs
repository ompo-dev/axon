import assert from 'node:assert/strict';
import {sha} from './gsm8k-source.mjs';
import {scalar,evaluate,agrees} from './math-program-source.mjs';
import {workingExamples,workingFrame} from './working-pointer-source.mjs';

const operations=new Map([['math-p032','+'],['math-p033','-'],['math-p034','*'],['math-p035','/']]);
const key=(program,a,b)=>sha(JSON.stringify([program,...(['math-p032','math-p034'].includes(program)?[a,b].sort():[a,b])]));
export function repairExamples(source,rollout){
  const canonical=workingExamples(source),question=canonical[0].input.text.split(' '),keys=[],goals=[];
  const qKey=index=>`q:${index}`,sourceRef=ref=>({kind:ref.kind,index:ref.index-(ref.kind==='question'?1:0)}),refKey=(ref,known)=>ref.kind==='question'?qKey(ref.index):known[ref.index];
  for(const action of source.actions){const left=refKey(sourceRef(action.a),keys),right=refKey(sourceRef(action.b),keys);assert(left&&right&&operations.has(action.program));const k=key(action.program,left,right);goals.push({program:action.program,left,right,key:k});keys.push(k);}
  const finalPosition=canonical.at(-1).targets[0].left,goalKey=finalPosition<question.length?qKey(finalPosition):keys[(finalPosition-question.length-3)/4];assert(goalKey);
  const own=[],ownKeys=[],pairs=[],refusals=[];let noncanonical=false;
  function available(nodes){const map=new Map();for(const [i,w]of question.entries()){try{map.set(qKey(i),{ref:{kind:'question',index:i},value:scalar(w)});}catch{}}
    const expressions=[];for(const [i,node]of nodes.entries()){const a=map.get(refKey(node.left,expressions)),b=map.get(refKey(node.right,expressions));assert(a&&b);const expr=key(node.program,refKey(node.left,expressions),refKey(node.right,expressions));expressions.push(expr);map.set(expr,{ref:{kind:'register',index:i},value:node.value});}return map;
  }
  const position=ref=>ref.kind==='question'?ref.index:question.length+3+4*ref.index;
  for(const event of rollout.trace){
    if(!Object.hasOwn(event,'computed'))break;assert(operations.has(event.prediction.label));const map=available(own),a=map.get(refKey(event.left_origin,ownKeys)),b=map.get(refKey(event.right_origin,ownKeys));assert(a&&b);const value=evaluate([operations.get(event.prediction.label),'x0','x1'],[a.value,b.value]);assert(agrees(value,event.computed));
    const expr=key(event.prediction.label,refKey(event.left_origin,ownKeys),refKey(event.right_origin,ownKeys));own.push({program:event.prediction.label,left:{...event.left_origin},right:{...event.right_origin},value:event.computed});ownKeys.push(expr);noncanonical ||= expr!==keys[own.length-1];if(!noncanonical)continue;
    const nodes=structuredClone(own),plan=[];let refused;
    try{while(true){const known=available(nodes),root=known.get(goalKey);if(root){assert(agrees(root.value,source.final));plan.push({nodes:structuredClone(nodes),label:'math-done',left:root.ref,right:null});break;}
        if(nodes.length>=8)throw Error('repair action quota');const next=goals.find(g=>!known.has(g.key)&&known.has(g.left)&&known.has(g.right));assert(next,'Missing topological repair');const left=known.get(next.left),right=known.get(next.right),n=evaluate([operations.get(next.program),'x0','x1'],[left.value,right.value]);assert(Math.abs(n)<=1e9);
        plan.push({nodes:structuredClone(nodes),label:next.program,left:left.ref,right:right.ref});nodes.push({program:next.program,left:left.ref,right:right.ref,value:n});
      }}catch(error){refused=error.message;}
    if(refused){refusals.push({prefix:own.length,error:refused});continue;}
    for(const action of plan){const target={label:action.label,left:position(action.left),right:action.right===null?null:position(action.right)},targets=[target];if(['math-p032','math-p034'].includes(target.label)&&target.left!==target.right)targets.push({...target,left:target.right,right:target.left});
      const rich={input:workingFrame(question,action.nodes),targets},blind={input:workingFrame(question,action.nodes,true),targets};for(const t of targets){assert(rich.input.candidates.includes(t.left));assert(t.right===null||rich.input.candidates.includes(t.right));}
      const teacherIndex=Math.min(action.nodes.length,canonical.length-1);pairs.push({working:rich,blind,teacher_index:teacherIndex,donor_prefix:own.length,registers:action.nodes.length});
    }
  }
  return {pairs,refusals};
}
