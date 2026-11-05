import assert from 'node:assert/strict';
import {scalar} from './math-program-source.mjs';
import {canonicalExamples,focusExample} from './math-actor-source.mjs';

export function pointerExamples(episode){
  return canonicalExamples(episode.examples).map((original,step)=>{
    const example=focusExample(original,step),words=example.text.split(' '),candidates=[];let left,right=null;
    for(const [i,word]of words.entries()){
      try{scalar(word);candidates.push(i);}catch{}
      if(example.tags[i]==='B:a')left=i;
      if(example.tags[i]==='B:b')right=i;
      if(example.tags[i]==='B:both'){left=i;right=i;}
    }
    if(words.length>544||Buffer.byteLength(example.text)>8192||words.some(w=>Buffer.byteLength(w)>256)||!candidates.length||candidates.length>32)throw Error('pointer source quota');
    assert(candidates.includes(left));assert.equal(right===null,example.intent==='math-done');if(right!==null)assert(candidates.includes(right));
    const targets=[{label:example.intent,left,right}];
    if(['math-p032','math-p034'].includes(example.intent)&&left!==right)targets.push({label:example.intent,left:right,right:left});
    return {input:{text:example.text,candidates},targets};
  });
}
