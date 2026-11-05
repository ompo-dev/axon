import assert from 'node:assert/strict';
import ts from '../ui/node_modules/typescript/lib/typescript.js';

export function scalar(token){
  if(typeof token!=='string')throw Error('number token');
  const s=token.replace(/^\$/u,'').replace(/[.,?!;:]$/u,'');
  if(!/^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?$/u.test(s))throw Error('decimal token');
  const n=Number(s);if(!Number.isFinite(n)||Math.abs(n)>1e9)throw Error('number bounds');return n===0?0:n;
}
export function question(text,maximumTokens=64){
  if(![64,512].includes(maximumTokens))throw Error('question interface');
  if(typeof text!=='string'||Buffer.byteLength(text)>4096)throw Error('question bytes');
  const normalized=text.replace(/\s+/gu,' ').trim();if(!normalized||/\p{Cc}/u.test(normalized))throw Error('question controls');
  const words=normalized.toLowerCase().replace(/[.?!]+$/u,'').split(' ');
  if(words.length>maximumTokens||words.some(w=>Buffer.byteLength(w)>256))throw Error('question tokens');
  const quantities=[];for(const [index,word]of words.entries()){try{quantities.push({index,value:scalar(word),word});}catch{}}
  return {text:normalized,words,quantities};
}
export function expression(text){
  if(typeof text!=='string'||!text.trim()||text.length>256)throw Error('expression size');
  const source=ts.createSourceFile('arithmetic.ts',`(${text})`,ts.ScriptTarget.Latest,true,ts.ScriptKind.TS);
  if(source.parseDiagnostics.length||source.statements.length!==1||!ts.isExpressionStatement(source.statements[0]))throw Error('expression syntax');
  let nodes=0;
  function visit(node,depth=0){
    if(++nodes>127||depth>16)throw Error('expression quota');
    if(ts.isParenthesizedExpression(node))return visit(node.expression,depth+1);
    if(ts.isNumericLiteral(node)){if(!/^(?:0|[1-9]\d*)(?:\.\d+)?$/u.test(node.getText(source)))throw Error('literal syntax');return scalar(node.getText(source));}
    if(ts.isPrefixUnaryExpression(node)&&[ts.SyntaxKind.MinusToken,ts.SyntaxKind.PlusToken].includes(node.operator)&&ts.isNumericLiteral(node.operand)){const n=visit(node.operand,depth+1);return node.operator===ts.SyntaxKind.MinusToken?-n:n;}
    if(ts.isBinaryExpression(node)){
      const op=new Map([[ts.SyntaxKind.PlusToken,'+'],[ts.SyntaxKind.MinusToken,'-'],[ts.SyntaxKind.AsteriskToken,'*'],[ts.SyntaxKind.SlashToken,'/']]).get(node.operatorToken.kind);
      if(op)return [op,visit(node.left,depth+1),visit(node.right,depth+1)];
    }
    throw Error('expression node');
  }
  return visit(source.statements[0].expression);
}
export function evaluate(tree,input=[]){
  let n;if(typeof tree==='number')n=tree;else if(typeof tree==='string'){assert(/^x[0-3]$/u.test(tree));n=input[Number(tree[1])];}else{
    assert(Array.isArray(tree)&&tree.length===3);const a=evaluate(tree[1],input),b=evaluate(tree[2],input);
    switch(tree[0]){case '+':n=a+b;break;case '-':n=a-b;break;case '*':n=a*b;break;case '/':if(b===0)throw Error('zero division');n=a/b;break;default:throw Error('operator');}
    if(n===0&&a!==0&&b!==0&&['*','/'].includes(tree[0]))throw Error('underflow');
  }
  if(!Number.isFinite(n))throw Error('nonfinite result');return n===0?0:n;
}
export const agrees=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-9*(1+Math.abs(b));
export function finalScalar(answer){
  if(typeof answer!=='string'||answer.length>16384)throw Error('answer size');
  const match=/\n#### ([+-]?(?:0|[1-9]\d*)(?:\.\d+)?)\s*$/u.exec(answer);if(!match)throw Error('final scalar');return scalar(match[1]);
}
export function teaching(questionText,answer,maximumTokens=64){
  const q=question(questionText,maximumTokens),prior=[];if(q.quantities.length===0)throw Error('no quantities');
  const final=finalScalar(answer),matches=[...answer.matchAll(/<<([^<>]+)>>/gu)];
  if(matches.length===0||matches.length>16)throw Error('annotation count');
  function bind(tree){
    if(typeof tree!=='number')return [tree[0],bind(tree[1]),bind(tree[2])];
    const choices=q.quantities.filter(v=>v.value===tree).map(v=>`q${v.index}`);
    choices.push(...prior.filter(v=>v.value===tree).map(v=>v.tree));
    const distinct=new Map(choices.map(c=>[JSON.stringify(c),c]));
    if(distinct.size!==1)throw Error(distinct.size?'ambiguous binding':'unbound literal');return distinct.values().next().value;
  }
  for(const match of matches){
    const parts=match[1].split('=');if(parts.length!==2)throw Error('annotation equality');
    const ast=expression(parts[0]),value=scalar(parts[1].trim());if(!agrees(evaluate(ast),value))throw Error('rounded/inconsistent annotation');
    const tree=bind(ast);let count=0;
    function bounded(node){if(++count>12)throw Error('expanded program quota');if(Array.isArray(node)){bounded(node[1]);bounded(node[2]);}}
    bounded(tree);prior.push({value,tree});
  }
  if(!agrees(prior.at(-1).value,final))throw Error('final mismatch');
  const order=[],ops=[];
  function compile(tree){
    if(ops.length>=12)throw Error('expanded program quota');
    if(typeof tree==='string'){
      if(!order.includes(tree))order.push(tree);const index=order.indexOf(tree);ops.push(`x${index}`);return `x${index}`;
    }
    const a=compile(tree[1]),b=compile(tree[2]);ops.push(tree[0]);return [tree[0],a,b];
  }
  const tree=compile(prior.at(-1).tree);
  if(order.length<1||order.length>4||ops.length>12)throw Error('expanded program quota');
  const tags=q.words.map(()=>'O');for(const [i,key]of order.entries())tags[Number(key.slice(1))]=`B:x${i}`;
  const values=order.map(key=>q.quantities.find(v=>v.index===Number(key.slice(1))).value);
  if(!agrees(evaluate(tree,values),final))throw Error('expanded mismatch');
  return {...q,tree,arity:order.length,ops,key:JSON.stringify(ops),tags,values,final};
}
export function numbersOnly(example){
  const indices=example.words.map((_,i)=>i).filter(i=>example.quantities.some(q=>q.index===i));
  if(indices.length===0)throw Error('no quantities');return {text:indices.map(i=>example.words[i]).join(' '),tags:indices.map(i=>example.tags[i])};
}
export function tuples(arity,start,count){
  return Array.from({length:count},(_,j)=>Array.from({length:arity},(_,i)=>1+((j+start+1)*(i*7+3)+i*i*5)%23));
}
