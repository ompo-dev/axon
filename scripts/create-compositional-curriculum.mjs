import fs from 'node:fs/promises';
import path from 'node:path';
const base=JSON.parse(await fs.readFile(new URL('../evals/curricula/structural-v3.json',import.meta.url),'utf8'));
const N=n=>['num',n], X=x=>['var',x], A=(op,...xs)=>[op,...xs], D=(f,x)=>A('d',f,X(x));
const ex=(input,output)=>({input,output}), U=s=>['utterance',...s.split(' ')];
const protocol={version:1,id:path.basename(process.argv[2]??'compositional-v1.json','.json'),lessons:structuredClone(base.lessons),cases:[],dialogues:structuredClone(base.dialogues)};
const reserved=process.argv[3]??'';
if(reserved&&!/^[a-z0-9_-]{1,24}$/.test(reserved))throw Error('Invalid reserved tag');
function teach(id,examples,admission){protocol.lessons.push({id,guards:[],examples,admission});}
function normalize(t){
  if(!Array.isArray(t))return t;
  const [op,...xs]=t,v=xs.map(normalize);
  if(['add','sub','mul','div'].includes(op)&&v.every(t=>t[0]==='num')){
    const a=v[0][1],b=v[1][1],n=op==='add'?a+b:op==='sub'?a-b:op==='mul'?a*b:a/b;
    if(Number.isSafeInteger(n))return N(n);
  }
  return A(op,...v);
}
function derivative(t,x){
  const [op,a,b]=t;
  if(op==='num')return N(0);if(op==='var')return N(a===x?1:0);
  if(op==='add'||op==='sub')return A(op,derivative(a,x),derivative(b,x));
  if(op==='mul')return A('add',A('mul',derivative(a,x),b),A('mul',a,derivative(b,x)));
  if(op==='div')return A('div',A('sub',A('mul',derivative(a,x),b),A('mul',a,derivative(b,x))),A('pow',b,N(2)));
  if(op==='pow')return b[1]===0?N(0):A('mul',A('mul',b,A('pow',a,N(b[1]-1))),derivative(a,x));
  if(op==='exp')return A('mul',A('exp',a),derivative(a,x));
  throw Error('Unsupported independent generator derivative');
}
for(const op of ['add','sub','mul','div']){
  const make=i=>{const a=[-7,3,11,-4,2,9,17,-13,23,31,41,47][i],b=[3,-5,7,11,-2,13,5,19,-7,17,23,29][i];return ex(A(op,N(op==='div'?a*b:a),N(b)),N(op==='add'?a+b:op==='sub'?a-b:op==='mul'?a*b:a));};
  teach(`evaluate-${op}`,Array.from({length:8},(_,i)=>make(i)),[make(9),make(11)]);
}
const discoveries=[
  {id:'circuit-a',f:(a,b)=>a*a+b,samples:[[1,7],[3,-2],[-4,5],[6,1],[2,19],[7,-11],[-8,3],[13,29]]},
  {id:'circuit-b',f:(a,b)=>(a-b)*(a-b),samples:[[1,7],[3,-2],[-4,5],[6,1],[2,19],[7,-11],[-8,3],[13,29]]},
  {id:'circuit-c',f:(a,b)=>a*a+b*b,samples:[[1,7],[3,-2],[-4,5],[6,1],[2,19],[7,-11],[-8,3],[13,29]]},
  {id:'circuit-d',f:(a,b)=>Math.sqrt(a*a+b*b),samples:[[3,4],[5,12],[8,15],[7,24],[20,21],[9,40],[12,35],[11,60]]},
];
for(const task of discoveries){const make=([a,b])=>ex(A(task.id,a,b),N(task.f(a,b)));teach(`discover-${task.id}`,task.samples.map(make),(task.id==='circuit-d'?[[28,45],[33,56]]:[[43,21],[-31,17]]).map(make));}
const names=['ana maria','bruno','caio silva neto','dora lima','eva','fabio de almeida','gabi costa','heitor'];
const places=['rio de janeiro','natal','belo horizonte','recife velho','palmas','sao paulo','porto alegre','vitoria'];
for(const [id,make] of [
  ['fact',(a,b)=>ex(U(`${a} mora em ${b}`),A('tell',a,'mora',b))],
  ['question',(a)=>ex(U(`onde ${a} mora`),A('ask',a,'mora'))],
  ['alias',(a)=>ex(U(`em que cidade ${a} mora`),A('ask',a,'mora'))],
  ['capital',(a,b)=>ex(U(`${a} tem capital ${b}`),A('tell',a,'capital',b))],
  ['path',(a)=>ex(U(`qual a capital do pais de ${a}`),A('path',a,'mora','capital'))],
])teach(`phrase-${id}`,names.map((a,i)=>make(a,places[i])),[make('iris do monte','vila nova'),make('joao costa junior','cidade do norte')]);
const add=(family,input,expected,forbid_heads=['d','lim','solve'])=>protocol.cases.push({id:`composed-${protocol.cases.length}`,family,input,expected,forbid_heads});
if(!reserved)for(const c of base.cases)add(c.family,c.input,c.expected===null?null:normalize(c.expected),c.forbid_heads);
for(let i=0;i<700;i++){
  const x=`${reserved}composed_x${i}`,f=i%2?A('pow',A('add',A('mul',N(2+i%7),X(x)),N(i%13)),N(2+i%8)):A('exp',A('mul',N(1+i%9),X(x)));
  let input=f,answer=f;const order=i%3===0?3:2;
  for(let k=0;k<order;k++){input=D(input,x);answer=normalize(derivative(answer,x));}
  add(`derivative-order-${order}`,input,answer);
}
for(let i=0;i<250;i++)for(const task of discoveries){
  let a=i+(reserved?10101:101),b=2*i-(reserved?3079:79);
  if(task.id==='circuit-d'){const m=i+(reserved?1041:41),n=i%29+1;a=m*m-n*n;b=2*m*n;}
  add('discovered-program',A(task.id,a,b),N(task.f(a,b)));
}
for(let i=0;i<600;i++){
  const a=`${reserved?reserved+' ':''}pessoa nova ${i} do vale`,b=`cidade ${i} da serra`;
  add('multiword-language',U(`${a} mora em ${b}`),A('tell',a,'mora',b),['utterance']);
}
const reply=(status,text=null)=>({status,text});
for(let i=0;i<100;i++){
  const who=`pessoa ${i} da silva`,country=`pais ${i} do sul`,city=`cidade ${i} do norte`,next=`novo lugar ${i}`;
  protocol.dialogues.push({id:`multiword-dialogue-${i}`,restart_at:[2,6],turns:[
    {text:`${who} mora em ${country}`,expected:reply('recorded')},
    {text:`onde ${who} mora`,expected:reply('answer',`${who} mora em ${country}`)},
    {text:'onde ele mora',expected:reply('answer',`${who} mora em ${country}`)},
    {text:`${country} tem capital ${city}`,expected:reply('recorded')},
    {text:`qual a capital do pais de ${who}`,expected:reply('answer',`a capital de ${country} e ${city}`)},
    {text:`${who} mora em ${next}`,expected:reply('recorded')},
    {text:'onde ele mora',expected:reply('answer',`${who} mora em ${next}`)},
    {text:`em que cidade ${who} mora`,expected:reply('answer',`${who} mora em ${next}`)},
    {text:'ana mora em bruno mora em recife',expected:reply('clarify')},
    {text:'explique livremente o significado da vida',expected:reply('unknown')},
  ]});
}
const output=path.resolve(process.argv[2]??'evals/curricula/compositional-v1.json');
await fs.writeFile(output,JSON.stringify(protocol),{flag:'wx'});
console.log(JSON.stringify({output,lessons:protocol.lessons.length,cases:protocol.cases.length,dialogue_turns:protocol.dialogues.reduce((n,d)=>n+d.turns.length,0)}));
