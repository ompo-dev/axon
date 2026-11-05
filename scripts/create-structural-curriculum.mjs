import fs from 'node:fs/promises';
import path from 'node:path';
const N=n=>['num',n], X=x=>['var',x], A=(op,...args)=>[op,...args];
const D=(f,x)=>A('d',f,X(x)), L=(f,x,a)=>A('lim',f,X(x),N(a));
const utterance=s=>['utterance',...s.split(' ')];
const names=['ana','bruno','caio','dora','eva','fabio','gabi','heitor'];
const cities=['recife','natal','manaus','belem','maceio','palmas','olinda','vitoria'];
const sample=(i,offset=0)=>[X(`u${i+offset}`),N(i*7+offset+2),A('add',X(`u${i+offset}`),N(i+offset+4)),A('mul',N(i+offset+2),X(`u${i+offset}`))][(i+offset)%4];
const lessons=[];
function teach(id,make,guards=[]){
  lessons.push({id,examples:Array.from({length:8},(_,i)=>make(i)),guards,admission:[make(20),make(23)]});
}
const ex=(input,output)=>({input,output});
teach('d-constant',i=>ex(D(N(i*13-20),`x${i}`),N(0)));
teach('d-variable',i=>ex(D(X(`x${i}`),`x${i}`),N(1)));
teach('d-other-variable',i=>ex(D(X(`y${i}`),`x${i}`),N(0)),[{kind:'different',left:[1,1],right:[2,1]}]);
for(const op of ['add','sub'])teach(`d-${op}`,i=>{const f=sample(i),g=sample(i,3),x=`x${i}`;return ex(D(A(op,f,g),x),A(op,D(f,x),D(g,x)));});
teach('d-product',i=>{const f=sample(i),g=sample(i,3),x=`x${i}`;return ex(D(A('mul',f,g),x),A('add',A('mul',D(f,x),g),A('mul',f,D(g,x))));});
teach('d-quotient',i=>{const f=sample(i),g=sample(i,3),x=`x${i}`;return ex(D(A('div',f,g),x),A('div',A('sub',A('mul',D(f,x),g),A('mul',f,D(g,x))),A('pow',g,N(2))));});
teach('d-zero-power',i=>ex(D(A('pow',sample(i),N(0)),`x${i}`),N(0)));
teach('d-power-chain',i=>{const f=sample(i),n=i+2,x=`x${i}`;return ex(D(A('pow',f,N(n)),x),A('mul',A('mul',N(n),A('pow',f,A('sub',N(n),N(1)))),D(f,x)));},[{kind:'integer_range',path:[1,2,1],min:1,max:1000}]);
for(const op of ['sin','exp'])teach(`d-${op}-chain`,i=>{const f=sample(i),x=`x${i}`;return ex(D(A(op,f),x),A('mul',A(op==='sin'?'cos':'exp',f),D(f,x)));});
teach('lim-constant',i=>ex(L(N(i*11+3),`x${i}`,i-5),N(i*11+3)));
teach('lim-variable',i=>ex(L(X(`x${i}`),`x${i}`,i-9),N(i-9)));
teach('lim-other-variable',i=>ex(L(X(`y${i}`),`x${i}`,i-9),X(`y${i}`)),[{kind:'different',left:[1,1],right:[2,1]}]);
for(const op of ['add','sub','mul'])teach(`lim-${op}`,i=>{const f=sample(i),g=sample(i,3),x=`x${i}`,a=i-4;return ex(L(A(op,f,g),x,a),A(op,L(f,x,a),L(g,x,a)));});
teach('lim-power',i=>{const f=sample(i),x=`x${i}`,a=i-8,n=i+1;return ex(L(A('pow',f,N(n)),x,a),A('pow',L(f,x,a),N(n)));},[{kind:'integer_range',path:[1,2,1],min:0,max:1000}]);
for(const op of ['sin','exp'])teach(`lim-${op}`,i=>ex(L(A(op,sample(i)),`x${i}`,i-8),A(op,L(sample(i),`x${i}`,i-8))));
const removable=(x,n,a)=>L(A('div',A('sub',A('pow',X(x),N(n)),A('pow',N(a),N(n))),A('sub',X(x),N(a))),x,a);
teach('lim-removable-powers',i=>ex(removable(`x${i}`,i+2,i-11),A('mul',N(i+2),A('pow',N(i-11),A('sub',N(i+2),N(1))))),[{kind:'integer_range',path:[1,1,1,2,1],min:1,max:1000}]);
for(const sign of [1,-1])teach(`solve-linear-${sign===1?'positive':'negative'}`,i=>ex(A('solve',A('add',A('mul',N(sign*(i+2)),X(`x${i}`)),N(i*11-7)),N(i*5+17),X(`x${i}`)),A('div',A('sub',N(i*5+17),N(i*11-7)),N(sign*(i+2)))),[{kind:'integer_range',path:[1,1,1,1],min:sign===1?1:-1000,max:sign===1?1000:-1}]);
teach('say-city',i=>ex(A('reply',`p${i}`,'mora',`c${i+7}`),A('say',`p${i}`,'mora','em',`c${i+7}`)));
teach('say-capital',i=>ex(A('reply',`p${i}`,'capital',`c${i+7}`),A('say','a','capital','de',`p${i}`,'e',`c${i+7}`)));
teach('tell-city',i=>ex(utterance(`${names[i%8]}${i} mora em ${cities[i%8]}${i+9}`),A('tell',`${names[i%8]}${i}`,'mora',`${cities[i%8]}${i+9}`)));
teach('tell-city-alias',i=>ex(utterance(`${names[i%8]}${i} reside em ${cities[i%8]}${i+9}`),A('tell',`${names[i%8]}${i}`,'mora',`${cities[i%8]}${i+9}`)));
teach('ask-city',i=>ex(utterance(`onde p${i} mora`),A('ask',`p${i}`,'mora')));
teach('ask-city-alias',i=>ex(utterance(`em que cidade p${i} mora`),A('ask',`p${i}`,'mora')));
teach('ask-pronoun',i=>ex(utterance(`onde ele verbo${i}`),A('refer',`verbo${i}`)));
teach('focus-many',i=>ex(utterance(`falamos de p${i} e q${i+11}`),A('focus',`p${i}`,`q${i+11}`)));
teach('retract-city',i=>ex(utterance(`p${i} nao mora em c${i+8}`),A('retract',`p${i}`,'mora',`c${i+8}`)));
teach('tell-capital',i=>ex(utterance(`pais${i} tem capital cidade${i+5}`),A('tell',`pais${i}`,'capital',`cidade${i+5}`)));
teach('ask-capital-path',i=>ex(utterance(`qual a capital do pais de p${i}`),A('path',`p${i}`,'mora','capital')));

function derivative(f,x){
  const [op,a,b]=f;
  if(op==='num')return N(0);if(op==='var')return N(a===x?1:0);
  if(op==='add'||op==='sub')return A(op,derivative(a,x),derivative(b,x));
  if(op==='mul')return A('add',A('mul',derivative(a,x),b),A('mul',a,derivative(b,x)));
  if(op==='div')return A('div',A('sub',A('mul',derivative(a,x),b),A('mul',a,derivative(b,x))),A('pow',b,N(2)));
  if(op==='pow')return b[1]===0?N(0):A('mul',A('mul',b,A('pow',a,A('sub',b,N(1)))),derivative(a,x));
  if(op==='sin'||op==='exp')return A('mul',A(op==='sin'?'cos':'exp',a),derivative(a,x));
  throw Error(`unsupported oracle ${op}`);
}
function limit(f,x,n){const [op,a,b]=f;if(op==='num')return f;if(op==='var')return a===x?N(n):f;return A(op,limit(a,x,n),...(b?[op==='pow'?b:limit(b,x,n)]:[]));}
let state=Number(process.argv[3]??91307531);
if(!Number.isSafeInteger(state)||state<1||state>2147483647)throw Error('Invalid data generator RNG');
const tag=process.argv[3]?`r${state}_`:'';
const rand=n=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)%n;};
function expression(x,depth){
  if(!depth)return rand(3)?X(x):N(rand(23)-11);
  const op=['add','sub','mul','pow','sin','exp'][rand(6)];
  const f=expression(x,depth-1);
  return op==='pow'?A(op,f,N(rand(6))):['sin','exp'].includes(op)?A(op,f):A(op,f,expression(x,depth-1));
}
const cases=[];const add=(family,input,expected,forbid_heads=['d','lim','solve'])=>cases.push({id:`case-${cases.length}`,family,input,expected,forbid_heads});
for(let i=0;i<1200;i++){const x=`${tag}reserved${i}`,f=expression(x,2+rand(3));add('symbolic-derivative',D(f,x),derivative(f,x));}
for(let i=0;i<600;i++){const x=`${tag}quotient${i}`,f=A('div',expression(x,2),A('add',A('pow',X(x),N(2)),N(1+rand(9))));add('quotient-derivative',D(f,x),derivative(f,x));}
for(let i=0;i<1200;i++){const x=`${tag}limit${i}`,f=expression(x,2+rand(2)),a=rand(13)-6;add('continuous-limit',L(f,x,a),limit(f,x,a));}
for(let i=0;i<600;i++){const x=`${tag}hole${i}`,n=2+rand(14),a=rand(25)-12;add('removable-limit',removable(x,n,a),A('mul',N(n),A('pow',N(a),A('sub',N(n),N(1)))));}
for(let i=0;i<600;i++){const x=`${tag}solve${i}`,a=(rand(2)?1:-1)*(1+rand(99)),b=rand(99)-49,c=rand(99)-49;add('linear-equation',A('solve',A('add',A('mul',N(a),X(x)),N(b)),N(c),X(x)),A('div',A('sub',N(c),N(b)),N(a)));}
for(let i=0;i<600;i++)add('language-roles',utterance(`${tag}pessoa${i} reside em ${tag}lugar${i+13}`),A('tell',`${tag}pessoa${i}`,'mora',`${tag}lugar${i+13}`),['utterance']);
for(let i=0;i<100;i++){
  add('unsupported-refusal',D(A('log',X(`${tag}bad${i}`)),`${tag}bad${i}`),null);
  add('singular-refusal',L(A('div',N(1),X(`${tag}bad${i}`)),`${tag}bad${i}`,0),null);
  add('zero-coefficient-refusal',A('solve',A('add',A('mul',N(0),X(`${tag}bad${i}`)),N(i)),N(i+1),X(`${tag}bad${i}`)),null);
}
const reply=(status,text=null)=>({status,text});
const dialogues=[];
for(let i=0;i<100;i++){
  const who=`pessoa${i}`,other=`outra${i}`,country=`pais${i}`,city=`capital${i}`,next=`novo${i}`;
  const learnedAlias=j=>ex(utterance(`aprendiz${j} habita em lugar${j+19}`),A('tell',`aprendiz${j}`,'mora',`lugar${j+19}`));
  dialogues.push({id:`dialogue-${i}`,restart_at:[5,9,19],updates:[{before_turn:17,teaching:{id:'new-habita',examples:Array.from({length:8},(_,j)=>learnedAlias(j+1000)),admission:[learnedAlias(2001),learnedAlias(2002)],guards:[]}}],turns:[
    {text:`onde ${who} mora`,expected:reply('unknown')},
    {text:`${who} reside em ${country}`,expected:reply('recorded')},
    {text:'onde ele mora',expected:reply('answer',`${who} mora em ${country}`)},
    {text:`${country} tem capital ${city}`,expected:reply('recorded')},
    {text:`qual a capital do pais de ${who}`,expected:reply('answer',`a capital de ${country} e ${city}`)},
    {text:`em que cidade ${who} mora`,expected:reply('answer',`${who} mora em ${country}`)},
    {text:`falamos de ${who} e ${other}`,expected:reply('context')},
    {text:'onde ele mora',expected:reply('clarify')},
    {text:`${who} mora em ${next}`,expected:reply('recorded')},
    {text:'onde ele mora',expected:reply('answer',`${who} mora em ${next}`)},
    {text:`${who} nao mora em ${next}`,expected:reply('retracted')},
    {text:`onde ${who} mora`,expected:reply('unknown')},
    {text:`${who} mora em ${country}`,expected:reply('recorded')},
    {text:`${who} mora em ${next}`,source:'other-observer',expected:reply('recorded')},
    {text:`onde ${who} mora`,expected:reply('conflict')},
    {text:`fale livremente sobre ${who}`,expected:reply('unknown')},
    {text:`aprendiz${i} habita em aprendida${i}`,expected:reply('unknown')},
    {text:`aprendiz${i} habita em aprendida${i}`,expected:reply('recorded')},
    {text:`onde aprendiz${i} mora`,expected:reply('answer',`aprendiz${i} mora em aprendida${i}`)},
    {text:'onde ele mora',expected:reply('answer',`aprendiz${i} mora em aprendida${i}`)},
    {text:`em que cidade aprendiz${i} mora`,expected:reply('answer',`aprendiz${i} mora em aprendida${i}`)},
  ]});
}
const output=path.resolve(process.argv[2]??'evals/curricula/structural-v1.json');
await fs.writeFile(output,JSON.stringify({version:1,id:path.basename(output,'.json'),lessons,cases,dialogues}),{flag:'wx'});
console.log(JSON.stringify({output,lessons:lessons.length,cases:cases.length,dialogues:dialogues.length,turns:dialogues.reduce((n,d)=>n+d.turns.length,0)}));
