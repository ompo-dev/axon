import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

const destination=process.argv[2];
if(!destination)throw new Error('usage: bun scripts/create-agent-curriculum.mjs <new.json> [offset]');
const offset=Number(process.argv[3]??0);
const order=process.argv[4]??'';
if(order&&!/^[a-z0-9-]{1,32}$/.test(order))throw new Error('order id');
if(!Number.isSafeInteger(offset)||offset<0)throw new Error('offset');
const protocol=JSON.parse(fs.readFileSync(path.resolve('evals/curricula/compositional-v1.json'),'utf8'));
protocol.id=`agent-v1-${offset}${order?`-${order}`:''}`;
const frame=(i,n,a,b)=>[
  ['goal','avalie',n,'com',a,'e',b],
  ['goal','use',n,'em',a,'e',b],
  ['goal','resultado','de',n,'para',a,'e',b],
  ['goal','passe',a,'e',b,'por',n],
][i];
const grouped=(j)=>j%2 ? ['goal','use',`inner${j}`,'em',11+j,'e',19+j] : ['goal','resultado','de',`inner${j}`,'para',5+j,'e',17+j];
for(let f=0;f<4;f++)for(let shape=0;shape<4;shape++){
  const examples=Array.from({length:6},(_,j)=>{
    const a=shape&1?grouped(j):3+j*2;
    const b=shape&2?grouped(j+9):13+j*3;
    return {input:frame(f,`tool${j}`,a,b),output:['invoke',`tool${j}`,a,b]};
  });
  protocol.lessons.push({id:`goal-frame-${f}-${shape}`,examples:examples.slice(0,4),admission:examples.slice(4)});
}
const functions=[(a,b)=>a+b,(a,b)=>a*b,(a,b)=>a*a+b,(a,b)=>(a-b)**2,(a,b)=>a*a+b*b,(a,b)=>2*a+3*b,(a,b)=>a*b+b,(a,b)=>a+b*b];
const pairs=[[0,0],[1,1],[2,2],[3,8],[5,2],[-3,4],[7,-2],[0,9],[8,0],[4,11],[-5,-6],[9,3],[2,-7],[11,5],[6,-4],[13,2],[3,17],[19,-3],[10,7],[23,4],[12,15],[-11,9],[21,6],[14,-8]];
const text=(i,n,a,b)=>[
  `avalie ${n} com ${a} e ${b}`,
  `use ${n} em ${a} e ${b}`,
  `resultado de ${n} para ${a} e ${b}`,
  `passe ${a} e ${b} por ${n}`,
][i%4];
const episodes=[];
for(let epoch=1;epoch<=3;epoch++)for(let task=0;task<8;task++){
  const fn=functions[(task+epoch-1)%functions.length],other=functions[(task+epoch+2)%functions.length];
  const name=`instrumento${task}`,aux=`auxiliar${task}`;
  const interfaces=[[name,fn],[aux,other]].map(([name,fn])=>({name,epoch,observations:pairs.map(input=>({input,output:fn(...input)}))}));
  if(order)for(const teacher of interfaces){
    const key=o=>createHash('sha256').update(JSON.stringify([order,teacher.name,epoch,o.input])).digest('hex');
    teacher.observations.sort((a,b)=>key(a).localeCompare(key(b)));
  }
  const goals=[];
  for(let i=0;i<20;i++){
    const a=31+offset+3*i+task,b=17+2*i+epoch,c=5+i;
    goals.push({text:text(i,name,a,b),expected:fn(a,b)});
    goals.push({text:text(i+1,name,`(${text(i+2,aux,a,b)})`,c),expected:fn(other(a,b),c)});
    goals.push({text:text(i+2,name,`(${text(i,aux,a,c)})`,`(${text(i+1,aux,b,c)})`),expected:fn(other(a,c),other(b,c))});
  }
  // Unlearned wording and unauthorized interfaces must refuse before any experiment.
  goals.push({text:`invente uma resposta sobre ${name}`,expected:null});
  goals.push({text:text(0,'desconhecido',39,51),expected:null});
  episodes.push({id:`epoch${epoch}-task${task}`,interfaces,goals});
}
protocol.agent={budget:{queries:24,confirmations:4,search_work:4000000,active:true,inherit:true},episodes};
fs.writeFileSync(destination,JSON.stringify(protocol,null,2)+'\n',{flag:'wx'});
process.stdout.write(JSON.stringify({destination,lessons:protocol.lessons.length,episodes:episodes.length,goals:episodes.reduce((n,e)=>n+e.goals.length,0),offset})+'\n');
