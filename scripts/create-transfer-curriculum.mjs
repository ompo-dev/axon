import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { oracle } from '../ui/curriculum.mjs';
const prior=JSON.parse(await fs.readFile('evals/curricula/mixed-128-v1.json','utf8'));
const stages=[];
const version=process.argv[2]??'transfer-v2';if(!/^[a-z0-9_-]+$/.test(version))throw Error('Invalid fixture id');
const m=(id,arity,oracle,parents=[])=>stages.push({id,family:'hierarchical-math',kind:'composed',required:true,arity,oracle,parents});
for(const id of ['square','double','product','sum']){const s=prior.stages.find(s=>s.id===id);m(id,s.arity,s.oracle);}
m('four-product',2,['x0','x1','mul',2,'mul',2,'mul'],['product','double']);
const discriminant=prior.stages.find(s=>s.id==='discriminant').oracle;
m('discriminant',3,discriminant,['square','four-product']);
m('root-difference',2,['x1','sqrt','x0','sub']);
m('quadratic-numerator',3,[...discriminant,'sqrt','x1','sub'],['discriminant','root-difference']);
m('negative-root-sum',2,[0,'x0','sub','x1','sqrt','sub']);
m('quadratic-negative-numerator',3,[0,'x1','sub',...discriminant,'sqrt','sub'],['discriminant','negative-root-sum']);
for(const id of ['quadratic-plus','quadratic-minus']){const s=prior.stages.find(s=>s.id===id);m(id,s.arity,s.oracle,[id==='quadratic-plus'?'quadratic-numerator':'quadratic-negative-numerator','double']);}
for(const id of ['difference','sum-three']){const s=prior.stages.find(s=>s.id===id);m(id,s.arity,s.oracle,['sum']);}
m('third',1,['x0',3,'div']);
for(const id of ['hypotenuse','norm-three','norm-four','distance-two','mean-three','percentage','fahrenheit-celsius']){
    const s=prior.stages.find(s=>s.id===id),parents=id==='distance-two'?['difference','hypotenuse']:id==='mean-three'?['sum-three','third']:stages.filter(p=>['square','double','sum','hypotenuse','norm-three'].includes(p.id)).map(p=>p.id);m(id,s.arity,s.oracle,parents);
}
const definitions=[
 ['invoice','product',['preco','quantidade']],['change','difference',['pagamento','custo']],
 ['speed','quotient',['distancia','tempo']],['travel','product',['velocidade','tempo']],
 ['rectangle','product',['base','altura']],['triangle','triangle-area',['base','altura']],
 ['norm-two','hypotenuse',['horizontal','vertical']],['norm-three','norm-three',['horizontal','vertical','profundidade'],'rel-norm-two'],
 ['stock','difference',['necessidade','estoque']],['energy','kinetic-energy',['massa','velocidade']],
 ['linear','linear-map',['fator','entrada','deslocamento']],['mean','mean-two',['primeiro','segundo']],
];
for(const [id,math,labels,parent] of definitions){
    const s=prior.stages.find(s=>s.id===math),examples={};let serial=0;
    for(const [split,count] of [['train',12],['admission',12],['test',128],['transfer',128]]){
        const rows=[];
        while(rows.length<count){
            const n=++serial,x=labels.map((_,j)=>(parseInt(createHash('sha256').update(`${id}/${n}/${j}`).digest('hex').slice(0,8),16)%997+1)/(split==='train'?13:split==='admission'?17:split==='test'?19:23));
            const y=oracle(s.oracle,x);if(!Number.isFinite(y))continue;
            const pairs=labels.map((label,j)=>`${label} ${x[j]}`);
            if(n%2)pairs.reverse();
            let input;
            if(split==='train')input=n%3===0?`agora calcule ${id} com ${pairs.join(' e ')}`:n%3===1?`${id} ${pairs.join(' com ')} agora`:`calcule ${id} ${pairs.join(' e ')}`;
            else if(split==='admission')input=`${pairs.join(' e ')} calcule ${id} agora`;
            else if(split==='test')input=`com ${pairs.join(' agora ')} ${id} calcule`;
            else input=`agora ${id} e ${pairs.join(' com ')} calcule agora`;
            rows.push({input,output:y,arguments:x});
        }examples[split]=rows;
    }
    stages.push({id:`rel-${id}`,family:'relational-calculation',kind:'relational',required:true,arity:s.arity,oracle:s.oracle,parents:parent?[parent]:[],examples});
}
const source={schema:'axon-curriculum/v1',id:version,title:'Hierarchical math and learned role binding',splits:{train:12,admission:12,test:128,transfer:128},budget:{...prior.budget,semantic_work_per_stage:20000000},stages};
await fs.writeFile(`evals/curricula/${version}.json`,JSON.stringify(source,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({stages:stages.length,relational:definitions.length}));
