import fs from 'node:fs';
const [protocolPath,dataPath,csvPath]=process.argv.slice(2);
if(!csvPath)throw new Error('usage: bun scripts/create-algorithm-curriculum.mjs <protocol.json> <algorithm-data.json> <records.csv>');
const p=JSON.parse(fs.readFileSync('evals/curricula/compositional-v1.json','utf8'));
p.id=protocolPath.split(/[\\/]/).at(-1).replace(/\.json$/,'');p.algorithms=dataPath.split(/[\\/]/).at(-1);
const forms=[(op,arg)=>['goal','aplique',op,'em',arg],(op,arg)=>['goal','processe',arg,'usando',op],(op,arg)=>['goal','execute',op,'sobre',arg],(op,arg)=>['goal','quero',op,'para',arg]];
for(let f=0;f<forms.length;f++){
  const rows=Array.from({length:7},(_,j)=>{
    const op=`ferramenta${j}`,arg=j%2?['goal','aplique',`auxiliar${j}`,'em',`arquivo${j}`]:`dados${j}`;
    return {input:forms[f](op,arg),output:['invoke',op,arg]};
  });
  p.lessons.push({id:`algorithm-goal-${f}`,examples:rows.slice(0,5),admission:rows.slice(5)});
}
const text=(f,op,arg)=>[`aplique ${op} em ${arg}`,`processe ${arg} usando ${op}`,`execute ${op} sobre ${arg}`,`quero ${op} para ${arg}`][f%4];
const numbers=(j,n=3+j%8)=>Array.from({length:n},(_,i)=>((j+7)*(i+3)*13+i*i*7)%31-15);
const rows=j=>Array.from({length:3+j%7},(_,i)=>[`item${j}-${i}`,1+(j*3+i)%8,1+(j*5+i*7)%17]);
const palettes=['red','blue','green','gray','black'];
const matrix=j=>Array.from({length:2+j%5},(_,r)=>Array.from({length:3+j%7},(_,c)=>(j+r*3+c*7)%4));
const sum=v=>v.reduce((a,b)=>a+b,0);
const definitions=[
 ['somar','numeric-sum',j=>numbers(j),sum],
 ['quadrados','numeric-sum-squares',j=>numbers(j),v=>sum(v.map(n=>n*n))],
 ['produto','numeric-product',j=>numbers(j,2+j%5).map(n=>n%5),v=>v.reduce((a,b)=>a*b,1)],
 ['maior','numeric-maximum',j=>numbers(j),v=>Math.max(...v)],
 ['menor','numeric-minimum',j=>numbers(j),v=>Math.min(...v)],
 ['contar','numeric-equality-count',j=>[numbers(j),j%7-3],([v,q])=>v.filter(n=>n===q).length],
 ['selecionar','numeric-equality-filter',j=>[numbers(j),j%7-3],([v,q])=>v.filter(n=>n===q)],
 ['somarmaiores','numeric-conditional-sum',j=>[numbers(j),j%7-3],([v,q])=>sum(v.filter(n=>n>q))],
 ['contarmaiores','numeric-conditional-count',j=>[numbers(j),j%7-3],([v,q])=>v.filter(n=>n>q).length],
 ['palavras','text-equality-count',j=>[Array.from({length:3+j%9},(_,i)=>palettes[(i*i+j)%5]),palettes[j%5]],([v,q])=>v.filter(n=>n===q).length],
 ['selecionarpalavras','text-equality-filter',j=>[Array.from({length:3+j%9},(_,i)=>palettes[(i*i+j)%5]),palettes[j%5]],([v,q])=>v.filter(n=>n===q)],
 ['total','record-weighted-reduction',rows,v=>sum(v.map(r=>r[1]*r[2]))],
 ['escolherregistros','record-conditional-filter',j=>[rows(j),5+j%8],([v,q])=>v.filter(r=>r[2]>q)],
 ['relatorio','record-learned-pipeline',j=>[rows(j),5+j%8],([v,q])=>sum(v.filter(r=>r[2]>q).map(r=>r[1]*r[2]))],
 ['cores','visual-count-via-learned-call',j=>[matrix(j),j%4],([grid,q])=>sum(grid.map(row=>row.filter(n=>n===q).length))],
 ['recolorir','visual-conditional-nested-loop',matrix,grid=>grid.map(row=>row.map(n=>n===0?4:n))],
];
const lessons=definitions.map(([id,family,input,compute])=>{
  const all=Array.from({length:9},(_,i)=>{const x=input(10+i);return {input:x,output:compute(x)};});
  return {id,family,examples:all.slice(0,5),admission:all.slice(5)};
});
// Explicit empty cases prevent an arbitrary nonempty initializer being accepted as a reduction.
for(const id of ['somar','quadrados','produto']){const t=lessons.find(t=>t.id===id);t.examples.unshift({input:[],output:id==='produto'?1:0});}
for(const [id,fn] of [['maior',v=>Math.max(...v)],['menor',v=>Math.min(...v)]]){
  const t=lessons.find(t=>t.id===id);for(const x of [[-13,-4,-25],[13,4,25],[7],[-9]])t.examples.push({input:x,output:fn(x)});
}
const requests=[];
for(const [id,family,input,compute] of definitions)for(let j=0;j<100;j++){
  const x=input(1000+j);requests.push({id:`${id}-${j}`,family,text:text(j,id,'dados'),bindings:{dados:x},expected:compute(x)});
}
for(const n of [32,128,512,1024,2048]){
  const x=numbers(794,n);requests.push({id:`large-sum-${n}`,family:'longer-than-training',text:text(n,'somar','dados'),bindings:{dados:x},expected:sum(x)});
  const q=4,pair=[x,q];if(n<2048)requests.push({id:`large-count-${n}`,family:'longer-than-training',text:text(n+1,'palavras','dados'),bindings:{dados:pair},expected:x.filter(v=>v===q).length});
}
for(let j=0;j<100;j++){
  const pair=[numbers(307+j),j%7-3];const selected=pair[0].filter(n=>n===pair[1]);
  requests.push({id:`composition-${j}`,family:'written-learned-composition',text:text(j,'somar',`(${text(j+1,'selecionar','dados')})`),bindings:{dados:pair},expected:sum(selected)});
}
requests.push({id:'unlearned-wording',family:'unsupported-language',text:'descubra tudo sobre os meus arquivos',bindings:{},expected:null});
requests.push({id:'missing-binding',family:'unsupported-binding',text:'aplique somar em segredo',bindings:{},expected:null});
const changeInput=[2,3,4];
const newTeaching={id:'somar',family:'unannounced-rule-change',examples:[{input:[],output:0},...[[2,3],[4,5,2],[3,-2],[6,1,2]].map(input=>({input,output:sum(input)+input.length}))],admission:[[7,2,3],[-3,2,2],[8,3]].map(input=>({input,output:sum(input)+input.length}))};
const changes=[{operation:'somar',input:changeInput,actual:12,teaching:newTeaching,requests:[
 {id:'changed-incremented-sum',family:'unsignalled-change',text:'aplique somar em dados',bindings:{dados:[7,3,2]},expected:15},
 {id:'retained-old-squares',family:'retention-after-change',text:'aplique quadrados em dados',bindings:{dados:[2,3,4]},expected:29},
]}];
const fileRows=Array.from({length:400},(_,i)=>[`pedido${i}`,1+i%7,2+(i*13)%23]);const threshold=12;
const fileDemo={csv:csvPath.split(/[\\/]/).at(-1),text:'aplique relatorio em dados',parameter:threshold,expected:sum(fileRows.filter(r=>r[2]>threshold).map(r=>r[1]*r[2]))};
const c={version:1,lessons,requests,changes,policy_trials:24,file_demo:fileDemo};
for(const [path,data] of [[protocolPath,p],[dataPath,c]])fs.writeFileSync(path,JSON.stringify(data,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(csvPath,'pedido,quantidade,valor\n'+fileRows.map(r=>r.join(',')).join('\n')+'\n',{flag:'wx'});
console.log(JSON.stringify({lessons:lessons.length,requests:requests.length,file_rows:fileRows.length,policy_trials:c.policy_trials}));
