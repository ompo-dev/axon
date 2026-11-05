// Reproducible initial fixture only. Edit the resulting JSON for subsequent curricula.
import fs from 'node:fs/promises';
import path from 'node:path';
const x=n=>[`x${n}`],c=n=>[n],op=(name,...args)=>[...args.flat(),name];
const add=(a,b)=>op('add',a,b),sub=(a,b)=>op('sub',a,b),mul=(a,b)=>op('mul',a,b),div=(a,b)=>op('div',a,b),sqrt=a=>op('sqrt',a),sq=a=>mul(a,a),abs=a=>sqrt(sq(a));
const sum=(...args)=>args.reduce(add),norm=(...args)=>sqrt(sum(...args.map(sq)));
const disc=sub(sq(x(1)),mul(c(4),mul(x(0),x(2))));
const minimum=(a,b)=>div(sub(add(a,b),abs(sub(a,b))),c(2));
const maximum=(a,b)=>div(add(add(a,b),abs(sub(a,b))),c(2));
const math=[];
const m=(id,family,arity,oracle,parents=[],required=true)=>math.push({id,family,kind:'numeric',required,arity,oracle,parents});
m('sum','arithmetic',2,add(x(0),x(1)));
m('difference','arithmetic',2,sub(x(0),x(1)));
m('product','arithmetic',2,mul(x(0),x(1)));
m('quotient','arithmetic',2,div(x(0),x(1)));
m('square','powers',1,sq(x(0)));
m('double','affine',1,mul(x(0),c(2)));
m('triple','affine',1,mul(x(0),c(3)),['double']);
m('half','affine',1,div(x(0),c(2)));
m('negative','affine',1,sub(c(0),x(0)));
m('absolute','geometry',1,abs(x(0)),['square']);
m('sum-three','arithmetic',3,sum(x(0),x(1),x(2)),['sum']);
m('sum-four','arithmetic',4,sum(x(0),x(1),x(2),x(3)),['sum']);
m('mean-two','statistics',2,div(add(x(0),x(1)),c(2)),['sum','half']);
m('mean-three','statistics',3,div(sum(x(0),x(1),x(2)),c(3)),['sum-three','triple']);
m('mean-four','statistics',4,div(sum(x(0),x(1),x(2),x(3)),c(4)),['sum-four','half']);
m('difference-squares','polynomial',2,sub(sq(x(0)),sq(x(1))),['square','difference']);
m('product-sum-difference','polynomial',4,mul(add(x(0),x(1)),sub(x(2),x(3))),['sum','difference']);
m('square-sum','polynomial',2,sq(add(x(0),x(1))),['square','sum']);
m('square-difference','polynomial',2,sq(sub(x(0),x(1))),['square','difference']);
m('sum-squares','polynomial',2,add(sq(x(0)),sq(x(1))),['square']);
m('hypotenuse','geometry',2,norm(x(0),x(1)),['square','sum-squares']);
m('norm-three','geometry',3,norm(x(0),x(1),x(2)),['hypotenuse','square']);
m('norm-four','geometry',4,norm(x(0),x(1),x(2),x(3)),['hypotenuse']);
m('distance-two','geometry',4,norm(sub(x(0),x(2)),sub(x(1),x(3))),['hypotenuse','difference']);
m('ratio-norms','rational',4,div(norm(x(0),x(1)),norm(x(2),x(3))),['hypotenuse','quotient']);
m('linear-map','affine',3,add(mul(x(0),x(1)),x(2)),['sum','product']);
m('dot-two','linear-algebra',4,add(mul(x(0),x(1)),mul(x(2),x(3))),['sum','product']);
m('quadratic-value','polynomial',4,sum(mul(x(1),sq(x(0))),mul(x(2),x(0)),x(3)),['square','linear-map']);
m('discriminant','algebra',3,disc,['square','product']);
m('quadratic-plus','algebra',3,div(add(sub(c(0),x(1)),sqrt(disc)),mul(c(2),x(0))),['discriminant','negative'],true);
m('quadratic-minus','algebra',3,div(sub(sub(c(0),x(1)),sqrt(disc)),mul(c(2),x(0))),['discriminant','negative'],true);
m('geometric-mean','statistics',2,sqrt(mul(x(0),x(1))),['product']);
m('harmonic-mean','statistics',2,div(mul(c(2),mul(x(0),x(1))),add(x(0),x(1))),['product','sum']);
m('reciprocal','rational',1,div(c(1),x(0)));
m('cube','powers',1,mul(sq(x(0)),x(0)),['square']);
m('fourth-power','powers',1,sq(sq(x(0))),['square']);
m('relative-change','rational',2,div(sub(x(1),x(0)),x(0)),['difference','quotient']);
m('percentage','affine',1,mul(x(0),c(100)),['double']);
m('rectangle-perimeter','geometry',2,mul(c(2),add(x(0),x(1))),['double','sum']);
m('triangle-area','geometry',2,div(mul(x(0),x(1)),c(2)),['product','half']);
m('box-volume','geometry',3,mul(mul(x(0),x(1)),x(2)),['product']);
m('kinetic-energy','physics',2,div(mul(x(0),sq(x(1))),c(2)),['square','half']);
m('motion-displacement','physics',3,add(mul(x(0),x(1)),div(mul(x(2),sq(x(1))),c(2))),['kinetic-energy','product']);
m('weighted-mean','statistics',4,div(add(mul(x(0),x(1)),mul(x(2),x(3))),add(x(1),x(3))),['dot-two','sum']);
m('minimum-two','ordering',2,minimum(x(0),x(1)),['absolute','mean-two']);
m('maximum-two','ordering',2,maximum(x(0),x(1)),['absolute','mean-two']);
m('positive-part','ordering',1,div(add(x(0),abs(x(0))),c(2)),['absolute','half']);
m('interpolation','affine',3,add(x(0),mul(x(2),sub(x(1),x(0)))),['difference','linear-map']);
m('ratio-products','rational',4,div(mul(x(0),x(1)),mul(x(2),x(3))),['product','quotient']);
m('difference-products','linear-algebra',4,sub(mul(x(0),x(1)),mul(x(2),x(3))),['product','difference']);
m('inverse-square','rational',1,div(c(1),sq(x(0))),['square','reciprocal']);
m('balanced-ratio','rational',2,div(sub(x(0),x(1)),add(x(0),x(1))),['sum','difference']);
m('cross-ratio','rational',4,div(mul(sub(x(0),x(1)),sub(x(2),x(3))),mul(sub(x(0),x(3)),sub(x(2),x(1)))),['difference','ratio-products']);
m('distance-one','geometry',2,abs(sub(x(0),x(1))),['absolute','difference']);
m('soft-sign','rational',1,div(x(0),add(c(1),abs(x(0)))),['absolute']);
m('sine-control','transcendental',1,op('sin',x(0)),[],false);
m('exponential-control','transcendental',1,op('exp',x(0)),[],false);
m('maximum-three','ordering',3,maximum(maximum(x(0),x(1)),x(2)),['maximum-two']);
m('energy-change','physics',3,div(mul(x(0),sub(sq(x(2)),sq(x(1)))),c(2)),['difference-squares','half']);
m('root-quadratic','algebra',3,sqrt(add(mul(x(1),sq(x(0))),x(2))),['square','linear-map']);
m('cosine-vectors','linear-algebra',4,div(add(mul(x(0),x(1)),mul(x(2),x(3))),mul(norm(x(0),x(2)),norm(x(1),x(3)))),['hypotenuse','dot-two']);
m('cubic-value','polynomial',4,sum(mul(x(1),mul(sq(x(0)),x(0))),mul(x(2),x(0)),x(3)),['cube','linear-map']);
m('fahrenheit-celsius','affine',1,div(mul(sub(x(0),c(32)),c(5)),c(9)),['difference','quotient']);
m('resistor-parallel','physics',2,div(mul(x(0),x(1)),add(x(0),x(1))),['product','sum']);
if(math.length!==64)throw Error(`Expected 64 math stages, got ${math.length}`);
const texts=[];
const t=(id,family,input,output,transfer_input)=>texts.push({id,family,kind:'text',required:true,input,output,...(transfer_input?{transfer_input}:{})});
t('person-first','extraction','{first} {last}','{first}');
t('person-last','extraction','{first} {last}','{last}');
t('person-reverse','formatting','{first} {last}','{last}, {first}');
t('person-upper','case','{first} {last}','{first|upper} {last|upper}');
t('person-lower','case','{first|upper} {last|upper}','{first|lower} {last|lower}');
t('customer-name','records','nome={first} {last};cidade={city}','{first} {last}');
t('customer-city','records','nome={first} {last};cidade={city}','{city}','cidade={city};nome={first} {last}');
t('customer-label','formatting','nome={first};sobrenome={last}','{last} / {first}');
t('email-user','extraction','{user}@{domain}','{user}');
t('email-domain','extraction','{user}@{domain}','{domain}');
t('email-lower','case','{user|upper}@{domain|upper}','{user|lower}@{domain|lower}');
t('email-label','formatting','contato:{user}@{domain}','E-mail: {user}@{domain}');
t('file-extension','files','{filename}.{ext}','{ext}');
t('file-stem','files','{filename}.{ext}','{filename}');
t('file-path','files','pasta/{filename}.{ext}','{filename}.{ext}');
t('file-rename','files','{filename}.{ext}','backup-{filename}.{ext}');
t('ticket-code','records','ticket={code};status={status}','{code}');
t('ticket-state','records','ticket={code};status={status}','{status}');
t('ticket-label','formatting','{code}|{status}','[{code}] {status}');
t('order-code','records','pedido:{code} cliente:{first}','{code}');
t('order-owner','records','pedido:{code} cliente:{first}','{first}');
t('phone-extract','extraction','telefone={phone};nome={first}','{phone}');
t('phone-label','formatting','{first};{phone}','{first}: {phone}');
t('date-reorder','dates','{year}-{month}-{day}','{day}/{month}/{year}');
t('date-year','dates','{year}-{month}-{day}','{year}');
t('date-month','dates','{year}-{month}-{day}','{month}');
t('date-day','dates','{year}-{month}-{day}','{day}');
t('date-compact','dates','{day}/{month}/{year}','{year}{month}{day}');
t('city-code','records','{city} [{code}]','{city}');
t('city-upper','case','cidade={city}','{city|upper}');
t('payment-amount','records','pago={amount};ref={code}','{amount}');
t('payment-reference','records','pago={amount};ref={code}','{code}');
t('payment-label','formatting','{code}|{amount}','REF {code}: {amount}');
t('product-label','formatting','{filename};{code}','{code} - {filename}');
t('log-state','records','registro[{code}] resultado={status}','{status}');
t('log-reference','records','registro[{code}] resultado={status}','{code}');
t('csv-swap','formatting','{first},{city},{code}','{code},{first},{city}');
t('html-content','extraction','<nome>{first}</nome>','{first}');
t('initials','formatting','{first} {last}','{first}.{last}');
texts.push({id:'free-causal-reading',family:'free-reading',kind:'text',required:false,examples:{
  train:[['A rua molhou porque choveu.','choveu'],['A sala clareou porque acenderam a luz.','acenderam a luz'],['A planta morreu porque faltou agua.','faltou agua'],['O vidro quebrou porque caiu.','caiu']].map(([input,output])=>({input,output})),
  admission:[['Sem combustivel, o carro parou.','faltou combustivel'],['A prova foi adiada devido a chuva.','choveu'],['O gelo virou agua ao receber calor.','recebeu calor'],['A porta abriu quando Lia girou a chave.','Lia girou a chave']].map(([input,output])=>({input,output})),
  test:[['Bia nao levou casaco e sentiu frio.','nao levou casaco'],['O alarme tocou depois que entrou fumaca.','entrou fumaca'],['O time venceu por marcar mais gols.','marcou mais gols'],['Faltou energia e a maquina desligou.','faltou energia']].map(([input,output])=>({input,output})),
  transfer:[['O passeio acabou cedo: uma tempestade chegou.','chegou uma tempestade'],['O arquivo sumiu pois foi apagado.','foi apagado'],['A comida esfriou enquanto esperavam.','esperaram'],['O cano rompeu pelo excesso de pressao.','excesso de pressao']].map(([input,output])=>({input,output}))
}});
if(texts.length!==40)throw Error(`Expected 40 text stages, got ${texts.length}`);
const wordDefinitions=[
 ['written-sum','sum','some {x0} com {x1}','qual o total de {x0} e {x1}'],
 ['written-difference','difference','subtraia {x1} de {x0}','quanto sobra de {x0} retirando {x1}'],
 ['written-product','product','multiplique {x0} por {x1}','produto entre {x0} e {x1}'],
 ['written-quotient','quotient','divida {x0} por {x1}','quociente de {x0} e {x1}'],
 ['written-square','square','calcule quadrado de {x0}','quanto vale {x0} elevado ao quadrado'],
 ['written-double','double','calcule dobro de {x0}','duplique o valor {x0}'],
 ['written-half','half','calcule metade de {x0}','qual metade do numero {x0}'],
 ['written-absolute','absolute','calcule modulo de {x0}','distancia de {x0} ate zero'],
 ['written-mean','mean-two','media entre {x0} e {x1}','valor medio de {x0} com {x1}'],
 ['rectangle-area','product','retangulo base {x0} altura {x1}','qual area de retangulo com lados {x0} e {x1}'],
 ['triangle-area-text','triangle-area','triangulo base {x0} altura {x1}','area triangular com base {x0} e altura {x1}'],
 ['rectangle-perimeter-text','rectangle-perimeter','perimetro retangulo {x0} por {x1}','soma dos lados de retangulo {x0} por {x1}'],
 ['stock-needed','difference','necessidade {x0} estoque {x1}','temos {x1} mas precisamos de {x0}'],
 ['invoice-total','product','preco {x0} quantidade {x1}','custo de {x1} unidades a {x0} cada'],
 ['invoice-change','difference','pagamento {x0} custo {x1}','troco pagando {x0} por custo {x1}'],
 ['distance-travelled','product','velocidade {x0} tempo {x1}','distancia em {x1} horas a {x0} por hora'],
 ['average-speed','quotient','distancia {x0} tempo {x1}','velocidade para {x0} em {x1} horas'],
 ['ohm-voltage','product','resistencia {x0} corrente {x1}','tensao para corrente {x1} e resistencia {x0}'],
 ['box-volume-text','box-volume','caixa largura {x0} altura {x1} profundidade {x2}','volume de caixa com medidas {x0} {x1} {x2}'],
 ['hypotenuse-text','hypotenuse','catetos {x0} e {x1}','qual hipotenusa com catetos {x0} e {x1}'],
 ['linear-text','linear-map','produto {x0} por {x1} acrescido {x2}','multiplique {x1} por {x0} e adicione {x2}'],
 ['sum-three-text','sum-three','some {x0} mais {x1} mais {x2}','total entre tres valores {x0} {x1} {x2}'],
 ['geometric-mean-text','geometric-mean','media geometrica {x0} e {x1}','raiz do produto de {x0} por {x1}'],
 ['energy-text','kinetic-energy','massa {x0} velocidade {x1}','energia cinetica com velocidade {x1} e massa {x0}']
];
const words=wordDefinitions.map(([id,parent,input,transfer_input])=>{const s=math.find(s=>s.id===parent);return {id,family:'written-calculation',kind:'word_math',required:true,arity:s.arity,oracle:s.oracle,input,transfer_input};});
const alpha=n=>{let s='';do{s=String.fromCharCode(97+n%26)+s;n=Math.floor(n/26)-1;}while(n>=0);return s;};
const fields={};
const first=['Ana','Bruno','Carolina','Diego','Elisa','Fabio','Gabriela','Hugo','Iara','Jonas','Karina','Luis','Marina','Natan','Olivia','Paulo','Rita','Sonia','Tales','Vera'];
for(const key of ['first','last','city','user','domain','filename','ext','code','status','phone','amount','day','month','year']){
  fields[key]={};
  for(const [split,offset] of [['train',0],['admission',1000],['test',2000],['transfer',4000]]){
    fields[key][split]=Array.from({length:256},(_,i)=>{
      const n=offset+i,suffix=alpha(n);
      if(key==='first')return first[i%first.length]+suffix;
      if(key==='last')return ['Silva','Costa','Almeida','Fernandes','Rocha'][i%5]+suffix;
      if(key==='city')return ['Recife','Salvador','Natal','Curitiba','Manaus'][i%5]+suffix;
      if(key==='user')return `usuario${suffix}`;
      if(key==='domain')return `empresa${suffix}.com`;
      if(key==='filename')return `relatorio${suffix}`;
      if(key==='ext')return `ext${suffix}`;
      if(key==='code')return `AX${n+10}`;
      if(key==='status')return `estado${suffix}`;
      if(key==='phone')return String(11900000000+n);
      if(key==='amount')return String(10000+n);
      // Deliberately variable-width fields prevent fixed-position memorization.
      return String((key==='year'?2000:key==='month'?1:10)+n);
    });
  }
}
const source={schema:'axon-curriculum/v1',id:'mixed-128-v1',title:'128 etapas: matematica, texto e problemas escritos',splits:{train:12,admission:12,test:128,transfer:32},budget:{candidates_per_stage:5000,total_candidates:400000,work_per_stage:8000000,total_numeric_work:640000000,text_work_per_stage:2000000,semantic_work_per_stage:2000000},stages:[...math,...texts,...words],fields};
if(source.stages.length!==128)throw Error('Wrong stage count');
const output=path.resolve(process.argv[2]??'evals/curricula/mixed-128-v1.json');
await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(source,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({file:output,stages:source.stages.length,numeric:math.length,text:texts.length,word_math:words.length}));
