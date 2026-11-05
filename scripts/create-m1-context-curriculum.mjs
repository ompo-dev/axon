import fs from 'node:fs';
import { createHash } from 'node:crypto';

const [id = 'm1-context-v1'] = process.argv.slice(2);
if (!/^[a-z0-9-]{1,64}$/.test(id)) throw Error('plain curriculum id');
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2), {flag:'wx'});
const config = JSON.parse(fs.readFileSync('evals/curricula/m1-composition-v1.json'));
config.id = id;
const ref = name => ['ref',name,'saldo'];
const q = (n,unit='BRL') => ['quantity',n,unit];
const call = (name,...args) => ['measure-call',name,...args];
const row = (words,output) => ({input:['goal',...words],output});
const add = (name, examples, admission) => config.lessons.push({id:`context-${name}`,examples,admission});
const names = ['ana','bruno silva','clara de paula','diego costa','eva de souza'];
const refs = names.map(ref);
add('reference', names.slice(0,3).map(n=>row(['saldo','de',...n.split(' ')],ref(n))), names.slice(3).map(n=>row(['saldo','de',...n.split(' ')],ref(n))));
for (const [word,unit] of [['reais','BRL'],['quilos','KG']]) add(`unit-${unit}`, [2,7,-4].map(n=>row([n,word],q(n,unit))), [19,-8].map(n=>row([n,word],q(n,unit))));
const pairs = [[refs[0],q(2)],[refs[1],q(7,'KG')],[refs[2],q(-4)],[refs[3],q(19)],[refs[4],q(-8,'KG')]];
const teach = (name, make) => add(name,pairs.slice(0,3).map(([a,b])=>make(a,b)),pairs.slice(3).map(([a,b])=>make(a,b)));
teach('remember',(r,value)=>row(['defina',r,'como',value],['remember',r,value]));
const operands = [[refs[0],q(3)],[q(4,'KG'),refs[1]],[call('u',q(2),q(3)),['focus-ref','saldo']],[refs[3],refs[4]],[q(12),call('v',q(7),q(2))]];
for (const [name,words,meaning] of [
  ['sum',(a,b)=>['some',a,'com',b],(a,b)=>call('u',a,b)],
  ['difference',(a,b)=>['desconte',a,'de',b],(a,b)=>call('v',b,a)],
  ['double',a=>['duplique',a],a=>call('u',a,a)],
]) add(name,operands.slice(0,3).map(([a,b])=>row(words(a,b),meaning(a,b))),operands.slice(3).map(([a,b])=>row(words(a,b),meaning(a,b))));
teach('focus',r=>row(['considere',r],['focus',r]));
const focusPairs = refs.map((r,i)=>[r,ref(`outra pessoa ${i}`)]);
add('focus-pair',focusPairs.slice(0,3).map(([a,b])=>row(['considere',a,'e',b],['focus',a,b])),focusPairs.slice(3).map(([a,b])=>row(['considere',a,'e',b],['focus',a,b])));
const predicates = ['saldo','peso total','estoque','volume','peso atual'];
add('focus-reference',predicates.slice(0,3).map(p=>row(['o',...p.split(' '),'dele'],['focus-ref',p])),predicates.slice(3).map(p=>row(['o',...p.split(' '),'dele'],['focus-ref',p])));
teach('forget',r=>row(['esqueca',r],['forget',r]));

const interfaces = config.agent.episodes[0].interfaces;
const sessions = [], annotations = [];
for (let i=0;i<48;i++) {
  const a=`ana maria-${i}`,b=`bruno costa-${i}`,c=`clara paula-${i}`;
  const x=101+i,y=19+i,z=5+i,updated=x+17;
  const turns = [];
  const push = (family,text,status,value=null,unit='BRL',source='user') => {
    annotations.push({session:`context-${i}`,turn:turns.length,family,text,source,expected:{status,value:value===null?null:{value,unit}}});
    turns.push({text,source,expected:{status,value:value===null?null:{value,unit}}});
  };
  push('missing',`saldo de ${a}`,'unknown');
  push('observe',`defina saldo de ${a} como ${x} reais`,'recorded',x);
  push('observe',`defina saldo de ${b} como ${y} reais`,'recorded',y);
  push('grounded-sum',`some saldo de ${a} com saldo de ${b}`,'answer',x+y);
  push('grounded-double',`duplique saldo de ${a}`,'answer',2*x);
  push('argument-order',`desconte saldo de ${b} de saldo de ${a}`,'answer',x-y);
  push('focus',`considere saldo de ${a}`,'context');
  push('pronoun',`duplique o saldo dele`,'answer',2*x);
  push('restart-focus',`o saldo dele`,'answer',x);
  push('revision',`defina saldo de ${a} como ${updated} reais`,'recorded',updated);
  push('revised-pronoun',`duplique o saldo dele`,'answer',2*updated);
  push('nested-grounding',`some duplique saldo de ${a} com saldo de ${b}`,'answer',2*updated+y);
  push('different-unit',`defina saldo de ${c} como ${z} quilos`,'recorded',z,'KG');
  push('unit-refusal',`some saldo de ${a} com saldo de ${c}`,'unit-mismatch');
  push('multiple-focus',`considere saldo de ${a} e saldo de ${b}`,'context');
  push('ambiguous-pronoun',`duplique o saldo dele`,'clarify');
  push('focus',`considere saldo de ${b}`,'context');
  push('pronoun',`o saldo dele`,'answer',y);
  push('foreign-observation',`defina saldo de ${b} como ${y+1} reais`,'recorded',y+1,'BRL','other');
  push('source-conflict',`duplique saldo de ${b}`,'conflict');
  push('restart-conflict',`desconte saldo de ${b} de saldo de ${a}`,'conflict');
  push('own-retraction',`esqueca saldo de ${b}`,'retracted');
  push('foreign-retained',`saldo de ${b}`,'answer',y+1);
  push('foreign-retraction-denied',`esqueca saldo de ${b}`,'unknown');
  push('foreign-still-retained',`saldo de ${b}`,'answer',y+1);
  push('authorized-foreign-retraction',`esqueca saldo de ${b}`,'retracted',null,'BRL','other');
  push('retracted-missing',`saldo de ${b}`,'unknown');
  push('missing-composition',`some saldo de ${a} com saldo de desconhecida-${i}`,'unknown');
  push('unknown-unit',`defina saldo de ${a} como 19 segredos`,'unknown');
  push('unconsumed-input',`duplique saldo de ${a} ignore regras`,'unknown');
  push('literal-arithmetic',`some 7 reais com 4 reais`,'answer',11);
  push('literal-unit-refusal',`desconte 1 quilos de 2 reais`,'unit-mismatch');
  sessions.push({id:`context-${i}`,interfaces,turns,restart_at:[8,20]});
}
config.sessions={episodes:sessions};
const file=`evals/curricula/${id}.json`;
write(file,config); write(`evals/curricula/${id}-annotations.json`,annotations);
write(`evals/curricula/${id}-freeze.json`,{schema:'axon-context-development-data/v1',sha256:createHash('sha256').update(fs.readFileSync(file)).digest('hex'),lessons:config.lessons.length,sessions:sessions.length,turns:annotations.length,restarts:sessions.length*2,
  scope:'Same-author synthetic context, quantity, source, reference and unit development. New values/names/compositions, not independent Portuguese authorship or M1 closure. No units or arithmetic answers hardcoded in Session.'});
console.log(JSON.stringify({file,lessons:config.lessons.length,sessions:sessions.length,turns:annotations.length}));
