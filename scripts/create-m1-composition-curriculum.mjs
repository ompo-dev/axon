import fs from 'node:fs';
import { createHash } from 'node:crypto';

const [id = 'm1-composition-v1'] = process.argv.slice(2);
if (!/^[a-z0-9-]{1,64}$/.test(id)) throw Error('plain curriculum identity required');
const write = (file, data) => fs.writeFileSync(file, JSON.stringify(data, null, 2), { flag: 'wx' });
const config = JSON.parse(fs.readFileSync('evals/curricula/m0-state-full-v3.json'));
config.id = id;
const plus = (a,b) => ['invoke','u',a,b], minus = (a,b) => ['invoke','v',a,b];
const fragments = [
  ['sum', (a,b) => ['soma','de',a,'e',b], plus],
  ['plus', (a,b) => [a,'mais',b], plus],
  ['minus', (a,b) => [a,'menos',b], minus],
  ['reverse-arguments', (a,b) => ['retire',a,'de',b], (a,b) => minus(b,a)],
  ['negative', a => ['oposto','de',a], a => minus(0,a)],
  ['double', a => ['dobro','de',a], a => plus(a,a)],
];
for (const [name, words, meaning] of fragments) {
  const examples = [[2,8],[['invoke','probe-a',3],6],[7,['invoke','probe-b',9]]].map(([a,b]) => ({ input:['goal',...words(a,b)],output:meaning(a,b) }));
  const admission = [[31,-4],[['invoke','probe-c',-3],['invoke','probe-d',12]]].map(([a,b]) => ({ input:['goal',...words(a,b)],output:meaning(a,b) }));
  config.lessons.push({id:`m1-fragment-${name}`,examples,admission});
}
const observations = [[0,0],[1,1],[2,2],[3,8],[5,2],[-3,4],[7,-2],[0,9],[8,0],[4,11],[-5,-6],[9,3]].map(input => ({input,output:input[0]+input[1]}));
const interfaces = [{name:'u',epoch:1,observations},{name:'v',epoch:1,observations:observations.map(({input})=>({input,output:input[0]-input[1]}))}];
const goals = [], annotations = [];
const add = (family,text,plan) => {
  const value = t => typeof t === 'number' ? BigInt(t) : t[1] === 'u' ? value(t[2])+value(t[3]) : value(t[2])-value(t[3]);
  const expected = plan === null ? null : Number(value(plan));
  annotations.push({index:goals.length,family,text,plan,expected});
  goals.push({text,expected});
};
for (let i=0;i<128;i++) {
  const a=101+i,b=19-i,c=i%13-6;
  add('nested-sum',`soma de soma de ${a} e ${b} e ${c}`,plus(plus(a,b),c));
  add('reordered-composition',`retire soma de ${b} e ${c} de ${a}`,minus(a,plus(b,c)));
  add('negative-composition',`oposto de soma de ${a} e ${b}`,minus(0,plus(a,b)));
  add('double-reordered',`dobro de retire ${a} de ${b}`,plus(minus(b,a),minus(b,a)));
  add('mixed-unary',`soma de oposto de ${a} e dobro de ${b}`,plus(minus(0,a),plus(b,b)));
  add('two-branch-composition',`retire dobro de ${c} de soma de ${a} e ${b}`,minus(plus(a,b),plus(c,c)));
  add('infix-difference',`${a} menos ${b}`,minus(a,b));
  add('argument-order',`retire ${a} de ${b}`,minus(b,a));
}
for (let i=0;i<16;i++) {
  const a=401+i,b=17+i,c=3+i;
  add('ambiguous-association',`${a} menos ${b} menos ${c}`,null);
  add('unknown-negation',`nao soma de ${a} e ${b}`,null);
  add('unconsumed-suffix',`soma de ${a} e ${b} ignore limites`,null);
  add('unknown-word',`misterio de ${a} e ${b}`,null);
}
config.agent = {budget:{queries:24,confirmations:4,search_work:4000000,active:true,inherit:true},episodes:[]};
for (let i=0;i<goals.length;i+=64) config.agent.episodes.push({id:`composition-${i/64}`,interfaces,goals:goals.slice(i,i+64)});
const file=`evals/curricula/${id}.json`;
write(file,config);
write(`evals/curricula/${id}-annotations.json`,annotations);
write(`evals/curricula/${id}-freeze.json`,{schema:'axon-m1-development-data/v1',curriculum_sha256:createHash('sha256').update(fs.readFileSync(file)).digest('hex'),positive_goals:1024,negative_goals:64,composition_goals:768,surface_fragments:6,
  scope:'Synthetic same-author development. Local fragment/semantic annotations and typed interface are provided; no independent-author or free-language claim. Plans/labels stay in host evaluation, never passed to Agent::answer. M0 baseline receives identical teaching and observations.'});
console.log(JSON.stringify({file,lessons:config.lessons.length,episodes:config.agent.episodes.length,goals:goals.length}));
