import fs from 'node:fs';
import crypto from 'node:crypto';
const [protocolPath,predictionsPath,receiptPath]=process.argv.slice(2);
if(!receiptPath)throw new Error('usage: bun scripts/audit-agent.mjs <protocol.json> <agent-predictions.json> <new-receipt.json>');
const protocol=JSON.parse(fs.readFileSync(protocolPath,'utf8'));
const rows=JSON.parse(fs.readFileSync(predictionsPath,'utf8'));
if(!protocol.agent){
  const descriptors={interfaces:Array.from({length:8},(_,i)=>({name:`instrumento${i}`,epoch:3}))};
  const render=t=>Array.isArray(t)?`(${t.slice(1).map(render).join(' ')})`:String(t);
  const errors=[];let correct=0;
  if(rows.length!==protocol.cases.length)throw new Error('missing reserve rows');
  for(let i=0;i<rows.length;i++){
    const c=protocol.cases[i],r=rows[i];if(c.id!==r.id||JSON.stringify(c.input)!==JSON.stringify(r.input))throw new Error('reserve identity');
    let expected=null;try{expected=interpret(render(c.input).slice(1,-1),descriptors);}catch{}
    const actual=r.actual===null?null:BigInt(r.actual);
    if((c.expected===null?null:BigInt(c.expected))!==expected)errors.push(`reserve label mismatch ${c.id}`);
    if(actual===expected)correct++;else errors.push(`reserve prediction mismatch ${c.id}`);
  }
  const receipt={checked:rows.length,correct,errors,scope:'Independent BigInt arithmetic and separate constrained-language parser; frozen epoch 3, no teacher.'};
  fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({...receipt,errors:errors.slice(0,10)}));
  process.exit(errors.length?1:0);
}
const episodes=new Map(protocol.agent.episodes.map(e=>[e.id,e]));
// Independently implemented integer specification. Do not evaluate labels or candidate expressions.
function calculate(name,epoch,a,b){
  const m=/^(instrumento|auxiliar)([0-7])$/.exec(name);if(!m)throw new Error('unknown interface');
  const kind=(Number(m[2])+epoch+(m[1]==='instrumento'?-1:2))%8;
  switch(kind){
    case 0:return a+b;
    case 1:return a*b;
    case 2:return a*a+b;
    case 3:return a*a-2n*a*b+b*b;
    case 4:return a*a+b*b;
    case 5:return a+a+b+b+b;
    case 6:return (a+1n)*b;
    case 7:return b*b+a;
  }
}
function interpret(text,episode){
  const tokens=text.replaceAll('(',' ( ').replaceAll(')',' ) ').trim().split(/\s+/);let i=0;
  function parse(nested=false){
    const v=[];let closed=!nested;
    while(i<tokens.length){const t=tokens[i++];if(t==='(')v.push(parse(true));else if(t===')'){if(!nested)throw new Error('group');closed=true;break;}else v.push(/^-?\d+$/.test(t)?BigInt(t):t);}
    if(!closed)throw new Error('group');
    let name,a,b;
    if(v.length===6&&v[0]==='avalie'&&v[2]==='com'&&v[4]==='e') [name,a,b]=[v[1],v[3],v[5]];
    else if(v.length===6&&v[0]==='use'&&v[2]==='em'&&v[4]==='e') [name,a,b]=[v[1],v[3],v[5]];
    else if(v.length===7&&v[0]==='resultado'&&v[1]==='de'&&v[3]==='para'&&v[5]==='e') [name,a,b]=[v[2],v[4],v[6]];
    else if(v.length===6&&v[0]==='passe'&&v[2]==='e'&&v[4]==='por') [name,a,b]=[v[5],v[1],v[3]];
    else throw new Error('unlearned language');
    if(typeof a!=='bigint'||typeof b!=='bigint')throw new Error('operand');
    const iface=episode.interfaces.find(t=>t.name===name);if(!iface)throw new Error('unauthorized');
    return calculate(name,iface.epoch,a,b);
  }
  return parse();
}
let oracle=0,checked=0,correct=0,children=0;const errors=[],variants={};
for(const e of episodes.values())for(const t of e.interfaces)for(const o of t.observations){
  if(calculate(t.name,t.epoch,...o.input.map(BigInt))!==BigInt(o.output))errors.push(`teacher mismatch ${e.id}/${t.name}`);oracle++;
}
const identities=new Set();
for(const row of rows){
  if(!['active','passive','no-library'].includes(row.variant))throw new Error('unknown control variant');
  const id=JSON.stringify([row.variant,row.episode,row.goal]);if(identities.has(id))throw new Error('duplicate goal');identities.add(id);
  const e=episodes.get(row.episode),g=e?.goals[row.goal];if(!g||row.input!==g.text)throw new Error('goal identity');
  let expected=null;try{expected=interpret(g.text,e);}catch{}
  const actual=row.actual===null?null:BigInt(row.actual),child=row.child_actual===null?null:BigInt(row.child_actual);
  const ok=actual===expected,childOk=child===expected;
  if((g.expected===null?null:BigInt(g.expected))!==expected)errors.push(`label mismatch ${id}`);
  if(row.correct!==ok||row.child_correct!==childOk)errors.push(`incorrect correctness receipt ${id}`);
  const observations=row.events.filter(e=>['observation','refutation'].includes(e.phase));
  if(observations.length!==row.queries)errors.push(`query accounting mismatch ${id}`);
  for(const ev of observations){
    const teacher=e.interfaces.find(t=>t.name===ev.operation&&t.epoch===ev.epoch);
    const o=teacher?.observations.find(o=>JSON.stringify(o.input)===JSON.stringify(ev.input));
    if(!o||o.output!==ev.actual)errors.push(`non-teacher observation ${id}`);
  }
  variants[row.variant]??={correct:0,total:0,queries:0};variants[row.variant].correct+=Number(ok);variants[row.variant].total++;variants[row.variant].queries+=row.queries;
  checked++;correct+=Number(ok);children+=Number(childOk);
}
const expectedRows=protocol.agent.episodes.reduce((n,e)=>n+e.goals.length,0)*3;
if(checked!==expectedRows)errors.push('missing rows');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const receipt={checked,correct,children,teacher_observations_checked:oracle,variants,errors,protocol_sha256:sha(protocolPath),predictions_sha256:sha(predictionsPath),checker_sha256:sha(new URL(import.meta.url)),scope:'Independent integer arithmetic and constrained-language interpreter, not externally authored natural-language evaluation.'};
fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({...receipt,errors:errors.slice(0,10),error_count:errors.length}));if(errors.length||correct!==checked||children!==checked)process.exitCode=1;
