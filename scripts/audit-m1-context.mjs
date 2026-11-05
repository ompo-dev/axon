import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';

const [directory, output, profile='m1-context-v1'] = process.argv.slice(2);
assert(['m1-context-v1','m1-lexical-v2'].includes(profile),'Unknown integration profile');
if (!output) throw Error('usage: bun scripts/audit-m1-context.mjs <campaign.outputs> <new-audit.json>');
const annotations=JSON.parse(await fs.readFile('evals/curricula/m1-context-v1-annotations.json'));
const raw=await fs.readFile(`${directory}/session-predictions.json`), rows=JSON.parse(raw);
assert.equal(rows.length,annotations.length*3);
const equal=isDeepStrictEqual;
const literal=text=>{
  const m=/^(-?\d+) (reais|quilos)$/.exec(text);
  return m ? {value:BigInt(m[1]),unit:m[2]==='reais'?'BRL':'KG',proof:[]} : null;
};
function oracle() {
  const facts=new Map(); let focus=[];
  const read=who=>{
    const sources=[...(facts.get(who)?.values()??[])];
    if (!sources.length) return {issue:'unknown'};
    if (new Set(sources.map(q=>`${q.value}/${q.unit}`)).size>1) return {issue:'conflict'};
    return {...sources[0],proof:sources.flatMap(q=>q.proof)};
  };
  const expression=text=>{
    const q=literal(text);if(q)return q;
    if(text==='o saldo dele')return focus.length===1?read(focus[0]):{issue:'clarify'};
    if(text.startsWith('saldo de '))return read(text.slice('saldo de '.length));
    if(text.startsWith('duplique ')){
      const a=expression(text.slice(9));return a.issue?a:{...a,value:2n*a.value};
    }
    const prefix=text.startsWith('some ')?'some ':text.startsWith('desconte ')?'desconte ':null;
    if(prefix){
      const rest=text.slice(prefix.length), separator=prefix==='some '?' com ':' de ';
      const candidates=[];
      for(let i=rest.indexOf(separator);i>=0;i=rest.indexOf(separator,i+1)){
        const a=expression(rest.slice(0,i)),b=expression(rest.slice(i+separator.length));
        if(a.issue==='unknown'||b.issue==='unknown')continue;
        if(a.issue||b.issue){candidates.push({issue:a.issue??b.issue});continue;}
        if(a.unit!==b.unit){candidates.push({issue:'unit-mismatch'});continue;}
        candidates.push({value:prefix==='some '?a.value+b.value:b.value-a.value,unit:a.unit,proof:[...a.proof,...b.proof]});
      }
      return candidates.length===1?candidates[0]:{issue:candidates.length?'clarify':'unknown'};
    }
    return {issue:'unknown'};
  };
  return (text,source,observed)=>{
    let m=/^defina saldo de (.+) como (-?\d+) (reais|quilos)$/.exec(text);
    if(m){
      const value={value:BigInt(m[2]),unit:m[3]==='reais'?'BRL':'KG',proof:observed.evidence};
      const sources=facts.get(m[1])??new Map();sources.set(source,value);facts.set(m[1],sources);
      return {status:'recorded',value};
    }
    if(text.startsWith('esqueca saldo de ')){
      const sources=facts.get(text.slice('esqueca saldo de '.length));
      const exists=sources?.has(source);if(exists)sources.delete(source);
      return {status:exists?'retracted':'unknown',value:null};
    }
    if(text.startsWith('considere saldo de ')){
      const rest=text.slice('considere saldo de '.length), names=rest.split(' e saldo de ');
      const resolved=names.map(read),issue=resolved.find(q=>q.issue)?.issue;
      if(issue)return {status:issue,value:null};
      focus=[...new Set(names)];return {status:'context',value:null};
    }
    const result=expression(text);
    return result.issue?{status:result.issue,value:null}:{status:'answer',value:result};
  };
}
const targets=new Map(), expected=new Map();
for(const row of rows.filter(r=>r.variant==='context')){
  const evaluate=targets.get(row.session)??oracle();targets.set(row.session,evaluate);
  assert(row.actual,`core error: ${row.session}/${row.turn}: ${row.error}`);
  if(row.actual.status==='recorded'){
    assert.equal(row.episodes_after,row.episodes_before+1);
    assert.deepEqual(row.actual.evidence,[row.episodes_after]);
  }
  const result=evaluate(row.text,row.source,row.actual);
  const value=result.value?{value:Number(result.value.value),unit:result.value.unit}:null;
  if(value)assert(Number.isSafeInteger(value.value));
  const target={status:result.status,value};
  const a=annotations.find(a=>a.session===row.session && a.turn===row.turn);
  assert(a && a.text===row.text && a.source===row.source);
  assert(equal(target,a.expected),`generator/oracle disagreement: ${row.session}/${row.turn}: ${JSON.stringify(target)}`);
  expected.set(`${row.session}/${row.turn}`,{...target,family:a.family});
  if(result.status==='answer'){
    const proof=[...new Set(result.value.proof)].sort((a,b)=>a-b);
    assert(equal(proof,row.actual.evidence),`evidence mismatch: ${row.session}/${row.turn}`);
  }
}
const variants={};
for(const row of rows){
  const target=expected.get(`${row.session}/${row.turn}`);assert(target);
  const actual=row.actual?{status:row.actual.status,value:row.actual.value}:null;
  const passed=equal(actual,{status:target.status,value:target.value});
  assert.equal(row.correct,passed);
  assert.equal(row.expected_status,target.status);assert(equal(row.expected_value,target.value));
  assert(row.work>=0 && row.work<=250000);
  if(['unknown','clarify','conflict','unit-mismatch'].includes(row.actual?.status)){
    assert.equal(row.queries,0);assert.equal(row.episodes_before,row.episodes_after);
  }
  if(row.actual){
    assert.equal(row.actual.work,row.work);
    assert(row.actual.evidence.every(n=>Number.isInteger(n)&&n>0&&n<=row.episodes_after));
  }
  const group=variants[row.variant]??={correct:0,total:0,answers:0,refusals:0,restarts:0,work:0,families:{}};
  group.correct+=Number(passed);group.total++;group.work+=row.work;group.restarts+=Number(row.restarted);
  group.answers+=Number(passed&&target.status==='answer');
  group.refusals+=Number(passed&&['unknown','clarify','conflict','unit-mismatch'].includes(target.status));
  const f=group.families[target.family]??={correct:0,total:0};f.correct+=Number(passed);f.total++;
}
assert.equal(variants.context.correct,1536);assert.equal(variants.context.total,1536);
assert.equal(variants.context.restarts,96);
assert(variants.context.correct>variants['no-memory'].correct && variants.context.correct>variants['no-composition'].correct);
const result=JSON.parse(await fs.readFile(`${directory}/structural-result.json`));
assert(result.admitted);
assert.equal(result.protocol,profile);
assert.deepEqual([result.correct,result.dialogue_correct,result.algorithm_correct,result.agent_correct,result.session_correct,result.rules],[7400,3100,profile==='m1-lexical-v2'?4821:4693,1088,1536,126]);
const compiled=await fs.readFile(`${directory}/structural.axstruct`,'utf8');
assert(!compiled.includes('ana maria-')&&!compiled.includes('bruno costa-'));
const report={schema:'axon-m1-context-development-audit/v1',variants,independent_ledger_arithmetic:true,personal_names_absent_from_compiled_seed:true,predictions_sha256:createHash('sha256').update(raw).digest('hex'),m1_complete:false,
  scope:'Independent BigInt/ledger reference over the authored surface grammar; exact quantity, reference, source and refusal checks. Same-author synthetic development, not independent authorship or general conversation. Source labels are trusted caller metadata, not authenticated user identities.'};
await fs.writeFile(output,JSON.stringify(report,null,2),{flag:'wx'});
console.log(JSON.stringify({variants,m1_complete:false}));
