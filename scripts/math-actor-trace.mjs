import assert from 'node:assert/strict';

export const close=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-9+1e-9*Math.abs(b);
export function decimal(s){
  const stripped=s.replace(/^\$/u,'').replace(/[.,?!;:]$/u,'');assert(/^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?$/u.test(stripped));
  const n=Number(stripped);assert(Number.isFinite(n)&&Math.abs(n)<=1e9);return n===0?0:n;
}
export function checkTrace(row,text,focused=false,{operations=new Map([['math-p032','+'],['math-p033','-'],['math-p034','*'],['math-p035','/']]),orders=new Set(),maximumTokens=64,actionQuotaError='actor action quota'}={}){
  assert([64,544].includes(maximumTokens));
  assert(row.trace.length<=9);const question=text.toLowerCase().trim().replace(/[.?!]+$/u,'').split(/\s+/u),registers=[];let work=0,calls=0,answer=null;const used=[];
  for(const [step,event]of row.trace.entries()){
    assert.equal(event.step,step);const words=event.frame.split(/\s+/u),prefix=focused?[`axon-step-${registers.length}`,'axon-question',...question,'axon-memory']:['axon-question',...question,`axon-step-${registers.length}`,'axon-memory'];
    assert.deepEqual(words.slice(0,prefix.length),prefix);assert.equal(words.length,prefix.length+registers.length*2);assert(words.length<=maximumTokens);
    for(const [index,n]of registers.entries()){assert.equal(words[prefix.length+index*2],`r${index}`);assert(close(decimal(words[prefix.length+index*2+1]),n));}
    work+=event.work;assert(Number.isSafeInteger(event.work)&&event.work>=0);
    if(!event.prediction){assert(event.error);continue;}
    const [head,intent,slots]=event.prediction;assert.equal(head,'intent');assert(Array.isArray(slots)&&slots[0]==='slots');
    const values=new Map();let parseError=false;
    try{for(const pair of slots.slice(1)){assert.equal(pair.length,2);const [role,token]=pair;assert(['a','b','both'].includes(role));assert(words.some(w=>w.replace(/[.?!]+$/u,'')===token));const n=decimal(token);for(const key of role==='both'?['a','b']:[role]){assert(!values.has(key));values.set(key,n);}}}catch(e){assert(event.error,`Invalid action without refusal: ${e.message}`);parseError=true;}
    if(parseError)continue;
    if(intent==='math-done'){
      assert.equal(values.size,1);assert(values.has('a'));assert(close(event.answer,values.get('a')));answer=event.answer;assert.equal(step,row.trace.length-1);continue;
    }
    assert(operations.has(intent));assert.equal(values.size,2);assert(values.has('a')&&values.has('b'));
    if(registers.length===8){assert.equal(event.error,actionQuotaError);continue;}
    calls++;const a=values.get('a'),b=values.get('b'),op=operations.get(intent);let n;
    switch(op){case '+':n=a+b;break;case '-':n=a-b;break;case '*':n=a*b;break;case '/':n=a/b;break;default:assert.fail('operator');}
    const domain=(!Number.isFinite(n))||(op==='/'&&b===0)||(n===0&&a!==0&&b!==0&&['*','/'].includes(op))||Math.abs(n)>1e9;
    if(domain){assert(event.error);assert(!Object.hasOwn(event,'computed'));continue;}
    assert(!event.error);assert(close(event.computed,n));registers.push(event.computed);used.push(intent);
  }
  assert.equal(work,row.language_work);assert.equal(calls,row.arithmetic_evaluations);
  if(row.answer!==null){assert.equal(row.error,null);assert(close(row.answer,answer));}else{assert(row.error);assert.equal(answer,null);}
  return {calculations:registers.length,order:used.join(','),new_order:!orders.has(used.join(','))};
}
