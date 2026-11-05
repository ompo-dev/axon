import assert from 'node:assert/strict';

export function sealedQuestions(raw,expected=1000){
  assert(raw.byteLength<=1000000,'SVAMP source quota');const records=JSON.parse(raw),accesses={answers:0,equations:0},ids=new Set();assert(Array.isArray(records)&&records.length===expected,'SVAMP cardinality');
  const inputs=records.map(row=>{
    assert.deepEqual(Object.keys(row).sort(),['Answer','Body','Equation','ID','Question','Type']);
    for(const [field,key]of [['Answer','answers'],['Equation','equations']])Object.defineProperty(row,field,{enumerable:true,get(){accesses[key]++;throw Error('Sealed SVAMP annotation');}});
    assert(typeof row.ID==='string'&&row.ID.length>0&&row.ID.length<=96&&!ids.has(row.ID));ids.add(row.ID);assert(typeof row.Body==='string'&&typeof row.Question==='string'&&row.Body.length&&row.Question.length);
    return {id:row.ID,text:row.Body+' '+row.Question};
  });return {inputs,accesses,sealed_records:records};
}
export function answerScalar(value){
  assert(typeof value==='number'||typeof value==='string','SVAMP answer type');
  if(typeof value==='string')assert(/^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?$/u.test(value),'SVAMP decimal answer');
  const number=Number(value);assert(Number.isFinite(number)&&Math.abs(number)<=1e9,'SVAMP answer bounds');return number===0?0:number;
}
