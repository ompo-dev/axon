import fs from 'node:fs';
const destination=process.argv[2];if(!destination)throw new Error('new protocol path required');
const functions=[(a,b)=>a+b,(a,b)=>a*b,(a,b)=>a*a+b,(a,b)=>(a-b)*(a-b),(a,b)=>a*a+b*b,(a,b)=>2n*a+3n*b,(a,b)=>a*b+b,(a,b)=>a+b*b];
const frame=(i,n,a,b)=>[
  ['goal','avalie',n,'com',a,'e',b],['goal','use',n,'em',a,'e',b],
  ['goal','resultado','de',n,'para',a,'e',b],['goal','passe',a,'e',b,'por',n],
][i%4];
const cases=[];
for(let t=0;t<8;t++)for(let i=0;i<150;i++){
  let input=41+i*2,value=BigInt(input);const task=(t+2)%8;
  let term=input;
  const depth=2+i%3;
  for(let d=0;d<depth;d++){
    // Keep deeper paths in the additive/product families so the exact result fits i64.
    const index=d===depth-1?task:(d%2);
    const op=(index+6)%8;
    const operand=3+(i*7+d)%13;
    term=frame(i+d,`instrumento${op}`,term,operand);
    value=functions[index](value,BigInt(operand));
  }
  if(value>BigInt(Number.MAX_SAFE_INTEGER))throw new Error('unsafe JSON result');
  cases.push({id:`depth-${t}-${i}`,family:`written-composition-depth-${depth}`,input:term,expected:Number(value),forbid_heads:['goal','invoke']});
}
const negatives=[['qual','seria','o','resultado','se','eu','mudasse','tudo'],['explique','por','que','a','formula','funciona'],['derive','sin','de','x']];
for(let i=0;i<negatives.length;i++)cases.push({id:`unlearned-${i}`,family:'unlearned-natural-requests',input:['goal',...negatives[i]],expected:null,forbid_heads:['goal','invoke']});
fs.writeFileSync(destination,JSON.stringify({id:'agent-deep-reserve-v1',cases},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({destination,cases:cases.length,depths:[2,3,4],scope:'New composition depths in taught frames, same numeric families; not an external language benchmark.'}));
