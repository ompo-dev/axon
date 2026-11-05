import test from 'node:test';
import assert from 'node:assert/strict';
import {scalar,question,expression,evaluate,finalScalar,teaching,numbersOnly,tuples} from './math-program-source.mjs';

test('arithmetic AST is bounded and has no executable syntax',()=>{
  assert.equal(evaluate(expression('(-2+3.5)*4/2')),3);
  for(const s of ['process.exit()','a+1','(()=>1)()','1;2','1,2','2**3','1%2','1e3','0x10','[1][0]','1/*x*/+2;evil()','NaN','Infinity','1/0'])assert.throws(()=>evaluate(expression(s)),s);
  assert.throws(()=>expression('('.repeat(30)+'1'+')'.repeat(30)));
  assert.throws(()=>evaluate(['*',1e-200,1e-200]));
});
test('decimal interface and question quota never silently truncate',()=>{
  for(const [s,n]of [['$5.20,',5.2],['-3',-3],['+0.3?',.3],['12.',12]])assert.equal(scalar(s),n);
  for(const s of ['12%','1,200','2e3','01','NaN','$','3/2','1..','1_000','1000000001'])assert.throws(()=>scalar(s),s);
  assert.throws(()=>question(Array(65).fill('a').join(' ')));assert.throws(()=>question('a'.repeat(257)));
  assert.throws(()=>question('bad\u0000 text'));assert.equal(question('x\ny  4').text,'x y 4');
});
test('supervised traces expand intermediate values with explicit argument roles',()=>{
  const e=teaching('Sam buys 3 bags with 8 stones and removes 5 stones. How many?', 'First <<3*8=24>> then <<24-5=19>>.\n#### 19');
  assert.deepEqual(e.ops,['x0','x1','*','x2','-']);assert.equal(evaluate(e.tree,[7,9,4]),59);
  assert.deepEqual(numbersOnly(e),{text:'3 8 5',tags:['B:x0','B:x1','B:x2']});
  const reverse=teaching('Sam had 5 and needed 12. Difference?', '<<12-5=7>>\n#### 7');
  assert.deepEqual(reverse.values,[12,5]);assert.deepEqual(numbersOnly(reverse).tags,['B:x1','B:x0']);
  assert.equal(finalScalar('x\n#### -2.5'),-2.5);assert.equal(tuples(4,0,8).length,8);
});
test('ambiguous, implicit, rounded and unsupported source supervision refuses',()=>{
  for(const [q,a]of [
    ['3 bags and 3 stones','<<3*3=9>>\n#### 9'],
    ['3 items cost 7 each','<<3*7=21>><<21/100=0.21>>\n#### 0.21'],
    ['2 and 3','<<2/3=0.67>>\n#### 0.67'],
    ['2 and 3','<<2+3=5>>\n#### 7'],
    ['2 and 3','No trace\n#### 5'],
    ['2 and 3','<<2+3=5>><<5+2=7>>\n#### 7 dollars'],
    ['3 and 6 and 2','<<3*2=6>><<6+2=8>>\n#### 8']
  ])assert.throws(()=>teaching(q,a));
});
