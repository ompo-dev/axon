// Teacher data only. No Portuguese-to-operation dispatch is shipped in the runtime.
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '..');
const header = ['rule','sentence','kind','owner','item','before','after','focus_owner','focus_item','arguments','answer'];
const rows = [];
const cases = [
  ['Ana','Bruno','livros',18,5,3], ['Carla','Davi','moedas',25,7,9],
  ['Eva','Fabio','pecas',13,2,6], ['Gabi','Hugo','flores',37,11,4],
  ['Iara','Joao','cartas',19,6,8], ['Lia','Mauro','pontos',42,9,12],
  ['Nina','Otavio','frutas',31,4,7], ['Paula','Rui','bolas',28,8,5],
];
function add(rule,sentence,kind,owner='',item='',before='',after='',focusOwner='',focusItem='',args='',answer='') {
  rows.push([rule,sentence,kind,owner,item,before,after,focusOwner,focusItem,args,answer]);
}
for (const [a,b,u,n,q,m] of cases) {
  add('01-declare',`${a} tinha ${n} ${u}`,'change',a,u,'_',n);
  add('02-possess',`${a} possui ${n} ${u}`,'change',a,u,'_',n);
  add('03-receive',`${a} recebeu ${q} ${u}`,'change',a,u,n,n+q);
  add('04-spend',`${a} gastou ${q} ${u}`,'change',a,u,n,n-q);
  add('05-transfer',`${a} emprestou ${q} ${u} para ${b}`,'change',`${a}|${b}`,`${u}|${u}`,`${n}|${m}`,`${n-q}|${m+q}`);
  add('06-return',`${b} devolveu ${q} ${u} para ${a}`,'change',`${b}|${a}`,`${u}|${u}`,`${n}|${m}`,`${n-q}|${m+q}`);
  add('07-negation',`${a} nao recebeu ${q} ${u}`,'ignore');
  add('08-reference',`Ela recebeu ${q} ${u}`,'change',a,u,n,n+q,a);
  add('09-query',`Quantos ${u} ficaram com ${a}`,'question',a,u,n,'','','','',n);
  add('10-reference-query',`Quantos ${u} ela tem`,'question',a,u,n,'',a,'','',n);
  add('11-total',`Quantos ${u} ${a} e ${b} tem juntos`,'question',`${a}|${b}`,`${u}|${u}`,`${n}|${m}`,'','','','',n+m);
  add('12-compare',`Quantos ${u} ${a} tem a mais que ${b}`,'question',`${a}|${b}`,`${u}|${u}`,`${n}|${m}`,'','','','',n-m);
  add('13-product',`Quanto custa comprar ${q} unidades a ${m} por unidade`,'calculate','','','','','','',`${q}|${m}`,q*m);
  add('14-affine',`Qual x satisfaz ${q} vezes x mais ${m} igual a ${n}`,'calculate','','','','','','',`${q}|${m}|${n}`,(n-m)/q);
}
for (const [a,b] of [[3,4],[5,12],[8,15],[7,24],[9,40],[20,21],[12,35],[11,60]]) {
  add('15-norm',`Qual a norma de ${a} e ${b}`,'calculate','','','','','','',`${a}|${b}`,Math.hypot(a,b));
}
await writeFile(path.join(root,'evals/fixtures/relational-v1-lessons.tsv'), [header,...rows].map(r=>r.join('\t')).join('\n')+'\n', {flag:'wx'});
console.log(`Published ${rows.length} teacher demonstrations in ${new Set(rows.map(r=>r[0])).size} groups.`);
