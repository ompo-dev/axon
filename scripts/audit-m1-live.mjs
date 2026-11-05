import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {restoreRuns} from '../ui/run-ledger.mjs';

const [workflowFile,exportFile,output]=process.argv.slice(2);
if(!output)throw Error('usage: bun scripts/audit-m1-live.mjs <workflow.json> <export.jsonl> <new-audit.json>');
const workflow=JSON.parse(await fs.readFile(workflowFile,'utf8'));
const raw=await fs.readFile(exportFile);
const rows=raw.toString('utf8').trim().split(/\r?\n/).map(line=>JSON.parse(line));
assert.equal(workflow.status,'complete');assert.equal(workflow.exit_code,0);
assert.equal(rows.length,workflow.trace.records);
let core=0,decisions=0,turns=0,filePlans=0,fileResults=0;
const grounds=new Map(),plans=new Map();
for(let i=0;i<rows.length;i++) {
  const e=rows[i];assert.equal(e.run,workflow.id);assert.equal(e.coreGaps,0);
  if(i)assert.equal(e.seq,rows[i-1].seq+1);
  if(e.coreSeq!==undefined)assert.equal(e.coreSeq,++core);
  const key=`${e.variant}:${e.stage}:${e.turn}`;
  if(e.method==='Session::ground'){assert(!grounds.has(key));grounds.set(key,e.seq);decisions++;}
  if(e.method==='Session::turn'){assert(grounds.get(key)<e.seq);turns++;}
  if(e.method==='Agent::file_plan'){plans.set(e.case,e);filePlans++;}
  if(e.method==='Agent::file_result') {
    const value=JSON.parse(e.actual_json);fileResults++;
    assert(value.seed_unchanged&&value.teacher_queries===0);
    if(value.plan!==null&&value.plan!==undefined) {
      const plan=plans.get(e.case-1);assert(plan&&plan.seq<e.seq);
      assert.deepEqual(JSON.parse(plan.plans_json),value.plan);
    } else assert.notEqual(value.status,'answer');
  }
}
assert.equal(rows.at(-1).event,'end');assert.equal(rows.at(-1).status,'complete');
assert.equal(decisions,4608);assert.equal(turns,4608);assert.equal(filePlans,6);assert.equal(fileResults,9);
const restored=await restoreRuns(path.resolve('.axon-memory/observatory'));
assert.equal(restored.current.id,workflow.id);assert.equal(restored.current.status,'complete');
assert.equal(restored.current.trace.records,rows.length);assert.equal(restored.current.trace.gaps,0);
assert.equal(restored.current.result.file_correct,9);assert.equal(restored.current.result.file_total,9);
assert.equal(restored.current.cases.filter(e=>e.method==='Agent::file_result').length,9);
const report={schema:'axon-m1-live-audit/v1',run:workflow.id,records:rows.length,core_records:core,gaps:0,decisions,turns,file_plans:filePlans,file_results:fileResults,
  decisions_before_results:true,planned_files_before_results:true,completed_ledger_replay:true,
  export_sha256:createHash('sha256').update(raw).digest('hex'),elapsed_ms:workflow.elapsed_ms,
  scope:'Actual completed local workflow and restored ledger. Invalid schemas/unrecognized requests need not invent a plan. Consumed authored development, not M1 approval.'};
await fs.writeFile(output,JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
