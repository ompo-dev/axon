import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {sha} from './gsm8k-source.mjs';
import {checkPointerTrace} from './relational-pointer-trace.mjs';

const [candidate,out]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/replay-pointer-workspace.mjs <candidate.exe> <new-output>');
const output=path.resolve(out);assert(output.startsWith(process.cwd()+path.sep));await fs.mkdir(output);const write=(file,value)=>fs.writeFile(path.join(output,file),JSON.stringify(value,null,2),{flag:'wx'}),binary=path.join(output,'candidate.exe');await fs.copyFile(candidate,binary,fs.constants.COPYFILE_EXCL);
const plans=[{run:'.axon-memory/m1-relational-pointer-v2',index:1,kinds:['teacher','own','development']},{run:'.axon-memory/m1-pointer-augmentation-v1',index:1,kinds:['development']}],results=[],calls=[];
const strip=value=>JSON.parse(JSON.stringify(value,(key,v)=>['workspace_bytes','peak_model_workspace_bytes'].includes(key)?undefined:v));
await write('freeze.json',{candidate_sha256:sha(await fs.readFile(binary)),plans,script_sha256:sha(await fs.readFile(process.argv[1])),scope:'Frozen weights and requests only, no retraining or gold. Only accounted workspace is permitted to change; positions, decisions, arithmetic, traces, margins and work must remain identical.'});
for(const [planIndex,plan]of plans.entries()){
  const study=JSON.parse(await fs.readFile(path.join(plan.run,'predictions-ready.json'))),model=study.results[plan.index],dir=path.join(plan.run,`policy-${plan.index}`),number=path.join(output,`model-${planIndex}.axnum`);await fs.copyFile(path.join(dir,'combined/structural.axnum'),number,fs.constants.COPYFILE_EXCL);assert.equal(sha(await fs.readFile(number)),model.number_sha256);
  for(const kind of plan.kinds){const before=JSON.parse(await fs.readFile(path.join(dir,`${kind}-predictions.json`))),inputs=kind==='teacher'?before.map(r=>({id:r.id,text:r.frame})):JSON.parse(await fs.readFile(path.join(dir,`${kind}-inputs.json`))),batch=kind==='teacher'?32:4,after=[];
    assert.equal(before.length,inputs.length);
    for(let start=0;start<inputs.length;start+=batch){const name=`model-${planIndex}-${kind}-${start}`,input=path.join(output,`${name}-inputs.json`),receipt=path.join(output,`${name}-predictions.json`);await write(`${name}-inputs.json`,inputs.slice(start,start+batch));const args=[kind==='teacher'?'pointer-frames-batch':'pointer-batch',number,path.join(output,`${name}-birth`),input,receipt];
      const {stdout,stderr}=await promisify(execFile)(binary,args,{windowsHide:true,timeout:300000,maxBuffer:4000000});await fs.writeFile(path.join(output,`${name}.stdout.log`),stdout,{flag:'wx'});await fs.writeFile(path.join(output,`${name}.stderr.log`),stderr,{flag:'wx'});const summary=JSON.parse(stdout);assert(summary.seed_unchanged);const rows=JSON.parse(await fs.readFile(receipt));assert.deepEqual(rows.map(r=>r.id),inputs.slice(start,start+batch).map(r=>r.id));after.push(...rows);calls.push({name,args,exit_code:0});
    }
    let frames=0,accountedSaved=0;
    for(const [i,row]of after.entries()){assert.deepEqual(strip(row),strip(before[i]),'Frozen behavior changed');if(kind!=='teacher')checkPointerTrace(row,inputs[i].text);const newFrames=kind==='teacher'?[row]:row.trace,oldFrames=kind==='teacher'?[before[i]]:before[i].trace;
      for(const [j,event]of newFrames.entries()){if(event.workspace_bytes===null||event.workspace_bytes===undefined)continue;const saved=2*event.frame.split(' ').length*16*4;assert.equal(oldFrames[j].workspace_bytes-event.workspace_bytes,saved);frames++;accountedSaved+=saved;}
      if(kind!=='teacher')assert(row.peak_model_workspace_bytes<=before[i].peak_model_workspace_bytes);
    }
    assert.equal(sha(await fs.readFile(number)),model.number_sha256);const result={run:plan.run,index:plan.index,kind,rows_identical_except_workspace:after.length,measured_frames:frames,total_accounted_bytes_saved_across_frames:accountedSaved,not_process_rss:true};results.push(result);console.log(JSON.stringify(result));
  }
}
await write('audit.json',{schema:'axon-pointer-workspace-replay/v1',verified:true,results,calls,teacher_queries:0,models_refitted:0,m1_complete:false,m2_permitted:false});
