import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {isDeepStrictEqual} from 'node:util';
import {sha} from './gsm8k-source.mjs';
import {scalar} from './math-program-source.mjs';

const [run,out]=process.argv.slice(2);if(!out)throw Error('usage: node scripts/diagnose-pointer-state.mjs <frozen-study> <new-output>');await fs.mkdir(out);
const read=async file=>JSON.parse(await fs.readFile(path.join(run,file))),study=await read('predictions-ready.json'),frozen=await read('freeze.json'),cohorts=await read('cohorts.json');assert.equal(sha(await fs.readFile(path.join(run,'cohorts.json'))),frozen.cohorts_sha256);
const masked=text=>text.split(' ').map(word=>{try{scalar(word);return'<scalar>';}catch{return word;}}).join(' '),results=[];
for(const model of study.results){
  const page=model.policy.numbers_only?'free':'full',episodes=cohorts[model.policy.cohort],directory=`policy-${model.index}`,teacher=await read(`${directory}/teacher-predictions.json`),teacherById=new Map(teacher.map(r=>[r.id,r])),own=await read(`${directory}/own-predictions.json`);assert.equal(sha(await fs.readFile(path.join(run,directory,'own-predictions.json'))),model.own.predictions_sha256);assert.equal(own.length,episodes.length);
  let compared=0,alias=0,differentValues=0,pastDeviated=0,samePrediction=0,samePredictionOnDifferentValues=0;
  const witnesses=[];
  for(const [i,row]of own.entries()){const episode=episodes[i];assert.equal(row.id,episode.id);let historyDiverged=false;
    for(const event of row.trace){const taught=teacherById.get(`${row.id}:${event.step}`);if(!taught)continue;compared++;const equivalent=masked(taught.frame)===masked(event.frame);if(equivalent){alias++;const same=isDeepStrictEqual(event.prediction,taught.prediction);samePrediction+=Number(same);const different=event.frame!==taught.frame;differentValues+=Number(different);pastDeviated+=Number(historyDiverged);samePredictionOnDifferentValues+=Number(same&&different);
        if(model.training.candidate_lexemes_retained===0){assert(same,'Anonymous input alias changed selection');assert.equal(event.margin,taught.margin);assert.equal(event.work,taught.work);}
        if((different||historyDiverged)&&witnesses.length<8)witnesses.push({id:row.id,step:event.step,different_register_values:different,earlier_action_deviated_from_provided_trace:historyDiverged,teacher_frame:taught.frame,own_frame:event.frame,teacher_prediction:taught.prediction,own_prediction:event.prediction});
      }
      const targets=episode[page].examples[event.step]?.targets;if(targets&&!targets.some(t=>isDeepStrictEqual(t,event.prediction)))historyDiverged=true;
    }
  }
  results.push({index:model.index,name:model.policy.name,compared_frames:compared,numeric_masked_aliases:alias,aliases_with_different_register_values:differentValues,aliases_after_noncanonical_action:pastDeviated,same_prediction_on_alias:samePrediction,same_prediction_on_different_values:samePredictionOnDifferentValues,anonymous:model.training.candidate_lexemes_retained===0,witnesses});
}
const report={schema:'axon-pointer-state-observability/v1',script_sha256:sha(await fs.readFile(process.argv[1])),study_sha256:sha(await fs.readFile(path.join(run,'predictions-ready.json'))),results,new_inferences:0,new_fits:0,new_corpus_answers_read:0,m1_complete:false,m2_permitted:false,scope:'Training-only representation diagnosis using already frozen native teacher/own traces. Anonymous state aliases cannot distinguish numeric register changes; inputs also omit the actual operator/operand provenance carried only in the external trace. Not a mathematical impossibility proof for all solutions: a policy may recompute from the question, and this does not establish the sole cause of public failure. Motive for explicit learned causal working state, not a new M1 gate or capability.'};await fs.writeFile(path.join(out,'audit.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(results.map(({witnesses,...r})=>r)));
