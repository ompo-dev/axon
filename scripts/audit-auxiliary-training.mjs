import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {sha} from './conllu-source.mjs';
import {loadPublicPos, shuffledLabels} from './auxiliary-pos-source.mjs';
import {loadLexicalGraph,rewireEdges} from './lexical-graph-source.mjs';

const [run, output] = process.argv.slice(2);
if (!output) throw Error('usage: node scripts/audit-auxiliary-training.mjs <frozen-run> <new-output>');
await fs.mkdir(output);
const write = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2), {flag: 'wx'});
const read = async name => JSON.parse(await fs.readFile(path.join(run, name)));
const frozen = await read('freeze.json'), audit = await read('audit.json');
for (const file of ['scripts/auxiliary-pos-source.mjs', 'scripts/conllu-source.mjs']) {
  assert.equal(sha(await fs.readFile(file)), frozen.sources[file], 'Source helper changed since this experiment');
}
const protocols = Object.entries(frozen.sources).filter(([, hash]) => hash === frozen.protocol_sha256);
assert.equal(protocols.length, 1);
const protocol = await read(path.join('source', protocols[0][0]));
const byteStudy = protocol.schema === 'axon-public-byte-convolution/v1';
const orderedStudy = protocol.schema === 'axon-public-ordered-context/v1';
const adaptiveStudy = protocol.schema === 'axon-public-adaptive-training/v1';
const rolesStudy = protocol.schema === 'axon-public-role-schema/v1';
const lexicalStudy = protocol.schema === 'axon-public-lexical-graph/v1';
const conditionalStudy=protocol.schema==='axon-public-conditional-loss/v1';
const splitStudy=protocol.schema==='axon-public-split-mixers/v1';
if (byteStudy) assert.equal(protocol.policies.length, 5);
if (orderedStudy) assert.equal(protocol.policies.length, 7);
if (adaptiveStudy) assert.equal(protocol.policies.length, 6);
if (rolesStudy) assert.equal(protocol.policies.length, 3);
if (lexicalStudy) assert.equal(protocol.policies.length, 4);
if (conditionalStudy)assert.equal(protocol.policies.length,6);
if (splitStudy)assert.equal(protocol.policies.length,6);
let lexicalGraph,rewiredGraph;
if(protocol.lexical_source) {
  lexicalGraph=await loadLexicalGraph({...protocol.lexical_source,facts:path.join(run,'lexical-source/facts.tsv'),vocabulary:path.join(run,'lexical-source/vocabulary.axseq'),license_file:path.join(run,'lexical-source/LICENSE')});
  rewiredGraph=rewireEdges(lexicalGraph.edges,19374611);
  assert.deepEqual(await read('lexical-edges.json'),lexicalGraph.edges);assert.deepEqual(await read('lexical-rewired-edges.json'),rewiredGraph.edges);
  assert.deepEqual(await read('lexical-source.json'),lexicalGraph.statistics);assert.deepEqual(await read('lexical-rewiring.json'),rewiredGraph.statistics);
  assert.equal(sha(await fs.readFile(path.join(run,'lexical-edges.json'))),frozen.lexical_graph.edges_sha256);
  assert.equal(sha(await fs.readFile(path.join(run,'lexical-rewired-edges.json'))),frozen.lexical_graph.rewired_edges_sha256);
  const degrees=edges=>{const result=new Map();for(const edge of edges)for(const word of edge)result.set(word,(result.get(word)??0)+1);return [...result].sort(([a],[b])=>Buffer.compare(Buffer.from(a),Buffer.from(b)));};
  assert.deepEqual(degrees(lexicalGraph.edges),degrees(rewiredGraph.edges));
}
let source = {examples: [], origins: [], stats: {source_words: 0, words: 0, frames: 0, source_sentences: 0}};
if (protocol.auxiliary_source) {
source = await loadPublicPos({...protocol.auxiliary_source, corpus: path.join(run, 'auxiliary-source')});
assert.deepEqual(await read('auxiliary-examples.json'), source.examples);
assert.deepEqual(await read('auxiliary-origins.json'), source.origins);
assert.equal(sha(await fs.readFile(path.join(run, 'auxiliary-examples.json'))), frozen.auxiliary.examples_sha256);
assert.equal(sha(await fs.readFile(path.join(run, 'auxiliary-origins.json'))), frozen.auxiliary.origins_sha256);
const provenance = await read('auxiliary-source/provenance.json');
for (const name of ['LICENSE.txt', 'README.md']) {
  assert.equal(sha(await fs.readFile(path.join(run, 'auxiliary-source', name))), provenance.sources[name].sha256);
}
} else {
  assert.equal(frozen.auxiliary, null);
  assert.equal(protocol.auxiliary_policies, undefined);
}
await fs.copyFile(process.argv[1], path.join(output, 'auditor.mjs'), fs.constants.COPYFILE_EXCL);
await write('freeze.json', {run_freeze_sha256: sha(await fs.readFile(path.join(run, 'freeze.json'))),
  auditor_sha256: sha(await fs.readFile(process.argv[1])),
  scope: 'Host-only audit of frozen training artifacts, source transforms, deployed shapes and telemetry; no new predictions or test annotations.'});
const control = await read('policy-0-train.json');
assert.equal(audit.policies.length, protocol.policies.length);
const results = []; let lexicon, metadata, inputCoverage;
for (const p of audit.policies) {
  const i = p.index, training = await read(`policy-${i}-train.json`), extra = protocol.auxiliary_policies?.[i];
  assert.deepEqual(training.examples, control.examples);
  assert.deepEqual(training.budget, protocol.policies[i]);
  if (extra) assert.deepEqual(training.auxiliary, {weight: extra.weight, interval: extra.interval,
    examples: extra.shuffle_labels ? shuffledLabels(source.examples, 19374611) : source.examples});
  else assert.equal(training.auxiliary, undefined);
  const lexicalPolicy=protocol.lexical_policies?.[i];
  if(lexicalPolicy)assert.deepEqual(training.lexical_graph,{weight:lexicalPolicy.weight,interval:lexicalPolicy.interval,edges:lexicalPolicy.rewire?rewiredGraph.edges:lexicalGraph.edges});
  else assert.equal(training.lexical_graph,undefined);
  const page = await fs.readFile(path.join(run, `policy-${i}/semantic-lexical-encoder-v1.axseq`));
  assert.deepEqual(page, await fs.readFile(path.join(run, `policy-${i}-dev-0-birth/semantic-lexical-encoder-v1.axseq`)),
    'Training export must match the payload actually born for inference');
  const convolution = training.budget.byte_convolution ?? false;
  const ordered = training.budget.ordered_context ?? false;
  const typed = training.budget.typed_roles ?? false;
  const split = training.budget.split_heads ?? false;
  assert.equal(page.subarray(0, 8).toString(), split ? 'AXSQ0013' : typed ? 'AXSQ0012' : ordered ? 'AXSQ0011' : convolution ? 'AXSQ0010' : 'AXSQ0009');
  const end = 12 + page.readUInt32LE(8), header = JSON.parse(page.subarray(12, end));
  assert.equal(header.byte_convolution ?? false, convolution);
  assert.equal(header.ordered_context ?? false, ordered);
  assert.equal(header.typed_roles ?? false, typed);
  assert.equal(header.split_heads ?? false, split);
  if(split)assert(!typed&&!ordered&&!convolution&&header.relative_regions&&header.hidden>0);
  assert.equal(header.freeze_byte_convolution, undefined, 'Training policy is not deployed state');
  assert.equal(header.freeze_ordered_context, undefined);
  assert.equal(header.mean_token_loss, undefined);
  assert.equal(header.minibatch, undefined);
  assert.equal(header.optimizer, undefined);
  assert.equal(header.lexical_graph,undefined);
  assert.equal(header.constrained_loss,undefined);
  if (convolution) assert.equal(header.characters, 0);
  if (ordered) {assert.equal(convolution, false); assert.equal(header.characters, 0);}
  const words = page.subarray(end, end + header.word_bytes);
  if (i === 0) {
    lexicon = Buffer.from(words); metadata = header.metadata;
    const vocabulary = new Set(words.toString('utf8').split('\0').slice(0, -1)), byPos = {}, unknown = new Set();
    assert.equal(vocabulary.size, header.words);
    let known = 0, unavailable = 0;
    for (const e of source.examples) {
      const tokens = e.text.toLowerCase().split(/\s+/u); assert.equal(tokens.length, e.labels.length);
      tokens.forEach((word, j) => {
        const counts = byPos[e.labels[j]] ??= {known: 0, unknown: 0};
        if (vocabulary.has(word)) {known++; counts.known++;}
        else {unavailable++; counts.unknown++; unknown.add(word);}
      });
    }
    inputCoverage = protocol.auxiliary_source ? {known_word_tokens: known, unknown_word_tokens: unavailable, unknown_word_types: unknown.size,
      character_buckets: header.characters, by_pos: byPos,
      scope: 'Train-only lexical coverage. With zero character buckets all unseen spellings share the unknown-word row, but their contexts may differ; this is not a causal explanation or an accuracy estimate.'} : null;
  }
  else {assert.deepEqual(words, lexicon); assert.deepEqual(header.metadata, metadata);}
  const rows = 3 + header.words + header.characters + header.metadata.intents.length + header.metadata.tags.length;
  const classes = header.metadata.intents.length + header.metadata.tags.length;
  const width = header.dimensions * 7, units = header.hidden ?? 0;
  const mixers = units * (1 + Number(split));
  const outputWeights = units ? mixers * width + classes * units : classes * width;
  const extraWeights = ordered ? 2 * header.dimensions ** 2 : convolution ? 258 * 8 + header.dimensions * 24 : 0;
  const extraBytes = ordered ? 2 * header.dimensions * (header.dimensions + 8) : convolution ? 258 * (8 + 4) + header.dimensions * 24 + header.dimensions * 8 : 0;
  const roleStride=Math.ceil(header.metadata.tags.length/7),roleCells=typed?header.metadata.intents.length*roleStride:0;
  const roleBytes=typed?roleCells+4:0;
  assert.equal(page.length, end + header.word_bytes + rows * (header.dimensions + 4) + outputWeights + (classes + mixers) * 8 + extraBytes + roleBytes + 8,
    'Deployed payload must contain only primary heads and declared parameters');
  assert.equal(rows * header.dimensions + outputWeights + extraWeights, p.training.summary.weights);
  const matching = audit.policies.find(q => (q.training.summary.budget.byte_convolution ?? false) === convolution
    && (q.training.summary.budget.ordered_context ?? false) === ordered && (q.training.summary.budget.hidden ?? 0) === units && (q.training.summary.budget.typed_roles??false)===typed && (q.training.summary.budget.split_heads??false)===split);
  assert.equal(p.training.summary.weights, matching.training.summary.weights);
  assert.equal(p.training.summary.retained_model_bytes, matching.training.summary.retained_model_bytes);
  if (protocol.expected_numbers?.[i]) assert.equal(p.number_sha256, protocol.expected_numbers[i]);
  const events = (await fs.readFile(path.join(run, `policy-${i}-training.stdout.jsonl`), 'utf8')).trim().split(/\r?\n/u).map(JSON.parse);
  const summary = events.pop(); assert.deepEqual(summary, p.training.summary);
  assert.equal(events.length, training.budget.epochs);
  const counts = {sentences: 0, words: 0, errors: 0};
  events.forEach((e, n) => {
    assert.equal(e.coreSeq, n + 1); assert.equal(e.epoch, n + 1);
    for (const field of Object.keys(counts)) counts[field] += e[`auxiliary_${field}`];
  });
  for (const [field, count] of Object.entries(counts)) assert.equal(count, summary.auxiliary[field]);
  assert.equal(counts.sentences, extra ? Math.floor(training.examples.length * training.budget.epochs / extra.interval) : 0);
  assert.equal(summary.auxiliary.head_deployed, false);
  let optimizerSteps = null;
  if (adaptiveStudy||rolesStudy) {
    const batch=training.budget.minibatch??0,expected=batch?Math.ceil(training.examples.length/batch):0;
    for(const event of events)assert.equal(event.optimizer_steps,expected);
    optimizerSteps=events.reduce((n,e)=>n+e.optimizer_steps,0);
    assert.deepEqual(summary.optimizer,{kind:batch?'minibatch-adam':'online-sgd',steps:optimizerSteps,batch_sentences:batch,state_deployed:false,gradient_clip_norm:batch?1:null});
    assert.equal(optimizerSteps,expected*training.budget.epochs);
  }
  let lexicalUpdates=null;
  if(lexicalStudy||conditionalStudy||splitStudy) {
    lexicalUpdates=events.reduce((n,e)=>n+e.lexical_edge_updates,0);
    assert(events.every(e=>e.lexical_edge_updates===(lexicalPolicy?.weight>0?training.examples.length:0)));
    assert.equal(lexicalUpdates,lexicalPolicy?.weight>0?272832:0);
    assert.deepEqual(summary.lexical_graph,{configured:!!lexicalPolicy,weight:lexicalPolicy?Math.fround(lexicalPolicy.weight):null,interval:lexicalPolicy?.interval??null,edges:lexicalPolicy?lexicalGraph.edges.length:null,updates:lexicalUpdates,edges_deployed:false});
  }
  let normalizationTerms=null;
  if(conditionalStudy||splitStudy) {
    const support=new Map();for(const example of training.examples) {const roles=support.get(example.intent)??new Set();for(const tag of example.tags)if(tag!=='O')roles.add(tag.slice(2));support.set(example.intent,roles);}
    let expected=0;
    for(const example of training.examples) {let previous='O';for(const tag of example.tags) {expected+=training.budget.constrained_loss?1+support.get(example.intent).size+Number(previous!=='O'):header.metadata.tags.length;previous=tag;}}
    for(const event of events)assert.equal(event.primary_tag_normalization_terms,expected);
    normalizationTerms=events.reduce((n,e)=>n+e.primary_tag_normalization_terms,0);assert.equal(normalizationTerms,summary.primary_tag_normalization_terms);
  }
  if(typed) {
    const expected=Buffer.alloc(roleCells);for(let intent=0;intent<header.metadata.intents.length;intent++)expected[intent*roleStride]=1;
    for(const example of training.examples)for(const tag of example.tags)if(tag!=='O') {
      const intent=header.metadata.intents.indexOf(example.intent);assert(intent>=0);
      for(const prefix of ['B:','I:']) {const index=header.metadata.tags.indexOf(prefix+tag.slice(2));assert(index>0);expected[intent*roleStride+Math.floor(index/7)]|=1<<(index%7);}
    }
    const outputStart=end+header.word_bytes+rows*(header.dimensions+4),scales=(classes+units+1)*4;
    assert.equal(page.readFloatLE(outputStart+(classes+units)*4),1/16);
    assert.deepEqual(page.subarray(outputStart+scales+outputWeights,outputStart+scales+outputWeights+roleCells),expected);
    const comparison=protocol.neural_controls?.[i];if(rolesStudy)assert(comparison);
    if(comparison) {
    const previousAudit=JSON.parse(await fs.readFile(path.join(comparison.run,'audit.json'))),previous=previousAudit.policies.find(p=>p.index===comparison.index);assert(previous);
    assert.equal(sha(await fs.readFile(path.join(comparison.run,`policy-${comparison.index}/structural.axnum`))),previous.number_sha256);
    const old=await fs.readFile(path.join(comparison.run,`policy-${comparison.index}/semantic-lexical-encoder-v1.axseq`)),oldEnd=12+old.readUInt32LE(8),oldHeader=JSON.parse(old.subarray(12,oldEnd));
    const {typed_roles:_,...unrestrictedHeader}=header;assert.deepEqual(unrestrictedHeader,oldHeader);
    const oldOutputStart=oldEnd+oldHeader.word_bytes+rows*(header.dimensions+4),oldScales=(classes+units)*4;
    assert.deepEqual(page.subarray(end,outputStart),old.subarray(oldEnd,oldOutputStart),'Vocabulary/input quantization must be unchanged');
    assert.deepEqual(page.subarray(outputStart,outputStart+oldScales),old.subarray(oldOutputStart,oldOutputStart+oldScales),'Neural scales must be unchanged');
    assert.deepEqual(page.subarray(outputStart+scales,outputStart+scales+outputWeights),old.subarray(oldOutputStart+oldScales,oldOutputStart+oldScales+outputWeights),'Neural matrix bytes must be unchanged');
    const biasBytes=(classes+units)*4;assert.deepEqual(page.subarray(page.length-8-biasBytes,page.length-8),old.subarray(old.length-8-biasBytes,old.length-8),'Neural biases must be unchanged');
    }
  }
  results.push({index: i, counts, page_bytes: page.length, deployed_weights: summary.weights,
    ...((adaptiveStudy||rolesStudy)?{optimizer_steps:optimizerSteps,optimizer_state_deployed:false}:{}),
    ...((rolesStudy||conditionalStudy)?{typed_roles:typed,categorical_schema_bytes:roleBytes}:{}),
    ...((lexicalStudy||conditionalStudy||splitStudy)?{lexical_edge_updates:lexicalUpdates,lexical_edges_deployed:false}:{}),
    ...(splitStudy?{split_mixers:split,mixer_units:mixers,primary_tag_normalization_terms:normalizationTerms}:{}),
    ...(conditionalStudy?{constrained_loss:training.budget.constrained_loss??false,primary_tag_normalization_terms:normalizationTerms}:{}),
    byte_convolution: convolution, convolution_frozen_during_training: training.budget.freeze_byte_convolution ?? false,
    ordered_context: ordered, recurrent_weights_frozen_during_training: training.budget.freeze_ordered_context ?? false,
    extra_matrix_coefficients: extraWeights, retained_model_bytes: summary.retained_model_bytes,
    primary_data_identical: true, vocabulary_identical: true, primary_metadata_identical: true,
    auxiliary_head_deployed: false, shuffled_labels: extra?.shuffle_labels ?? null});
}
if(splitStudy) {
  assert.deepEqual(results.map(r=>r.split_mixers),[false,false,true,true,false,true]);
  assert.deepEqual(results.map(r=>r.deployed_weights),[86272,77280,80864,93440,86272,93440]);
  assert.deepEqual(results.map(r=>r.lexical_edge_updates),[0,0,0,0,272832,272832]);
  assert(results.every(r=>r.primary_tag_normalization_terms===211575104));
} else if(conditionalStudy) {
  assert.deepEqual(results.map(r=>r.typed_roles),[false,true,true,true,true,true]);
  assert.deepEqual(results.map(r=>r.constrained_loss),[false,false,false,true,true,true]);
  assert.deepEqual(results.map(r=>r.lexical_edge_updates),[0,0,0,0,0,272832]);
  assert(results[3].primary_tag_normalization_terms<results[1].primary_tag_normalization_terms);
} else if (lexicalStudy) {
  assert.deepEqual(results.map(r=>r.lexical_edge_updates),[0,272832,272832,272832]);
  assert(results.every(r=>!r.byte_convolution&&!r.ordered_context&&r.deployed_weights===86272));
} else if (rolesStudy) {
  assert.deepEqual(results.map(r=>r.typed_roles),[false,true,true]);
  assert.deepEqual(results.map(r=>r.optimizer_steps),[0,0,8544]);
} else if (adaptiveStudy) {
  assert(results.every(r=>!r.ordered_context&&!r.byte_convolution&&r.counts.words===0));
  assert.deepEqual(results.map(r=>r.optimizer_steps),[0,0,8544,8544,8544,8544]);
} else if (orderedStudy) {
  assert.deepEqual(results.map(r => r.ordered_context), [false, false, false, true, true, true, true]);
  assert.deepEqual(results.map(r => r.recurrent_weights_frozen_during_training), [false, false, false, true, false, true, false]);
  assert(results.every(r => !r.byte_convolution && r.counts.words === 0));
} else if (byteStudy) {
  assert.deepEqual(results.map(r => r.byte_convolution), [false, false, true, true, true]);
  assert.deepEqual(results.map(r => r.convolution_frozen_during_training), [false, false, true, false, false]);
  assert.equal(results[1].counts.words, results[4].counts.words);
  for (const i of [0, 2, 3]) assert.equal(results[i].counts.words, 0);
} else {
  assert.equal(results[1].counts.words, results[2].counts.words);
  assert.equal(results[1].counts.words, results[3].counts.words);
}
const report = {schema: splitStudy?'axon-split-mixers-training-audit/v1':conditionalStudy?'axon-conditional-loss-training-audit/v1':lexicalStudy ? 'axon-lexical-graph-training-audit/v1' : rolesStudy ? 'axon-role-schema-training-audit/v1' : adaptiveStudy ? 'axon-adaptive-training-audit/v1' : orderedStudy ? 'axon-ordered-training-audit/v3' : byteStudy ? 'axon-byte-auxiliary-training-audit/v2' : 'axon-auxiliary-training-audit/v1', results,
  ...((lexicalStudy||conditionalStudy||splitStudy)?{lexical_source:lexicalGraph.statistics,rewiring:rewiredGraph.statistics}:{}),
  source_words: source.stats.source_words, core_auxiliary_words: source.stats.words, frames: source.stats.frames,
  source_sentences: source.stats.source_sentences, input_coverage: inputCoverage,
  reserved_annotations_parsed: false, auxiliary_development_read: false,
  new_predictions: 0, m1_complete: false,
  scope: 'Training and payload integrity only. Extra source information/work/byte coefficients are explicit. Online POS errors are not held-out POS or semantic accuracy; numerical results never authorize effects.'};
await write('audit.json', report); console.log(JSON.stringify(report));
