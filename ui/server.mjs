import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs/promises';
import { createReadStream, createWriteStream, existsSync } from 'node:fs';
import { QueryMemory, mergeSeedCandidates } from './trace.js';
import { fileURLToPath } from 'node:url';
import { randomBytes, randomUUID } from 'node:crypto';
import os from 'node:os';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parse } from 'csv-parse/sync';
import { workflows, validateLabOptions, executeWorkflow, stopWorker } from './laboratory.mjs';
import { saveRun, restoreRuns } from './run-ledger.mjs';

const exec = promisify(execFile);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(HERE, 'dist');
const METHODS = [
  ['M01', 'BehaviorLearner::hypothesize', 'src/behavior.rs'],
  ['M02', 'TransferLearner::choose_probe', 'src/transfer.rs'],
  ['M03', 'TransferLearner::observe', 'src/transfer.rs'],
  ['M04', 'TransferLearner::compile', 'src/transfer.rs'],
  ['M05', 'conformance_probe', 'src/transfer.rs'],
  ['M06', 'factor::learn', 'src/factor.rs'],
  ['M07', 'FactorModel::plan', 'src/factor.rs'],
  ['M08', 'FactorTracker::observe', 'src/factor.rs'],
  ['M09', 'audit_machine', 'src/behavior_lab.rs'],
  ['M10', 'factor_lab::audit', 'src/factor_lab.rs'],
  ['M11', 'KnowledgeStore::investigate', 'src/knowledge.rs'],
  ['M12', 'solve_task', 'src/solver.rs'],
  ['M13', 'GoalController::propose', 'src/goal_controller.rs'],
  ['M14', 'GoalController::feedback', 'src/goal_controller.rs'],
  ['M15', 'DemandRuntime::observe_behavior', 'src/demand.rs'],
  ['M16', 'GoalController::rebind', 'src/goal_controller.rs'],
  ['M17', 'ControllerReplay::summary', 'src/bin/axon_uic_controller_lab.rs'],
  ['M18', 'Genome::birth', 'src/genome.rs'],
  ['M19', 'Genome::offspring', 'src/genome.rs'],
  ['M20', 'Genome::consolidate', 'src/genome.rs'],
  ['M21', 'LanguageFrame::learn', 'src/grounded_language.rs'],
  ['M22', 'LexiconView::snapshot', 'src/lexicon.rs'],
  ['M23', 'GroundedLanguage::answer', 'src/grounded_language.rs'],
  ['M24', 'Investigator::next_probe', 'src/cumulative.rs'],
  ['M25', 'Investigator::acquire', 'src/cumulative.rs'],
  ['M26', 'StudySeed::consolidate', 'src/cumulative.rs'],
  ['M27', 'Casework::plan_files', 'src/casework.rs'],
  ['M28', 'Rule::learn', 'src/semantic.rs'],
  ['M29', 'Session::narrative', 'src/semantic.rs'],
  ['M30', 'Relational::selection', 'src/semantic_lab.rs'],
  ['M31', 'Nursery::commit', 'src/nursery.rs'],
  ['M32', 'Nursery::grow', 'src/nursery.rs'],
  ['M33', 'Nursery::resume', 'src/nursery_lab.rs'],
  ['M34', 'ConstructiveSearch::run', 'src/numeric_constructive.rs'],
  ['M35', 'CircuitDna::program', 'src/numeric_constructive.rs'],
];

export async function catalogFiles(root) {
  const realRoot = await fs.realpath(root);
  const entries = [];
  async function walk(dir, depth) {
    if (depth > 4 || entries.length >= 400) return;
    let children;
    try { children = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
    children.sort((a, b) => a.name.localeCompare(b.name));
    for (const child of children) {
      if (entries.length >= 400 || child.isSymbolicLink()) continue;
      const full = path.join(dir, child.name);
      if (child.isDirectory()) await walk(full, depth + 1);
      else if (child.isFile() && /\.(rs|md|axon)$/i.test(child.name)) {
        const resolved = await fs.realpath(full);
        const relative = path.relative(realRoot, resolved);
        if (relative.startsWith('..') || path.isAbsolute(relative)) continue;
        const stat = await fs.stat(resolved);
        entries.push({ id: entries.length, path: relative.replaceAll('\\', '/'), bytes: stat.size,
          kind: child.name.endsWith('.rs') ? 'code' : child.name.endsWith('.axon') ? 'task' : 'paper' });
      }
    }
  }
  for (const folder of ['docs', 'src', 'tests', 'examples']) await walk(path.join(realRoot, folder), 0);
  return entries;
}

export function validateRun(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Invalid request');
  if (!['transfer', 'factor', 'knowledge', 'math', 'index', ...workflows.map(w=>w.id)].includes(input.mode)) throw new Error('Unknown task');
  const laboratory = workflows.some(w=>w.id===input.mode);
  const allowed = new Set(laboratory?['mode','seed','count','strategy','curriculum','suite','validation']:['mode','seed','dimensions','novel','paused','query','task']);
  if (Object.keys(input).some(key => !allowed.has(key))) throw new Error('Unknown option');
  const seed = input.seed ?? 719319;
  const dimensions = input.dimensions ?? 3;
  if (!Number.isSafeInteger(seed) || seed < 1 || seed > 2147483647) throw new Error('Seed must be 1..2147483647');
  if (!Number.isInteger(dimensions) || dimensions < 2 || dimensions > 5) throw new Error('Channels must be 2..5');
  for (const key of ['paused', 'novel']) if (input[key] !== undefined && typeof input[key] !== 'boolean') throw new Error('Invalid flag');
  if (input.paused && !['factor', 'transfer'].includes(input.mode)) throw new Error('This task does not support stepping');
  if (input.mode === 'knowledge' && (typeof input.query !== 'string' || !input.query.trim() || input.query.length > 2000)) throw new Error('Enter a query');
  if (input.mode === 'math' && (typeof input.task !== 'string' || !input.task.trim() || Buffer.byteLength(input.task) > 32768)) throw new Error('Invalid task source');
  return laboratory?{mode:input.mode,...(input.mode==='structural'?{}:{seed}),...validateLabOptions(input)}:{...input,seed,dimensions,novel:input.novel??false,paused:input.paused??false};
}

export async function evolutionCatalog(root) {
  const base = path.join(root, '.axon-memory', 'evolution');
  const realRoot = await fs.realpath(root);
  async function boundedRead(file,limit=256000) {
    const stat = await fs.lstat(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > limit) throw new Error('Invalid report');
    const relative = path.relative(realRoot, await fs.realpath(file));
    if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Report outside workspace');
    return { text: await fs.readFile(file, 'utf8'), modified: stat.mtimeMs };
  }
  let names;
  try { names = await fs.readdir(base); } catch { return []; }
  const campaigns = [];
  for (const name of names.filter(n => /^reserved-\d+-(hybrid|mutation|no-inheritance|balanced)-v\d+\.json$/.test(n)).sort().slice(-40)) {
    try {
      const report = await boundedRead(path.join(base, name));
      const data = JSON.parse(report.text);
      for (const field of ['seed','lifecycle_ms','max_sampled_working_set_bytes','binary_bytes','exit_code']) {
        if (typeof data[field] !== 'number' || !Number.isFinite(data[field]) || data[field] < 0) throw new Error('Invalid resource measurement');
      }
      if (!Number.isSafeInteger(data.seed) || !['hybrid','mutation-only','no-inheritance','balanced'].includes(data.evolution_mode)) continue;
      const csv = await boundedRead(path.join(base, name.replace(/\.json$/, '.outputs'), 'results.csv'));
      const cases = parse(csv.text, { columns: true, skip_empty_lines: true, max_record_size: 16000 });
      if (!cases.length || cases.length > 64 || data.mode !== 'evolution') continue;
      const numeric = ['case','generation','evaluations','work','active_genes','instructions','dependencies','capsule_bytes','correct','heldout','training_ns','local_proposals','residual_errors'];
      const optional = new Set(['active_genes','instructions','dependencies','capsule_bytes','residual_errors']);
      for (const row of cases) for (const field of numeric) {
        if (optional.has(field) && row[field] === '' && row.status !== 'Admitted') {row[field]=null;continue;}
        if (typeof row[field] !== 'string' || !row[field].trim()) throw new Error('Missing experiment value');
        row[field] = Number(row[field]);
        if (!Number.isFinite(row[field]) || row[field] < 0) throw new Error('Invalid experiment value');
      }
      campaigns.push({ id: name, seed: data.seed, mode: data.evolution_mode, modified: report.modified,
        lifecycleMs: data.lifecycle_ms, sampledRss: data.max_sampled_working_set_bytes,
        binaryBytes: data.binary_bytes, exitCode: data.exit_code, cases });
    } catch { /* Incomplete or invalid stored reports are not presented as measurements. */ }
  }
  for(const name of names.filter(n=>/^integrated-reserve-\d+-v\d+\.json$/.test(n)).sort().slice(-8)){
    try{
      const report=await boundedRead(path.join(base,name),4000000),data=JSON.parse(report.text);
      if(data.schema!=='axon-integrated-reserve-v1'||!Array.isArray(data.runs)||data.runs.length>32)continue;
      for(const run of data.runs){
        if(run.status!=='complete'||run.exit_code!==0||!Number.isSafeInteger(run.options?.seed)||!['hybrid','mutation-only','balanced'].includes(run.options.strategy)||!Array.isArray(run.cases)||run.cases.length!==data.tasks||!Number.isFinite(run.elapsed_ms)||run.elapsed_ms<0)continue;
        const cases=run.cases.map(c=>({case:c.case,status:c.outcome,generation:c.generation,evaluations:c.evaluations,work:c.work,active_genes:c.active_genes??null,instructions:c.instructions??null,dependencies:c.dependencies??null,capsule_bytes:c.capsule_bytes??null,correct:c.correct,heldout:c.heldout,training_ns:null,local_proposals:null,residual_errors:c.residual_errors??null,expression:c.expression??''}));
        if(cases.some(c=>['case','generation','evaluations','work','correct','heldout'].some(k=>!Number.isFinite(c[k])||c[k]<0)))continue;
        campaigns.push({id:`${name}:${run.id}`,seed:run.options.seed,mode:run.options.strategy,modified:report.modified,lifecycleMs:run.elapsed_ms,sampledRss:null,binaryBytes:null,exitCode:run.exit_code,cases});
      }
    }catch{/* Missing evidence stays unavailable. */}
  }
  return campaigns.sort((a,b)=>b.modified-a.modified);
}

export async function artifactFiles(dataRoot,id) {
  const folder=path.join(dataRoot,'runs',id,'artifacts');let root;
  try{const stat=await fs.lstat(folder);if(!stat.isDirectory()||stat.isSymbolicLink())return [];root=await fs.realpath(folder);}catch{return [];}
  const realRoot=await fs.realpath(dataRoot),scope=path.relative(realRoot,root),files=[];
  if(scope.startsWith('..')||path.isAbsolute(scope))return [];
  const allowed=/^[a-zA-Z0-9_.-]+\.(csv|tsv|txt|axevo|axmath|axstruct|axcont|axnum|axreact|axdna|axgoal|axbehavior|axtext|axseed|axsem|axstudy|axep)$/;
  async function walk(directory,depth){
    const entries=await fs.readdir(directory,{withFileTypes:true});
    // Publish this directory's seeds before nested births consume the bounded catalog.
    for(const entry of entries.sort((a,b)=>Number(b.isFile())-Number(a.isFile())||a.name.localeCompare(b.name)).slice(0,2048)){
      if(files.length>=1024)return;
      if(entry.isSymbolicLink()||!/^[-a-zA-Z0-9_.]+$/.test(entry.name))continue;
      const file=path.join(directory,entry.name),real=await fs.realpath(file),relative=path.relative(root,real);
      if(relative.startsWith('..')||path.isAbsolute(relative))continue;
      if(entry.isDirectory()){if(depth<2)await walk(file,depth+1);continue;}
      if(!entry.isFile()||(!allowed.test(entry.name)&&!/^audit[-a-zA-Z0-9]*\.json$/.test(entry.name)&&!(/^(curriculum|prepared|curriculum-results|model-index|[a-z0-9_-]+-(result|predictions))\.json$/.test(entry.name))))continue;
      const stat=await fs.lstat(file);if(stat.size<=16000000)files.push({name:path.relative(folder,file).replaceAll('\\','/'),bytes:stat.size});
    }
  }
  await walk(folder,0);
  return files.sort((a,b)=>a.name.localeCompare(b.name));
}

/** @param {{root?: string, port?: number, autoSnapshot?: boolean, storageRoot?: string}} [options] */
export async function createObservatory({ root = path.resolve(HERE, '..'), port = 4317, autoSnapshot = true, storageRoot } = {}) {
  const sources = await catalogFiles(root);
  const dataRoot = storageRoot ?? path.join(root, '.axon-memory', 'observatory');
  const memory = path.join(dataRoot, 'project.axm');
  const extension = process.platform === 'win32' ? '.exe' : '';
  const cli = path.join(root, 'target', 'release', `axon${extension}`);
  const observer = path.join(root, 'target', 'release', `axon-observer${extension}`);
  const nonce = randomBytes(24).toString('hex');
  const session = randomUUID();
  const clients = new Set();
  const events = [];
  const jobs = new Map();
  const queryMemory = new QueryMemory();
  const delivery = []; let deliveryTimer = null;
  function deliver() {
    deliveryTimer = null;
    const records = delivery.splice(0);
    for (let offset = 0; offset < records.length; offset += 64) {
      const batch = records.slice(offset, offset + 64);
      const packet = `id: ${batch.at(-1).seq}\ndata: ${JSON.stringify({event:'batch',records:batch})}\n\n`;
      for (const client of clients) {
        if (client.writableLength > 8_000_000) { client.destroy(); clients.delete(client); continue; }
        client.write(packet);
      }
    }
  }
  let sequence = 0;
  let graph = { nodes: [], edges: [], models: [] };
  let current = null;
  let starting = false;
  let stopping = false;
  let actualPort = port;
  const publicJob = job => ({ id: job.id, mode: job.mode, status: job.status, started: job.started,
    finished: job.finished, options: job.options, trace: job.trace, counters: job.counters,
    latest: job.latest, result: job.result, output: job.output, progress: job.progress, cases: job.cases,
    samples: job.samples, algorithmInvestigation:job.algorithmInvestigation, algorithmDrift:job.algorithmDrift, algorithmInquiry:job.algorithmInquiry, candidates:job.candidates??[], workflow: job.workflow, exitCode: job.exitCode, recovered: job.recovered ?? false });
  const state = () => ({ token: nonce, session, port: actualPort, coreReady: existsSync(cli) && existsSync(observer),
    indexed: existsSync(memory), sources, methods: METHODS.map(([id, name, file]) => ({ id, name, file })),
    graph, memory: [...queryMemory.nodes.values()], outputs: [...queryMemory.outputs.values()], current: current && publicJob(current),
    runs: [...jobs.values()].map(publicJob).reverse(), workflows: workflows.map(w=>({...w,available:w.binary?existsSync(path.join(root,'target','release',`${w.binary}${extension}`)):true})),
    events: events.slice(-160), streamSequence: sequence });
  function send(event) {
    const item = { ...event, session, seq: ++sequence, time: Date.now(), run: Object.hasOwn(event, 'run') ? event.run : current?.id ?? null };
    if (item.event === 'graph') graph = { nodes: item.nodes, edges: item.edges, models: item.models, phase: item.phase };
    if (item.run === current?.id) queryMemory.accept(item);
    if (current && item.run === current.id && item.event === 'control') current.status = item.status;
    const owner = jobs.get(item.run);
    const logged = owner?.logStream && !owner.logStream.writableEnded && !owner.trace.logError;
    if (logged) {
      owner.trace.records++;
      if(item.method)owner.trace.methods[item.method]=(owner.trace.methods[item.method]??0)+1;
      if(item.coreSeq!==undefined){if(item.coreSeq!==owner.trace.lastCoreSeq+1)owner.trace.gaps++;owner.trace.lastCoreSeq=item.coreSeq;}
      if(item.telemetry)owner.trace.mode=item.telemetry;
      item.coreGaps = owner.trace.gaps;
    }
    const summary = item.event === 'graph' ? { event: 'graph', seq: item.seq, time: item.time, run: item.run,
      detail: `${item.nodes.length} predictive states / ${item.edges.length} transitions` } : item;
    events.push(summary);
    if (events.length > 750) events.shift();
    const data = JSON.stringify(item);
    if(clients.size){delivery.push(item);if(!deliveryTimer)deliveryTimer=setTimeout(deliver,16);}
    if (owner) {
      if (item.experiment) {
        owner.candidates=mergeSeedCandidates(owner.candidates,item);
        owner.progress = item;
        if(item.event==='algorithm_investigation')owner.algorithmInvestigation=item;
        if(item.event==='algorithm_investigation'&&['query_selected','query_observed'].includes(item.phase))owner.algorithmInquiry=item;
        if(item.event==='algorithm_drift')owner.algorithmDrift=item;
        if (Number.isInteger(item.case)) {
          const before=owner.cases.find(c=>c.case===item.case)??{case:item.case};
          const next={...before,...item};
          owner.cases=owner.cases.filter(c=>c.case!==item.case);owner.cases.push(next);owner.cases.sort((a,b)=>a.case-b.case);
        }
        if (['evolution_progress','continuous_progress','controller_feedback','study_query','validation','plan','prediction','seed_selection','genome','nursery_progress','dna_growth','investigation','agent_result','algorithm_investigation','algorithm_drift','algorithm_result'].includes(item.event)) {
          owner.samples.push(item);if(owner.samples.length>256)owner.samples.shift();
        }
        if (item.event==='lab_result') owner.result=item;
      }
      if (item.event === 'query') { owner.counters.queryCalls++; owner.counters[item.cacheHit ? 'cached' : 'physical']++; }
      if (item.event === 'step') owner.counters.steps++;
      if (item.event === 'error' || item.event === 'query_error') owner.counters.errors++;
      for (const key of ['queries','latencyNs','cacheNodes','hypotheses','initialHypotheses','residual']) {
        if (typeof item[key] === 'number') owner.latest[key] = item[key];
      }
      if (item.event === 'result') owner.result = item;
      if (['output','process_output','test_output','process_stderr'].includes(item.event)) owner.output = `${owner.output}${item.text ?? ''}\n`.slice(-128000);
    }
    if (logged) {
      owner.logBytes += Buffer.byteLength(data) + 1;
      if(owner.logBytes>256_000_000&&!owner.trace.limit){owner.trace.limit=true;stopWorker(owner.process);send({event:'error',run:owner.id,detail:'Complete trace exceeded 256 MB; task stopped, not silently sampled.'});}
      const writable=owner.logStream.write(`${data}\n`);
      if(!writable&&owner.process?.stdout&&!owner.draining){owner.draining=true;const input=owner.process.stdout;input.pause();owner.logStream.once('drain',()=>{owner.draining=false;input.resume();});}
    }
    return item;
  }
  async function start(options) {
    if (starting) throw new Error('A task is already starting');
    if (!existsSync(cli) || !existsSync(observer)) throw new Error('Build axon and axon-observer release binaries first');
    if (current && (['running', 'paused', 'pausing', 'starting'].includes(current.status) || current.process?.exitCode===null)) throw new Error('A task is already active');
    if (options.mode === 'knowledge' && !existsSync(memory)) throw new Error('Index local sources first');
    starting = true;
    try {
    await fs.mkdir(path.join(dataRoot, 'runs'), { recursive: true });
    await fs.mkdir(path.join(dataRoot, 'events'), { recursive: true });
    const id = randomUUID();
    const directory = path.join(dataRoot, 'runs', id);
    await fs.mkdir(directory);
    const output = path.join(directory, 'artifacts');
    const job = { id, mode: options.mode, options, started: Date.now(), finished: null, status: options.paused ? 'paused' : 'running',
      process: null, logPath: path.join(dataRoot, 'events', `${id}.jsonl`), logBytes: 0, cancelled: false,
      counters: {physical:0,cached:0,queryCalls:0,steps:0,errors:0}, latest: {}, result: null, output: '',
      progress:null,cases:[],samples:[],candidates:[],workflow:workflows.find(w=>w.id===options.mode)??null,exitCode:null,
      trace:{mode:'entry-exit-only',records:0,lastCoreSeq:0,gaps:0,logError:null,limit:false,methods:Object.create(null)} };
    job.logStream=createWriteStream(job.logPath,{flags:'wx'});
    job.log=new Promise(resolve=>{job.logStream.once('finish',resolve);job.logStream.once('error',error=>{job.trace.logError=error.message;job.status='error';stopWorker(job.process);send({event:'error',run:job.id,status:'error',trace:job.trace,detail:`Trace storage failed: ${error.message}`});resolve();});});
    job.persist=async()=>{await job.log;if(!job.trace.logError){const file=await fs.open(job.logPath,'r+');try{await file.sync();}finally{await file.close();}}await saveRun(directory,publicJob(job));};
    try{await saveRun(directory,publicJob(job));}catch(error){job.logStream.end();throw error;}
    queryMemory.clear();
    graph={nodes:[],edges:[],models:[],phase:'collecting'};
    current = job; jobs.set(id, job);
    if (jobs.size > 32) jobs.delete(jobs.keys().next().value);
    const publish = event => send({ ...event, run: job.id });
    publish({ event: 'job', status: job.status, mode: job.mode, detail: `begin ${job.mode}`, method: job.workflow ? 'executeWorkflow' : job.mode === 'knowledge' ? 'KnowledgeStore::investigate' : job.mode === 'math' ? 'solve_task' : 'observer' });
    if (job.workflow) {
      job.trace.mode=job.workflow.telemetry;
      void executeWorkflow({root,output,job,options,publish}).then(code=>{job.exitCode=code;job.status=job.cancelled?'cancelled':code===0&&!job.trace.logError&&!job.trace.limit?'complete':'error';})
        .catch(error=>{job.status=job.cancelled?'cancelled':'error';publish({event:'error',detail:error.message});})
        .finally(async()=>{job.finished=Date.now();publish({event:'end',status:job.status,exitCode:job.exitCode,elapsedMs:job.finished-job.started,trace:{...job.trace,records:job.trace.records+1}});job.logStream.end();await job.persist().catch(error=>send({event:'error',run:job.id,detail:`Ledger: ${error.message}`}));});
    } else if (['factor', 'transfer'].includes(options.mode)) {
      const child = spawn(observer, [options.mode, `${options.seed}`, `${options.dimensions}`, options.novel ? '1' : '0', options.paused ? '1' : '0', output],
        { cwd: root, windowsHide: true, shell: false, stdio: ['pipe', 'pipe', 'pipe'] });
      job.process = child;
      child.stdin.on('error', () => {});
      let pending = '';
      child.stdout.setEncoding('utf8');
      child.stdout.on('data', chunk => {
        pending += chunk;
        if (pending.length > 512000) { child.kill(); publish({ event: 'error', detail: 'Observer message limit' }); return; }
        let index;
        while ((index = pending.indexOf('\n')) !== -1) {
          const line = pending.slice(0, index).trim(); pending = pending.slice(index + 1);
          if (!line) continue;
          try { send({ ...JSON.parse(line), run: job.id }); } catch { send({ event: 'error', detail: 'Invalid observer record', run: job.id }); }
        }
      });
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', data => send({ event: 'error', detail: data.slice(0, 4000), run: job.id }));
      child.on('error', error => { job.status = 'error'; send({ event: 'error', detail: error.message, run: job.id }); });
      child.on('close', code => { job.finished = Date.now(); job.status = job.cancelled ? 'cancelled' : code === 0&&!job.trace.logError&&!job.trace.limit ? 'complete' : 'error';
        job.exitCode=code;send({ event: 'end', status: job.status, elapsedMs: job.finished - job.started, run: job.id, trace:{...job.trace,records:job.trace.records+1} }); job.logStream.end();void job.persist().catch(error=>send({event:'error',run:job.id,detail:`Ledger: ${error.message}`})); });
    } else {
      const runCli = async args => {
        const child = spawn(cli, args, { cwd: root, windowsHide: true, shell: false, stdio: ['ignore', 'pipe', 'pipe'] });
        job.process = child;
        let result = ''; let errors = '';
        return new Promise((resolve, reject) => {
          child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
          child.stdout.on('data', text => { result += text; if (result.length > 1000000) child.kill(); });
          child.stderr.on('data', text => { errors += text; });
          child.on('error', reject);
          child.on('close', code => code === 0 ? resolve(result) : reject(new Error(errors.slice(0, 8000) || 'Task interrupted')));
        });
      };
      (async () => {
        if (options.mode === 'index') {
          for (const folder of ['docs', 'src', 'tests']) {
            if (job.cancelled) break;
            publish({ event: 'stage', method: 'KnowledgeStore::learn', detail: `index ${folder}` });
            const output = await runCli(['learn', path.join(root, folder), '--memory', memory]);
            publish({ event: 'output', method: 'KnowledgeStore::learn', text: output, detail: `indexed ${folder}` });
          }
        } else if (options.mode === 'knowledge') {
          const text = await runCli(['investigate', options.query, '--memory', memory, '--limit', '4']);
          publish({ event: 'output', method: 'KnowledgeStore::investigate', text, detail: options.query });
        } else {
          await fs.mkdir(output);
          const task = path.join(output, 'task.axon');
          await fs.writeFile(task, options.task, { flag: 'wx' });
          const text = await runCli(['solve', task, '--artifact-dir', path.join(output, 'artifacts')]);
          publish({ event: 'output', method: 'solve_task', text, detail: 'exact structured task result' });
        }
        job.status = job.cancelled ? 'cancelled' : 'complete';
      })().catch(error => { job.status = job.cancelled ? 'cancelled' : 'error'; publish({ event: 'error', detail: error.message }); })
        .finally(async() => { job.finished = Date.now();job.exitCode=job.status==='complete'?0:null; publish({ event: 'end', status: job.status, elapsedMs: job.finished - job.started, trace:{...job.trace,records:job.trace.records+1} });job.logStream.end();await job.persist().catch(error=>send({event:'error',run:job.id,detail:`Ledger: ${error.message}`})); });
    }
    return { id, status: job.status };
    } finally { starting = false; }
  }
  const server = http.createServer(async (req, res) => {
    const json = (code, value) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(value)); };
    const hosts = [`127.0.0.1:${actualPort}`, `localhost:${actualPort}`];
    if (!hosts.includes(req.headers.host)) return json(403, { error: 'Localhost only' });
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' blob: data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      if (req.method === 'GET' && url.pathname === '/api/state') return json(200, state());
      if (req.method === 'GET' && url.pathname === '/api/evolution') return json(200, { campaigns: await evolutionCatalog(root) });
      if (req.method === 'GET' && url.pathname === '/api/events') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
        res.write(`data: ${JSON.stringify({ event: 'connected', session, status: current?.status ?? 'idle' })}\n\n`);
        clients.add(res); req.on('close', () => clients.delete(res)); return;
      }
      if (req.method === 'GET' && url.pathname === '/api/source') {
        const raw = url.searchParams.get('id');
        if (!/^\d+$/.test(raw ?? '')) return json(400, { error: 'Invalid source id' });
        const source = sources[Number(raw)];
        if (!source) return json(404, { error: 'Source not found' });
        const full = await fs.realpath(path.join(root, source.path));
        const relative = path.relative(await fs.realpath(root), full);
        if (relative.startsWith('..') || path.isAbsolute(relative)) return json(403, { error: 'Source outside workspace' });
        const limit = source.path === 'docs/AXON.md' ? 1048576 : 128000;
        if ((await fs.stat(full)).size > limit) return json(413, { error: 'Source exceeds inspection limit' });
        return json(200, { ...source, text: await fs.readFile(full, 'utf8') });
      }
      if (req.method === 'GET' && url.pathname === '/api/export') {
        const id = url.searchParams.get('id'); const job = jobs.get(id);
        if (!job) return json(404, { error: 'Run not found' });
        if(job.finished||!job.logStream)await job.log;else await new Promise(resolve=>job.logStream.write('',resolve));
        if (!existsSync(job.logPath)) return json(404, { error: 'No log available yet' });
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Content-Disposition': `attachment; filename="axon-${id}.jsonl"` });
        const file = createReadStream(job.logPath); file.on('error', () => res.destroy()); file.pipe(res); return;
      }
      if(req.method==='GET'&&url.pathname==='/api/artifacts') {
        const job=jobs.get(url.searchParams.get('id'));if(!job)return json(404,{error:'Run not found'});
        return json(200,{files:await artifactFiles(dataRoot,job.id)});
      }
      if(req.method==='GET'&&url.pathname==='/api/artifact') {
        const job=jobs.get(url.searchParams.get('id'));if(!job)return json(404,{error:'Run not found'});
        const files=await artifactFiles(dataRoot,job.id),entry=files.find(f=>f.name===url.searchParams.get('name'));
        if(!entry)return json(404,{error:'Artifact not found'});
        res.writeHead(200,{'Content-Type':'application/octet-stream','Content-Length':entry.bytes,'Content-Disposition':`attachment; filename="${entry.name}"`});
        const file=createReadStream(path.join(dataRoot,'runs',job.id,'artifacts',entry.name));file.on('error',()=>res.destroy());file.pipe(res);return;
      }
      if (req.method === 'POST' && ['/api/run', '/api/control'].includes(url.pathname)) {
        const origin = req.headers.origin;
        if ((origin && !hosts.some(host => origin === `http://${host}`)) || req.headers['x-axon-session'] !== nonce) return json(403, { error: 'Invalid local session' });
        if (!req.headers['content-type']?.startsWith('application/json')) return json(415, { error: 'JSON required' });
        let body = ''; for await (const chunk of req) { body += chunk; if (Buffer.byteLength(body) > 40000) return json(413, { error: 'Request too large' }); }
        let data; try { data = JSON.parse(body); } catch { return json(400, { error: 'Invalid JSON' }); }
        if (url.pathname === '/api/run') return json(200, await start(validateRun(data)));
        if (!data || Object.keys(data).length !== 1 || !['pause', 'resume', 'step', 'cancel'].includes(data.command)) return json(400, { error: 'Invalid control' });
        if (!current?.process || !['running', 'paused', 'pausing'].includes(current.status)) return json(409, { error: 'No active task' });
        if (data.command === 'cancel') { current.cancelled = true; current.process.stdin?.write('stop\n');
          const child = current.process; setTimeout(() => { if (child.exitCode === null) stopWorker(child); }, 500).unref();
        } else if (['factor', 'transfer'].includes(current.mode)) {
          current.process.stdin.write(`${data.command}\n`);
          if (data.command === 'pause') current.status = 'pausing';
        } else return json(409, { error: 'This task does not support stepping' });
        return json(200, { status: current.status });
      }
      if (req.method === 'GET' && (['/', '/index.html'].includes(url.pathname) || /^\/assets\/[\w.-]+\.(js|css)$/.test(url.pathname))) {
        const file = path.join(DIST, url.pathname.startsWith('/assets/') ? url.pathname.slice(1) : 'index.html');
        if (!existsSync(file)) return json(503, { error: 'Build the React workspace first: cd ui && bun run build' });
        const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html';
        const stat = await fs.stat(file);
        res.writeHead(200, { 'Content-Type': `${type}; charset=utf-8`, 'Content-Length': stat.size }); createReadStream(file).pipe(res); return;
      }
      json(404, { error: 'Not found' });
    } catch (error) { if (!res.headersSent) json(400, { error: error.message }); else res.destroy(); }
  });
  const restored=await restoreRuns(dataRoot);
  for(const item of restored.runs){const job={...item,logPath:path.join(dataRoot,'events',`${item.id}.jsonl`),log:Promise.resolve(),recovered:true};jobs.set(job.id,job);}
  if(restored.current){current=jobs.get(restored.current.id);graph=restored.graph;queryMemory.restore(restored.memory);queryMemory.outputs=new Map(restored.outputs.map(o=>[o.id,o]));events.push(...restored.events);}
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
  actualPort = server.address().port;
  if (autoSnapshot && existsSync(observer)) {
    const bundle = path.join(root, '.ecc/benchmarks/factor-resource-project-demo-20261013-v1.outputs/models/factors.axseed');
    if (existsSync(bundle)) exec(observer, ['snapshot', bundle, 'demo/factor/workflow-v1/5'], { cwd: root, windowsHide: true, maxBuffer: 1000000 })
      .then(({ stdout }) => { if (current) return; for (const line of stdout.trim().split('\n')) { try { send({ ...JSON.parse(line), run: null }); } catch {} } })
      .catch(error => send({ event: 'error', detail: `Snapshot: ${error.message}` }));
  }
  const heartbeat = setInterval(() => { for (const client of clients) client.write(': keepalive\n\n'); }, 15000); heartbeat.unref();
  return { server, url: `http://127.0.0.1:${actualPort}`, state, async close() {
    stopping = true; clearInterval(heartbeat); if(deliveryTimer)clearTimeout(deliveryTimer); if(current?.process)stopWorker(current.process);
    for(const job of jobs.values())if(job.logStream&&!job.logStream.writableEnded)job.logStream.end();
    for (const client of clients) client.end(); clients.clear();
    await new Promise(resolve => server.close(resolve));
  } };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.AXON_UI_PORT || process.argv[2] || 4317);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid port');
  const storageRoot=process.env.AXON_UI_EPHEMERAL==='1'?await fs.mkdtemp(path.join(os.tmpdir(),'axon-ui-preview-')):undefined;
  const app = await createObservatory({ port, storageRoot });
  console.log(`AXON Workspace ${app.url}`);
  const stop = async () => { await app.close(); process.exit(0); };
  process.on('SIGINT', stop); process.on('SIGTERM', stop);
}
