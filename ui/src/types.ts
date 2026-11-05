export type Mode =
  | 'transfer'
  | 'factor'
  | 'knowledge'
  | 'math'
  | 'index'
  | 'evolution'
  | 'reactive'
  | 'continuous'
  | 'seed'
  | 'controller'
  | 'genome'
  | 'language'
  | 'cumulative'
  | 'semantic'
  | 'nursery'
  | 'constructive'
  | 'curriculum'
  | 'integration'
  | 'structural'
  | 'tests';
export type Status =
  | 'running'
  | 'paused'
  | 'pausing'
  | 'starting'
  | 'complete'
  | 'error'
  | 'cancelled'
  | 'interrupted';
export interface GraphNode {
  id: string;
  factor: number;
  state: number;
  context: string;
}
export interface GraphEdge {
  source: string;
  target: string;
  action: number;
  output: number;
  factor: number;
}
export interface Model {
  name: string;
  states: number;
  bytes: number;
  context: string;
  rows: string[];
}
export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  models: Model[];
  phase?: string;
}
export interface RecordEvent {
  event: string;
  seq: number;
  session: string;
  time: number;
  run: string | null;
  method?: string;
  detail?: string;
  status?: Status;
  mode?: Mode;
  telemetry?: string;
  coreSeq?: number;
  coreGaps?: number;
  coreNs?: number;
  unixMs?: number;
  latencyNs?: number;
  queries?: number;
  calls?: number;
  cacheNodes?: number;
  cacheHit?: boolean;
  hypotheses?: number;
  initialHypotheses?: number;
  residual?: number;
  exact?: boolean;
  checkedEdges?: number;
  states?: number;
  bytes?: number;
  actions?: number[];
  reply?: number[];
  frames?: number[][];
  context?: string;
  action?: number;
  fromStates?: number[];
  toStates?: number[];
  nodes?: GraphNode[];
  edges?: GraphEdge[];
  models?: Model[];
  phase?: string;
  stage?: string;
  text?: string;
  trace?: Trace;
  elapsedMs?: number;
  experiment?: string;
  case?: number;
  channel?: number;
  outcome?: string;
  generation?: number;
  evaluations?: number;
  templates?: number;
  operation?: string;
  mining_work?: number;
  library_programs?: number;
  library_windows?: number;
  inquiry?: string;
  last_inquiry?: string;
  query_count?: number;
  query_hypotheses?: number;
  query_survivors?: number;
  invalidated?: number;
  work?: number;
  language_work?: number;
  session_correct?: number;
  session_total?: number;
  file_correct?: number;
  file_total?: number;
  plans_json?: string;
  actual_json?: string | null;
  meanings?: number;
  grounded?: number;
  evidence?: number[];
  turn?: number;
  loss?: number;
  local_work?: number;
  truncated?: number;
  diversity?: number;
  active_genes?: number;
  instructions?: number;
  capsule_bytes?: number;
  dependencies?: number;
  correct?: number;
  total?: number;
  dialogue_correct?: number;
  dialogue_turns?: number;
  derived_programs?: number;
  total_rules?: number;
  agent_correct?: number;
  agent_total?: number;
  algorithm_correct?: number;
  algorithm_total?: number;
  agent_portable?: number;
  portable_correct?: number;
  variant?: string;
  refutations?: number;
  acquisitions?: number;
  programs?: string[];
  heldout?: number;
  scored?: number;
  observation?: number;
  observations?: number;
  promotions?: number;
  drifts?: number;
  reuses?: number;
  passed?: number | boolean;
  tasks?: number;
  goal_tasks?: number;
  domain?: string;
  transfer_correct?: number;
  transfer_total?: number;
  goal_admitted?: number;
  all_goals_passed?: boolean;
  expression?: string;
  wiring?: number[];
  arity?: number;
  output_signal?: number;
  identity?: string;
  exitCode?: number | null;
  candidate?: number;
  genotype?: string;
  origin_revision?: string;
  revision?: string;
  parent_a?: string;
  parent_b?: string;
  solved?: number;
  modules?: number;
  retention_ok?: boolean;
  violations?: number;
  frontier?: boolean;
  name?: string;
  survivors?: number;
  capabilities?: number;
  steps?: string[];
  physical_correct?: number;
  baseline_correct?: number;
  predicted?: number | null;
  actual?: number | null;
  peak_bytes?: number;
  peak_modules?: number;
  selected?: number[];
  candidates?: number;
  artifact?: string;
  attempts?: number;
  acquired?: number;
  population?: number;
  genes?: number;
  mutations?: number;
  refinement?: boolean;
  bindings?: boolean;
  refused?: number;
  wrong?: number;
  negative_pass?: number;
  dictionary_facts?: number;
  arm?: string;
  task?: number;
  novel?: boolean;
  admitted?: boolean | number;
  borrowed?: boolean;
  questions?: number;
  probe_work?: number;
  search_work?: number | null;
  disagreement?: number;
  skills?: number;
  encoded_bytes?: number;
  seed_digits?: number;
  unavailable?: number;
  domain_refusals?: number;
  checked?: number;
  file_plans?: number;
  item?: string;
  needed?: number;
  units?: number;
  cost_cents?: number;
  budget_cents?: number;
  choices?: string;
  stock_hash?: string;
  offers_hash?: string;
  model?: string;
  episode?: number;
  optimizer_nodes?: number;
  optimizer_iterations?: number;
  family?: string;
  sentence?: string;
  unique?: number;
  random_best?: number;
  natural_correct?: number;
  natural_checked?: number;
  negative_correct?: number;
  negative_checked?: number;
  new_candidates?: number;
  previous_candidates?: number;
  growths?: number;
  unfinished?: number;
  imported?: number;
}
export interface Workflow {
  id: Mode;
  label: string;
  kind: string;
  telemetry: string;
  available: boolean;
}
export interface Trace {
  mode: string;
  records: number;
  lastCoreSeq: number;
  gaps: number;
  methods: Record<string, number>;
  logError?: string | null;
}
export interface Counters {
  physical: number;
  cached: number;
  queryCalls: number;
  steps: number;
  errors: number;
}
export interface Job {
  id: string;
  mode: Mode;
  status: Status;
  started: number;
  finished: number | null;
  trace: Trace;
  options?: RunOptions;
  counters: Counters;
  latest: Record<string, number>;
  result: RecordEvent | null;
  output: string;
  progress: RecordEvent | null;
  cases: RecordEvent[];
  samples: RecordEvent[];
  algorithmInvestigation?: RecordEvent;
  algorithmInquiry?: RecordEvent;
  algorithmDrift?: RecordEvent;
  candidates: RecordEvent[];
  workflow: Workflow | null;
  exitCode: number | null;
  recovered?: boolean;
}
export interface Source {
  id: number;
  path: string;
  bytes: number;
  kind: string;
}
export interface MemoryNode {
  id: string;
  parent: string | null;
  depth: number;
  context: string;
  root?: boolean;
  active?: boolean;
  output: number[];
}
export interface Output {
  id: string;
  context: string;
  output: number;
  physical: number;
  cached: number;
  executed: number;
}
export interface Snapshot {
  token: string;
  session: string;
  port: number;
  coreReady: boolean;
  indexed: boolean;
  sources: Source[];
  methods: { id: string; name: string; file: string }[];
  graph: Graph;
  memory: MemoryNode[];
  outputs: Output[];
  current: Job | null;
  runs: Job[];
  events: RecordEvent[];
  streamSequence: number;
  workflows: Workflow[];
}
export interface RunOptions {
  mode: Mode;
  seed?: number;
  dimensions?: number;
  novel?: boolean;
  paused?: boolean;
  query?: string;
  task?: string;
  count?: number;
  strategy?:
    | 'balanced'
    | 'hybrid'
    | 'mutation-only'
    | 'no-inheritance'
    | 'elitist'
    | 'diverse'
    | 'reinforced';
  curriculum?: string;
  suite?: 'rust' | 'rust-minimal' | 'ui' | 'check';
  validation?: 'development' | 'reserved';
}
export interface EvolutionCase {
  case: number;
  status: string;
  generation: number;
  evaluations: number;
  work: number;
  active_genes: number | null;
  instructions: number | null;
  dependencies: number | null;
  capsule_bytes: number | null;
  correct: number;
  heldout: number;
  training_ns: number | null;
  local_proposals: number | null;
  residual_errors: number | null;
  expression: string;
}
export interface Campaign {
  id: string;
  seed: number;
  mode: string;
  modified: number;
  lifecycleMs: number;
  sampledRss: number | null;
  binaryBytes: number | null;
  exitCode: number;
  cases: EvolutionCase[];
}
