import { useCallback, useEffect, useRef, useState } from 'react';
import { QueryMemory, mergeSeedCandidates } from '../trace.js';
import type { Snapshot, RecordEvent, RunOptions, Job, Campaign, Mode, Status } from './types';

export const busy = (job: Job | null) =>
  !!job && ['running', 'paused', 'pausing', 'starting'].includes(job.status);
export const fmt = (n: number | undefined | null) =>
  n == null ? '--' : new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 2 }).format(n);
export const bytes = (n: number) =>
  n < 1024 ? `${fmt(n)} B` : n < 1048576 ? `${fmt(n / 1024)} KB` : `${fmt(n / 1048576)} MB`;
export const latency = (ns: number | undefined) =>
  ns == null || !Number.isFinite(ns) || ns < 0
    ? '--'
    : ns < 1000
      ? `${fmt(ns)} ns`
      : ns < 1000000
        ? `${fmt(ns / 1000)} us`
        : `${fmt(ns / 1000000)} ms`;
export const label: Record<Mode | Status, string> = {
  transfer: 'Transferencia',
  factor: 'Composicao',
  knowledge: 'Investigacao',
  math: 'Tarefa exata',
  index: 'Indexacao',
  evolution: 'Evolucao',
  seed: 'Seeds herdaveis',
  controller: 'Objetivos e procedimentos',
  genome: 'Genoma e nascimento',
  language: 'Linguagem e matematica',
  cumulative: 'Aprendizagem cumulativa',
  semantic: 'Texto, mundo e matematica / R1',
  nursery: 'Viveiro de seeds',
  constructive: 'DNA construtivo',
  curriculum: 'Curriculo JSON',
  integration: 'Heranca e fluxo integrado',
  structural: 'Estruturas, dialogo e calculo',
  reactive: 'Aprendizagem reativa',
  continuous: 'Aprendizagem continua',
  tests: 'Testes',
  interrupted: 'Interrompido',
  running: 'Executando',
  paused: 'Pausado',
  pausing: 'Pausando',
  complete: 'Concluido',
  cancelled: 'Cancelado',
  error: 'Erro',
  starting: 'Iniciando',
};
export function describe(e: RecordEvent) {
  if (e.detail) return e.detail;
  if (e.event === 'query')
    return `[${e.actions?.join(', ')}] -> [${(e.frames?.at(-1) ?? e.reply)?.join(', ')}]`;
  if (e.event === 'step') return `acao ${e.action} -> [${e.reply?.join(', ')}]`;
  if (e.event === 'result')
    return `${fmt(e.checkedEdges)} transicoes verificadas / residual ${e.residual ?? '--'}`;
  if (e.experiment)
    return `${e.event} / ${e.case != null ? `caso ${e.case}` : e.channel != null ? `canal ${e.channel}` : e.experiment} / ${e.outcome ?? e.phase ?? fmt(e.work)}`;
  return e.method ?? e.event;
}

export function preserveObservedAlgorithmEvidence(snapshot: Snapshot, previous: Snapshot | null) {
  const before = previous?.current,
    current = snapshot.current;
  if (!before || !current || snapshot.session !== previous?.session || current.id !== before.id)
    return;
  for (const key of ['algorithmInvestigation', 'algorithmDrift', 'algorithmInquiry'] as const) {
    const event = before[key];
    // An older backend may omit these fields; never carry evidence across runs or into the future.
    if (
      current[key] === undefined &&
      event?.coreSeq != null &&
      event.coreSeq <= current.trace.lastCoreSeq
    ) {
      current[key] = event;
    }
  }
}

export function useAxon() {
  const [data, setData] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [sending, setSending] = useState(false);
  const [transportGaps, setTransportGaps] = useState(0);
  const ref = useRef<Snapshot | null>(null);
  const refreshRef = useRef<() => void>(() => {});

  useEffect(() => {
    let alive = true,
      syncing = false,
      timer: ReturnType<typeof setTimeout> | undefined;
    let pending: RecordEvent[] = [];
    let reconnecting = false;
    const memory = new QueryMemory();
    const flush = () => {
      timer = undefined;
      if (!alive || syncing || !ref.current) return;
      let s = ref.current;
      let current = s.current
        ? {
            ...s.current,
            counters: { ...s.current.counters },
            latest: { ...s.current.latest },
            trace: { ...s.current.trace, methods: { ...s.current.trace.methods } },
          }
        : null;
      let graph = s.graph,
        events = [...s.events],
        sequence = s.streamSequence;
      for (const e of pending.splice(0)) {
        if (e.session !== s.session || e.seq <= sequence) continue;
        if (e.seq !== sequence + 1) setTransportGaps((n) => n + 1);
        sequence = e.seq;
        if (e.event === 'job') {
          memory.clear();
          graph = { nodes: [], edges: [], models: [], phase: 'collecting' };
          events = [];
          current = {
            id: e.run!,
            mode: e.mode!,
            status: e.status!,
            started: e.time,
            finished: null,
            trace: { mode: 'entry-exit-only', records: 0, lastCoreSeq: 0, gaps: 0, methods: {} },
            counters: { physical: 0, cached: 0, queryCalls: 0, steps: 0, errors: 0 },
            latest: {},
            result: null,
            output: '',
            progress: null,
            cases: [],
            samples: [],
            candidates: [],
            workflow: s.workflows.find((w) => w.id === e.mode) ?? null,
            exitCode: null,
          };
        }
        if (e.run && current && e.run !== current.id) continue;
        if (e.event === 'graph')
          graph = { nodes: e.nodes!, edges: e.edges!, models: e.models!, phase: e.phase };
        if (current && e.run === current.id) {
          memory.accept(e);
          current.trace.records++;
          if (e.method)
            current.trace.methods[e.method] = (current.trace.methods[e.method] ?? 0) + 1;
          if (e.coreSeq != null) current.trace.lastCoreSeq = e.coreSeq;
          if (e.coreGaps != null) current.trace.gaps = e.coreGaps;
          if (e.telemetry) current.trace.mode = e.telemetry;
          if (e.event === 'algorithm_investigation') current.algorithmInvestigation = e;
          if (
            e.event === 'algorithm_investigation' &&
            ['query_selected', 'query_observed'].includes(e.phase ?? '')
          )
            current.algorithmInquiry = e;
          if (e.event === 'algorithm_drift') current.algorithmDrift = e;
          if (e.event === 'query') {
            current.counters.queryCalls++;
            current.counters[e.cacheHit ? 'cached' : 'physical']++;
          }
          if (e.event === 'step') current.counters.steps++;
          if (['error', 'query_error'].includes(e.event)) current.counters.errors++;
          for (const key of [
            'queries',
            'latencyNs',
            'cacheNodes',
            'hypotheses',
            'initialHypotheses',
            'residual',
          ] as const)
            if (e[key] != null) current.latest[key] = e[key]!;
          if (['result', 'lab_result'].includes(e.event)) current.result = e;
          if (['output', 'process_output', 'test_output', 'process_stderr'].includes(e.event))
            current.output = `${current.output}${e.text ?? ''}\n`.slice(-128000);
          if (e.experiment) {
            current.candidates = mergeSeedCandidates(current.candidates, e);
            current.progress = e;
            if (e.case != null) {
              const before = current.cases.find((c) => c.case === e.case);
              current.cases = [
                ...current.cases.filter((c) => c.case !== e.case),
                { ...before, ...e },
              ].sort((a, b) => a.case! - b.case!);
            }
          }
          if (
            [
              'evolution_progress',
              'continuous_progress',
              'controller_feedback',
              'study_query',
              'validation',
              'plan',
              'prediction',
              'seed_selection',
              'genome',
              'nursery_progress',
              'dna_growth',
            ].includes(e.event)
          )
            current.samples = [...current.samples, e].slice(-256);
          if (e.status && ['job', 'start', 'control', 'end', 'error'].includes(e.event))
            current.status = e.status;
          if (e.event === 'end') {
            current.finished = e.time;
            if (e.trace) current.trace = e.trace;
            current.exitCode = e.exitCode ?? null;
          }
        }
        events.push(
          e.event === 'graph'
            ? {
                ...e,
                nodes: undefined,
                edges: undefined,
                models: undefined,
                detail: `${graph.nodes.length} estados / ${graph.edges.length} transicoes`,
              }
            : e,
        );
      }
      s = {
        ...s,
        current,
        graph,
        events: events.slice(-750),
        streamSequence: sequence,
        memory: [...memory.nodes.values()],
        outputs: [...memory.outputs.values()],
      };
      ref.current = s;
      setData(s);
      if (current?.finished && s.runs[0]?.finished !== current.finished) void refresh();
    };
    const schedule = () => {
      if (!timer) timer = setTimeout(flush, 100);
    };
    async function refresh() {
      if (syncing || !alive) return;
      syncing = true;
      try {
        const response = await fetch('/api/state');
        if (!response.ok) throw new Error('Nao foi possivel conectar ao nucleo local.');
        const snapshot: Snapshot = await response.json();
        if (!alive) return;
        preserveObservedAlgorithmEvidence(snapshot, ref.current);
        if (reconnecting && ref.current?.session === snapshot.session) {
          const before = ref.current.streamSequence;
          const delivered = new Set(
            pending
              .filter(
                (e) =>
                  e.session === snapshot.session &&
                  e.seq > before &&
                  e.seq <= snapshot.streamSequence,
              )
              .map((e) => e.seq),
          );
          if (snapshot.streamSequence - before > delivered.size) setTransportGaps((n) => n + 1);
        }
        reconnecting = false;
        if (ref.current?.session !== snapshot.session) {
          pending = pending.filter((e) => e.session === snapshot.session);
          setTransportGaps(0);
        }
        memory.restore(snapshot.memory);
        memory.outputs = new Map(snapshot.outputs.map((o) => [o.id, o]));
        ref.current = snapshot;
        setData(snapshot);
        if (snapshot.current?.mode === 'evolution' && snapshot.current.finished)
          void refreshCampaigns();
      } catch (e) {
        if (alive) setError(String(e));
      } finally {
        syncing = false;
        if (pending.length) schedule();
      }
    }
    refreshRef.current = () => void refresh();
    const stream = new EventSource('/api/events');
    stream.onmessage = (message) => {
      if (!alive) return;
      try {
        const packet = JSON.parse(message.data);
        if (packet.event === 'connected') {
          setConnected(true);
          void refresh();
        } else {
          pending.push(...(packet.event === 'batch' ? packet.records : [packet]));
          schedule();
        }
      } catch {
        setError('Registro de telemetria invalido.');
      }
    };
    stream.onerror = () => {
      if (!alive) return;
      reconnecting = true;
      setConnected(false);
    };
    const interval = setInterval(() => {
      if (!busy(ref.current?.current ?? null)) void refresh();
    }, 10000);
    function refreshCampaigns() {
      return fetch('/api/evolution')
        .then((r) => {
          if (!r.ok) throw new Error('Falha ao ler experimentos.');
          return r.json();
        })
        .then((r) => {
          if (alive) setCampaigns(r.campaigns);
        })
        .catch((e) => {
          if (alive) setError(String(e));
        });
    }
    void refreshCampaigns();
    return () => {
      alive = false;
      stream.close();
      clearInterval(interval);
      clearTimeout(timer);
    };
  }, []);

  const post = useCallback(
    async (route: string, value: unknown) => {
      if (!ref.current || !connected) throw new Error('Conexao local indisponivel.');
      setSending(true);
      setError('');
      try {
        const response = await fetch(`/api/${route}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Axon-Session': ref.current.token },
          body: JSON.stringify(value),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error ?? 'Falha na operacao.');
        refreshRef.current();
        return result;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        throw e;
      } finally {
        setSending(false);
      }
    },
    [connected],
  );
  const run = (value: RunOptions) => post('run', value);
  const control = (command: 'pause' | 'resume' | 'step' | 'cancel') => post('control', { command });
  return {
    data,
    connected,
    error,
    setError,
    campaigns,
    sending,
    transportGaps,
    run,
    control,
    refresh: () => refreshRef.current(),
  };
}
