import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Cpu,
  Database,
  FileCode2,
  FileText,
  FlaskConical,
  Folder,
  GitBranch,
  LayoutDashboard,
  LayoutGrid,
  Link,
  Menu,
  Moon,
  Network,
  PanelLeftClose,
  Pause,
  Play,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  StepForward,
  Sun,
  Terminal,
  X,
  Zap,
} from 'lucide-react';
import {
  Badge,
  colored,
  Empty,
  Expand,
  IconButton,
  Modal,
  Segments,
  Sparkline,
  Ticks,
  Widget,
} from './components';
import { busy, bytes, describe, fmt, label, latency, useAxon } from './useAxon';
import { CoreDiagram, GraphView, palette } from './visuals';
import { Laboratory } from './Laboratory';
import type { Campaign, GraphNode, Mode, RunOptions, Source } from './types';

const views = [
  { id: 'overview', name: 'Visao geral', icon: LayoutDashboard },
  { id: 'network', name: 'Rede de estados', icon: Network },
  { id: 'evolution', name: 'Seeds e evolucao', icon: GitBranch },
  { id: 'laboratory', name: 'Laboratorio', icon: FlaskConical },
  { id: 'knowledge', name: 'Conhecimento', icon: Database },
  { id: 'runs', name: 'Execucoes', icon: Activity },
  { id: 'sources', name: 'Arquivos', icon: Folder },
] as const;
type View = (typeof views)[number]['id'];
const panels = [
  'Runtime',
  'Modelos',
  'Memoria',
  'Conhecimento',
  'Fluxo de eventos',
  'Recursos',
] as const;
const taskDefault =
  'task average_stream {\n  data numbers: Vec<u64> = [2, 4, 9, 12, 18]\n  goal derive IncrementalArtifact<AverageExactU64>\n}';
const taskNames = [
  'Quadrado',
  'Produto escalado',
  'Hipotenusa 2D',
  'Discriminante',
  'Raiz positiva',
  'Raiz negativa',
  'Norma 3D',
  'Distancia entre pontos',
];

function download(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }),
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function App() {
  const reducedMotion = useReducedMotion();
  const axon = useAxon(),
    s = axon.data,
    j = s?.current ?? null;
  const [view, setView] = useState<View>('overview');
  const [dark, setDark] = useState(() => localStorage.getItem('axon.theme') === 'dark');
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem('axon.sidebar') === 'collapsed',
  );
  const [mobileMenu, setMobileMenu] = useState(false),
    [modal, setModal] = useState<'run' | 'search' | 'panels' | 'settings' | 'help' | null>(null);
  const [inspection, setInspection] = useState<{ title: string; value: unknown } | null>(null);
  const [hidden, setHidden] = useState<string[]>(() => {
    try {
      const stored: unknown = JSON.parse(localStorage.getItem('axon.panels') ?? '[]');
      return Array.isArray(stored)
        ? stored.filter(
            (p): p is string => typeof p === 'string' && panels.some((name) => name === p),
          )
        : [];
    } catch {
      return [];
    }
  });
  const [query, setQuery] = useState(''),
    [sourceFilter, setSourceFilter] = useState(''),
    [sourceKind, setSourceKind] = useState('TODOS');
  const [gridMode, setGridMode] = useState('SAIDAS'),
    [resourceMode, setResourceMode] = useState('MODELOS');
  const [campaignId, setCampaignId] = useState(''),
    [tab, setTab] = useState('TODOS');
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    localStorage.setItem('axon.theme', dark ? 'dark' : 'light');
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  }, [dark]);
  useEffect(() => {
    localStorage.setItem('axon.sidebar', collapsed ? 'collapsed' : 'expanded');
  }, [collapsed]);
  useEffect(() => {
    localStorage.setItem('axon.panels', JSON.stringify(hidden));
  }, [hidden]);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setModal('search');
      }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, []);
  const inspect = (title: string, value: unknown) => setInspection({ title, value });
  const go = (v: View) => {
    setView(v);
    setMobileMenu(false);
  };
  const invoke = (f: () => Promise<unknown>) => void f().catch(() => {});
  const graph = s?.graph ?? { nodes: [], edges: [], models: [] };
  const active = axon.connected && busy(j),
    steppable = j && ['factor', 'transfer'].includes(j.mode);
  const online = axon.connected && !!s?.coreReady;
  const graphBytes = graph.models.reduce((n, m) => n + m.bytes, 0);
  const events = s?.events.filter((e) => !j || e.run === j.id) ?? [];
  const latencies = events.filter((e) => e.latencyNs != null).map((e) => e.latencyNs! / 1e6);
  const last = events.at(-1);
  const routes = Object.entries(j?.trace.methods ?? {}).sort((a, b) => b[1] - a[1]);
  const campaign = axon.campaigns.find((c) => c.id === campaignId) ?? axon.campaigns[0];
  const status = !axon.connected
    ? 'Desconectado'
    : j
      ? label[j.status]
      : online
        ? 'Pronto'
        : 'Nucleo indisponivel';
  const footer = (
    <>
      <span className="mono">
        {active ? 'EXECUCAO LOCAL' : j ? 'ULTIMA EXECUCAO' : 'SNAPSHOT LOCAL'}
      </span>
      <span className="push" />
      <Badge tone={online ? 'green' : 'neutral'}>{axon.connected ? 'Conectado' : 'Offline'}</Badge>
    </>
  );
  const card = (
    name: string,
    children: ReactNode,
    foot: ReactNode = footer,
    tools?: ReactNode,
    sub?: string,
  ) => (
    <Widget
      title={name}
      sub={sub}
      tools={tools}
      onClose={view === 'overview' ? () => setHidden((v) => [...v, name]) : undefined}
      onInspect={() =>
        inspect(
          name,
          name === 'Runtime'
            ? j
            : name === 'Modelos'
              ? graph
              : name === 'Memoria'
                ? s?.memory
                : name === 'Conhecimento'
                  ? s?.sources
                  : name === 'Fluxo de eventos'
                    ? events
                    : { graphBytes, sources: s?.sources },
        )
      }
      footer={foot}
    >
      {children}
    </Widget>
  );

  const runtime = card(
    'Runtime',
    <>
      <div className="instrument runtime-diagram">
        <div className="runtime-stat">
          <span className="eyebrow">MODELOS OBSERVADOS</span>
          <div className="metric-value">
            {fmt(graph.models.length)}
            <small>/ {fmt(graph.nodes.length)} estados</small>
          </div>
        </div>
        <CoreDiagram
          graph={graph}
          active={active}
          event={events.findLast((e) => e.event === 'step')}
        />
        <div className="instrument-bottom mono">
          <span>
            {j
              ? `${label[j.mode].toUpperCase()} / ${status.toUpperCase()}`
              : 'SNAPSHOT / SEM EXECUCAO ATIVA'}
          </span>
          <span>
            {j?.latest.latencyNs != null
              ? `${latency(j.latest.latencyNs)} / ULTIMA CONSULTA`
              : 'LATENCIA NAO MEDIDA'}
          </span>
        </div>
      </div>
      <div className="instrument runtime-rows">
        {[
          {
            name: 'observer',
            meta: 'RUST / EVENTOS DO ALGORITMO',
            available: online,
            state:
              active && steppable
                ? j?.status === 'paused'
                  ? 'PAUSED'
                  : j?.status === 'pausing'
                    ? 'PAUSING'
                    : 'RUNNING'
                : online
                  ? 'READY'
                  : 'OFFLINE',
          },
          {
            name: 'solver',
            meta: 'RUST / TAREFA TIPADA',
            available: online,
            state: active && j?.mode === 'math' ? 'RUNNING' : online ? 'READY' : 'OFFLINE',
          },
          {
            name: 'knowledge',
            meta: `MEMORIA LOCAL / ${s?.indexed ? 'INDEXADA' : 'SEM INDICE'}`,
            available: !!s?.indexed,
            state:
              active && ['knowledge', 'index'].includes(j?.mode ?? '')
                ? 'RUNNING'
                : s?.indexed
                  ? 'READY'
                  : 'EMPTY',
          },
          {
            name: 'telemetry',
            meta: 'SSE / LOOPBACK / JSONL',
            available: axon.connected,
            state: axon.connected ? 'CONNECTED' : 'OFFLINE',
          },
        ].map((row) => (
          <div className="runtime-row" key={row.name}>
            <i className={`status-dot ${row.available ? 'on' : ''}`} />
            <div>
              <strong>{row.name}</strong>
              <span className="mono">{row.meta}</span>
            </div>
            <Badge
              tone={row.state === 'RUNNING' || row.state === 'CONNECTED' ? 'green' : 'neutral'}
            >
              {row.state}
            </Badge>
            <span className="mono row-tail">
              {row.name === 'observer'
                ? `${fmt(j?.trace.records ?? 0)} EVT`
                : row.name === 'telemetry'
                  ? `PORTA ${s?.port ?? '--'}`
                  : 'LOCAL'}
            </span>
          </div>
        ))}
      </div>
    </>,
    <>
      <span className="mono">
        <i className={`status-dot ${online ? 'on' : ''}`} /> {status.toUpperCase()} /{' '}
        {fmt(j?.counters.queryCalls ?? 0)} CONSULTAS
      </span>
      <span className="push" />
      <button
        className="primary"
        disabled={!online || axon.sending || active}
        onClick={() => setModal('run')}
      >
        <Zap size={15} />
        Executar AXON
      </button>
    </>,
    <button className="select-pill" title="Runtime local" onClick={() => setModal('settings')}>
      <Cpu size={15} />
      local
      <ChevronDown size={14} />
    </button>,
  );

  const models = card(
    'Modelos',
    <>
      <div className="instrument">
        <div className="section-top">
          <div>
            <span className="eyebrow">ESTADOS PREDITIVOS</span>
            <div className="metric-value">
              {fmt(graph.nodes.length)}
              <small>/ {fmt(graph.edges.length)} transicoes</small>
            </div>
          </div>
          <Sparkline
            values={events.filter((e) => e.event === 'probe').map((e) => e.hypotheses ?? 0)}
          />
        </div>
        <p className="mono metric-note">
          {graph.phase === 'stored-snapshot'
            ? 'SNAPSHOT ARMAZENADO'
            : (graph.phase ?? 'SEM GRAFO').toUpperCase()}{' '}
          / {bytes(graphBytes)} CODIFICADOS
        </p>
        <div className="three-metrics">
          <div>
            <span className="eyebrow">AUDITORIA</span>
            <strong>
              {j?.result?.exact === true
                ? 'Exata'
                : j?.result?.exact === false
                  ? 'Refutada'
                  : 'Nao executada'}
            </strong>
            <Ticks label="Modelos observados" value={graph.models.length} total={8} />
          </div>
          <div>
            <span className="eyebrow">RESIDUAL</span>
            <strong>{fmt(j?.result?.residual)}</strong>
            <Ticks
              label="Erros registrados"
              color="var(--pink)"
              value={j?.counters.errors ?? 0}
              total={8}
            />
          </div>
          <div>
            <span className="eyebrow">VERIFICADAS</span>
            <strong>{fmt(j?.result?.checkedEdges)}</strong>
            <Ticks
              label="Modelos auditados"
              color="var(--teal)"
              value={j?.result?.exact ? graph.models.length : 0}
              total={8}
            />
          </div>
        </div>
      </div>
      <div className="instrument">
        <div className="section-top">
          <h3>
            Modelos <span className="muted">/ {graph.models.length}</span>
          </h3>
          <Segments
            items={['SAIDAS', 'ACOES']}
            value={gridMode}
            onChange={setGridMode}
            label="Metrica das transicoes"
          />
        </div>
        <div className="model-matrix">
          {graph.models.map((m, i) => (
            <div className="matrix-row" key={m.context}>
              <button
                className="matrix-label"
                title={m.context}
                onClick={() => inspect(m.context, m)}
              >
                {m.name.toUpperCase()}
                <span>{m.states} estados</span>
              </button>
              <div className="matrix-cells">
                {graph.edges
                  .filter((e) => e.factor === i)
                  .slice(0, 16)
                  .map((e) => (
                    <button
                      key={`${e.source}-${e.action}`}
                      title={`${e.source} -> ${e.target} / a${e.action} / y${e.output}`}
                      onClick={() => inspect('Transicao', e)}
                      style={{
                        background: palette[i % 5],
                        opacity:
                          0.35 +
                          (gridMode === 'ACOES'
                            ? e.action / Math.max(1, ...graph.edges.map((e) => e.action))
                            : e.output / Math.max(1, ...graph.edges.map((e) => e.output))) *
                            0.65,
                      }}
                      aria-label={`Transicao ${e.source} acao ${e.action}`}
                    />
                  ))}
              </div>
              <span className="mono">{bytes(m.bytes)}</span>
            </div>
          ))}
          {!graph.models.length && <Empty icon={Network} title="Nenhum modelo recebido" />}
        </div>
        <div className="instrument-bottom mono">
          <span>
            {Math.min(16, graph.edges.filter((e) => e.factor === 0).length)} TRANSICOES / MODELO
          </span>
          <span>{gridMode === 'SAIDAS' ? 'COR = VALOR DA SAIDA' : 'COR = INDICE DA ACAO'}</span>
        </div>
      </div>
    </>,
    <>
      <span className="mono">
        {graph.models.length} MODELOS / {fmt(graph.edges.length)} TRANSICOES
      </span>
      <span className="push" />
      <button className="primary" onClick={() => go('network')}>
        <Network size={14} />
        Abrir rede
      </button>
    </>,
  );

  const memory = card(
    'Memoria',
    <>
      <div className="instrument">
        <div className="section-top">
          <div>
            <span className="eyebrow">PREFIXOS OBSERVADOS</span>
            <div className="metric-value">
              {fmt(s?.memory.length ?? 0)}
              <small>registros</small>
            </div>
          </div>
          <Badge tone={active ? 'green' : 'neutral'}>{active ? 'Em uso' : 'Historico'}</Badge>
        </div>
        <p className="mono metric-note">
          {fmt(j?.counters.physical ?? 0)} FISICAS / {fmt(j?.counters.cached ?? 0)} CACHE / SEM
          TOKENS DE LLM
        </p>
        <div className="memory-tiles">
          {(s?.memory ?? []).slice(-192).map((n) => (
            <button
              key={n.id}
              style={{
                background: palette[Math.min(n.depth, 4)],
                opacity: n.root ? 0.4 : 0.45 + Math.min(n.depth, 8) * 0.06,
              }}
              title={`${n.id} / profundidade ${n.depth}`}
              aria-label={`Prefixo ${n.id}`}
              onClick={() => inspect('Prefixo observado', n)}
            />
          ))}
        </div>
        {!s?.memory.length && (
          <Empty icon={BrainCircuit} title="Nenhuma consulta recebida nesta sessao" />
        )}
        <div className="legend">
          {['RAIZES', 'PROF. 1', 'PROF. 2', 'PROF. 3', 'PROF. 4+'].map((v, i) => (
            <span key={v}>
              <i style={{ background: palette[i] }} />
              {v}
            </span>
          ))}
        </div>
      </div>
      <div className="instrument">
        <div className="section-top">
          <h3>Saidas observadas</h3>
          <span className="mono">{fmt(s?.outputs.length ?? 0)} VALORES</span>
        </div>
        {(s?.outputs ?? []).slice(-4).map((o) => (
          <button className="list-row" key={o.id} onClick={() => inspect('Saida observada', o)}>
            <i className="dot" style={{ background: 'var(--teal)' }} />
            <strong>{o.output}</strong>
            <span className="mono truncate">{o.context.split('/').at(-1)}</span>
            <span className="push" />
            <span className="mono">{o.physical + o.cached + o.executed} OBS.</span>
          </button>
        ))}
        {!s?.outputs.length && <div className="quiet-empty mono">SEM RESPOSTAS REGISTRADAS</div>}
      </div>
    </>,
    <>
      <span className="mono">
        {Math.min(192, s?.memory.length ?? 0)} PREFIXOS VISIVEIS / CACHE DE CONSULTAS
      </span>
      <span className="push" />
      <Expand onClick={() => inspect('Memoria completa', s?.memory)} />
    </>,
  );

  const knowledge = card(
    'Conhecimento',
    <>
      <div className="instrument knowledge-summary">
        <div className="knowledge-graphic">
          <div className="stack-layer layer-3" />
          <div className="stack-layer layer-2" />
          <div className="stack-layer layer-1">
            <Database size={32} strokeWidth={1.2} />
          </div>
          <span className="knowledge-point p1">
            <FileCode2 size={18} />
          </span>
          <span className="knowledge-point p2">
            <FileText size={18} />
          </span>
          <span className="knowledge-point p3">
            <BookOpen size={18} />
          </span>
        </div>
        <h3>{s?.indexed ? 'Memoria local indexada' : 'Fontes locais'}</h3>
        <p className="muted">{fmt(s?.sources.length ?? 0)} arquivos / docs, src e tests</p>
        <div className="center-actions">
          <button
            className="solid"
            disabled={!online || active || axon.sending}
            onClick={() => invoke(() => axon.run({ mode: 'index' }))}
          >
            <Link size={15} />
            {s?.indexed ? 'Reindexar fontes' : 'Indexar fontes'}
          </button>
          <button className="secondary" onClick={() => go('sources')}>
            <Folder size={15} />
            Abrir arquivos
          </button>
        </div>
      </div>
      <div className="instrument sources-summary">
        {[
          { name: 'Codigo-fonte', kind: 'code', icon: FileCode2, detail: 'src / Rust' },
          { name: 'Documentacao', kind: 'paper', icon: BookOpen, detail: 'docs / Markdown' },
          { name: 'Tarefas', kind: 'task', icon: FileText, detail: 'examples / AXON' },
        ].map(({ name, kind, icon: Icon, detail }) => (
          <button
            key={kind}
            className="source-summary-row"
            onClick={() => {
              setSourceKind(kind === 'code' ? 'CODIGO' : kind === 'paper' ? 'DOCS' : 'TAREFAS');
              go('sources');
            }}
          >
            <span className="source-icon">
              <Icon size={19} strokeWidth={1.5} />
            </span>
            <div>
              <strong>{name}</strong>
              <span className="muted">{detail}</span>
            </div>
            <span className="push" />
            <span className="mono">
              {fmt(s?.sources.filter((f) => f.kind === kind).length ?? 0)}
            </span>
            <ChevronRight size={15} className="muted" />
          </button>
        ))}
      </div>
    </>,
    <>
      <span className="mono">
        {s?.indexed ? 'INDICE DISPONIVEL' : 'AINDA NAO INDEXADO'} / LOCAL
      </span>
      <span className="push" />
      <button className="secondary" onClick={() => go('knowledge')}>
        <Search size={14} />
        Investigar
      </button>
    </>,
    <span className="mono">{fmt(s?.sources.length ?? 0)} FONTES</span>,
  );

  const flow = card(
    'Fluxo de eventos',
    <>
      <div className="instrument">
        <div className="section-top">
          <div>
            <span className="eyebrow">REGISTROS DA EXECUCAO</span>
            <div className="metric-value">
              {fmt(j?.trace.records ?? 0)}
              <small>eventos</small>
            </div>
          </div>
          <Badge tone={active && j?.status !== 'paused' ? 'green' : 'neutral'}>
            {active ? (j?.status === 'paused' ? 'Pausado' : 'Live') : 'Sem atividade'}
          </Badge>
        </div>
        <div
          className="event-wave"
          role="img"
          aria-label="Latencias medidas das consultas recentes"
        >
          {latencies.slice(-80).map((v, i) => (
            <i
              key={i}
              style={{
                height: `${8 + Math.min(1, v / Math.max(0.001, ...latencies)) * 58}px`,
                background: i % 3 === 0 ? 'var(--orange)' : 'var(--orange-border)',
              }}
            />
          ))}
          {!latencies.length && <span className="mono muted">NENHUMA LATENCIA RECEBIDA</span>}
        </div>
        <div className="event-highlight">
          <span className="mono">ULTIMO</span>
          <span>{last ? describe(last) : 'Nenhum evento nesta sessao'}</span>
        </div>
      </div>
      <div className="instrument">
        <div className="section-top">
          <h3>Rotas de processamento</h3>
          <span className="mono">
            {j?.trace.mode === 'complete-query-trace' ? 'QUERY TRACE' : 'ENTRY / EXIT'}
          </span>
        </div>
        <div className="routing-flow">
          <svg
            viewBox="0 0 450 130"
            role="img"
            aria-label="Contagem real de checkpoints por metodo"
          >
            <circle cx="15" cy="65" r="4" fill="var(--surface)" stroke="var(--muted)" />
            {routes.slice(0, 4).map(([method, n], i) => (
              <g key={method}>
                <path
                  d={`M20 65H75C110 65 107 ${16 + i * 32} 150 ${16 + i * 32}H290`}
                  stroke={palette[i]}
                  opacity=".45"
                  fill="none"
                  strokeWidth="2"
                />
                {active &&
                  !reducedMotion &&
                  events.some((e) => e.method === method && Date.now() - e.time < 850) && (
                    <motion.path
                      className="route-pulse"
                      key={`${method}-${n}`}
                      d={`M20 65H75C110 65 107 ${16 + i * 32} 150 ${16 + i * 32}H290`}
                      stroke={palette[i]}
                      fill="none"
                      strokeWidth="2.5"
                      initial={{ pathLength: 0, opacity: 1 }}
                      animate={{ pathLength: 1, opacity: 0 }}
                      transition={{ duration: 0.75 }}
                    />
                  )}
                <circle cx="290" cy={16 + i * 32} r="3" fill={palette[i]} />
                <text x="305" y={20 + i * 32} fill="var(--muted)" fontSize="10">
                  {method.split('::').at(-1)?.slice(0, 18)}
                </text>
                <text x="440" y={20 + i * 32} textAnchor="end" fill="var(--ink)" fontSize="10">
                  {n}
                </text>
              </g>
            ))}
          </svg>
        </div>
        <div className="instrument-bottom mono">
          <span>{fmt(j?.trace.gaps ?? 0)} LACUNAS NO NUCLEO</span>
          <span>{fmt(axon.transportGaps)} NA ENTREGA</span>
        </div>
      </div>
    </>,
    <>
      <span className="mono">
        {last
          ? `${Math.max(0, Math.round((now - last.time) / 1000))} S DESDE ULTIMO EVENTO`
          : 'AGUARDANDO EXECUCAO'}
      </span>
      <span className="push" />
      <button className="secondary" onClick={() => go('runs')}>
        <Terminal size={14} />
        Abrir log
      </button>
    </>,
  );

  const resourceGroups =
    resourceMode === 'MODELOS'
      ? graph.models.map((m, i) => ({
          name: m.name.toUpperCase(),
          value: m.bytes,
          color: palette[i % 5],
        }))
      : ['code', 'paper', 'task'].map((kind, i) => ({
          name: { code: 'CODIGO', paper: 'DOCS', task: 'TAREFAS' }[kind],
          value: s?.sources.filter((f) => f.kind === kind).reduce((n, f) => n + f.bytes, 0) ?? 0,
          color: palette[i],
        }));
  const totalResource = resourceGroups.reduce((n, g) => n + g.value, 0);
  const resources = card(
    'Recursos',
    <>
      <div className="instrument">
        <div className="section-top">
          <div>
            <span className="eyebrow">
              {resourceMode === 'MODELOS' ? 'MODELOS CODIFICADOS' : 'CATALOGO / TAMANHO EM DISCO'}
            </span>
            <div className="metric-value">{bytes(totalResource)}</div>
          </div>
          <Segments
            items={['MODELOS', 'ARQUIVOS']}
            value={resourceMode}
            onChange={setResourceMode}
            label="Escopo dos recursos"
          />
        </div>
        <p className="mono metric-note">
          {resourceMode === 'MODELOS'
            ? 'EXCLUI CACHE, PROCESSO E INTERFACE'
            : 'EXCLUI BINARIOS, MEMORIAS E LOGS'}
        </p>
        <div className="cost-bar">
          {resourceGroups
            .filter((g) => g.value > 0)
            .map((g) => (
              <span
                key={g.name}
                style={{
                  width: `${(g.value / Math.max(1, totalResource)) * 100}%`,
                  background: g.color,
                }}
              />
            ))}
        </div>
        <div className="cost-legend">
          {resourceGroups.map((g) => (
            <div key={g.name}>
              <span className="eyebrow">
                <i className="dot" style={{ background: g.color }} />
                {g.name}
              </span>
              <strong>{bytes(g.value)}</strong>
              <span className="mono">
                {fmt(totalResource ? (g.value / totalResource) * 100 : 0)}%
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="instrument resource-limits">
        <h3>Escopo das medicoes</h3>
        {[
          {
            name: 'Consultas fisicas',
            value: fmt(j?.counters.physical ?? 0),
            color: 'var(--orange)',
            n: j?.counters.physical ?? 0,
          },
          {
            name: 'Consultas em cache',
            value: fmt(j?.counters.cached ?? 0),
            color: 'var(--blue)',
            n: j?.counters.cached ?? 0,
          },
          {
            name: 'Passos confirmados',
            value: fmt(j?.counters.steps ?? 0),
            color: 'var(--teal)',
            n: j?.counters.steps ?? 0,
          },
        ].map((g) => (
          <div className="resource-row" key={g.name}>
            <i className="dot" style={{ background: g.color }} />
            <strong>{g.name}</strong>
            <Ticks
              label={g.name}
              value={j?.counters.queryCalls ? (g.n / j.counters.queryCalls) * 30 : 0}
              color={g.color}
            />
            <span className="mono">{g.value}</span>
          </div>
        ))}
        <div className="instrument-bottom mono">
          <span>RSS ATUAL: NAO MEDIDO</span>
          <span>ENERGIA: NAO MEDIDA</span>
        </div>
      </div>
    </>,
    <>
      <span className="mono">
        {resourceMode === 'MODELOS' ? 'ENCODED BYTES' : 'FILESYSTEM BYTES'} / SEM ESTIMATIVA
      </span>
      <span className="push" />
      <button
        className="secondary"
        onClick={() =>
          download(
            { scope: resourceMode, groups: resourceGroups, time: now },
            'axon-resources.json',
          )
        }
      >
        <ArrowDownToLine size={14} />
        Exportar
      </button>
    </>,
  );

  async function openSource(source: Source) {
    try {
      const response = await fetch(`/api/source?id=${source.id}`);
      const body = await response.json();
      if (!response.ok) throw new Error(body.error);
      inspect(source.path, body.text);
    } catch (e) {
      axon.setError(String(e));
    }
  }
  function nodeInspect(node: GraphNode) {
    inspect(node.id, { node, transitions: graph.edges.filter((e) => e.source === node.id) });
  }
  const sourceList = (limit?: number) => {
    const filtered = (s?.sources ?? []).filter(
      (f) =>
        f.path.toLowerCase().includes(sourceFilter.toLowerCase()) &&
        (sourceKind === 'TODOS' ||
          f.kind === { CODIGO: 'code', DOCS: 'paper', TAREFAS: 'task' }[sourceKind]),
    );
    return (
      <div className="source-browser">
        <div className="source-browser-tools">
          <label className="search-field">
            <Search size={16} />
            <input
              placeholder="Buscar arquivos..."
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              aria-label="Buscar arquivos"
            />
          </label>
          <Segments
            items={['TODOS', 'CODIGO', 'DOCS', 'TAREFAS']}
            value={sourceKind}
            onChange={setSourceKind}
            label="Tipo de arquivo"
          />
        </div>
        <div className="file-table">
          <div className="file-table-heading mono">
            <span>ARQUIVO</span>
            <span>TIPO</span>
            <span>TAMANHO</span>
          </div>
          {filtered.slice(0, limit ?? 400).map((f) => (
            <button className="file-table-row" key={f.id} onClick={() => void openSource(f)}>
              {f.kind === 'code' ? <FileCode2 size={18} /> : <FileText size={18} />}
              <span>{f.path}</span>
              <span className="mono muted">{f.kind.toUpperCase()}</span>
              <span className="mono muted">{bytes(f.bytes)}</span>
              <ArrowUpRight size={14} className="muted" />
            </button>
          ))}
          {!filtered.length && <Empty icon={Search} title="Nenhum arquivo encontrado" />}
        </div>
        <span className="mono muted table-count">
          {fmt(filtered.length)} ARQUIVOS / CATALOGO LIMITADO A 400
        </span>
      </div>
    );
  };
  const cards: Record<string, ReactNode> = {
    Runtime: runtime,
    Modelos: models,
    Memoria: memory,
    Conhecimento: knowledge,
    'Fluxo de eventos': flow,
    Recursos: resources,
  };

  return (
    <div className="workspace">
      <aside
        className={`sidebar ${collapsed ? 'collapsed' : ''} ${mobileMenu ? 'mobile-open' : ''}`}
      >
        <div className="sidebar-brand">
          <button
            className="brand-mark"
            title={collapsed ? 'Expandir navegacao' : 'AXON'}
            aria-label={collapsed ? 'Expandir navegacao' : 'AXON'}
            onClick={() => {
              if (collapsed) setCollapsed(false);
              else go('overview');
            }}
          >
            <Zap size={21} fill="currentColor" />
          </button>
          <button
            className="brand-name"
            onClick={() => setModal('settings')}
            title="Workspace AXON"
          >
            AXON <ChevronDown size={14} />
          </button>
          <IconButton
            icon={PanelLeftClose}
            label={collapsed ? 'Expandir navegacao' : 'Recolher navegacao'}
            onClick={() => setCollapsed((v) => !v)}
          />
        </div>
        <button className="sidebar-search" onClick={() => setModal('search')} title="Buscar">
          <Search size={16} />
          <span>Buscar</span>
          <span className="search-key">
            <Search size={11} />
          </span>
        </button>
        <nav aria-label="Navegacao principal">
          {views.slice(0, 4).map((v) => (
            <button
              key={v.id}
              title={v.name}
              className={`nav-item ${view === v.id ? 'selected' : ''}`}
              onClick={() => go(v.id)}
            >
              <v.icon size={19} strokeWidth={1.65} />
              <span>{v.name}</span>
              {v.id === 'overview' && active && <span className="nav-count">1</span>}
            </button>
          ))}
          <div className="nav-section">
            <span>Laboratorio</span>
            <ChevronDown size={13} />
          </div>
          {views.slice(4).map((v) => (
            <button
              key={v.id}
              title={v.name}
              className={`nav-item ${view === v.id ? 'selected' : ''}`}
              onClick={() => go(v.id)}
            >
              <v.icon size={19} strokeWidth={1.65} />
              <span>{v.name}</span>
            </button>
          ))}
          <button className="nav-item" title="Nova execucao" onClick={() => setModal('run')}>
            <Play size={18} />
            <span>Nova execucao</span>
          </button>
        </nav>
        <div className="nav-section favorites">
          <span>Favoritos</span>
          <ChevronDown size={13} />
        </div>
        <button
          className="nav-item favorite"
          title="AXON.md"
          onClick={() => {
            const f = s?.sources.find((f) => f.path === 'docs/AXON.md');
            if (f) void openSource(f);
          }}
        >
          <FileText size={18} />
          <span>Documento mestre</span>
        </button>
        <button
          className="nav-item favorite"
          title="Resultados evolutivos"
          onClick={() => go('evolution')}
        >
          <Folder size={18} />
          <span>Resultados evolutivos</span>
        </button>
        <div className="sidebar-bottom">
          <button className="nav-item" title="Preferencias" onClick={() => setModal('settings')}>
            <Settings2 size={18} />
            <span>Preferencias</span>
          </button>
          <button
            className="nav-item"
            title="Escopo e integridade"
            onClick={() => setModal('help')}
          >
            <CircleHelp size={18} />
            <span>Escopo e integridade</span>
          </button>
          <div className="sidebar-meter">
            <div>
              <span>Modelos observados</span>
              <strong>{graph.models.length}</strong>
            </div>
            <Ticks
              label="Quantidade de modelos observados"
              value={graph.models.length}
              total={8}
              color="var(--teal)"
            />
          </div>
          <button
            className="local-profile"
            onClick={() => setModal('settings')}
            title="Workspace local"
          >
            <span className="profile-mark">
              <Cpu size={21} />
              <i className={`status-dot ${online ? 'on' : ''}`} />
            </span>
            <span>
              <strong>Workspace local</strong>
              <small>127.0.0.1:{s?.port ?? '--'}</small>
            </span>
            <ChevronDown size={14} />
          </button>
        </div>
      </aside>
      {mobileMenu && (
        <button
          className="sidebar-backdrop"
          aria-label="Fechar navegacao"
          onClick={() => setMobileMenu(false)}
        />
      )}
      <div className="workspace-main">
        <header className="topbar">
          <IconButton
            className="mobile-toggle"
            icon={Menu}
            label="Abrir navegacao"
            onClick={() => setMobileMenu((v) => !v)}
          />
          <span className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <strong>{views.find((v) => v.id === view)?.name}</strong>
          </span>
          <div className="topbar-right">
            <span className={`connection-label ${online ? 'on' : ''}`}>
              <i className={`status-dot ${online ? 'on' : ''}`} />
              {online ? 'Nucleo conectado' : 'Sem conexao'}
            </span>
            <span className="topbar-divider" />
            <IconButton
              icon={dark ? Sun : Moon}
              label={dark ? 'Tema claro' : 'Tema escuro'}
              onClick={() => setDark((v) => !v)}
            />
            <IconButton
              icon={SlidersHorizontal}
              label="Preferencias"
              onClick={() => setModal('settings')}
            />
          </div>
        </header>
        <main className="page">
          <div className="page-heading">
            <div>
              <span className="eyebrow">AXON / LOCAL WORKSPACE</span>
              <h1>{views.find((v) => v.id === view)?.name}</h1>
            </div>
            <div className="page-actions">
              {view === 'overview' && (
                <button className="secondary" onClick={() => setModal('panels')}>
                  <LayoutGrid size={15} />
                  <span>Paineis</span>
                </button>
              )}
              <button
                className="primary"
                disabled={!online || active || axon.sending}
                onClick={() => setModal('run')}
              >
                <Plus size={16} />
                <span>Nova execucao</span>
              </button>
            </div>
          </div>
          <AnimatePresence>
            {axon.error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="error-banner"
                role="alert"
              >
                <ShieldCheck size={17} />
                <span>{axon.error}</span>
                <IconButton icon={X} label="Fechar aviso" onClick={() => axon.setError('')} />
              </motion.div>
            )}
          </AnimatePresence>
          {active && (
            <div className="run-bar">
              <Badge tone="green">{status}</Badge>
              <span className="mono truncate">
                {j?.id.slice(0, 8)} / {label[j!.mode]}
              </span>
              <span className="mono">{fmt(j?.trace.records)} EVENTOS</span>
              <span className="push" />
              {steppable && (
                <>
                  <IconButton
                    icon={j?.status === 'paused' ? Play : Pause}
                    label={j?.status === 'paused' ? 'Retomar' : 'Pausar'}
                    disabled={axon.sending || j?.status === 'pausing'}
                    onClick={() =>
                      invoke(() => axon.control(j?.status === 'paused' ? 'resume' : 'pause'))
                    }
                  />
                  <IconButton
                    icon={StepForward}
                    label="Avancar um checkpoint"
                    disabled={axon.sending || j?.status !== 'paused'}
                    onClick={() => invoke(() => axon.control('step'))}
                  />
                </>
              )}
              <IconButton
                icon={Square}
                label="Cancelar execucao"
                disabled={axon.sending}
                onClick={() => invoke(() => axon.control('cancel'))}
              />
            </div>
          )}
          {view === 'overview' && (
            <div className="dashboard-grid">
              {panels
                .filter((p) => !hidden.includes(p))
                .map((p) => (
                  <div key={p} className="widget-slot">
                    {cards[p]}
                  </div>
                ))}
              {hidden.length === panels.length && (
                <button className="restore-panels" onClick={() => setHidden([])}>
                  <Plus size={20} />
                  Restaurar paineis
                </button>
              )}
            </div>
          )}
          {view === 'network' && (
            <>
              <Widget
                title="Rede de estados"
                sub={graph.phase ?? 'Sem modelo'}
                tools={
                  <Badge tone={active ? 'green' : 'neutral'}>{active ? 'Live' : 'Snapshot'}</Badge>
                }
                footer={
                  <>
                    <span className="mono">LAYOUT DE MODELOS FORMAIS / NAO ANATOMIA NEURAL</span>
                    <span className="push" />
                    <button
                      className="secondary"
                      onClick={() => download(graph, 'axon-graph.json')}
                    >
                      <ArrowDownToLine size={14} />
                      Exportar grafo
                    </button>
                  </>
                }
              >
                <div className="instrument">
                  <GraphView graph={graph} events={events} active={active} onSelect={nodeInspect} />
                </div>
              </Widget>
              <div className="network-models">
                {graph.models.map((m, i) => (
                  <button
                    key={m.context}
                    className="network-model-item"
                    onClick={() => inspect(m.context, m)}
                  >
                    <i className="dot" style={{ background: palette[i % 5] }} />
                    <strong>{m.name.toUpperCase()}</strong>
                    <span className="mono">
                      {m.states} ESTADOS / {bytes(m.bytes)}
                    </span>
                    <ArrowUpRight size={15} />
                  </button>
                ))}
              </div>
            </>
          )}
          {view === 'evolution' && (
            <>
              <div className="evolution-toolbar">
                <Badge>Experimentos armazenados</Badge>
                <label className="select-pill">
                  <GitBranch size={15} />
                  <select
                    aria-label="Selecionar campanha"
                    value={campaign?.id ?? ''}
                    onChange={(e) => setCampaignId(e.target.value)}
                  >
                    {axon.campaigns.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.seed} / {c.mode}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {campaign ? (
                <Evolution campaign={campaign} inspect={inspect} />
              ) : (
                <Empty icon={FlaskConical} title="Nenhum experimento reservado encontrado" />
              )}
            </>
          )}
          {view === 'laboratory' && (
            <Laboratory
              data={s}
              connected={axon.connected}
              sending={axon.sending}
              onRun={axon.run}
              onCancel={() => axon.control('cancel')}
              inspect={inspect}
            />
          )}
          {view === 'knowledge' && (
            <>
              <div className="dashboard-grid">
                {knowledge}
                {memory}
              </div>
              <section className="query-section">
                <h2>Investigar fontes locais</h2>
                <form
                  className="query-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    invoke(() => axon.run({ mode: 'knowledge', query }));
                  }}
                >
                  <Search size={20} />
                  <input
                    aria-label="Consulta de conhecimento"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="O que investigar?"
                    maxLength={2000}
                  />
                  <button
                    className="primary"
                    disabled={!s?.indexed || !query.trim() || active || axon.sending}
                  >
                    <ArrowRight size={16} />
                    Investigar
                  </button>
                </form>
                {!s?.indexed && <span className="mono muted">INDICE LOCAL AINDA NAO CRIADO</span>}
                {j?.mode === 'knowledge' && (
                  <pre className="output-code">
                    {j.output || 'Aguardando resposta do nucleo...'}
                  </pre>
                )}
              </section>
            </>
          )}
          {view === 'sources' && <section className="unframed-section">{sourceList()}</section>}
          {view === 'runs' && (
            <>
              <div className="runs-layout">
                <div className="run-list">
                  <h2>
                    Esta sessao <span className="muted">/ {s?.runs.length ?? 0}</span>
                  </h2>
                  {s?.runs.map((r) => (
                    <button
                      className={`run-list-item ${j?.id === r.id ? 'selected' : ''}`}
                      key={r.id}
                      onClick={() => inspect(`Execucao ${r.id}`, r)}
                    >
                      <Activity size={17} />
                      <div>
                        <strong>{label[r.mode]}</strong>
                        <span className="mono">
                          {r.id.slice(0, 8)} / {new Date(r.started).toLocaleTimeString('pt-BR')}
                        </span>
                      </div>
                      <Badge
                        tone={
                          r.status === 'complete'
                            ? 'green'
                            : r.status === 'error'
                              ? 'red'
                              : 'neutral'
                        }
                      >
                        {label[r.status]}
                      </Badge>
                    </button>
                  ))}
                  {!s?.runs.length && (
                    <Empty icon={Activity} title="Nenhuma execucao nesta sessao" />
                  )}
                </div>
                <div className="log-panel">
                  <div className="log-header">
                    <h2>Eventos do nucleo</h2>
                    <Segments
                      items={['TODOS', 'QUERIES', 'RESULTADOS']}
                      value={tab}
                      onChange={setTab}
                      label="Filtro do log"
                    />
                    <span className="push" />
                    {j && (
                      <a className="secondary" href={`/api/export?id=${j.id}`} download>
                        <ArrowDownToLine size={14} />
                        JSONL
                      </a>
                    )}
                  </div>
                  <div className="event-table">
                    <div className="event-table-heading mono">
                      <span>SEQ</span>
                      <span>EVENTO</span>
                      <span>REGISTRO</span>
                    </div>
                    {events
                      .filter(
                        (e) =>
                          tab === 'TODOS' ||
                          (tab === 'QUERIES' && e.event === 'query') ||
                          (tab === 'RESULTADOS' &&
                            ['result', 'output', 'end', 'error'].includes(e.event)),
                      )
                      .slice(-200)
                      .reverse()
                      .map((e) => (
                        <button
                          className="event-table-row"
                          key={e.seq}
                          onClick={() => inspect(`Evento ${e.seq}`, e)}
                        >
                          <span className="mono muted">{e.seq}</span>
                          <span className={`event-type ${e.event === 'error' ? 'red' : ''}`}>
                            {e.event}
                          </span>
                          <span className="truncate">{describe(e)}</span>
                          <ArrowUpRight size={13} />
                        </button>
                      ))}
                  </div>
                  <div className="log-footer mono">
                    JANELA RECENTE / LOG JSONL INTEGRAL / {fmt(j?.trace.gaps ?? 0)} LACUNAS DO
                    NUCLEO
                  </div>
                  {j?.output && <pre className="output-code">{j.output}</pre>}
                </div>
              </div>
            </>
          )}
          <footer className="page-footer mono">
            <span>
              <i className={`status-dot ${online ? 'on' : ''}`} /> AXON LOCAL
            </span>
            <span>
              {!axon.connected
                ? 'CONEXAO INTERROMPIDA'
                : active
                  ? j?.status === 'paused'
                    ? 'PAUSADO'
                    : j?.status === 'pausing'
                      ? 'PAUSANDO'
                      : 'PROCESSANDO'
                  : j
                    ? 'EXECUCAO ENCERRADA'
                    : 'SEM EXECUCAO ATIVA'}
            </span>
            <span>{s?.session?.slice(0, 8) ?? 'CONECTANDO'}</span>
          </footer>
        </main>
      </div>
      {modal === 'run' && (
        <Modal title="Nova execucao" onClose={() => setModal(null)}>
          <RunForm
            sending={axon.sending}
            disabled={!online || active}
            indexed={!!s?.indexed}
            onRun={async (options) => {
              await axon.run(options);
              setModal(null);
            }}
          />
        </Modal>
      )}
      {modal === 'panels' && (
        <Modal title="Paineis do workspace" onClose={() => setModal(null)}>
          <div className="settings-list">
            {panels.map((p) => (
              <label key={p}>
                <span>{p}</span>
                <input
                  className="switch"
                  type="checkbox"
                  checked={!hidden.includes(p)}
                  onChange={(e) =>
                    setHidden((v) => (e.target.checked ? v.filter((n) => n !== p) : [...v, p]))
                  }
                />
              </label>
            ))}
          </div>
        </Modal>
      )}
      {modal === 'settings' && (
        <Modal title="Preferencias" onClose={() => setModal(null)}>
          <div className="settings-list">
            <label>
              <span>Tema escuro</span>
              <input
                className="switch"
                type="checkbox"
                checked={dark}
                onChange={(e) => setDark(e.target.checked)}
              />
            </label>
            <label>
              <span>Navegacao compacta</span>
              <input
                className="switch"
                type="checkbox"
                checked={collapsed}
                onChange={(e) => setCollapsed(e.target.checked)}
              />
            </label>
            <div>
              <span>Servidor</span>
              <span className="mono">127.0.0.1:{s?.port ?? '--'}</span>
            </div>
            <div>
              <span>Runtime da interface</span>
              <span className="mono">BUN / REACT / TYPESCRIPT</span>
            </div>
            <div>
              <span>Integridade do nucleo</span>
              <span className="mono">{fmt(j?.trace.gaps ?? 0)} LACUNAS</span>
            </div>
          </div>
          <button
            className="secondary"
            onClick={() => {
              setHidden([]);
              axon.refresh();
            }}
          >
            <LayoutGrid size={15} />
            Restaurar paineis e atualizar
          </button>
        </Modal>
      )}
      {modal === 'search' && (
        <Modal title="Buscar no workspace" wide onClose={() => setModal(null)}>
          <label className="search-field global-search">
            <Search size={18} />
            <input
              autoFocus
              placeholder="Arquivo, modelo ou pagina..."
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              aria-label="Busca global"
            />
          </label>
          <div className="search-results">
            {views
              .filter((v) => v.name.toLowerCase().includes(sourceFilter.toLowerCase()))
              .map((v) => (
                <button
                  key={v.id}
                  onClick={() => {
                    go(v.id);
                    setModal(null);
                  }}
                >
                  <v.icon size={18} />
                  {v.name}
                  <ArrowRight size={14} />
                </button>
              ))}
            {(s?.sources ?? [])
              .filter(
                (f) => sourceFilter && f.path.toLowerCase().includes(sourceFilter.toLowerCase()),
              )
              .slice(0, 20)
              .map((f) => (
                <button
                  key={f.id}
                  onClick={() => {
                    setModal(null);
                    void openSource(f);
                  }}
                >
                  <FileCode2 size={18} />
                  <span className="truncate">{f.path}</span>
                  <ArrowUpRight size={14} />
                </button>
              ))}
          </div>
        </Modal>
      )}
      {modal === 'help' && (
        <Modal title="Escopo e integridade" onClose={() => setModal(null)}>
          <div className="scope-copy">
            <ShieldCheck size={26} />
            <h3>Dados locais. Limites explicitos.</h3>
            <p>
              Runtime, consultas e auditorias vem do executavel Rust iniciado por este workspace. O
              grafo representa estados e transicoes de modelos formais, nao neuronios biologicos ou
              pensamentos.
            </p>
            <p>
              Seeds e evolucao apresenta experimentos salvos, nao treinamento acontecendo agora.
              Latencias no observador incluem instrumentacao e nao sao benchmarks de eficiencia.
            </p>
            <p>
              RSS atual, consumo de energia, conversacao livre e voz nao estao instrumentados.
              Outros processos AXON nao sao monitorados automaticamente.
            </p>
            <p>
              As tarefas de solver e conhecimento fornecem entrada e saida, sem um trace completo do
              processamento interno.
            </p>
          </div>
        </Modal>
      )}
      {inspection && (
        <Modal title={inspection.title} wide onClose={() => setInspection(null)}>
          <div className="inspection-toolbar">
            <span className="mono">DADOS ORIGINAIS / SEM RESUMO GERADO</span>
            <button
              className="secondary"
              onClick={() => download(inspection.value, 'axon-inspection.json')}
            >
              <ArrowDownToLine size={14} />
              Exportar
            </button>
          </div>
          <pre className="inspection-code">
            {typeof inspection.value === 'string'
              ? inspection.value
              : JSON.stringify(inspection.value ?? null, null, 2)}
          </pre>
        </Modal>
      )}
    </div>
  );
}

function RunForm({
  onRun,
  sending,
  disabled,
  indexed,
}: {
  onRun: (v: RunOptions) => Promise<void>;
  sending: boolean;
  disabled: boolean;
  indexed: boolean;
}) {
  const [mode, setMode] = useState<Mode>('transfer'),
    [seed, setSeed] = useState(719319),
    [dimensions, setDimensions] = useState(3),
    [novel, setNovel] = useState(false),
    [paused, setPaused] = useState(false),
    [task, setTask] = useState(taskDefault),
    [query, setQuery] = useState('BehaviorMemory drift context');
  const [error, setError] = useState('');
  const finite = ['transfer', 'factor'].includes(mode);
  async function submit(e: FormEvent) {
    e.preventDefault();
    setError('');
    try {
      await onRun({
        mode,
        ...(finite
          ? { seed, dimensions, novel, paused }
          : mode === 'math'
            ? { task }
            : mode === 'knowledge'
              ? { query }
              : {}),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  return (
    <form className="run-form" onSubmit={submit}>
      <label>
        Experimento
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
          aria-label="Experimento"
        >
          {(['transfer', 'factor', 'math', 'index', 'knowledge'] as const).map((m) => (
            <option key={m} value={m}>
              {label[m]}
            </option>
          ))}
        </select>
      </label>
      {finite && (
        <>
          <label>
            Seed
            <input
              type="number"
              aria-label="Seed"
              min={1}
              max={2147483647}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              required
            />
          </label>
          {mode === 'factor' && (
            <label>
              Canais <span className="mono">{dimensions}</span>
              <input
                type="range"
                min={2}
                max={5}
                step={1}
                value={dimensions}
                onChange={(e) => setDimensions(Number(e.target.value))}
                aria-label="Canais"
                style={colored('var(--blue)')}
              />
              <div className="range-labels mono">
                <span>2</span>
                <span>3</span>
                <span>4</span>
                <span>5</span>
              </div>
            </label>
          )}
          <label className="check-row">
            <span>Mecanismo desconhecido</span>
            <input
              className="switch"
              type="checkbox"
              checked={novel}
              onChange={(e) => setNovel(e.target.checked)}
            />
          </label>
          <label className="check-row">
            <span>Iniciar pausado</span>
            <input
              className="switch"
              type="checkbox"
              checked={paused}
              onChange={(e) => setPaused(e.target.checked)}
            />
          </label>
        </>
      )}
      {mode === 'math' && (
        <label>
          Tarefa AXON
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            aria-label="Tarefa AXON"
            spellCheck={false}
            maxLength={32768}
            rows={8}
          />
        </label>
      )}
      {mode === 'knowledge' && (
        <label>
          Consulta
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            maxLength={2000}
            required
            aria-label="Consulta"
          />
          {!indexed && <span className="form-error">Crie o indice local primeiro.</span>}
        </label>
      )}
      {mode === 'index' && (
        <div className="index-paths mono">
          <Folder size={16} />
          DOCS / SRC / TESTS
        </div>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <footer>
        <span className="mono muted">RUST / EXECUCAO REAL LOCAL</span>
        <button
          className="primary"
          disabled={disabled || sending || (mode === 'knowledge' && !indexed)}
          type="submit"
        >
          <Play size={14} />
          {sending ? 'Iniciando...' : 'Executar'}
        </button>
      </footer>
    </form>
  );
}
function Evolution({
  campaign: c,
  inspect,
}: {
  campaign: Campaign;
  inspect: (title: string, v: unknown) => void;
}) {
  const admitted = c.cases.filter((v) => v.status === 'Admitted'),
    correct = c.cases.reduce((n, v) => n + v.correct, 0),
    heldout = c.cases.reduce((n, v) => n + v.heldout, 0),
    capsules = admitted.reduce((n, v) => n + (v.capsule_bytes ?? 0), 0);
  return (
    <Widget
      title="Seeds e evolucao"
      sub={String(c.seed)}
      tools={
        <button className="select-pill" onClick={() => inspect('Medicao original', c)}>
          <FlaskConical size={15} />
          {c.mode}
          <SlidersHorizontal size={14} />
        </button>
      }
      footer={
        <>
          <span className="mono">RESERVA SINTETICA / NAO AGI / NAO LIVE</span>
          <span className="push" />
          <button className="primary" onClick={() => download(c, `${c.id}`)}>
            <ArrowDownToLine size={14} />
            Exportar resultado
          </button>
        </>
      }
    >
      <div className="instrument evolution-summary">
        <div>
          <span className="eyebrow">TAREFAS ADMITIDAS</span>
          <div className="metric-value">
            {admitted.length}
            <small>/ {c.cases.length}</small>
          </div>
          <p className="mono muted">
            {fmt(correct)} CORRETOS / {fmt(heldout)} CASOS CONFIGURADOS
          </p>
        </div>
        <div className="evolution-stats">
          <div>
            <span className="eyebrow">PROCESSO COMPLETO</span>
            <strong>{fmt(c.lifecycleMs)} ms</strong>
          </div>
          <div>
            <span className="eyebrow">RSS AMOSTRADO</span>
            <strong>{c.sampledRss == null ? 'Nao medido' : bytes(c.sampledRss)}</strong>
          </div>
          <div>
            <span className="eyebrow">CAPSULAS ADMITIDAS</span>
            <strong>{bytes(capsules)}</strong>
          </div>
          <div>
            <span className="eyebrow">BINARIO DO LAB</span>
            <strong>{c.binaryBytes == null ? 'Nao registrado' : bytes(c.binaryBytes)}</strong>
          </div>
        </div>
      </div>
      <div className="instrument evolution-cases">
        <div className="evolution-heading mono">
          <span>CAPACIDADE</span>
          <span>GERACAO</span>
          <span>OPERACOES</span>
          <span>RESERVA</span>
          <span>ESTADO</span>
        </div>
        {c.cases.map((v) => (
          <button
            key={v.case}
            className="evolution-row"
            onClick={() => inspect(taskNames[v.case] ?? `Caso ${v.case}`, v)}
          >
            <span>
              <i className="dot" style={{ background: palette[v.case % 5] }} />
              <strong>{taskNames[v.case] ?? `Caso ${v.case}`}</strong>
              <code>{v.expression || 'Sem programa admitido'}</code>
            </span>
            <span className="mono">{v.generation}</span>
            <span className="mono">{fmt(v.instructions)}</span>
            <span className="mono">
              {fmt(v.correct)} / {fmt(v.heldout)}
            </span>
            <Badge
              tone={
                v.status === 'Admitted' ? 'green' : v.status.includes('Refuted') ? 'red' : 'neutral'
              }
            >
              {v.status === 'Admitted'
                ? 'Admitido'
                : v.status === 'WorkLimit'
                  ? 'Limite'
                  : v.status.includes('Refuted')
                    ? 'Refutado'
                    : 'Abstencao'}
            </Badge>
          </button>
        ))}
      </div>
      <div className="instrument">
        <div className="section-top">
          <h3>Proveniencia</h3>
          <Badge>Historico</Badge>
        </div>
        <p className="mono muted break-all">{c.id}</p>
        <div className="instrument-bottom mono">
          <span>{new Date(c.modified).toLocaleString('pt-BR')}</span>
          <span>ARITMETICA / GRAMATICA / ROTULOS FORNECIDOS</span>
        </div>
      </div>
    </Widget>
  );
}
