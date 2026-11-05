import { useEffect, useState, type FormEvent } from 'react';
import { ArrowDownToLine, ArrowRight, Workflow, Play, Search, Square } from 'lucide-react';
import { Relational } from './Relational';
import { Nursery } from './Nursery';
import { Badge, Empty, Sparkline, Widget } from './components';
import { busy, bytes, fmt, label } from './useAxon';
import type { Job, Mode, RecordEvent, RunOptions, Snapshot } from './types';
import { Cumulative } from './Cumulative';

const cases = [
  'Quadrado',
  'Produto escalado',
  'Hipotenusa 2D',
  'Discriminante',
  'Raiz positiva',
  'Raiz negativa',
  'Norma 3D',
  'Distancia entre pontos',
];
const operators = ['+', '-', '*', '/', 'sqrt'];
const constructiveCases = [
  'Quadrado',
  'Norma 2D',
  'Distancia 2D',
  'Razao de normas',
  'Produto composto',
  'Seno / controle negativo',
];
const arithmeticCases = ['Soma', 'Subtracao', 'Produto', 'Divisao', ...constructiveCases];
const constructiveValidation = (event: RecordEvent) =>
  ['prospective-final', 'post-selection', 'development-recheck'].includes(event.phase ?? '');

function Genome({
  record,
  inspect,
}: {
  record: RecordEvent;
  inspect: (title: string, value: unknown) => void;
}) {
  const arity = record.arity ?? 0,
    wiring = record.wiring ?? [],
    base = arity + 4;
  const genes = Array.from({ length: wiring.length / 7 }, (_, i) => wiring.slice(i * 7, i * 7 + 7));
  const usedSignals = new Set([
    record.output_signal,
    ...genes.filter((g) => g[6] === 1).flatMap((g) => g.slice(2, 2 + g[1])),
  ]);
  const location = (id: number) => ({ x: 48 + (id % 8) * 98, y: 40 + Math.floor(id / 8) * 76 });
  const height = 80 + Math.ceil((base + genes.length) / 8) * 76;
  return (
    <div className="genome-scroll">
      <svg
        viewBox={`0 0 820 ${height}`}
        role="group"
        aria-label="Genoma adquirido, conexoes e genes ativos"
      >
        {genes.flatMap((g, i) =>
          g.slice(2, 2 + g[1]).map((input, a) => {
            const from = location(input),
              to = location(base + i);
            return (
              <path
                key={`${i}-${a}`}
                d={`M${from.x},${from.y + 10} L${to.x},${to.y - 10}`}
                stroke={g[6] ? 'var(--teal)' : 'var(--line)'}
                strokeWidth={g[6] ? 1.3 : 0.7}
                opacity={g[6] ? 0.6 : 0.3}
                fill="none"
              />
            );
          }),
        )}
        {Array.from({ length: base + genes.length }, (_, id) => {
          const pos = location(id),
            g = genes[id - base],
            active = g ? g[6] === 1 : usedSignals.has(id);
          const text =
            id < arity
              ? `x${id}`
              : id < base
                ? ['-1', '0', '1', '2'][id - arity]
                : (operators[g[0]] ?? `f${g[0] - 5}`);
          return (
            <g
              key={id}
              role="button"
              tabIndex={0}
              aria-label={`Sinal ${id}, ${text}, ${active ? 'ativo' : 'inativo'}`}
              onClick={() =>
                inspect(`Sinal ${id}`, { id, gene: g ?? null, output: record.output_signal === id })
              }
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  inspect(`Sinal ${id}`, { id, gene: g ?? null });
                }
              }}
            >
              <circle
                cx={pos.x}
                cy={pos.y}
                r={14}
                fill="var(--surface)"
                stroke={
                  record.output_signal === id
                    ? 'var(--orange)'
                    : active
                      ? 'var(--teal)'
                      : 'var(--line)'
                }
                strokeWidth={record.output_signal === id ? 2 : 1}
              />
              <text x={pos.x} y={pos.y + 4} textAnchor="middle" fill="var(--ink)" fontSize="10">
                {text}
              </text>
              <text x={pos.x} y={pos.y + 30} textAnchor="middle" fill="var(--muted)" fontSize="9">
                s{id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function SeedPopulation({
  job,
  inspect,
}: {
  job: Job;
  inspect: (title: string, value: unknown) => void;
}) {
  return (
    <Widget
      title="Populacao e descendencia"
      sub="SELECAO POR DESENVOLVIMENTO"
      onInspect={() => inspect('Populacao de genotipos', job.candidates)}
      footer={<span className="mono">RETENCAO + CAPACIDADES + TRABALHO + BYTES + INSTRUCOES</span>}
    >
      <div
        className="lab-table-scroll population-scroll"
        tabIndex={0}
        role="region"
        aria-label="Populacao de seeds"
      >
        <table className="lab-table">
          <thead>
            <tr>
              <th>Candidato / genotipo</th>
              <th>Rodada</th>
              <th>Capacidades</th>
              <th>{job.mode === 'genome' ? 'Procedimentos' : 'Modulos'}</th>
              <th>Trabalho</th>
              <th>DNA</th>
              <th>Replay</th>
              <th>Frente</th>
              <th>Artefato</th>
            </tr>
          </thead>
          <tbody>
            {job.candidates.map((c) => (
              <tr key={c.candidate}>
                <td>
                  <button
                    className="lab-case"
                    aria-label={`Inspecionar candidato ${c.candidate}`}
                    onClick={() => inspect(`Candidato ${c.candidate}`, c)}
                  >
                    #{c.candidate} / {c.genotype?.slice(0, 8)}
                  </button>
                </td>
                <td>{fmt(c.generation)}</td>
                <td>
                  {fmt(c.solved)} / {fmt(c.tasks ?? c.checked)}
                </td>
                <td>{fmt(c.modules ?? c.genes)}</td>
                <td>{fmt(c.work)}</td>
                <td>
                  {c.capsule_bytes == null && c.encoded_bytes == null
                    ? '--'
                    : bytes(c.capsule_bytes ?? c.encoded_bytes ?? 0)}
                </td>
                <td>
                  {c.retention_ok === true ? 'Valido' : c.retention_ok === false ? 'Falhou' : '--'}
                </td>
                <td>
                  <Badge tone={c.frontier ? 'green' : 'neutral'}>
                    {c.frontier === undefined ? 'Avaliando' : c.frontier ? 'Selecionado' : 'Fora'}
                  </Badge>
                </td>
                <td>
                  {c.event === 'seed_evaluation' && (
                    <a
                      title={`Baixar DNA do candidato ${c.candidate}`}
                      aria-label={`Baixar DNA do candidato ${c.candidate}`}
                      href={`/api/artifact?id=${job.id}&name=${encodeURIComponent(c.artifact ?? `candidate-${c.candidate}.axdna`)}`}
                    >
                      <ArrowDownToLine size={15} />
                    </a>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Widget>
  );
}
function ArtifactLinks({ run, files }: { run: string; files: { name: string; bytes: number }[] }) {
  const [query, setQuery] = useState('');
  useEffect(() => setQuery(''), [run]);
  const matches = files.filter((file) =>
    file.name.toLowerCase().includes(query.trim().toLowerCase()),
  );
  return (
    <>
      <div className="artifact-toolbar">
        <label className="search-field">
          <Search size={15} aria-hidden="true" />
          <input
            type="search"
            aria-label="Buscar artefatos"
            placeholder="Buscar artefatos"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <span className="mono">
          {fmt(matches.length)} / {fmt(files.length)}
        </span>
      </div>
      <div className="lab-files" role="region" aria-label="Artefatos exportaveis" tabIndex={0}>
        {matches.map((file) => (
          <a
            key={file.name}
            className="secondary"
            href={`/api/artifact?id=${run}&name=${encodeURIComponent(file.name)}`}
          >
            <ArrowDownToLine size={14} />
            <span>{file.name}</span>
            <small>{bytes(file.bytes)}</small>
          </a>
        ))}
      </div>
    </>
  );
}

export function Laboratory({
  data,
  connected,
  sending,
  onRun,
  onCancel,
  inspect,
}: {
  data: Snapshot | null;
  connected: boolean;
  sending: boolean;
  onRun: (options: RunOptions) => Promise<unknown>;
  onCancel: () => Promise<unknown>;
  inspect: (title: string, value: unknown) => void;
}) {
  const [mode, setMode] = useState<Mode>('evolution'),
    [seed, setSeed] = useState('82641973'),
    [count, setCount] = useState('1000');
  const [strategy, setStrategy] = useState<RunOptions['strategy']>('balanced'),
    [curriculum, setCurriculum] = useState<RunOptions['curriculum']>('transfer');
  const [constructiveStrategy, setConstructiveStrategy] =
      useState<RunOptions['strategy']>('diverse'),
    [constructiveCurriculum, setConstructiveCurriculum] =
      useState<RunOptions['curriculum']>('arithmetic');
  const [suite, setSuite] = useState<RunOptions['suite']>('rust'),
    [selected, setSelected] = useState(''),
    [geneCase, setGeneCase] = useState('');
  const [validation, setValidation] = useState<RunOptions['validation']>('development');
  const [jsonCurriculum, setJsonCurriculum] = useState('mixed-128-v1');
  const [files, setFiles] = useState<{ name: string; bytes: number }[]>([]),
    [fileError, setFileError] = useState('');
  const revisions = files.filter((f) => f.name.startsWith('revision-'));
  const current = data?.current ?? null,
    job = selected ? (data?.runs.find((r) => r.id === selected) ?? current) : current;
  const running = connected && busy(current),
    workflow = data?.workflows.find((w) => w.id === mode);
  const genomes = job?.cases.filter((c) => c.wiring) ?? [];
  const genome = genomes.find((c) => String(c.case) === geneCase) ?? genomes.at(-1);
  const procedures = job?.cases.filter((c) => c.steps) ?? [];
  const procedure = procedures.find((c) => String(c.case) === geneCase) ?? procedures.at(-1);
  const losses = (job?.samples ?? [])
    .filter((e) => e.case === genome?.case && e.loss != null)
    .map((e) => e.loss!);
  const channels = [
    ...new Map(
      (job?.samples ?? []).filter((e) => e.channel != null).map((e) => [e.channel, e]),
    ).values(),
  ];
  useEffect(() => {
    let alive = true;
    setFiles([]);
    setFileError('');
    if (job?.id && !busy(job))
      fetch(`/api/artifacts?id=${job.id}`)
        .then(async (r) => {
          if (!r.ok) throw new Error('Artefatos indisponiveis');
          return r.json();
        })
        .then((r) => {
          if (alive) setFiles(r.files);
        })
        .catch((e) => {
          if (alive) setFileError(String(e));
        });
    return () => {
      alive = false;
    };
  }, [job?.id, job?.status]);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSelected('');
    await onRun({
      mode,
      ...(mode === 'structural' ? {} : { seed: Number(seed) }),
      ...(mode === 'tests' ? { suite } : mode === 'structural' ? {} : { count: Number(count) }),
      ...(mode === 'evolution' ? { strategy, curriculum } : {}),
      ...(mode === 'constructive'
        ? { strategy: constructiveStrategy, curriculum: constructiveCurriculum }
        : {}),
      ...(mode === 'curriculum' || mode === 'structural' ? { curriculum: jsonCurriculum } : {}),
      ...(['seed', 'controller', 'genome', 'language', 'cumulative', 'semantic'].includes(mode)
        ? { validation }
        : {}),
    }).catch(() => {});
  };
  const progress = job?.progress,
    result = job?.result;
  const constructiveNames =
    job?.options?.curriculum === 'arithmetic' ? arithmeticCases : constructiveCases;
  const validations =
    job?.mode === 'constructive' || job?.mode === 'curriculum'
      ? job.cases.filter(
          (c) =>
            (job.mode === 'curriculum' || constructiveValidation(c)) &&
            Number.isFinite(c.correct) &&
            Number.isFinite(c.checked),
        )
      : [];
  const observedTotals = validations.length
    ? validations.reduce(
        (sum, c) => ({ correct: sum.correct + c.correct!, checked: sum.checked + c.checked! }),
        { correct: 0, checked: 0 },
      )
    : null;
  const modelBytes =
    genome?.capsule_bytes ??
    (job?.mode === 'constructive'
      ? files.find((f) => f.name === `task-${genome?.case}.axevo`)?.bytes
      : undefined);
  const activeGenes =
    genome?.active_genes ??
    (genome?.wiring
      ? Array.from(
          { length: genome.wiring.length / 7 },
          (_, i) => genome.wiring![i * 7 + 6],
        ).filter((v) => v === 1).length
      : undefined);
  const residency = job?.samples.findLast((e) => e.peak_bytes != null) ?? progress;
  const algorithmSearch =
    job?.algorithmInvestigation ??
    job?.samples.findLast((e) => e.event === 'algorithm_investigation' && e.templates != null);
  const algorithmDrift =
    job?.algorithmDrift ?? job?.samples.findLast((e) => e.event === 'algorithm_drift');
  const algorithmInquiry =
    job?.algorithmInquiry ??
    job?.samples.findLast(
      (e) =>
        e.event === 'algorithm_investigation' &&
        ['query_selected', 'query_observed'].includes(e.phase ?? ''),
    );
  const sessionTurn = job?.cases.findLast(
    (e) => e.method === 'Session::turn' && e.variant === 'context',
  );
  const sessionInterpretation = job?.cases.findLast(
    (e) => e.method === 'Session::ground' && e.variant === 'context',
  );
  return (
    <div className="laboratory">
      <form className="lab-controls" onSubmit={submit}>
        <label>
          Experimento
          <select
            aria-label="Experimento"
            value={mode}
            onChange={(e) => {
              const next = e.target.value as Mode;
              setMode(next);
              if (next === 'continuous') setCount('3000');
              if (next === 'seed' && seed === '82641973') setSeed('70143863');
              if (next === 'controller' && seed === '82641973') setSeed('72918463');
              if (next === 'genome') setSeed('84371921');
              if (next === 'language') setSeed('71582347');
              if (next === 'cumulative') setSeed('39172841');
              if (next === 'semantic') setSeed('61423917');
              if (next === 'nursery') {
                setSeed('61423917');
                setCount('1000');
              }
              if (next === 'constructive') {
                setSeed('75218341');
                setCount('5000');
              }
              if (next === 'curriculum') {
                setSeed('75218341');
                setCount('5000');
              }
              if (next === 'structural') setJsonCurriculum('compositional-v1');
            }}
          >
            {(data?.workflows ?? []).map((w) => (
              <option key={w.id} value={w.id} disabled={!w.available}>
                {w.label}
              </option>
            ))}
          </select>
        </label>
        {mode !== 'integration' && mode !== 'structural' && (
          <label>
            {mode === 'constructive' || mode === 'curriculum' ? 'RNG da busca' : 'Seed aleatoria'}
            <input
              type="number"
              min="1"
              max="2147483647"
              required
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </label>
        )}
        {mode === 'tests' ? (
          <label>
            Suite
            <select
              aria-label="Suite"
              value={suite}
              onChange={(e) => setSuite(e.target.value as RunOptions['suite'])}
            >
              <option value="rust">Rust completo</option>
              <option value="rust-minimal">Rust sem features</option>
              <option value="ui">Unidades e API</option>
              <option value="check">Clippy</option>
            </select>
          </label>
        ) : mode === 'integration' || mode === 'structural' ? null : (
          <label>
            {mode === 'nursery'
              ? 'Novas seeds por lote'
              : mode === 'constructive'
                ? 'Novos grafos por tarefa'
                : mode === 'curriculum'
                  ? 'Limite de grafos por etapa'
                  : mode === 'continuous'
                    ? 'Observacoes por canal'
                    : 'Validacoes por caso'}
            <input
              type="number"
              required
              min={mode === 'continuous' ? 1000 : 100}
              max="10000"
              value={count}
              onChange={(e) => setCount(e.target.value)}
            />
          </label>
        )}
        {mode === 'evolution' && (
          <>
            <label>
              Busca
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as RunOptions['strategy'])}
              >
                <option value="balanced">Composicao balanceada</option>
                <option value="hybrid">Hibrida anterior</option>
                <option value="mutation-only">Apenas mutacoes</option>
                <option value="no-inheritance">Sem heranca</option>
              </select>
            </label>
            <label>
              Curriculo
              <select
                value={curriculum}
                onChange={(e) => setCurriculum(e.target.value as RunOptions['curriculum'])}
              >
                <option value="transfer">8 problemas / transferencia</option>
                <option value="core">7 problemas / controle anterior</option>
              </select>
            </label>
          </>
        )}
        {(mode === 'curriculum' || mode === 'structural') && (
          <label>
            Curriculo JSON
            <input
              aria-label="Curriculo JSON"
              value={jsonCurriculum}
              required
              maxLength={64}
              pattern={'[a-z0-9][a-z0-9_\\-]{0,63}'}
              onChange={(e) => setJsonCurriculum(e.target.value)}
            />
          </label>
        )}
        {mode === 'constructive' && (
          <>
            <label>
              Busca construtiva
              <select
                aria-label="Busca construtiva"
                value={constructiveStrategy}
                onChange={(e) => setConstructiveStrategy(e.target.value as RunOptions['strategy'])}
              >
                <option value="reinforced">Especialistas + reforco de busca</option>
                <option value="diverse">Especialistas sem reforco</option>
                <option value="elitist">Erro medio / controle</option>
              </select>
            </label>
            <label>
              Curriculo construtivo
              <select
                aria-label="Curriculo construtivo"
                value={constructiveCurriculum}
                onChange={(e) =>
                  setConstructiveCurriculum(e.target.value as RunOptions['curriculum'])
                }
              >
                <option value="arithmetic">9 etapas + controle negativo</option>
                <option value="legacy">5 etapas + controle negativo</option>
              </select>
            </label>
          </>
        )}
        {['seed', 'controller', 'genome', 'language', 'cumulative', 'semantic'].includes(mode) && (
          <label>
            Protocolo
            <select
              aria-label="Protocolo"
              value={validation}
              onChange={(e) => setValidation(e.target.value as RunOptions['validation'])}
            >
              <option value="development">Desenvolvimento</option>
              <option value="reserved">Reserva final</option>
            </select>
          </label>
        )}
        <div className="lab-commands">
          <button
            className="primary"
            disabled={!connected || !workflow?.available || running || sending}
            type="submit"
          >
            <Play size={15} />
            Executar
          </button>
          <button
            className="secondary"
            type="button"
            disabled={!running || sending}
            onClick={() => void onCancel().catch(() => {})}
          >
            <Square size={15} />
            Cancelar
          </button>
        </div>
      </form>
      <div className="evolution-toolbar">
        <Badge tone={running ? 'green' : 'neutral'}>
          {job ? label[job.status] : 'Sem execucao'}
        </Badge>
        {job && !busy(job) && <Badge>Historico</Badge>}
        <select
          aria-label="Execucao do laboratorio"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          <option value="">Execucao atual</option>
          {data?.runs.map((r) => (
            <option value={r.id} key={r.id}>
              {label[r.mode]} / {r.id.slice(0, 8)} / {label[r.status]}
            </option>
          ))}
        </select>
        {job && (
          <a className="secondary" href={`/api/export?id=${job.id}`}>
            <ArrowDownToLine size={15} />
            JSONL completo
          </a>
        )}
      </div>
      <Widget
        title="Aquisicao e verificacao"
        sub={job?.workflow?.telemetry ?? 'Sem checkpoint'}
        onInspect={() => inspect('Execucao instrumentada', job)}
        footer={
          <>
            <span className="mono">
              {job?.recovered
                ? 'RECUPERADO DO DISCO'
                : running && job?.id === current?.id
                  ? 'PROCESSO LOCAL ATIVO'
                  : 'REGISTRO LOCAL'}
            </span>
            <span className="push" />
            <span className="mono">EXIT {fmt(job?.exitCode)}</span>
          </>
        }
      >
        <div className="lab-metrics instrument">
          <div>
            <span className="eyebrow">FASE</span>
            <strong>{progress?.phase ?? progress?.event ?? '--'}</strong>
          </div>
          <div>
            <span className="eyebrow">TRABALHO / ULTIMO CHECKPOINT</span>
            <strong>{fmt(progress?.work ?? job?.cases.at(-1)?.work)}</strong>
          </div>
          <div>
            <span className="eyebrow">AVALIACOES / ULTIMO CASO</span>
            <strong>{fmt(progress?.evaluations ?? job?.cases.at(-1)?.evaluations)}</strong>
          </div>
          <div>
            <span className="eyebrow">
              {job?.mode === 'seed' ? 'REPLAY DOS MODULOS / CORRETOS' : 'VALIDACAO / CORRETOS'}
            </span>
            <strong>
              {fmt(observedTotals?.correct ?? result?.correct)} /{' '}
              {fmt(
                observedTotals?.checked ??
                  result?.heldout ??
                  result?.scored ??
                  result?.checked ??
                  result?.total,
              )}
            </strong>
          </div>
          <div>
            <span className="eyebrow">REGISTROS DO NUCLEO</span>
            <strong>{fmt(job?.trace.lastCoreSeq)}</strong>
          </div>
          <div>
            <span className="eyebrow">LACUNAS DO NUCLEO</span>
            <strong>{fmt(job?.trace.gaps)}</strong>
          </div>
          {job?.mode === 'cumulative' && (
            <>
              <div>
                <span className="eyebrow">SEM MODELO ADMITIDO</span>
                <strong>{fmt(result?.unavailable)}</strong>
              </div>
              <div>
                <span className="eyebrow">RECUSAS DE DOMINIO</span>
                <strong>{fmt(result?.domain_refusals)}</strong>
              </div>
              <div>
                <span className="eyebrow">RESPOSTAS ERRADAS</span>
                <strong>{fmt(result?.wrong)}</strong>
              </div>
              <div>
                <span className="eyebrow">PROGRAMAS HERDADOS</span>
                <strong>{fmt(result?.skills)}</strong>
              </div>
            </>
          )}
          {job?.mode === 'language' && (
            <>
              <div>
                <span className="eyebrow">RECUSAS DE DOMINIO</span>
                <strong>{fmt(result?.refused)}</strong>
              </div>
              <div>
                <span className="eyebrow">RESPOSTAS ERRADAS</span>
                <strong>{fmt(result?.wrong)}</strong>
              </div>
              <div>
                <span className="eyebrow">FATOS DO DICIONARIO</span>
                <strong>{fmt(result?.dictionary_facts)}</strong>
              </div>
              <div>
                <span className="eyebrow">TESTES NEGATIVOS</span>
                <strong>{fmt(result?.negative_pass)} / 4</strong>
              </div>
            </>
          )}
          {result?.promotions != null && (
            <>
              <div>
                <span className="eyebrow">PROMOCOES</span>
                <strong>{fmt(result.promotions)}</strong>
              </div>
              <div>
                <span className="eyebrow">MUDANCAS DETECTADAS</span>
                <strong>{fmt(result.drifts)}</strong>
              </div>
            </>
          )}
        </div>
        {(job?.mode === 'constructive' ||
          job?.mode === 'curriculum' ||
          job?.mode === 'integration') && (
          <div className="lab-metrics instrument">
            {result?.goal_tasks != null && (
              <div>
                <span className="eyebrow">
                  {job.mode === 'curriculum'
                    ? 'METAS COMPLETAS / TESTE E TRANSFERENCIA'
                    : job.mode === 'integration'
                      ? 'METAS / PROTOCOLO INTEGRADO'
                      : 'METAS ADMITIDAS / CURRICULO'}
                </span>
                <strong>
                  {fmt(result.goal_admitted)} / {fmt(result.goal_tasks)}
                </strong>
                {result.all_goals_passed != null && (
                  <Badge tone={result.all_goals_passed ? 'green' : 'neutral'}>
                    {result.all_goals_passed
                      ? job.mode === 'integration'
                        ? 'Protocolo aprovado'
                        : 'Curriculo aprovado'
                      : 'Metas pendentes'}
                  </Badge>
                )}
              </div>
            )}
            <div>
              <span className="eyebrow">
                {job.mode === 'integration' ? 'VARIANTES ADMITIDAS' : 'CAPACIDADES ADMITIDAS'}
              </span>
              <strong>
                {fmt(typeof result?.admitted === 'number' ? result.admitted : undefined)} /{' '}
                {fmt(result?.tasks)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">
                {job.mode === 'integration' ? 'VARIANTES AVALIADAS' : 'NOVOS GRAFOS / LOTE'}
              </span>
              <strong>
                {fmt(job.mode === 'integration' ? result?.candidates : result?.new_candidates)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">
                {job.mode === 'curriculum'
                  ? 'TRANSFERENCIA / ACERTOS'
                  : job.mode === 'integration'
                    ? 'MUDANCAS / REUSOS'
                    : 'GENES / ULTIMO CASO'}
              </span>
              <strong>
                {job.mode === 'curriculum'
                  ? `${fmt(validations.reduce((n, c) => n + (c.transfer_correct ?? 0), 0))} / ${fmt(validations.reduce((n, c) => n + (c.transfer_total ?? 0), 0))}`
                  : job.mode === 'integration'
                    ? `${fmt(result?.drifts)} / ${fmt(result?.reuses)}`
                    : fmt(job.cases.at(-1)?.genes)}
              </strong>
            </div>
          </div>
        )}
        {job?.mode === 'structural' && (
          <div className="lab-metrics instrument">
            <div>
              <span className="eyebrow">ESTRUTURAS / ACERTOS</span>
              <strong>
                {fmt(result?.correct)} / {fmt(result?.total)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">DIALOGO / TURNOS CORRETOS</span>
              <strong>
                {fmt(result?.dialogue_correct)} / {fmt(result?.dialogue_turns)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">REGRAS COMPILADAS</span>
              <strong>{fmt(result?.total_rules ?? result?.acquired)}</strong>
            </div>
            <div>
              <span className="eyebrow">PROGRAMAS NUMERICOS INFERIDOS</span>
              <strong>{fmt(result?.derived_programs)}</strong>
            </div>
          </div>
        )}
        {job?.mode === 'structural' && (result?.agent_total ?? 0) > 0 && (
          <div className="lab-metrics instrument">
            <div>
              <span className="eyebrow">OBJETIVOS DO AGENTE</span>
              <strong>
                {fmt(result?.agent_correct)} / {fmt(result?.agent_total)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">DESCENDENTE / SEM PROFESSOR</span>
              <strong>
                {fmt(result?.agent_portable)} / {fmt(result?.agent_total)}
              </strong>
            </div>
          </div>
        )}
        {job?.mode === 'structural' && (result?.algorithm_total ?? 0) > 0 && (
          <div className="lab-metrics instrument">
            <div>
              <span className="eyebrow">ALGORITMOS / RESPOSTAS DO DESCENDENTE</span>
              <strong>
                {fmt(result?.algorithm_correct)} / {fmt(result?.algorithm_total)}
              </strong>
            </div>
          </div>
        )}
        {(result?.file_total ?? 0) > 0 && (
          <div
            className="lab-metrics instrument"
            role="region"
            aria-label="Arquivos com esquema aprendido"
          >
            <div>
              <span className="eyebrow">ARQUIVOS / RESULTADOS E RECUSAS CORRETOS</span>
              <strong>
                {fmt(result?.file_correct)} / {fmt(result?.file_total)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">PLANOS E RESPOSTAS REGISTRADOS</span>
              <button
                className="icon-button"
                title="Inspecionar planos e resultados reais dos arquivos"
                aria-label="Inspecionar arquivos com esquema aprendido"
                onClick={() =>
                  inspect(
                    'Arquivos com esquema aprendido',
                    job?.cases.filter(
                      (c) => c.method === 'Agent::file_plan' || c.method === 'Agent::file_result',
                    ),
                  )
                }
              >
                <Search size={15} />
              </button>
            </div>
          </div>
        )}
        {((result?.session_total ?? 0) > 0 || sessionTurn || sessionInterpretation) && (
          <div className="lab-metrics instrument" role="region" aria-label="Sessao contextual">
            <div>
              <span className="eyebrow">SESSAO / TURNOS CORRETOS</span>
              <strong>
                {fmt(result?.session_correct ?? sessionTurn?.correct)} /{' '}
                {fmt(result?.session_total ?? sessionTurn?.total)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">SIGNIFICADOS / FUNDAMENTADOS</span>
              <strong>
                {fmt(sessionInterpretation?.meanings)} / {fmt(sessionInterpretation?.grounded)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">ULTIMA COMPOSICAO / UNIDADES</span>
              <strong>{fmt(sessionInterpretation?.work)}</strong>
            </div>
            <div>
              <span className="eyebrow">ULTIMO TURNO / EVIDENCIAS</span>
              <button
                className="icon-button"
                title="Inspecionar pedido, referencias e resposta real"
                aria-label="Inspecionar sessao contextual"
                onClick={() => {
                  let reply: unknown = sessionTurn?.actual_json;
                  try {
                    reply = JSON.parse(sessionTurn?.actual_json ?? 'null');
                  } catch {
                    /* Preserve raw malformed evidence. */
                  }
                  inspect('Sessao contextual registrada', {
                    interpretation: sessionInterpretation,
                    turn: sessionTurn,
                    reply,
                  });
                }}
              >
                <Search size={15} />
              </button>
            </div>
          </div>
        )}
        {job?.cases.some((c) => c.method === 'Session::control') && (
          <div
            className="lab-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Controles da sessao contextual"
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Aprendizado</th>
                  <th>Turnos corretos</th>
                  <th>Consultas numericas</th>
                  <th>Composicao / unidades</th>
                </tr>
              </thead>
              <tbody>
                {job.cases
                  .filter((c) => c.method === 'Session::control')
                  .map((c) => (
                    <tr key={c.case}>
                      <td>
                        <button
                          className="lab-case"
                          onClick={() => inspect('Controle da sessao contextual', c)}
                        >
                          {c.variant === 'context'
                            ? 'Memoria + composicao'
                            : c.variant === 'no-memory'
                              ? 'Sem consulta a memoria'
                              : 'Sem composicao'}
                        </button>
                      </td>
                      <td>
                        {fmt(c.correct)} / {fmt(c.total)}
                      </td>
                      <td>{fmt(c.queries)}</td>
                      <td>{fmt(c.work)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
        {algorithmSearch && (
          <div
            className="lab-metrics instrument"
            role="region"
            aria-label="Ultima busca de abstracoes"
          >
            <div>
              <span className="eyebrow">PROCEDIMENTO / ULTIMA INVESTIGACAO</span>
              <strong>{algorithmSearch.operation}</strong>
            </div>
            <div>
              <span className="eyebrow">PADROES PROPOSTOS / INVESTIGACAO</span>
              <strong>{fmt(algorithmSearch.templates)}</strong>
            </div>
            <div>
              <span className="eyebrow">MINERACAO / UNIDADES INTERNAS</span>
              <strong>{fmt(algorithmSearch.mining_work)}</strong>
            </div>
            <div>
              <span className="eyebrow">BIBLIOTECA / ULTIMA INVESTIGACAO</span>
              <strong>{fmt(algorithmSearch.library_programs)}</strong>
            </div>
            <div>
              <span className="eyebrow">JANELAS / BUSCA FINAL</span>
              <strong>{fmt(algorithmSearch.library_windows)}</strong>
            </div>
            <div>
              <span className="eyebrow">REGRAS SUSPENSAS / ULTIMA MUDANCA</span>
              <strong>{fmt(algorithmDrift?.invalidated)}</strong>
            </div>
            {algorithmInquiry?.last_inquiry && (
              <>
                <div>
                  <span className="eyebrow">RESPOSTAS / ULTIMA AQUISICAO COM CONSULTA</span>
                  <strong>
                    {algorithmInquiry.operation}: {fmt(algorithmInquiry.query_count)}
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">HIPOTESES / AMOSTRA ANTES E DEPOIS</span>
                  <strong>
                    {fmt(algorithmInquiry.query_hypotheses)} /{' '}
                    {fmt(algorithmInquiry.query_survivors)}
                  </strong>
                </div>
                <div>
                  <span className="eyebrow">ULTIMA CONSULTA REGISTRADA</span>
                  <button
                    className="icon-button"
                    title="Inspecionar pergunta, previsoes e resposta registrada"
                    aria-label="Inspecionar consulta de aprendizagem"
                    onClick={() => {
                      let query: unknown = algorithmInquiry.last_inquiry;
                      try {
                        query = JSON.parse(algorithmInquiry.last_inquiry ?? 'null');
                      } catch {
                        // Keep malformed evidence inspectable without crashing the workspace.
                      }
                      inspect('Consulta de aprendizagem registrada', query);
                    }}
                  >
                    <Search size={15} />
                  </button>
                </div>
              </>
            )}
          </div>
        )}
        {job?.cases.some((c) => c.method === 'Algorithm::control') && (
          <div
            className="lab-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Controles dos algoritmos"
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Aprendizado</th>
                  <th>Procedimentos</th>
                  <th>Respostas</th>
                  <th>Trabalho interno / busca e verificacao</th>
                </tr>
              </thead>
              <tbody>
                {job.cases
                  .filter((c) => c.method === 'Algorithm::control')
                  .map((c) => (
                    <tr key={c.case}>
                      <td>
                        <button
                          className="lab-case"
                          onClick={() => inspect('Controle dos algoritmos', c)}
                        >
                          {c.variant === 'passive'
                            ? 'Consultas passivas / mesma biblioteca'
                            : c.variant === 'inherited'
                              ? 'Com biblioteca herdada'
                              : c.variant === 'default-order'
                                ? 'Ordem padrao + biblioteca'
                                : 'Sem biblioteca na busca'}
                        </button>
                      </td>
                      <td>{fmt(c.acquired)}</td>
                      <td>
                        {fmt(c.correct)} / {fmt(c.total)}
                      </td>
                      <td>{fmt(c.work)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
        {job?.cases.some((c) => c.method === 'Agent::campaign') && (
          <div
            className="lab-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Controles da investigacao"
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Politica</th>
                  <th>Objetivos</th>
                  <th>Descendente</th>
                  <th>Consultas</th>
                  <th>Refutacoes</th>
                  <th>Trabalho registrado</th>
                  <th>Composicao concluida</th>
                </tr>
              </thead>
              <tbody>
                {job.cases
                  .filter((c) => c.method === 'Agent::campaign')
                  .map((c) => (
                    <tr key={c.case}>
                      <td>
                        <button
                          className="lab-case"
                          onClick={() => inspect('Controle da investigacao', c)}
                        >
                          {c.variant === 'active'
                            ? 'Ativa + heranca'
                            : c.variant === 'passive'
                              ? 'Ordem passiva'
                              : 'Sem biblioteca na busca'}
                        </button>
                      </td>
                      <td>
                        {fmt(c.correct)} / {fmt(c.total)}
                      </td>
                      <td>{fmt(c.portable_correct)}</td>
                      <td>{fmt(c.queries)}</td>
                      <td>{fmt(c.refutations)}</td>
                      <td>{fmt(c.work)}</td>
                      <td>{fmt(c.language_work)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
        {channels.length > 0 && (
          <div
            className="lab-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Checkpoints por canal"
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>{job?.mode === 'controller' ? 'Ultimo objetivo' : 'Canal'}</th>
                  <th>Fase</th>
                  <th>Observacao</th>
                  <th>
                    {job?.mode === 'controller' ? 'Corretos / fluxo' : 'Corretos / avaliados'}
                  </th>
                  <th>Promocoes</th>
                  <th>Mudancas</th>
                </tr>
              </thead>
              <tbody>
                {channels.map((c) => (
                  <tr key={c.channel}>
                    <td>{c.channel}</td>
                    <td>{c.phase}</td>
                    <td>{fmt(c.observation)}</td>
                    <td>
                      {fmt(c.correct)} / {fmt(c.scored)}
                    </td>
                    <td>{fmt(c.promotions)}</td>
                    <td>{fmt(c.drifts)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {['seed', 'genome'].includes(job?.mode ?? '') && (
          <div className="lab-metrics instrument">
            <div>
              <span className="eyebrow">CANDIDATOS AVALIADOS</span>
              <strong>{fmt(result?.candidates)}</strong>
            </div>
            <div>
              <span className="eyebrow">MELHOR COBERTURA / DESENVOLVIMENTO</span>
              <strong>
                {fmt(result?.solved)} / {fmt(result?.tasks)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">FRENTE FINAL</span>
              <strong>{fmt(result?.survivors)}</strong>
            </div>
          </div>
        )}
        {job?.mode === 'genome' && (
          <div className="lab-metrics instrument">
            <div>
              <span className="eyebrow">TENTATIVAS DE COMPOSICAO</span>
              <strong>{fmt(result?.attempts)}</strong>
            </div>
            <div>
              <span className="eyebrow">COMPOSICOES ADMITIDAS / TREINO</span>
              <strong>{fmt(result?.acquired)}</strong>
            </div>
          </div>
        )}
        {job?.mode === 'controller' && (
          <div className="lab-metrics instrument">
            <div>
              <span className="eyebrow">METAS ATINGIDAS / SIMULADOR</span>
              <strong>
                {fmt(result?.physical_correct)} / {fmt(result?.heldout)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">ROTA FIXA / DECISOES CORRETAS</span>
              <strong>
                {fmt(result?.baseline_correct)} / {fmt(result?.heldout)}
              </strong>
            </div>
            <div>
              <span className="eyebrow">MUDANCAS OBSERVADAS</span>
              <strong>{fmt(result?.drifts)}</strong>
            </div>
            <div>
              <span className="eyebrow">NOVAS CONSULTAS AO MECANISMO</span>
              <strong>{fmt(result?.queries)}</strong>
            </div>
            <div>
              <span className="eyebrow">PAYLOAD ATIVO / ULTIMO FLUXO</span>
              <strong>{residency?.peak_bytes == null ? '--' : bytes(residency.peak_bytes)}</strong>
            </div>
            <div>
              <span className="eyebrow">MODULOS / PICO DO ULTIMO FLUXO</span>
              <strong>{fmt(residency?.peak_modules)}</strong>
            </div>
          </div>
        )}
        {job?.mode === 'structural' ? (
          <div
            className="lab-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Resultados por familia"
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Familia</th>
                  <th>Acertos</th>
                  <th>Total</th>
                  <th>Resultado</th>
                </tr>
              </thead>
              <tbody>
                {job.cases
                  .filter((c) => c.method === 'StructuralSeed::transfer')
                  .map((c) => (
                    <tr key={c.case}>
                      <td>
                        <button
                          className="lab-case"
                          onClick={() => inspect(`Familia ${c.stage}`, c)}
                        >
                          {c.stage ?? '--'}
                        </button>
                      </td>
                      <td>{fmt(c.correct)}</td>
                      <td>{fmt(c.total)}</td>
                      <td>
                        {c.admitted === true
                          ? 'Aprovada no protocolo'
                          : c.admitted === false
                            ? 'Falhou no protocolo'
                            : '--'}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : job?.mode === 'cumulative' ? null : job?.cases.length ? (
          <div
            className="lab-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Resultados por caso"
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Caso</th>
                  <th>Resultado</th>
                  <th>Geracao</th>
                  <th>Trabalho</th>
                  <th>Validacao</th>
                  {job.mode === 'curriculum' && <th>Transferencia</th>}
                  <th>Bytes</th>
                </tr>
              </thead>
              <tbody>
                {job.cases.map((c) => (
                  <tr key={c.case}>
                    <td>
                      <button
                        className="lab-case"
                        onClick={() => inspect(`Caso ${c.case}`, c)}
                        aria-label={`Inspecionar caso ${c.case}`}
                      >
                        {job.mode === 'evolution'
                          ? cases[c.case!]
                          : job.mode === 'constructive'
                            ? constructiveNames[c.case!]
                            : (c.name ?? `Caso ${c.case}`)}
                      </button>
                    </td>
                    <td>
                      {(job.mode === 'constructive' ||
                        job.mode === 'curriculum' ||
                        job.mode === 'integration') &&
                      c.admitted === true
                        ? 'Admitido'
                        : (job.mode === 'constructive' ||
                              job.mode === 'curriculum' ||
                              job.mode === 'integration') &&
                            c.admitted === false
                          ? 'Nao admitido'
                          : (c.outcome ?? c.phase ?? c.event)}
                    </td>
                    <td>{fmt(c.generation)}</td>
                    <td>{fmt(c.work ?? c.search_work)}</td>
                    <td>
                      {c.event === 'verification'
                        ? `${fmt(c.correct)} / ${fmt(c.heldout)}`
                        : (job.mode === 'constructive' && constructiveValidation(c)) ||
                            job.mode === 'curriculum' ||
                            job.mode === 'integration'
                          ? `${fmt(c.correct)} / ${fmt(c.checked)}`
                          : '--'}
                    </td>
                    {job.mode === 'curriculum' && (
                      <td>
                        {fmt(c.transfer_correct)} / {fmt(c.transfer_total)}
                      </td>
                    )}
                    <td>
                      {fmt(
                        c.capsule_bytes ??
                          (job.mode === 'curriculum'
                            ? files.find(
                                (f) =>
                                  f.name ===
                                  `${c.name}.${c.domain === 'numeric' ? 'axevo' : c.domain === 'composed' ? 'axmath' : c.domain === 'text' ? 'axtext' : 'axsem'}`,
                              )?.bytes
                            : job.mode === 'constructive'
                              ? files.find((f) => f.name === `task-${c.case}.axevo`)?.bytes
                              : undefined),
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty
            icon={Play}
            title={job?.mode === 'tests' ? 'Saida do executor abaixo' : 'Sem casos adquiridos'}
          />
        )}
      </Widget>
      {job?.mode === 'cumulative' && <Cumulative job={job} inspect={inspect} />}
      {job?.mode === 'semantic' && <Relational job={job} inspect={inspect} />}
      {job?.mode === 'nursery' && <Nursery job={job} inspect={inspect} />}
      {job && job.mode !== 'cumulative' && (job.candidates ?? []).length > 0 && (
        <SeedPopulation job={job} inspect={inspect} />
      )}
      {procedure && (
        <Widget
          title="Procedimento aprendido"
          sub={procedure.name}
          onInspect={() => inspect('Composicao admitida', procedure)}
          tools={
            <select
              aria-label="Procedimento"
              value={String(procedure.case)}
              onChange={(e) => setGeneCase(e.target.value)}
            >
              {procedures.map((p) => (
                <option key={p.case} value={p.case}>
                  {p.name}
                </option>
              ))}
            </select>
          }
          footer={
            <>
              <span className="mono">
                {job?.mode === 'genome'
                  ? 'HERANCA + NOVA COMPOSICAO'
                  : 'REFERENCIAS IMUTAVEIS / SIMULACAO RESETAVEL'}
              </span>
              <span className="push" />
              <span className="mono">{fmt(procedure.steps?.length)} ETAPAS</span>
            </>
          }
        >
          <ol className="procedure-stages" aria-label="Etapas do procedimento adquirido">
            {procedure.steps?.map((step, i) => (
              <li key={`${i}:${step}`}>
                <button
                  className="secondary"
                  title={step}
                  onClick={() =>
                    inspect(`Etapa ${i + 1}`, {
                      step,
                      procedure: procedure.identity,
                      source: procedure.expression,
                    })
                  }
                >
                  <Workflow size={16} />
                  <span>{step}</span>
                </button>
                {i + 1 < procedure.steps!.length && <ArrowRight size={16} aria-hidden="true" />}
              </li>
            ))}
          </ol>
        </Widget>
      )}
      {genome && (
        <Widget
          title="Genoma adquirido"
          sub={genome.name ?? `CASO ${genome.case} / ${genome.identity ?? ''}`}
          tools={
            <select
              aria-label="Genoma"
              value={String(genome.case)}
              onChange={(e) => setGeneCase(e.target.value)}
            >
              {genomes.map((g) => (
                <option value={g.case} key={g.case}>
                  {g.name ??
                    (job?.mode === 'constructive' ? constructiveNames[g.case!] : cases[g.case!]) ??
                    `Caso ${g.case}`}
                </option>
              ))}
            </select>
          }
          onInspect={() => inspect('Genoma e programa', genome)}
          footer={
            <>
              <span className="mono">
                {fmt(activeGenes)} GENES ATIVOS / {fmt(genome.instructions)} INSTRUCOES
              </span>
              <span className="push" />
              <span className="mono">{modelBytes == null ? '--' : bytes(modelBytes)}</span>
            </>
          }
        >
          <Genome record={genome} inspect={inspect} />
          <div className="lab-expression">
            <code>{genome.expression}</code>
            <Sparkline values={losses} />
          </div>
        </Widget>
      )}
      <Widget
        title="Saida do processo"
        sub={job?.mode === 'tests' ? 'COMPILADOR + HARNESS / EXIT REAL' : 'STDOUT + STDERR'}
        onInspect={() => inspect('Saida do processo', job?.output)}
        footer={<span className="mono">ULTIMOS 128 KB / LOG COMPLETO EM JSONL</span>}
      >
        <pre className="lab-output" tabIndex={0} aria-label="Saida textual do processo">
          {job?.output || 'Sem saida textual recebida'}
        </pre>
      </Widget>
      {files.length > 0 && (
        <section className="unframed-section">
          <h2>Artefatos desta execucao</h2>
          <ArtifactLinks
            run={job!.id}
            files={files.filter((f) => !f.name.startsWith('revision-'))}
          />
          {revisions.length > 0 && (
            <details className="lab-revisions" key={job!.id}>
              <summary>Linhagem / {revisions.length} revisoes</summary>
              <ArtifactLinks run={job!.id} files={revisions} />
            </details>
          )}
        </section>
      )}
      {fileError && <p role="alert">{fileError}</p>}
    </div>
  );
}
