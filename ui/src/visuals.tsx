import { useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Plus, Minus, Scan } from 'lucide-react';
import { IconButton } from './components';
import type { Graph, RecordEvent, GraphNode } from './types';

export const palette = [
  'var(--blue)',
  'var(--teal)',
  'var(--purple)',
  'var(--orange)',
  'var(--pink)',
];
export function CoreDiagram({
  graph,
  event,
  active,
}: {
  graph: Graph;
  event?: RecordEvent;
  active: boolean;
}) {
  const reduced = useReducedMotion();
  const models = graph.models.slice(0, 5);
  const positions = [
    [114, 103],
    [185, 33],
    [445, 102],
    [426, 222],
    [114, 222],
  ];
  const chip = (x: number, y: number, name: string, index: number) => (
    <g key={name} transform={`translate(${x} ${y})`}>
      <path
        d="M-29 0 L0 -16 L29 0 L29 14 L0 31 L-29 14Z"
        fill="var(--surface)"
        stroke="var(--diagram-stroke)"
        strokeWidth="1.4"
      />
      <path d="M-29 0 L0 17 L29 0 M0 17V31" fill="none" stroke="var(--diagram-stroke)" />
      <path
        d="M-15 0 L0 -8 L15 0 L0 9Z"
        fill="var(--surface-soft)"
        stroke="var(--diagram-stroke)"
      />
      <circle cy="1" r="3" fill={palette[index % palette.length]} />
      <text y="49" textAnchor="middle" fill="var(--muted)" fontSize="10" fontFamily="var(--mono)">
        {name}
      </text>
    </g>
  );
  return (
    <svg
      viewBox="0 0 560 300"
      className="core-diagram"
      role="img"
      aria-label={`${graph.models.length} modelos observados no grafo`}
    >
      {models.map((m, i) => (
        <path
          key={m.context}
          d={`M${positions[i][0]} ${positions[i][1] + 16} L280 177`}
          stroke="var(--diagram-stroke)"
          fill="none"
          strokeDasharray="3 4"
          opacity=".6"
        />
      ))}
      <path
        d="M175 170 L280 109 L386 170 L386 202 L280 263 L175 202Z"
        fill="var(--surface-soft)"
        stroke="var(--diagram-stroke)"
        strokeWidth="1.3"
      />
      <path
        d="M175 170 L280 232 L386 170 M280 232V263"
        fill="none"
        stroke="var(--diagram-stroke)"
      />
      <path
        d="M185 185 L280 241 L376 185 M185 200L280 256L376 200"
        fill="none"
        stroke="var(--diagram-stroke)"
        strokeDasharray="2 5"
        opacity=".45"
      />
      <path
        d="M226 135 L280 104 L334 135 L334 161 L280 192 L226 161Z"
        fill="var(--orange-soft)"
        stroke="var(--orange-border)"
        strokeWidth="1.5"
      />
      <path d="M226 135L280 166L334 135 M280 166V192" fill="none" stroke="var(--orange-border)" />
      <ellipse cx="280" cy="135" rx="11" ry="6" fill="none" stroke="var(--orange)" />
      <circle cx="280" cy="135" r="2.5" fill="var(--orange)" />
      <path d="M258 201L280 188L302 201L280 214Z" fill="var(--ink)" />
      <path d="M273 201L280 197L287 201L280 205Z" fill="var(--surface)" />
      {models.map((m, i) =>
        chip(positions[i][0], positions[i][1], `${m.name.toUpperCase()} / ${m.states}S`, i),
      )}
      {active &&
        event?.event === 'step' &&
        Date.now() - event.time < 1000 &&
        !reduced &&
        models.map((m, i) => {
          const context =
            (event.fromStates?.length ?? 0) > 1 ? `${event.context}/factor-${i}` : event.context;
          return (
            context === m.context && (
              <motion.path
                key={`${event.seq}-${i}`}
                d={`M280 177L${positions[i][0]} ${positions[i][1] + 16}`}
                stroke={palette[i]}
                fill="none"
                strokeWidth="2"
                initial={{ pathLength: 0, opacity: 1 }}
                animate={{ pathLength: 1, opacity: 0 }}
                transition={{ duration: 0.8 }}
              />
            )
          );
        })}
      <circle cx="330" cy="137" r="2.5" fill={active ? 'var(--teal)' : 'var(--diagram-stroke)'} />
      <text x="353" y="58" fill="var(--orange)" fontSize="11" fontFamily="var(--mono)">
        AXON CORE
      </text>
      <path d="M346 54H317V115" stroke="var(--orange-border)" fill="none" />
    </svg>
  );
}

export function GraphView({
  graph,
  events,
  active,
  onSelect,
}: {
  graph: Graph;
  events: RecordEvent[];
  active: boolean;
  onSelect: (node: GraphNode) => void;
}) {
  const [zoom, setZoom] = useState(1),
    [pan, setPan] = useState({ x: 0, y: 0 });
  const reduced = useReducedMotion();
  const [drag, setDrag] = useState<{ x: number; y: number; pan: { x: number; y: number } } | null>(
    null,
  );
  const points = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>();
    graph.models.forEach((m, f) => {
      const cx =
        graph.models.length === 1 ? 400 : 115 + (f / Math.max(1, graph.models.length - 1)) * 570;
      const cy = graph.models.length > 2 ? 220 + (f % 2) * 100 : 280;
      const nodes = graph.nodes.filter((n) => n.context === m.context);
      nodes.forEach((n, i) => {
        const a = -Math.PI / 2 + (i / Math.max(1, nodes.length)) * Math.PI * 2;
        map.set(n.id, { x: cx + Math.cos(a) * 63, y: cy + Math.sin(a) * 76 });
      });
    });
    return map;
  }, [graph]);
  const step = events.findLast((e) => e.event === 'step');
  const recent = active && step && Date.now() - step.time < 1000 ? step : null;
  return (
    <div className="graph-view">
      <div className="graph-caption">
        <span className="mono">
          {graph.phase === 'stored-snapshot'
            ? 'SNAPSHOT ARMAZENADO'
            : (graph.phase ?? 'SEM MODELO').toUpperCase()}
        </span>
        <span className="mono">
          {graph.nodes.length} ESTADOS / {graph.edges.length} TRANSICOES
        </span>
      </div>
      <svg
        viewBox="0 0 800 520"
        role="group"
        aria-label="Grafo real de estados e transicoes"
        className="network-svg"
        onPointerDown={(e) => {
          if ((e.target as Element).closest('[data-node]')) return;
          setDrag({ x: e.clientX, y: e.clientY, pan });
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (drag) {
            const rect = e.currentTarget.getBoundingClientRect();
            setPan({
              x: drag.pan.x + ((e.clientX - drag.x) * 800) / rect.width,
              y: drag.pan.y + ((e.clientY - drag.y) * 520) / rect.height,
            });
          }
        }}
        onPointerUp={() => setDrag(null)}
        onPointerCancel={() => setDrag(null)}
      >
        <defs>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0 0L6 3L0 6" fill="var(--diagram-stroke)" />
          </marker>
        </defs>
        <g
          transform={`translate(${400 + pan.x} ${260 + pan.y}) scale(${zoom}) translate(-400 -260)`}
        >
          {graph.models.map((m) => {
            const n = graph.nodes.find((n) => n.context === m.context),
              p = n && points.get(n.id);
            return (
              p && (
                <text
                  key={m.context}
                  x={p.x}
                  y={p.y - 34}
                  textAnchor="middle"
                  fontSize="11"
                  fontFamily="var(--mono)"
                  fill="var(--muted)"
                >
                  {m.name.toUpperCase()}
                </text>
              )
            );
          })}
          {graph.edges.map((e) => {
            const a = points.get(e.source),
              b = points.get(e.target);
            if (!a || !b) return null;
            const path =
              e.source === e.target
                ? `M${a.x} ${a.y - 12} C${a.x + 40} ${a.y - 55} ${a.x - 40} ${a.y - 55} ${a.x} ${a.y - 12}`
                : `M${a.x} ${a.y} Q${(a.x + b.x) / 2 + (e.action - 1.5) * 15} ${(a.y + b.y) / 2 - 15} ${b.x} ${b.y}`;
            const source = graph.nodes.find((n) => n.id === e.source),
              target = graph.nodes.find((n) => n.id === e.target);
            const context =
              (recent?.fromStates?.length ?? 0) > 1
                ? `${recent?.context}/factor-${e.factor}`
                : recent?.context;
            const lit =
              !!recent &&
              context === source?.context &&
              recent.fromStates?.[e.factor] === source?.state &&
              recent.toStates?.[e.factor] === target?.state &&
              recent.action === e.action &&
              recent.reply?.[e.factor] === e.output;
            return (
              <g key={`${e.source}-${e.action}`}>
                <path
                  d={path}
                  fill="none"
                  stroke={palette[e.factor % 5]}
                  strokeWidth={lit ? 2 : 1}
                  opacity={lit ? 1 : 0.22}
                  markerEnd="url(#arrow)"
                >
                  <title>{`${e.source} -> ${e.target} / acao ${e.action} / saida ${e.output}`}</title>
                </path>
                {lit && !reduced && (
                  <motion.path
                    key={recent.seq}
                    d={path}
                    fill="none"
                    stroke="var(--orange)"
                    strokeWidth="2.5"
                    initial={{ pathLength: 0, opacity: 1 }}
                    animate={{ pathLength: 1, opacity: 0 }}
                    transition={{ duration: 0.8 }}
                  />
                )}
              </g>
            );
          })}
          {graph.nodes.map((n) => {
            const p = points.get(n.id);
            return (
              p && (
                <g
                  key={n.id}
                  data-node={n.id}
                  tabIndex={0}
                  role="button"
                  aria-label={`Inspecionar ${n.id}`}
                  onClick={() => onSelect(n)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelect(n);
                    }
                  }}
                  className="graph-node"
                >
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r="13"
                    fill="var(--surface)"
                    stroke={palette[n.factor % 5]}
                    strokeWidth="1.5"
                  />
                  <circle cx={p.x} cy={p.y} r="3" fill={palette[n.factor % 5]} />
                  <text
                    x={p.x}
                    y={p.y + 29}
                    textAnchor="middle"
                    fill="var(--muted)"
                    fontSize="10"
                    fontFamily="var(--mono)"
                  >
                    s{n.state}
                  </text>
                </g>
              )
            );
          })}
          {!graph.nodes.length && (
            <text x="400" y="260" textAnchor="middle" fill="var(--muted)" fontSize="14">
              Nenhum grafo recebido
            </text>
          )}
        </g>
      </svg>
      <div className="graph-controls">
        <IconButton
          icon={Minus}
          label="Afastar grafo"
          onClick={() => setZoom((v) => Math.max(0.5, v - 0.15))}
        />
        <output>{Math.round(zoom * 100)}%</output>
        <IconButton
          icon={Plus}
          label="Aproximar grafo"
          onClick={() => setZoom((v) => Math.min(3, v + 0.15))}
        />
        <IconButton
          icon={Scan}
          label="Enquadrar grafo"
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
        />
      </div>
    </div>
  );
}
