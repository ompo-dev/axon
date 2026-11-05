import { Widget } from './components';
import { fmt } from './useAxon';
import type { Job } from './types';

export function Nursery({
  job,
  inspect,
}: {
  job: Job;
  inspect: (title: string, value: unknown) => void;
}) {
  const checkpoint = job.samples.findLast((e) => e.event === 'nursery_progress');
  const state = job.result ?? checkpoint;
  const growth = job.samples.filter((e) => e.event === 'dna_growth');
  const checkpoints = job.samples.filter((e) => e.event === 'nursery_progress');
  const progress =
    job.result && checkpoints.at(-1)?.candidates !== job.result.candidates
      ? [...checkpoints, job.result]
      : checkpoints;
  return (
    <Widget
      title="Viveiro de seeds"
      onInspect={() => inspect('Registro persistente de evolucao', state)}
      footer={
        <span className="mono">
          {job.result?.outcome ?? 'BUSCA EM ANDAMENTO'} /{' '}
          {job.result?.admitted === true ? 'CURRICULO APROVADO' : 'SEM VENCEDOR INTEGRAL'}
        </span>
      }
    >
      <div className="lab-metrics instrument">
        <div>
          <span className="eyebrow">AVALIADAS / REGISTRO</span>
          <strong>{fmt(state?.candidates)}</strong>
        </div>
        <div>
          <span className="eyebrow">DNA UNICO RESERVADO</span>
          <strong>{fmt(state?.unique)}</strong>
        </div>
        <div>
          <span className="eyebrow">HISTORICOS / NAO REAVALIADOS</span>
          <strong>{fmt(state?.imported)}</strong>
        </div>
        <div>
          <span className="eyebrow">NOVAS NESTE LOTE</span>
          <strong>{fmt(job.result?.new_candidates)}</strong>
        </div>
        <div>
          <span className="eyebrow">CASAS DECIMAIS / GENOTIPO</span>
          <strong>{fmt(state?.genes)}</strong>
        </div>
        <div>
          <span className="eyebrow">MELHOR / DESAFIOS</span>
          <strong>
            {fmt(state?.correct)} / {fmt(state?.checked)}
          </strong>
        </div>
        <div>
          <span className="eyebrow">RESERVAS INCONCLUSIVAS</span>
          <strong>{fmt(state?.unfinished)}</strong>
        </div>
      </div>
      <div className="lab-table-scroll" tabIndex={0} role="region" aria-label="Crescimento do DNA">
        <table className="lab-table">
          <thead>
            <tr>
              <th>DNA / casas</th>
              <th>Avaliacoes anteriores</th>
            </tr>
          </thead>
          <tbody>
            {growth.map((e, i) => (
              <tr key={i}>
                <td>{fmt(e.genes)}</td>
                <td>{fmt(e.candidates)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div
        className="lab-table-scroll population-scroll"
        tabIndex={0}
        role="region"
        aria-label="Evolucao sem repeticao"
      >
        <table className="lab-table">
          <thead>
            <tr>
              <th>Avaliadas</th>
              <th>DNA unico</th>
              <th>Casas</th>
              <th>Melhor DNA</th>
              <th>Corretos</th>
            </tr>
          </thead>
          <tbody>
            {progress.map((e, i) => (
              <tr key={i}>
                <td>{fmt(e.candidates)}</td>
                <td>{fmt(e.unique)}</td>
                <td>{fmt(e.genes)}</td>
                <td>{e.genotype ?? '--'}</td>
                <td>
                  {fmt(e.correct)} / {fmt(e.checked)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Widget>
  );
}
