import { Widget } from './components';
import { bytes, fmt } from './useAxon';
import type { Job } from './types';

export function Relational({
  job,
  inspect,
}: {
  job: Job;
  inspect: (title: string, value: unknown) => void;
}) {
  const result = job.result;
  const generations = job.samples.filter((e) => e.event === 'seed_selection');
  const births = job.samples.filter((e) => e.event === 'genome');
  const predictions = job.samples.filter((e) => e.event === 'prediction');
  const arms = job.samples.filter((e) => e.event === 'validation');
  return (
    <>
      <Widget
        title="Composicao e generalizacao"
        onInspect={() => inspect('Protocolo R1 / P0', result)}
        footer={<span className="mono">{result?.outcome ?? 'AVALIACAO EM ANDAMENTO'}</span>}
      >
        <div className="lab-metrics instrument">
          <div>
            <span className="eyebrow">CANDIDATOS AVALIADOS</span>
            <strong>{fmt(result?.candidates)}</strong>
          </div>
          <div>
            <span className="eyebrow">ESTRUTURAS DISTINTAS</span>
            <strong>{fmt(result?.unique)}</strong>
          </div>
          <div>
            <span className="eyebrow">TEXTOS FORA DA GRAMATICA</span>
            <strong>
              {fmt(result?.natural_correct)} / {fmt(result?.natural_checked)}
            </strong>
          </div>
          <div>
            <span className="eyebrow">RECUSAS CORRETAS</span>
            <strong>
              {fmt(result?.negative_correct)} / {fmt(result?.negative_checked)}
            </strong>
          </div>
          <div>
            <span className="eyebrow">RESPOSTAS ERRADAS</span>
            <strong>{fmt(result?.wrong)}</strong>
          </div>
        </div>
        <div
          className="lab-table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Selecao estrutural R1"
        >
          <table className="lab-table">
            <thead>
              <tr>
                <th>Rodada</th>
                <th>Melhor / desenvolvimento</th>
                <th>Controle aleatorio</th>
                <th>Diversidade</th>
              </tr>
            </thead>
            <tbody>
              {generations.map((e) => (
                <tr key={e.generation}>
                  <td>{fmt(e.generation)}</td>
                  <td>
                    {fmt(e.correct)} / {fmt(e.checked)}
                  </td>
                  <td>
                    {fmt(e.random_best)} / {fmt(e.checked)}
                  </td>
                  <td>
                    {fmt(e.unique)} / {fmt(e.population)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div
          className="lab-table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Validacao independente R1"
        >
          <table className="lab-table">
            <thead>
              <tr>
                <th>Individuo</th>
                <th>Corretos</th>
                <th>Recusas</th>
                <th>Errados</th>
              </tr>
            </thead>
            <tbody>
              {arms.map((e) => (
                <tr key={e.arm}>
                  <td>{e.arm === 'winner-3' ? 'Consolidado' : e.arm}</td>
                  <td>
                    {fmt(e.correct)} / {fmt(e.checked)}
                  </td>
                  <td>{fmt(e.refused)}</td>
                  <td>{fmt(e.wrong)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Widget>
      <Widget
        title="Nascimento e evidencias"
        onInspect={() => inspect('Nascimentos reais', births)}
      >
        <div className="lab-metrics instrument">
          {births.map((e) => (
            <div key={e.name}>
              <span className="eyebrow">{e.name === 'winner-3' ? 'Consolidado' : e.name}</span>
              <strong>{e.encoded_bytes == null ? '--' : bytes(e.encoded_bytes)}</strong>
              <small>{fmt(e.seed_digits)} digitos</small>
            </div>
          ))}
        </div>
        <div
          className="lab-table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Respostas observadas R1"
        >
          <table className="lab-table">
            <thead>
              <tr>
                <th>Individuo / caso</th>
                <th>Entrada</th>
                <th>Resposta</th>
                <th>Esperado</th>
                <th>Resultado</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((e, i) => (
                <tr key={`${e.arm}:${e.name}:${i}`}>
                  <td>
                    <button className="lab-case" onClick={() => inspect(`Resposta ${e.name}`, e)}>
                      {e.arm} / {e.name}
                    </button>
                  </td>
                  <td style={{ whiteSpace: 'normal', minWidth: 200 }}>{e.sentence}</td>
                  <td>{fmt(e.predicted)}</td>
                  <td>{fmt(e.actual)}</td>
                  <td>
                    {e.outcome ?? '--'} / {e.passed ? 'Correto' : 'Nao resolvido'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Widget>
    </>
  );
}
