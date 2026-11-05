import { Microscope, FileCheck2 } from 'lucide-react';
import { Empty, Widget } from './components';
import { bytes, fmt } from './useAxon';
import type { Job, RecordEvent } from './types';

const tasks: Record<number, string> = {
  0: 'Soma',
  1: 'Produto',
  2: 'Diferenca',
  4: 'Norma 2D',
  8: 'Norma 3D / escala',
  9: 'Norma 3D * escala',
  10: 'Norma 3D - deslocamento',
  11: 'Distancia 2D',
};
const arms: Record<string, string> = {
  cold: 'Sem heranca',
  'founder-random': 'Fundador / aleatoria',
  'learned-random': 'Aprendido / aleatoria',
  'learned-active': 'Aprendido / ativa',
};
const cents = (value?: number) =>
  value == null
    ? '--'
    : new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(value / 100);
export function Cumulative({
  job,
  inspect,
}: {
  job: Job;
  inspect: (title: string, value: unknown) => void;
}) {
  const comparisons = job.cases.filter((c) => c.event === 'validation' && c.arm);
  const plans = job.samples.filter((e) => e.event === 'plan');
  const probe = job.samples.findLast((e) => e.event === 'study_query');
  const cell = (c: RecordEvent) => (c.admitted ? 'Admitido' : 'Sem modelo');
  return (
    <>
      <Widget
        title="Investigacao e heranca"
        onInspect={() => inspect('Populacao cumulativa', job.candidates)}
      >
        <div className="lab-metrics instrument">
          <div>
            <span className="eyebrow">PERGUNTAS / ULTIMA TAREFA</span>
            <strong>{fmt(probe?.questions)}</strong>
          </div>
          <div>
            <span className="eyebrow">HIPOTESES RESTANTES</span>
            <strong>{fmt(probe?.survivors)}</strong>
          </div>
          <div>
            <span className="eyebrow">SEPARACAO DA PERGUNTA</span>
            <strong>{fmt(probe?.disagreement)}</strong>
          </div>
          <div>
            <span className="eyebrow">DIGITOS / GENOMA COMPLETO</span>
            <strong>{fmt(job.result?.seed_digits)}</strong>
          </div>
        </div>
        <div
          className="lab-table-scroll"
          role="region"
          aria-label="Populacao cumulativa"
          tabIndex={0}
        >
          <table className="lab-table">
            <thead>
              <tr>
                <th>Geracao</th>
                <th>Seed</th>
                <th>Admissoes</th>
                <th>Perguntas</th>
                <th>Busca</th>
                <th>Programas</th>
                <th>Codificado</th>
                <th>Selecao</th>
              </tr>
            </thead>
            <tbody>
              {job.candidates.map((c) => (
                <tr key={c.candidate}>
                  <td>{fmt(c.generation)}</td>
                  <td>{fmt(c.candidate)}</td>
                  <td>{fmt(typeof c.admitted === 'number' ? c.admitted : undefined)} / 5</td>
                  <td>{fmt(c.questions)}</td>
                  <td>{fmt(c.work)}</td>
                  <td>{fmt(c.skills)}</td>
                  <td>{c.encoded_bytes == null ? '--' : bytes(c.encoded_bytes)}</td>
                  <td>{c.frontier ? 'Selecionada' : 'Avaliada'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Widget>
      <Widget
        title="Comparacoes pareadas"
        sub={`${comparisons.length} CHECKPOINTS`}
        onInspect={() => inspect('Validacao cumulativa', comparisons)}
      >
        {comparisons.length ? (
          <div
            className="lab-table-scroll"
            role="region"
            aria-label="Comparacoes de aprendizado"
            tabIndex={0}
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Grupo</th>
                  <th>Tarefa</th>
                  <th>Familia</th>
                  <th>Perguntas</th>
                  <th>Busca</th>
                  <th>Admissao</th>
                  <th>Corretos</th>
                  <th>Dominio</th>
                  <th>Sem modelo</th>
                  <th>Errados</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((c) => (
                  <tr key={c.case}>
                    <td>{arms[c.arm!] ?? c.arm}</td>
                    <td>{tasks[c.task!] ?? c.task}</td>
                    <td>{c.novel ? 'Nova' : 'Conhecida'}</td>
                    <td>{fmt(c.questions)}</td>
                    <td>{fmt(c.search_work)}</td>
                    <td>{cell(c)}</td>
                    <td>
                      {fmt(c.correct)} / {fmt(c.heldout)}
                    </td>
                    <td>{fmt(c.domain_refusals)}</td>
                    <td>{fmt(c.unavailable)}</td>
                    <td>{fmt(c.wrong)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty icon={Microscope} title="Sem comparacoes registradas" />
        )}
      </Widget>
      <Widget
        title="Planos de arquivos"
        sub="DRY-RUN / SEM COMPRA"
        onInspect={() => inspect('Recibos dos planos', plans)}
      >
        {plans.length ? (
          <div
            className="lab-table-scroll"
            role="region"
            aria-label="Planos de arquivos"
            tabIndex={0}
          >
            <table className="lab-table">
              <thead>
                <tr>
                  <th>Episodio</th>
                  <th>Item</th>
                  <th>Necessidade</th>
                  <th>Unidades</th>
                  <th>Custo</th>
                  <th>Limite</th>
                  <th>Pacotes</th>
                  <th>Estoque / hash</th>
                  <th>Ofertas / hash</th>
                </tr>
              </thead>
              <tbody>
                {plans.map((p) => (
                  <tr key={p.coreSeq}>
                    <td>{fmt(p.episode)}</td>
                    <td>{p.item}</td>
                    <td>{fmt(p.needed)}</td>
                    <td>{fmt(p.units)}</td>
                    <td>{cents(p.cost_cents)}</td>
                    <td>{cents(p.budget_cents)}</td>
                    <td>{p.choices || '--'}</td>
                    <td className="mono">{p.stock_hash}</td>
                    <td className="mono">{p.offers_hash}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty icon={FileCheck2} title="Sem planos registrados" />
        )}
      </Widget>
    </>
  );
}
