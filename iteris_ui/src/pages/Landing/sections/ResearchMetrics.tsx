/** ResearchMetrics — glass metrics strip with 5 key research numbers. */
import React from 'react';

const METRICS = [
  { value: '0.912',   label: 'Best Dice Score',    sub: 'DDPG · CAMUS'               },
  { value: '3',       label: 'DRL Agent Families', sub: 'DQN · DDQN · DDPG'          },
  { value: '2',       label: 'Datasets Evaluated', sub: 'CAMUS · BRISC'              },
  { value: '<0.001',  label: 'p-value',            sub: 'Wilcoxon · 5-fold CV'       },
  { value: '20',      label: 'Episode Steps',      sub: 'Boundary refinement / image'},
];

export const ResearchMetrics: React.FC = () => (
  <section aria-label="Research metrics" className="py-20 px-6 lg:px-10" style={{
    background: 'rgba(3,5,8,0.9)',
    borderTop: '1px solid rgba(56,189,248,0.08)',
    borderBottom: '1px solid rgba(56,189,248,0.08)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
  }}>
    <div className="mx-auto max-w-6xl">
      <ol className="flex flex-col sm:flex-row items-stretch" aria-label="Key research metrics">
        {METRICS.map((m, i) => (
          <li key={m.label} className="flex flex-col gap-2 px-8 py-8 sm:py-0 flex-1 first:pl-0 last:pr-0"
            style={{ borderLeft: i > 0 ? '1px solid rgba(255,255,255,0.06)' : 'none' }}>
            <span className="font-mono font-bold text-[2.8rem] leading-none bg-iteris-gradient bg-clip-text text-transparent"
              style={{ WebkitBackgroundClip: 'text' }} aria-label={m.value}>{m.value}</span>
            <span className="font-heading font-semibold text-sm text-landing-text">{m.label}</span>
            <span className="text-[11px] leading-snug" style={{ color: 'rgba(240,249,255,0.32)' }}>{m.sub}</span>
          </li>
        ))}
      </ol>
    </div>
  </section>
);
export default ResearchMetrics;
