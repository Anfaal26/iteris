/**
 * ResearchMetrics — Section 4: full-width dark strip with 5 key research metrics.
 * Metric numbers use the iteris gradient via bg-clip-text.
 */

import React from 'react';

interface Metric {
  value: string;
  label: string;
  subLabel: string;
}

const METRICS: Metric[] = [
  { value: '0.912', label: 'Best Dice Score', subLabel: 'DDPG · CAMUS' },
  { value: '3', label: 'DRL Agent Families', subLabel: 'DQN · DDQN · DDPG' },
  { value: '2', label: 'Datasets Evaluated', subLabel: 'CAMUS (Cardiac US) · BRISC (Brain MRI)' },
  { value: 'p < 0.001', label: 'Statistical Significance', subLabel: 'Wilcoxon signed-rank · 5-fold CV' },
  { value: '20', label: 'Episode Steps', subLabel: 'DRL boundary refinement per image' },
];

/** Section 4 — Research metrics strip. */
export const ResearchMetrics: React.FC = () => (
  <section
    aria-label="Research metrics"
    className="bg-landing-footer py-16 px-6"
  >
    <div className="mx-auto max-w-6xl">
      <ol className="flex flex-col sm:flex-row items-stretch divide-y sm:divide-y-0 sm:divide-x divide-white/[0.08]" aria-label="Key research metrics">
        {METRICS.map((m) => (
          <li
            key={m.label}
            className="flex flex-col gap-1.5 px-8 py-6 sm:py-0 flex-1 first:pl-0 last:pr-0"
          >
            {/* Gradient number */}
            <span
              className="font-mono font-bold text-[2.5rem] leading-none bg-iteris-gradient bg-clip-text text-transparent"
              aria-label={m.value}
            >
              {m.value}
            </span>
            <span className="font-heading font-bold text-sm text-landing-text">
              {m.label}
            </span>
            <span className="text-xs text-landing-text/35 leading-snug">
              {m.subLabel}
            </span>
          </li>
        ))}
      </ol>
    </div>
  </section>
);

export default ResearchMetrics;
