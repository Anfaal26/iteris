/**
 * QualitativeGridSection — 3×3 placeholder grids (CAMUS and BRISC), columns
 * Baseline / Dueling DDQN / TD3. No fabricated Dice scores — cells stay
 * empty until real inference output is wired up (spec §5).
 */
import React from 'react';

/** Props for QualitativeGridSection. */
export interface QualitativeGridSectionProps {
  id?: string;
}

type Difficulty = 'Easy' | 'Medium' | 'Hard';
type Agent = 'Baseline' | 'Dueling DDQN' | 'TD3';

const DIFFICULTIES: Difficulty[] = ['Easy', 'Medium', 'Hard'];
const AGENTS: Agent[] = ['Baseline', 'Dueling DDQN', 'TD3'];

const STRUCTURE_LABELS: Record<'camus' | 'brisc', Record<Difficulty, string>> = {
  camus: {
    Easy: 'LV Endo + Epi + LA',
    Medium: 'LV Endo + Epi',
    Hard: 'LA boundary (low SNR)',
  },
  brisc: {
    Easy: 'Glioma (large)',
    Medium: 'Meningioma',
    Hard: 'Pituitary (small)',
  },
};

interface GridProps {
  dataset: 'camus' | 'brisc';
  label: string;
}

const PlaceholderGrid: React.FC<GridProps> = ({ dataset, label }) => (
  <div>
    <h3 className="font-heading text-base font-semibold text-text mb-4">{label}</h3>
    {/* Column headers */}
    <div className="grid grid-cols-4 gap-2 mb-1">
      <div />
      {AGENTS.map((agent) => (
        <div key={agent} className="text-center text-xs font-body font-semibold text-muted">
          {agent}
        </div>
      ))}
    </div>
    {/* Grid rows */}
    {DIFFICULTIES.map((diff) => (
      <div key={diff} className="grid grid-cols-4 gap-2 mb-2 items-center">
        {/* Row header */}
        <div className="text-xs font-body font-semibold text-muted text-right pr-2">
          {diff}
        </div>
        {AGENTS.map((agent) => {
          const structLabel = STRUCTURE_LABELS[dataset][diff];
          return (
            <div
              key={agent}
              className="relative aspect-square rounded border border-border bg-bg flex flex-col items-center justify-center gap-1 overflow-hidden"
              aria-label={`${dataset} ${diff} ${agent} — pending inference output`}
            >
              <span className="text-[10px] font-mono text-muted text-center leading-tight px-1">
                {structLabel}
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono text-muted border border-border">
                pending
              </span>
            </div>
          );
        })}
      </div>
    ))}
  </div>
);

/**
 * Qualitative segmentation grid with placeholder cells. Real inference images
 * and per-cell Dice will replace these once the workspace backend is wired
 * up for batch export.
 */
export const QualitativeGridSection: React.FC<QualitativeGridSectionProps> = ({
  id = 'figures',
}) => {
  return (
    <section id={id} aria-labelledby="figures-heading" className="py-12 scroll-mt-16">
      <h2 id="figures-heading" className="font-heading text-xl font-bold text-text mb-8">
        Qualitative Results
      </h2>
      <p className="text-sm font-body text-muted mb-8">
        3 × 3 grids stratified by difficulty (Easy / Medium / Hard) and method
        (Baseline / Dueling DDQN / TD3). Inference images and per-cell Dice scores
        will replace these placeholders once evaluation output exists — see the{' '}
        <a href="/workspace" className="text-accent hover:underline">
          Workspace
        </a>{' '}
        for live single-scan inference in the meantime.
      </p>

      <div className="grid gap-12 lg:grid-cols-2">
        <PlaceholderGrid dataset="camus" label="CAMUS — Echocardiography" />
        <PlaceholderGrid dataset="brisc" label="BRISC — Brain MRI" />
      </div>
    </section>
  );
};

export default QualitativeGridSection;
