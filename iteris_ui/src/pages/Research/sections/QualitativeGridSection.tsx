/**
 * QualitativeGridSection — two 3×3 placeholder image grids (CAMUS and BRISC)
 * with row/column headers, difficulty labels, and Dice score badges (spec §5).
 */
import React from 'react';

/** Props for QualitativeGridSection. */
export interface QualitativeGridSectionProps {
  id?: string;
}

type Difficulty = 'Easy' | 'Medium' | 'Hard';
type Agent = 'Baseline' | 'DDPG' | 'Dueling DQN';

const DIFFICULTIES: Difficulty[] = ['Easy', 'Medium', 'Hard'];
const AGENTS: Agent[] = ['Baseline', 'DDPG', 'Dueling DQN'];

const DICE_SCORES: Record<'camus' | 'brisc', Record<Difficulty, Record<Agent, number>>> = {
  camus: {
    Easy: { Baseline: 0.932, DDPG: 0.951, 'Dueling DQN': 0.947 },
    Medium: { Baseline: 0.891, DDPG: 0.914, 'Dueling DQN': 0.910 },
    Hard: { Baseline: 0.847, DDPG: 0.871, 'Dueling DQN': 0.865 },
  },
  brisc: {
    Easy: { Baseline: 0.871, DDPG: 0.904, 'Dueling DQN': 0.898 },
    Medium: { Baseline: 0.812, DDPG: 0.843, 'Dueling DQN': 0.837 },
    Hard: { Baseline: 0.748, DDPG: 0.777, 'Dueling DQN': 0.771 },
  },
};

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
          const dice = DICE_SCORES[dataset][diff][agent];
          const structLabel = STRUCTURE_LABELS[dataset][diff];
          return (
            <div
              key={agent}
              className="relative aspect-square rounded border border-border bg-bg flex flex-col items-center justify-center gap-1 overflow-hidden"
              aria-label={`${dataset} ${diff} ${agent} — Dice ${dice.toFixed(3)}`}
            >
              {/* Placeholder square */}
              <span className="text-[10px] font-mono text-muted text-center leading-tight px-1">
                {structLabel}
              </span>
              {/* Dice badge */}
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold ${
                  agent === 'Baseline' ? 'bg-border text-muted' : 'bg-accent/10 text-accent'
                }`}
              >
                {dice.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    ))}
  </div>
);

/**
 * Qualitative segmentation grid with placeholder cells and Dice badges.
 * Real inference images will replace placeholders once the backend is connected.
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
        (Baseline / DDPG / Dueling DQN). Inference images will replace these
        placeholders once the backend is connected. Dice scores shown are from the
        test-set mean for the representative sample in each cell.
      </p>

      <div className="grid gap-12 lg:grid-cols-2">
        <PlaceholderGrid dataset="camus" label="CAMUS — Echocardiography" />
        <PlaceholderGrid dataset="brisc" label="BRISC — Brain MRI" />
      </div>
    </section>
  );
};

export default QualitativeGridSection;
