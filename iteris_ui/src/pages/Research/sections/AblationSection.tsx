/**
 * AblationSection — ablation study table from spec §5.
 * Five rows: Discrete vs Continuous, Replay buffer, Target network,
 * Episode-start reward, Per-structure reward.
 */
import React from 'react';

/** Props for AblationSection. */
export interface AblationSectionProps {
  id?: string;
}

interface AblationRow {
  target: string;
  camusDice: number;
  briscDice: number;
  delta: string;
  interpretation: string;
}

const ABLATION_ROWS: AblationRow[] = [
  {
    target: 'Discrete vs Continuous action space',
    camusDice: 0.908,
    briscDice: 0.835,
    delta: '−0.004 / −0.005',
    interpretation:
      'Continuous (DDPG) outperforms best discrete agent; coarse grid limits fine boundary adjustments.',
  },
  {
    target: 'Replay buffer (remove)',
    camusDice: 0.891,
    briscDice: 0.814,
    delta: '−0.021 / −0.026',
    interpretation:
      'Large degradation confirms experience replay is essential for sample efficiency.',
  },
  {
    target: 'Target network (remove)',
    camusDice: 0.883,
    briscDice: 0.807,
    delta: '−0.029 / −0.033',
    interpretation:
      'Training instability; Q-values diverge on 18% of training runs without target network.',
  },
  {
    target: 'Episode-start reward (remove)',
    camusDice: 0.897,
    briscDice: 0.819,
    delta: '−0.015 / −0.021',
    interpretation:
      'Agent takes longer to reach productive exploration region; convergence slows by ~30%.',
  },
  {
    target: 'Per-structure reward weighting (remove)',
    camusDice: 0.903,
    briscDice: 0.828,
    delta: '−0.009 / −0.012',
    interpretation:
      'Thin structures (LV endo, pituitary) degraded disproportionately without up-weighting.',
  },
];

// Full model reference Dice for delta colour-coding
const FULL_CAMUS = 0.912;
const FULL_BRISC = 0.840;

/**
 * Ablation study table showing the contribution of each design choice.
 */
export const AblationSection: React.FC<AblationSectionProps> = ({ id = 'ablations' }) => {
  return (
    <section id={id} aria-labelledby="ablations-heading" className="py-12 scroll-mt-16">
      <h2 id="ablations-heading" className="font-heading text-xl font-bold text-text mb-4">
        Ablation Study
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        Each row removes or replaces one component from the full DDPG model and re-evaluates
        on both test sets. Delta is reported relative to the full model (CAMUS 0.912, BRISC
        0.840).
      </p>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm font-body" aria-label="Ablation study results">
          <thead>
            <tr className="bg-bg">
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                Ablation target
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                CAMUS Dice
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                BRISC Dice
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                Δ (C / B)
              </th>
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                Interpretation
              </th>
            </tr>
          </thead>
          <tbody>
            {ABLATION_ROWS.map((row, i) => {
              const camusColor =
                row.camusDice < FULL_CAMUS - 0.01 ? 'text-error' : 'text-warning';
              const briscColor =
                row.briscDice < FULL_BRISC - 0.01 ? 'text-error' : 'text-warning';
              return (
                <tr key={row.target} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                  <td className="px-4 py-2 text-text font-medium">{row.target}</td>
                  <td className={`px-4 py-2 text-right font-mono text-xs ${camusColor}`}>
                    {row.camusDice.toFixed(3)}
                  </td>
                  <td className={`px-4 py-2 text-right font-mono text-xs ${briscColor}`}>
                    {row.briscDice.toFixed(3)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-error">
                    {row.delta}
                  </td>
                  <td className="px-4 py-2 text-muted text-xs">{row.interpretation}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default AblationSection;
