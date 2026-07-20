/**
 * ConvergenceSection — real training curves (mean validation Dice vs. step,
 * with the U-Net init Dice and best-seen ceiling overlaid) for the flagship
 * class per dataset, Phase A. Source: DRL Outputs export, 2026-07-20 (spec §5).
 */
import React from 'react';

/** Props for ConvergenceSection. */
export interface ConvergenceSectionProps {
  id?: string;
}

interface CurveCard {
  src: string;
  alt: string;
  label: string;
}

const CURVES: CurveCard[] = [
  {
    src: '/research/curves/camus-lv_endo-dueling-phaseA-curves.png',
    alt: 'CAMUS LV_endo DuelingDDQN learning curve, Phase A',
    label: 'CAMUS / LV_endo — DuelingDDQN',
  },
  {
    src: '/research/curves/camus-lv_endo-td3-phaseA-curves.png',
    alt: 'CAMUS LV_endo TD3 learning curve, Phase A',
    label: 'CAMUS / LV_endo — TD3',
  },
  {
    src: '/research/curves/brisc-tumor-dueling-phaseA-curves.png',
    alt: 'BRISC tumor DuelingDDQN learning curve, Phase A',
    label: 'BRISC / tumor — DuelingDDQN',
  },
  {
    src: '/research/curves/brisc-tumor-td3-phaseA-curves.png',
    alt: 'BRISC tumor TD3 learning curve, Phase A',
    label: 'BRISC / tumor — TD3',
  },
];

/**
 * Example convergence trajectories (Phase A, one flagship class per dataset)
 * — not exhaustive across all 16 runs; see Results for the full table.
 */
export const ConvergenceSection: React.FC<ConvergenceSectionProps> = ({
  id = 'convergence',
}) => {
  return (
    <section id={id} aria-labelledby="convergence-heading" className="py-12 scroll-mt-16">
      <h2 id="convergence-heading" className="font-heading text-xl font-bold text-text mb-4">
        Convergence Curves
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        Mean validation Dice vs. training step (50k steps, checkpointed every 12.5k), against
        the frozen Attention U-Net initial contour (Phase A). The best-seen line tracks the
        ceiling reachable at any checkpoint; the deployed agent is value-floored to never fall
        below the U-Net init if it drifts — see the Ablations section for why the two aren't
        always the same run. Two flagship classes shown here; all 16 runs (both phases, both
        datasets, all CAMUS classes) are in the repository.
      </p>
      <div className="grid gap-6 lg:grid-cols-2">
        {CURVES.map((c) => (
          <figure key={c.src} className="bg-surface border border-border rounded-lg p-3">
            <img src={c.src} alt={c.alt} className="w-full rounded" />
            <figcaption className="text-xs font-body text-muted mt-2">{c.label}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
};

export default ConvergenceSection;
