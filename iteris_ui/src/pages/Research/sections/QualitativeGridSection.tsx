/**
 * QualitativeGridSection — real best/median/worst refinement cases (U-Net
 * init vs. agent-refined vs. ground truth), Phase A. Source: DRL Outputs
 * export, 2026-07-20 (spec §5).
 */
import React from 'react';

/** Props for QualitativeGridSection. */
export interface QualitativeGridSectionProps {
  id?: string;
}

interface CaseGrid {
  src: string;
  alt: string;
  label: string;
}

const GRIDS: CaseGrid[] = [
  {
    src: '/research/qualitative/camus-lv_endo-dueling-phaseA-comparison.png',
    alt: 'CAMUS LV_endo DuelingDDQN best/median/worst refinement cases',
    label: 'CAMUS / LV_endo — DuelingDDQN',
  },
  {
    src: '/research/qualitative/camus-lv_endo-td3-phaseA-comparison.png',
    alt: 'CAMUS LV_endo TD3 best/median/worst refinement cases',
    label: 'CAMUS / LV_endo — TD3',
  },
  {
    src: '/research/qualitative/brisc-tumor-dueling-phaseA-comparison.png',
    alt: 'BRISC tumor DuelingDDQN best/median/worst refinement cases',
    label: 'BRISC / tumor — DuelingDDQN',
  },
  {
    src: '/research/qualitative/brisc-tumor-td3-phaseA-comparison.png',
    alt: 'BRISC tumor TD3 best/median/worst refinement cases',
    label: 'BRISC / tumor — TD3',
  },
];

/**
 * Real best/median/worst-gain qualitative grids per flagship class, Phase A.
 */
export const QualitativeGridSection: React.FC<QualitativeGridSectionProps> = ({
  id = 'figures',
}) => {
  return (
    <section id={id} aria-labelledby="figures-heading" className="py-12 scroll-mt-16">
      <h2 id="figures-heading" className="font-heading text-xl font-bold text-text mb-4">
        Qualitative Results
      </h2>
      <p className="text-sm font-body text-muted mb-8">
        Each grid picks the best-gain, median-gain, and worst-gain test case for that run
        (U-Net init contour → agent-refined contour → ground truth), against the Attention
        U-Net baseline (Phase A). Run inference on your own scan in the{' '}
        <a href="/workspace" className="text-accent hover:underline">
          Workspace
        </a>
        .
      </p>

      <div className="grid gap-8 lg:grid-cols-2">
        {GRIDS.map((g) => (
          <figure key={g.src} className="bg-surface border border-border rounded-lg p-3">
            <img src={g.src} alt={g.alt} className="w-full rounded" />
            <figcaption className="text-xs font-body text-muted mt-2">{g.label}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
};

export default QualitativeGridSection;
