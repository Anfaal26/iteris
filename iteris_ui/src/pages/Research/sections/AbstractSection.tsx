/**
 * AbstractSection — ArXiv-style project abstract with title, authors,
 * institution info, date, and badge pills (spec §5).
 */
import React from 'react';

/** Props for AbstractSection. */
export interface AbstractSectionProps {
  id?: string;
}

/**
 * Renders the paper abstract in an ArXiv-style layout.
 * No inline colours — only Tailwind token classes.
 */
export const AbstractSection: React.FC<AbstractSectionProps> = ({ id = 'abstract' }) => {
  return (
    <section id={id} aria-labelledby="abstract-heading" className="py-12 scroll-mt-16">
      <div className="max-w-2xl">
        {/* Badge pills */}
        <div className="flex flex-wrap gap-2 mb-6">
          <span className="px-3 py-1 rounded-full text-xs font-body font-medium border border-border text-muted">
            Taylor's University
          </span>
          <span className="px-3 py-1 rounded-full text-xs font-body font-medium border border-border text-muted">
            PRJ63504
          </span>
        </div>

        {/* Title */}
        <h1
          id="abstract-heading"
          className="font-heading text-2xl font-bold text-text leading-snug mb-4"
        >
          Iteris: Deep Reinforcement Learning for Adaptive Contour
          Refinement in Medical Image Segmentation
        </h1>

        {/* Authors */}
        <p className="font-body text-sm text-muted mb-1">
          Mohammad Anfaal Hossain&thinsp;·&thinsp;Capstone Team PRJ63504
        </p>
        <p className="font-body text-sm text-muted mb-1">
          School of Computer Science and Digital Technology, Taylor's University
        </p>
        <p className="font-body text-sm text-muted mb-6">
          <span className="px-2 py-0.5 rounded-full border border-border text-xs mr-2">
            Evaluated 2026-07-20
          </span>
          Academic year 2025/2026
        </p>

        {/* Abstract body */}
        <div
          className="prose max-w-none font-body text-sm text-text leading-relaxed space-y-3"
          aria-label="Abstract"
        >
          <p>
            Classical segmentation networks commit to a boundary in a single forward pass.
            This work asks what happens when the model can look again. We frame boundary
            refinement as a sequential decision-making problem: a frozen U-Net produces an
            initial contour, and a deep reinforcement learning (DRL) agent revises it
            step by step by displacing contour vertices along outward normals within
            angular sectors, learning through interaction with a reward signal rather than
            through imitation of ground truth.
          </p>
          <p>
            We compare two DRL formulations on the same contour-refinement environment — a
            discrete Dueling Double DQN over angular sectors, and a continuous TD3
            (Twin Delayed DDPG) agent — against two segmentation backbones, a compact
            Lite U-Net and an Attention Residual U-Net, on two independent benchmarks: the
            CAMUS cardiac ultrasound dataset (LV endocardium, LV epicardium, left atrium)
            and the BRISC brain MRI dataset (glioma, meningioma, pituitary). The reward is a
            baseline-centred, potential-based shaping term (Φ = K·(Dice − Dice₀)) chosen
            specifically to remove the path-dependence and discount drag that made earlier
            reward formulations collapse to a do-nothing policy at the baseline.
          </p>
          <p>
            Across 16 evaluation runs (2 agents × 2 backbones × 4 classes), the pattern is
            consistent: against the strong Attention U-Net (Phase A), neither agent
            reliably exceeds it — mean CAMUS Dice 0.896 (DuelingDDQN) / 0.885 (TD3) vs. 0.900
            baseline; BRISC 0.867 / 0.874 vs. 0.870. Against the weaker Lite U-Net (Phase B),
            DuelingDDQN edges past on CAMUS (0.860 vs. 0.847, +1.3&nbsp;pp) but not on BRISC
            (0.753 vs. 0.819, −6.6&nbsp;pp). Discrete and continuous action spaces perform
            near-identically head-to-head, and deployed Dice correlates 0.98–1.00 with the
            backbone's own Dice across every run — evidence the agents are not yet adding an
            independent, structure-aware signal beyond what the backbone already provides.
            See{' '}
            <a href="#results" className="text-accent hover:underline">
              Results
            </a>{' '}
            for the full breakdown and{' '}
            <a href="#ablations" className="text-accent hover:underline">
              Ablations
            </a>{' '}
            for the diagnosis.
          </p>
        </div>
      </div>
    </section>
  );
};

export default AbstractSection;
