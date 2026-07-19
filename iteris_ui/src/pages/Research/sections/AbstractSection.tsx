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
            In progress
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
            Evaluation is not yet complete — the discrete and continuous agents are
            currently in training against the Lite U-Net baseline, with the Attention U-Net
            serving as the upper-bound competitor. Result, convergence, and ablation figures
            on this page will populate as evaluation runs finish; see the{' '}
            <a href="#results" className="text-accent hover:underline">
              Results
            </a>{' '}
            section for current status.
          </p>
        </div>
      </div>
    </section>
  );
};

export default AbstractSection;
