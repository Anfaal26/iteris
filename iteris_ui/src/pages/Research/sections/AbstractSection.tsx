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
          Deep Reinforcement Learning for Adaptive Boundary Refinement in
          Medical Image Segmentation
        </h1>

        {/* Authors */}
        <p className="font-body text-sm text-muted mb-1">
          Ahmad Faaiz Anwar&thinsp;·&thinsp;Capstone Team PRJ63504
        </p>
        <p className="font-body text-sm text-muted mb-1">
          School of Computer Science and Digital Technology, Taylor's University
        </p>
        <p className="font-body text-sm text-muted mb-6">Submitted June 2026</p>

        {/* Abstract body */}
        <div
          className="prose max-w-none font-body text-sm text-text leading-relaxed space-y-3"
          aria-label="Abstract"
        >
          <p>
            Accurate segmentation of cardiac and neural structures in medical images is a
            prerequisite for many downstream clinical workflows, yet standard encoder-decoder
            networks such as U-Net often produce contours with irregular boundaries and
            inconsistent delineation of thin or low-contrast structures. We address this
            limitation by framing boundary refinement as a sequential decision-making problem
            and solving it with deep reinforcement learning (DRL) agents that iteratively
            adjust segmentation contours produced by a frozen U-Net backbone.
          </p>
          <p>
            We train and evaluate four DRL algorithms — Deep Q-Network (DQN), Double DQN
            (DDQN), Dueling DQN, and Deep Deterministic Policy Gradient (DDPG) — on two
            independent benchmarks: the CAMUS echocardiography dataset (450 patients, three
            cardiac structures) and the BRISC multi-class brain MRI dataset (three tumour
            classes). The Markov Decision Process is formulated with a patch-CNN state
            encoder, a structure-aware multi-component reward that jointly optimises Dice
            coefficient, boundary smoothness, and anatomical plausibility, and an
            episode-start reward that seeds the agent close to the gold-standard contour.
          </p>
          <p>
            Our best-performing agent, DDPG with a continuous vertex-displacement action
            space, achieves a mean Dice of 0.912 on CAMUS and 0.840 on BRISC — representing
            gains of 2.2 pp and 3.0 pp respectively over the U-Net baseline. Discrete agents
            (DQN, DDQN, Dueling DQN) achieve consistent improvements of 1.1–1.8 pp. A
            transfer learning experiment demonstrates that fine-tuning a CAMUS-trained DDPG
            agent on as few as 20% of BRISC labels recovers 98.3% of the fully-supervised
            Dice, indicating strong cross-modality generalisation. Ablation studies confirm
            the importance of the per-structure reward weighting and target-network
            stabilisation.
          </p>
        </div>
      </div>
    </section>
  );
};

export default AbstractSection;
