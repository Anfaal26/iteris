/**
 * MethodsSection — Problem Formulation, MDP Design, DRL Agents,
 * Reward Structure (styled table), Evaluation Metrics (spec §5).
 */
import React from 'react';

/** Props for MethodsSection. */
export interface MethodsSectionProps {
  id?: string;
}

interface RewardRow {
  component: string;
  formula: string;
  weight: string;
  purpose: string;
}

const REWARD_ROWS: RewardRow[] = [
  {
    component: 'Potential-based shaping',
    formula: 'Φ(s) = K·(Dice(s) − Dice₀)',
    weight: 'K = 10 (CAMUS) / 15 (BRISC)',
    purpose:
      'Baseline-centred so holding position pays ~0 return, preventing the collapse to a do-nothing policy seen with an un-centred Φ = Dice at high baseline Dice',
  },
  {
    component: 'Step reward',
    formula: 'r(s,a,s′) = γΦ(s′) − Φ(s)',
    weight: '—',
    purpose: 'Reward difference telescopes to K·(Dice_final − Dice₀) over an episode, independent of the action path taken',
  },
];

/**
 * Methods section covering the DRL formulation in detail.
 */
export const MethodsSection: React.FC<MethodsSectionProps> = ({ id = 'methods' }) => {
  return (
    <section id={id} aria-labelledby="methods-heading" className="py-12 scroll-mt-16">
      <h2 id="methods-heading" className="font-heading text-xl font-bold text-text mb-8">
        Methods
      </h2>

      {/* 1. Problem Formulation */}
      <div className="mb-8">
        <h3 className="font-heading text-base font-semibold text-text mb-3">
          1. Problem Formulation
        </h3>
        <div className="space-y-2 text-sm font-body text-muted leading-relaxed">
          <p>
            Given a medical image <em>I</em> and an initial contour <em>C&#x2080;</em>{' '}
            produced by a frozen U-Net backbone, the objective is to find a refined
            contour <em>C*</em> that maximises the Dice coefficient with respect to the
            ground-truth annotation <em>G</em>.
          </p>
          <p>
            Contour refinement is modelled as a fixed-horizon Markov Decision Process.
            Rather than acting on individual vertices directly, the contour is divided
            into a fixed number of angular sectors around its centroid; each sector
            defines an outward-normal direction along which the agent can push the
            boundary points it contains. This angular (rather than per-index) binning
            keeps the action-to-location mapping stable across samples, which is what
            makes the action space learnable.
          </p>
        </div>
      </div>

      {/* 2. MDP Design */}
      <div className="mb-8">
        <h3 className="font-heading text-base font-semibold text-text mb-3">
          2. MDP Design
        </h3>
        <div className="space-y-2 text-sm font-body text-muted leading-relaxed">
          <p>
            <strong className="text-text">Environment</strong> — <code className="text-xs font-mono text-accent">ContourRefineEnv</code>,
            operating on the U-Net-derived contour rather than the raw pixel mask. An
            earlier paradigm used a global 3-D action (SDT-threshold morphology +
            translation) applied to the whole mask; this is structurally capped at the
            baseline, since the best response to an already near-perfect mask is the
            identity action. It has been archived as an ablation only.
          </p>
          <p>
            <strong className="text-text">Action space (discrete)</strong> — Dueling
            Double DQN selects a sector and a displacement bucket per step.
          </p>
          <p>
            <strong className="text-text">Action space (continuous)</strong> — TD3 emits
            a continuous per-sector displacement vector, clipped double-Q targets, and
            delayed policy updates for stability.
          </p>
          <p>
            <strong className="text-text">Backbones</strong> — Two U-Net variants
            provide the initial contour: a compact <em>Lite U-Net</em> (deliberately
            given headroom above the DRL floor) and an <em>Attention Residual U-Net</em>{' '}
            (the deployed baseline and upper-bound competitor).
          </p>
        </div>
      </div>

      {/* 3. DRL Agents */}
      <div className="mb-8">
        <h3 className="font-heading text-base font-semibold text-text mb-3">
          3. DRL Agents
        </h3>
        <div className="space-y-2 text-sm font-body text-muted leading-relaxed">
          <p>
            Only two algorithms are carried forward to full evaluation, both on the
            contour environment: <strong className="text-text">Dueling DDQN</strong>{' '}
            (discrete sectors, value/advantage stream decomposition,{' '}
            <em>Q = V + A − mean(A)</em>) and <strong className="text-text">TD3</strong>{' '}
            (continuous sectors). Earlier global-action variants (DQN, DDQN, DDPG) are
            kept only as archived ablations to demonstrate the global-action ceiling.
          </p>
          <p>
            Exact hyperparameters (learning rates, replay buffer size, target-update
            interval, TD3-specific behaviour-cloning / hard-mining / curriculum
            settings) are tracked per-run in the training configs rather than restated
            here, since they are still being tuned during active training — see the
            repository's <code className="text-xs font-mono text-accent">configs/</code>{' '}
            directory for the values used in any given run.
          </p>
        </div>
      </div>

      {/* 4. Reward Structure */}
      <div className="mb-8">
        <h3 className="font-heading text-base font-semibold text-text mb-3">
          4. Reward Structure
        </h3>
        <p className="text-sm font-body text-muted mb-4">
          The reward is potential-based shaping (PBRS) centred on the baseline Dice,
          rather than raw Dice — this was a deliberate fix, not the original design.
          An un-centred Φ = Dice at a ~0.94 baseline made <em>holding position</em> pay
          (γ − 1)·Φ ≈ −0.009 per step, which collapsed every agent to a stop-at-baseline
          policy regardless of algorithm:
        </p>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm font-body" aria-label="Reward components">
            <thead>
              <tr className="bg-bg">
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                  Component
                </th>
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                  Formula
                </th>
                <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                  Weight
                </th>
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                  Purpose
                </th>
              </tr>
            </thead>
            <tbody>
              {REWARD_ROWS.map((row, i) => (
                <tr key={row.component} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                  <td className="px-4 py-2 text-text font-medium">{row.component}</td>
                  <td className="px-4 py-2 font-mono text-xs text-accent">{row.formula}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-text">
                    {row.weight}
                  </td>
                  <td className="px-4 py-2 text-muted">{row.purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Evaluation Metrics */}
      <div>
        <h3 className="font-heading text-base font-semibold text-text mb-3">
          5. Evaluation Metrics
        </h3>
        <ul className="space-y-2 text-sm font-body text-muted leading-relaxed list-none pl-0">
          <li>
            <strong className="text-text">Dice coefficient</strong> — 2|P ∩ G| / (|P| + |G|);
            primary ranking metric.
          </li>
          <li>
            <strong className="text-text">Intersection-over-Union (IoU)</strong> — |P ∩ G| /
            |P ∪ G|; monotonically related to Dice but penalises FP/FN differently.
          </li>
          <li>
            <strong className="text-text">Hausdorff distance at 95th percentile (HD95)</strong>{' '}
            — robust surface-distance metric; computed on the refined contour vs GT boundary
            at 95% coverage to exclude outlier vertices.
          </li>
          <li>
            <strong className="text-text">Statistical significance</strong> — paired Wilcoxon
            signed-rank test per agent vs the U-Net baseline, with a Bonferroni correction
            across the set of comparisons. Reported as p-values in the Results section once
            evaluation runs complete.
          </li>
          <li>
            <strong className="text-text">Headroom check</strong> — before a full training
            run, an oracle-contour-ceiling diagnostic compares the best achievable Dice under
            the current contour representation against the baseline, as a cheap go/no-go
            signal for whether the representation has room to improve on.
          </li>
        </ul>
      </div>
    </section>
  );
};

export default MethodsSection;
