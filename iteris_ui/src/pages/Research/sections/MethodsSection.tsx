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
    component: 'Dice reward',
    formula: 'r_Dice = ΔDice(t)',
    weight: '0.50',
    purpose: 'Primary segmentation quality signal',
  },
  {
    component: 'Boundary smoothness',
    formula: 'r_smooth = −κ(t)',
    weight: '0.20',
    purpose: 'Penalises high-curvature contour artefacts',
  },
  {
    component: 'Anatomical plausibility',
    formula: 'r_anat = IoU(hull(t))',
    weight: '0.15',
    purpose: 'Enforces convex-hull shape prior',
  },
  {
    component: 'Episode-start bonus',
    formula: 'r_init = Dice(s_0)',
    weight: '0.10',
    purpose: 'Seeds agent near gold-standard contour',
  },
  {
    component: 'Per-structure weight',
    formula: 'w_s · r_s',
    weight: '0.05',
    purpose: 'Up-weights thin structures (LV endo, pituitary)',
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
            Given a medical image <em>I</em> and an initial segmentation mask{' '}
            <em>M&#x2080;</em> produced by a frozen U-Net, the objective is to find a
            refined mask <em>M*</em> that maximises the Dice coefficient with respect to
            the ground-truth annotation <em>G</em>.
          </p>
          <p>
            Boundary refinement is modelled as a fixed-horizon Markov Decision Process
            with horizon <em>T</em> = 50 steps. The agent operates on a local patch
            centred at each contour vertex and emits actions that displace vertices in
            either a discrete 8-direction grid (DQN / DDQN / Dueling DQN) or a
            continuous 2-D offset vector (DDPG).
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
            <strong className="text-text">State space</strong> — A 64 × 64 patch
            centred at the current contour vertex, concatenated with a one-hot
            structure ID and the current step normalised to [0, 1]. Encoded by a
            lightweight CNN (3 × Conv + 2 × FC) to a 256-dimensional embedding.
          </p>
          <p>
            <strong className="text-text">Action space (discrete)</strong> — 8 cardinal
            and diagonal unit displacements, each scaling a step size that decays
            linearly from 4 px (step 0) to 1 px (step 49).
          </p>
          <p>
            <strong className="text-text">Action space (continuous)</strong> — DDPG
            emits a 2-D continuous displacement vector clipped to [−4, 4] px.
          </p>
          <p>
            <strong className="text-text">Transition</strong> — Each action moves the
            target vertex and triggers a full mask recompute via polygon fill; adjacent
            vertices are smoothed with a Gaussian kernel (σ = 1.0) to prevent isolated
            spikes.
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
            All discrete agents share a common replay-buffer (capacity 10<sup>5</sup>,
            uniform sampling) and target-network update interval of 500 steps. DQN uses
            single-network Q-evaluation. DDQN decouples action selection (online) from
            value estimation (target) to reduce overestimation bias. Dueling DQN adds
            separate value and advantage streams recombined as{' '}
            <em>Q = V + A − mean(A)</em>.
          </p>
          <p>
            DDPG uses an Ornstein–Uhlenbeck exploration process (θ = 0.15, σ = 0.2)
            and soft target updates (τ = 0.005). The actor and critic each have two
            hidden layers of 256 units. Batch normalisation is applied after the first
            hidden layer of the critic.
          </p>
        </div>
      </div>

      {/* 4. Reward Structure */}
      <div className="mb-8">
        <h3 className="font-heading text-base font-semibold text-text mb-3">
          4. Reward Structure
        </h3>
        <p className="text-sm font-body text-muted mb-4">
          The composite reward at step <em>t</em> is a weighted sum of five components:
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
            signed-rank test (α = 0.05) for each agent vs U-Net baseline. Reported as p-values
            in the results table.
          </li>
        </ul>
      </div>
    </section>
  );
};

export default MethodsSection;
