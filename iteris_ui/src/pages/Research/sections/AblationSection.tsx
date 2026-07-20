/**
 * AblationSection — the three-cap diagnosis for "agent doesn't beat baseline"
 * (representation / information / optimization), with real supporting
 * evidence from the 2026-07-20 evaluation pass (spec §5).
 */
import React from 'react';

/** Props for AblationSection. */
export interface AblationSectionProps {
  id?: string;
}

interface CapRow {
  cap: string;
  question: string;
  status: string;
}

const CAP_ROWS: CapRow[] = [
  {
    cap: 'Representation cap',
    question:
      'Every output is a smoothing B-spline fit of the largest connected component — the agent can never emit the raw U-Net mask itself, so its ceiling sits below a do-nothing policy by a fixed projection "tax."',
    status:
      'Consistent with results: deployed Dice correlates 0.98 (Phase A) / 1.00 (Phase B) with the U-Net init Dice — almost none of the variance in final performance is explained by anything the agent does.',
  },
  {
    cap: 'Information cap',
    question:
      'A GT-blind refiner sees the same image with less capacity than the U-Net and no privileged channels — on ambiguous regions the U-Net\'s residual error may simply not be resolvable without a learned shape prior.',
    status:
      'Consistent with results: both agents beat baseline more often on Phase B (weak, high-headroom Lite U-Net) than Phase A (strong Attention U-Net) — the ceiling tracks how much room the baseline leaves, not the agent\'s learning signal.',
  },
  {
    cap: 'Optimization / STOP-timing cap',
    question:
      'TD3 has no learned STOP action, so a deployed policy can drift past its best checkpoint before the episode ends.',
    status:
      'Consistent with results: drift correlates −1.00 (Phase A) / −0.99 (Phase B) with Δ IoU — the more a run drifts post-peak, the worse its deployed IoU, and TD3 loses to baseline more often than DuelingDDQN (which has an implicit floor via value-clipping).',
  },
];

/**
 * The three-cap diagnosis, with real correlation evidence from the
 * 2026-07-20 evaluation pass backing each row.
 */
export const AblationSection: React.FC<AblationSectionProps> = ({ id = 'ablations' }) => {
  return (
    <section id={id} aria-labelledby="ablations-heading" className="py-12 scroll-mt-16">
      <h2 id="ablations-heading" className="font-heading text-xl font-bold text-text mb-4">
        Ablation Study
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        When a refinement agent fails to beat the U-Net baseline, three distinct causes get
        conflated. All three are consistent with the evaluation pass below — this isn't a
        controlled ablation (no component was individually switched off and re-run), but the
        correlation structure across all 16 runs points the same way on every cap.
      </p>

      <div className="overflow-x-auto rounded-lg border border-border mb-8">
        <table className="w-full text-sm font-body" aria-label="Ablation diagnosis">
          <thead>
            <tr className="bg-bg">
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                Cap
              </th>
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                What it means
              </th>
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                Evidence
              </th>
            </tr>
          </thead>
          <tbody>
            {CAP_ROWS.map((row, i) => (
              <tr key={row.cap} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                <td className="px-4 py-2 text-text font-medium align-top whitespace-nowrap">
                  {row.cap}
                </td>
                <td className="px-4 py-2 text-muted text-xs align-top">{row.question}</td>
                <td className="px-4 py-2 text-muted text-xs align-top">{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/drl_metric_correlation_PhaseA.png"
            alt="DRL metric correlation matrix, Phase A"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            Metric correlations, Phase A — deploy_dice/init_dice = 0.98.
          </figcaption>
        </figure>
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/drl_metric_correlation_PhaseB.png"
            alt="DRL metric correlation matrix, Phase B"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            Metric correlations, Phase B — deploy_dice/init_dice = 1.00.
          </figcaption>
        </figure>
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/discrete_vs_continuous_head_to_head.png"
            alt="Head-to-head DuelingDDQN vs TD3 deployed Dice, same class and phase"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            DuelingDDQN vs. TD3, same class &amp; phase — near-identical along the diagonal.
          </figcaption>
        </figure>
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/absolute_dice_by_action_space.png"
            alt="Deployed Dice by action space (discrete vs continuous), by phase"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            Discrete vs. continuous action space — no consistent advantage either way.
          </figcaption>
        </figure>
      </div>
    </section>
  );
};

export default AblationSection;
