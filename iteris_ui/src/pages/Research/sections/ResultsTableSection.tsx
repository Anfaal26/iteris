/**
 * ResultsTableSection — real evaluation output (2026-07-20 run), grouped by
 * phase: Phase A pairs DRL agents against the deployed AttentionResUNet
 * baseline, Phase B against the LiteUNet baseline. Source: master_comparison.csv
 * (spec §5).
 */
import React, { useState } from 'react';
import { EVAL_ROWS, baselineFor, type Phase } from '../data/evaluationResults';

/** Props for ResultsTableSection. */
export interface ResultsTableSectionProps {
  id?: string;
}

const CLASSES: Array<{ dataset: 'CAMUS' | 'BRISC'; className: string; label: string }> = [
  { dataset: 'BRISC', className: 'tumor', label: 'BRISC / tumor' },
  { dataset: 'CAMUS', className: 'LV_endo', label: 'CAMUS / LV endo' },
  { dataset: 'CAMUS', className: 'LV_epi', label: 'CAMUS / LV epi' },
  { dataset: 'CAMUS', className: 'LA', label: 'CAMUS / LA' },
];

const PHASES: Array<{ id: Phase; label: string; baseline: string }> = [
  { id: 'Phase A', label: 'Phase A', baseline: 'Attention U-Net (deployed)' },
  { id: 'Phase B', label: 'Phase B', baseline: 'Lite U-Net' },
];

function deltaClass(delta: number): string {
  if (delta > 0.005) return 'text-success font-semibold';
  if (delta > -0.005) return 'text-warning';
  return 'text-error';
}

function fmtDelta(delta: number): string {
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${(delta * 100).toFixed(1)}pp`;
}

const PhaseTable: React.FC<{ phase: Phase }> = ({ phase }) => (
  <div className="overflow-x-auto rounded-lg border border-border mb-4">
    <table className="w-full text-sm font-body" aria-label={`Results — ${phase}`}>
      <thead>
        <tr className="bg-bg">
          <th rowSpan={2} className="text-left px-4 py-2 font-semibold text-muted border-b border-r border-border align-bottom">
            Class
          </th>
          <th colSpan={2} className="text-center px-4 py-2 font-semibold text-muted border-b border-r border-border">
            Baseline
          </th>
          <th colSpan={3} className="text-center px-4 py-2 font-semibold text-accent border-b border-r border-border">
            DuelingDDQN
          </th>
          <th colSpan={3} className="text-center px-4 py-2 font-semibold text-uncertainty border-b border-border">
            TD3
          </th>
        </tr>
        <tr className="bg-bg">
          {['Dice ↑', 'HD95 ↓'].map((h) => (
            <th key={`base-${h}`} className="text-right px-4 py-1.5 font-medium text-muted border-b border-border text-xs">
              {h}
            </th>
          ))}
          {['Dice ↑', 'HD95 ↓', 'Δ vs baseline'].map((h, i) => (
            <th
              key={`dd-${h}`}
              className={`text-right px-4 py-1.5 font-medium text-muted border-b text-xs ${i === 2 ? 'border-r border-border' : 'border-border'}`}
            >
              {h}
            </th>
          ))}
          {['Dice ↑', 'HD95 ↓', 'Δ vs baseline'].map((h) => (
            <th key={`td3-${h}`} className="text-right px-4 py-1.5 font-medium text-muted border-b border-border text-xs">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {CLASSES.map((c, i) => {
          const baseline = baselineFor(c.dataset, c.className, phase);
          const dueling = EVAL_ROWS.find(
            (r) => r.dataset === c.dataset && r.className === c.className && r.phase === phase && r.model === 'DuelingDDQN',
          );
          const td3 = EVAL_ROWS.find(
            (r) => r.dataset === c.dataset && r.className === c.className && r.phase === phase && r.model === 'TD3',
          );
          if (!baseline || !dueling || !td3) return null;
          const duelingDelta = dueling.dice - baseline.dice;
          const td3Delta = td3.dice - baseline.dice;
          return (
            <tr key={c.label} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
              <td className="px-4 py-2 border-r border-border text-text font-medium">{c.label}</td>
              <td className="px-4 py-2 text-right font-mono text-xs text-muted">{baseline.dice.toFixed(3)}</td>
              <td className="px-4 py-2 text-right font-mono text-xs text-muted border-r border-border">
                {baseline.hd95.toFixed(1)}
              </td>
              <td className="px-4 py-2 text-right font-mono text-xs text-text">{dueling.dice.toFixed(3)}</td>
              <td className="px-4 py-2 text-right font-mono text-xs text-text">{dueling.hd95.toFixed(1)}</td>
              <td className={`px-4 py-2 text-right font-mono text-xs border-r border-border ${deltaClass(duelingDelta)}`}>
                {fmtDelta(duelingDelta)}
              </td>
              <td className="px-4 py-2 text-right font-mono text-xs text-text">{td3.dice.toFixed(3)}</td>
              <td className="px-4 py-2 text-right font-mono text-xs text-text">{td3.hd95.toFixed(1)}</td>
              <td className={`px-4 py-2 text-right font-mono text-xs ${deltaClass(td3Delta)}`}>{fmtDelta(td3Delta)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

/**
 * Real results table, phase-tabbed, plus the aggregate evaluation figures.
 */
export const ResultsTableSection: React.FC<ResultsTableSectionProps> = ({ id = 'results' }) => {
  const [phase, setPhase] = useState<Phase>('Phase A');
  const activePhase = PHASES.find((p) => p.id === phase)!;

  return (
    <section id={id} aria-labelledby="results-heading" className="py-12 scroll-mt-16">
      <h2 id="results-heading" className="font-heading text-xl font-bold text-text mb-2">
        Results
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        Full evaluation run, 2026-07-20. Two phases compare the DRL agents against a different
        U-Net backbone: Phase A against the deployed Attention U-Net (the stronger, low-headroom
        baseline), Phase B against the Lite U-Net (weaker, more headroom). Same agents, same
        contour environment — only the initial contour differs.
      </p>

      {/* Phase tabs */}
      <div role="tablist" aria-label="Result phase" className="flex gap-1 mb-4 border-b border-border">
        {PHASES.map((p) => (
          <button
            key={p.id}
            role="tab"
            aria-selected={phase === p.id}
            type="button"
            onClick={() => setPhase(p.id)}
            className={[
              'px-4 py-2 text-sm font-body font-medium border-b-2 -mb-px transition-colors duration-panel ease-out',
              phase === p.id ? 'border-accent text-accent' : 'border-transparent text-muted hover:text-text',
            ].join(' ')}
          >
            {p.label}
          </button>
        ))}
      </div>
      <p className="text-xs font-body text-muted mb-4">
        Baseline: <span className="text-text">{activePhase.baseline}</span>
      </p>

      <PhaseTable phase={phase} />

      <p className="text-xs font-body text-muted italic mb-10">
        Δ vs baseline in percentage points of Dice. HD95 in millimetres (CAMUS) / pixels (BRISC).
        Paired significance testing (Wilcoxon, Bonferroni-corrected) has not been run on this
        evaluation pass — deltas above are point estimates, not yet tested for significance.
      </p>

      {/* Aggregate figures */}
      <div className="grid gap-6 sm:grid-cols-2">
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/unet_baseline_landscape.png"
            alt="U-Net baselines by dataset/class, split by phase — bar chart"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            Attention U-Net vs. Lite U-Net baseline Dice, every class.
          </figcaption>
        </figure>
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/drl_before_after_dumbbell.png"
            alt="Before (baseline) to after (deployed) Dice, every run, grouped by phase"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            Baseline → deployed Dice for every one of the 16 runs.
          </figcaption>
        </figure>
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/drl_win_tie_loss.png"
            alt="Win/tie/loss per agent, by phase, tie band plus or minus 0.005 Dice"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            Win/tie/loss vs. baseline (±0.005 Dice tie band) — DuelingDDQN ties every Phase B
            run; both agents mostly lose to the strong Phase A baseline.
          </figcaption>
        </figure>
        <figure className="bg-surface border border-border rounded-lg p-3">
          <img
            src="/research/figures/multi_metric_radar.png"
            alt="Multi-metric radar by dataset — Dice, IoU, BIoU, precision, sensitivity"
            className="w-full rounded"
          />
          <figcaption className="text-xs font-body text-muted mt-2">
            All models overlaid across five metrics, by dataset.
          </figcaption>
        </figure>
      </div>
    </section>
  );
};

export default ResultsTableSection;
