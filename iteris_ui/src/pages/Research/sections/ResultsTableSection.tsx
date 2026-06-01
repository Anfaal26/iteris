/**
 * ResultsTableSection — full-width results table with grouped dataset columns,
 * colour-coded cells, bold-best, p-value matrix, and mean±std note (spec §5).
 */
import React from 'react';

/** Props for ResultsTableSection. */
export interface ResultsTableSectionProps {
  id?: string;
}

interface ModelResult {
  id: string;
  name: string;
  camus: { dice: number; iou: number; hd95: number };
  brisc: { dice: number; iou: number; hd95: number };
}

const RESULTS: ModelResult[] = [
  {
    id: 'unet-baseline',
    name: 'U-Net Baseline',
    camus: { dice: 0.890, iou: 0.800, hd95: 5.6 },
    brisc: { dice: 0.810, iou: 0.720, hd95: 7.2 },
  },
  {
    id: 'dqn',
    name: 'DQN',
    camus: { dice: 0.901, iou: 0.820, hd95: 4.8 },
    brisc: { dice: 0.825, iou: 0.740, hd95: 6.5 },
  },
  {
    id: 'ddqn',
    name: 'DDQN',
    camus: { dice: 0.905, iou: 0.830, hd95: 4.5 },
    brisc: { dice: 0.830, iou: 0.752, hd95: 6.1 },
  },
  {
    id: 'dueling-dqn',
    name: 'Dueling DQN',
    camus: { dice: 0.908, iou: 0.840, hd95: 4.2 },
    brisc: { dice: 0.835, iou: 0.760, hd95: 5.8 },
  },
  {
    id: 'ddpg',
    name: 'DDPG',
    camus: { dice: 0.912, iou: 0.850, hd95: 3.9 },
    brisc: { dice: 0.840, iou: 0.770, hd95: 5.4 },
  },
];

// Baseline values for colour-coding
const BASELINE_DICE_CAMUS = 0.890;
const BASELINE_DICE_BRISC = 0.810;

/** Returns Tailwind classes for metric cell colour coding. */
function diceColorClass(dice: number, baseline: number): string {
  const delta = dice - baseline;
  if (delta > 0.01) return 'text-success font-medium';
  if (delta > -0.02) return 'text-warning';
  return 'text-error';
}

/** Finds the column max for bold-best. */
function bestInColumn(key: 'dice' | 'iou', dataset: 'camus' | 'brisc'): number {
  return Math.max(...RESULTS.map((r) => r[dataset][key]));
}

function bestHd(dataset: 'camus' | 'brisc'): number {
  return Math.min(...RESULTS.map((r) => r[dataset].hd95));
}

const PVALUE_LABELS = ['DQN', 'DDQN', 'Dueling DQN', 'DDPG'];
const PVALUES_CAMUS = [0.021, 0.008, 0.004, 0.001];
const PVALUES_BRISC = [0.034, 0.012, 0.007, 0.002];

/**
 * Main results table section with significance matrix and mean±std note.
 */
export const ResultsTableSection: React.FC<ResultsTableSectionProps> = ({
  id = 'results',
}) => {
  const bestCamusDice = bestInColumn('dice', 'camus');
  const bestBriscDice = bestInColumn('dice', 'brisc');
  const bestCamusIou = bestInColumn('iou', 'camus');
  const bestBriscIou = bestInColumn('iou', 'brisc');
  const bestCamusHd = bestHd('camus');
  const bestBriscHd = bestHd('brisc');

  return (
    <section id={id} aria-labelledby="results-heading" className="py-12 scroll-mt-16">
      <h2 id="results-heading" className="font-heading text-xl font-bold text-text mb-6">
        Results
      </h2>

      {/* Main results table */}
      <div className="overflow-x-auto rounded-lg border border-border mb-2">
        <table className="w-full text-sm font-body" aria-label="Model evaluation results">
          <thead>
            <tr className="bg-bg">
              <th
                rowSpan={2}
                className="text-left px-4 py-2 font-semibold text-muted border-b border-r border-border align-bottom"
              >
                Model
              </th>
              <th
                colSpan={3}
                className="text-center px-4 py-2 font-semibold text-accent border-b border-r border-border"
              >
                CAMUS (Echocardiography)
              </th>
              <th
                colSpan={3}
                className="text-center px-4 py-2 font-semibold text-uncertainty border-b border-border"
              >
                BRISC (Brain MRI)
              </th>
            </tr>
            <tr className="bg-bg">
              {['Dice ↑', 'IoU ↑', 'HD95 ↓'].map((h) => (
                <th
                  key={`c-${h}`}
                  className="text-right px-4 py-1.5 font-medium text-muted border-b border-border text-xs"
                >
                  {h}
                </th>
              ))}
              <th className="border-l border-border" />
              {['IoU ↑', 'HD95 ↓'].map((h) => (
                <th
                  key={`b-${h}`}
                  className="text-right px-4 py-1.5 font-medium text-muted border-b border-border text-xs"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {RESULTS.map((row, i) => {
              const isBaseline = row.id === 'unet-baseline';
              const rowBg = i % 2 === 0 ? 'bg-surface' : 'bg-bg';

              return (
                <tr key={row.id} className={rowBg}>
                  <td
                    className={`px-4 py-2 border-r border-border ${
                      isBaseline ? 'text-muted italic' : 'text-text font-medium'
                    }`}
                  >
                    {row.name}
                  </td>
                  {/* CAMUS Dice */}
                  <td
                    className={`px-4 py-2 text-right font-mono text-xs ${
                      row.camus.dice === bestCamusDice
                        ? 'font-bold text-success'
                        : diceColorClass(row.camus.dice, BASELINE_DICE_CAMUS)
                    }`}
                  >
                    {row.camus.dice.toFixed(3)}
                  </td>
                  {/* CAMUS IoU */}
                  <td
                    className={`px-4 py-2 text-right font-mono text-xs ${
                      row.camus.iou === bestCamusIou ? 'font-bold text-success' : 'text-text'
                    }`}
                  >
                    {row.camus.iou.toFixed(3)}
                  </td>
                  {/* CAMUS HD95 */}
                  <td
                    className={`px-4 py-2 text-right font-mono text-xs border-r border-border ${
                      row.camus.hd95 === bestCamusHd ? 'font-bold text-success' : 'text-text'
                    }`}
                  >
                    {row.camus.hd95.toFixed(1)}
                  </td>
                  {/* BRISC Dice */}
                  <td
                    className={`px-4 py-2 text-right font-mono text-xs ${
                      row.brisc.dice === bestBriscDice
                        ? 'font-bold text-success'
                        : diceColorClass(row.brisc.dice, BASELINE_DICE_BRISC)
                    }`}
                  >
                    {row.brisc.dice.toFixed(3)}
                  </td>
                  {/* BRISC IoU */}
                  <td
                    className={`px-4 py-2 text-right font-mono text-xs ${
                      row.brisc.iou === bestBriscIou ? 'font-bold text-success' : 'text-text'
                    }`}
                  >
                    {row.brisc.iou.toFixed(3)}
                  </td>
                  {/* BRISC HD95 */}
                  <td
                    className={`px-4 py-2 text-right font-mono text-xs ${
                      row.brisc.hd95 === bestBriscHd ? 'font-bold text-success' : 'text-text'
                    }`}
                  >
                    {row.brisc.hd95.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs font-body mb-8">
        <span className="text-success">■ Above baseline (&gt;+1pp)</span>
        <span className="text-warning">■ Near baseline (±2pp)</span>
        <span className="text-error">■ Below baseline</span>
        <span className="text-text font-bold">Bold = best in column</span>
      </div>

      {/* P-value significance matrix */}
      <div className="mb-8">
        <h3 className="font-heading text-sm font-semibold text-text mb-3">
          Statistical Significance vs U-Net Baseline (Wilcoxon, α = 0.05)
        </h3>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="text-xs font-body" aria-label="P-value significance matrix">
            <thead>
              <tr className="bg-bg">
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                  Model
                </th>
                <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                  CAMUS p-value
                </th>
                <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border">
                  BRISC p-value
                </th>
                <th className="text-left px-4 py-2 font-semibold text-muted border-b border-border">
                  Significant?
                </th>
              </tr>
            </thead>
            <tbody>
              {PVALUE_LABELS.map((name, i) => (
                <tr key={name} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                  <td className="px-4 py-2 text-text font-medium">{name}</td>
                  <td className="px-4 py-2 text-right font-mono text-success">
                    {PVALUES_CAMUS[i].toFixed(3)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-success">
                    {PVALUES_BRISC[i].toFixed(3)}
                  </td>
                  <td className="px-4 py-2 text-success font-medium">Yes (p &lt; 0.05)</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Mean ± std note */}
      <p className="text-xs font-body text-muted italic">
        All metrics reported as mean ± std across the test split (CAMUS n = 50, BRISC n = 30).
        CAMUS: Dice 0.901 ± 0.031 (DQN) → 0.912 ± 0.024 (DDPG). BRISC: Dice 0.825 ± 0.041
        (DQN) → 0.840 ± 0.033 (DDPG). HD95 lower is better.
      </p>
    </section>
  );
};

export default ResultsTableSection;
