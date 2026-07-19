/**
 * ResultsTableSection — results table sourced from the live model registry.
 * No fabricated numbers: metric cells render "—" and the section is marked
 * pending until real evaluation output exists (spec §5).
 */
import React from 'react';
import type { ModelRecord } from '@/api/contract';
import modelsRaw from '@/content/models.yaml';

/** Props for ResultsTableSection. */
export interface ResultsTableSectionProps {
  id?: string;
}

const models = modelsRaw as ModelRecord[];

function fmt(value: number | null, digits = 3): string {
  return value == null ? '—' : value.toFixed(digits);
}

/**
 * Results table — reads the same registry as /models, so it can only ever
 * report real numbers, never placeholders that look like data.
 */
export const ResultsTableSection: React.FC<ResultsTableSectionProps> = ({
  id = 'results',
}) => {
  const evaluatedCount = models.filter((m) => m.diceCamus != null || m.diceBrisc != null).length;

  return (
    <section id={id} aria-labelledby="results-heading" className="py-12 scroll-mt-16">
      <div className="flex items-center gap-3 mb-6">
        <h2 id="results-heading" className="font-heading text-xl font-bold text-text">
          Results
        </h2>
        <span className="px-2 py-0.5 rounded-full border border-border text-xs font-body text-muted">
          {evaluatedCount} / {models.length} evaluated
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border mb-2">
        <table className="w-full text-sm font-body" aria-label="Model evaluation results">
          <thead>
            <tr className="bg-bg">
              <th className="text-left px-4 py-2 font-semibold text-muted border-b border-r border-border">
                Model
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border text-xs">
                CAMUS Dice ↑
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border text-xs">
                BRISC Dice ↑
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border text-xs">
                IoU ↑
              </th>
              <th className="text-right px-4 py-2 font-semibold text-muted border-b border-border text-xs">
                HD95 ↓
              </th>
            </tr>
          </thead>
          <tbody>
            {models.map((row, i) => (
              <tr key={row.id} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                <td
                  className={`px-4 py-2 border-r border-border ${
                    row.deployed ? 'text-text font-medium' : 'text-muted italic'
                  }`}
                >
                  {row.name}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-muted">
                  {fmt(row.diceCamus)}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-muted">
                  {fmt(row.diceBrisc)}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-muted">
                  {fmt(row.iou)}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-muted">
                  {fmt(row.hd, 1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs font-body text-muted italic">
        Dueling DDQN and TD3 are currently training against the Lite U-Net baseline; the
        Attention U-Net is deployed as the reference checkpoint. This table reads directly
        from the model registry, so it will populate automatically as runs complete —
        statistical significance (paired Wilcoxon, Bonferroni-corrected) will be reported
        alongside once there is evaluation output to test.
      </p>
    </section>
  );
};

export default ResultsTableSection;
