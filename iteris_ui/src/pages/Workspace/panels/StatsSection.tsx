/**
 * StatsSection — the expanded results block that sits below the viewer in the
 * single-scroll page (redesign). On a softly lighter surface than the near-black
 * viewer: a per-class Dice / IoU / Hausdorff table with a Δ-vs-baseline column,
 * plus a row of small stat chips for DRL-specific numbers (refinement steps,
 * inference latency). Presentational only — all data comes from the two results.
 */

import React from 'react';
import type { PredictResponse, StructureMetrics } from '@/api/contract';
import { structureColor } from '@/tokens';

export interface StatsSectionProps {
  result: PredictResponse;
  /** Baseline U-Net result for the same image, when available (Δ column). */
  baselineResult?: PredictResponse | null;
}

/** A small labelled stat chip. */
const StatChip: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex flex-col rounded-lg bg-bg border border-border px-3 py-2">
    <span className="text-[10px] font-heading uppercase tracking-wider text-muted">{label}</span>
    <span className="text-sm font-mono text-text mt-0.5">{value}</span>
  </div>
);

/** Signed, colour-coded delta cell (green = better, muted = flat, red = worse). */
function DeltaCell({ value, invert = false }: { value: number | null; invert?: boolean }) {
  if (value === null) return <span className="text-muted/50">—</span>;
  const better = invert ? value < 0 : value > 0;
  const flat = Math.abs(value) < (invert ? 0.05 : 0.002);
  const cls = flat ? 'text-muted' : better ? 'text-success' : 'text-error';
  const sign = value > 0 ? '+' : '';
  return <span className={`font-mono ${cls}`}>{sign}{value.toFixed(invert ? 2 : 3)}</span>;
}

export const StatsSection: React.FC<StatsSectionProps> = ({ result, baselineResult }) => {
  const { metrics, refinementSteps, inferenceMs, modelId } = result;
  const baseByStructure = new Map<string, StructureMetrics>(
    (baselineResult?.metrics.structures ?? []).map((s) => [s.structure, s]),
  );
  const isDrl = refinementSteps !== undefined;
  const hasGt = metrics.dice > 0; // real metrics only when a GT mask was attached

  return (
    <section
      aria-label="Detailed results"
      className="px-6 py-6 bg-surface"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider">
          Detailed results
        </h2>
        <span className="text-[11px] font-mono text-muted/70">{modelId}</span>
      </div>

      {/* DRL / timing stat chips */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-5">
        <StatChip label="Overall Dice" value={hasGt ? metrics.dice.toFixed(3) : 'n/a'} />
        <StatChip label="Overall IoU" value={hasGt ? metrics.iou.toFixed(3) : 'n/a'} />
        {isDrl && <StatChip label="Refinement steps" value={String(refinementSteps)} />}
        <StatChip label="Inference" value={`${Math.round(inferenceMs)} ms`} />
      </div>

      {/* Per-class table */}
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm font-body">
          <thead>
            <tr className="text-[11px] font-heading uppercase tracking-wider text-muted border-b border-border">
              <th className="text-left font-medium px-3 py-2">Structure</th>
              <th className="text-right font-medium px-3 py-2">Dice</th>
              <th className="text-right font-medium px-3 py-2">IoU</th>
              <th className="text-right font-medium px-3 py-2">Hausdorff</th>
              <th className="text-right font-medium px-3 py-2">Δ Dice vs U-Net</th>
            </tr>
          </thead>
          <tbody>
            {metrics.structures.map((s) => {
              const base = baseByStructure.get(s.structure);
              const delta = base && hasGt ? s.dice - base.dice : null;
              return (
                <tr key={s.structure} className="border-b border-border/60 last:border-0">
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                        style={{ backgroundColor: structureColor[s.structure] ?? 'var(--muted)' }}
                        aria-hidden="true"
                      />
                      <span className="text-text">{s.label}</span>
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-text">{hasGt ? s.dice.toFixed(3) : '—'}</td>
                  <td className="px-3 py-2 text-right font-mono text-text">{hasGt ? s.iou.toFixed(3) : '—'}</td>
                  <td className="px-3 py-2 text-right font-mono text-text">{hasGt ? s.hd.toFixed(2) : '—'}</td>
                  <td className="px-3 py-2 text-right"><DeltaCell value={delta} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {!hasGt && (
        <p className="text-[11px] font-body text-muted mt-3">
          Attach a ground-truth mask in the sidebar to compute Dice / IoU / Hausdorff for this scan.
        </p>
      )}
    </section>
  );
};

export default StatsSection;
