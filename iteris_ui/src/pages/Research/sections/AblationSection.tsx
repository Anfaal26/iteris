/**
 * AblationSection — the three-cap diagnosis for "agent doesn't beat baseline"
 * (representation / information / optimization), pending the headroom_report
 * oracle-ceiling numbers that will quantify each row (spec §5).
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
    status: 'Diagnosed — fix: raise spline point count / near-zero smoothing, or edit residual-on-logits so do-nothing reproduces the U-Net mask exactly.',
  },
  {
    cap: 'Information cap',
    question:
      'A GT-blind refiner sees the same image with less capacity than the U-Net and no privileged channels — on ambiguous regions the U-Net\'s residual error may simply not be resolvable without a learned shape prior.',
    status: 'Diagnosed — the fixed smoothing spline is a uniform, not selective, prior; a learned one is the candidate fix.',
  },
  {
    cap: 'Optimization / STOP-timing cap',
    question:
      'PBRS, TD3+BC, hard-example mining, curriculum, and an HD-composite reward only help the agent reach its ceiling faster — they cannot raise the ceiling itself. TD3 also has no learned STOP action, so a deployed policy can drift past its best checkpoint.',
    status: 'Decisive test: headroom_report oracle-contour-ceiling vs. baseline, plus a forced-STOP-at-t=0 evaluation to isolate drift.',
  },
];

/**
 * Qualitative breakdown of the three ceilings on agent performance, pending
 * the quantitative headroom_report numbers that will fill this table in.
 */
export const AblationSection: React.FC<AblationSectionProps> = ({ id = 'ablations' }) => {
  return (
    <section id={id} aria-labelledby="ablations-heading" className="py-12 scroll-mt-16">
      <div className="flex items-center gap-3 mb-4">
        <h2 id="ablations-heading" className="font-heading text-xl font-bold text-text">
          Ablation Study
        </h2>
        <span className="px-2 py-0.5 rounded-full border border-border text-xs font-body text-muted">
          ○ Quantitative pass pending
        </span>
      </div>
      <p className="text-sm font-body text-muted mb-6">
        When a refinement agent fails to beat the U-Net baseline, three distinct causes get
        conflated. The quantitative ablation below is pending the{' '}
        <code className="text-xs font-mono text-accent">headroom_report</code> oracle-ceiling
        run for each cap; the diagnosis itself is already established.
      </p>

      <div className="overflow-x-auto rounded-lg border border-border">
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
                Status
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
    </section>
  );
};

export default AblationSection;
