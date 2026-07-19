/**
 * ConvergenceSection — CAMUS / BRISC convergence curves. Pending state until
 * real training logs exist; no synthetic curves (spec §5).
 */
import React from 'react';

/** Props for ConvergenceSection. */
export interface ConvergenceSectionProps {
  id?: string;
}

const PendingCard: React.FC<{ title: string }> = ({ title }) => (
  <div className="bg-surface border border-border rounded-lg p-4 flex flex-col items-center justify-center text-center gap-2 min-h-[220px]">
    <h3 className="font-heading text-sm font-semibold text-text">{title}</h3>
    <span className="px-2 py-0.5 rounded-full border border-border text-xs font-body text-muted">
      ○ Training in progress
    </span>
    <p className="text-xs font-body text-muted max-w-[28ch]">
      Mean episode Dice vs. training steps will render here once logged runs are available.
    </p>
  </div>
);

/**
 * Side-by-side convergence curves for CAMUS and BRISC — pending until logs exist.
 */
export const ConvergenceSection: React.FC<ConvergenceSectionProps> = ({
  id = 'convergence',
}) => {
  return (
    <section id={id} aria-labelledby="convergence-heading" className="py-12 scroll-mt-16">
      <h2 id="convergence-heading" className="font-heading text-xl font-bold text-text mb-4">
        Convergence Curves
      </h2>
      <p className="text-sm font-body text-muted mb-6">
        Mean episode Dice vs. training steps for Dueling DDQN and TD3 on the contour
        environment, evaluated against the frozen Lite U-Net initial contour.
      </p>
      <div className="grid gap-6 lg:grid-cols-2">
        <PendingCard title="CAMUS" />
        <PendingCard title="BRISC" />
      </div>
    </section>
  );
};

export default ConvergenceSection;
