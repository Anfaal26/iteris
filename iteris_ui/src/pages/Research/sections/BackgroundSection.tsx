/**
 * BackgroundSection — concise related-work context: why contour refinement
 * is framed as sequential decision-making, plus a compact architecture +
 * formula summary for the two agents used on this page (Dueling DDQN, TD3).
 * Diagrams are HTML/CSS box-and-arrow, not raster images, so they stay
 * crisp and theme-aware. (spec §5 extension)
 */
import React from 'react';

/** Props for BackgroundSection. */
export interface BackgroundSectionProps {
  id?: string;
}

const Box: React.FC<{ children: React.ReactNode; accent?: boolean; small?: boolean }> = ({
  children,
  accent,
  small,
}) => (
  <div
    className={[
      'rounded-lg border px-3 text-center font-body leading-snug shrink-0',
      small ? 'py-1.5 text-xs' : 'py-2.5 text-sm',
      accent ? 'border-accent/40 bg-accent/10 text-accent font-medium' : 'border-border bg-surface text-text',
    ].join(' ')}
  >
    {children}
  </div>
);

const Arrow: React.FC = () => (
  <span className="text-muted text-lg shrink-0" aria-hidden="true">
    →
  </span>
);

interface Ref {
  label: string;
  href: string;
}

const REFS: Ref[] = [
  { label: 'Wang et al., 2016 — Dueling Network Architectures for DRL', href: 'https://arxiv.org/abs/1511.06581' },
  { label: 'Fujimoto et al., 2018 — Addressing Function Approximation Error in Actor-Critic Methods (TD3)', href: 'https://arxiv.org/abs/1802.09477' },
  { label: 'Dong et al. — Left Ventricle Contouring in Cardiac Images via DRL', href: 'https://arxiv.org/abs/2106.04127' },
  { label: 'MARL-MambaContour, 2026 — Multi-Agent DRL for Active Contour Optimisation', href: 'https://arxiv.org/abs/2506.18679' },
  { label: 'AI in Radiology: 2025 Trends & FDA Approvals — IntuitionLabs', href: 'https://intuitionlabs.ai/articles/ai-radiology-trends-2025' },
];

/**
 * Background/related-work: framing, the two agent architectures with core
 * formulas and diagrams, and a one-line note on why this matters clinically.
 */
export const BackgroundSection: React.FC<BackgroundSectionProps> = ({ id = 'background' }) => {
  return (
    <section id={id} aria-labelledby="background-heading" className="py-12 scroll-mt-16">
      <h2 id="background-heading" className="font-heading text-xl font-bold text-text mb-4">
        Background
      </h2>
      <p className="text-sm font-body text-muted leading-relaxed mb-8">
        Framing segmentation refinement as sequential decision-making rather than a single
        forward pass has direct precedent: DRL contour agents have been applied to left-ventricle
        delineation in cardiac imaging, and multi-agent DRL contour optimisation is active work
        as of 2026. This project narrows that idea to two agents on the same environment,
        differing only in action space.
      </p>

      {/* Dueling DDQN */}
      <div className="mb-10">
        <h3 className="font-heading text-base font-semibold text-text mb-2">
          Dueling DDQN <span className="text-muted font-normal">— discrete</span>
        </h3>
        <p className="text-sm font-body text-muted leading-relaxed mb-3">
          Splits the Q-network into a scalar state-value stream <em>V(s)</em> and a per-action
          advantage stream <em>A(s,a)</em>, recombined so the two are individually identifiable:
        </p>
        <p className="font-mono text-sm text-accent bg-surface border border-border rounded-lg px-4 py-2 mb-4 overflow-x-auto">
          Q(s,a) = V(s) + ( A(s,a) − (1/|A|)&nbsp;Σ<sub>a′</sub> A(s,a′) )
        </p>
        <p className="text-sm font-body text-muted leading-relaxed mb-4">
          The benefit for a sector-based action space: <em>V(s)</em> updates from every step
          regardless of which sector was pushed, so the agent learns "is this contour good"
          separately from "which sector should move" — useful when many sectors are irrelevant
          on any given step.
        </p>
        <div className="flex items-center gap-2 flex-wrap overflow-x-auto">
          <Box small>State s</Box>
          <Arrow />
          <Box small>Shared conv/FC encoder</Box>
          <Arrow />
          <div className="flex flex-col gap-1.5">
            <Box small>Value V(s)</Box>
            <Box small>Advantage A(s,a)</Box>
          </div>
          <Arrow />
          <Box small>Combine</Box>
          <Arrow />
          <Box small accent>Q(s,a) per sector</Box>
        </div>
      </div>

      {/* TD3 */}
      <div className="mb-8">
        <h3 className="font-heading text-base font-semibold text-text mb-2">
          TD3 <span className="text-muted font-normal">— continuous</span>
        </h3>
        <p className="text-sm font-body text-muted leading-relaxed mb-3">
          Twin Delayed DDPG fixes DDPG's Q-value overestimation with three changes: two critics
          (take the minimum), a policy updated less often than the critics, and noise injected
          into the target action:
        </p>
        <p className="font-mono text-sm text-accent bg-surface border border-border rounded-lg px-4 py-2 mb-4 overflow-x-auto">
          y = r + γ·min(Q₁′(s′, ã), Q₂′(s′, ã)),&nbsp; ã = π′(s′) + ε,&nbsp; ε ~ clip(N(0,σ), −c, c)
        </p>
        <div className="flex items-center gap-2 flex-wrap overflow-x-auto mb-3">
          <Box small>State s</Box>
          <Arrow />
          <Box small accent>Actor π(s)</Box>
          <Arrow />
          <Box small>Action a (+ clipped noise)</Box>
          <Arrow />
          <div className="flex flex-col gap-1.5">
            <Box small>Critic Q₁(s,a)</Box>
            <Box small>Critic Q₂(s,a)</Box>
          </div>
          <Arrow />
          <Box small accent>min(Q₁,Q₂) → target</Box>
        </div>
        <p className="text-xs font-body text-muted">
          Actor and target networks update every <em>d</em> critic steps (delayed policy update);
          targets track the online networks by Polyak averaging (rate τ).
        </p>
      </div>

      <p className="text-sm font-body text-muted leading-relaxed mb-6">
        Radiology alone has over a thousand FDA-authorised AI-enabled devices as of late 2025 —
        the majority of all AI medical device clearances — which is exactly why this page
        reports pending/negative results plainly rather than only the favourable ones: a
        capstone prototype is not a clinical tool, but the reporting habits should still hold.
      </p>

      <ul className="space-y-1 list-none pl-0">
        {REFS.map((r) => (
          <li key={r.href} className="text-xs font-body text-muted">
            <a href={r.href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
              {r.label} ↗
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default BackgroundSection;
