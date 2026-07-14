/** ResearchContext — glass pull-quote with ice-blue left border and inner ambient glow. */
import React from 'react';

export const ResearchContext: React.FC = () => (
  <section aria-label="Research context" className="bg-landing-bg py-24 px-6 lg:px-10">
    <div className="mx-auto max-w-3xl">
      <figure className="rounded-2xl p-10 relative overflow-hidden" style={{
        background: 'rgba(56,189,248,0.04)',
        border: '1px solid rgba(56,189,248,0.12)',
        borderLeft: '3px solid rgba(56,189,248,0.6)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
      }}>
        <div className="absolute inset-0 pointer-events-none" aria-hidden="true"
          style={{ background: 'radial-gradient(ellipse at 0% 50%, rgba(56,189,248,0.06) 0%, transparent 60%)' }} />
        <blockquote className="relative font-heading text-xl sm:text-2xl font-medium leading-snug text-landing-text">
          "Classical segmentation commits to a boundary in a single forward pass.
          This work asks what happens when the model can look again — a reinforcement
          learning agent that revises the initial U-Net contour step by step, learning
          through interaction rather than through imitation."
        </blockquote>
        <figcaption className="relative mt-7 text-[13px]" style={{ color: 'rgba(240,249,255,0.38)' }}>
          — PRJ63504 Capstone · Taylor's University · 2025
        </figcaption>
      </figure>
    </div>
  </section>
);
export default ResearchContext;
