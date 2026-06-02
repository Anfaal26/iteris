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
          "A key novelty of this work is the real-time iteration playback interface:
          rather than presenting only a final segmentation mask, ITERIS exposes the
          full 20-step DRL boundary-refinement trajectory — enabling researchers to
          inspect precisely where and how the agent improves upon the initial U-Net prediction."
        </blockquote>
        <figcaption className="relative mt-7 text-[13px]" style={{ color: 'rgba(240,249,255,0.38)' }}>
          — PRJ63504 Capstone · Taylor's University · 2024
        </figcaption>
      </figure>
    </div>
  </section>
);
export default ResearchContext;
