/**
 * ResearchContext — Section 6: pull-quote about iteration playback novelty.
 * Centred quote block with teal left border and light bg wash.
 */

import React from 'react';

/** Section 6 — Research context pull quote. */
export const ResearchContext: React.FC = () => (
  <section
    aria-label="Research context"
    className="bg-landing-bg py-24 px-6"
  >
    <div className="mx-auto max-w-3xl">
      <figure
        className="rounded-xl bg-white/[0.03] border border-white/[0.06] pl-8 pr-8 py-10
                   border-l-4"
        style={{ borderLeftColor: 'var(--color-gradient-a)' }}
      >
        <blockquote className="font-heading text-xl sm:text-2xl font-medium text-landing-text leading-snug">
          "A key novelty of this work is the real-time iteration playback interface:
          rather than presenting only a final segmentation mask, ITERIS exposes the
          full 20-step DRL boundary-refinement trajectory, enabling researchers to
          inspect precisely where and how the agent improves upon the initial U-Net
          prediction."
        </blockquote>
        <figcaption className="mt-6 text-sm text-landing-text/40">
          — PRJ63504 Capstone Project, Taylor's University, 2024
        </figcaption>
      </figure>
    </div>
  </section>
);

export default ResearchContext;
