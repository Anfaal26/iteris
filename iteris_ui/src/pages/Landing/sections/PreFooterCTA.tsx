/**
 * PreFooterCTA — Section 7: full-width dark CTA section.
 */

import React from 'react';

/** Section 7 — Pre-footer call to action. */
export const PreFooterCTA: React.FC = () => (
  <section
    aria-label="Call to action"
    className="bg-landing-footer py-24 px-6 text-center"
  >
    <div className="mx-auto max-w-2xl flex flex-col items-center gap-8">
      <h2 className="font-heading font-bold text-4xl sm:text-5xl text-landing-text leading-tight">
        Ready to explore?
      </h2>

      <div className="flex flex-col sm:flex-row items-center gap-4">
        <a
          href="/workspace"
          className="rounded-full bg-grad-a px-7 py-3 font-heading font-semibold text-landing-bg text-sm
                     hover:opacity-90 transition-opacity duration-panel ease-out"
        >
          Try Iteris →
        </a>
        <a
          href="/research"
          className="rounded-full border border-white/20 px-7 py-3 font-heading font-semibold text-landing-text text-sm
                     hover:border-white/40 hover:bg-white/[0.04] transition-colors duration-panel ease-out"
        >
          View Research
        </a>
      </div>

      <p className="text-xs text-landing-text/25 max-w-sm leading-relaxed">
        ITERIS is a research prototype developed as part of PRJ63504 at Taylor's University.
        All segmentation results are for research purposes only and must not be used for
        clinical decision-making.
      </p>
    </div>
  </section>
);

export default PreFooterCTA;
