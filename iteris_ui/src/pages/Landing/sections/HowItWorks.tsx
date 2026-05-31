/**
 * HowItWorks — Section 3: 4-step process, numbered 01–04.
 */

import React from 'react';

interface Step {
  num: string;
  title: string;
  description: string;
}

const STEPS: Step[] = [
  {
    num: '01',
    title: 'Upload an Image',
    description: 'Drop a DICOM or PNG ultrasound or MRI scan into the workstation.',
  },
  {
    num: '02',
    title: 'Choose a Model',
    description: 'Select from DQN, DDQN, Dueling DQN or DDPG from the model library.',
  },
  {
    num: '03',
    title: 'Run Segmentation',
    description: 'The DRL agent refines boundaries in under a minute with live feedback.',
  },
  {
    num: '04',
    title: 'Explore Results',
    description: 'Compare masks, step through playback, and export Dice / IoU metrics.',
  },
];

/** Section 3 — How it works, 4 numbered steps. */
export const HowItWorks: React.FC = () => (
  <section
    aria-label="How it works"
    className="bg-landing-bg py-24 px-6"
  >
    <div className="mx-auto max-w-6xl flex flex-col gap-14">
      {/* Header */}
      <div className="flex flex-col gap-3">
        <span className="font-mono text-xs tracking-widest uppercase text-grad-a">
          How It Works
        </span>
        <h2 className="font-heading font-bold text-3xl sm:text-4xl text-landing-text max-w-xl">
          From Upload to Insight in Under a Minute.
        </h2>
      </div>

      {/* Steps row */}
      <ol className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8" aria-label="Process steps">
        {STEPS.map((step, idx) => (
          <li key={step.num} className="flex flex-col gap-4 relative">
            {/* Connector dot between steps (not on last) */}
            {idx < STEPS.length - 1 && (
              <div
                className="hidden lg:block absolute top-3 left-full w-full h-px bg-white/[0.08]"
                aria-hidden="true"
              >
                <span className="absolute right-0 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-grad-a/40" />
              </div>
            )}

            {/* Number */}
            <span className="font-mono text-2xl font-bold text-grad-a leading-none">
              {step.num}
            </span>

            {/* Content */}
            <div className="flex flex-col gap-1.5">
              <h3 className="font-heading font-bold text-base text-landing-text">
                {step.title}
              </h3>
              <p className="text-sm leading-relaxed text-landing-text/40">
                {step.description}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  </section>
);

export default HowItWorks;
