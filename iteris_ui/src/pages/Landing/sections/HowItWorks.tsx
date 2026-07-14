/** HowItWorks — 4-step numbered sequence in glass cards. */
import React from 'react';

const STEPS = [
  { num: '01', title: 'Upload a Scan',      description: 'Drop a DICOM, NIfTI, or PNG study — or pick from the curated CAMUS/BRISC sample library.' },
  { num: '02', title: 'Choose a Model',     description: 'Select Dueling DQN (discrete) or TD3 (continuous) from the model library, or benchmark against the Attention U-Net baseline.' },
  { num: '03', title: 'Run Segmentation',   description: 'Preprocessing, normalisation, and DRL boundary refinement complete in seconds with live feedback.' },
  { num: '04', title: 'Explore Results',    description: 'Step through iteration playback, wipe-compare masks, export Dice and Hausdorff as JSON.' },
];

export const HowItWorks: React.FC = () => (
  <section aria-label="How it works" className="bg-landing-bg py-28 px-6 lg:px-10" style={{
    backgroundImage: 'linear-gradient(var(--scan-grid) 1px, transparent 1px), linear-gradient(90deg, var(--scan-grid) 1px, transparent 1px)',
    backgroundSize: '48px 48px',
  }}>
    <div className="mx-auto max-w-6xl flex flex-col gap-16">
      <div className="flex flex-col gap-4 max-w-xl">
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase" style={{ color: 'rgba(56,189,248,0.7)' }}>How It Works</p>
        <h2 className="font-heading font-bold text-3xl sm:text-4xl lg:text-[3.25rem] text-landing-text leading-tight" style={{ letterSpacing: '-0.02em' }}>
          From Upload to Insight<br />in Under a Minute.
        </h2>
      </div>
      <ol className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5" aria-label="Process steps">
        {STEPS.map((step, idx) => (
          <li key={step.num} className="relative flex flex-col gap-5 rounded-2xl p-6" style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.07)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
          }}>
            {idx < STEPS.length - 1 && (
              <div className="hidden lg:block absolute top-10 left-full w-5 pointer-events-none" aria-hidden="true"
                style={{ height: '1px', background: 'linear-gradient(90deg, rgba(56,189,248,0.25), transparent)' }} />
            )}
            <span className="font-mono font-bold text-4xl leading-none bg-iteris-gradient bg-clip-text text-transparent"
              style={{ WebkitBackgroundClip: 'text' }} aria-label={`Step ${step.num}`}>{step.num}</span>
            <div className="flex flex-col gap-2">
              <h3 className="font-heading font-bold text-[15px] text-landing-text">{step.title}</h3>
              <p className="text-[13px] leading-relaxed" style={{ color: 'rgba(240,249,255,0.42)' }}>{step.description}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  </section>
);
export default HowItWorks;
