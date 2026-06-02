/** ModelPreview — glass model tag cards, DDPG gets BEST badge. */
import React from 'react';
import type { ModelRecord } from '@/api/contract';

export interface ModelPreviewProps { models: ModelRecord[]; }

const FAMILY_LABEL: Record<string, string> = {
  'baseline':      'Baseline',
  'discrete-drl':  'Discrete DRL',
  'continuous-drl':'Continuous DRL',
};

const ModelTag: React.FC<{ model: ModelRecord }> = ({ model }) => {
  const isBest = model.id === 'ddpg';
  const bestDice = model.diceCamus ?? model.diceBrisc;
  return (
    <article className="group relative flex flex-col gap-3 rounded-2xl p-5 transition-all duration-slide ease-out" style={{
      background: isBest ? 'rgba(56,189,248,0.07)' : 'rgba(255,255,255,0.03)',
      border: isBest ? '1px solid rgba(56,189,248,0.22)' : '1px solid rgba(255,255,255,0.07)',
      backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
    }}>
      {isBest && (
        <>
          <div className="absolute inset-x-0 top-0 h-px rounded-t-2xl pointer-events-none" aria-hidden="true"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(56,189,248,0.8), transparent)' }} />
          <span className="absolute -top-2.5 right-4 rounded-full px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider"
            style={{ background: 'var(--color-gradient-b)', color: '#030508' }}>BEST</span>
        </>
      )}
      <h3 className="font-heading font-bold text-[14px] text-landing-text">{model.name}</h3>
      <p className="text-[11px] uppercase tracking-wider" style={{ color: 'rgba(56,189,248,0.6)' }}>
        {FAMILY_LABEL[model.family] ?? model.family}
      </p>
      {bestDice != null && (
        <p className="font-mono text-[13px]" style={{ color: 'var(--color-gradient-b)' }}>
          Dice {bestDice.toFixed(3)}
        </p>
      )}
    </article>
  );
};

export const ModelPreview: React.FC<ModelPreviewProps> = ({ models }) => (
  <section aria-label="Model preview" className="bg-landing-bg py-28 px-6 lg:px-10">
    <div className="mx-auto max-w-6xl flex flex-col gap-12">
      <div className="flex flex-col gap-4">
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase" style={{ color: 'rgba(56,189,248,0.7)' }}>Algorithm Comparison</p>
        <h2 className="font-heading font-bold text-3xl sm:text-4xl lg:text-[3.25rem] text-landing-text leading-tight" style={{ letterSpacing: '-0.02em' }}>
          Three Algorithms.<br />Two Datasets. Complete Evaluation.
        </h2>
        <p className="text-[14px] leading-relaxed max-w-lg" style={{ color: 'rgba(240,249,255,0.45)' }}>
          Every agent benchmarked against U-Net baseline on CAMUS cardiac ultrasound and BRISC brain MRI. Dice, IoU, and Hausdorff distance reported for all.
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {models.map((m) => <ModelTag key={m.id} model={m} />)}
      </div>
      <a href="/models" className="text-[13px] font-body no-underline transition-colors duration-panel ease-out w-fit flex items-center gap-2"
        style={{ color: 'var(--color-gradient-b)' }}>
        Explore all models in the library
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="M2 6h8M6 2l4 4-4 4" />
        </svg>
      </a>
    </div>
  </section>
);
export default ModelPreview;
