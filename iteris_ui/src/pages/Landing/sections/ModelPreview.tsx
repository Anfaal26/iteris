/** ModelPreview — model architecture cards with family colour coding. */
import React from 'react';
import type { ModelRecord } from '@/api/contract';

export interface ModelPreviewProps { models: ModelRecord[]; }

const FAMILY_CFG: Record<string, {
  bg: string; border: string; accentBar: string;
  chipBg: string; chipBorder: string; chipColor: string;
  glow: string; label: string;
}> = {
  'baseline': {
    bg: 'rgba(148,163,184,0.04)',
    border: 'rgba(148,163,184,0.15)',
    accentBar: 'linear-gradient(90deg, transparent, rgba(148,163,184,0.45), transparent)',
    chipBg: 'rgba(148,163,184,0.09)',
    chipBorder: 'rgba(148,163,184,0.2)',
    chipColor: 'rgba(203,213,225,0.85)',
    glow: 'rgba(148,163,184,0.06)',
    label: 'Baseline',
  },
  'discrete-drl': {
    bg: 'rgba(56,189,248,0.05)',
    border: 'rgba(56,189,248,0.16)',
    accentBar: 'linear-gradient(90deg, transparent, rgba(56,189,248,0.65), transparent)',
    chipBg: 'rgba(56,189,248,0.09)',
    chipBorder: 'rgba(56,189,248,0.22)',
    chipColor: 'rgba(56,189,248,0.9)',
    glow: 'rgba(56,189,248,0.07)',
    label: 'Discrete',
  },
  'continuous-drl': {
    bg: 'rgba(167,139,250,0.05)',
    border: 'rgba(167,139,250,0.16)',
    accentBar: 'linear-gradient(90deg, transparent, rgba(167,139,250,0.65), transparent)',
    chipBg: 'rgba(167,139,250,0.09)',
    chipBorder: 'rgba(167,139,250,0.22)',
    chipColor: 'rgba(167,139,250,0.9)',
    glow: 'rgba(167,139,250,0.07)',
    label: 'Continuous',
  },
};

const ModelTag: React.FC<{ model: ModelRecord }> = ({ model }) => {
  const cfg = FAMILY_CFG[model.family] ?? FAMILY_CFG['baseline'];
  return (
    <article
      className="group relative flex flex-col gap-0 rounded-2xl overflow-hidden transition-all duration-[250ms] ease-out"
      style={{
        background: cfg.bg,
        border: `1px solid ${cfg.border}`,
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
      }}
    >
      {/* Top accent line */}
      <div
        className="absolute top-0 inset-x-0 h-px pointer-events-none"
        aria-hidden="true"
        style={{ background: cfg.accentBar }}
      />
      {/* Hover radial glow from top */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-[250ms] pointer-events-none"
        style={{ background: `radial-gradient(ellipse at 50% -5%, ${cfg.glow} 0%, transparent 55%)` }}
      />

      <div className="relative flex flex-col gap-3 px-5 py-5">
        {/* Family chip */}
        <span
          className="self-start font-mono text-[10px] uppercase tracking-[0.12em] px-2.5 py-1 rounded-full"
          style={{
            background: cfg.chipBg,
            border: `1px solid ${cfg.chipBorder}`,
            color: cfg.chipColor,
          }}
        >
          {cfg.label}
        </span>

        {/* Name */}
        <h3 className="font-heading font-bold text-[15px] leading-snug text-landing-text">
          {model.name}
        </h3>

        {/* Deployment status */}
        <p className="font-mono text-[11px]" style={{ color: 'rgba(240,249,255,0.28)' }}>
          {model.deployed ? '● Deployed' : '○ Evaluation pending'}
        </p>
      </div>
    </article>
  );
};

export const ModelPreview: React.FC<ModelPreviewProps> = ({ models }) => (
  <section aria-label="Model preview" className="bg-landing-bg py-28 px-6 lg:px-10">
    <div className="mx-auto max-w-6xl flex flex-col gap-12">
      <div className="flex flex-col gap-4 max-w-xl">
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase" style={{ color: 'rgba(56,189,248,0.7)' }}>
          Algorithm Comparison
        </p>
        <h2
          className="font-heading font-bold text-3xl sm:text-4xl lg:text-[3.25rem] text-landing-text leading-tight"
          style={{ letterSpacing: '-0.02em' }}
        >
          Four Models.<br />Two Datasets. Full Benchmark.
        </h2>
        <p className="text-[14px] leading-relaxed" style={{ color: 'rgba(240,249,255,0.45)' }}>
          Two baselines (Attention U-Net, Lite U-Net) evaluated against Dueling DQN (discrete)
          and TD3 (continuous) on CAMUS cardiac ultrasound and BRISC brain MRI. Dice, IoU,
          and Hausdorff distance reported across both datasets.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {models.map((m) => <ModelTag key={m.id} model={m} />)}
      </div>

      <a
        href="/models"
        className="text-[13px] font-body no-underline transition-colors duration-panel ease-out w-fit flex items-center gap-2"
        style={{ color: 'var(--color-gradient-b)' }}
      >
        Explore all models in the library
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="M2 6h8M6 2l4 4-4 4" />
        </svg>
      </a>
    </div>
  </section>
);
export default ModelPreview;
