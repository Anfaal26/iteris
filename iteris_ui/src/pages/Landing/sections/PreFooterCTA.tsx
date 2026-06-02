/** PreFooterCTA — centred CTA with ambient scanner-glow orb. */
import React from 'react';

export const PreFooterCTA: React.FC = () => (
  <section aria-label="Call to action" className="relative py-32 px-6 text-center overflow-hidden"
    style={{ background: 'var(--color-landing-footer)' }}>
    {/* Ambient glow */}
    <div aria-hidden="true" className="absolute inset-0 pointer-events-none flex items-center justify-center">
      <div style={{
        width: '60vw', height: '40vw', maxWidth: 700, maxHeight: 400, borderRadius: '50%',
        background: 'radial-gradient(ellipse, rgba(56,189,248,0.1) 0%, transparent 65%)',
        filter: 'blur(60px)',
      }} />
    </div>
    <div className="relative mx-auto max-w-2xl flex flex-col items-center gap-8">
      <p className="font-mono text-[11px] tracking-[0.18em] uppercase" style={{ color: 'rgba(56,189,248,0.7)' }}>Get Started</p>
      <h2 className="font-heading font-bold text-4xl sm:text-5xl lg:text-6xl text-landing-text leading-tight" style={{ letterSpacing: '-0.02em' }}>
        Ready to explore?
      </h2>
      <p className="text-[15px] leading-relaxed max-w-md" style={{ color: 'rgba(240,249,255,0.45)' }}>
        Run inference on real medical scans, replay DRL boundary refinement step by step, and generate LLM-powered interpretations.
      </p>
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <a href="/workspace"
          className="flex items-center gap-2 rounded-full px-7 py-3.5 font-heading font-semibold text-sm no-underline transition-all duration-panel ease-out"
          style={{ background: 'var(--color-gradient-b)', color: '#030508' }}>
          Open Workstation
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M2 7h10M7 2l5 5-5 5" />
          </svg>
        </a>
        <a href="/research"
          className="flex items-center gap-2 rounded-full px-7 py-3.5 font-heading font-semibold text-sm no-underline transition-all duration-panel ease-out"
          style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)', color: 'rgba(240,249,255,0.8)' }}>
          View Research
        </a>
      </div>
      <p className="text-[11px] leading-relaxed max-w-sm" style={{ color: 'rgba(240,249,255,0.2)' }}>
        Research prototype · PRJ63504 · Taylor's University · Not for clinical use
      </p>
    </div>
  </section>
);
export default PreFooterCTA;
