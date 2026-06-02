/**
 * Landing page (spec §4) — medical-imaging glassmorphism redesign.
 *
 * Visual language:
 *  - Deep black (#030508) base — MRI lightbox
 *  - Ambient cold-blue glow orbs (scanner light through tissue)
 *  - Subtle DICOM crosshair scan-grid texture
 *  - Frosted glass cards and surfaces throughout
 *  - Ice-white → scanner-blue → deep-ocean gradient (#bae6fd → #38bdf8 → #0ea5e9)
 *  - Slow scan-line pulse on hero (ultrasound beam aesthetic)
 */

import { Suspense, lazy } from 'react';
import { Navbar } from '@/components';
import type { ModelRecord } from '@/api/contract';

import { FeatureStrip }    from './sections/FeatureStrip';
import { HowItWorks }      from './sections/HowItWorks';
import { ResearchMetrics } from './sections/ResearchMetrics';
import { ModelPreview }    from './sections/ModelPreview';
import { ResearchContext } from './sections/ResearchContext';
import { PreFooterCTA }    from './sections/PreFooterCTA';
import { LandingFooter }   from './sections/LandingFooter';

const HeroScene = lazy(() => import('./HeroScene'));

import rawModels from '@/content/models.yaml';
const models = rawModels as ModelRecord[];

const NAV_ITEMS = [
  { label: 'Research', href: '/research' },
  { label: 'Models',   href: '/models'   },
  { label: 'Datasets', href: '/datasets' },
];

// ── Ambient glow orbs (CSS only, no canvas) ────────────────────────────────
// Three overlapping radial gradients — cold blue, like MRI contrast agent
// dispersion or scanner light through soft tissue.
const AmbientGlows = () => (
  <div aria-hidden="true" className="absolute inset-0 overflow-hidden pointer-events-none">
    {/* Primary: large diffuse orb, top-right, scanner blue */}
    <div style={{
      position: 'absolute', top: '-10%', right: '-5%',
      width: '65vw', height: '65vw', maxWidth: 900, maxHeight: 900,
      borderRadius: '50%',
      background: 'radial-gradient(circle, rgba(56,189,248,0.12) 0%, rgba(14,165,233,0.05) 45%, transparent 70%)',
      filter: 'blur(40px)',
    }} />
    {/* Secondary: smaller, centre-left, ice */}
    <div style={{
      position: 'absolute', top: '30%', left: '-8%',
      width: '40vw', height: '40vw', maxWidth: 600, maxHeight: 600,
      borderRadius: '50%',
      background: 'radial-gradient(circle, rgba(186,230,253,0.07) 0%, transparent 65%)',
      filter: 'blur(60px)',
    }} />
    {/* Tertiary: deep, bottom-right, accent */}
    <div style={{
      position: 'absolute', bottom: '5%', right: '15%',
      width: '30vw', height: '30vw', maxWidth: 440, maxHeight: 440,
      borderRadius: '50%',
      background: 'radial-gradient(circle, rgba(14,165,233,0.09) 0%, transparent 60%)',
      filter: 'blur(50px)',
    }} />
  </div>
);

// ── DICOM scan-grid texture ──────────────────────────────────────────────────
const ScanGrid = () => (
  <div aria-hidden="true" className="absolute inset-0 pointer-events-none" style={{
    backgroundImage: `
      linear-gradient(var(--scan-grid) 1px, transparent 1px),
      linear-gradient(90deg, var(--scan-grid) 1px, transparent 1px)
    `,
    backgroundSize: '48px 48px',
    maskImage: 'radial-gradient(ellipse 80% 60% at 70% 40%, black 20%, transparent 80%)',
    WebkitMaskImage: 'radial-gradient(ellipse 80% 60% at 70% 40%, black 20%, transparent 80%)',
  }} />
);

// ── Slow scan-line pulse (ultrasound beam) ──────────────────────────────────
const ScanLine = () => (
  <>
    <style>{`
      @keyframes scanPulse {
        0%   { top: 60px; opacity: 0; }
        5%   { opacity: 0.35; }
        90%  { opacity: 0.18; }
        100% { top: 100%; opacity: 0; }
      }
    `}</style>
    <div
      aria-hidden="true"
      className="absolute inset-x-0 pointer-events-none"
      style={{
        height: '1px',
        background: 'linear-gradient(90deg, transparent 0%, rgba(56,189,248,0.6) 35%, rgba(186,230,253,0.9) 50%, rgba(56,189,248,0.6) 65%, transparent 100%)',
        animation: 'scanPulse 8s ease-in-out infinite',
        animationDelay: '2s',
      }}
    />
  </>
);

// ── Staggered entry keyframe ─────────────────────────────────────────────────
const EntryKeyframes = () => (
  <style>{`
    @keyframes fadeSlideUp {
      from { opacity: 0; transform: translateY(20px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
  `}</style>
);

// ── Hero section ─────────────────────────────────────────────────────────────
const HeroSection = () => (
  <section
    aria-label="Hero"
    className="relative flex min-h-screen w-full overflow-hidden bg-landing-bg"
  >
    <AmbientGlows />
    <ScanGrid />
    <ScanLine />

    {/* Left copy — 45% */}
    <div className="relative z-10 flex flex-col justify-center gap-10 px-8 sm:px-12 lg:px-16 xl:px-20
                    w-full lg:w-[48%] pt-32 pb-24">

      {/* Eyebrow chip — glass pill */}
      <div
        className="self-start animate-[fadeSlideUp_0.7s_ease-out_both]"
        style={{ animationDelay: '0ms' }}
      >
        <span
          className="inline-flex items-center gap-2 rounded-full px-4 py-1.5
                     font-mono text-[11px] tracking-widest uppercase"
          style={{
            background: 'rgba(56,189,248,0.08)',
            border: '1px solid rgba(186,230,253,0.15)',
            color: 'var(--color-gradient-b)',
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-grad-b animate-pulse" />
          Taylor's University · PRJ63504 Capstone
        </span>
      </div>

      {/* H1 — very large */}
      <h1
        className="font-heading font-bold leading-[1.04] text-landing-text
                   text-5xl sm:text-6xl lg:text-[5.5rem] xl:text-[6.5rem]
                   animate-[fadeSlideUp_0.7s_ease-out_both]"
        style={{ animationDelay: '120ms', letterSpacing: '-0.02em' }}
      >
        See How AI
        <br />
        <span
          className="bg-iteris-gradient bg-clip-text text-transparent"
          style={{ WebkitBackgroundClip: 'text' }}
        >
          Learns to See.
        </span>
      </h1>

      {/* Sub-headline */}
      <p
        className="font-body font-light leading-relaxed max-w-[380px]
                   text-[15px] sm:text-[17px] text-landing-text/50
                   animate-[fadeSlideUp_0.7s_ease-out_both]"
        style={{ animationDelay: '240ms' }}
      >
        A DRL-powered medical image segmentation workstation. Compare DQN,
        DDQN and DDPG agents on cardiac ultrasound and brain MRI — with full
        iteration playback.
      </p>

      {/* CTAs */}
      <div
        className="flex flex-wrap items-center gap-3 animate-[fadeSlideUp_0.7s_ease-out_both]"
        style={{ animationDelay: '360ms' }}
      >
        {/* Primary — filled glass */}
        <a
          href="/workspace"
          className="group flex items-center gap-2 rounded-full px-6 py-3
                     font-heading font-semibold text-sm text-landing-bg
                     transition-all duration-panel ease-out no-underline"
          style={{ background: 'var(--color-gradient-b)' }}
        >
          Open Workstation
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" className="transition-transform duration-panel group-hover:translate-x-0.5" aria-hidden="true">
            <path d="M2 7h10M7 2l5 5-5 5" />
          </svg>
        </a>

        {/* Secondary — ghost glass */}
        <a
          href="/research"
          className="flex items-center gap-2 rounded-full px-6 py-3
                     font-heading font-semibold text-sm text-landing-text/80
                     hover:text-landing-text
                     transition-all duration-panel ease-out no-underline"
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.12)',
          }}
        >
          View Research
        </a>
      </div>

      {/* Stat pills row */}
      <div
        className="flex flex-wrap gap-2 animate-[fadeSlideUp_0.7s_ease-out_both]"
        style={{ animationDelay: '480ms' }}
      >
        {[
          { val: '0.912', label: 'Best Dice · DDPG', dot: 'bg-grad-b' },
          { val: 'p < 0.001', label: 'Statistical Sig.', dot: 'bg-success' },
          { val: '20 steps', label: 'Playback depth', dot: 'bg-grad-a' },
        ].map((s) => (
          <span
            key={s.label}
            className="inline-flex items-center gap-2 rounded-full px-3 py-1.5
                       font-mono text-[11px] text-landing-text/70"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}
          >
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${s.dot}`} aria-hidden="true" />
            <strong className="text-landing-text font-bold">{s.val}</strong>
            <span className="text-landing-text/40">{s.label}</span>
          </span>
        ))}
      </div>

      {/* Scroll cue */}
      <p
        className="font-body text-[11px] text-landing-text/20 flex items-center gap-2
                   animate-[fadeIn_1s_ease-out_1.2s_both]"
        aria-hidden="true"
      >
        <span className="w-px h-5 bg-landing-text/15" />
        Scroll to explore
      </p>
    </div>

    {/* Right — Three.js lattice, full height */}
    <div
      className="absolute inset-y-0 right-0 w-full lg:w-[58%] pointer-events-none"
      aria-hidden="true"
    >
      <Suspense fallback={null}>
        <HeroScene className="w-full h-full" />
      </Suspense>
      {/* Left-edge dissolve into copy */}
      <div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(to right, var(--color-landing-bg) 0%, rgba(3,5,8,0.6) 20%, transparent 50%)',
        }}
      />
    </div>
  </section>
);

// ── Page ──────────────────────────────────────────────────────────────────────
export default function Landing() {
  return (
    <>
      <EntryKeyframes />

      <Navbar
        variant="dark"
        navItems={NAV_ITEMS}
        onSearch={() => {}}
        onSettings={() => {}}
      />

      <main className="bg-landing-bg text-landing-text font-body">
        <HeroSection />
        <FeatureStrip />
        <HowItWorks />
        <ResearchMetrics />
        <ModelPreview models={models} />
        <ResearchContext />
        <PreFooterCTA />
      </main>

      <LandingFooter />
    </>
  );
}
