/**
 * Landing page (spec §4) — dark-theme marketing page for ITERIS.
 *
 * 8 sections:
 *  1. Navbar (dark)
 *  2. Hero — Three.js lattice + copy
 *  3. Feature strip
 *  4. How It Works
 *  5. Research Metrics strip
 *  6. Model Preview
 *  7. Research Context pull-quote
 *  8. Pre-footer CTA
 *  9. Footer
 *
 * All colours resolve from CSS custom properties (no raw hex).
 * Three.js scene runs at 30fps (motion.landingFps). Staggered
 * hero entry uses CSS animation-delay with 150ms gaps (motion.staggerMs).
 */

import React, { Suspense, lazy } from 'react';
import { Navbar } from '@/components';
import type { ModelRecord } from '@/api/contract';

// Sections
import { FeatureStrip } from './sections/FeatureStrip';
import { HowItWorks } from './sections/HowItWorks';
import { ResearchMetrics } from './sections/ResearchMetrics';
import { ModelPreview } from './sections/ModelPreview';
import { ResearchContext } from './sections/ResearchContext';
import { PreFooterCTA } from './sections/PreFooterCTA';
import { LandingFooter } from './sections/LandingFooter';

// Three.js scene — lazy so it doesn't block initial paint
const HeroScene = lazy(() => import('./HeroScene'));

// Model data from YAML (typed via vite-plugin-yaml)
import rawModels from '@/content/models.yaml';
const models = rawModels as ModelRecord[];

// Nav links for the dark navbar
const NAV_ITEMS = [
  { label: 'Research', href: '/research' },
  { label: 'Models', href: '/models' },
  { label: 'Workspace', href: '/workspace' },
];

/**
 * Hero section (Section 1) — full viewport, left copy + right Three.js canvas.
 * Staggered entry animation via CSS custom delay classes.
 */
const HeroSection: React.FC = () => (
  <section
    aria-label="Hero"
    className="relative flex min-h-screen w-full overflow-hidden bg-landing-bg"
  >
    {/* Left content ~40% */}
    <div className="relative z-10 flex flex-col justify-center gap-8 px-8 sm:px-12 lg:px-16 py-32
                    w-full lg:w-[42%] lg:min-h-screen">
      {/* Eyebrow chip */}
      <span
        className="self-start rounded-full border border-grad-a/40 bg-grad-a/5
                   px-4 py-1.5 font-mono text-xs text-grad-a
                   animate-[fadeSlideUp_0.6s_ease-out_both]"
        style={{ animationDelay: '0ms' }}
      >
        Taylor&apos;s University · PRJ63504 Capstone
      </span>

      {/* H1 */}
      <h1
        className="font-heading font-bold text-5xl sm:text-6xl leading-[1.08] text-landing-text
                   animate-[fadeSlideUp_0.6s_ease-out_both]"
        style={{ animationDelay: '150ms' }}
      >
        See How AI
        <br />
        <span className="bg-iteris-gradient bg-clip-text text-transparent">
          Learns to See.
        </span>
      </h1>

      {/* Sub-headline — 17px DM Sans Light, 48% opacity, 3 lines max */}
      <p
        className="font-body font-light text-[17px] leading-relaxed text-landing-text/[0.48]
                   max-w-[340px] line-clamp-3
                   animate-[fadeSlideUp_0.6s_ease-out_both]"
        style={{ animationDelay: '300ms' }}
      >
        ITERIS is a DRL-powered medical image segmentation workstation. Compare
        DQN, DDQN and DDPG agents side-by-side and watch boundary refinement unfold
        step by step.
      </p>

      {/* CTA buttons */}
      <div
        className="flex flex-wrap items-center gap-4 animate-[fadeSlideUp_0.6s_ease-out_both]"
        style={{ animationDelay: '450ms' }}
      >
        <a
          href="/workspace"
          className="rounded-full bg-grad-a px-7 py-3 font-heading font-semibold text-sm
                     text-landing-bg hover:opacity-90 transition-opacity duration-panel ease-out"
        >
          Try Iteris →
        </a>
        <a
          href="/research"
          className="rounded-full border border-white/25 px-7 py-3 font-heading font-semibold text-sm
                     text-landing-text hover:border-white/50 hover:bg-white/[0.04]
                     transition-colors duration-panel ease-out"
        >
          View Our Research
        </a>
      </div>

      {/* Stat pill */}
      <div
        className="flex items-center gap-2 self-start animate-[fadeSlideUp_0.6s_ease-out_both]"
        style={{ animationDelay: '600ms' }}
      >
        <span
          className="inline-flex items-center gap-2 rounded-full border border-white/[0.08]
                     bg-white/[0.04] px-4 py-1.5 font-mono text-xs text-landing-text"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" />
          Best Dice 0.912 · DDPG
        </span>
      </div>

      {/* Scroll cue */}
      <p
        className="font-body text-xs text-landing-text/[0.22] mt-auto
                   animate-[fadeSlideUp_0.6s_ease-out_both]"
        style={{ animationDelay: '750ms' }}
        aria-hidden="true"
      >
        ↓ Scroll to explore
      </p>
    </div>

    {/* Right Three.js canvas ~60% */}
    <div className="absolute inset-y-0 right-0 w-full lg:w-[62%] pointer-events-none" aria-hidden="true">
      <Suspense fallback={null}>
        <HeroScene className="w-full h-full" />
      </Suspense>

      {/* Left-edge fade overlay so canvas blends into hero copy */}
      <div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(to right, var(--color-landing-bg) 0%, transparent 35%)',
        }}
      />
    </div>
  </section>
);

/**
 * Landing — full landing page with 8 content sections.
 * Page-level `data-theme` is intentionally omitted; the landing always
 * lives in the dark token context set by its Tailwind bg-landing-bg classes.
 */
const Landing: React.FC = () => (
  <>
    {/* Global fade-slide keyframe (reduce-motion handled globally in index.css) */}
    <style>{`
      @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0);    }
      }
    `}</style>

    <Navbar
      variant="dark"
      navItems={NAV_ITEMS}
      onSearch={() => {}}
      onSettings={() => {}}
    />

    <main className="bg-landing-bg text-landing-text font-body">
      {/* §4 Section 1 — Hero */}
      <HeroSection />

      {/* §4 Section 2 — Feature strip */}
      <FeatureStrip />

      {/* §4 Section 3 — How It Works */}
      <HowItWorks />

      {/* §4 Section 4 — Research Metrics */}
      <ResearchMetrics />

      {/* §4 Section 5 — Model Preview */}
      <ModelPreview models={models} />

      {/* §4 Section 6 — Research Context */}
      <ResearchContext />

      {/* §4 Section 7 — Pre-footer CTA */}
      <PreFooterCTA />
    </main>

    {/* §4 Section 8 — Footer */}
    <LandingFooter />
  </>
);

export default Landing;
