/** FeatureStrip — glassmorphism feature cards, medical-imaging aesthetic. */
import React from 'react';

interface FeatureCard { title: string; description: string; icon: React.ReactNode; hero?: boolean; }

const PlayIcon = () => (
  <svg width="20" height="20" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <circle cx="11" cy="11" r="9"/><polygon points="9,7.5 16,11 9,14.5" fill="currentColor" stroke="none"/>
  </svg>
);
const WipeIcon = () => (
  <svg width="20" height="20" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <line x1="11" y1="3" x2="11" y2="19" strokeDasharray="2 1.5"/>
    <rect x="3" y="5" width="6" height="12" rx="1.5" opacity="0.5"/><rect x="13" y="5" width="6" height="12" rx="1.5"/>
  </svg>
);
const SideBySideIcon = () => (
  <svg width="20" height="20" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="2" y="5" width="8" height="12" rx="1.5"/><rect x="12" y="5" width="8" height="12" rx="1.5"/>
  </svg>
);
const LibraryIcon = () => (
  <svg width="20" height="20" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="12" y="3" width="7" height="7" rx="1.5"/>
    <rect x="3" y="12" width="7" height="7" rx="1.5"/><rect x="12" y="12" width="7" height="7" rx="1.5"/>
  </svg>
);

const CARDS: FeatureCard[] = [
  { title: 'Iteration Playback', description: 'Watch the DRL agent refine boundaries across 20 episodes with live Δ Dice sparklines. Full trajectory visibility no other tool offers.', icon: <PlayIcon />, hero: true },
  { title: 'Wipe Comparison',   description: 'Drag a vertical divider to compare baseline and DRL masks pixel-perfect on the same scan with synced zoom.', icon: <WipeIcon /> },
  { title: 'Side-by-Side View', description: 'Three model outputs, shared viewport, per-column metrics, best-Dice badge highlighted automatically.', icon: <SideBySideIcon /> },
  { title: 'Model Library',     description: 'Performance tables, convergence curves, and one-click workspace load for both DRL architectures and the U-Net baselines.', icon: <LibraryIcon /> },
];

const GLASS: React.CSSProperties = { backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)' };

export const FeatureStrip: React.FC = () => (
  <section aria-label="Features" className="bg-landing-bg py-24 px-6 lg:px-10">
    <div className="mx-auto max-w-6xl">
      <p className="font-mono text-[11px] tracking-[0.18em] uppercase mb-10" style={{ color: 'rgba(56,189,248,0.7)' }}>Capabilities</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {CARDS.map((card) => (
          <article key={card.title}
            className="group relative flex flex-col gap-5 rounded-2xl p-6 transition-all duration-slide ease-out"
            style={card.hero
              ? { background: 'rgba(56,189,248,0.07)', border: '1px solid rgba(56,189,248,0.2)', ...GLASS }
              : { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', ...GLASS }
            }
          >
            {/* Hover glow */}
            <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-slide pointer-events-none"
              style={{ background: card.hero
                ? 'radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.1) 0%, transparent 65%)'
                : 'radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.06) 0%, transparent 65%)'
              }} />
            {/* Top shimmer — hero only */}
            {card.hero && (
              <div className="absolute inset-x-0 top-0 h-px rounded-t-2xl pointer-events-none" aria-hidden="true"
                style={{ background: 'linear-gradient(90deg, transparent 0%, rgba(56,189,248,0.7) 35%, rgba(186,230,253,0.6) 50%, rgba(56,189,248,0.7) 65%, transparent 100%)' }} />
            )}
            {/* Icon */}
            <div className="flex items-center justify-center w-10 h-10 rounded-xl" style={{
              background: card.hero ? 'rgba(56,189,248,0.12)' : 'rgba(255,255,255,0.05)',
              border: card.hero ? '1px solid rgba(56,189,248,0.25)' : '1px solid rgba(255,255,255,0.08)',
              color: card.hero ? 'var(--color-gradient-b)' : 'rgba(240,249,255,0.5)',
            }}>{card.icon}</div>
            <div className="flex flex-col gap-2">
              <h3 className="font-heading font-semibold text-[15px]" style={{ color: 'var(--color-landing-text)' }}>{card.title}</h3>
              <p className="text-[13px] leading-relaxed" style={{ color: 'rgba(240,249,255,0.42)' }}>{card.description}</p>
            </div>
          </article>
        ))}
      </div>
    </div>
  </section>
);
export default FeatureStrip;
