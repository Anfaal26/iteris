/**
 * FeatureStrip — Section 2: 4 feature cards in a horizontal row.
 * "Iteration Playback" card gets the hero accent treatment.
 */

import React from 'react';

interface FeatureCard {
  title: string;
  description: string;
  icon: React.ReactNode;
  hero?: boolean;
}

const PlayIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
    <polygon points="4,2 18,10 4,18" />
  </svg>
);
const WipeIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <line x1="10" y1="2" x2="10" y2="18" />
    <rect x="2" y="4" width="6" height="12" rx="1" />
    <rect x="12" y="4" width="6" height="12" rx="1" />
  </svg>
);
const SideBySideIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="2" y="4" width="7" height="12" rx="1" />
    <rect x="11" y="4" width="7" height="12" rx="1" />
  </svg>
);
const LibraryIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="3" y="3" width="6" height="6" rx="1" />
    <rect x="11" y="3" width="6" height="6" rx="1" />
    <rect x="3" y="11" width="6" height="6" rx="1" />
    <rect x="11" y="11" width="6" height="6" rx="1" />
  </svg>
);

const CARDS: FeatureCard[] = [
  {
    title: 'Iteration Playback',
    description: 'Watch the DRL agent refine boundaries step-by-step across 20 episodes with live Dice deltas.',
    icon: <PlayIcon />,
    hero: true,
  },
  {
    title: 'Wipe Comparison',
    description: 'Drag a vertical split to compare original and refined masks on the same image.',
    icon: <WipeIcon />,
  },
  {
    title: 'Side-by-Side View',
    description: 'Inspect two model outputs simultaneously with synchronised pan and zoom.',
    icon: <SideBySideIcon />,
  },
  {
    title: 'Model Library',
    description: 'Browse DQN, DDQN, Dueling DQN and DDPG with full performance tables per dataset.',
    icon: <LibraryIcon />,
  },
];

/** Section 2 — Feature strip with 4 cards. */
export const FeatureStrip: React.FC = () => (
  <section
    aria-label="Features"
    className="bg-landing-bg py-20 px-6"
  >
    <div className="mx-auto max-w-6xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {CARDS.map((card) => (
        <article
          key={card.title}
          className={[
            'relative rounded-xl p-6 flex flex-col gap-4',
            'border border-white/[0.08]',
            card.hero
              ? 'bg-white/[0.07]'
              : 'bg-white/[0.03]',
          ].join(' ')}
        >
          {/* Teal→violet top accent bar for hero card */}
          {card.hero && (
            <div
              className="absolute inset-x-0 top-0 h-[2px] rounded-t-xl bg-iteris-gradient"
              aria-hidden="true"
            />
          )}

          {/* Icon circle */}
          <div
            className={[
              'flex items-center justify-center w-10 h-10 rounded-full',
              card.hero
                ? 'bg-grad-a/10 text-grad-a'
                : 'bg-white/[0.06] text-landing-text/60',
            ].join(' ')}
          >
            {card.icon}
          </div>

          <div className="flex flex-col gap-1.5">
            <h3 className="font-heading font-semibold text-sm text-landing-text">
              {card.title}
            </h3>
            <p className="text-xs leading-relaxed text-landing-text/40">
              {card.description}
            </p>
          </div>
        </article>
      ))}
    </div>
  </section>
);

export default FeatureStrip;
