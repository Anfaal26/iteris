/**
 * ModelPreview — Section 5: row of model tag cards from models.yaml.
 * DDPG gets a "BEST" badge. Links to /models.
 */

import React from 'react';
import type { ModelRecord } from '@/api/contract';

/** Props for ModelPreview. */
export interface ModelPreviewProps {
  /** Model records from models.yaml. */
  models: ModelRecord[];
}

/** A single model tag card. */
const ModelTag: React.FC<{ model: ModelRecord }> = ({ model }) => {
  const isBest = model.id === 'ddpg';
  const bestDice = model.diceCamus ?? model.diceBrisc;

  return (
    <article
      className={[
        'relative flex flex-col gap-2 rounded-xl p-5',
        'border border-white/[0.08] bg-white/[0.03]',
        isBest ? 'ring-1 ring-grad-a/30' : '',
      ].join(' ')}
    >
      {isBest && (
        <span className="absolute -top-2.5 right-4 rounded-full bg-grad-a px-2 py-0.5 font-mono text-[10px] font-bold text-landing-bg uppercase tracking-wider">
          BEST
        </span>
      )}
      <h3 className="font-heading font-bold text-sm text-landing-text">
        {model.name}
      </h3>
      <p className="text-xs text-landing-text/40 capitalize">
        {model.family.replace(/-/g, ' ')}
      </p>
      {bestDice != null && (
        <p className="font-mono text-xs text-grad-a">
          Dice {bestDice.toFixed(3)}
        </p>
      )}
    </article>
  );
};

/** Section 5 — Model preview cards. */
export const ModelPreview: React.FC<ModelPreviewProps> = ({ models }) => (
  <section
    aria-label="Model preview"
    className="bg-landing-bg py-24 px-6"
  >
    <div className="mx-auto max-w-6xl flex flex-col gap-12">
      {/* Header */}
      <div className="flex flex-col gap-3">
        <span className="font-mono text-xs tracking-widest uppercase text-grad-a">
          Algorithm Comparison
        </span>
        <h2 className="font-heading font-bold text-3xl sm:text-4xl text-landing-text max-w-2xl">
          Three Algorithms. Two Datasets. Complete Evaluation.
        </h2>
        <p className="text-sm text-landing-text/40 max-w-xl leading-relaxed">
          Every DRL agent is benchmarked against the U-Net baseline on CAMUS cardiac
          ultrasound and BRISC brain MRI, with Dice, IoU, and Hausdorff distance reported.
        </p>
      </div>

      {/* Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {models.map((m) => (
          <ModelTag key={m.id} model={m} />
        ))}
      </div>

      {/* Link */}
      <a
        href="/models"
        className="text-sm text-grad-a hover:text-grad-b transition-colors duration-panel ease-out w-fit"
      >
        Explore all models in the library →
      </a>
    </div>
  </section>
);

export default ModelPreview;
