/**
 * ModelsSection — renders the live model registry (src/content/models.yaml)
 * via ModelCard, so this section can never drift from what /models shows.
 * (spec §5)
 */
import React from 'react';
import { ModelCard } from '@/components';
import type { ModelRecord } from '@/api/contract';
import modelsRaw from '@/content/models.yaml';

/** Props for ModelsSection. */
export interface ModelsSectionProps {
  id?: string;
}

const models = modelsRaw as ModelRecord[];

const ARCH: Partial<Record<ModelRecord['id'], string>> = {
  'unet-baseline': 'Attention Residual U-Net (Oktay et al. 2018 attention gates)',
  'lite-unet': 'Compact encoder/decoder, reduced channel width',
  'dueling-dqn': 'ContourRefineEnv · discrete angular sectors · V + A − mean(A)',
  td3: 'ContourRefineEnv · continuous per-sector displacement · clipped double-Q, delayed policy updates',
};

/**
 * Renders every registered model (baselines + DRL agents) with live status —
 * no hardcoded metrics, since none currently exist.
 */
export const ModelsSection: React.FC<ModelsSectionProps> = ({ id = 'models' }) => {
  const baselines = models.filter((m) => m.family === 'baseline');
  const agents = models.filter((m) => m.family !== 'baseline');

  return (
    <section id={id} aria-labelledby="models-heading" className="py-12 scroll-mt-16">
      <h2 id="models-heading" className="font-heading text-xl font-bold text-text mb-2">
        Models
      </h2>
      <p className="text-sm font-body text-muted mb-8">
        Sourced live from the model registry — the same data shown on{' '}
        <a href="/models" className="text-accent hover:underline">
          Model Library
        </a>
        . Metric pills read "—" until a model has been evaluated.
      </p>

      <div className="mb-8">
        <h3 className="font-heading text-sm font-semibold text-text mb-3">Baselines</h3>
        <div className="grid gap-4 sm:grid-cols-2">
          {baselines.map((model) => (
            <div key={model.id}>
              <ModelCard model={model} variant="expanded" className="w-full" />
              {ARCH[model.id] && (
                <p className="mt-1.5 text-xs font-mono text-muted px-1">{ARCH[model.id]}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="font-heading text-sm font-semibold text-text mb-3">DRL Agents</h3>
        <div className="grid gap-4 sm:grid-cols-2">
          {agents.map((model) => (
            <div key={model.id}>
              <ModelCard model={model} variant="expanded" className="w-full" />
              {ARCH[model.id] && (
                <p className="mt-1.5 text-xs font-mono text-muted px-1">{ARCH[model.id]}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ModelsSection;
