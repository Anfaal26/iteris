/**
 * ModelsSection — three algorithm cards (DQN, DDQN, Dueling DQN, DDPG) loaded
 * from models.yaml via ModelRecord. Baseline is shown for reference only.
 * (spec §5)
 */
import React from 'react';
import type { ModelRecord, ModelFamily } from '@/api/contract';
import modelsRaw from '@/content/models.yaml';

/** Props for ModelsSection. */
export interface ModelsSectionProps {
  id?: string;
}

const models = modelsRaw as ModelRecord[];

// DRL agents only (baseline excluded — shown separately)
const DRL_IDS = ['dqn', 'ddqn', 'dueling-dqn', 'ddpg'];

const FAMILY_LABEL: Record<ModelFamily, string> = {
  baseline: 'Baseline',
  'discrete-drl': 'Discrete DRL',
  'continuous-drl': 'Continuous DRL',
};

const FAMILY_CLASSES: Record<ModelFamily, string> = {
  baseline: 'bg-border text-muted',
  'discrete-drl': 'bg-accent/10 text-accent',
  'continuous-drl': 'bg-uncertainty/10 text-uncertainty',
};

interface HyperRow {
  param: string;
  value: string;
}

const HYPERPARAMS: Record<string, HyperRow[]> = {
  dqn: [
    { param: 'γ (discount)', value: '0.99' },
    { param: 'ε-greedy start', value: '1.0' },
    { param: 'ε-greedy end', value: '0.05' },
    { param: 'Replay buffer', value: '100,000' },
    { param: 'Batch size', value: '64' },
    { param: 'Target update', value: '500 steps' },
    { param: 'Loss', value: 'Huber (δ = 1)' },
  ],
  ddqn: [
    { param: 'γ (discount)', value: '0.99' },
    { param: 'ε-greedy start', value: '1.0' },
    { param: 'ε-greedy end', value: '0.05' },
    { param: 'Replay buffer', value: '100,000' },
    { param: 'Batch size', value: '64' },
    { param: 'Target update', value: '500 steps' },
    { param: 'Decoupled selection', value: 'Online network' },
  ],
  'dueling-dqn': [
    { param: 'γ (discount)', value: '0.99' },
    { param: 'ε-greedy start', value: '1.0' },
    { param: 'ε-greedy end', value: '0.05' },
    { param: 'Value stream', value: 'FC-256 → 1' },
    { param: 'Advantage stream', value: 'FC-256 → |A|' },
    { param: 'Aggregation', value: 'V + A − mean(A)' },
    { param: 'Target update', value: '500 steps' },
  ],
  ddpg: [
    { param: 'γ (discount)', value: '0.99' },
    { param: 'Actor LR', value: '1e-4' },
    { param: 'Critic LR', value: '1e-3' },
    { param: 'Soft update τ', value: '0.005' },
    { param: 'OU noise θ', value: '0.15' },
    { param: 'OU noise σ', value: '0.20' },
    { param: 'Batch size', value: '128' },
  ],
};

const ARCH: Record<string, string> = {
  dqn: 'Patch-CNN encoder (3 × Conv2d) → FC-256 → FC-|A|',
  ddqn: 'Patch-CNN encoder (3 × Conv2d) → FC-256 → FC-|A| (online + target)',
  'dueling-dqn': 'Patch-CNN → FC-256 → Value stream + Advantage stream',
  ddpg: 'Actor: FC-256 × 2 → tanh; Critic: FC-256 × 2 → Q-value',
};

/**
 * Renders DRL algorithm cards with architecture, description, and hyperparameter tables.
 */
export const ModelsSection: React.FC<ModelsSectionProps> = ({ id = 'models' }) => {
  const drlModels = models.filter((m) => DRL_IDS.includes(m.id));

  return (
    <section id={id} aria-labelledby="models-heading" className="py-12 scroll-mt-16">
      <h2 id="models-heading" className="font-heading text-xl font-bold text-text mb-8">
        Models
      </h2>

      <div className="space-y-6">
        {drlModels.map((model) => {
          const familyClass =
            FAMILY_CLASSES[model.family] ?? FAMILY_CLASSES['discrete-drl'];
          const familyLabel =
            FAMILY_LABEL[model.family] ?? model.family;
          const hypers = HYPERPARAMS[model.id] ?? [];
          const arch = ARCH[model.id] ?? '—';

          return (
            <article
              key={model.id}
              className="bg-surface border border-border rounded-lg p-6"
              aria-label={`Model: ${model.name}`}
            >
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <h3 className="font-heading text-base font-bold text-text">
                  {model.name}
                </h3>
                <span
                  className={`px-2 py-0.5 rounded text-xs font-body font-medium ${familyClass}`}
                >
                  {familyLabel}
                </span>
              </div>

              <p className="text-xs font-mono text-muted mb-3 border border-border rounded px-3 py-1.5 bg-bg">
                {arch}
              </p>

              <p className="text-sm font-body text-muted leading-relaxed mb-4">
                {model.description}
              </p>

              <div className="overflow-x-auto rounded border border-border">
                <table className="w-full text-xs font-body" aria-label={`${model.name} hyperparameters`}>
                  <thead>
                    <tr className="bg-bg">
                      <th className="text-left px-3 py-1.5 font-semibold text-muted border-b border-border">
                        Parameter
                      </th>
                      <th className="text-right px-3 py-1.5 font-semibold text-muted border-b border-border">
                        Value
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {hypers.map((row, i) => (
                      <tr key={row.param} className={i % 2 === 0 ? 'bg-surface' : 'bg-bg'}>
                        <td className="px-3 py-1.5 text-muted">{row.param}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-text">
                          {row.value}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
};

export default ModelsSection;
