/**
 * ModelsSection — model registry (src/content/models.yaml) enriched, for
 * display on this page only, with the real 2026-07-20 evaluation numbers
 * (the registry itself still reads null pending a site-wide update outside
 * the scope of the Research page). (spec §5)
 */
import React from 'react';
import { ModelCard } from '@/components';
import type { ModelRecord } from '@/api/contract';
import modelsRaw from '@/content/models.yaml';
import { EVAL_ROWS, type Phase } from '../data/evaluationResults';

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

// Maps registry model IDs to the real evaluation data's model name + which
// phase to display: baselines show their own phase; DRL agents show Phase A
// (paired with the actually-deployed Attention U-Net baseline).
const EVAL_LOOKUP: Partial<Record<ModelRecord['id'], { name: string; phase: Phase }>> = {
  'unet-baseline': { name: 'AttentionResUNet', phase: 'Phase A' },
  'lite-unet': { name: 'LiteUNet', phase: 'Phase B' },
  'dueling-dqn': { name: 'DuelingDDQN', phase: 'Phase A' },
  td3: { name: 'TD3', phase: 'Phase A' },
};

function mean(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

/** Derives display-only metrics from the real evaluation rows for one model/phase. */
function evalMetricsFor(id: ModelRecord['id']): Pick<ModelRecord, 'diceCamus' | 'diceBrisc' | 'iou' | 'hd'> | null {
  const lookup = EVAL_LOOKUP[id];
  if (!lookup) return null;
  const rows = EVAL_ROWS.filter((r) => r.model === lookup.name && r.phase === lookup.phase);
  const camusRows = rows.filter((r) => r.dataset === 'CAMUS');
  const briscRows = rows.filter((r) => r.dataset === 'BRISC');
  const iouRows = rows.filter((r): r is typeof r & { iou: number } => r.iou != null);
  return {
    diceCamus: camusRows.length ? mean(camusRows.map((r) => r.dice)) : null,
    diceBrisc: briscRows.length ? mean(briscRows.map((r) => r.dice)) : null,
    iou: iouRows.length ? mean(iouRows.map((r) => r.iou)) : null,
    hd: rows.length ? mean(rows.map((r) => r.hd95)) : null,
  };
}

const DESCRIPTION_OVERRIDE: Partial<Record<ModelRecord['id'], string>> = {
  'unet-baseline':
    'Attention Residual U-Net (Oktay et al. 2018 attention gates, ResNet-style encoder/decoder). Deployed baseline — evaluated 2026-07-20 as the Phase A reference.',
  'lite-unet':
    'Compact U-Net with a reduced encoder/decoder. Not deployed for inference, but evaluated as the Phase B (lower-headroom) baseline the DRL agents were designed to have room to improve on.',
  'dueling-dqn':
    'Dueling Double DQN with value/advantage stream decomposition over discrete angular sectors. Evaluated 2026-07-20; not yet deployed for interactive inference — see Results for the full breakdown.',
  td3: 'Twin Delayed DDPG with continuous per-sector displacement actions and clipped double-Q targets. Evaluated 2026-07-20; not yet deployed for interactive inference — see Results for the full breakdown.',
};

/**
 * Renders every registered model (baselines + DRL agents), display-enriched
 * with the real evaluation numbers this page's Results section reports.
 */
export const ModelsSection: React.FC<ModelsSectionProps> = ({ id = 'models' }) => {
  const baselines = models.filter((m) => m.family === 'baseline');
  const agents = models.filter((m) => m.family !== 'baseline');

  const renderCard = (model: ModelRecord) => {
    const evalMetrics = evalMetricsFor(model.id);
    const displayModel: ModelRecord = evalMetrics
      ? {
          ...model,
          ...evalMetrics,
          description: DESCRIPTION_OVERRIDE[model.id] ?? model.description,
        }
      : model;
    return (
      <div key={model.id}>
        <ModelCard model={displayModel} variant="expanded" className="w-full" />
        {ARCH[model.id] && (
          <p className="mt-1.5 text-xs font-mono text-muted px-1">{ARCH[model.id]}</p>
        )}
      </div>
    );
  };

  return (
    <section id={id} aria-labelledby="models-heading" className="py-12 scroll-mt-16">
      <h2 id="models-heading" className="font-heading text-xl font-bold text-text mb-2">
        Models
      </h2>
      <p className="text-sm font-body text-muted mb-8">
        Registry from{' '}
        <a href="/models" className="text-accent hover:underline">
          Model Library
        </a>
        , with metric pills enriched here from the real 2026-07-20 evaluation (mean Dice
        across classes; DRL agents shown at Phase A, paired with the deployed baseline).
        The Model Library itself still shows "evaluation pending" until the registry is
        updated site-wide.
      </p>

      <div className="mb-8">
        <h3 className="font-heading text-sm font-semibold text-text mb-3">Baselines</h3>
        <div className="grid gap-4 sm:grid-cols-2">{baselines.map(renderCard)}</div>
      </div>

      <div>
        <h3 className="font-heading text-sm font-semibold text-text mb-3">DRL Agents</h3>
        <div className="grid gap-4 sm:grid-cols-2">{agents.map(renderCard)}</div>
      </div>
    </section>
  );
};

export default ModelsSection;
