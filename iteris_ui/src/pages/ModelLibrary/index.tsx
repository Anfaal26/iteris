/**
 * Model Library — filterable model cards with training curves (spec §8).
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Navbar, ModelCard } from '@/components';
import type { ModelRecord, ModelFamily, DatasetId } from '@/api/contract';
import { api } from '@/api/client';
import { ROUTES } from '@/routes';
import { colorsHex } from '@/tokens';

const NAV_ITEMS = [
  { label: 'Workspace', href: ROUTES.workspace },
  { label: 'Model Library', href: ROUTES.models },
  { label: 'Dataset Explorer', href: ROUTES.datasets },
];

type SortKey = 'diceCamus' | 'diceBrisc' | 'iou' | 'hd';

const SORT_OPTIONS: { label: string; key: SortKey }[] = [
  { label: 'CAMUS Dice', key: 'diceCamus' },
  { label: 'BRISC Dice', key: 'diceBrisc' },
  { label: 'IoU', key: 'iou' },
  { label: 'HD', key: 'hd' },
];

const FAMILY_OPTIONS: { label: string; value: ModelFamily | 'all' }[] = [
  { label: 'All', value: 'all' },
  { label: 'Discrete DRL', value: 'discrete-drl' },
  { label: 'Continuous DRL', value: 'continuous-drl' },
];

const DATASET_OPTIONS: { label: string; value: DatasetId | 'all' }[] = [
  { label: 'All', value: 'all' },
  { label: 'CAMUS', value: 'camus' },
  { label: 'BRISC', value: 'brisc' },
];

/** Generate synthetic sinusoidal convergence data for a given target Dice over N steps. */
function convergenceData(targetDice: number, steps = 50) {
  return Array.from({ length: steps }, (_, i) => {
    const progress = i / (steps - 1);
    const noise = Math.sin(i * 1.3) * 0.005;
    const dice = 0.7 + (targetDice - 0.7) * (1 - Math.exp(-4 * progress)) + noise;
    return { step: (i + 1) * 1000, dice: Math.max(0, Math.min(1, dice)) };
  });
}

interface TrainingCurvesProps {
  model: ModelRecord;
}

const TrainingCurves: React.FC<TrainingCurvesProps> = ({ model }) => {
  const camusData = useMemo(
    () => (model.diceCamus != null ? convergenceData(model.diceCamus) : []),
    [model.diceCamus],
  );
  const briscData = useMemo(
    () => (model.diceBrisc != null ? convergenceData(model.diceBrisc) : []),
    [model.diceBrisc],
  );

  // Line colours from tokens
  const camusColor = colorsHex.accent;
  const briscColor = colorsHex.gradientC;

  return (
    <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border mt-4">
      {model.diceCamus != null && (
        <div>
          <p className="text-xs font-body text-muted mb-2">CAMUS Dice vs Steps</p>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={camusData}>
              <XAxis
                dataKey="step"
                tick={{ fontSize: 10, fill: 'var(--color-muted)' }}
                tickFormatter={(v: number) => `${v / 1000}k`}
              />
              <YAxis
                domain={[0.7, 1]}
                tick={{ fontSize: 10, fill: 'var(--color-muted)' }}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  fontSize: 11,
                }}
                labelFormatter={(v) => `Step ${v}`}
                formatter={(v: number) => [v.toFixed(3), 'Dice']}
              />
              <Line
                type="monotone"
                dataKey="dice"
                stroke={camusColor}
                dot={false}
                strokeWidth={1.5}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {model.diceBrisc != null && (
        <div>
          <p className="text-xs font-body text-muted mb-2">BRISC Dice vs Steps</p>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={briscData}>
              <XAxis
                dataKey="step"
                tick={{ fontSize: 10, fill: 'var(--color-muted)' }}
                tickFormatter={(v: number) => `${v / 1000}k`}
              />
              <YAxis
                domain={[0.7, 1]}
                tick={{ fontSize: 10, fill: 'var(--color-muted)' }}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  fontSize: 11,
                }}
                labelFormatter={(v) => `Step ${v}`}
                formatter={(v: number) => [v.toFixed(3), 'Dice']}
              />
              <Line
                type="monotone"
                dataKey="dice"
                stroke={briscColor}
                dot={false}
                strokeWidth={1.5}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

interface ExpandedModelCardProps {
  model: ModelRecord;
  isBest: boolean;
  onLoadInWorkspace: (id: ModelRecord['id']) => void;
}

const ExpandedModelCard: React.FC<ExpandedModelCardProps> = ({
  model,
  isBest,
  onLoadInWorkspace,
}) => {
  const [showCurves, setShowCurves] = useState(false);

  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <ModelCard model={model} variant="expanded" isBest={isBest} className="w-full border-0 p-0 shadow-none" />
        </div>
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          {model.selectable && (
            <button
              type="button"
              onClick={() => onLoadInWorkspace(model.id)}
              className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-heading font-semibold hover:bg-accent/90 transition-colors duration-panel ease-out"
            >
              Load in Workspace
            </button>
          )}
          <button
            type="button"
            aria-expanded={showCurves}
            onClick={() => setShowCurves((v) => !v)}
            className="text-xs font-body text-accent hover:underline transition-colors duration-panel ease-out"
          >
            {showCurves ? 'Hide training curves ↑' : 'View training curves ↓'}
          </button>
        </div>
      </div>

      {showCurves && <TrainingCurves model={model} />}
    </div>
  );
};

/**
 * Model Library page — filterable and sortable list of models with expandable
 * training curve charts and workspace navigation.
 */
export default function ModelLibrary() {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [familyFilter, setFamilyFilter] = useState<ModelFamily | 'all'>('all');
  const [datasetFilter, setDatasetFilter] = useState<DatasetId | 'all'>('all');
  const [sortKey, setSortKey] = useState<SortKey>('diceCamus');

  useEffect(() => {
    api.models().then(setModels).catch(() => {});
  }, []);

  const filtered = useMemo(() => {
    let list = models;
    if (familyFilter !== 'all') {
      list = list.filter((m) => m.family === familyFilter);
    }
    if (datasetFilter === 'camus') {
      list = list.filter((m) => m.diceCamus != null);
    } else if (datasetFilter === 'brisc') {
      list = list.filter((m) => m.diceBrisc != null);
    }
    return [...list].sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      // For HD: lower is better
      if (sortKey === 'hd') return (av as number) - (bv as number);
      return (bv as number) - (av as number);
    });
  }, [models, familyFilter, datasetFilter, sortKey]);

  const bestId = useMemo(() => {
    if (models.length === 0) return null;
    return [...models].sort((a, b) => (b.diceCamus ?? 0) - (a.diceCamus ?? 0))[0]?.id ?? null;
  }, [models]);

  const handleLoadInWorkspace = (id: ModelRecord['id']) => {
    navigate(ROUTES.workspace, { state: { selectedModel: id } });
  };

  return (
    <div className="min-h-screen bg-bg font-body text-text">
      <Navbar variant="light" navItems={NAV_ITEMS} />

      {/* Sticky filter bar */}
      <div
        className="sticky top-0 z-40 bg-surface/95 border-b border-border px-8 py-3 flex flex-wrap items-center gap-4"
        style={{ marginTop: 'var(--navbar-height)', backdropFilter: 'blur(8px)' }}
        aria-label="Model filters"
      >
        {/* Family filter */}
        <div className="flex items-center gap-1" role="group" aria-label="Algorithm family">
          {FAMILY_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              aria-pressed={familyFilter === opt.value}
              onClick={() => setFamilyFilter(opt.value as ModelFamily | 'all')}
              className={[
                'px-3 py-1.5 rounded-md text-xs font-body transition-colors duration-panel ease-out',
                familyFilter === opt.value
                  ? 'bg-accent text-white'
                  : 'border border-border text-muted hover:border-accent/50',
              ].join(' ')}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-border" aria-hidden="true" />

        {/* Dataset filter */}
        <div className="flex items-center gap-1" role="group" aria-label="Dataset">
          {DATASET_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              aria-pressed={datasetFilter === opt.value}
              onClick={() => setDatasetFilter(opt.value as DatasetId | 'all')}
              className={[
                'px-3 py-1.5 rounded-md text-xs font-body transition-colors duration-panel ease-out',
                datasetFilter === opt.value
                  ? 'bg-accent text-white'
                  : 'border border-border text-muted hover:border-accent/50',
              ].join(' ')}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-border" aria-hidden="true" />

        {/* Sort */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-body text-muted">Sort by:</span>
          <div className="flex items-center gap-1" role="group" aria-label="Sort metric">
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                type="button"
                aria-pressed={sortKey === opt.key}
                onClick={() => setSortKey(opt.key)}
                className={[
                  'px-3 py-1.5 rounded-md text-xs font-body transition-colors duration-panel ease-out',
                  sortKey === opt.key
                    ? 'bg-accent text-white'
                    : 'border border-border text-muted hover:border-accent/50',
                ].join(' ')}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <span className="ml-auto text-xs font-mono text-muted">
          {filtered.length} model{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Model cards list */}
      <main
        className="px-8 py-6 flex flex-col gap-4 max-w-5xl mx-auto"
        aria-label="Model list"
      >
        {filtered.length === 0 ? (
          <p className="text-sm font-body text-muted py-12 text-center">
            No models match the current filters.
          </p>
        ) : (
          filtered.map((model) => (
            <ExpandedModelCard
              key={model.id}
              model={model}
              isBest={model.id === bestId}
              onLoadInWorkspace={handleLoadInWorkspace}
            />
          ))
        )}
      </main>
    </div>
  );
}
