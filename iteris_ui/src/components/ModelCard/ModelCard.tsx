/**
 * ModelCard — shows a model's name, family badge, description, metric pills,
 * active state (2px teal left border), BEST badge, selectable/greyed state.
 * Supports compact (workspace) and expanded (library) variants.
 */

import React from 'react';
import type { ModelRecord } from '@/api/contract';

/** Props for the ModelCard component. */
export interface ModelCardProps {
  /** The model data to display. */
  model: ModelRecord;
  /** Layout variant. @default 'compact' */
  variant?: 'compact' | 'expanded';
  /** Whether this card is currently selected/active. */
  active?: boolean;
  /** Whether to show the BEST badge. */
  isBest?: boolean;
  /** Called when the user selects this card (if selectable). */
  onSelect?: (id: ModelRecord['id']) => void;
  /** Additional class names. */
  className?: string;
}

const FAMILY_BADGE: Record<string, { label: string; classes: string }> = {
  baseline: { label: 'Baseline', classes: 'bg-border text-muted' },
  'discrete-drl': { label: 'Discrete DRL', classes: 'bg-accent/10 text-accent' },
  'continuous-drl': { label: 'Continuous DRL', classes: 'bg-uncertainty/10 text-uncertainty' },
};

interface MetricPillProps {
  label: string;
  value: number | null;
}

const MetricPill: React.FC<MetricPillProps> = ({ label, value }) => (
  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-mono bg-bg border border-border">
    <span className="text-muted">{label}</span>
    <span className="text-text">{value != null ? value.toFixed(3) : '—'}</span>
  </span>
);

/**
 * Card representing a segmentation model. Compact variant suits the workspace
 * sidebar (~272 px); expanded shows full description and all metrics.
 */
export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  variant = 'compact',
  active = false,
  isBest = false,
  onSelect,
  className,
}) => {
  const badge = FAMILY_BADGE[model.family] ?? FAMILY_BADGE['baseline'];
  const isSelectable = model.selectable;

  const handleClick = () => {
    if (isSelectable && onSelect) onSelect(model.id);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && isSelectable && onSelect) {
      e.preventDefault();
      onSelect(model.id);
    }
  };

  return (
    <div
      role={isSelectable ? 'button' : undefined}
      tabIndex={isSelectable ? 0 : undefined}
      aria-pressed={isSelectable ? active : undefined}
      aria-label={isSelectable ? `Select model ${model.name}` : model.name}
      onClick={isSelectable ? handleClick : undefined}
      onKeyDown={isSelectable ? handleKeyDown : undefined}
      className={[
        'relative bg-surface border border-border rounded-lg',
        'transition-colors duration-panel ease-out',
        variant === 'compact' ? 'p-3 w-[272px]' : 'p-4',
        active ? 'border-l-2 border-l-accent' : '',
        isSelectable ? 'cursor-pointer hover:border-accent/50' : 'opacity-50 cursor-not-allowed',
        className ?? '',
      ].join(' ')}
    >
      {/* BEST badge */}
      {isBest && (
        <span className="absolute top-2 right-2 px-1.5 py-0.5 rounded text-xs font-heading font-semibold bg-accent text-white">
          BEST
        </span>
      )}

      <div className="flex items-start gap-2 mb-2">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-heading font-semibold text-text truncate">{model.name}</h3>
          <span
            className={`inline-block mt-0.5 px-1.5 py-0.5 rounded text-xs font-body ${badge.classes}`}
          >
            {badge.label}
          </span>
        </div>
      </div>

      {(variant === 'expanded' || !active) && (
        <p className="text-xs font-body text-muted mb-2 line-clamp-2">{model.description}</p>
      )}

      {/* Metric pills */}
      <div className="flex flex-wrap gap-1">
        {model.diceCamus != null && <MetricPill label="CAMUS" value={model.diceCamus} />}
        {model.diceBrisc != null && <MetricPill label="BRISC" value={model.diceBrisc} />}
        {model.iou != null && <MetricPill label="IoU" value={model.iou} />}
        {model.hd != null && <MetricPill label="HD" value={model.hd} />}
      </div>
    </div>
  );
};

export default ModelCard;
