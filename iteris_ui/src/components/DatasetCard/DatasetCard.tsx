/**
 * DatasetCard — expanded card showing dataset info with best-Dice badge.
 */

import React from 'react';
import type { DatasetId, Modality } from '@/api/contract';

/** Props for the DatasetCard component. */
export interface DatasetCardProps {
  /** Dataset identifier. */
  datasetId: DatasetId;
  /** Display name. */
  name: string;
  /** Imaging modality. */
  modality: Modality;
  /** Short description of the dataset. */
  description: string;
  /** Number of samples. */
  sampleCount: number;
  /** Best Dice achieved across all evaluated models. */
  bestDice: number;
  /** Whether this dataset is currently selected. */
  selected?: boolean;
  /** Called when the card is activated. */
  onSelect?: (id: DatasetId) => void;
  /** Additional class names. */
  className?: string;
}

const MODALITY_BADGE: Record<Modality, string> = {
  ultrasound: 'bg-accent/10 text-accent',
  mri: 'bg-uncertainty/10 text-uncertainty',
};

/**
 * Card component for the dataset browser / selection panel.
 * Shows modality, sample count, description, and best-Dice badge.
 */
export const DatasetCard: React.FC<DatasetCardProps> = ({
  datasetId,
  name,
  modality,
  description,
  sampleCount,
  bestDice,
  selected = false,
  onSelect,
  className,
}) => {
  const handleClick = () => onSelect?.(datasetId);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect?.(datasetId); }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`Select ${name} dataset`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={[
        'relative bg-surface border rounded-xl p-4 cursor-pointer',
        'transition-all duration-panel ease-out',
        selected ? 'border-accent ring-1 ring-accent' : 'border-border hover:border-accent/50',
        className ?? '',
      ].join(' ')}
    >
      {/* Best Dice badge */}
      <span className="absolute top-3 right-3 px-2 py-0.5 rounded text-xs font-heading font-semibold bg-success/15 text-success">
        Best Dice {bestDice.toFixed(3)}
      </span>

      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-heading font-semibold text-text">{name}</h3>
        <span className={`px-1.5 py-0.5 rounded text-xs font-body capitalize ${MODALITY_BADGE[modality]}`}>
          {modality}
        </span>
      </div>

      <p className="text-xs font-body text-muted mb-3 line-clamp-2">{description}</p>

      <div className="flex items-center gap-3 text-xs font-mono text-muted">
        <span>{sampleCount.toLocaleString()} samples</span>
        <span className="uppercase tracking-wide font-body">{datasetId.toUpperCase()}</span>
      </div>
    </div>
  );
};

export default DatasetCard;
