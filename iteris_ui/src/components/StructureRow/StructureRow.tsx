/**
 * StructureRow — one row in the per-structure results panel.
 * Shows colour swatch, structure name, Dice score, and HD.
 * Driven by `StructureMetrics` from the API contract.
 */

import React from 'react';
import type { StructureMetrics } from '@/api/contract';
import { structureColor } from '@/tokens';

/** Props for the StructureRow component. */
export interface StructureRowProps {
  /** Structure metrics data from the API. */
  metrics: StructureMetrics;
  /**
   * When true, renders a hatch pattern overlay on the swatch for colour-blind users.
   * @default false
   */
  hatchSwatch?: boolean;
  /** Additional class names. */
  className?: string;
}

/**
 * A single row showing mask colour swatch, structure label, Dice and HD metrics.
 * Includes optional hatch pattern on the swatch for colour-blind accessibility (spec §10).
 */
export const StructureRow: React.FC<StructureRowProps> = ({
  metrics,
  hatchSwatch = false,
  className,
}) => {
  const color = structureColor[metrics.structure] ?? 'var(--color-muted)';
  const swatchId = `hatch-${metrics.structure}`;

  return (
    <div
      className={[
        'flex items-center gap-3 py-1.5',
        className ?? '',
      ].join(' ')}
      role="row"
    >
      {/* Colour swatch with optional hatch overlay */}
      <span
        aria-hidden="true"
        className="relative flex-shrink-0 w-3 h-3 rounded-sm overflow-hidden"
        style={{ backgroundColor: color }}
      >
        {hatchSwatch && (
          <svg
            viewBox="0 0 4 4"
            width="12"
            height="12"
            className="absolute inset-0"
            aria-hidden="true"
          >
            <defs>
              <pattern id={swatchId} patternUnits="userSpaceOnUse" width="4" height="4">
                <line x1="0" y1="4" x2="4" y2="0" stroke="rgba(255,255,255,0.6)" strokeWidth="0.8" />
              </pattern>
            </defs>
            <rect width="4" height="4" fill={`url(#${swatchId})`} />
          </svg>
        )}
      </span>

      {/* Structure label */}
      <span className="flex-1 text-sm font-body text-text truncate">{metrics.label}</span>

      {/* Dice */}
      <div className="flex flex-col items-end">
        <span className="text-xs text-muted font-body">Dice</span>
        <span className="text-sm font-mono text-text">{metrics.dice.toFixed(3)}</span>
      </div>

      {/* HD */}
      <div className="flex flex-col items-end">
        <span className="text-xs text-muted font-body">HD</span>
        <span className="text-sm font-mono text-text">{metrics.hd.toFixed(1)}</span>
      </div>
    </div>
  );
};

export default StructureRow;
