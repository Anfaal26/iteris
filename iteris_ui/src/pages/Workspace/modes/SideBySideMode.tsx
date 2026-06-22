/**
 * SideBySideMode — three equal columns, each showing a different model's result
 * with a BEST badge on the highest-Dice column.
 */

import React from 'react';
import type { CompareResult } from '@/api/contract';

/** Props for SideBySideMode. */
export interface SideBySideModeProps {
  anatomyLabel: string;
  imageB64?: string;
  results: CompareResult[];
  visibleStructures: Set<string>;
  overlayOpacity: number;
}

/** Three-column side-by-side comparison view. */
export const SideBySideMode: React.FC<SideBySideModeProps> = ({
  anatomyLabel,
  imageB64,
  results,
  visibleStructures,
  overlayOpacity,
}) => {
  const bestIdx = results.reduce(
    (best, r, i) => (r.metrics.dice > results[best].metrics.dice ? i : best),
    0,
  );

  if (results.length === 0) {
    return (
      <div className="flex items-center justify-center w-full h-full bg-landing-bg">
        <p className="text-sm font-body text-muted">Run inference to compare models.</p>
      </div>
    );
  }

  return (
    <div className="flex w-full h-full bg-landing-bg gap-2 p-4">
      {results.map((r, idx) => (
        <div
          key={r.modelId}
          className={[
            'flex flex-col flex-1 items-center gap-2',
            idx === bestIdx ? 'ring-1 ring-accent rounded-lg p-1' : '',
          ].join(' ')}
        >
          {/* Model label + BEST badge */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-muted">{r.modelId}</span>
            {idx === bestIdx && (
              <span className="px-1.5 py-0.5 rounded text-xs font-heading font-semibold bg-accent text-white">
                BEST
              </span>
            )}
          </div>

          {/* Canvas + masks */}
          <div className="relative aspect-square w-full max-w-[320px]">
            {imageB64 ? (
              <img
                src={imageB64}
                alt={`${anatomyLabel} — ${r.modelId}`}
                className="absolute inset-0 w-full h-full object-fill rounded-lg bg-[#2a2f3a]"
              />
            ) : (
              <svg
                width="100%"
                height="100%"
                viewBox="0 0 256 256"
                aria-label={`${anatomyLabel} — ${r.modelId}`}
                role="img"
                className="rounded-lg"
              >
                <rect width="256" height="256" fill="#2a2f3a" rx="8" />
                <text
                  x="128"
                  y="128"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="#64748b"
                  fontSize="14"
                  fontFamily="system-ui"
                >
                  {anatomyLabel}
                </text>
              </svg>
            )}

            {r.masks.map((mask) =>
              visibleStructures.has(mask.structure) ? (
                <img
                  key={mask.structure}
                  src={mask.imageB64}
                  alt={`${mask.label} mask for ${r.modelId}`}
                  className="absolute inset-0 w-full h-full rounded-lg"
                  style={{ opacity: overlayOpacity, mixBlendMode: 'screen' }}
                />
              ) : null,
            )}
          </div>

          {/* Dice */}
          <p className="text-xs font-mono text-text">Dice: {r.metrics.dice.toFixed(3)}</p>
        </div>
      ))}
    </div>
  );
};

export default SideBySideMode;
