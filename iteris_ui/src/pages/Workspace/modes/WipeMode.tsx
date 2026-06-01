/**
 * WipeMode — split wipe comparison between U-Net baseline and selected DRL model.
 */

import React, { useState } from 'react';
import { WipeDivider } from '@/components';
import type { MaskLayer, Metrics } from '@/api/contract';

/** Props for WipeMode. */
export interface WipeModeProps {
  anatomyLabel: string;
  baselineMasks: MaskLayer[];
  drlMasks: MaskLayer[];
  visibleStructures: Set<string>;
  overlayOpacity: number;
  baselineMetrics?: Metrics;
  drlMetrics?: Metrics;
}

/** Wipe comparison view — left half baseline, right half DRL model. */
export const WipeMode: React.FC<WipeModeProps> = ({
  anatomyLabel,
  baselineMasks,
  drlMasks,
  visibleStructures,
  overlayOpacity,
  baselineMetrics,
  drlMetrics,
}) => {
  const [wipeValue, setWipeValue] = useState(50);

  const diceDelta =
    drlMetrics && baselineMetrics
      ? (drlMetrics.dice - baselineMetrics.dice).toFixed(3)
      : null;

  return (
    <div className="flex flex-col w-full h-full">
      {/* Wipe viewer */}
      <div className="relative flex-1 flex items-center justify-center bg-landing-bg overflow-hidden">
        <div className="relative" style={{ width: 256, height: 256 }}>
          {/* Base canvas */}
          <svg
            width="256"
            height="256"
            viewBox="0 0 256 256"
            aria-label={`${anatomyLabel} wipe comparison`}
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
              fontSize="12"
              fontFamily="system-ui"
            >
              {anatomyLabel}
            </text>
          </svg>

          {/* Left half — baseline masks clipped */}
          <div
            className="absolute inset-0 overflow-hidden rounded-lg"
            style={{ clipPath: `inset(0 ${100 - wipeValue}% 0 0)` }}
          >
            {baselineMasks.map((mask) =>
              visibleStructures.has(mask.structure) ? (
                <img
                  key={mask.structure}
                  src={mask.imageB64}
                  alt={`Baseline ${mask.label} mask`}
                  className="absolute inset-0 w-full h-full"
                  style={{ opacity: overlayOpacity, mixBlendMode: 'screen' }}
                />
              ) : null,
            )}
          </div>

          {/* Right half — DRL masks clipped */}
          <div
            className="absolute inset-0 overflow-hidden rounded-lg"
            style={{ clipPath: `inset(0 0 0 ${wipeValue}%)` }}
          >
            {drlMasks.map((mask) =>
              visibleStructures.has(mask.structure) ? (
                <img
                  key={mask.structure}
                  src={mask.imageB64}
                  alt={`DRL ${mask.label} mask`}
                  className="absolute inset-0 w-full h-full"
                  style={{ opacity: overlayOpacity, mixBlendMode: 'screen' }}
                />
              ) : null,
            )}
          </div>

          {/* Wipe divider */}
          <WipeDivider value={wipeValue} onChange={setWipeValue} height={256} />

          {/* Labels */}
          <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded text-xs font-mono bg-surface/80 text-muted">
            Baseline
          </div>
          <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded text-xs font-mono bg-surface/80 text-accent">
            DRL
          </div>
        </div>
      </div>

      {/* Metrics delta */}
      {diceDelta !== null && (
        <div className="flex items-center justify-center gap-4 py-3 border-t border-border bg-surface">
          <div className="text-center">
            <p className="text-xs font-body text-muted">Baseline Dice</p>
            <p className="text-sm font-mono text-text">{baselineMetrics!.dice.toFixed(3)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-body text-muted">DRL Dice</p>
            <p className="text-sm font-mono text-text">{drlMetrics!.dice.toFixed(3)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-body text-muted">Δ Dice</p>
            <p
              className={[
                'text-sm font-mono',
                parseFloat(diceDelta) >= 0 ? 'text-success' : 'text-error',
              ].join(' ')}
            >
              {parseFloat(diceDelta) >= 0 ? '+' : ''}{diceDelta}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default WipeMode;
