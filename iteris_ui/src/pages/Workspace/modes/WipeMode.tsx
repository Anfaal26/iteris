/**
 * WipeMode — split wipe comparison between any two chosen sources
 * (Attention U-Net / GT / DRL). The two sides are resolved upstream in
 * ImageViewer from the sidebar's source chips, so this component just renders
 * whatever overlay set each side carries.
 */

import React, { useState } from 'react';
import { WipeDivider } from '@/components';

/** One overlay image to stack on the scan (a structure mask or the GT mask). */
export interface WipeOverlay {
  id: string;
  imageB64: string;
}

/** A named side of the wipe (left or right of the divider). */
export interface WipeSide {
  label: string;
  overlays: WipeOverlay[];
}

export interface WipeModeProps {
  anatomyLabel: string;
  imageB64?: string;
  left: WipeSide;
  right: WipeSide;
  overlayOpacity: number;
}

export const WipeMode: React.FC<WipeModeProps> = ({
  anatomyLabel,
  imageB64,
  left,
  right,
  overlayOpacity,
}) => {
  const [wipeValue, setWipeValue] = useState(50);

  const overlayImgs = (overlays: WipeOverlay[], side: string) =>
    overlays.map((o) => (
      <img
        key={`${side}-${o.id}`}
        src={o.imageB64}
        alt={`${side} ${o.id}`}
        className="absolute inset-0 w-full h-full object-fill"
        style={{ opacity: overlayOpacity, mixBlendMode: 'screen' }}
      />
    ));

  return (
    <div className="flex flex-col w-full h-full">
      <div className="relative flex-1 flex items-center justify-center bg-landing-bg overflow-hidden">
        <div className="relative aspect-square" style={{ width: 'min(85%, 80vh, 640px)' }}>
          {imageB64 ? (
            <img
              src={imageB64}
              alt={anatomyLabel}
              className="absolute inset-0 w-full h-full object-fill rounded-lg bg-[#2a2f3a]"
            />
          ) : (
            <svg width="100%" height="100%" viewBox="0 0 256 256" role="img" aria-label={`${anatomyLabel} wipe comparison`} className="rounded-lg">
              <rect width="256" height="256" fill="#2a2f3a" rx="8" />
              <text x="128" y="128" textAnchor="middle" dominantBaseline="middle" fill="#64748b" fontSize="12" fontFamily="system-ui">
                {anatomyLabel}
              </text>
            </svg>
          )}

          {/* Left side — clipped to the wipe */}
          <div className="absolute inset-0 overflow-hidden rounded-lg" style={{ clipPath: `inset(0 ${100 - wipeValue}% 0 0)` }}>
            {overlayImgs(left.overlays, 'left')}
          </div>

          {/* Right side — clipped to the wipe */}
          <div className="absolute inset-0 overflow-hidden rounded-lg" style={{ clipPath: `inset(0 0 0 ${wipeValue}%)` }}>
            {overlayImgs(right.overlays, 'right')}
          </div>

          <WipeDivider value={wipeValue} onChange={setWipeValue} height={256} />

          <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded text-xs font-mono bg-surface/80 text-muted">
            {left.label}
          </div>
          <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded text-xs font-mono bg-surface/80 text-accent">
            {right.label}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WipeMode;
