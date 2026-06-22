/**
 * SingleMode — single model viewer with canvas placeholder and mask overlays.
 */

import React from 'react';
import type { MaskLayer } from '@/api/contract';

/** Props for SingleMode. */
export interface SingleModeProps {
  anatomyLabel: string;
  imageB64?: string;
  masks: MaskLayer[];
  visibleStructures: Set<string>;
  overlayOpacity: number;
  windowLevel: number;
  windowWidth: number;
}

/**
 * Single-model canvas view. Renders a grey placeholder with anatomy label,
 * then overlays mask images from inference results.
 */
export const SingleMode: React.FC<SingleModeProps> = ({
  anatomyLabel,
  imageB64,
  masks,
  visibleStructures,
  overlayOpacity,
  windowLevel,
  windowWidth,
}) => {
  // Simulated W/L: maps to brightness/contrast CSS filter
  const brightness = 0.6 + (windowLevel / 400) * 0.8;
  const contrast = 0.8 + (windowWidth / 400) * 0.4;

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-landing-bg overflow-hidden">
      {/* Canvas placeholder */}
      <div
        className="relative aspect-square"
        style={{
          width: 'min(85%, 85vh, 640px)',
          filter: `brightness(${brightness}) contrast(${contrast})`,
        }}
      >
        {/* Base image — real scan if available, grey placeholder otherwise */}
        {imageB64 ? (
          <img
            src={imageB64}
            alt={anatomyLabel}
            className="absolute inset-0 w-full h-full object-fill rounded-lg bg-[#2a2f3a]"
          />
        ) : (
          <svg
            width="100%"
            height="100%"
            viewBox="0 0 256 256"
            aria-label={`${anatomyLabel} scan placeholder`}
            role="img"
            className="rounded-lg"
          >
            <rect width="256" height="256" fill="#2a2f3a" rx="8" />
            {[64, 128, 192].map((v) => (
              <React.Fragment key={v}>
                <line x1={v} y1="0" x2={v} y2="256" stroke="#3a3f4a" strokeWidth="0.5" />
                <line x1="0" y1={v} x2="256" y2={v} stroke="#3a3f4a" strokeWidth="0.5" />
              </React.Fragment>
            ))}
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
        )}

        {/* Mask overlays */}
        {masks.map((mask) =>
          visibleStructures.has(mask.structure) ? (
            <img
              key={mask.structure}
              src={mask.imageB64}
              alt={`${mask.label} segmentation mask`}
              className="absolute inset-0 w-full h-full rounded-lg"
              style={{ opacity: overlayOpacity, mixBlendMode: 'screen' }}
            />
          ) : null,
        )}
      </div>
    </div>
  );
};

export default SingleMode;
