/**
 * SampleImageTile — thumbnail with modality, anatomy, and difficulty badges.
 * When `thumbnailB64` is empty, renders a procedural SVG placeholder.
 * Driven by `SampleImage` from the API contract.
 */

import React from 'react';
import type { SampleImage } from '@/api/contract';

/** Props for the SampleImageTile component. */
export interface SampleImageTileProps {
  /** Sample image data from the API. */
  image: SampleImage;
  /** Called when the tile is selected. */
  onSelect?: (id: string) => void;
  /** Whether this tile is currently selected. */
  selected?: boolean;
  /** Additional class names. */
  className?: string;
}

const DIFFICULTY_BADGE: Record<SampleImage['difficulty'], string> = {
  easy: 'bg-success/20 text-success',
  medium: 'bg-warning/20 text-warning',
  hard: 'bg-error/20 text-error',
};

/**
 * Tile component for the sample image browser.
 * Falls back to a procedural SVG grid placeholder when no thumbnail is provided.
 */
export const SampleImageTile: React.FC<SampleImageTileProps> = ({
  image,
  onSelect,
  selected = false,
  className,
}) => {
  const hasThumbnail = image.thumbnailB64.length > 0;

  const handleClick = () => onSelect?.(image.id);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect?.(image.id); }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`${image.anatomy} ${image.modality} sample, ${image.difficulty} difficulty`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={[
        'relative rounded-lg overflow-hidden border cursor-pointer',
        'transition-all duration-panel ease-out',
        selected ? 'border-accent ring-1 ring-accent' : 'border-border hover:border-accent/50',
        className ?? '',
      ].join(' ')}
    >
      {/* Thumbnail / placeholder */}
      <div className="aspect-square bg-bg">
        {hasThumbnail ? (
          <img
            src={`data:image/png;base64,${image.thumbnailB64}`}
            alt={`${image.anatomy} ${image.modality} scan`}
            className="w-full h-full object-cover"
          />
        ) : (
          <svg
            viewBox="0 0 64 64"
            className="w-full h-full"
            aria-label={`Placeholder for ${image.anatomy} scan`}
            role="img"
          >
            {/* Procedural grid placeholder */}
            <rect width="64" height="64" fill="var(--border)" />
            <line x1="0" y1="16" x2="64" y2="16" stroke="var(--muted)" strokeWidth="0.5" opacity="0.4" />
            <line x1="0" y1="32" x2="64" y2="32" stroke="var(--muted)" strokeWidth="0.5" opacity="0.4" />
            <line x1="0" y1="48" x2="64" y2="48" stroke="var(--muted)" strokeWidth="0.5" opacity="0.4" />
            <line x1="16" y1="0" x2="16" y2="64" stroke="var(--muted)" strokeWidth="0.5" opacity="0.4" />
            <line x1="32" y1="0" x2="32" y2="64" stroke="var(--muted)" strokeWidth="0.5" opacity="0.4" />
            <line x1="48" y1="0" x2="48" y2="64" stroke="var(--muted)" strokeWidth="0.5" opacity="0.4" />
            {/* Simple body silhouette hint */}
            <ellipse cx="32" cy="26" rx="10" ry="12" fill="none" stroke="var(--muted)" strokeWidth="1" opacity="0.4" />
            <rect x="24" y="38" width="16" height="16" rx="2" fill="none" stroke="var(--muted)" strokeWidth="1" opacity="0.4" />
          </svg>
        )}
      </div>

      {/* Badge overlay */}
      <div className="absolute bottom-0 inset-x-0 p-1.5 flex flex-wrap gap-1 bg-gradient-to-t from-landing-bg/60 to-transparent">
        <span className="px-1 py-0.5 rounded text-xs font-body bg-surface/80 text-muted capitalize">
          {image.modality}
        </span>
        <span className="px-1 py-0.5 rounded text-xs font-body bg-surface/80 text-muted capitalize">
          {image.anatomy}
        </span>
        <span
          className={`px-1 py-0.5 rounded text-xs font-body capitalize ${DIFFICULTY_BADGE[image.difficulty]}`}
        >
          {image.difficulty}
        </span>
      </div>
    </div>
  );
};

export default SampleImageTile;
