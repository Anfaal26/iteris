/**
 * LogoMark — nested-diamond-eye motif as inline SVG.
 * Uses `currentColor` so it inherits any text colour from its parent.
 */

import React from 'react';

/** Props for the LogoMark component. */
export interface LogoMarkProps {
  /** Width/height of the square SVG in pixels. @default 32 */
  size?: number;
  /** Accessible label for screen readers. @default "Iteris logo" */
  ariaLabel?: string;
  /** Additional class names. */
  className?: string;
}

/**
 * Inline SVG logomark: two concentric diamond frames with an eye pupil centre.
 * Renders in `currentColor`; wrap in a coloured element to tint it.
 */
export const LogoMark: React.FC<LogoMarkProps> = ({
  size = 32,
  ariaLabel = 'Iteris logo',
  className,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 32 32"
    fill="none"
    aria-label={ariaLabel}
    role="img"
    className={className}
    xmlns="http://www.w3.org/2000/svg"
  >
    {/* Outer diamond frame */}
    <polygon
      points="16,2 30,16 16,30 2,16"
      stroke="currentColor"
      strokeWidth="1.5"
      fill="none"
    />
    {/* Inner diamond frame */}
    <polygon
      points="16,7 25,16 16,25 7,16"
      stroke="currentColor"
      strokeWidth="1.5"
      fill="none"
    />
    {/* Eye iris */}
    <ellipse
      cx="16"
      cy="16"
      rx="5"
      ry="5"
      stroke="currentColor"
      strokeWidth="1.5"
      fill="none"
    />
    {/* Pupil */}
    <circle cx="16" cy="16" r="2" fill="currentColor" />
    {/* Eye corners (diamond → oval lens) */}
    <line x1="7" y1="16" x2="11" y2="16" stroke="currentColor" strokeWidth="1.5" />
    <line x1="21" y1="16" x2="25" y2="16" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

export default LogoMark;
