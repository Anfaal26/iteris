/**
 * LogoMark — glowing blue badge with a 3-bar equaliser glyph, as inline SVG.
 * Uses its own gradient fills, so it renders identically regardless of theme.
 */

import React, { useId } from 'react';

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
 * Inline SVG logomark: a glowing scanner-blue circular badge with three
 * white rounded bars (an equaliser/pulse glyph). Gradient and filter ids are
 * namespaced per-instance via useId so multiple copies (navbar + footer)
 * don't collide.
 */
export const LogoMark: React.FC<LogoMarkProps> = ({
  size = 32,
  ariaLabel = 'Iteris logo',
  className,
}) => {
  const uid = useId().replace(/:/g, '');
  const badgeGrad = `badgeGrad-${uid}`;
  const glowFilter = `glow-${uid}`;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      aria-label={ariaLabel}
      role="img"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id={badgeGrad} cx="35%" cy="28%" r="80%">
          <stop offset="0%" stopColor="#c8f4ff" />
          <stop offset="30%" stopColor="#38bdf8" />
          <stop offset="65%" stopColor="#0ea5e9" />
          <stop offset="100%" stopColor="#0a3d67" />
        </radialGradient>

        <filter id={glowFilter} x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="9" />
        </filter>
      </defs>

      {/* Ambient bloom behind the badge */}
      <circle cx="100" cy="100" r="90" fill="#38bdf8" opacity="0.45" filter={`url(#${glowFilter})`} />

      {/* Badge */}
      <circle cx="100" cy="100" r="82" fill={`url(#${badgeGrad})`} />
      <circle cx="100" cy="100" r="82" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="2" />

      {/* Equaliser bars */}
      <rect x="53" y="68" width="22" height="64" rx="11" fill="#ffffff" />
      <rect x="89" y="48" width="22" height="104" rx="11" fill="#ffffff" />
      <rect x="125" y="60" width="22" height="80" rx="11" fill="#ffffff" />
    </svg>
  );
};

export default LogoMark;
