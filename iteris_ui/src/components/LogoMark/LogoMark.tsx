/**
 * LogoMark — the Iteris glyph: three rounded bars of staggered height in a
 * vertical scanner-blue gradient, on a transparent ground.
 *
 * Deliberately badge-less. The earlier mark set white bars inside a glowing
 * circular badge, which read as hollow at navbar and favicon sizes — the bars
 * themselves are the mark, so they carry the gradient directly.
 *
 * Keep this in sync with public/favicon.svg: same geometry, standalone file
 * (a browser tab can't render a React component).
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

/** Bar geometry in the 200×200 viewBox: [x, y, height]. Width/radius shared. */
const BARS: [number, number, number][] = [
  [50, 50, 118],
  [88, 49, 91],
  [126, 24, 130],
];
const BAR_W = 24;

/**
 * Inline SVG logomark. The gradient id is namespaced per-instance via useId so
 * multiple copies (navbar + footer) don't collide in the same document.
 */
export const LogoMark: React.FC<LogoMarkProps> = ({
  size = 32,
  ariaLabel = 'Iteris logo',
  className,
}) => {
  const uid = useId().replace(/:/g, '');
  const barGrad = `barGrad-${uid}`;

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
        <linearGradient id={barGrad} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00b5f2" />
          <stop offset="100%" stopColor="#0a7cf0" />
        </linearGradient>
      </defs>

      {BARS.map(([x, y, h]) => (
        <rect
          key={x}
          x={x}
          y={y}
          width={BAR_W}
          height={h}
          rx={BAR_W / 2}
          fill={`url(#${barGrad})`}
        />
      ))}
    </svg>
  );
};

export default LogoMark;
