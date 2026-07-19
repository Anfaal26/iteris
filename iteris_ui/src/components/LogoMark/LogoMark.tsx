/**
 * LogoMark — radial geometric mark as inline SVG.
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
 * Inline SVG logomark: rotated interlocking petals around a central sparkle,
 * framed by concentric rings. Gradient ids are namespaced per-instance via
 * useId so multiple copies (e.g. navbar + favicon preview) don't collide.
 */
export const LogoMark: React.FC<LogoMarkProps> = ({
  size = 32,
  ariaLabel = 'Iteris logo',
  className,
}) => {
  const uid = useId().replace(/:/g, '');
  const outerLoopGrad = `outerLoopGrad-${uid}`;
  const innerLoopGrad = `innerLoopGrad-${uid}`;
  const geminiCore = `geminiCore-${uid}`;
  const petal = `geometric-petal-${uid}`;
  const ray = `sharp-ray-${uid}`;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 800 800"
      aria-label={ariaLabel}
      role="img"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id={outerLoopGrad} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00F2FE" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#0072ff" stopOpacity="0.85" />
        </linearGradient>

        <linearGradient id={innerLoopGrad} x1="100%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#38ef7d" stopOpacity="0.2" />
          <stop offset="50%" stopColor="#00f2fe" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#4facfe" stopOpacity="0.7" />
        </linearGradient>

        <linearGradient id={geminiCore} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="40%" stopColor="#d4f1f9" />
          <stop offset="100%" stopColor="#00c6ff" />
        </linearGradient>

        <g id={petal}>
          <path d="M 400,400 C 460,240 590,230 590,360 C 590,470 470,440 400,400 Z" fill={`url(#${outerLoopGrad})`} />
          <path d="M 400,400 C 440,280 530,270 530,360 C 530,420 450,420 400,400 Z" fill={`url(#${innerLoopGrad})`} />
        </g>

        <g id={ray}>
          <path d="M 400,400 L 415,260 L 400,160 L 385,260 Z" fill="#00F2FE" opacity="0.4" />
        </g>
      </defs>

      {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => (
        <g key={`petal-${deg}`} transform={`rotate(${deg} 400 400)`}>
          <use href={`#${petal}`} />
        </g>
      ))}

      {[22.5, 112.5, 202.5, 292.5].map((deg) => (
        <g key={`petal-mid-${deg}`} transform={`rotate(${deg} 400 400)`} opacity="0.5">
          <use href={`#${petal}`} />
        </g>
      ))}

      {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => (
        <g key={`ray-${deg}`} transform={`rotate(${deg} 400 400)`}>
          <use href={`#${ray}`} />
        </g>
      ))}

      <path
        d="M 400,260
           Q 400,400 260,400
           Q 400,400 400,540
           Q 400,400 540,400
           Q 400,400 400,260 Z"
        fill={`url(#${geminiCore})`}
      />

      <circle cx="400" cy="400" r="40" fill="none" stroke="#ffffff" strokeWidth="3" opacity="0.7" />
      <circle cx="400" cy="400" r="15" fill="#ffffff" />
    </svg>
  );
};

export default LogoMark;
