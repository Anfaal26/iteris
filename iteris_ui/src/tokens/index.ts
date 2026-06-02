/**
 * Typed mirror of the canonical CSS custom properties (src/index.css, spec §10).
 *
 * Components that need a token *value in JS* (e.g. Three.js materials, canvas
 * fill styles, recharts colours) import from here instead of writing literals.
 * Each entry resolves to the corresponding CSS variable so the CSS file stays
 * the single source of truth — change a colour there and it flows everywhere.
 */

const cssVar = (name: string): string => `var(${name})`;

/** Brand + clinical theme colours. */
export const colors = {
  bg: cssVar('--bg'),
  surface: cssVar('--surface'),
  text: cssVar('--text'),
  muted: cssVar('--muted'),
  border: cssVar('--border'),
  accent: cssVar('--color-accent'),
  success: cssVar('--color-success'),
  warning: cssVar('--color-warning'),
  error: cssVar('--color-error'),
  uncertainty: cssVar('--color-uncertainty'),
  landingBg: cssVar('--color-landing-bg'),
  landingText: cssVar('--color-landing-text'),
  gradientA: cssVar('--color-gradient-a'),
  gradientB: cssVar('--color-gradient-b'),
  gradientC: cssVar('--color-gradient-c'),
} as const;

/** Raw hex values for contexts that cannot resolve CSS vars (e.g. WebGL on canvas). */
export const colorsHex = {
  landingBg: '#030508',
  gradientA: '#bae6fd',   /* ice-white / MRI highlight  */
  gradientB: '#38bdf8',   /* scanner blue               */
  gradientC: '#0ea5e9',   /* deep imaging blue          */
  accent: '#0a7ea4',
} as const;

/** Mask overlay colours — reserved, never used in UI chrome (spec §10). */
export const maskColors = {
  lvEndo: cssVar('--mask-lv-endo'),
  lvEpi: cssVar('--mask-lv-epi'),
  la: cssVar('--mask-la'),
  glioma: cssVar('--mask-glioma'),
  meningioma: cssVar('--mask-meningioma'),
  pituitary: cssVar('--mask-pituitary'),
} as const;

/** Hex mask values for canvas compositing. */
export const maskColorsHex = {
  lvEndo: '#00c9a7',
  lvEpi: '#f59e0b',
  la: '#f87171',
  glioma: '#818cf8',
  meningioma: '#34d399',
  pituitary: '#fb923c',
} as const;

export const fonts = {
  heading: cssVar('--font-heading'),
  body: cssVar('--font-body'),
  mono: cssVar('--font-mono'),
} as const;

/** Motion durations in ms (numeric) — for JS-driven animation/timeouts. */
export const motion = {
  tooltipMs: 100,
  panelMs: 150,
  slideMs: 200,
  /** Landing Three.js scene target frame rate (spec §4). */
  landingFps: 30,
  /** Staggered entry delay between hero elements (spec §4). */
  staggerMs: 150,
  easeOut: 'cubic-bezier(0.16, 1, 0.3, 1)',
} as const;

/** Structure → mask colour map keyed by the structure ids used in the API contract. */
export const structureColor: Record<string, string> = {
  lv_endo: maskColors.lvEndo,
  lv_epi: maskColors.lvEpi,
  la: maskColors.la,
  glioma: maskColors.glioma,
  meningioma: maskColors.meningioma,
  pituitary: maskColors.pituitary,
};

export type ThemeName = 'clinical' | 'reading-room';
