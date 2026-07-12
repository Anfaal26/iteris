/**
 * Tailwind theme — every value maps to a CSS custom property in src/index.css.
 * No raw hex/px live here; this just exposes the canonical tokens as utility classes.
 * @type {import('tailwindcss').Config}
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        // Colours used anywhere with a Tailwind `/opacity` modifier (bg-accent/15,
        // border-border/50, text-landing-text/60, ...) MUST resolve through the
        // `rgb(var(--x-rgb) / <alpha-value>)` pattern — Tailwind can only inject
        // an alpha channel into a colour format it can parse at build time, and a
        // bare `var(--hex)` reference is opaque to it (the modifier silently
        // renders fully transparent otherwise). See the --*-rgb tokens in index.css.
        surface: 'rgb(var(--surface-rgb) / <alpha-value>)',
        'surface-2': 'var(--surface-2)',
        text: 'var(--text)',
        muted: 'rgb(var(--muted-rgb) / <alpha-value>)',
        border: 'rgb(var(--border-rgb) / <alpha-value>)',
        accent: 'rgb(var(--color-accent-rgb) / <alpha-value>)',
        success: 'rgb(var(--color-success-rgb) / <alpha-value>)',
        warning: 'rgb(var(--color-warning-rgb) / <alpha-value>)',
        error: 'rgb(var(--color-error-rgb) / <alpha-value>)',
        uncertainty: 'rgb(var(--color-uncertainty-rgb) / <alpha-value>)',
        'landing-bg': 'rgb(var(--color-landing-bg-rgb) / <alpha-value>)',
        'landing-text': 'rgb(var(--color-landing-text-rgb) / <alpha-value>)',
        'landing-footer': 'var(--color-landing-footer)',
        'grad-a': 'var(--color-gradient-a)',
        'grad-b': 'rgb(var(--color-gradient-b-rgb) / <alpha-value>)',
        'grad-c': 'var(--color-gradient-c)',
        mask: {
          'lv-endo': 'var(--mask-lv-endo)',
          'lv-epi': 'var(--mask-lv-epi)',
          la: 'var(--mask-la)',
          glioma: 'var(--mask-glioma)',
          meningioma: 'var(--mask-meningioma)',
          pituitary: 'var(--mask-pituitary)',
        },
      },
      fontFamily: {
        heading: 'var(--font-heading)',
        body: 'var(--font-body)',
        mono: 'var(--font-mono)',
      },
      transitionTimingFunction: { out: 'var(--ease-out)' },
      transitionDuration: {
        tooltip: 'var(--motion-tooltip)',
        panel: 'var(--motion-panel)',
        slide: 'var(--motion-slide)',
      },
      backgroundImage: {
        'iteris-gradient':
          'linear-gradient(90deg, var(--color-gradient-a), var(--color-gradient-b), var(--color-gradient-c))',
      },
      boxShadow: {
        card: 'var(--shadow-sm)',
        float: 'var(--shadow-md)',
      },
      spacing: {
        navbar: 'var(--navbar-height)',
        'panel-l': 'var(--control-panel-width)',
        'panel-r': 'var(--results-panel-width)',
        sidebar: 'var(--research-sidebar-width)',
      },
    },
  },
  plugins: [],
};
