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
        surface: 'var(--surface)',
        text: 'var(--text)',
        muted: 'var(--muted)',
        border: 'var(--border)',
        accent: 'var(--color-accent)',
        success: 'var(--color-success)',
        warning: 'var(--color-warning)',
        error: 'var(--color-error)',
        uncertainty: 'var(--color-uncertainty)',
        'landing-bg': 'var(--color-landing-bg)',
        'landing-text': 'var(--color-landing-text)',
        'landing-footer': 'var(--color-landing-footer)',
        'grad-a': 'var(--color-gradient-a)',
        'grad-b': 'var(--color-gradient-b)',
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
