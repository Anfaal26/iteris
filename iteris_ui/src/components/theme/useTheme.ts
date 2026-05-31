/**
 * useTheme — reads and sets the active ITERIS theme by toggling
 * `data-theme="reading-room"` on `document.documentElement`.
 *
 * The CSS in src/index.css re-maps all `--bg`, `--surface`, etc. tokens when
 * the attribute is present, so no JS colour values are needed here.
 */

import { useState, useEffect, useCallback } from 'react';
import type { ThemeName } from '@/tokens';

/** Return shape for the useTheme hook. */
export interface UseThemeReturn {
  /** Current active theme name. */
  theme: ThemeName;
  /** Toggle between 'clinical' and 'reading-room'. */
  toggleTheme: () => void;
  /** Directly set the theme. */
  setTheme: (t: ThemeName) => void;
}

/**
 * Hook for reading and writing the document-level ITERIS theme.
 * The theme is persisted to `localStorage` key `iteris-theme`.
 */
export function useTheme(): UseThemeReturn {
  const [theme, setThemeState] = useState<ThemeName>(() => {
    const stored = typeof window !== 'undefined'
      ? (localStorage.getItem('iteris-theme') as ThemeName | null)
      : null;
    return stored ?? 'clinical';
  });

  useEffect(() => {
    if (theme === 'reading-room') {
      document.documentElement.setAttribute('data-theme', 'reading-room');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    localStorage.setItem('iteris-theme', theme);
  }, [theme]);

  const setTheme = useCallback((t: ThemeName) => setThemeState(t), []);

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => (prev === 'clinical' ? 'reading-room' : 'clinical'));
  }, []);

  return { theme, toggleTheme, setTheme };
}
