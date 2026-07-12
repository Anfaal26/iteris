/**
 * ThemeToggle — sun/moon icon + label button that switches between the
 * clinical dark (default) and light themes by toggling `data-theme="light"`
 * on `document.documentElement`.
 */

import React from 'react';
import { useTheme } from '@/components/theme/useTheme';

/** Props for ThemeToggle. */
export interface ThemeToggleProps {
  /** Additional class names. */
  className?: string;
}

/**
 * Button that switches between the dark (default) and light themes.
 * Uses the `useTheme` hook; no JS colour values are needed.
 */
export const ThemeToggle: React.FC<ThemeToggleProps> = ({ className }) => {
  const { theme, toggleTheme } = useTheme();
  const isLight = theme === 'light';

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isLight ? 'Switch to dark theme' : 'Switch to light theme'}
      aria-pressed={isLight}
      className={[
        'flex items-center gap-1.5 px-2 py-1 rounded-md',
        'text-muted hover:text-text transition-colors duration-panel ease-out',
        'text-sm font-body select-none',
        className ?? '',
      ].join(' ')}
    >
      {isLight ? (
        /* Sun — currently light, offers switching to dark */
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
        >
          <circle cx="8" cy="8" r="3" />
          <path d="M8 1v1.5M8 13.5V15M15 8h-1.5M2.5 8H1M12.72 3.28l-1.06 1.06M4.34 11.66l-1.06 1.06M12.72 12.72l-1.06-1.06M4.34 4.34 3.28 3.28" />
        </svg>
      ) : (
        /* Moon — currently dark, offers switching to light */
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="currentColor"
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M14.5 9.54A6.5 6.5 0 0 1 6.46 1.5a.5.5 0 0 0-.6-.63A7 7 0 1 0 15.13 10.14a.5.5 0 0 0-.63-.6z" />
        </svg>
      )}
      <span>{isLight ? 'Light' : 'Dark'}</span>
    </button>
  );
};

export default ThemeToggle;
