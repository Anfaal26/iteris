/**
 * ReadingRoomToggle — moon icon + label button that toggles the reading-room
 * dark theme by setting `data-theme="reading-room"` on `document.documentElement`.
 */

import React from 'react';
import { useTheme } from '@/components/theme/useTheme';

/** Props for ReadingRoomToggle. */
export interface ReadingRoomToggleProps {
  /** Additional class names. */
  className?: string;
}

/**
 * Button that switches between clinical (light) and reading-room (dark) themes.
 * Uses the `useTheme` hook; no JS colour values are needed.
 */
export const ReadingRoomToggle: React.FC<ReadingRoomToggleProps> = ({ className }) => {
  const { theme, toggleTheme } = useTheme();
  const isReadingRoom = theme === 'reading-room';

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isReadingRoom ? 'Switch to clinical theme' : 'Switch to reading-room theme'}
      aria-pressed={isReadingRoom}
      className={[
        'flex items-center gap-1.5 px-2 py-1 rounded-md',
        'text-muted hover:text-text transition-colors duration-panel ease-out',
        'text-sm font-body select-none',
        className ?? '',
      ].join(' ')}
    >
      {/* Moon SVG — inline, no external asset dependency */}
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
      <span>{isReadingRoom ? 'Clinical' : 'Reading Room'}</span>
    </button>
  );
};

export default ReadingRoomToggle;
