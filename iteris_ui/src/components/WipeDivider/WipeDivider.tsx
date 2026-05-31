/**
 * WipeDivider — a draggable vertical 50%-divider for split-view wipe comparison.
 * 2px white line + circular drag handle + shadow.
 * Double-click snaps to centre. Keyboard-controllable (←/→, 10% increments).
 * Controlled: consumer provides `value` (0-100) and `onChange`.
 */

import React, { useRef, useCallback, useEffect, useState } from 'react';

/** Props for the WipeDivider component. */
export interface WipeDividerProps {
  /** Current position as percentage 0–100. @default 50 */
  value?: number;
  /** Called when the user drags or snaps the divider. */
  onChange?: (value: number) => void;
  /**
   * When true, show the sync indicator label.
   * @default false
   */
  synced?: boolean;
  /** Height of the wipe viewport in px; needed for the line. */
  height?: string | number;
  /** Additional class names on the container element. */
  className?: string;
}

const STEP = 10;

/**
 * Vertical wipe divider for image comparison.
 * - Drag handle repositions the divider.
 * - Double-click snaps to 50%.
 * - Arrow keys move by ±10%.
 */
export const WipeDivider: React.FC<WipeDividerProps> = ({
  value = 50,
  onChange,
  synced = false,
  height = '100%',
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [isDragging, setIsDragging] = useState(false);

  const clamp = (v: number) => Math.max(0, Math.min(100, v));

  const positionFromEvent = useCallback(
    (clientX: number): number => {
      const rect = containerRef.current?.parentElement?.getBoundingClientRect();
      if (!rect) return value;
      return clamp(((clientX - rect.left) / rect.width) * 100);
    },
    [value],
  );

  // Mouse move / up listeners attached to window during drag
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      onChange?.(positionFromEvent(e.clientX));
    };
    const onMouseUp = () => {
      if (dragging.current) {
        dragging.current = false;
        setIsDragging(false);
      }
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [onChange, positionFromEvent]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    setIsDragging(true);
  };

  const handleDoubleClick = () => onChange?.(50);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); onChange?.(clamp(value - STEP)); }
    if (e.key === 'ArrowRight') { e.preventDefault(); onChange?.(clamp(value + STEP)); }
    if (e.key === 'Home') { e.preventDefault(); onChange?.(0); }
    if (e.key === 'End') { e.preventDefault(); onChange?.(100); }
  };

  return (
    <div
      ref={containerRef}
      className={['absolute top-0 bottom-0 z-10 flex flex-col items-center', className ?? ''].join(' ')}
      style={{ left: `${value}%`, transform: 'translateX(-50%)' }}
    >
      {/* Vertical line */}
      <div
        aria-hidden="true"
        className="w-0.5 bg-white shadow-lg"
        style={{ height: typeof height === 'number' ? `${height}px` : height }}
      />

      {/* Drag handle */}
      <button
        type="button"
        aria-label={`Wipe divider at ${Math.round(value)}%. Drag or use arrow keys to move.`}
        aria-valuenow={Math.round(value)}
        aria-valuemin={0}
        aria-valuemax={100}
        role="slider"
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        onKeyDown={handleKeyDown}
        className={[
          'absolute top-1/2 -translate-y-1/2',
          'w-8 h-8 rounded-full bg-white shadow-lg border-2 border-border',
          'flex items-center justify-center cursor-grab',
          isDragging ? 'cursor-grabbing' : '',
          'transition-transform duration-tooltip ease-out',
          'hover:scale-110 focus-visible:scale-110',
        ].join(' ')}
        style={{
          boxShadow: '0 0 0 2px var(--color-accent), 0 4px 12px var(--color-border)',
        }}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
          <line x1="3" y1="1" x2="3" y2="11" stroke="var(--color-muted)" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="6" y1="1" x2="6" y2="11" stroke="var(--color-muted)" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="9" y1="1" x2="9" y2="11" stroke="var(--color-muted)" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </button>

      {/* Sync indicator */}
      {synced && (
        <span className="absolute bottom-2 bg-surface/80 text-xs font-mono text-muted px-1.5 py-0.5 rounded">
          Sync
        </span>
      )}
    </div>
  );
};

export default WipeDivider;
