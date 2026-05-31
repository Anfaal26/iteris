/**
 * IterationPlaybackTimeline — DRL iteration playback transport.
 * Shows step counter, progress bar with clickable markers, transport controls,
 * speed selector, sparkline of Δ Dice per step, and step annotations.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { IterationStep } from '@/api/contract';

/** Props for the IterationPlaybackTimeline component. */
export interface IterationPlaybackTimelineProps {
  /** Ordered iteration steps. */
  steps: IterationStep[];
  /** Currently visible step index (0-based). */
  currentStep?: number;
  /** Called when the user changes the active step. */
  onStepChange?: (step: number) => void;
  /** Additional class names. */
  className?: string;
}

const SPEEDS = [0.5, 1, 2, 4] as const;
type Speed = (typeof SPEEDS)[number];

const SPARKLINE_H = 32;
const SPARKLINE_W = 200;

function buildSparklinePath(deltas: number[], currentIdx: number): { path: string; rulePct: number } {
  if (deltas.length < 2) return { path: '', rulePct: 0 };
  const min = Math.min(...deltas);
  const max = Math.max(...deltas);
  const range = max - min || 1;
  const points = deltas.map((d, i) => {
    const x = (i / (deltas.length - 1)) * SPARKLINE_W;
    const y = SPARKLINE_H - ((d - min) / range) * (SPARKLINE_H - 4) - 2;
    return `${x},${y}`;
  });
  const path = `M ${points.join(' L ')}`;
  const rulePct = (currentIdx / Math.max(deltas.length - 1, 1)) * 100;
  return { path, rulePct };
}

/**
 * Full iteration playback timeline.
 * Does NOT animate clinical output (masks) — it only advances the step index
 * and lets the parent re-render the viewer accordingly via `onStepChange`.
 */
export const IterationPlaybackTimeline: React.FC<IterationPlaybackTimelineProps> = ({
  steps,
  currentStep = 0,
  onStepChange,
  className,
}) => {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const total = steps.length;
  const currentStepRef = useRef(currentStep);
  currentStepRef.current = currentStep;

  const go = useCallback(
    (idx: number) => onStepChange?.(Math.max(0, Math.min(total - 1, idx))),
    [onStepChange, total],
  );

  // Autoplay
  useEffect(() => {
    if (!playing) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    const ms = 800 / speed;
    intervalRef.current = setInterval(() => {
      const next = currentStepRef.current + 1;
      if (next >= total) {
        setPlaying(false);
      } else {
        onStepChange?.(next);
      }
    }, ms);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, total, onStepChange]);

  const deltas = steps.map((s) => s.deltaDice);
  const { path: sparkPath, rulePct } = buildSparklinePath(deltas, currentStep);
  const annotation = steps[currentStep]?.annotation ?? '';

  return (
    <div
      className={['bg-surface border border-border rounded-xl p-4 flex flex-col gap-3', className ?? ''].join(' ')}
      aria-label="Iteration playback timeline"
    >
      {/* Step counter */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-heading font-semibold text-text">
          Step <span className="font-mono">{currentStep + 1}</span>
          <span className="text-muted font-normal"> / {total}</span>
        </span>
        {/* Speed selector */}
        <div
          className="flex items-center gap-1"
          role="group"
          aria-label="Playback speed"
        >
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={speed === s}
              onClick={() => setSpeed(s)}
              className={[
                'px-2 py-0.5 rounded text-xs font-mono transition-colors duration-panel ease-out',
                speed === s
                  ? 'bg-accent text-white'
                  : 'text-muted hover:text-text border border-border',
              ].join(' ')}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* Progress bar with N clickable markers */}
      <div
        className="relative h-1.5 bg-border rounded-full"
        role="group"
        aria-label="Step markers"
      >
        {/* Fill */}
        <div
          className="absolute inset-y-0 left-0 bg-accent rounded-full transition-all duration-panel ease-out"
          style={{ width: total > 1 ? `${(currentStep / (total - 1)) * 100}%` : '0%' }}
          aria-hidden="true"
        />
        {/* Markers */}
        {steps.map((_, idx) => {
          const pct = total > 1 ? (idx / (total - 1)) * 100 : 0;
          return (
            <button
              key={idx}
              type="button"
              aria-label={`Go to step ${idx + 1}`}
              onClick={() => go(idx)}
              className={[
                'absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full -translate-x-1/2',
                'transition-colors duration-panel ease-out',
                idx === currentStep
                  ? 'bg-accent scale-125'
                  : idx < currentStep
                  ? 'bg-accent/60'
                  : 'bg-border hover:bg-accent/50',
              ].join(' ')}
              style={{ left: `${pct}%` }}
            />
          );
        })}
      </div>

      {/* Transport controls */}
      <div className="flex items-center justify-center gap-2">
        {/* Reset */}
        <button type="button" aria-label="Reset to first step" onClick={() => { setPlaying(false); go(0); }}
          className="w-8 h-8 flex items-center justify-center rounded-md text-muted hover:text-text transition-colors duration-panel ease-out">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
            <path d="M2 2v10l2-2V4L2 2zM5 7l7-5v10L5 7z" />
          </svg>
        </button>
        {/* Previous */}
        <button type="button" aria-label="Previous step" onClick={() => { setPlaying(false); go(currentStep - 1); }}
          className="w-8 h-8 flex items-center justify-center rounded-md text-muted hover:text-text transition-colors duration-panel ease-out">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
            <path d="M11 2L4 7l7 5V2zM3 2v10H2V2h1z" />
          </svg>
        </button>
        {/* Play / Pause */}
        <button
          type="button"
          aria-label={playing ? 'Pause playback' : 'Play iteration'}
          aria-pressed={playing}
          onClick={() => setPlaying((p) => !p)}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-accent text-white hover:opacity-90 transition-opacity duration-panel ease-out"
        >
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
              <rect x="2" y="2" width="4" height="10" /><rect x="8" y="2" width="4" height="10" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
              <path d="M3 2l9 5-9 5V2z" />
            </svg>
          )}
        </button>
        {/* Next */}
        <button type="button" aria-label="Next step" onClick={() => { setPlaying(false); go(currentStep + 1); }}
          className="w-8 h-8 flex items-center justify-center rounded-md text-muted hover:text-text transition-colors duration-panel ease-out">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
            <path d="M3 2l7 5-7 5V2zM11 2v10h1V2h-1z" />
          </svg>
        </button>
      </div>

      {/* Sparkline */}
      {sparkPath && (
        <div aria-hidden="true" className="relative">
          <p className="text-xs text-muted font-body mb-1">Δ Dice per step</p>
          <svg
            width="100%"
            height={SPARKLINE_H}
            viewBox={`0 0 ${SPARKLINE_W} ${SPARKLINE_H}`}
            preserveAspectRatio="none"
            className="overflow-visible"
          >
            <path d={sparkPath} fill="none" stroke="var(--color-accent)" strokeWidth="1.5" />
            {/* Current step vertical rule */}
            <line
              x1={`${rulePct}%`} y1="0"
              x2={`${rulePct}%`} y2={SPARKLINE_H}
              stroke="var(--color-warning)"
              strokeWidth="1"
              strokeDasharray="3,2"
            />
          </svg>
        </div>
      )}

      {/* Step annotation */}
      {annotation && (
        <p className="text-xs font-body text-muted border-t border-border pt-2 line-clamp-2">
          {annotation}
        </p>
      )}
    </div>
  );
};

export default IterationPlaybackTimeline;
