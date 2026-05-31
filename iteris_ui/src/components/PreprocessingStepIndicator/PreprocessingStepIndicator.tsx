/**
 * PreprocessingStepIndicator — horizontal 5-step pipeline indicator.
 * Completed steps show a green check + elapsed ms. Not a spinner.
 */

import React from 'react';

/** A single preprocessing step's status. */
export interface StepStatus {
  /** Step display label. */
  label: string;
  /** Whether this step has completed. */
  done: boolean;
  /** Elapsed time in ms. Only shown when done=true. */
  elapsedMs?: number;
}

/** Props for the PreprocessingStepIndicator component. */
export interface PreprocessingStepIndicatorProps {
  /** Array of exactly 5 step statuses. */
  steps?: StepStatus[];
  /** Additional class names. */
  className?: string;
}

const DEFAULT_STEPS: StepStatus[] = [
  { label: 'Load', done: false },
  { label: 'Normalise', done: false },
  { label: 'Resize', done: false },
  { label: 'Augment', done: false },
  { label: 'Ready', done: false },
];

/**
 * Displays the 5-stage preprocessing pipeline as a horizontal step indicator.
 * Completed steps show a green check mark and elapsed milliseconds.
 */
export const PreprocessingStepIndicator: React.FC<PreprocessingStepIndicatorProps> = ({
  steps = DEFAULT_STEPS,
  className,
}) => (
  <div
    className={['flex items-center gap-0', className ?? ''].join(' ')}
    role="list"
    aria-label="Preprocessing pipeline"
  >
    {steps.map((step, idx) => (
      <React.Fragment key={step.label}>
        <div
          role="listitem"
          aria-label={`${step.label}${step.done ? ` — completed${step.elapsedMs != null ? ` in ${step.elapsedMs}ms` : ''}` : ' — pending'}`}
          className="flex flex-col items-center gap-0.5 px-2"
        >
          {/* Icon */}
          <div
            className={[
              'w-6 h-6 rounded-full flex items-center justify-center',
              step.done ? 'bg-success' : 'bg-border',
            ].join(' ')}
          >
            {step.done ? (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                <polyline points="2,6 5,9 10,3" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            ) : (
              <span className="w-2 h-2 rounded-full bg-muted" aria-hidden="true" />
            )}
          </div>
          {/* Label */}
          <span
            className={[
              'text-xs font-body',
              step.done ? 'text-success' : 'text-muted',
            ].join(' ')}
          >
            {step.label}
          </span>
          {/* Elapsed */}
          {step.done && step.elapsedMs != null && (
            <span className="text-xs font-mono text-muted">{step.elapsedMs}ms</span>
          )}
        </div>

        {/* Connector */}
        {idx < steps.length - 1 && (
          <div
            aria-hidden="true"
            className={[
              'h-0.5 flex-1 min-w-[12px]',
              step.done ? 'bg-success' : 'bg-border',
            ].join(' ')}
          />
        )}
      </React.Fragment>
    ))}
  </div>
);

export default PreprocessingStepIndicator;
