/**
 * MetricCard — displays a large mono metric value with a status dot
 * and a one-line baseline context label.
 */

import React from 'react';

/** Status of a metric relative to the baseline. */
export type MetricStatus = 'success' | 'warning' | 'error';

/** Props for the MetricCard component. */
export interface MetricCardProps {
  /** Metric name shown as a small label above the value. */
  label: string;
  /** The formatted metric value string (e.g. "0.923"). */
  value: string;
  /** Status relative to baseline. */
  status: MetricStatus;
  /** One-line context string (e.g. "Baseline: 0.841"). */
  baselineLabel: string;
  /** Additional class names. */
  className?: string;
}

const STATUS_DOT: Record<MetricStatus, string> = {
  success: 'bg-success',
  warning: 'bg-warning',
  error: 'bg-error',
};

const STATUS_TEXT: Record<MetricStatus, string> = {
  success: 'Above baseline',
  warning: 'Near baseline',
  error: 'Below baseline',
};

/**
 * Compact card showing one aggregate metric (Dice, IoU, HD, etc.) with status colour.
 * All colours are resolved from Tailwind token classes — no hardcoded values.
 */
export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  status,
  baselineLabel,
  className,
}) => (
  <div
    className={[
      'bg-surface border border-border rounded-lg p-4 flex flex-col gap-1',
      className ?? '',
    ].join(' ')}
  >
    <span className="text-xs font-body text-muted uppercase tracking-wider">{label}</span>
    <div className="flex items-center gap-2">
      <span className="font-mono text-2xl font-semibold text-text">{value}</span>
      <span
        className={`w-2 h-2 rounded-full flex-shrink-0 ${STATUS_DOT[status]}`}
        aria-label={STATUS_TEXT[status]}
        title={STATUS_TEXT[status]}
      />
    </div>
    <span className="text-xs font-body text-muted">{baselineLabel}</span>
  </div>
);

export default MetricCard;
