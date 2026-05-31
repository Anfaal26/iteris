/**
 * ExportButtonGroup — three export actions:
 * 1. Download PNG mask
 * 2. Export JSON metrics
 * 3. Copy session link (via navigator.clipboard)
 */

import React, { useState } from 'react';

/** Props for the ExportButtonGroup component. */
export interface ExportButtonGroupProps {
  /** Called when "Download PNG mask" is activated. */
  onDownloadPng?: () => void;
  /** Called when "Export JSON metrics" is activated. */
  onExportJson?: () => void;
  /**
   * Session URL to copy. If provided, "Copy session link" uses this value.
   * If omitted, copies `window.location.href`.
   */
  sessionUrl?: string;
  /** Additional class names. */
  className?: string;
}

interface ExportButtonProps {
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  feedback?: boolean;
}

const ExportButton: React.FC<ExportButtonProps> = ({ label, icon, onClick, feedback }) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    title={label}
    className={[
      'flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-sm font-body',
      'transition-all duration-panel ease-out',
      feedback
        ? 'border-success text-success bg-success/10'
        : 'border-border text-muted hover:border-accent hover:text-accent bg-surface',
    ].join(' ')}
  >
    {icon}
    <span className="hidden sm:inline">{feedback ? 'Copied!' : label}</span>
  </button>
);

/**
 * Grouped export action buttons. Copy-link uses the async Clipboard API.
 * All callbacks are optional — pages provide their own download/export logic.
 */
export const ExportButtonGroup: React.FC<ExportButtonGroupProps> = ({
  onDownloadPng,
  onExportJson,
  sessionUrl,
  className,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopyLink = async () => {
    const url = sessionUrl ?? window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available (e.g. non-HTTPS) — silently ignore
    }
  };

  return (
    <div
      className={['flex items-center gap-2 flex-wrap', className ?? ''].join(' ')}
      role="group"
      aria-label="Export actions"
    >
      <ExportButton
        label="Download PNG mask"
        onClick={() => onDownloadPng?.()}
        icon={
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M7 1v8M4 6l3 3 3-3M2 11h10" />
          </svg>
        }
      />

      <ExportButton
        label="Export JSON metrics"
        onClick={() => onExportJson?.()}
        icon={
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="2" y="1" width="10" height="12" rx="1.5" />
            <line x1="4" y1="5" x2="10" y2="5" />
            <line x1="4" y1="7.5" x2="10" y2="7.5" />
            <line x1="4" y1="10" x2="7" y2="10" />
          </svg>
        }
      />

      <ExportButton
        label="Copy session link"
        onClick={handleCopyLink}
        feedback={copied}
        icon={
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M5.5 8.5a3 3 0 0 0 3.5.5l2-2a3 3 0 0 0-4.24-4.24L5.5 4" />
            <path d="M8.5 5.5a3 3 0 0 0-3.5-.5l-2 2a3 3 0 0 0 4.24 4.24L8.5 10" />
          </svg>
        }
      />
    </div>
  );
};

export default ExportButtonGroup;
