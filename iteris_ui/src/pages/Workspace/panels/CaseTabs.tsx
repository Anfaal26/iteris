/**
 * CaseTabs — the result tab strip shown above the viewer for a batch run.
 *
 * One tab per queued scan. Each carries a status dot (queued / running / done /
 * error) so progress is legible while the queue is still working, and selecting
 * a tab swaps the whole centre + right panel over to that case — same viewer,
 * same mask editor, same metrics. Hidden entirely for a single-scan run, where
 * a one-tab strip would be noise.
 */

import React from 'react';
import type { BatchItemStatus } from './ControlPanel';

export interface CaseTabsProps {
  cases: { id: string; label: string; status: BatchItemStatus }[];
  activeIndex: number;
  onSelect: (index: number) => void;
  /** Downloads every finished case's edited mask. */
  onDownloadAll: () => void;
  /** True while the queue is still running — the download is incomplete until it ends. */
  running: boolean;
}

const STATUS_DOT: Record<BatchItemStatus, string> = {
  queued: 'bg-muted/50',
  running: 'bg-accent animate-pulse',
  done: 'bg-success',
  error: 'bg-error',
};

export const CaseTabs: React.FC<CaseTabsProps> = ({
  cases,
  activeIndex,
  onSelect,
  onDownloadAll,
  running,
}) => {
  const anyDone = cases.some((c) => c.status === 'done');

  return (
    <div
      role="tablist"
      aria-label="Batch results"
      className="flex items-center gap-1 px-3 py-1.5 bg-surface border-b border-border overflow-x-auto flex-shrink-0"
    >
      {cases.map((c, i) => (
        <button
          key={c.id}
          type="button"
          role="tab"
          aria-selected={i === activeIndex}
          onClick={() => onSelect(i)}
          title={c.label}
          className={[
            'flex items-center gap-1.5 rounded-md px-2.5 py-1 flex-shrink-0 max-w-[180px]',
            'transition-colors duration-panel ease-out',
            i === activeIndex
              ? 'bg-accent/15 text-text border border-accent/40'
              : 'border border-transparent text-muted hover:text-text hover:bg-surface-2',
          ].join(' ')}
        >
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${STATUS_DOT[c.status]}`} aria-hidden="true" />
          <span className="text-xs font-body truncate">{c.label}</span>
        </button>
      ))}

      <button
        type="button"
        disabled={!anyDone || running}
        onClick={onDownloadAll}
        className={[
          'ml-auto flex-shrink-0 text-[11px] font-body px-2.5 py-1 rounded-md border',
          'transition-colors duration-panel ease-out',
          !anyDone || running
            ? 'border-border/50 text-muted/40 cursor-not-allowed'
            : 'border-border text-muted hover:text-text hover:border-accent/50',
        ].join(' ')}
      >
        Download all masks
      </button>
    </div>
  );
};

export default CaseTabs;
