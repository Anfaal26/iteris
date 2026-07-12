/**
 * ResultsPanel — right panel showing metrics, structure breakdown, inference details,
 * export actions, and the LLM interpretation panel (spec §6 right zone, §7).
 */

import React, { useState } from 'react';
import {
  MetricCard,
  StructureRow,
  ExportButtonGroup,
  LLMInterpretationPanel,
} from '@/components';
import type { MetricStatus } from '@/components';
import type { PredictResponse, InterpretRequest } from '@/api/contract';
import { api } from '@/api/client';

/** Props for the ResultsPanel component. */
export interface ResultsPanelProps {
  result: PredictResponse | null;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onExportJson?: () => void;
  onDownloadPng?: () => void;
}

function metricStatus(value: number, baseline: number): MetricStatus {
  if (value >= baseline + 0.02) return 'success';
  if (value >= baseline - 0.02) return 'warning';
  return 'error';
}

/** Right panel with metric cards, structure rows, accordion details, and export actions. */
export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  result,
  collapsed,
  onToggleCollapse,
  onExportJson,
  onDownloadPng,
}) => {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [showInterpret, setShowInterpret] = useState(false);
  const [interpretStream, setInterpretStream] = useState<AsyncIterable<string> | undefined>();
  const [interpretLoading, setInterpretLoading] = useState(false);

  const handleInterpret = () => {
    if (!result) return;
    setShowInterpret(true);
    setInterpretLoading(true);

    const req: InterpretRequest = {
      modelId: result.modelId,
      structures: result.metrics.structures.map((s) => s.structure),
      metrics: result.metrics,
      dataset: result.dataset,
      modality: result.dataset === 'camus' ? 'ultrasound' : 'mri',
    };

    const gen = api.interpret(req);
    setInterpretStream(gen);
    // Clear loading after a short delay (stream starts immediately)
    setTimeout(() => setInterpretLoading(false), 500);
  };

  // Collapsed: slim rail with just an expand chevron, freeing room for the viewer.
  if (collapsed) {
    return (
      <aside
        aria-label="Results panel (collapsed)"
        className="flex flex-col items-center h-full bg-surface border-l border-border py-4 w-12 flex-shrink-0"
      >
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Expand results panel"
          className="text-muted hover:text-text transition-colors duration-panel ease-out"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M15 6l-6 6 6 6" />
          </svg>
        </button>
      </aside>
    );
  }

  if (!result) {
    return (
      <aside
        aria-label="Results panel"
        className="flex flex-col h-full bg-surface border-l border-border p-4"
        style={{ width: 'var(--results-panel-width)', minWidth: 'var(--results-panel-width)' }}
      >
        <div className="flex items-center justify-between mb-4">
          <span className="text-[11px] font-heading uppercase tracking-wider text-muted">Summary</span>
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label="Collapse results panel"
            className="text-muted hover:text-text transition-colors duration-panel ease-out"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M9 6l6 6-6 6" />
            </svg>
          </button>
        </div>
        <p className="text-sm font-body text-muted m-auto text-center">
          Run inference to see results here.
        </p>
      </aside>
    );
  }

  const { metrics, sessionId, modelId, preprocessingMs, inferenceMs, imageWidth, imageHeight } =
    result;
  const bd = metrics.baselineDice;

  const summaryMetrics: { label: string; value: number; baseline: number; invertStatus?: boolean }[] = [
    { label: 'Dice', value: metrics.dice, baseline: bd },
    { label: 'IoU', value: metrics.iou, baseline: bd - 0.08 },
    { label: 'HD', value: metrics.hd, baseline: 5.6, invertStatus: true },
    { label: '95HD', value: metrics.hd95, baseline: 4.0, invertStatus: true },
  ];

  return (
    <aside
      aria-label="Results panel"
      className="flex flex-col gap-4 h-full overflow-y-auto bg-surface border-l border-border p-4"
      style={{ width: 'var(--results-panel-width)', minWidth: 'var(--results-panel-width)' }}
    >
      {/* Panel header with collapse control */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-heading uppercase tracking-wider text-muted">Summary</span>
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Collapse results panel"
          className="text-muted hover:text-text transition-colors duration-panel ease-out"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M9 6l6 6-6 6" />
          </svg>
        </button>
      </div>

      {/* Metric cards */}
      <section aria-label="Aggregate metrics">
        <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
          Metrics
        </h2>
        <div className="grid grid-cols-2 gap-2">
          {summaryMetrics.map((m) => {
            const status: MetricStatus = m.invertStatus
              ? m.value <= m.baseline - 1 ? 'success' : m.value <= m.baseline + 1 ? 'warning' : 'error'
              : metricStatus(m.value, m.baseline);
            return (
              <MetricCard
                key={m.label}
                label={m.label}
                value={m.value.toFixed(3)}
                status={status}
                baselineLabel={`Baseline: ${m.baseline.toFixed(m.invertStatus ? 1 : 3)}`}
              />
            );
          })}
        </div>
      </section>

      {/* Per-structure breakdown */}
      <section aria-label="Per-structure metrics" role="table">
        <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
          Structures
        </h2>
        <div className="flex flex-col divide-y divide-border">
          {metrics.structures.map((s) => (
            <StructureRow key={s.structure} metrics={s} />
          ))}
        </div>
      </section>

      {/* Inference details accordion */}
      <section aria-label="Inference details">
        <button
          type="button"
          aria-expanded={detailsOpen}
          onClick={() => setDetailsOpen((v) => !v)}
          className="w-full flex items-center justify-between text-xs font-heading font-semibold text-muted uppercase tracking-wider hover:text-text transition-colors duration-panel ease-out"
        >
          <span>Inference Details</span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className={['transition-transform duration-panel ease-out', detailsOpen ? 'rotate-180' : ''].join(' ')}
            aria-hidden="true"
          >
            <polyline points="2 4 6 8 10 4" />
          </svg>
        </button>
        {detailsOpen && (
          <dl className="mt-2 flex flex-col gap-1">
            {[
              { dt: 'Model', dd: modelId },
              { dt: 'Session ID', dd: sessionId },
              { dt: 'Preprocessing', dd: `${preprocessingMs} ms` },
              { dt: 'Inference', dd: `${inferenceMs} ms` },
              { dt: 'Dimensions', dd: `${imageWidth}×${imageHeight}` },
            ].map(({ dt, dd }) => (
              <div key={dt} className="flex items-center justify-between">
                <dt className="text-xs font-body text-muted">{dt}</dt>
                <dd className="text-xs font-mono text-text">{dd}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      {/* Export */}
      <section aria-label="Export actions">
        <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
          Export
        </h2>
        <ExportButtonGroup
          onDownloadPng={onDownloadPng}
          onExportJson={onExportJson}
          sessionUrl={`${window.location.origin}${window.location.pathname}?session=${sessionId}`}
        />
      </section>

      {/* Interpret with Claude */}
      {!showInterpret && (
        <button
          type="button"
          onClick={handleInterpret}
          className="w-full py-2.5 px-4 rounded-lg border border-accent/40 text-sm font-heading font-semibold text-accent hover:bg-accent/5 transition-colors duration-panel ease-out"
        >
          Interpret with Claude ✦
        </button>
      )}

      {showInterpret && (
        <LLMInterpretationPanel
          stream={interpretStream}
          loading={interpretLoading}
          className="flex-1"
        />
      )}
    </aside>
  );
};

export default ResultsPanel;
