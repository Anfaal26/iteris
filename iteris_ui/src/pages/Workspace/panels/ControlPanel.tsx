/**
 * ControlPanel — left sidebar (redesign).
 *
 * Restructured, not re-skinned: same dark tokens, mono micro-labels, pill
 * controls and rounded-xl cards as the rest of the app. Sections top→bottom:
 *   • Image      — side-by-side Scan / GT-mask dropzones + read-only detection chip
 *   • Batch      — collapsed by default; a multi-file picker for batch runs
 *   • (untitled) — Attention Res U-Net · DRL (always expanded: DuelingDDQN / TD3)
 *   • Data regime — Low / High segmented control, default flips with the model
 *   • View mode  — Single / Wipe (+ source chips) / Side-by-Side
 *   • ▷ run       — circular play button pinned to the bottom
 * The whole rail collapses to an icon strip via the chevron at its top edge.
 *
 * The panel is absolutely positioned over the workspace's left rail: collapsed it
 * exactly fills the permanently-reserved --control-panel-collapsed track, expanded
 * it overlays the centre column. Either way the centre never resizes or reflows.
 */

import React, { useState, useCallback, useRef } from 'react';
import { SampleImageTile } from '@/components';
import type {
  SampleImage,
  ModelId,
  DatasetId,
  Regime,
  ViewMode,
  WipeSource,
} from '@/api/contract';
import { isModelAvailable, isCombinationAvailable } from '@/api/contract';
import type { DetectionResult } from '@/lib/detectDataset';

/** Props for the ControlPanel component. */
export interface ControlPanelProps {
  samples: SampleImage[];
  selectedModel: ModelId;
  /** Detected (or assumed-default) dataset used for availability gating. */
  dataset: DatasetId;
  /** Result of auto-detecting the uploaded scan; null before a scan is present. */
  detection: DetectionResult | null;
  selectedRegime: Regime;
  viewMode: ViewMode;
  /** The two sources currently paired in Wipe mode. */
  wipeSources: [WipeSource, WipeSource];
  loading: boolean;
  collapsed: boolean;
  scanLabel?: string;
  gtMaskLabel?: string;
  onToggleCollapse: () => void;
  onModelSelect: (id: ModelId) => void;
  onRegimeChange: (r: Regime) => void;
  onViewModeChange: (m: ViewMode) => void;
  onWipeSourcesChange: (s: [WipeSource, WipeSource]) => void;
  onSampleSelect: (sample: SampleImage) => void;
  /** dataUrl is the full `data:<mime>;base64,...` string; file is the raw File (for detection). */
  onScanUpload: (dataUrl: string, file: File) => void;
  onGtMaskUpload: (dataUrl: string, file: File) => void;
  onRunInference: () => void;

  /* --- batch segmentation --- */
  /** Hard cap on files per batch run (half the backend's safe concurrent load). */
  maxBatchFiles: number;
  /** Queued/queued-and-run batch cases, for the in-panel progress list. */
  batchItems: { id: string; label: string; status: BatchItemStatus }[];
  onBatchUpload: (files: File[]) => void;
  onClearBatch: () => void;
}

/** Lifecycle of one queued batch image. */
export type BatchItemStatus = 'queued' | 'running' | 'done' | 'error';

/** Top-level model picker structure. DRL is a group; the rest are leaves. */
type PickerNode =
  | { kind: 'model'; id: ModelId; name: string }
  | {
      kind: 'group';
      label: string;
      children: { id: ModelId; name: string; sub: string }[];
    };

/**
 * Lite U-Net is deliberately absent: it exists in the research code as an
 * architecture-headroom reference but is not a deployed checkpoint, so it is
 * not offered here. DRL is a permanently-expanded group, never collapsible.
 */
const PICKER: PickerNode[] = [
  { kind: 'model', id: 'unet-baseline', name: 'Attention Res U-Net' },
  {
    kind: 'group',
    label: 'DRL',
    children: [
      { id: 'dueling-dqn', name: 'DuelingDDQN', sub: 'Discrete' },
      { id: 'td3', name: 'TD3', sub: 'Continuous' },
    ],
  },
];

const WIPE_SOURCE_LABEL: Record<WipeSource, string> = {
  'attention-unet': 'Attention U-Net',
  gt: 'GT',
  drl: 'DRL',
};

/** Reads a File as a data URL and forwards it (with the File) to `onLoad`. */
function readFile(file: File, onLoad: (dataUrl: string, file: File) => void) {
  const reader = new FileReader();
  reader.onload = () => onLoad(reader.result as string, file);
  reader.readAsDataURL(file);
}

/** A single dashed dropzone (Scan or GT mask). */
const DropZone: React.FC<{
  label: string;
  filled?: string;
  /** GT is secondary until a scan exists. */
  disabled?: boolean;
  onFile: (dataUrl: string, file: File) => void;
  icon: React.ReactNode;
}> = ({ label, filled, disabled, onFile, icon }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) readFile(file, onFile);
    },
    [disabled, onFile],
  );
  return (
    <div className="flex-1 min-w-0">
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
        onDragLeave={() => setOver(false)}
        aria-label={`Upload ${label}`}
        className={[
          'w-full flex flex-col items-center gap-1 rounded-lg border border-dashed px-2 py-3',
          'transition-colors duration-panel ease-out text-center',
          disabled
            ? 'border-border/50 opacity-50 cursor-not-allowed'
            : over
              ? 'border-accent bg-accent/5 cursor-pointer'
              : filled
                ? 'border-accent/40 cursor-pointer'
                : 'border-border hover:border-accent/50 cursor-pointer',
        ].join(' ')}
      >
        <span className={filled ? 'text-accent' : 'text-muted'} aria-hidden="true">
          {icon}
        </span>
        <span className="text-[11px] font-body text-muted truncate max-w-full">
          {filled ?? label}
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".dcm,.nii,.nii.gz,.png,.jpg,.jpeg"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) readFile(file, onFile);
        }}
      />
    </div>
  );
};

/** Uppercase mono section micro-label, matching the rest of the app. */
const SectionLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
    {children}
  </h2>
);

/** Left control panel — collapses to an icon rail. */
export const ControlPanel: React.FC<ControlPanelProps> = ({
  samples,
  selectedModel,
  dataset,
  detection,
  selectedRegime,
  viewMode,
  wipeSources,
  loading,
  collapsed,
  scanLabel,
  gtMaskLabel,
  onToggleCollapse,
  onModelSelect,
  onRegimeChange,
  onViewModeChange,
  onWipeSourcesChange,
  onSampleSelect,
  onScanUpload,
  onGtMaskUpload,
  onRunInference,
  maxBatchFiles,
  batchItems,
  onBatchUpload,
  onClearBatch,
}) => {
  const [batchOpen, setBatchOpen] = useState(false);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const canRun = (!!scanLabel || batchItems.length > 0) && !loading;

  const runButton = (
    <button
      type="button"
      onClick={onRunInference}
      disabled={!canRun}
      aria-label="Run inference"
      className={[
        'w-11 h-11 rounded-full flex items-center justify-center flex-shrink-0',
        'transition-colors duration-panel ease-out',
        canRun ? 'bg-accent hover:bg-accent/90' : 'bg-accent/40 cursor-not-allowed',
      ].join(' ')}
    >
      {loading ? (
        <svg width="18" height="18" viewBox="0 0 24 24" className="animate-spin text-white" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.3" />
          <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="text-white ml-0.5" aria-hidden="true">
          <path d="M8 5v14l11-7z" />
        </svg>
      )}
    </button>
  );

  // Collapsed: a slim rail with just expand + run, so the viewer gets the room.
  if (collapsed) {
    return (
      <aside
        aria-label="Control panel (collapsed)"
        className="absolute inset-y-0 left-0 z-20 flex flex-col items-center justify-between bg-surface border-r border-border py-4"
        style={{ width: 'var(--control-panel-collapsed)' }}
      >
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Expand control panel"
          className="text-muted hover:text-text transition-colors duration-panel ease-out"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M9 6l6 6-6 6" />
          </svg>
        </button>
        {runButton}
      </aside>
    );
  }

  return (
    <aside
      aria-label="Control panel"
      className="absolute inset-y-0 left-0 z-20 flex flex-col overflow-hidden bg-surface border-r border-border shadow-float"
      style={{ width: 'var(--control-panel-width)', minWidth: 'var(--control-panel-width)' }}
    >
      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-5">
        {/* Header with collapse chevron */}
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-heading uppercase tracking-wider text-muted">Workspace</span>
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label="Collapse control panel"
            className="text-muted hover:text-text transition-colors duration-panel ease-out"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M15 6l-6 6 6 6" />
            </svg>
          </button>
        </div>

        {/* Image — dual dropzones + detection chip */}
        <section aria-label="Image upload">
          <SectionLabel>Image</SectionLabel>
          <div className="flex gap-2">
            <DropZone
              label="Scan"
              filled={scanLabel}
              onFile={onScanUpload}
              icon={
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              }
            />
            <DropZone
              label="GT mask"
              filled={gtMaskLabel}
              disabled={!scanLabel}
              onFile={onGtMaskUpload}
              icon={
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="M12 21s-7-4.35-7-10a7 7 0 0 1 14 0c0 5.65-7 10-7 10z" />
                  <circle cx="12" cy="11" r="2.5" />
                </svg>
              }
            />
          </div>

          {/* Read-only detection chip — never a control */}
          {detection && (
            <div
              className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-2.5 py-1"
              title={`Detected from ${detection.source}`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-accent" aria-hidden="true" />
              <span className="text-[11px] font-body text-muted">
                {detection.confidence === 'low' ? 'Looks like' : 'Detected'}: {detection.label}
              </span>
            </div>
          )}

          {/* Samples */}
          {samples.length > 0 && (
            <>
              <p className="text-[11px] font-body text-muted mt-3 mb-1.5">Or choose a sample:</p>
              <div className="grid grid-cols-2 gap-2">
                {samples.slice(0, 4).map((s) => (
                  <SampleImageTile key={s.id} image={s} onSelect={() => onSampleSelect(s)} />
                ))}
              </div>
            </>
          )}
        </section>

        {/* Batch segmentation — collapsed by default so the single-scan flow
            stays the obvious one. Runs are driven by the same ▷ button below. */}
        <section aria-label="Batch segmentation">
          <button
            type="button"
            aria-expanded={batchOpen}
            onClick={() => setBatchOpen((v) => !v)}
            className="w-full flex items-center justify-between text-xs font-heading font-semibold text-muted uppercase tracking-wider hover:text-text transition-colors duration-panel ease-out"
          >
            <span>Batch segmentation</span>
            <span className="flex items-center gap-1.5">
              {batchItems.length > 0 && (
                <span className="text-[10px] font-mono text-accent normal-case tracking-normal">
                  {batchItems.length}/{maxBatchFiles}
                </span>
              )}
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className={['transition-transform duration-panel ease-out', batchOpen ? 'rotate-180' : ''].join(' ')}
                aria-hidden="true"
              >
                <polyline points="2 4 6 8 10 4" />
              </svg>
            </span>
          </button>

          {batchOpen && (
            <div className="mt-2">
              <button
                type="button"
                disabled={loading}
                onClick={() => batchInputRef.current?.click()}
                className={[
                  'w-full flex flex-col items-center gap-1 rounded-lg border border-dashed px-2 py-3 text-center',
                  'transition-colors duration-panel ease-out',
                  loading
                    ? 'border-border/50 opacity-50 cursor-not-allowed'
                    : 'border-border hover:border-accent/50 cursor-pointer',
                ].join(' ')}
              >
                <span className="text-muted" aria-hidden="true">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                    <path d="M12 11v5M9.5 13.5 12 11l2.5 2.5" />
                  </svg>
                </span>
                <span className="text-[11px] font-body text-muted">Choose scans</span>
              </button>
              <input
                ref={batchInputRef}
                type="file"
                multiple
                accept=".dcm,.nii,.nii.gz,.png,.jpg,.jpeg"
                className="hidden"
                onChange={(e) => {
                  const files = Array.from(e.target.files ?? []);
                  if (files.length) onBatchUpload(files);
                  // Reset so re-picking the same files fires change again.
                  e.target.value = '';
                }}
              />
              <p className="text-[11px] font-body text-muted mt-1.5">
                Maximum {maxBatchFiles} files per run. Press ▷ to start; results open as tabs.
              </p>

              {batchItems.length > 0 && (
                <>
                  <ul className="mt-2 flex flex-col gap-1">
                    {batchItems.map((item) => (
                      <li
                        key={item.id}
                        className="flex items-center gap-2 rounded-md bg-surface-2 px-2 py-1.5"
                      >
                        <span
                          className={[
                            'w-1.5 h-1.5 rounded-full flex-shrink-0',
                            item.status === 'done'
                              ? 'bg-success'
                              : item.status === 'error'
                                ? 'bg-error'
                                : item.status === 'running'
                                  ? 'bg-accent animate-pulse'
                                  : 'bg-muted/50',
                          ].join(' ')}
                          aria-hidden="true"
                        />
                        <span className="text-[11px] font-body text-muted truncate flex-1">
                          {item.label}
                        </span>
                        <span className="text-[10px] font-mono text-muted/70 flex-shrink-0">
                          {item.status}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <button
                    type="button"
                    disabled={loading}
                    onClick={onClearBatch}
                    className="mt-1.5 text-[11px] font-body text-muted hover:text-text disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-panel ease-out"
                  >
                    Clear queue
                  </button>
                </>
              )}
            </div>
          )}
        </section>

        {/* Model picker — no section title; the options speak for themselves */}
        <section aria-label="Model selection">
          <div className="flex flex-col gap-1.5">
            {PICKER.map((node) => {
              if (node.kind === 'model') {
                const enabled = isModelAvailable(dataset, node.id);
                const active = selectedModel === node.id;
                return (
                  <button
                    key={node.id}
                    type="button"
                    disabled={!enabled}
                    aria-pressed={active}
                    onClick={() => onModelSelect(node.id)}
                    className={[
                      'w-full text-left rounded-lg border px-3 py-2 text-sm font-body',
                      'transition-colors duration-panel ease-out',
                      active
                        ? 'border-accent bg-accent/10 text-text'
                        : enabled
                          ? 'border-border text-muted hover:border-accent/50'
                          : 'border-border/50 text-muted/40 cursor-not-allowed',
                    ].join(' ')}
                  >
                    {node.name}
                    {!enabled && <span className="ml-1 text-[10px]">· soon</span>}
                  </button>
                );
              }
              // DRL group — expanded, boxed, selected by default.
              const groupActive = node.children.some((c) => c.id === selectedModel);
              return (
                <div
                  key={node.label}
                  className={[
                    'rounded-lg border px-2.5 py-2',
                    groupActive ? 'border-accent' : 'border-border',
                  ].join(' ')}
                >
                  {/* No collapse affordance — the children are always visible */}
                  <div className="px-0.5 pb-1.5">
                    <span className="text-sm font-body font-medium text-text">{node.label}</span>
                  </div>
                  <div className="flex flex-col gap-1">
                    {node.children.map((c) => {
                      const enabled = isModelAvailable(dataset, c.id);
                      const active = selectedModel === c.id;
                      return (
                        <button
                          key={c.id}
                          type="button"
                          disabled={!enabled}
                          aria-pressed={active}
                          onClick={() => onModelSelect(c.id)}
                          className={[
                            'w-full flex items-center justify-between rounded-md px-2 py-1.5 text-left',
                            'transition-colors duration-panel ease-out',
                            active
                              ? 'bg-accent/15 text-text'
                              : enabled
                                ? 'text-muted hover:bg-surface-2'
                                : 'text-muted/40 cursor-not-allowed',
                          ].join(' ')}
                        >
                          <span className="text-sm font-body">
                            {c.name}
                            {!enabled && <span className="ml-1 text-[10px]">· soon</span>}
                          </span>
                          <span className="text-[11px] font-body text-muted/70">{c.sub}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Data regime — default flips with the model, unavailable options greyed */}
        <section aria-label="Data regime">
          <SectionLabel>Data regime</SectionLabel>
          <div className="flex rounded-lg border border-border overflow-hidden">
            {(['low', 'high'] as Regime[]).map((r) => {
              const enabled = isCombinationAvailable(dataset, selectedModel, r);
              const active = selectedRegime === r;
              return (
                <button
                  key={r}
                  type="button"
                  disabled={!enabled}
                  aria-pressed={active}
                  onClick={() => onRegimeChange(r)}
                  className={[
                    'flex-1 py-1.5 text-xs font-body capitalize transition-colors duration-panel ease-out',
                    active
                      ? 'bg-accent text-white'
                      : enabled
                        ? 'text-muted hover:text-text'
                        : 'text-muted/30 cursor-not-allowed',
                  ].join(' ')}
                >
                  {r} data
                </button>
              );
            })}
          </div>
        </section>

        {/* View mode */}
        <section aria-label="View mode">
          <SectionLabel>View mode</SectionLabel>
          <div className="flex flex-col gap-1">
            {(
              [
                { id: 'single', label: 'Single model' },
                { id: 'wipe', label: 'Wipe comparison' },
                { id: 'side-by-side', label: 'Side-by-side' },
              ] as { id: ViewMode; label: string }[]
            ).map(({ id, label }) => (
              <button
                key={id}
                type="button"
                aria-pressed={viewMode === id}
                onClick={() => onViewModeChange(id)}
                className={[
                  'w-full py-1.5 px-3 rounded-md text-xs font-body text-left',
                  'transition-colors duration-panel ease-out',
                  viewMode === id
                    ? 'bg-accent/15 text-accent border border-accent/30'
                    : 'border border-transparent text-muted hover:text-text',
                ].join(' ')}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Wipe source chips — pick any two available sources */}
          {viewMode === 'wipe' && (
            <div className="mt-2">
              <p className="text-[11px] font-body text-muted mb-1.5">Compare (pick two):</p>
              <div className="flex gap-1.5 flex-wrap">
                {(['attention-unet', 'gt', 'drl'] as WipeSource[]).map((src) => {
                  const selected = wipeSources.includes(src);
                  // GT is only available once a mask is attached.
                  const enabled = src !== 'gt' || !!gtMaskLabel;
                  return (
                    <button
                      key={src}
                      type="button"
                      disabled={!enabled}
                      aria-pressed={selected}
                      onClick={() => {
                        if (!enabled) return;
                        if (selected) {
                          // Keep at least two selected — swap out the *other* one instead.
                          const other = wipeSources.find((s) => s !== src)!;
                          onWipeSourcesChange([src, other]);
                        } else {
                          onWipeSourcesChange([wipeSources[1], src]);
                        }
                      }}
                      className={[
                        'text-[11px] px-2.5 py-1 rounded-full border transition-colors duration-panel ease-out',
                        selected
                          ? 'bg-accent/15 text-accent border-accent'
                          : enabled
                            ? 'text-muted border-border hover:border-accent/50'
                            : 'text-muted/30 border-border/50 cursor-not-allowed',
                      ].join(' ')}
                    >
                      {WIPE_SOURCE_LABEL[src]}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Run — circular play button pinned to the bottom */}
      <div className="flex justify-center border-t border-border py-3 flex-shrink-0">
        {runButton}
      </div>
    </aside>
  );
};

export default ControlPanel;
