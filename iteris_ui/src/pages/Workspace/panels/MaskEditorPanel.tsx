/**
 * MaskEditorPanel — right panel: manual correction toolkit for the AI-predicted
 * segmentation mask (redesign §4). Replaces the old metrics panel; metrics now
 * live only in the post-inference StatsSection on the main scroll.
 *
 * Layout mirrors the left ControlPanel: absolutely positioned over the reserved
 * right rail, so collapsing/expanding never resizes the centre column. Contents
 * are grouped into compact labelled sections in the same dark-token visual
 * language as the rest of the workspace (mono micro-labels, pill controls).
 *
 * All editing state comes from `useMaskEditor` (lifted into Workspace) — this
 * component is purely the control surface for it.
 */

import React from 'react';
import type { MaskEditor, EditorTool } from '../hooks/useMaskEditor';

export interface MaskEditorPanelProps {
  editor: MaskEditor;
  collapsed: boolean;
  onToggleCollapse: () => void;
  /** True once inference has produced a mask to edit. */
  hasResult: boolean;
  /** Shown as a compact case summary above the toolkit. */
  summary?: { modelId: string; sessionId: string; dimensions: string };
  /** Editing only binds to the single-model view; other modes are read-only. */
  editingAvailable: boolean;
  onSaveMask: () => void;
}

/** Uppercase mono micro-label, matching the left panel. */
const SectionLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
    {children}
  </h2>
);

/** Labelled slider row with a mono value readout. */
const SliderRow: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  format?: (n: number) => string;
  onChange: (n: number) => void;
}> = ({ label, value, min, max, step = 1, format, onChange }) => (
  <div className="flex items-center gap-2">
    <label className="text-[11px] font-body text-muted w-16 flex-shrink-0">{label}</label>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={label}
      onChange={(e) => onChange(Number(e.target.value))}
      className="flex-1 accent-accent min-w-0"
    />
    <span className="text-[11px] font-mono text-text w-10 text-right flex-shrink-0">
      {format ? format(value) : value}
    </span>
  </div>
);

const TOOLS: { id: EditorTool; label: string; icon: React.ReactNode }[] = [
  {
    id: 'brush',
    label: 'Brush',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d="M9.5 14.5 3 21l6.5-1.5L20 9a2.8 2.8 0 0 0-4-4L5.5 15.5z" />
      </svg>
    ),
  },
  {
    id: 'eraser',
    label: 'Eraser',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d="M4 16.5 10.5 10l7 7-3 3H7z" />
        <path d="M10.5 10 15 5.5a2.1 2.1 0 0 1 3 0l3 3a2.1 2.1 0 0 1 0 3L17.5 17" />
      </svg>
    ),
  },
  {
    id: 'lasso',
    label: 'Lasso',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <ellipse cx="12" cy="9" rx="8" ry="5.5" />
        <path d="M7 13.5c-1 2 0 4 2 4.5" />
      </svg>
    ),
  },
  {
    id: 'fill',
    label: 'Fill',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d="M4 12 12 4l8 8-8 8z" />
        <path d="M20 15c1.2 1.6 1.8 2.7 1.8 3.4a1.8 1.8 0 1 1-3.6 0c0-.7.6-1.8 1.8-3.4z" />
      </svg>
    ),
  },
  {
    id: 'pan',
    label: 'Pan',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d="M12 3v9M12 21v-4M3 12h9M21 12h-4" />
        <circle cx="12" cy="12" r="2.5" />
      </svg>
    ),
  },
];

/** Small square icon button used by the history / zoom rows. */
const IconButton: React.FC<{
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}> = ({ label, disabled, onClick, children }) => (
  <button
    type="button"
    title={label}
    aria-label={label}
    disabled={disabled}
    onClick={onClick}
    className={[
      'flex-1 flex items-center justify-center rounded-md border py-1.5',
      'transition-colors duration-panel ease-out',
      disabled
        ? 'border-border/50 text-muted/40 cursor-not-allowed'
        : 'border-border text-muted hover:text-text hover:border-accent/50',
    ].join(' ')}
  >
    {children}
  </button>
);

export const MaskEditorPanel: React.FC<MaskEditorPanelProps> = ({
  editor,
  collapsed,
  onToggleCollapse,
  hasResult,
  summary,
  editingAvailable,
  onSaveMask,
}) => {
  const collapseChevron = (expandDirection: 'in' | 'out') => (
    <button
      type="button"
      onClick={onToggleCollapse}
      aria-label={expandDirection === 'out' ? 'Expand mask editor panel' : 'Collapse mask editor panel'}
      className="text-muted hover:text-text transition-colors duration-panel ease-out"
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d={expandDirection === 'out' ? 'M15 6l-6 6 6 6' : 'M9 6l6 6-6 6'} />
      </svg>
    </button>
  );

  if (collapsed) {
    return (
      <aside
        aria-label="Mask editor (collapsed)"
        className="absolute inset-y-0 right-0 z-20 flex flex-col items-center bg-surface border-l border-border py-4"
        style={{ width: 'var(--control-panel-collapsed)' }}
      >
        {collapseChevron('out')}
      </aside>
    );
  }

  const disabled = !hasResult || !editor.ready;

  return (
    <aside
      aria-label="Mask editor"
      className="absolute inset-y-0 right-0 z-20 flex flex-col overflow-hidden bg-surface border-l border-border shadow-float"
      style={{ width: 'var(--results-panel-width)', minWidth: 'var(--results-panel-width)' }}
    >
      <div className="flex items-center justify-between px-4 pt-4 pb-2 flex-shrink-0">
        <span className="text-[11px] font-heading uppercase tracking-wider text-muted">Mask editor</span>
        {collapseChevron('in')}
      </div>

      <div className="flex-1 overflow-y-auto px-4 pb-4 flex flex-col gap-5">
        {/* Compact case summary — no metrics here; those live on the main scroll */}
        {summary && (
          <dl className="flex flex-col gap-1 rounded-lg border border-border bg-surface-2 px-3 py-2">
            {[
              { dt: 'Model', dd: summary.modelId },
              { dt: 'Session', dd: summary.sessionId },
              { dt: 'Size', dd: summary.dimensions },
            ].map(({ dt, dd }) => (
              <div key={dt} className="flex items-center justify-between gap-2">
                <dt className="text-[11px] font-body text-muted">{dt}</dt>
                <dd className="text-[11px] font-mono text-text truncate">{dd}</dd>
              </div>
            ))}
          </dl>
        )}

        {disabled && (
          <p className="text-sm font-body text-muted">
            Run inference to edit the predicted mask here.
          </p>
        )}

        {!disabled && !editingAvailable && (
          <p className="text-[11px] font-body text-warning">
            Switch the view mode to “Single model” to paint — wipe and side-by-side are read-only.
          </p>
        )}

        <fieldset disabled={disabled} className="contents">
          {/* Tools */}
          <section aria-label="Tools">
            <SectionLabel>Tools</SectionLabel>
            <div className="grid grid-cols-5 gap-1">
              {TOOLS.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  title={t.label}
                  aria-label={t.label}
                  aria-pressed={editor.tool === t.id}
                  disabled={disabled}
                  onClick={() => editor.setTool(t.id)}
                  className={[
                    'flex items-center justify-center rounded-md border py-2',
                    'transition-colors duration-panel ease-out',
                    disabled
                      ? 'border-border/50 text-muted/40 cursor-not-allowed'
                      : editor.tool === t.id
                        ? 'border-accent bg-accent/15 text-accent'
                        : 'border-border text-muted hover:text-text hover:border-accent/50',
                  ].join(' ')}
                >
                  {t.icon}
                </button>
              ))}
            </div>

            <div className="mt-2 flex flex-col gap-2">
              <SliderRow
                label="Brush"
                value={editor.brushSize}
                min={1}
                max={64}
                onChange={editor.setBrushSize}
                format={(n) => `${n}px`}
              />
              <SliderRow
                label="Eraser"
                value={editor.eraserSize}
                min={1}
                max={64}
                onChange={editor.setEraserSize}
                format={(n) => `${n}px`}
              />
            </div>

            {editor.tool === 'lasso' && (
              <div className="mt-2 flex rounded-lg border border-border overflow-hidden">
                {(['add', 'remove'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    aria-pressed={editor.lassoMode === m}
                    onClick={() => editor.setLassoMode(m)}
                    className={[
                      'flex-1 py-1.5 text-xs font-body capitalize transition-colors duration-panel ease-out',
                      editor.lassoMode === m ? 'bg-accent text-white' : 'text-muted hover:text-text',
                    ].join(' ')}
                  >
                    {m}
                  </button>
                ))}
              </div>
            )}
          </section>

          {/* Label / class palette — the same reserved mask colours as the overlay */}
          <section aria-label="Label">
            <SectionLabel>Label</SectionLabel>
            {editor.palette.length === 0 ? (
              <p className="text-[11px] font-body text-muted">No classes yet.</p>
            ) : (
              <div className="flex flex-col gap-1">
                {editor.palette.map((p) => (
                  <button
                    key={p.structure}
                    type="button"
                    aria-pressed={editor.activeColor === p.color}
                    disabled={disabled}
                    onClick={() => editor.setActiveColor(p.color)}
                    className={[
                      'w-full flex items-center gap-2 rounded-md border px-2 py-1.5 text-left',
                      'transition-colors duration-panel ease-out',
                      editor.activeColor === p.color
                        ? 'border-accent bg-accent/10 text-text'
                        : 'border-border text-muted hover:border-accent/50',
                    ].join(' ')}
                  >
                    <span
                      className="w-3.5 h-3.5 rounded-sm flex-shrink-0 border border-black/20"
                      style={{ backgroundColor: p.color }}
                      aria-hidden="true"
                    />
                    <span className="text-xs font-body truncate">{p.label}</span>
                  </button>
                ))}
              </div>
            )}
          </section>

          {/* Display — overlay opacity, layer visibility, W/L, zoom */}
          <section aria-label="Display">
            <SectionLabel>Display</SectionLabel>
            <div className="flex flex-col gap-2">
              <SliderRow
                label="Opacity"
                value={editor.opacity}
                min={0}
                max={1}
                step={0.05}
                onChange={editor.setOpacity}
                format={(n) => `${Math.round(n * 100)}%`}
              />
              <label className="flex items-center gap-2 text-[11px] font-body text-muted">
                <input
                  type="checkbox"
                  checked={editor.maskVisible}
                  onChange={(e) => editor.setMaskVisible(e.target.checked)}
                  className="accent-accent"
                />
                Show mask overlay
              </label>
              <SliderRow
                label="Level"
                value={editor.windowLevel}
                min={0}
                max={400}
                onChange={editor.setWindowLevel}
              />
              <SliderRow
                label="Window"
                value={editor.windowWidth}
                min={0}
                max={400}
                onChange={editor.setWindowWidth}
              />
              <div className="flex items-center gap-1">
                <span className="text-[11px] font-body text-muted w-16 flex-shrink-0">Zoom</span>
                <IconButton label="Zoom out" onClick={() => editor.setZoom(Math.max(0.5, editor.zoom - 0.25))}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                    <path d="M5 12h14" />
                  </svg>
                </IconButton>
                <span className="text-[11px] font-mono text-text w-10 text-center flex-shrink-0">
                  {editor.zoom.toFixed(2)}×
                </span>
                <IconButton label="Zoom in" onClick={() => editor.setZoom(Math.min(6, editor.zoom + 0.25))}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                    <path d="M12 5v14M5 12h14" />
                  </svg>
                </IconButton>
                <IconButton label="Reset view" onClick={editor.resetView}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                    <path d="M3 12a9 9 0 1 0 3-6.7" />
                    <path d="M3 4v5h5" />
                  </svg>
                </IconButton>
              </div>
            </div>
          </section>

          {/* History */}
          <section aria-label="History">
            <SectionLabel>History</SectionLabel>
            <div className="flex gap-1">
              <IconButton label="Undo" disabled={disabled || !editor.canUndo} onClick={editor.undo}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="M9 14 4 9l5-5" />
                  <path d="M4 9h10a6 6 0 0 1 0 12h-3" />
                </svg>
              </IconButton>
              <IconButton label="Redo" disabled={disabled || !editor.canRedo} onClick={editor.redo}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="m15 14 5-5-5-5" />
                  <path d="M20 9H10a6 6 0 0 0 0 12h3" />
                </svg>
              </IconButton>
            </div>
            <button
              type="button"
              disabled={disabled || !editor.dirty}
              onClick={editor.resetToPrediction}
              className={[
                'mt-2 w-full rounded-md border py-1.5 text-xs font-body',
                'transition-colors duration-panel ease-out',
                disabled || !editor.dirty
                  ? 'border-border/50 text-muted/40 cursor-not-allowed'
                  : 'border-border text-muted hover:text-text hover:border-accent/50',
              ].join(' ')}
            >
              Reset to AI prediction
            </button>
          </section>

          {/* Save / export */}
          <section aria-label="Save mask">
            <SectionLabel>Save</SectionLabel>
            <button
              type="button"
              disabled={disabled}
              onClick={onSaveMask}
              className={[
                'w-full rounded-lg border py-2 text-sm font-heading font-semibold',
                'transition-colors duration-panel ease-out',
                disabled
                  ? 'border-border/50 text-muted/40 cursor-not-allowed'
                  : 'border-accent/40 text-accent hover:bg-accent/5',
              ].join(' ')}
            >
              Download edited mask (PNG)
            </button>
            {/* TODO(backend): no endpoint exists for persisting a corrected mask
                (the FastAPI app exposes /predict, /compare, /infer, /interpret and
                /chat only). Server-side persistence is deliberately NOT invented
                here — export is client-side until an API contract is agreed. */}
            <p className="text-[11px] font-body text-muted mt-2">
              Saves locally. Server-side persistence needs a backend endpoint that does not exist yet.
            </p>
          </section>
        </fieldset>
      </div>
    </aside>
  );
};

export default MaskEditorPanel;
