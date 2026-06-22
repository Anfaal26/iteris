/**
 * ControlPanel — left sidebar with upload, preprocessing, model selection,
 * and analysis controls (spec §6 Zone 1–4).
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  SampleImageTile,
  PreprocessingStepIndicator,
  ModelCard,
} from '@/components';
import type {
  SampleImage,
  ModelRecord,
  ModelId,
  DatasetId,
  ViewMode,
} from '@/api/contract';
import type { StepStatus } from '@/components';
import { ROUTES } from '@/routes';

/** Props for the ControlPanel component. */
export interface ControlPanelProps {
  samples: SampleImage[];
  models: ModelRecord[];
  selectedModel: ModelId;
  selectedDataset: DatasetId;
  viewMode: ViewMode;
  playbackEnabled: boolean;
  loading: boolean;
  activeImageLabel?: string;
  onModelSelect: (id: ModelId) => void;
  onDatasetChange: (id: DatasetId) => void;
  onViewModeChange: (m: ViewMode) => void;
  onPlaybackToggle: (v: boolean) => void;
  onSampleSelect: (sample: SampleImage) => void;
  /** dataUrl is the full `data:<mime>;base64,...` string from FileReader. */
  onImageUpload: (dataUrl: string, filename: string) => void;
  onRunInference: () => void;
}

const MODEL_GROUPS: { label: string; family: string }[] = [
  { label: 'Discrete DRL', family: 'discrete-drl' },
  { label: 'Continuous DRL', family: 'continuous-drl' },
  { label: 'Baseline', family: 'baseline' },
];

const PREPROCESSING_LABELS = ['Load', 'Normalise', 'Resize', 'Augment', 'Ready'];

function usePreprocessingSteps(active: boolean): StepStatus[] {
  const [steps, setSteps] = useState<StepStatus[]>(
    PREPROCESSING_LABELS.map((label) => ({ label, done: false })),
  );

  useEffect(() => {
    if (!active) {
      setSteps(PREPROCESSING_LABELS.map((label) => ({ label, done: false })));
      return;
    }
    let cancelled = false;
    const interval = 800 / PREPROCESSING_LABELS.length;
    PREPROCESSING_LABELS.forEach((_, idx) => {
      const start = Date.now();
      setTimeout(() => {
        if (cancelled) return;
        setSteps((prev) =>
          prev.map((s, i) =>
            i === idx
              ? { ...s, done: true, elapsedMs: Math.round(Date.now() - start) }
              : s,
          ),
        );
      }, interval * (idx + 1));
    });
    return () => { cancelled = true; };
  }, [active]);

  return steps;
}

/** Left control panel with upload, preprocessing, model selection, and controls. */
export const ControlPanel: React.FC<ControlPanelProps> = ({
  samples,
  models,
  selectedModel,
  selectedDataset,
  viewMode,
  playbackEnabled,
  loading,
  activeImageLabel,
  onModelSelect,
  onDatasetChange,
  onViewModeChange,
  onPlaybackToggle,
  onSampleSelect,
  onImageUpload,
  onRunInference,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const steps = usePreprocessingSteps(!!activeImageLabel);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        onImageUpload(reader.result as string, file.name);
      };
      reader.readAsDataURL(file);
    },
    [onImageUpload],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      onImageUpload(reader.result as string, file.name);
    };
    reader.readAsDataURL(file);
  };

  return (
    <aside
      aria-label="Control panel"
      className="flex flex-col gap-4 h-full overflow-y-auto bg-surface border-r border-border p-4"
      style={{ width: 'var(--control-panel-width)', minWidth: 'var(--control-panel-width)' }}
    >
      {/* Zone 1 — Upload */}
      <section aria-label="Image upload">
        <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
          Image
        </h2>

        {activeImageLabel ? (
          <div className="border border-border rounded-lg p-3 bg-bg flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-body text-text truncate">{activeImageLabel}</span>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="text-xs font-body text-accent hover:underline ml-2 flex-shrink-0"
              >
                Replace
              </button>
            </div>
          </div>
        ) : (
          <div
            role="region"
            aria-label="Drop zone for DICOM, NIfTI, or PNG files"
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
            className={[
              'border-2 border-dashed rounded-lg p-4 flex flex-col items-center gap-2 cursor-pointer',
              'transition-colors duration-panel ease-out',
              isDragOver ? 'border-accent bg-accent/5' : 'border-border hover:border-accent/50',
            ].join(' ')}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="text-muted"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="text-xs font-body text-muted text-center">
              Drop DICOM, NIfTI, or PNG
            </p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".dcm,.nii,.nii.gz,.png,.jpg"
          className="hidden"
          aria-label="Upload image file"
          onChange={handleFileChange}
        />

        {/* Sample grid */}
        <p className="text-xs font-body text-muted mt-3 mb-2">Or choose a sample:</p>
        <div className="grid grid-cols-2 gap-2">
          {samples.slice(0, 6).map((s) => (
            <SampleImageTile
              key={s.id}
              image={s}
              onSelect={() => onSampleSelect(s)}
            />
          ))}
        </div>
      </section>

      {/* Zone 2 — Preprocessing */}
      {activeImageLabel && (
        <section aria-label="Preprocessing status">
          <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-2">
            Preprocessing
          </h2>
          <PreprocessingStepIndicator steps={steps} className="w-full" />
        </section>
      )}

      {/* Zone 3 — Model selection */}
      <section aria-label="Model selection">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider">
            Model
          </h2>
          <a
            href={ROUTES.research}
            className="text-xs font-body text-accent hover:underline"
          >
            View full research results →
          </a>
        </div>
        <div className="flex flex-col gap-2">
          {MODEL_GROUPS.map((group) => {
            const groupModels = models.filter((m) => m.family === group.family);
            if (groupModels.length === 0) return null;
            return (
              <div key={group.family}>
                <p className="text-xs font-body text-muted mb-1">{group.label}</p>
                {groupModels.map((m) => (
                  <ModelCard
                    key={m.id}
                    model={m}
                    variant="compact"
                    active={selectedModel === m.id}
                    onSelect={(id) => onModelSelect(id as ModelId)}
                    className="mb-1 w-full"
                  />
                ))}
              </div>
            );
          })}
        </div>
      </section>

      {/* Zone 4 — Analysis Controls */}
      <section aria-label="Analysis controls">
        <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-3">
          Analysis
        </h2>

        {/* Dataset selector */}
        <div className="mb-3">
          <p className="text-xs font-body text-muted mb-1">Dataset</p>
          <div className="flex rounded-lg border border-border overflow-hidden">
            {(['camus', 'brisc'] as DatasetId[]).map((d) => (
              <button
                key={d}
                type="button"
                aria-pressed={selectedDataset === d}
                onClick={() => onDatasetChange(d)}
                className={[
                  'flex-1 py-1.5 text-xs font-body transition-colors duration-panel ease-out',
                  selectedDataset === d
                    ? 'bg-accent text-white'
                    : 'text-muted hover:text-text',
                ].join(' ')}
              >
                {d.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Mode selector */}
        <div className="mb-3">
          <p className="text-xs font-body text-muted mb-1">View mode</p>
          <div className="flex flex-col gap-1">
            {(
              [
                { id: 'single', label: 'Single Model' },
                { id: 'wipe', label: 'Wipe Comparison' },
                { id: 'side-by-side', label: 'Side-by-Side' },
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
                    : 'border border-border text-muted hover:border-accent/50',
                ].join(' ')}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Iteration Playback toggle */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-body text-muted">Iteration Playback</span>
          <button
            type="button"
            role="switch"
            aria-checked={playbackEnabled}
            onClick={() => onPlaybackToggle(!playbackEnabled)}
            className={[
              'relative inline-flex h-5 w-9 items-center rounded-full',
              'transition-colors duration-panel ease-out',
              playbackEnabled ? 'bg-accent' : 'bg-border',
            ].join(' ')}
          >
            <span
              className={[
                'inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow',
                'transition-transform duration-panel ease-out',
                playbackEnabled ? 'translate-x-4' : 'translate-x-0.5',
              ].join(' ')}
            />
          </button>
        </div>

        {/* Upload GT mask */}
        <button
          type="button"
          className="w-full mb-3 py-2 text-xs font-body text-muted border border-dashed border-border rounded-md hover:border-accent/50 transition-colors duration-panel ease-out"
        >
          Upload GT mask
        </button>

        {/* Run Inference */}
        <button
          type="button"
          onClick={onRunInference}
          disabled={loading}
          aria-label="Run inference"
          className={[
            'w-full h-[52px] rounded-lg text-sm font-heading font-semibold text-white',
            'transition-colors duration-panel ease-out',
            loading
              ? 'bg-accent/60 cursor-not-allowed'
              : 'bg-accent hover:bg-accent/90',
          ].join(' ')}
        >
          {loading ? 'Running…' : 'Run Inference'}
        </button>
      </section>
    </aside>
  );
};

export default ControlPanel;
