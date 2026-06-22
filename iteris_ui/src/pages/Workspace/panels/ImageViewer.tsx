/**
 * ImageViewer — centre canvas zone with floating toolbar and mode overlays (spec §6).
 */

import React, { useState } from 'react';
import { IterationPlaybackTimeline, ExportButtonGroup } from '@/components';
import type {
  MaskLayer,
  ViewMode,
  IterationStep,
  CompareResult,
  Metrics,
} from '@/api/contract';
import { structureColor } from '@/tokens';
import { SingleMode } from '../modes/SingleMode';
import { WipeMode } from '../modes/WipeMode';
import { SideBySideMode } from '../modes/SideBySideMode';

/** Props for ImageViewer. */
export interface ImageViewerProps {
  anatomyLabel: string;
  /** Full `data:<mime>;base64,...` URL of the active image, for display only. */
  imageB64?: string;
  masks: MaskLayer[];
  baselineMasks: MaskLayer[];
  viewMode: ViewMode;
  playbackEnabled: boolean;
  stepSequence?: IterationStep[];
  compareResults?: CompareResult[];
  baselineMetrics?: Metrics;
  drlMetrics?: Metrics;
  hasResult: boolean;
}

/**
 * Centre image viewer with floating toolbar and mode overlays.
 * Manages W/L sliders, visibility toggles, and opacity locally.
 */
export const ImageViewer: React.FC<ImageViewerProps> = ({
  anatomyLabel,
  imageB64,
  masks,
  baselineMasks,
  viewMode,
  playbackEnabled,
  stepSequence,
  compareResults,
  baselineMetrics,
  drlMetrics,
  hasResult,
}) => {
  const [windowLevel, setWindowLevel] = useState(200);
  const [windowWidth, setWindowWidth] = useState(200);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [currentStep, setCurrentStep] = useState(0);
  const [visibleStructures, setVisibleStructures] = useState<Set<string>>(
    () => new Set(masks.map((m) => m.structure)),
  );

  // Update visible structures when masks change
  React.useEffect(() => {
    setVisibleStructures(new Set(masks.map((m) => m.structure)));
  }, [masks]);

  const toggleStructure = (structureId: string) => {
    setVisibleStructures((prev) => {
      const next = new Set(prev);
      if (next.has(structureId)) {
        next.delete(structureId);
      } else {
        next.add(structureId);
      }
      return next;
    });
  };

  // Determine which masks to display (step-based if playback active)
  const displayMasks =
    playbackEnabled && stepSequence && stepSequence[currentStep]
      ? stepSequence[currentStep].masks
      : masks;

  return (
    <div
      className="relative flex-1 flex flex-col overflow-hidden bg-bg"
      aria-label="Image viewer"
    >
      {/* Main viewer area */}
      <div className="flex-1 relative overflow-hidden">
        {viewMode === 'single' && (
          <SingleMode
            anatomyLabel={anatomyLabel}
            imageB64={imageB64}
            masks={displayMasks}
            visibleStructures={visibleStructures}
            overlayOpacity={overlayOpacity}
            windowLevel={windowLevel}
            windowWidth={windowWidth}
          />
        )}
        {viewMode === 'wipe' && (
          <WipeMode
            anatomyLabel={anatomyLabel}
            imageB64={imageB64}
            baselineMasks={baselineMasks}
            drlMasks={displayMasks}
            visibleStructures={visibleStructures}
            overlayOpacity={overlayOpacity}
            baselineMetrics={baselineMetrics}
            drlMetrics={drlMetrics}
          />
        )}
        {viewMode === 'side-by-side' && (
          <SideBySideMode
            anatomyLabel={anatomyLabel}
            imageB64={imageB64}
            results={compareResults ?? []}
            visibleStructures={visibleStructures}
            overlayOpacity={overlayOpacity}
          />
        )}
      </div>

      {/* Floating toolbar */}
      <div
        className="absolute bottom-4 inset-x-4 bg-surface/95 border border-border rounded-xl px-4 py-3 flex flex-wrap items-center gap-4"
        style={{ backdropFilter: 'blur(8px)' }}
        aria-label="Viewer toolbar"
      >
        {/* W/L sliders */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-body text-muted whitespace-nowrap">
            W/L
          </label>
          <input
            type="range"
            min="0"
            max="400"
            value={windowWidth}
            onChange={(e) => setWindowWidth(Number(e.target.value))}
            aria-label="Window width"
            className="w-20 accent-accent"
          />
          <input
            type="range"
            min="0"
            max="400"
            value={windowLevel}
            onChange={(e) => setWindowLevel(Number(e.target.value))}
            aria-label="Window level"
            className="w-20 accent-accent"
          />
        </div>

        {/* Structure visibility toggles */}
        {hasResult && masks.length > 0 && (
          <div className="flex items-center gap-1.5" role="group" aria-label="Structure visibility">
            {masks.map((mask) => (
              <button
                key={mask.structure}
                type="button"
                aria-pressed={visibleStructures.has(mask.structure)}
                onClick={() => toggleStructure(mask.structure)}
                title={mask.label}
                className={[
                  'w-6 h-6 rounded border-2 transition-opacity duration-panel ease-out',
                  visibleStructures.has(mask.structure) ? 'opacity-100' : 'opacity-30',
                ].join(' ')}
                style={{
                  backgroundColor: structureColor[mask.structure] ?? 'var(--color-muted)',
                  borderColor: structureColor[mask.structure] ?? 'var(--color-muted)',
                }}
              />
            ))}
          </div>
        )}

        {/* Overlay opacity slider */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-body text-muted whitespace-nowrap">
            Opacity
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(Number(e.target.value))}
            aria-label="Overlay opacity"
            className="w-20 accent-accent"
          />
        </div>

        {/* Export button group */}
        {hasResult && (
          <div className="ml-auto">
            <ExportButtonGroup />
          </div>
        )}
      </div>

      {/* Iteration Playback Timeline */}
      {playbackEnabled && stepSequence && stepSequence.length > 0 && (
        <div className="border-t border-border p-3">
          <IterationPlaybackTimeline
            steps={stepSequence}
            currentStep={currentStep}
            onStepChange={setCurrentStep}
          />
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
