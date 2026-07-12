/**
 * ImageViewer — centre canvas zone with floating toolbar and mode overlays.
 * Occupies the top of the single-scroll page; Wipe mode compares any two of the
 * sidebar-selected sources (Attention U-Net / GT / DRL).
 */

import React, { useState } from 'react';
import { IterationPlaybackTimeline, ExportButtonGroup } from '@/components';
import type { MaskLayer, ViewMode, IterationStep, CompareResult, WipeSource } from '@/api/contract';
import { structureColor } from '@/tokens';
import { SingleMode } from '../modes/SingleMode';
import { WipeMode, type WipeSide } from '../modes/WipeMode';
import { SideBySideMode } from '../modes/SideBySideMode';

export interface ImageViewerProps {
  anatomyLabel: string;
  /** Full `data:<mime>;base64,...` URL of the active image, for display only. */
  imageB64?: string;
  masks: MaskLayer[];
  baselineMasks: MaskLayer[];
  /** Ground-truth mask preview URL, when attached (a Wipe source). */
  gtMaskUrl?: string;
  viewMode: ViewMode;
  wipeSources: [WipeSource, WipeSource];
  playbackEnabled: boolean;
  stepSequence?: IterationStep[];
  compareResults?: CompareResult[];
  hasResult: boolean;
}

const WIPE_SOURCE_LABEL: Record<WipeSource, string> = {
  'attention-unet': 'Attention U-Net',
  gt: 'GT',
  drl: 'DRL',
};

export const ImageViewer: React.FC<ImageViewerProps> = ({
  anatomyLabel,
  imageB64,
  masks,
  baselineMasks,
  gtMaskUrl,
  viewMode,
  wipeSources,
  playbackEnabled,
  stepSequence,
  compareResults,
  hasResult,
}) => {
  const [windowLevel, setWindowLevel] = useState(200);
  const [windowWidth, setWindowWidth] = useState(200);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [currentStep, setCurrentStep] = useState(0);
  const [visibleStructures, setVisibleStructures] = useState<Set<string>>(
    () => new Set(masks.map((m) => m.structure)),
  );

  React.useEffect(() => {
    setVisibleStructures(new Set(masks.map((m) => m.structure)));
  }, [masks]);

  const toggleStructure = (structureId: string) => {
    setVisibleStructures((prev) => {
      const next = new Set(prev);
      if (next.has(structureId)) next.delete(structureId);
      else next.add(structureId);
      return next;
    });
  };

  const displayMasks =
    playbackEnabled && stepSequence && stepSequence[currentStep]
      ? stepSequence[currentStep].masks
      : masks;

  /** Resolve a wipe source to a renderable side (respects visibility toggles). */
  const resolveSide = (src: WipeSource): WipeSide => {
    if (src === 'gt') {
      return {
        label: WIPE_SOURCE_LABEL.gt,
        overlays: gtMaskUrl ? [{ id: 'gt', imageB64: gtMaskUrl }] : [],
      };
    }
    const layers = src === 'attention-unet' ? baselineMasks : displayMasks;
    return {
      label: WIPE_SOURCE_LABEL[src],
      overlays: layers
        .filter((m) => visibleStructures.has(m.structure))
        .map((m) => ({ id: m.structure, imageB64: m.imageB64 })),
    };
  };

  return (
    <div className="relative flex-1 flex flex-col overflow-hidden bg-bg" aria-label="Image viewer">
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
            left={resolveSide(wipeSources[0])}
            right={resolveSide(wipeSources[1])}
            overlayOpacity={overlayOpacity}
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
        className="absolute bottom-4 inset-x-4 bg-surface border border-border rounded-xl px-4 py-3 flex flex-wrap items-center gap-4 shadow-float"
        aria-label="Viewer toolbar"
      >
        <div className="flex items-center gap-2">
          <label className="text-xs font-body text-muted whitespace-nowrap">W/L</label>
          <input type="range" min="0" max="400" value={windowWidth} onChange={(e) => setWindowWidth(Number(e.target.value))} aria-label="Window width" className="w-20 accent-accent" />
          <input type="range" min="0" max="400" value={windowLevel} onChange={(e) => setWindowLevel(Number(e.target.value))} aria-label="Window level" className="w-20 accent-accent" />
        </div>

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

        <div className="flex items-center gap-2">
          <label className="text-xs font-body text-muted whitespace-nowrap">Opacity</label>
          <input type="range" min="0" max="1" step="0.05" value={overlayOpacity} onChange={(e) => setOverlayOpacity(Number(e.target.value))} aria-label="Overlay opacity" className="w-20 accent-accent" />
        </div>

        {hasResult && (
          <div className="ml-auto">
            <ExportButtonGroup />
          </div>
        )}
      </div>

      {playbackEnabled && stepSequence && stepSequence.length > 0 && (
        <div className="border-t border-border p-3">
          <IterationPlaybackTimeline steps={stepSequence} currentStep={currentStep} onStepChange={setCurrentStep} />
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
