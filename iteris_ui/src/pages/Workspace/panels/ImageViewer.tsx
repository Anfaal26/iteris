/**
 * ImageViewer — centre canvas zone with floating toolbar and mode overlays.
 * Occupies the top of the single-scroll page; Wipe mode compares any two of the
 * sidebar-selected sources (Attention U-Net / GT / DRL).
 */

import React, { useState } from 'react';
import { IterationPlaybackTimeline, ExportButtonGroup } from '@/components';
import type { MaskLayer, ViewMode, IterationStep, CompareResult, WipeSource } from '@/api/contract';
import { structureColor } from '@/tokens';
import type { MaskEditor } from '../hooks/useMaskEditor';
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
  /**
   * Manual mask-editing state. Owns window/level and overlay opacity too — those
   * controls moved to the right-hand toolkit panel, so the viewer only consumes
   * them here rather than holding its own copies.
   */
  editor: MaskEditor;
  /** Export the current metrics JSON (kept reachable from the viewer toolbar). */
  onExportJson?: () => void;
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
  editor,
  onExportJson,
}) => {
  const { windowLevel, windowWidth, opacity: overlayOpacity, blendMode } = editor;
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
            overlayBlend={blendMode}
            windowLevel={windowLevel}
            windowWidth={windowWidth}
            editor={editor}
            editing={hasResult}
          />
        )}
        {viewMode === 'wipe' && (
          <WipeMode
            anatomyLabel={anatomyLabel}
            imageB64={imageB64}
            left={resolveSide(wipeSources[0])}
            right={resolveSide(wipeSources[1])}
            overlayOpacity={overlayOpacity}
            overlayBlend={blendMode}
          />
        )}
        {viewMode === 'side-by-side' && (
          <SideBySideMode
            anatomyLabel={anatomyLabel}
            imageB64={imageB64}
            results={compareResults ?? []}
            visibleStructures={visibleStructures}
            overlayOpacity={overlayOpacity}
            overlayBlend={blendMode}
          />
        )}
      </div>

      {/* Floating toolbar — only once there is something to put in it */}
      {hasResult && (
      <div
        className="absolute bottom-4 inset-x-4 bg-surface border border-border rounded-xl px-4 py-3 flex flex-wrap items-center gap-4 shadow-float"
        aria-label="Viewer toolbar"
      >
        {/* Window/level and overlay opacity live in the right-hand mask toolkit */}
        {viewMode !== 'single' && hasResult && masks.length > 0 && (
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

        <div className="ml-auto">
          <ExportButtonGroup onExportJson={onExportJson} />
        </div>
      </div>
      )}

      {playbackEnabled && stepSequence && stepSequence.length > 0 && (
        <div className="border-t border-border p-3">
          <IterationPlaybackTimeline steps={stepSequence} currentStep={currentStep} onStepChange={setCurrentStep} />
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
