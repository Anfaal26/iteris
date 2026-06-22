/**
 * Workspace — three-zone clinical workstation (spec §6 + §7).
 * Left: ControlPanel, Centre: ImageViewer, Right: ResultsPanel.
 */

import { useState, useEffect } from 'react';
import { Navbar } from '@/components';
import type {
  ModelId,
  DatasetId,
  ViewMode,
  SampleImage,
  ModelRecord,
  PredictResponse,
  CompareResponse,
} from '@/api/contract';
import { api } from '@/api/client';
import { ROUTES } from '@/routes';
import { ControlPanel } from './panels/ControlPanel';
import { ImageViewer } from './panels/ImageViewer';
import { ResultsPanel } from './panels/ResultsPanel';

const NAV_ITEMS = [
  { label: 'Workspace', href: ROUTES.workspace },
  { label: 'Model Library', href: ROUTES.models },
  { label: 'Dataset Explorer', href: ROUTES.datasets },
];

/** Default placeholder b64 PNG (1×1 grey pixel). */
const PLACEHOLDER_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

/**
 * Main Workspace page with three-zone layout (spec §6).
 */
export default function Workspace() {
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);

  const [selectedModel, setSelectedModel] = useState<ModelId>('unet-baseline');
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('camus');
  const [viewMode, setViewMode] = useState<ViewMode>('single');
  const [playbackEnabled, setPlaybackEnabled] = useState(false);
  const [loading, setLoading] = useState(false);

  const [activeImage, setActiveImage] = useState<
    { b64: string; previewUrl: string; label: string } | null
  >(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [baselineResult, setBaselineResult] = useState<PredictResponse | null>(null);
  const [compareResponse, setCompareResponse] = useState<CompareResponse | null>(null);
  const [gtMask, setGtMask] = useState<{ b64: string; label: string } | null>(null);

  useEffect(() => {
    api.models().then(setModels).catch(() => {});
    api.samples().then(setSamples).catch(() => {});
  }, []);

  const handleSampleSelect = (sample: SampleImage) => {
    setActiveImage({
      b64: PLACEHOLDER_B64,
      previewUrl: `data:image/png;base64,${PLACEHOLDER_B64}`,
      label: `${sample.anatomy} (${sample.modality})`,
    });
    setResult(null);
    setBaselineResult(null);
    setCompareResponse(null);
  };

  /** dataUrl is the full `data:<mime>;base64,...` string straight from FileReader. */
  const handleImageUpload = (dataUrl: string, filename: string) => {
    const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
    setActiveImage({ b64, previewUrl: dataUrl, label: filename });
    setResult(null);
    setBaselineResult(null);
    setCompareResponse(null);
    setGtMask(null);
  };

  const handleGtMaskUpload = (dataUrl: string) => {
    const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
    setGtMask({ b64, label: 'attached' });
  };

  const handleRunInference = async () => {
    if (!activeImage) return;
    setLoading(true);
    try {
      const predictResult = await api.predict({
        imageB64: activeImage.b64,
        modelId: selectedModel,
        dataset: selectedDataset,
        mode: viewMode,
        playback: playbackEnabled,
        gtMaskB64: gtMask?.b64,
      });
      setResult(predictResult);

      // If wipe mode, also get baseline result (skip if it's the same model).
      if ((viewMode === 'wipe' || viewMode === 'side-by-side') && selectedModel !== 'unet-baseline') {
        const baselineRes = await api.predict({
          imageB64: activeImage.b64,
          modelId: 'unet-baseline',
          dataset: selectedDataset,
          mode: 'single',
        });
        setBaselineResult(baselineRes);
      }

      // If side-by-side, compare across whatever models are actually deployed
      // (DRL agents are excluded server-side until trained — see models.yaml).
      if (viewMode === 'side-by-side') {
        const deployedIds = models.filter((m) => m.deployed).map((m) => m.id);
        const compareRes = await api.compare({
          imageB64: activeImage.b64,
          modelIds: deployedIds,
          dataset: selectedDataset,
        });
        setCompareResponse(compareRes);
      }
    } catch {
      // Error is surfaced by the loading state clearing
    } finally {
      setLoading(false);
    }
  };

  const handleExportJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.metrics, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `iteris-metrics-${result.sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-bg">
      <Navbar
        variant="light"
        navItems={NAV_ITEMS}
      />

      {/* Three-zone layout below navbar */}
      <div
        className="flex flex-row flex-1 overflow-hidden"
        style={{ marginTop: 'var(--navbar-height)' }}
      >
        <ControlPanel
          samples={samples}
          models={models}
          selectedModel={selectedModel}
          selectedDataset={selectedDataset}
          viewMode={viewMode}
          playbackEnabled={playbackEnabled}
          loading={loading}
          activeImageLabel={activeImage?.label}
          onModelSelect={setSelectedModel}
          onDatasetChange={setSelectedDataset}
          onViewModeChange={setViewMode}
          onPlaybackToggle={setPlaybackEnabled}
          onSampleSelect={handleSampleSelect}
          onImageUpload={handleImageUpload}
          onGtMaskUpload={handleGtMaskUpload}
          gtMaskLabel={gtMask?.label}
          onRunInference={handleRunInference}
        />

        <ImageViewer
          anatomyLabel={activeImage?.label ?? 'Select an image'}
          imageB64={activeImage?.previewUrl}
          masks={result?.masks ?? []}
          baselineMasks={baselineResult?.masks ?? []}
          viewMode={viewMode}
          playbackEnabled={playbackEnabled}
          stepSequence={result?.stepSequence}
          compareResults={compareResponse?.results}
          baselineMetrics={baselineResult?.metrics}
          drlMetrics={result?.metrics}
          hasResult={!!result}
        />

        <ResultsPanel
          result={result}
          onExportJson={handleExportJson}
        />
      </div>
    </div>
  );
}
