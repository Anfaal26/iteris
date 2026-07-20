/**
 * Workspace — clinical workstation (redesign).
 *
 * Layout: a collapsible ControlPanel (left), a centre column, and a collapsible
 * MaskEditorPanel (right). Both side panels are ABSOLUTELY positioned over
 * permanently-reserved rails, so collapsing or expanding either one never
 * resizes or reflows the centre — its width is a function of the viewport only.
 *
 * The centre column's scroll is state-driven: locked (`overflow-hidden`) until
 * inference completes, then `overflow-y-auto`, revealing the detailed stats and
 * the "Ask about this result" chat below the fold. A persistent chat bubble
 * opens the same thread from any scroll position.
 *
 * Dataset/modality are auto-detected from the uploaded scan (no manual toggle);
 * the data regime default flips with the model family (DRL → low, U-Net → high).
 */

import { useState, useEffect, useRef } from 'react';
import { Navbar } from '@/components';
import type {
  ModelId,
  DatasetId,
  Regime,
  ViewMode,
  WipeSource,
  SampleImage,
  PredictResponse,
  CompareResponse,
  ChatMessage,
} from '@/api/contract';
import {
  defaultRegime,
  availableRegimes,
  isCombinationAvailable,
  AVAILABLE_COMBINATIONS,
} from '@/api/contract';
import { api, ApiError } from '@/api/client';
import { detectDataset, type DetectionResult } from '@/lib/detectDataset';
import { ROUTES } from '@/routes';
import { useMaskEditor, NO_MASKS } from './hooks/useMaskEditor';
import { ControlPanel } from './panels/ControlPanel';
import { ImageViewer } from './panels/ImageViewer';
import { MaskEditorPanel } from './panels/MaskEditorPanel';
import { StatsSection } from './panels/StatsSection';
import { ChatThread } from './panels/ChatThread';
import { ChatBubble } from './panels/ChatBubble';

const NAV_ITEMS = [
  { label: 'Workspace', href: ROUTES.workspace },
  { label: 'Model Library', href: ROUTES.models },
  { label: 'Dataset Explorer', href: ROUTES.datasets },
];

const PLACEHOLDER_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

const CHAT_SUGGESTIONS = [
  'Why did DRL improve the LV boundary?',
  'Compare to Attention U-Net',
  'What drove the Hausdorff distance?',
];

/** Strip the `data:*;base64,` prefix if present. */
function toBareB64(dataUrl: string): string {
  return dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
}

export default function Workspace() {
  const [samples, setSamples] = useState<SampleImage[]>([]);

  // DRL is the default family, selected & expanded (redesign) → default low regime.
  const [selectedModel, setSelectedModel] = useState<ModelId>('dueling-dqn');
  const [selectedRegime, setSelectedRegime] = useState<Regime>(defaultRegime('dueling-dqn'));
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('single');
  const [wipeSources, setWipeSources] = useState<[WipeSource, WipeSource]>(['attention-unet', 'drl']);
  const [playbackEnabled] = useState(false);
  const [loading, setLoading] = useState(false);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);

  const [activeImage, setActiveImage] = useState<
    { b64: string; previewUrl: string; label: string } | null
  >(null);
  const [gtMask, setGtMask] = useState<{ b64: string; previewUrl: string; label: string } | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [baselineResult, setBaselineResult] = useState<PredictResponse | null>(null);
  const [compareResponse, setCompareResponse] = useState<CompareResponse | null>(null);

  // Shared chat thread (inline section + floating bubble render the same state).
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatStreaming, setChatStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  /** Last question asked, so a failed turn can be retried verbatim. */
  const lastQuestionRef = useRef<string | null>(null);

  const statsRef = useRef<HTMLDivElement>(null);

  // Manual mask-correction state, shared by the centre canvas and the right
  // panel toolkit. NO_MASKS is a module constant so the hook's mask-loading
  // effect doesn't re-fire on every render before a result exists.
  const editor = useMaskEditor(
    result?.masks ?? NO_MASKS,
    result?.imageWidth ?? 256,
    result?.imageHeight ?? 256,
  );

  // Dataset used for availability gating: detected, else the DRL default target.
  const dataset: DatasetId = detection?.dataset ?? 'camus';

  useEffect(() => {
    api.samples().then(setSamples).catch(() => {});
  }, []);

  // When the model changes, snap the regime to the family default, falling back
  // to whatever single regime is actually trained for this (dataset, model).
  useEffect(() => {
    const preferred = defaultRegime(selectedModel);
    if (isCombinationAvailable(dataset, selectedModel, preferred)) {
      setSelectedRegime(preferred);
      return;
    }
    const [only] = availableRegimes(dataset, selectedModel);
    if (only) setSelectedRegime(only);
    else setSelectedRegime(preferred);
  }, [selectedModel, dataset]);

  const resetResults = () => {
    setResult(null);
    setBaselineResult(null);
    setCompareResponse(null);
    setChatMessages([]);
    setChatError(null);
  };

  const handleSampleSelect = (sample: SampleImage) => {
    setActiveImage({
      b64: PLACEHOLDER_B64,
      previewUrl: `data:image/png;base64,${PLACEHOLDER_B64}`,
      label: `${sample.anatomy} (${sample.modality})`,
    });
    setDetection({
      dataset: sample.dataset,
      modality: sample.modality,
      label: sample.dataset === 'camus' ? 'cardiac echo (CAMUS)' : 'brain MRI (BRISC)',
      source: 'filename',
      confidence: 'high',
    });
    resetResults();
  };

  const handleScanUpload = async (dataUrl: string, file: File) => {
    setActiveImage({ b64: toBareB64(dataUrl), previewUrl: dataUrl, label: file.name });
    setGtMask(null);
    resetResults();
    try {
      setDetection(await detectDataset(file));
    } catch {
      setDetection(null);
    }
  };

  const handleGtMaskUpload = (dataUrl: string, file: File) => {
    setGtMask({ b64: toBareB64(dataUrl), previewUrl: dataUrl, label: file.name });
  };

  const handleRunInference = async () => {
    if (!activeImage) return;
    setLoading(true);
    setChatMessages([]);
    setChatError(null);
    try {
      const predictResult = await api.predict({
        imageB64: activeImage.b64,
        modelId: selectedModel,
        dataset,
        regime: selectedRegime,
        mode: viewMode,
        playback: playbackEnabled,
        gtMaskB64: gtMask?.b64,
      });
      setResult(predictResult);

      // Baseline reference for the Δ column / wipe, unless the baseline IS the run.
      if (selectedModel !== 'unet-baseline') {
        const baselineRes = await api.predict({
          imageB64: activeImage.b64,
          modelId: 'unet-baseline',
          dataset,
          regime: 'high',
          mode: 'single',
          gtMaskB64: gtMask?.b64,
        });
        setBaselineResult(baselineRes);
      } else {
        setBaselineResult(null);
      }

      if (viewMode === 'side-by-side') {
        // Compare across every model with a live checkpoint for this dataset.
        const ids = Array.from(
          new Set(
            AVAILABLE_COMBINATIONS.filter((c) => c.dataset === dataset).map((c) => c.modelId),
          ),
        );
        const compareRes = await api.compare({
          imageB64: activeImage.b64,
          modelIds: ids,
          dataset,
          regime: selectedRegime,
        });
        setCompareResponse(compareRes);
      }
    } catch {
      // Error surfaces via loading clearing; a toast layer handles messaging.
    } finally {
      setLoading(false);
    }
  };

  const handleExportJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.metrics, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `iteris-metrics-${result.sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  /**
   * Save the manually-corrected mask. There is no backend endpoint for
   * persisting an edited mask (the API exposes /predict, /compare, /infer,
   * /interpret, /chat only), so this downloads the composited PNG locally
   * rather than inventing a server contract. See MaskEditorPanel's TODO.
   */
  const handleSaveMask = () => {
    if (!result) return;
    editor.exportPng(`iteris-mask-${result.sessionId}.png`);
  };

  /**
   * Send a chat turn and stream the grounded answer into the thread.
   * `base` is the thread the turn is appended to — retry passes the thread with
   * the failed user turn already removed, so a retry doesn't duplicate it.
   */
  const runChat = async (text: string, base: ChatMessage[]) => {
    if (!result || chatStreaming) return;
    setChatError(null);
    lastQuestionRef.current = text;
    const thread: ChatMessage[] = [...base, { role: 'user', content: text }];
    setChatMessages([...thread, { role: 'assistant', content: '' }]);
    setChatStreaming(true);
    try {
      const stream = api.chat({
        messages: thread,
        context: {
          modelId: result.modelId,
          dataset: result.dataset,
          regime: selectedRegime,
          metrics: result.metrics,
          hasGroundTruth: !!gtMask,
        },
      });
      let acc = '';
      for await (const chunk of stream) {
        acc += chunk;
        setChatMessages([...thread, { role: 'assistant', content: acc }]);
      }
    } catch (cause) {
      // Drop the empty assistant placeholder; the error renders inline instead.
      setChatMessages(thread);
      setChatError(
        cause instanceof ApiError
          ? cause.message
          : 'Could not reach the analysis service. Check your connection and retry.',
      );
    } finally {
      setChatStreaming(false);
    }
  };

  const sendChat = (text: string) => runChat(text, chatMessages);

  /** Re-send the last question after a failure. */
  const retryChat = () => {
    const question = lastQuestionRef.current;
    if (!question || chatStreaming) return;
    const last = chatMessages[chatMessages.length - 1];
    runChat(question, last?.role === 'user' ? chatMessages.slice(0, -1) : chatMessages);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-bg">
      <Navbar variant="light" navItems={NAV_ITEMS} />

      <div
        className="relative flex flex-row flex-1 overflow-hidden"
        style={{ marginTop: 'var(--navbar-height)' }}
      >
        {/* Left rail — reserved track. The panel itself overlays this (and the
            centre when expanded), so the centre column's box never changes. */}
        <div
          aria-hidden="true"
          className="flex-shrink-0"
          style={{ width: 'var(--control-panel-collapsed)' }}
        />

        <ControlPanel
          samples={samples}
          selectedModel={selectedModel}
          dataset={dataset}
          detection={detection}
          selectedRegime={selectedRegime}
          viewMode={viewMode}
          wipeSources={wipeSources}
          loading={loading}
          collapsed={sidebarCollapsed}
          scanLabel={activeImage?.label}
          gtMaskLabel={gtMask?.label}
          onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
          onModelSelect={setSelectedModel}
          onRegimeChange={setSelectedRegime}
          onViewModeChange={setViewMode}
          onWipeSourcesChange={setWipeSources}
          onSampleSelect={handleSampleSelect}
          onScanUpload={handleScanUpload}
          onGtMaskUpload={handleGtMaskUpload}
          onRunInference={handleRunInference}
        />

        {/* Centre — scroll unlocks only once inference has produced a result */}
        <main
          className={[
            'flex-1 min-w-0 bg-bg',
            result ? 'overflow-y-auto' : 'overflow-hidden',
          ].join(' ')}
        >
          {/* Viewer: fills the locked first screen, then yields to ~60vh so the
              stats section peeks in and invites the (now unlocked) scroll. */}
          <div
            className={[
              'flex flex-col',
              result ? 'h-[60vh] min-h-[360px]' : 'h-full',
            ].join(' ')}
          >
            <ImageViewer
              anatomyLabel={activeImage?.label ?? 'Upload a scan to begin'}
              imageB64={activeImage?.previewUrl}
              masks={result?.masks ?? []}
              baselineMasks={baselineResult?.masks ?? []}
              gtMaskUrl={gtMask?.previewUrl}
              viewMode={viewMode}
              wipeSources={wipeSources}
              playbackEnabled={playbackEnabled}
              stepSequence={result?.stepSequence}
              compareResults={compareResponse?.results}
              hasResult={!!result}
              editor={editor}
              onExportJson={handleExportJson}
            />
          </div>

          {/* Stats — lighter surface, a gentle step-up from the near-black viewer */}
          <div ref={statsRef} className="bg-gradient-to-b from-bg to-surface">
            {result ? (
              <StatsSection result={result} baselineResult={baselineResult} />
            ) : (
              <div className="px-6 py-8 text-sm font-body text-muted">
                Run inference to see per-class Dice, IoU, Hausdorff, and the baseline delta.
              </div>
            )}

            {/* Chat — below the stats, absent from the initial viewport */}
            <section aria-label="Ask about this result" className="px-6 pb-10 pt-2 border-t border-border/60">
              <h2 className="text-xs font-heading font-semibold text-muted uppercase tracking-wider mb-3">
                Ask about this result
              </h2>
              <ChatThread
                messages={chatMessages}
                streaming={chatStreaming}
                disabled={!result}
                suggestions={CHAT_SUGGESTIONS}
                onSend={sendChat}
                variant="inline"
                error={chatError}
                onRetry={retryChat}
              />
            </section>
          </div>
        </main>

        {/* Right rail — reserved track, mirroring the left */}
        <div
          aria-hidden="true"
          className="flex-shrink-0"
          style={{ width: 'var(--control-panel-collapsed)' }}
        />

        <MaskEditorPanel
          editor={editor}
          collapsed={resultsCollapsed}
          onToggleCollapse={() => setResultsCollapsed((v) => !v)}
          hasResult={!!result}
          editingAvailable={viewMode === 'single'}
          summary={
            result
              ? {
                  modelId: result.modelId,
                  sessionId: result.sessionId,
                  dimensions: `${result.imageWidth}×${result.imageHeight}`,
                }
              : undefined
          }
          onSaveMask={handleSaveMask}
        />
      </div>

      {/* Persistent chat affordance — same thread, any scroll position */}
      <ChatBubble
        messages={chatMessages}
        streaming={chatStreaming}
        disabled={!result}
        suggestions={CHAT_SUGGESTIONS}
        onSend={sendChat}
        error={chatError}
        onRetry={retryChat}
      />
    </div>
  );
}
