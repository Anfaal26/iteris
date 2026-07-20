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
 * the "Ask about this result" chat below the fold.
 *
 * Work is modelled as a list of CASES rather than a single image. A normal
 * single-scan run is a one-case list; a batch run is up to MAX_BATCH_FILES of
 * them, processed one at a time and surfaced as tabs above the viewer. Every
 * downstream panel reads the ACTIVE case, so batch and single share one code
 * path instead of forking into a parallel "batch mode".
 *
 * Dataset/modality are auto-detected per case from the uploaded scan (no manual
 * toggle); the data regime default flips with the model family (DRL → low,
 * U-Net → high).
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
import { ControlPanel, type BatchItemStatus } from './panels/ControlPanel';
import { CaseTabs } from './panels/CaseTabs';
import { ImageViewer } from './panels/ImageViewer';
import { MaskEditorPanel } from './panels/MaskEditorPanel';
import { StatsSection } from './panels/StatsSection';
import { ChatThread } from './panels/ChatThread';

const NAV_ITEMS = [
  { label: 'Workspace', href: ROUTES.workspace },
  { label: 'Model Library', href: ROUTES.models },
  { label: 'Dataset Explorer', href: ROUTES.datasets },
];

/**
 * Batch cap — half the number of inferences the backend can safely serve at
 * once, so a batch run never monopolises the Space. Raise both halves together
 * if the deployment's capacity changes.
 */
const MAX_BATCH_FILES = 5;

const PLACEHOLDER_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

const CHAT_SUGGESTIONS = [
  'Why did DRL improve the LV boundary?',
  'Compare to Attention U-Net',
  'What drove the Hausdorff distance?',
];

/** One scan and everything derived from it. A batch run is just several. */
interface WorkCase {
  id: string;
  label: string;
  /** Bare base64 (no data-URI prefix), as the API expects. */
  imageB64: string;
  /** Full data-URI, for display. */
  previewUrl: string;
  detection: DetectionResult | null;
  status: BatchItemStatus;
  result: PredictResponse | null;
  baselineResult: PredictResponse | null;
}

/** Strip the `data:*;base64,` prefix if present. */
function toBareB64(dataUrl: string): string {
  return dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;
}

/** Reads a File as a data URL. */
function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

let caseSeq = 0;
/** Monotonic case id — also the mask editor's session key. */
function nextCaseId(): string {
  caseSeq += 1;
  return `case-${caseSeq}`;
}

export default function Workspace() {
  const [samples, setSamples] = useState<SampleImage[]>([]);

  // DRL is the default family, selected & expanded (redesign) → default low regime.
  const [selectedModel, setSelectedModel] = useState<ModelId>('dueling-dqn');
  const [selectedRegime, setSelectedRegime] = useState<Regime>(defaultRegime('dueling-dqn'));
  const [viewMode, setViewMode] = useState<ViewMode>('single');
  const [wipeSources, setWipeSources] = useState<[WipeSource, WipeSource]>(['attention-unet', 'drl']);
  const [playbackEnabled] = useState(false);
  const [loading, setLoading] = useState(false);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);

  const [cases, setCases] = useState<WorkCase[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [gtMask, setGtMask] = useState<{ b64: string; previewUrl: string; label: string } | null>(null);
  const [compareResponse, setCompareResponse] = useState<CompareResponse | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatStreaming, setChatStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  /** Last question asked, so a failed turn can be retried verbatim. */
  const lastQuestionRef = useRef<string | null>(null);

  const statsRef = useRef<HTMLDivElement>(null);

  const activeCase = cases[activeIndex] ?? null;
  const result = activeCase?.result ?? null;
  const baselineResult = activeCase?.baselineResult ?? null;
  const detection = activeCase?.detection ?? null;
  const isBatch = cases.length > 1;

  // Dataset used for availability gating: detected, else the DRL default target.
  const dataset: DatasetId = detection?.dataset ?? 'camus';

  // Manual mask-correction state, shared by the centre canvas and the right
  // panel toolkit. NO_MASKS is a module constant so the hook's mask-loading
  // effect doesn't re-fire on every render before a result exists. The case id
  // is the session key, so each batch tab keeps its own edits and undo stack.
  const editor = useMaskEditor(
    result?.masks ?? NO_MASKS,
    result?.imageWidth ?? 256,
    result?.imageHeight ?? 256,
    dataset,
    activeCase?.id ?? 'empty',
  );

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

  /** Replace the queue with a fresh set of cases and clear derived state. */
  const loadCases = (next: WorkCase[]) => {
    setCases(next);
    setActiveIndex(0);
    setCompareResponse(null);
    setChatMessages([]);
    setChatError(null);
  };

  const handleSampleSelect = (sample: SampleImage) => {
    loadCases([
      {
        id: nextCaseId(),
        label: `${sample.anatomy} (${sample.modality})`,
        imageB64: PLACEHOLDER_B64,
        previewUrl: `data:image/png;base64,${PLACEHOLDER_B64}`,
        detection: {
          dataset: sample.dataset,
          modality: sample.modality,
          label: sample.dataset === 'camus' ? 'cardiac echo (CAMUS)' : 'brain MRI (BRISC)',
          source: 'filename',
          confidence: 'high',
        },
        status: 'queued',
        result: null,
        baselineResult: null,
      },
    ]);
  };

  const handleScanUpload = async (dataUrl: string, file: File) => {
    const id = nextCaseId();
    loadCases([
      {
        id,
        label: file.name,
        imageB64: toBareB64(dataUrl),
        previewUrl: dataUrl,
        detection: null,
        status: 'queued',
        result: null,
        baselineResult: null,
      },
    ]);
    setGtMask(null);
    try {
      const detected = await detectDataset(file);
      setCases((prev) => prev.map((c) => (c.id === id ? { ...c, detection: detected } : c)));
    } catch {
      // Detection is best-effort; the CAMUS default still lets the run proceed.
    }
  };

  /** Queue up to MAX_BATCH_FILES scans; extras beyond the cap are dropped. */
  const handleBatchUpload = async (files: File[]) => {
    const accepted = files.slice(0, MAX_BATCH_FILES);
    const built = await Promise.all(
      accepted.map(async (file) => {
        const dataUrl = await readAsDataUrl(file);
        let detected: DetectionResult | null = null;
        try {
          detected = await detectDataset(file);
        } catch {
          detected = null;
        }
        return {
          id: nextCaseId(),
          label: file.name,
          imageB64: toBareB64(dataUrl),
          previewUrl: dataUrl,
          detection: detected,
          status: 'queued' as BatchItemStatus,
          result: null,
          baselineResult: null,
        };
      }),
    );
    setGtMask(null);
    loadCases(built);
  };

  const handleClearBatch = () => loadCases([]);

  /** Patch one case in place by index. */
  const patchCase = (index: number, patch: Partial<WorkCase>) => {
    setCases((prev) => prev.map((c, i) => (i === index ? { ...c, ...patch } : c)));
  };

  /**
   * Run inference over the whole queue, one case at a time.
   *
   * Sequential on purpose: the backend is a single free-tier Space, and firing
   * MAX_BATCH_FILES requests at once is exactly the load the cap exists to
   * avoid. `loading` stays true for the entire queue, so the ▷ button spins
   * until every case has finished.
   *
   * A batch run skips the U-Net baseline pass (and the side-by-side compare)
   * that a single run makes for its Δ-vs-baseline column — those would double
   * or triple the request count for the same cap.
   */
  const handleRunInference = async () => {
    if (cases.length === 0 || loading) return;
    const queue = cases;
    const batch = queue.length > 1;

    setLoading(true);
    setChatMessages([]);
    setChatError(null);
    setCompareResponse(null);
    setCases((prev) =>
      prev.map((c) => ({ ...c, status: 'queued' as BatchItemStatus, result: null, baselineResult: null })),
    );

    for (let i = 0; i < queue.length; i += 1) {
      const item = queue[i];
      const caseDataset: DatasetId = item.detection?.dataset ?? 'camus';
      patchCase(i, { status: 'running' });
      try {
        const predictResult = await api.predict({
          imageB64: item.imageB64,
          modelId: selectedModel,
          dataset: caseDataset,
          regime: selectedRegime,
          mode: viewMode,
          playback: playbackEnabled,
          gtMaskB64: batch ? undefined : gtMask?.b64,
        });

        let baseline: PredictResponse | null = null;
        if (!batch && selectedModel !== 'unet-baseline') {
          baseline = await api.predict({
            imageB64: item.imageB64,
            modelId: 'unet-baseline',
            dataset: caseDataset,
            regime: 'high',
            mode: 'single',
            gtMaskB64: gtMask?.b64,
          });
        }

        patchCase(i, { status: 'done', result: predictResult, baselineResult: baseline });
      } catch {
        patchCase(i, { status: 'error' });
      }
    }

    if (!batch && viewMode === 'side-by-side') {
      // Compare across every model with a live checkpoint for this dataset.
      const ids = Array.from(
        new Set(AVAILABLE_COMBINATIONS.filter((c) => c.dataset === dataset).map((c) => c.modelId)),
      );
      try {
        setCompareResponse(
          await api.compare({
            imageB64: queue[0].imageB64,
            modelIds: ids,
            dataset,
            regime: selectedRegime,
          }),
        );
      } catch {
        // Comparison is supplementary; the primary result already rendered.
      }
    }

    setLoading(false);
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
   * Download every finished case's mask as separate PNGs.
   *
   * Not a zip: bundling one would mean pulling in an archiving dependency for
   * at most MAX_BATCH_FILES files. Spaced out so the browser doesn't treat the
   * burst as a popup and block all but the first.
   */
  const handleDownloadAll = () => {
    cases.forEach((c, i) => {
      if (c.status !== 'done' || !c.result) return;
      const url = editor.toDataUrl(c.id);
      if (!url) return;
      window.setTimeout(() => {
        const a = document.createElement('a');
        a.href = url;
        a.download = `iteris-mask-${c.label.replace(/\.[^.]+$/, '')}.png`;
        a.click();
      }, i * 250);
    });
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
          scanLabel={isBatch ? undefined : activeCase?.label}
          gtMaskLabel={gtMask?.label}
          onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
          onModelSelect={setSelectedModel}
          onRegimeChange={setSelectedRegime}
          onViewModeChange={setViewMode}
          onWipeSourcesChange={setWipeSources}
          onSampleSelect={handleSampleSelect}
          onScanUpload={handleScanUpload}
          onGtMaskUpload={(dataUrl, file) =>
            setGtMask({ b64: toBareB64(dataUrl), previewUrl: dataUrl, label: file.name })
          }
          onRunInference={handleRunInference}
          maxBatchFiles={MAX_BATCH_FILES}
          batchItems={isBatch ? cases.map((c) => ({ id: c.id, label: c.label, status: c.status })) : []}
          onBatchUpload={handleBatchUpload}
          onClearBatch={handleClearBatch}
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
            {isBatch && (
              <CaseTabs
                cases={cases.map((c) => ({ id: c.id, label: c.label, status: c.status }))}
                activeIndex={activeIndex}
                onSelect={setActiveIndex}
                onDownloadAll={handleDownloadAll}
                running={loading}
              />
            )}
            <ImageViewer
              anatomyLabel={activeCase?.label ?? 'Upload a scan to begin'}
              imageB64={activeCase?.previewUrl}
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
    </div>
  );
}
