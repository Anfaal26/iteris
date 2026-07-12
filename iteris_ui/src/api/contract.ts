/**
 * ITERIS API CONTRACT (spec §11) — the frozen interface between this UI and the
 * FastAPI backend (built separately as part of the ML work). Every page and the
 * mock client bind to these types; the backend must satisfy the same shapes.
 * Changing an endpoint means changing this file first.
 */

/** Which dataset / trained-model family is in play. */
export type DatasetId = 'camus' | 'brisc';

/** Imaging modality, derived from the dataset. */
export type Modality = 'ultrasound' | 'mri';

/** Algorithm families and the deployed checkpoints. */
export type ModelFamily = 'baseline' | 'discrete-drl' | 'continuous-drl';
export type ModelId =
  | 'unet-baseline' // Attention Res U-Net
  | 'lite-unet' // Lite U-Net (compact baseline)
  | 'dqn'
  | 'ddqn'
  | 'dueling-dqn' // DuelingDDQN — discrete DRL refinement
  | 'ddpg'
  | 'td3'; // TD3 — continuous DRL refinement

/**
 * Data regime a checkpoint was trained under. DRL agents default to `low`,
 * the U-Net baselines to `high` (see availableRegimes / DEFAULT_REGIME below).
 */
export type Regime = 'low' | 'high';

/** Viewing modes in the workstation (spec §6). */
export type ViewMode = 'single' | 'wipe' | 'side-by-side';

/**
 * Mask source a Wipe-comparison pane can show. Any two may be paired
 * (a source is greyed out when its mask doesn't exist for the current image).
 */
export type WipeSource = 'attention-unet' | 'gt' | 'drl';

/** Difficulty stratification for sample images (spec §5/§6). */
export type Difficulty = 'easy' | 'medium' | 'hard';

/** Anatomical structures / tumour classes, keyed to mask colours. */
export type StructureId =
  | 'lv_endo'
  | 'lv_epi'
  | 'la'
  | 'glioma'
  | 'meningioma'
  | 'pituitary';

/** Per-structure segmentation quality metrics. */
export interface StructureMetrics {
  structure: StructureId;
  label: string;
  dice: number;
  iou: number;
  hd: number;
  hd95: number;
}

/** Aggregate metrics returned for a prediction. */
export interface Metrics {
  dice: number;
  iou: number;
  hd: number;
  hd95: number;
  /** Per-structure breakdown shown in the right results panel. */
  structures: StructureMetrics[];
  /** Baseline (U-Net) overall Dice for delta/colour-coding context. */
  baselineDice: number;
}

/** A single mask layer: base64 PNG (RGBA) + the structure it represents. */
export interface MaskLayer {
  structure: StructureId;
  label: string;
  /** data-URI or bare base64 PNG with transparent background. */
  imageB64: string;
  color: string;
}

/** One step of DRL iteration playback (spec §6 Iteration Playback Mode). */
export interface IterationStep {
  step: number;
  masks: MaskLayer[];
  deltaDice: number;
  /** Pre-generated natural-language annotation for this step. */
  annotation: string;
}

/** POST /predict response. */
export interface PredictResponse {
  sessionId: string;
  modelId: ModelId;
  dataset: DatasetId;
  masks: MaskLayer[];
  metrics: Metrics;
  /** Present only when playback mode was requested. */
  stepSequence?: IterationStep[];
  /** DRL refinement episode length (agent steps taken). Absent for U-Net baselines. */
  refinementSteps?: number;
  preprocessingMs: number;
  inferenceMs: number;
  imageWidth: number;
  imageHeight: number;
}

export interface PredictRequest {
  imageB64: string;
  modelId: ModelId;
  dataset: DatasetId;
  /** Data regime the requested checkpoint was trained under. Defaults per model. */
  regime?: Regime;
  mode: ViewMode;
  playback?: boolean;
  gtMaskB64?: string;
}

/** GET /models item. */
export interface ModelRecord {
  id: ModelId;
  name: string;
  family: ModelFamily;
  description: string;
  /** Best Dice per dataset (null if not evaluated on that dataset). */
  diceCamus: number | null;
  diceBrisc: number | null;
  iou: number | null;
  hd: number | null;
  deployed: boolean;
  /** Whether the model can be selected for refinement (baseline cannot). */
  selectable: boolean;
}

/** POST /compare response. */
export interface CompareResult {
  modelId: ModelId;
  masks: MaskLayer[];
  metrics: Metrics;
}
export interface CompareResponse {
  results: CompareResult[];
}
export interface CompareRequest {
  imageB64: string;
  modelIds: ModelId[];
  dataset: DatasetId;
  regime?: Regime;
}

/** POST /interpret request (response is a text/event-stream). */
export interface InterpretRequest {
  modelId: ModelId;
  structures: StructureId[];
  metrics: Metrics;
  dataset: DatasetId;
  modality: Modality;
  difficulty?: Difficulty;
  tumorType?: 'glioma' | 'meningioma' | 'pituitary';
  gtMetrics?: Pick<Metrics, 'dice'>;
}

/** The five labelled sections the interpretation renders into (spec §7). */
export type InterpretationSection =
  | 'segmentation-summary'
  | 'clinical-significance'
  | 'metric-interpretation'
  | 'performance-analysis'
  | 'literature-references';

/** One turn in the workspace "Ask about this result" thread. */
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * POST /chat request — a grounded conversation about the current result.
 * The backend receives the running thread plus a compact snapshot of what's on
 * screen (model, dataset, metrics) so answers cite the actual numbers. Response
 * is a text/event-stream, same transport as /interpret.
 */
export interface ChatRequest {
  messages: ChatMessage[];
  context: {
    modelId: ModelId;
    dataset: DatasetId;
    regime: Regime;
    metrics: Metrics;
    /** Whether a ground-truth mask was attached (metrics are real vs. unavailable). */
    hasGroundTruth: boolean;
  };
}

/** GET /datasets/samples item (spec §6 sample tiles). */
export interface SampleImage {
  id: string;
  thumbnailB64: string;
  modality: Modality;
  anatomy: string;
  difficulty: Difficulty;
  bestDice: number;
  dataset: DatasetId;
}

/* ------------------------------------------------------------------ *
 *  /infer — the DRL-specific endpoint (Part 2 backend)
 *
 *  /predict only ever serves the deployed U-Net baseline; every DRL model id
 *  is actually served by POST /infer on the same FastAPI app, keyed by
 *  (dataset, modelFamily, algo, regime). The client translates transparently
 *  (see api/client.ts predict()) so callers never see this split — they just
 *  call api.predict() with a DRL modelId and get back a PredictResponse.
 * ------------------------------------------------------------------ */

/** Algo id the /infer registry keys on — the backend-facing name for a ModelId. */
export type AlgoId = 'dqn' | 'ddqn' | 'duelingddqn' | 'ddpg' | 'td3';

/** Maps a frontend ModelId to the backend's (family, algo) pair, or null for
 * non-DRL models (unet-baseline / lite-unet), which stay on /predict. */
export function drlBackend(modelId: ModelId): { algo: AlgoId } | null {
  switch (modelId) {
    case 'dqn': return { algo: 'dqn' };
    case 'ddqn': return { algo: 'ddqn' };
    case 'dueling-dqn': return { algo: 'duelingddqn' };
    case 'ddpg': return { algo: 'ddpg' };
    case 'td3': return { algo: 'td3' };
    default: return null;
  }
}

export interface InferRequest {
  imageB64: string;
  dataset: DatasetId;
  modelFamily: 'drl';
  algo: AlgoId;
  regime: Regime;
  gtMaskB64?: string;
}

export interface InferResponse {
  sessionId: string;
  dataset: DatasetId;
  algo: AlgoId;
  regime: Regime;
  masks: MaskLayer[];
  metrics: Metrics;
  refinementSteps?: number;
  inferenceMs: number;
  imageWidth: number;
  imageHeight: number;
}

/** GET /health response. */
export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  modelsLoaded: number;
  gpuAvailable: boolean;
  datasetsAvailable: DatasetId[];
}

/* ------------------------------------------------------------------ *
 *  Availability
 *
 *  Which (dataset, model, regime) combinations the backend can actually
 *  serve today. This mirrors the server-side checkpoint registry keyed by
 *  (dataset, class, regime, algo): the UI greys out anything not listed
 *  here so a user can never fire a request the backend would 404 (see the
 *  redesign's "grey out unavailable combinations" rule). Adding a trained
 *  checkpoint is a one-line change here + one entry server-side.
 * ------------------------------------------------------------------ */

/** Every combination with a live checkpoint. Extend as models are trained. */
export const AVAILABLE_COMBINATIONS: ReadonlyArray<{
  dataset: DatasetId;
  modelId: ModelId;
  regime: Regime;
}> = [
  // Attention Res U-Net baseline — the deployed CAMUS/BRISC checkpoint (high regime).
  { dataset: 'camus', modelId: 'unet-baseline', regime: 'high' },
  { dataset: 'brisc', modelId: 'unet-baseline', regime: 'high' },
  // DuelingDDQN, CAMUS LV endocardium, high regime — the one DRL checkpoint hosted so far.
  { dataset: 'camus', modelId: 'dueling-dqn', regime: 'high' },
];

/** Default regime for a model family: DRL → low, U-Net baselines → high. */
export function defaultRegime(modelId: ModelId): Regime {
  return modelId === 'unet-baseline' || modelId === 'lite-unet' ? 'high' : 'low';
}

/** True when a live checkpoint exists for this exact (dataset, model, regime). */
export function isCombinationAvailable(
  dataset: DatasetId,
  modelId: ModelId,
  regime: Regime,
): boolean {
  return AVAILABLE_COMBINATIONS.some(
    (c) => c.dataset === dataset && c.modelId === modelId && c.regime === regime,
  );
}

/** Regimes with a live checkpoint for this (dataset, model). May be empty. */
export function availableRegimes(dataset: DatasetId, modelId: ModelId): Regime[] {
  return (['low', 'high'] as Regime[]).filter((r) =>
    isCombinationAvailable(dataset, modelId, r),
  );
}

/** True when any regime of this (dataset, model) is serveable. */
export function isModelAvailable(dataset: DatasetId, modelId: ModelId): boolean {
  return availableRegimes(dataset, modelId).length > 0;
}
