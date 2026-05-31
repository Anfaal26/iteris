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
  | 'unet-baseline'
  | 'dqn'
  | 'ddqn'
  | 'dueling-dqn'
  | 'ddpg';

/** Viewing modes in the workstation (spec §6). */
export type ViewMode = 'single' | 'wipe' | 'side-by-side';

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
  preprocessingMs: number;
  inferenceMs: number;
  imageWidth: number;
  imageHeight: number;
}

export interface PredictRequest {
  imageB64: string;
  modelId: ModelId;
  dataset: DatasetId;
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

/** GET /health response. */
export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  modelsLoaded: number;
  gpuAvailable: boolean;
  datasetsAvailable: DatasetId[];
}
