/**
 * Mock implementation of the Iteris API (spec §11). Produces deterministic,
 * spec-accurate data so the entire UI runs without a backend. Mask layers are
 * lightweight SVG data-URIs (valid <img> sources) drawn as anatomical ellipses.
 *
 * These mocks are seeded from src/content/*.yaml where possible so model names
 * and metrics stay in one place; structural defaults live here.
 */
import modelsData from '@/content/models.yaml';
import samplesData from '@/content/samples.yaml';
import type {
  ChatRequest,
  CompareRequest,
  CompareResponse,
  HealthResponse,
  InterpretRequest,
  IterationStep,
  MaskLayer,
  Metrics,
  ModelId,
  ModelRecord,
  PredictRequest,
  PredictResponse,
  SampleImage,
  StructureId,
  StructureMetrics,
} from './contract';
import { maskColorsHex } from '@/tokens';

const STRUCTURES: Record<
  PredictRequest['dataset'],
  { id: StructureId; label: string; color: string }[]
> = {
  camus: [
    { id: 'lv_endo', label: 'LV Endocardium', color: maskColorsHex.lvEndo },
    { id: 'lv_epi', label: 'LV Epicardium', color: maskColorsHex.lvEpi },
    { id: 'la', label: 'Left Atrium', color: maskColorsHex.la },
  ],
  brisc: [
    { id: 'glioma', label: 'Glioma', color: maskColorsHex.glioma },
    { id: 'meningioma', label: 'Meningioma', color: maskColorsHex.meningioma },
    { id: 'pituitary', label: 'Pituitary Tumor', color: maskColorsHex.pituitary },
  ],
};

const SIZE = 256;

function svgMask(color: string, cx: number, cy: number, rx: number, ry: number): string {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${SIZE}' height='${SIZE}'><ellipse cx='${cx}' cy='${cy}' rx='${rx}' ry='${ry}' fill='${color}' fill-opacity='0.55' stroke='${color}' stroke-width='2'/></svg>`;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

function buildMasks(dataset: PredictRequest['dataset'], spread = 1): MaskLayer[] {
  return STRUCTURES[dataset].map((s, i) => ({
    structure: s.id,
    label: s.label,
    color: s.color,
    imageB64: svgMask(
      s.color,
      SIZE / 2 + (i - 1) * 18 * spread,
      SIZE / 2 + (i - 1) * 14 * spread,
      48 - i * 6,
      40 - i * 5,
    ),
  }));
}

function round(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function buildMetrics(dataset: PredictRequest['dataset'], dice: number): Metrics {
  const baselineDice = dataset === 'camus' ? 0.89 : 0.79;
  const structures: StructureMetrics[] = STRUCTURES[dataset].map((s, i) => ({
    structure: s.id,
    label: s.label,
    dice: round(dice - i * 0.02),
    iou: round(dice - 0.08 - i * 0.02),
    hd: round(3.2 + i * 0.9),
    hd95: round(2.1 + i * 0.7),
  }));
  return {
    dice: round(dice),
    iou: round(dice - 0.08),
    hd: round(3.2),
    hd95: round(2.1),
    structures,
    baselineDice,
  };
}

export async function health(): Promise<HealthResponse> {
  return {
    status: 'ok',
    modelsLoaded: 6,
    gpuAvailable: false,
    datasetsAvailable: ['camus', 'brisc'],
  };
}

export async function models(): Promise<ModelRecord[]> {
  return modelsData as ModelRecord[];
}

export async function samples(): Promise<SampleImage[]> {
  return samplesData as SampleImage[];
}

/** DRL agents refine a contour over an episode; baselines do a single pass. */
function isDrl(modelId: ModelId): boolean {
  return modelId === 'dqn' || modelId === 'ddqn' || modelId === 'dueling-dqn' || modelId === 'td3';
}

export async function predict(body: PredictRequest): Promise<PredictResponse> {
  const dice = body.dataset === 'camus' ? 0.912 : 0.84;
  const masks = buildMasks(body.dataset);
  const response: PredictResponse = {
    sessionId: `mock-${body.modelId}-${body.dataset}`,
    modelId: body.modelId,
    dataset: body.dataset,
    masks,
    metrics: buildMetrics(body.dataset, dice),
    refinementSteps: isDrl(body.modelId) ? 18 : undefined,
    preprocessingMs: 180,
    inferenceMs: isDrl(body.modelId) ? 940 : 640,
    imageWidth: SIZE,
    imageHeight: SIZE,
  };
  if (body.playback) {
    const total = 20;
    const init = (body.dataset === 'camus' ? 0.89 : 0.79) - 0.03;
    response.stepSequence = Array.from({ length: total }, (_, i): IterationStep => {
      const progress = (i + 1) / total;
      return {
        step: i + 1,
        masks: buildMasks(body.dataset, 1 - progress * 0.9),
        deltaDice: round((dice - init) * (1 / total) + (Math.sin(i) * 0.002)),
        annotation: `Step ${i + 1}: agent nudges the contour toward the high-gradient boundary; Dice trending toward ${round(init + (dice - init) * progress)}.`,
      };
    });
  }
  return response;
}

export async function compare(body: CompareRequest): Promise<CompareResponse> {
  return {
    results: body.modelIds.map((id) => {
      const dice = id === 'unet-baseline'
        ? (body.dataset === 'camus' ? 0.89 : 0.79)
        : (body.dataset === 'camus' ? 0.912 : 0.84);
      return {
        modelId: id,
        masks: buildMasks(body.dataset),
        metrics: buildMetrics(body.dataset, dice),
      };
    }),
  };
}

const SECTION_TEXT: Record<string, (b: InterpretRequest) => string> = {
  'Segmentation Summary': (b) =>
    `The ${b.modelId.toUpperCase()} agent delineated ${b.structures.length} structure(s) on a ${b.dataset === 'camus' ? 'CAMUS echocardiography' : 'BRISC T1-weighted MRI'} image, reaching an overall Dice of ${b.metrics.dice}.`,
  'Clinical Significance': () =>
    'Boundary precision at this level supports reproducible volumetric measurement; sub-voxel error margins matter for downstream quantitative analysis.',
  'Metric Interpretation': (b) =>
    `A Dice of ${b.metrics.dice} against a baseline of ${b.metrics.baselineDice} corresponds to a small-millimetre boundary error for typical structure sizes.`,
  'Performance Analysis': () =>
    'The agent appears to have learned to follow high-gradient image edges while penalising anatomically implausible excursions, yielding the above-baseline result.',
  'Literature References': () =>
    'See e.g. Leclerc et al., IEEE TMI (CAMUS); Bakas et al., Scientific Data (brain tumour segmentation). Metrics reported per 5-fold CV.',
};

export async function* interpret(body: InterpretRequest): AsyncGenerator<string> {
  for (const [header, fn] of Object.entries(SECTION_TEXT)) {
    yield `\n## ${header}\n`;
    const words = fn(body).split(' ');
    for (const w of words) {
      yield `${w} `;
      await new Promise((r) => setTimeout(r, 12));
    }
  }
}

/**
 * Mock chat — a grounded, streamed answer that references the live metrics so
 * the thread feels real without a backend. The last user turn steers the reply.
 */
export async function* chat(body: ChatRequest): AsyncGenerator<string> {
  const { context } = body;
  const last = [...body.messages].reverse().find((m) => m.role === 'user')?.content ?? '';
  const q = last.toLowerCase();
  const m = context.metrics;
  const modelName = context.modelId.toUpperCase();
  const gt = context.hasGroundTruth;

  let reply: string;
  if (/lv|endo|boundary|improve/.test(q)) {
    reply = `On the LV endocardium, ${modelName} pushed the contour toward the high-gradient wall the baseline under-segmented, lifting per-class Dice to ${m.structures[0]?.dice ?? m.dice}. The refinement steps concentrate where the initial mask leaked into papillary muscle.`;
  } else if (/compare|attention|u-?net|baseline/.test(q)) {
    reply = gt
      ? `Versus the Attention U-Net baseline (Dice ${m.baselineDice}), this run reaches ${m.dice} — a ${(m.dice - m.baselineDice).toFixed(3)} delta, mostly from tightened boundaries rather than new regions.`
      : `No ground-truth mask is attached, so Dice/IoU aren't computed for this image. Attach a GT mask to get a real delta versus the ${m.baselineDice} baseline.`;
  } else if (/hausdorff|hd|worst|outlier/.test(q)) {
    reply = `Hausdorff distance here is ${m.hd} (95th-pct ${m.hd95}), so the worst-case boundary excursion is small — the agent avoided the large stray components that drive HD up.`;
  } else {
    reply = gt
      ? `This ${context.dataset.toUpperCase()} result from ${modelName} scores Dice ${m.dice} / IoU ${m.iou} across ${m.structures.length} structure(s). Ask about a specific structure, the baseline delta, or the refinement trajectory.`
      : `This ${context.dataset.toUpperCase()} run used ${modelName}. Metrics need a ground-truth mask to compute — attach one in the sidebar and I can quantify the boundaries.`;
  }

  for (const w of reply.split(' ')) {
    yield `${w} `;
    await new Promise((r) => setTimeout(r, 14));
  }
}
