/**
 * Typed Iteris API client. Routes to the live FastAPI backend or to bundled
 * mocks depending on `appConfig.useMocks`, so the UI runs standalone before the
 * ML endpoints exist. All network errors surface as `ApiError` for toast display.
 */
import { appConfig } from '@/config/app.config';
import type {
  ChatRequest,
  CompareRequest,
  CompareResponse,
  HealthResponse,
  InferRequest,
  InferResponse,
  InterpretRequest,
  ModelRecord,
  PredictRequest,
  PredictResponse,
  Regime,
  SampleImage,
} from './contract';
import { defaultRegime, drlBackend } from './contract';
import * as mock from './mock';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${appConfig.apiBaseUrl}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(
      `Network error calling ${path}: ${(cause as Error).message}`,
    );
  }
  if (!res.ok) {
    throw new ApiError(`Request to ${path} failed`, res.status);
  }
  return (await res.json()) as T;
}

/**
 * Calls the same-origin `/api/infer` Vercel serverless function (a thin proxy
 * to the HF Space's POST /infer) and translates its response back into the
 * PredictResponse shape callers already expect from api.predict().
 */
async function inferViaDrlRoute(
  body: PredictRequest,
  algo: import('./contract').AlgoId,
  regime: Regime,
): Promise<PredictResponse> {
  const reqBody: InferRequest = {
    imageB64: body.imageB64,
    dataset: body.dataset,
    modelFamily: 'drl',
    algo,
    regime,
    gtMaskB64: body.gtMaskB64,
  };
  let res: Response;
  try {
    res = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody),
    });
  } catch (cause) {
    throw new ApiError(`Network error calling /api/infer: ${(cause as Error).message}`);
  }
  if (!res.ok) {
    // A structured 404 (not-registered) still deserves a real ApiError, not a
    // generic failure — the message helps a caller show "not trained yet".
    const detail = await res.json().catch(() => null);
    throw new ApiError(detail?.detail ?? `/api/infer failed (${res.status})`, res.status);
  }
  const infer = (await res.json()) as InferResponse;
  return {
    sessionId: infer.sessionId,
    modelId: body.modelId,
    dataset: infer.dataset,
    masks: infer.masks,
    metrics: infer.metrics,
    refinementSteps: infer.refinementSteps,
    preprocessingMs: 0,
    inferenceMs: infer.inferenceMs,
    imageWidth: infer.imageWidth,
    imageHeight: infer.imageHeight,
  };
}

export const api = {
  health(): Promise<HealthResponse> {
    return appConfig.useMocks ? mock.health() : request('/health');
  },

  models(): Promise<ModelRecord[]> {
    return appConfig.useMocks ? mock.models() : request('/models');
  },

  samples(): Promise<SampleImage[]> {
    return appConfig.useMocks
      ? mock.samples()
      : request('/datasets/samples');
  },

  /**
   * Runs inference for any model. /predict only ever serves the deployed
   * U-Net baseline; DRL model ids are transparently routed to the same-origin
   * `/api/infer` Vercel function (NOT appConfig.apiBaseUrl, which may point
   * straight at the HF Space) — that function is the only thing holding the
   * Space URL/token, so neither ever reaches the browser.
   */
  predict(body: PredictRequest): Promise<PredictResponse> {
    if (appConfig.useMocks) return mock.predict(body);

    const backend = drlBackend(body.modelId);
    if (!backend) {
      return request('/predict', { method: 'POST', body: JSON.stringify(body) });
    }
    return inferViaDrlRoute(body, backend.algo, body.regime ?? defaultRegime(body.modelId));
  },

  compare(body: CompareRequest): Promise<CompareResponse> {
    return appConfig.useMocks
      ? mock.compare(body)
      : request('/compare', { method: 'POST', body: JSON.stringify(body) });
  },

  /**
   * Streamed Claude interpretation. Yields text chunks as they arrive so the
   * panel can render a typewriter effect (spec §7).
   */
  async *interpret(body: InterpretRequest): AsyncGenerator<string> {
    if (appConfig.useMocks) {
      yield* mock.interpret(body);
      return;
    }
    const res = await fetch(`${appConfig.apiBaseUrl}/interpret`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      throw new ApiError('Interpretation request failed', res.status);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  },

  /**
   * Streamed answer to a workspace chat turn, grounded in the current result.
   * Same event-stream transport as interpret().
   */
  async *chat(body: ChatRequest): AsyncGenerator<string> {
    if (appConfig.useMocks) {
      yield* mock.chat(body);
      return;
    }
    const res = await fetch(`${appConfig.apiBaseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      throw new ApiError('Chat request failed', res.status);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  },
};
