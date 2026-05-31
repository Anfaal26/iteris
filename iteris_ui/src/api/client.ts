/**
 * Typed Iteris API client. Routes to the live FastAPI backend or to bundled
 * mocks depending on `appConfig.useMocks`, so the UI runs standalone before the
 * ML endpoints exist. All network errors surface as `ApiError` for toast display.
 */
import { appConfig } from '@/config/app.config';
import type {
  CompareRequest,
  CompareResponse,
  HealthResponse,
  InterpretRequest,
  ModelRecord,
  PredictRequest,
  PredictResponse,
  SampleImage,
} from './contract';
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

  predict(body: PredictRequest): Promise<PredictResponse> {
    return appConfig.useMocks
      ? mock.predict(body)
      : request('/predict', { method: 'POST', body: JSON.stringify(body) });
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
};
