/**
 * Runtime configuration, resolved from Vite env vars (see .env.example).
 * Nothing here is hardcoded — every value has an env override and a safe default.
 */

interface AppConfig {
  /** Base URL for backend API calls. */
  apiBaseUrl: string;
  /** When true, the UI uses bundled mock data instead of the live backend. */
  useMocks: boolean;
}

function bool(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  return value.toLowerCase() === 'true';
}

export const appConfig: AppConfig = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? '/api',
  useMocks: bool(import.meta.env.VITE_USE_MOCKS, true),
};
