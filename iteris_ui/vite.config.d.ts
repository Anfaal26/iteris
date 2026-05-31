/**
 * Vite configuration for the Iteris UI.
 *
 * - `@` resolves to `src` so imports never use brittle relative paths.
 * - `vite-plugin-yaml` lets content config (`src/content/*.yaml`) import as typed objects,
 *   keeping all copy/metrics/model data out of component code (YAML-driven, no hardcoding).
 * - The dev server proxies `/api` to the FastAPI backend whose URL is supplied via env,
 *   so no backend host is ever hardcoded in the bundle.
 */
declare const _default: import("vite").UserConfigFnObject;
export default _default;
