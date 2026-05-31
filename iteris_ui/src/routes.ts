/**
 * Canonical route table (spec §3). Imported by the router and any nav component
 * so paths are defined exactly once.
 */
export const ROUTES = {
  landing: '/',
  research: '/research',
  workspace: '/workspace',
  models: '/models',
  datasets: '/datasets',
} as const;

export type RouteKey = keyof typeof ROUTES;
