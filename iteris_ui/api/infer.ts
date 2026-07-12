/**
 * Vercel serverless function — thin proxy to the HF Space's POST /infer.
 *
 * The Space URL (and an optional auth token, if the Space is ever locked down)
 * live only in this function's environment, never in client-bundled code. The
 * browser calls same-origin `/api/infer`; this forwards the body server-side
 * and relays the response back. See iteris_ui/src/api/client.ts inferViaDrlRoute.
 *
 * Written against ONLY native Node.js `http` primitives (statusCode/setHeader/
 * end, and reading the request as a stream) rather than the `.status()/.json()/
 * .send()` convenience methods some Vercel setups attach — those aren't
 * guaranteed present for a plain Vite project's /api functions (that's what
 * caused the 500s: those methods didn't exist here, and the very first call to
 * one of them threw before any response was ever sent). `req`/`res` ARE a
 * Node `http.IncomingMessage`/`ServerResponse` under the hood regardless of
 * framework, so these low-level methods always work.
 *
 * Required env vars (set in the Vercel project, not committed):
 *   ITERIS_SPACE_URL    e.g. https://anfaal26-iteris-api.hf.space
 *   ITERIS_SPACE_TOKEN  optional — forwarded as `Authorization: Bearer <token>`
 *                       if the Space ever requires auth (public Spaces don't).
 *
 * Free-tier HF Spaces sleep after inactivity and take ~30s-1min to wake on the
 * first request, and a DRL contour-refinement rollout itself takes real
 * compute time — maxDuration below extends this function's timeout well past
 * Vercel's 10s Hobby-plan default so a cold start doesn't get killed mid-flight.
 */

export const config = { maxDuration: 60 };

import type { IncomingMessage, ServerResponse } from 'http';

/** Reads the full request body as a string — no reliance on auto body-parsing. */
function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(body));
}

export default async function handler(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'method_not_allowed' });
    return;
  }

  const spaceUrl = process.env.ITERIS_SPACE_URL;
  if (!spaceUrl) {
    sendJson(res, 501, {
      error: 'not_configured',
      detail: 'ITERIS_SPACE_URL is not set on this Vercel project.',
    });
    return;
  }

  let body: string;
  try {
    body = await readBody(req);
  } catch (cause) {
    sendJson(res, 400, { error: 'bad_request', detail: String(cause) });
    return;
  }

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (process.env.ITERIS_SPACE_TOKEN) {
    headers.Authorization = `Bearer ${process.env.ITERIS_SPACE_TOKEN}`;
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${spaceUrl.replace(/\/$/, '')}/infer`, {
      method: 'POST',
      headers,
      body,
    });
  } catch (cause) {
    sendJson(res, 502, { error: 'upstream_unreachable', detail: String(cause) });
    return;
  }

  const text = await upstream.text();
  res.statusCode = upstream.status;
  res.setHeader('Content-Type', 'application/json');
  res.end(text);
}
