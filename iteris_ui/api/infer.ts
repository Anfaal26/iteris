/**
 * Vercel serverless function — thin proxy to the HF Space's POST /infer.
 *
 * The Space URL (and an optional auth token, if the Space is ever locked down)
 * live only in this function's environment, never in client-bundled code. The
 * browser calls same-origin `/api/infer`; this forwards the body server-side
 * and relays the response back. See iteris_ui/src/api/client.ts inferViaDrlRoute.
 *
 * Uses the standard Vercel Node.js Function contract (req/res with the
 * `.status()/.json()/.send()` helpers Vercel's Node runtime attaches) rather
 * than the Fetch Request/Response signature — that signature is a Next.js App
 * Router convention, not guaranteed for a plain Vite project's /api functions.
 * `req.body` arrives pre-parsed for a JSON content-type, same as any other
 * Vercel Node function.
 *
 * Required env vars (set in the Vercel project, not committed):
 *   ITERIS_SPACE_URL    e.g. https://anfaal26-iteris-api.hf.space
 *   ITERIS_SPACE_TOKEN  optional — forwarded as `Authorization: Bearer <token>`
 *                       if the Space ever requires auth (public Spaces don't).
 *
 * Free-tier HF Spaces sleep after inactivity and take ~30s-1min to wake on the
 * first request — this proxy does not add its own timeout beyond the platform
 * default, so a cold start surfaces as a slow (not failed) response. The
 * frontend should show a loading state for DRL runs rather than treating a
 * slow /api/infer call as an error.
 */

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }

  const spaceUrl = process.env.ITERIS_SPACE_URL;
  if (!spaceUrl) {
    res.status(501).json({
      error: 'not_configured',
      detail: 'ITERIS_SPACE_URL is not set on this Vercel project.',
    });
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
      body: JSON.stringify(req.body),
    });
  } catch (cause) {
    res.status(502).json({ error: 'upstream_unreachable', detail: String(cause) });
    return;
  }

  const text = await upstream.text();
  res.status(upstream.status).setHeader('Content-Type', 'application/json');
  res.send(text);
}
