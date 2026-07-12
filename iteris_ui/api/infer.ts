/**
 * Vercel serverless function — thin proxy to the HF Space's POST /infer.
 *
 * The Space URL (and an optional auth token, if the Space is ever locked down)
 * live only in this function's environment, never in client-bundled code. The
 * browser calls same-origin `/api/infer`; this forwards the body server-side
 * and relays the response back. See iteris_ui/src/api/client.ts inferViaDrlRoute.
 *
 * Signature per Vercel's current documented contract for a plain (non-Next.js)
 * project: a NAMED export per HTTP method (`export function POST(request)`),
 * taking and returning the standard Fetch `Request`/`Response`. Two earlier
 * attempts at this file used a default-exported handler and a Node
 * `(req, res)` signature respectively — neither matches Vercel's actual
 * routing contract for /api functions outside Next.js, so both crashed with
 * an uncaught error before any response was ever sent (surfaced as a bare
 * 500 with no useful body). This is the form Vercel's docs show verbatim for
 * "other frameworks": https://vercel.com/docs/functions/functions-api-reference
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

export const config = { runtime: 'nodejs', maxDuration: 60 };

const jsonResponse = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

export async function POST(request: Request): Promise<Response> {
  const spaceUrl = process.env.ITERIS_SPACE_URL;
  if (!spaceUrl) {
    return jsonResponse(501, {
      error: 'not_configured',
      detail: 'ITERIS_SPACE_URL is not set on this Vercel project.',
    });
  }

  const body = await request.text();
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
    return jsonResponse(502, { error: 'upstream_unreachable', detail: String(cause) });
  }

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { 'Content-Type': 'application/json' },
  });
}
