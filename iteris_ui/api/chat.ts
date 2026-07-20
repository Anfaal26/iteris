/**
 * Vercel serverless function — the workspace "Ask about this result" chat.
 *
 * Proxies the thread to Groq's OpenAI-compatible chat-completions API and
 * relays the answer back as a plain UTF-8 text stream (the transport
 * src/api/client.ts `chat()` already reads: raw text chunks, not SSE frames).
 *
 * Signature matches api/infer.ts — a NAMED export per HTTP method taking and
 * returning the standard Fetch `Request`/`Response`. That is Vercel's documented
 * contract for /api functions in a non-Next.js project; a default export or a
 * Node `(req, res)` handler crashes before responding. See the comment block in
 * api/infer.ts for the history there.
 *
 * The API key lives ONLY in this function's environment — it is never bundled
 * into client code, and the browser only ever calls same-origin `/api/chat`.
 *
 * Required env vars (set in the Vercel project, not committed):
 *   GROQ_API_KEY   Groq free-tier key (https://console.groq.com/keys)
 *   GROQ_MODEL     optional — defaults to llama-3.3-70b-versatile.
 *                  Use llama-3.1-8b-instant if latency matters more than depth.
 */

export const config = { runtime: 'nodejs', maxDuration: 60 };

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const DEFAULT_MODEL = 'llama-3.3-70b-versatile';

/** Message shape shared with src/api/contract.ts ChatMessage. */
interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

/** Compact snapshot of what's on screen, mirroring ChatRequest['context']. */
interface ChatContext {
  modelId?: string;
  dataset?: string;
  regime?: string;
  hasGroundTruth?: boolean;
  metrics?: {
    dice?: number;
    iou?: number;
    hd?: number;
    hd95?: number;
    baselineDice?: number;
    structures?: { label?: string; dice?: number; iou?: number; hd?: number }[];
  };
}

const jsonResponse = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

/**
 * System prompt — specialises a general model for this workstation: it explains
 * segmentation output and the three deployed model families, and is explicit
 * that Iteris is a research tool, not a diagnostic one.
 */
function systemPrompt(context: ChatContext): string {
  const m = context.metrics ?? {};
  const perStructure = (m.structures ?? [])
    .map((s) => `  - ${s.label}: Dice ${s.dice}, IoU ${s.iou}, HD ${s.hd}`)
    .join('\n');

  return [
    'You are the analysis assistant embedded in Iteris, a medical-image segmentation',
    'research workstation (Taylor\'s University capstone PRJ63504). You explain results',
    'to researchers and ML students.',
    '',
    'What the system does: an Attention Residual U-Net produces a coarse segmentation',
    'mask; a deep-RL agent then refines the mask BOUNDARY by deforming its contour —',
    'pushing contiguous angular sectors of the contour along their outward normals.',
    'Two agents are deployed: DuelingDDQN (discrete action space) and TD3 (continuous).',
    'Datasets: CAMUS (cardiac ultrasound — LV endocardium, LV epicardium, left atrium)',
    'and BRISC (brain MRI — glioma, meningioma, pituitary tumour). Results are reported',
    'per data regime: Phase A (full data) and Phase B (low data, ~150 images).',
    '',
    'Answer questions about: the current result and its metrics, why a boundary moved,',
    'how Dice / IoU / Hausdorff / HD95 should be read, how the models behave and differ,',
    'and general medical-imaging and machine-learning concepts. Be concise and concrete;',
    'cite the actual numbers below rather than inventing any. If a number is not given,',
    'say it is unavailable instead of guessing.',
    '',
    'CRITICAL: Iteris is a research tool, NOT a diagnostic device. Never give a diagnosis,',
    'a clinical recommendation, or a statement about a patient\'s condition. If asked, say',
    'plainly that these outputs are not a substitute for clinical judgement and must not',
    'be used for patient care.',
    '',
    'Current result on screen:',
    `  model: ${context.modelId ?? 'unknown'}`,
    `  dataset: ${context.dataset ?? 'unknown'}   data regime: ${context.regime ?? 'unknown'}`,
    `  ground-truth mask attached: ${context.hasGroundTruth ? 'yes' : 'no'}`,
    context.hasGroundTruth
      ? `  overall Dice ${m.dice}, IoU ${m.iou}, HD ${m.hd}, HD95 ${m.hd95}, U-Net baseline Dice ${m.baselineDice}`
      : '  metrics unavailable — no ground-truth mask was attached, so Dice/IoU/HD are not computed.',
    perStructure ? `  per structure:\n${perStructure}` : '',
  ]
    .filter(Boolean)
    .join('\n');
}

export async function POST(request: Request): Promise<Response> {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return jsonResponse(501, {
      error: 'not_configured',
      detail: 'GROQ_API_KEY is not set on this Vercel project.',
    });
  }

  let body: { messages?: ChatTurn[]; context?: ChatContext };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return jsonResponse(400, { error: 'bad_request', detail: 'Body must be JSON.' });
  }

  const turns = (body.messages ?? []).filter((m) => typeof m?.content === 'string');
  if (turns.length === 0) {
    return jsonResponse(400, { error: 'bad_request', detail: 'No messages supplied.' });
  }

  let upstream: Response;
  try {
    upstream = await fetch(GROQ_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: process.env.GROQ_MODEL || DEFAULT_MODEL,
        stream: true,
        temperature: 0.3,
        max_tokens: 900,
        messages: [
          { role: 'system', content: systemPrompt(body.context ?? {}) },
          // Only the last 20 turns — the free tier has a modest token budget and
          // older turns add little once the thread is grounded in the same result.
          ...turns.slice(-20).map((m) => ({ role: m.role, content: m.content })),
        ],
      }),
    });
  } catch (cause) {
    return jsonResponse(502, { error: 'upstream_unreachable', detail: String(cause) });
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => '');
    // 429 is the common free-tier outcome; relay the status so the UI can offer
    // a retry rather than showing a generic failure.
    return jsonResponse(upstream.status === 429 ? 429 : 502, {
      error: upstream.status === 429 ? 'rate_limited' : 'upstream_error',
      detail: detail.slice(0, 500) || `Groq returned ${upstream.status}.`,
    });
  }

  // Groq streams OpenAI-style SSE ("data: {json}\n\n", terminated by [DONE]).
  // Unwrap it to bare text so the client can just decode chunks as they land.
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';

  const stream = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split('\n');
      // Keep the trailing partial line for the next chunk.
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const parsed = JSON.parse(payload);
          const text = parsed?.choices?.[0]?.delta?.content;
          if (text) controller.enqueue(encoder.encode(text));
        } catch {
          // A malformed frame is not worth failing the whole answer over.
        }
      }
    },
  });

  return new Response(upstream.body.pipeThrough(stream), {
    status: 200,
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  });
}
