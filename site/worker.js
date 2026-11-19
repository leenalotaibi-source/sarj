// Cloudflare Worker: keeps the AI keys OFF the public page.
// The page (index.html, constant AI_PROXY) sends its Gemini requests here; this worker adds the secret key.
// Setup (Cloudflare dashboard): Workers & Pages -> Create -> Hello World -> Edit code -> paste this file -> Deploy.
// Then Settings -> Variables and Secrets -> add a Secret named GEMINI_API_KEY (and ANTHROPIC_API_KEY for "/").
// Finally put the worker URL in index.html: const AI_PROXY = "https://<name>.<account>.workers.dev";

const ALLOWED_ORIGINS = ["https://reemaothsa.github.io"]; // only your site may use this worker
const GEMINI_MODELS = new Set([                            // only what the page asks for
  "gemini-3.8-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite",
  "gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts", "gemini-3.5-transcribe",
]);
const MAX_BODY = 6_000_000;                                // a few minutes of 16 kHz audio at most
const CLAUDE_MODEL = "claude-haiku-5-5";
const MAX_TURNS = 30;
const MAX_CHARS = 6000;

function cors(origin) {
  const ok = ALLOWED_ORIGINS.includes(origin);
  return {
    "Access-Control-Allow-Origin": ok ? origin : ALLOWED_ORIGINS[0],
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Vary": "Origin",
  };
}
const json = (body, status, origin) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...cors(origin) } });

async function gemini(request, env, origin, model) {
  if (!GEMINI_MODELS.has(model)) return json({ error: "model not allowed" }, 404, origin);
  if (!env.GEMINI_API_KEY) return json({ error: "GEMINI_API_KEY secret is missing" }, 500, origin);
  const body = await request.text();
  if (body.length > MAX_BODY) return json({ error: "request too large" }, 413, origin);
  const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
    body,
  });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json", ...cors(origin) } });
}

async function claude(request, env, origin) {
  if (!env.ANTHROPIC_API_KEY) return json({ error: "ANTHROPIC_API_KEY secret is missing" }, 500, origin);
  let body;
  try { body = await request.json(); } catch { return json({ error: "bad json" }, 400, origin); }
  let messages = Array.isArray(body.messages) ? body.messages : [];
  messages = messages
    .filter(m => (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .slice(-MAX_TURNS)
    .map(m => ({ role: m.role, content: m.content.slice(0, MAX_CHARS) }));
  if (!messages.length || messages[0].role !== "user") return json({ error: "conversation must start with user" }, 400, origin);
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model: CLAUDE_MODEL, max_tokens: 200, messages }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) return json({ error: (data.error && data.error.message) || "upstream error" }, r.status === 429 ? 429 : 502, origin);
  const text = (data.content || []).filter(c => c.type === "text").map(c => c.text).join(" ").trim();
  return json({ text }, 200, origin);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    if (request.method === "OPTIONS") return new Response(null, { headers: cors(origin) });
    if (request.method !== "POST") return json({ error: "POST only" }, 405, origin);
    if (!ALLOWED_ORIGINS.includes(origin)) return json({ error: "origin not allowed" }, 403, origin);
    const path = new URL(request.url).pathname;
    const m = path.match(/^\/gemini\/([\w.\-]+)$/);
    if (m) return gemini(request, env, origin, m[1]);
    return claude(request, env, origin);
  },
};
