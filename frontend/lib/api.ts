// Shared API base — set NEXT_PUBLIC_API_URL in .env.local for local dev,
// or in your deployment environment for production.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

/** Parse a Response safely: returns parsed JSON or throws a user-friendly Error. */
export async function parseJSON<T>(res: Response): Promise<T> {
  const ct = res.headers.get("content-type") ?? ""
  if (!ct.includes("application/json")) {
    // Non-JSON body (HTML error page, proxy error, empty response, etc.)
    const text = await res.text().catch(() => "")
    throw new Error(
      `Unexpected response (HTTP ${res.status})${text ? `: ${text.slice(0, 200)}` : ""}`
    )
  }
  const data = await res.json()
  if (!res.ok) {
    throw new Error(data?.detail ?? `Request failed (HTTP ${res.status})`)
  }
  return data as T
}

/**
 * Poll `GET /api/runs/{runId}` until the run reaches a terminal status.
 *
 * The pipeline runs in the background, so `POST /api/runs` returns a `running`
 * record immediately and the client waits here — no single long-lived request
 * that can time out on a proxy or load balancer.
 */
export async function pollRun<T extends { status: string }>(
  runId: string,
  { intervalMs = 2000, timeoutMs = 15 * 60_000 }: { intervalMs?: number; timeoutMs?: number } = {}
): Promise<T> {
  const deadline = Date.now() + timeoutMs
  while (true) {
    const run = await parseJSON<T>(await fetch(`${API_BASE}/api/runs/${runId}`))
    if (run.status !== "running" && run.status !== "queued") return run
    if (Date.now() > deadline) {
      throw new Error("Run is taking longer than expected — check back later.")
    }
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}
