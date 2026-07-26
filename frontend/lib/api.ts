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
