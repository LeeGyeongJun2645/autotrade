const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function req(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try { const err = await res.json(); detail = err.detail ?? detail } catch {}
    throw new Error(detail)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  get: (path) => req('GET', path),
  post: (path, body) => req('POST', path, body),
  delete: (path) => req('DELETE', path),
  streamUrl: (path) => BASE + path,
}
