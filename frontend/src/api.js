const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function req(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail ?? res.statusText)
  return data
}

export const api = {
  get: (path) => req('GET', path),
  post: (path, body) => req('POST', path, body),
  delete: (path) => req('DELETE', path),
  streamUrl: (path) => BASE + path,
}
