import { useState, useEffect } from 'react'
import { api } from '../api.js'

export function useSSE() {
  const [positions, setPositions] = useState({})
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const es = new EventSource(api.streamUrl('/stream'))

    es.onopen = () => {
      setConnected(true)
      setError(null)
    }

    es.addEventListener('positions', (e) => {
      try {
        setPositions(JSON.parse(e.data))
      } catch {
        // malformed JSON 무시
      }
    })

    es.onerror = () => {
      setConnected(false)
      setError('서버 연결 끊김 — 자동 재연결 중...')
    }

    return () => es.close()
  }, [])

  return { positions, connected, error }
}
