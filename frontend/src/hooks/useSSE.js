import { useState, useEffect } from 'react'
import { api } from '../api.js'

export function useSSE() {
  const [positions, setPositions] = useState({})
  const [simLogs, setSimLogs] = useState([])
  const [mlSignals, setMlSignals] = useState({})
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const es = new EventSource(api.streamUrl('/stream'))

    es.onopen = () => {
      setConnected(true)
      setError(null)
    }

    es.addEventListener('positions', (e) => {
      try { setPositions(JSON.parse(e.data)) } catch {}
    })

    es.addEventListener('simlog', (e) => {
      try { setSimLogs(JSON.parse(e.data)) } catch {}
    })

    es.addEventListener('mlsignal', (e) => {
      try { setMlSignals(JSON.parse(e.data)) } catch {}
    })

    es.onerror = () => {
      setConnected(false)
      setError('서버 연결 끊김 — 자동 재연결 중...')
    }

    return () => es.close()
  }, [])

  return { positions, simLogs, mlSignals, connected, error }
}
