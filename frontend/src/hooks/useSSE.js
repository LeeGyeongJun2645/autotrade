import { useState, useEffect } from 'react'
import { api } from '../api.js'

export function useSSE() {
  const [positions, setPositions] = useState({})
  const [simLogs, setSimLogs] = useState([])
  const [mlSignals, setMlSignals] = useState({})
  const [agents, setAgents] = useState([])
  const [fundingArb, setFundingArb] = useState([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const es = new EventSource(api.streamUrl('/stream'))

    es.onopen = () => {
      setConnected(true)
      setError(null)
    }

    es.addEventListener('positions', (e) => {
      try { setPositions(JSON.parse(e.data)) } catch (err) { console.warn('[SSE] positions 파싱 실패:', err) }
    })

    es.addEventListener('simlog', (e) => {
      try { setSimLogs(JSON.parse(e.data)) } catch (err) { console.warn('[SSE] simlog 파싱 실패:', err) }
    })

    es.addEventListener('mlsignal', (e) => {
      try { setMlSignals(JSON.parse(e.data)) } catch (err) { console.warn('[SSE] mlsignal 파싱 실패:', err) }
    })

    es.addEventListener('agents', (e) => {
      try { setAgents(JSON.parse(e.data)) } catch (err) { console.warn('[SSE] agents 파싱 실패:', err) }
    })

    es.addEventListener('funding_arb', (e) => {
      try { setFundingArb(JSON.parse(e.data)) } catch (err) { console.warn('[SSE] funding_arb 파싱 실패:', err) }
    })

    es.onerror = () => {
      setConnected(false)
      setError('서버 연결 끊김 — 자동 재연결 중...')
    }

    return () => es.close()
  }, [])

  return { positions, simLogs, mlSignals, agents, fundingArb, connected, error }
}
