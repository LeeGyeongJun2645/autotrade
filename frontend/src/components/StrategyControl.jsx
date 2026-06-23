import { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

function Section({ title, children }) {
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-5 space-y-4">
      <h3 className="font-semibold text-sm text-gray-300 uppercase tracking-wide">{title}</h3>
      {children}
    </div>
  )
}

function TagList({ items, onRemove }) {
  if (!items.length) return <p className="text-gray-500 text-sm">없음</p>
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((item) => (
        <span key={item} className="flex items-center gap-1 bg-gray-700 rounded-full px-3 py-1 text-sm font-mono">
          {item}
          <button
            onClick={() => onRemove(item)}
            className="text-gray-400 hover:text-red-400 ml-1 transition-colors"
          >
            ×
          </button>
        </span>
      ))}
    </div>
  )
}

function AddInput({ placeholder, onAdd, loading }) {
  const [value, setValue] = useState('')

  const submit = async () => {
    const v = value.trim().toUpperCase()
    if (!v) return
    await onAdd(v)
    setValue('')
  }

  return (
    <div className="flex gap-2">
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && submit()}
        placeholder={placeholder}
        className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
      />
      <button
        onClick={submit}
        disabled={loading || !value.trim()}
        className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
      >
        추가
      </button>
    </div>
  )
}

export default function StrategyControl() {
  const [status, setStatus] = useState(null)
  const [symbols, setSymbols] = useState({ kis: [], upbit: [] })
  const [actionLoading, setActionLoading] = useState(false)
  const [msg, setMsg] = useState(null)

  const fetchStatus = useCallback(async () => {
    try {
      const [s, sym] = await Promise.all([
        api.get('/scheduler/status'),
        api.get('/symbols'),
      ])
      setStatus(s)
      setSymbols(sym)
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 10_000)
    return () => clearInterval(id)
  }, [fetchStatus])

  const showMsg = (type, text) => {
    setMsg({ type, text })
    setTimeout(() => setMsg(null), 3000)
  }

  const toggleScheduler = async () => {
    setActionLoading(true)
    try {
      const path = status?.running ? '/scheduler/stop' : '/scheduler/start'
      await api.post(path)
      await fetchStatus()
      showMsg('ok', status?.running ? '스케줄러를 중지했습니다.' : '스케줄러를 시작했습니다.')
    } catch (e) {
      showMsg('error', e.message)
    } finally {
      setActionLoading(false)
    }
  }

  const addKis = async (symbol) => {
    try {
      await api.post('/symbols/kis', { symbol })
      await fetchStatus()
      showMsg('ok', `KIS 종목 추가: ${symbol}`)
    } catch (e) {
      showMsg('error', e.message)
    }
  }

  const removeKis = async (symbol) => {
    try {
      await api.delete(`/symbols/kis/${symbol}`)
      await fetchStatus()
    } catch (e) {
      showMsg('error', e.message)
    }
  }

  const addUpbit = async (ticker) => {
    try {
      await api.post('/symbols/upbit', { ticker })
      await fetchStatus()
      showMsg('ok', `Upbit 티커 추가: ${ticker}`)
    } catch (e) {
      showMsg('error', e.message)
    }
  }

  const removeUpbit = async (ticker) => {
    try {
      await api.delete(`/symbols/upbit/${ticker}`)
      await fetchStatus()
    } catch (e) {
      showMsg('error', e.message)
    }
  }

  return (
    <div className="space-y-4">
      {msg && (
        <div className={`text-sm px-4 py-2 rounded-lg ${msg.type === 'ok' ? 'bg-green-900/50 text-green-300 border border-green-700' : 'bg-red-900/50 text-red-300 border border-red-700'}`}>
          {msg.text}
        </div>
      )}

      {/* 스케줄러 */}
      <Section title="스케줄러">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className={`w-3 h-3 rounded-full ${status?.running ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
            <span className="text-sm">{status?.running ? '실행 중' : '중지됨'}</span>
            <span className="text-xs text-gray-500">포지션 {status?.position_count ?? 0}개</span>
          </div>
          <button
            onClick={toggleScheduler}
            disabled={actionLoading}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 ${status?.running ? 'bg-red-700 hover:bg-red-600' : 'bg-green-700 hover:bg-green-600'}`}
          >
            {actionLoading ? '...' : status?.running ? '중지' : '시작'}
          </button>
        </div>
      </Section>

      {/* KIS 감시 종목 */}
      <Section title="KIS 감시 종목">
        <TagList items={symbols.kis} onRemove={removeKis} />
        <AddInput placeholder="종목코드 (예: 005930)" onAdd={addKis} />
      </Section>

      {/* Upbit 감시 티커 */}
      <Section title="Upbit 감시 티커">
        <TagList items={symbols.upbit} onRemove={removeUpbit} />
        <AddInput placeholder="마켓 코드 (예: KRW-BTC)" onAdd={addUpbit} />
      </Section>
    </div>
  )
}
