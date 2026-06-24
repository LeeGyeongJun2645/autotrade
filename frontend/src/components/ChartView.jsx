import { useState, useEffect, useRef, useCallback } from 'react'
import { createChart, CandlestickSeries } from 'lightweight-charts'
import { api } from '../api.js'

const UPBIT_TICKERS = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']
const INTERVALS = [
  { value: 'minutes/5',   label: '5분' },
  { value: 'minutes/15',  label: '15분' },
  { value: 'minutes/30',  label: '30분' },
  { value: 'minutes/60',  label: '1시간' },
  { value: 'minutes/240', label: '4시간' },
  { value: 'days',        label: '일봉' },
  { value: 'weeks',       label: '주봉' },
  { value: 'months',      label: '월봉' },
]

// 분봉/시간봉은 Unix timestamp, 일봉 이상은 날짜 문자열
const toChartTime = (dateStr, isIntraday) =>
  isIntraday ? Math.floor(new Date(dateStr).getTime() / 1000) : dateStr.slice(0, 10)
const LEVEL_STYLES = {
  BUY:  { dot: 'bg-green-400',  text: 'text-green-400' },
  SELL: { dot: 'bg-red-400',    text: 'text-red-400' },
  WARN: { dot: 'bg-yellow-400', text: 'text-yellow-400' },
  RISK: { dot: 'bg-orange-400', text: 'text-orange-400' },
  INFO: { dot: 'bg-gray-500',   text: 'text-gray-300' },
}

// ── 시뮬레이션 로그 패널 ──────────────────────────────────────────
function SimLogPanel({ logs, connected }) {
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 flex flex-col" style={{ height: 280 }}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 shrink-0">
        <h3 className="text-sm font-semibold">시뮬레이션 로그</h3>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-gray-400">{connected ? '실시간' : '연결 중...'}</span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-1 font-mono text-xs">
        {logs.length === 0 ? (
          <p className="text-gray-500 text-center pt-8">
            전략 관리 탭에서 스케줄러를 시작하면 신호가 표시됩니다
          </p>
        ) : (
          logs.map((log, i) => {
            const s = LEVEL_STYLES[log.level] ?? LEVEL_STYLES.INFO
            return (
              <div key={i} className="flex items-start gap-2 leading-5">
                <span className="text-gray-500 shrink-0 w-16">{log.time}</span>
                <span className={`shrink-0 w-2 h-2 rounded-full mt-1.5 ${s.dot}`} />
                <span className="text-indigo-300 shrink-0 w-24 truncate">{log.symbol}</span>
                <span className={s.text}>{log.message}</span>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

// ── 캔들 차트 ────────────────────────────────────────────────────
function CandleChart({ ticker, interval, positions }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const [loading, setLoading] = useState(false)
  const [info, setInfo] = useState(null)
  const posRef = useRef(positions)
  useEffect(() => { posRef.current = positions }, [positions])

  const buildChart = useCallback(async () => {
    if (!containerRef.current) return
    setLoading(true)

    // 기존 차트 완전 제거
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }
    containerRef.current.innerHTML = ''

    try {
      const isIntraday = interval.startsWith('minutes')
      const data = await api.get(`/chart/upbit/${ticker}?interval=${encodeURIComponent(interval)}&count=150`)
      const candles = [...data].reverse().map((d) => ({
        time: toChartTime(d.date, isIntraday),
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }))

      const chart = createChart(containerRef.current, {
        layout: { background: { color: '#1f2937' }, textColor: '#9ca3af' },
        grid: { vertLines: { color: '#374151' }, horzLines: { color: '#374151' } },
        crosshair: { mode: 1 },
        rightPriceScale: { borderColor: '#374151' },
        timeScale: { borderColor: '#374151', timeVisible: true },
        width: containerRef.current.clientWidth,
        height: 380,
      })

      const series = chart.addSeries(CandlestickSeries, {
        upColor: '#22c55e', downColor: '#ef4444',
        borderUpColor: '#22c55e', borderDownColor: '#ef4444',
        wickUpColor: '#22c55e', wickDownColor: '#ef4444',
      })

      series.setData(candles)

      // 포지션 매수 마커
      const pos = posRef.current[ticker]
      if (pos?.opened_at) {
        series.setMarkers([{
          time: pos.opened_at.slice(0, 10),
          position: 'belowBar',
          color: '#22c55e',
          shape: 'arrowUp',
          text: `매수 ${Number(pos.entry_price).toLocaleString('ko-KR')}`,
        }])
      }

      chart.timeScale().fitContent()
      chartRef.current = chart

      // 반응형
      const ro = new ResizeObserver(() => {
        if (containerRef.current && chartRef.current) {
          chartRef.current.applyOptions({ width: containerRef.current.clientWidth })
        }
      })
      ro.observe(containerRef.current)

      // 현재가 정보
      if (candles.length >= 2) {
        const last = candles.at(-1)
        const prev = candles.at(-2)
        setInfo({ price: last.close, change: ((last.close - prev.close) / prev.close) * 100 })
      }
    } catch (e) {
      console.error('차트 오류:', e)
    } finally {
      setLoading(false)
    }
  }, [ticker, interval]) // positions 제외 → 포지션 변경 시 차트 재생성 안 함

  useEffect(() => {
    buildChart()
    return () => {
      chartRef.current?.remove()
      chartRef.current = null
    }
  }, [buildChart])

  const fmt = (n) => Number(n).toLocaleString('ko-KR')

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-bold text-sm">{ticker}</span>
          {info && (
            <>
              <span className="font-mono text-lg font-semibold">{fmt(info.price)}원</span>
              <span className={`text-sm font-mono ${info.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {info.change >= 0 ? '+' : ''}{info.change.toFixed(2)}%
              </span>
            </>
          )}
        </div>
        <button onClick={buildChart} className="text-xs text-gray-400 hover:text-white transition-colors">
          새로고침
        </button>
      </div>
      <div className="relative bg-gray-900" style={{ minHeight: 380 }}>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-gray-900">
            <span className="text-gray-400 text-sm">차트 로딩 중...</span>
          </div>
        )}
        <div ref={containerRef} />
      </div>
    </div>
  )
}

// ── 메인 컴포넌트 ─────────────────────────────────────────────────
export default function ChartView({ sse }) {
  const [ticker, setTicker] = useState('KRW-BTC')
  const [interval, setInterval] = useState('days')
  const { simLogs, connected, positions } = sse

  return (
    <div className="space-y-4">
      {/* 종목 선택 */}
      <div className="flex gap-1 bg-gray-800 rounded-lg p-1 border border-gray-700 w-fit">
        {UPBIT_TICKERS.map((t) => (
          <button
            key={t}
            onClick={() => setTicker(t)}
            className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-colors ${ticker === t ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            {t.replace('KRW-', '')}
          </button>
        ))}
      </div>

      {/* 인터벌 선택 */}
      <div className="flex gap-1 bg-gray-800 rounded-lg p-1 border border-gray-700 w-fit">
        {INTERVALS.map((iv) => (
          <button
            key={iv.value}
            onClick={() => setInterval(iv.value)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${interval === iv.value ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            {iv.label}
          </button>
        ))}
      </div>

      <CandleChart ticker={ticker} interval={interval} positions={positions} />
      <SimLogPanel logs={simLogs} connected={connected} />
    </div>
  )
}
