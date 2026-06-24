import { useState, useEffect, useRef, useCallback } from 'react'
import { createChart, CandlestickSeries, LineSeries } from 'lightweight-charts'
import { api } from '../api.js'
import { useSSE } from '../hooks/useSSE.js'

const UPBIT_TICKERS = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']
const INTERVALS = [
  { value: 'minutes/60', label: '1시간' },
  { value: 'days', label: '일봉' },
  { value: 'weeks', label: '주봉' },
  { value: 'months', label: '월봉' },
]

const LEVEL_STYLES = {
  BUY:  { dot: 'bg-green-400', text: 'text-green-400' },
  SELL: { dot: 'bg-red-400',   text: 'text-red-400' },
  WARN: { dot: 'bg-yellow-400', text: 'text-yellow-400' },
  RISK: { dot: 'bg-orange-400', text: 'text-orange-400' },
  INFO: { dot: 'bg-gray-400',  text: 'text-gray-300' },
}

function SimLogPanel({ logs, connected }) {
  const endRef = useRef(null)

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 flex flex-col h-80">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <h3 className="text-sm font-semibold">시뮬레이션 로그</h3>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-gray-400">{connected ? '실시간' : '연결 중...'}</span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-1 font-mono text-xs">
        {logs.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            스케줄러를 시작하면 전략 신호가 여기에 표시됩니다
          </p>
        ) : (
          logs.map((log, i) => {
            const style = LEVEL_STYLES[log.level] ?? LEVEL_STYLES.INFO
            return (
              <div key={i} className="flex items-start gap-2">
                <span className="text-gray-500 shrink-0 w-16">{log.time}</span>
                <span className={`shrink-0 w-2 h-2 rounded-full mt-1 ${style.dot}`} />
                <span className="text-indigo-300 shrink-0 w-24 truncate">{log.symbol}</span>
                <span className={style.text}>{log.message}</span>
              </div>
            )
          })
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}

function CandleChart({ ticker, interval, positions }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const [loading, setLoading] = useState(true)
  const [currentPrice, setCurrentPrice] = useState(null)
  const [priceChange, setPriceChange] = useState(null)

  const loadChart = useCallback(async () => {
    if (!containerRef.current) return
    setLoading(true)

    try {
      const data = await api.get(`/chart/upbit/${ticker}?interval=${interval}&count=150`)

      // 최신순 → 오래된 순 정렬
      const candles = [...data].reverse().map((d) => ({
        time: d.date.slice(0, 10),
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }))

      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }

      const chart = createChart(containerRef.current, {
        layout: {
          background: { color: '#1f2937' },
          textColor: '#9ca3af',
        },
        grid: {
          vertLines: { color: '#374151' },
          horzLines: { color: '#374151' },
        },
        crosshair: { mode: 1 },
        rightPriceScale: { borderColor: '#374151' },
        timeScale: {
          borderColor: '#374151',
          timeVisible: true,
        },
        width: containerRef.current.clientWidth,
        height: 380,
      })

      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#22c55e',
        borderDownColor: '#ef4444',
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      })

      candleSeries.setData(candles)

      // 포지션 마커 추가
      const pos = positions[ticker]
      if (pos) {
        candleSeries.setMarkers([{
          time: pos.opened_at?.slice(0, 10),
          position: 'belowBar',
          color: '#22c55e',
          shape: 'arrowUp',
          text: `매수 ${Number(pos.entry_price).toLocaleString('ko-KR')}`,
        }])
      }

      chart.timeScale().fitContent()
      chartRef.current = chart
      seriesRef.current = candleSeries

      // 현재가 + 변동률
      if (candles.length >= 2) {
        const last = candles[candles.length - 1]
        const prev = candles[candles.length - 2]
        setCurrentPrice(last.close)
        setPriceChange(((last.close - prev.close) / prev.close) * 100)
      }

      // 반응형 리사이즈
      const ro = new ResizeObserver(() => {
        chart.applyOptions({ width: containerRef.current?.clientWidth ?? 600 })
      })
      ro.observe(containerRef.current)

    } catch (e) {
      console.error('차트 로드 실패', e)
    } finally {
      setLoading(false)
    }
  }, [ticker, interval, positions])

  useEffect(() => {
    loadChart()
    return () => { chartRef.current?.remove(); chartRef.current = null }
  }, [loadChart])

  const fmt = (n) => Number(n).toLocaleString('ko-KR')
  const positive = priceChange >= 0

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      {/* 헤더 */}
      <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-bold text-sm">{ticker}</span>
          {currentPrice && (
            <>
              <span className="font-mono text-lg font-semibold">{fmt(currentPrice)}원</span>
              <span className={`text-sm font-mono ${positive ? 'text-green-400' : 'text-red-400'}`}>
                {positive ? '+' : ''}{priceChange?.toFixed(2)}%
              </span>
            </>
          )}
        </div>
        <button onClick={loadChart} className="text-xs text-gray-400 hover:text-white transition-colors">
          새로고침
        </button>
      </div>

      {/* 차트 */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
            <span className="text-gray-400 text-sm">차트 로딩 중...</span>
          </div>
        )}
        <div ref={containerRef} />
      </div>
    </div>
  )
}

export default function ChartView() {
  const [ticker, setTicker] = useState('KRW-BTC')
  const [interval, setInterval] = useState('days')
  const { positions, simLogs, connected } = useSSE()

  return (
    <div className="space-y-4">
      {/* 종목 + 인터벌 선택 */}
      <div className="flex gap-2 flex-wrap">
        <div className="flex gap-1 bg-gray-800 rounded-lg p-1 border border-gray-700">
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
        <div className="flex gap-1 bg-gray-800 rounded-lg p-1 border border-gray-700">
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
      </div>

      {/* 캔들 차트 */}
      <CandleChart ticker={ticker} interval={interval} positions={positions} />

      {/* 시뮬레이션 로그 */}
      <SimLogPanel logs={simLogs} connected={connected} />
    </div>
  )
}
