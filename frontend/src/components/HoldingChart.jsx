import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, CandlestickSeries, LineSeries } from 'lightweight-charts'
import { api } from '../api.js'

const sign = n => n >= 0 ? '+' : ''
const fmt  = n => Number(Math.round(n)).toLocaleString('ko-KR')

function MiniChart({ ticker, isCoin, entryPrice }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    const chart = createChart(container, {
      width:  container.clientWidth,
      height: 140,
      layout: { background: { color: 'transparent' }, textColor: '#9ca3af' },
      grid:   { vertLines: { color: '#374151' }, horzLines: { color: '#374151' } },
      rightPriceScale: { borderColor: '#374151', scaleMargins: { top: 0.15, bottom: 0.1 } },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: { mode: 1 },
      handleScroll: false,
      handleScale: false,
    })
    chartRef.current = chart

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor:   '#22c55e',
      downColor: '#ef4444',
      borderUpColor:   '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor:   '#22c55e',
      wickDownColor: '#ef4444',
    })

    // 매수가 기준선
    if (entryPrice) {
      const lineSeries = chart.addSeries(LineSeries, {
        color: '#f59e0b',
        lineWidth: 1,
        lineStyle: 2, // dashed
        priceLineVisible: false,
        lastValueVisible: false,
      })
      candleSeries._entryLine = lineSeries
    }

    const url = isCoin
      ? `/chart/upbit/${ticker}?interval=minutes%2F5&count=80`
      : `/chart/kis/${ticker}?count=60`

    api.get(url).then(data => {
      if (!Array.isArray(data) || data.length === 0) { setError('데이터 없음'); return }
      const sorted = [...data].sort((a, b) => a.date < b.date ? -1 : 1)
      const toTs = dateStr => {
        const d = new Date(dateStr)
        return isCoin ? Math.floor(d.getTime() / 1000) : dateStr.slice(0, 10)
      }
      const candles = sorted.map(d => ({
        time:  toTs(d.date),
        open:  Number(d.open),
        high:  Number(d.high),
        low:   Number(d.low),
        close: Number(d.close),
      }))
      candleSeries.setData(candles)

      // 매수가 라인 (전체 범위)
      if (entryPrice && candleSeries._entryLine && candles.length >= 2) {
        candleSeries._entryLine.setData([
          { time: candles[0].time,  value: entryPrice },
          { time: candles[candles.length - 1].time, value: entryPrice },
        ])
      }
      chart.timeScale().fitContent()
    }).catch(() => setError('차트 로드 실패'))

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth })
    })
    ro.observe(container)

    return () => { ro.disconnect(); chart.remove() }
  }, [ticker, isCoin, entryPrice])

  if (error) return <div className="text-xs text-gray-600 text-center py-4">{error}</div>
  return <div ref={containerRef} className="w-full" />
}


export default function HoldingChart({ agents = [] }) {
  const [expanded, setExpanded] = useState({})
  const [newsScores, setNewsScores] = useState({})

  // 보유 포지션 수집
  const holdings = []
  for (const ag of agents) {
    if (!ag.positions) continue
    for (const [ticker, pos] of Object.entries(ag.positions)) {
      if (!holdings.find(h => h.ticker === ticker)) {
        holdings.push({
          ticker,
          isCoin:     ticker.startsWith('KRW-'),
          entryPrice: pos.entry_price ?? pos.avg_price ?? 0,
          pnlPct:     pos.unrealized_pnl_pct ?? 0,
          qty:        pos.qty ?? 0,
          enteredAt:  pos.entered_at ?? '',
          agents:     [ag.agent_id],
        })
      } else {
        holdings.find(h => h.ticker === ticker)?.agents.push(ag.agent_id)
      }
    }
  }

  const loadNewsScore = useCallback(async (ticker) => {
    if (newsScores[ticker] !== undefined) return
    try {
      const data = await api.get(`/news/${ticker}/headlines`)
      setNewsScores(prev => ({ ...prev, [ticker]: data.news_score ?? null }))
    } catch {
      setNewsScores(prev => ({ ...prev, [ticker]: null }))
    }
  }, [newsScores])

  const toggle = (ticker) => {
    setExpanded(prev => ({ ...prev, [ticker]: !prev[ticker] }))
    loadNewsScore(ticker)
  }

  if (holdings.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-4 text-center text-xs text-gray-500">
        현재 에이전트 보유 종목 없음
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">보유 종목 차트</h3>
        <span className="text-xs text-gray-500">{holdings.length}개 종목</span>
      </div>

      <div className="divide-y divide-gray-700">
        {holdings.map(h => {
          const isOpen = expanded[h.ticker]
          const ns = newsScores[h.ticker]
          const nsColor = ns == null ? 'text-gray-500'
            : ns > 0.2 ? 'text-green-400' : ns < -0.3 ? 'text-red-400' : 'text-yellow-400'

          return (
            <div key={h.ticker}>
              {/* 종목 헤더 — 클릭으로 차트 토글 */}
              <button
                onClick={() => toggle(h.ticker)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-700/30 transition-colors text-left"
              >
                <div className="flex items-center gap-3">
                  <span className={`w-2 h-2 rounded-full shrink-0 ${h.isCoin ? 'bg-yellow-400' : 'bg-blue-400'}`} />
                  <div>
                    <span className="font-mono font-bold text-white text-sm">
                      {h.ticker.replace('KRW-', '')}
                    </span>
                    <span className="text-gray-500 text-xs ml-2">
                      {h.agents.slice(0, 3).join(', ')}{h.agents.length > 3 ? ` +${h.agents.length - 3}` : ''}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {/* 뉴스 감성 점수 */}
                  {ns !== undefined && (
                    <span className={`text-xs font-mono ${nsColor}`}>
                      뉴스 {ns != null ? (ns > 0 ? '+' : '') + ns.toFixed(2) : '—'}
                    </span>
                  )}
                  {/* 미실현 손익 */}
                  <span className={`font-mono text-sm font-semibold ${h.pnlPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {sign(h.pnlPct)}{h.pnlPct.toFixed(2)}%
                  </span>
                  <span className={`text-gray-400 text-xs transition-transform ${isOpen ? 'rotate-180' : ''}`}>▼</span>
                </div>
              </button>

              {/* 차트 패널 */}
              {isOpen && (
                <div className="px-3 pb-3 bg-gray-900/40">
                  {/* 상세 정보 */}
                  <div className="flex gap-4 text-xs text-gray-400 mb-2 pt-2">
                    <span>매수가 <span className="text-white font-mono">{fmt(h.entryPrice)}원</span></span>
                    <span>수량 <span className="text-white font-mono">
                      {h.qty < 1 ? h.qty.toFixed(6) : fmt(h.qty)}
                    </span></span>
                    {h.enteredAt && (
                      <span>진입 <span className="text-white">{h.enteredAt.slice(5, 16)}</span></span>
                    )}
                    {ns != null && (
                      <span>감성 <span className={nsColor}>{ns > 0 ? '+' : ''}{ns.toFixed(2)}</span></span>
                    )}
                  </div>
                  <MiniChart ticker={h.ticker} isCoin={h.isCoin} entryPrice={h.entryPrice} />
                  <p className="text-xs text-gray-600 mt-1 text-center">
                    5분봉 — 주황 점선: 매수가 기준선
                  </p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
