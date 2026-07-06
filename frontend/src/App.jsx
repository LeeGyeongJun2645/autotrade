import { useState, useEffect } from 'react'
import { api } from './api.js'
import { useSSE } from './hooks/useSSE.js'
import Dashboard from './components/Dashboard.jsx'
import StrategyControl from './components/StrategyControl.jsx'
import TradeHistory from './components/TradeHistory.jsx'
import ChartView from './components/ChartView.jsx'
import AgentDashboard from './components/AgentDashboard.jsx'
import NewsBanner from './components/NewsBanner.jsx'

const TABS = [
  { id: 'dashboard', label: '대시보드' },
  { id: 'chart',     label: '차트·시뮬' },
  { id: 'agents',    label: 'AI 경쟁' },
  { id: 'strategy',  label: '전략 관리' },
  { id: 'backtest',  label: '백테스트' },
]

export default function App() {
  const [tab, setTab]   = useState('dashboard')
  const [mode, setMode] = useState(null)
  const sse = useSSE()

  useEffect(() => {
    api.get('/health')
      .then((d) => setMode(d.mode))
      .catch(() => setMode('UNKNOWN'))
  }, [])

  // 뉴스 배너는 대시보드·에이전트 탭에서만 표시
  const showBanners = tab === 'dashboard' || tab === 'agents'

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* 헤더 */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold">⚡ AutoTrade</span>
            {mode && (
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border
                ${mode === 'PAPER'
                  ? 'border-yellow-500 text-yellow-400 bg-yellow-900/30'
                  : 'border-green-500 text-green-400 bg-green-900/30'}`}>
                {mode}
              </span>
            )}
          </div>
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors
                  ${tab === t.id
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'}`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {showBanners ? (
        /* 3-column 레이아웃: [코인뉴스] | [메인] | [주식/DART] */
        <div className="max-w-screen-2xl mx-auto flex gap-0" style={{ minHeight: 'calc(100vh - 57px)' }}>

          {/* 왼쪽 — 코인 뉴스 */}
          <aside className="w-64 shrink-0 sticky top-[57px] self-start" style={{ height: 'calc(100vh - 57px)' }}>
            <NewsBanner side="left" defaultCategory="coin" />
          </aside>

          {/* 메인 콘텐츠 */}
          <main className="flex-1 min-w-0 px-4 py-6">
            {tab === 'dashboard' && <Dashboard sse={sse} />}
            {tab === 'agents'    && <AgentDashboard agents={sse.agents} />}
          </main>

          {/* 오른쪽 — 주식·DART */}
          <aside className="w-64 shrink-0 sticky top-[57px] self-start" style={{ height: 'calc(100vh - 57px)' }}>
            <NewsBanner side="right" defaultCategory="stock" />
          </aside>
        </div>
      ) : (
        /* 단일 컬럼 레이아웃 */
        <main className="max-w-6xl mx-auto px-6 py-6">
          {tab === 'chart'    && <ChartView sse={sse} />}
          {tab === 'strategy' && <StrategyControl />}
          {tab === 'backtest' && <TradeHistory />}
        </main>
      )}
    </div>
  )
}
