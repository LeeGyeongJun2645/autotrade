import { useState, useEffect } from 'react'
import { api } from './api.js'
import Dashboard from './components/Dashboard.jsx'
import StrategyControl from './components/StrategyControl.jsx'
import TradeHistory from './components/TradeHistory.jsx'
import ChartView from './components/ChartView.jsx'

const TABS = [
  { id: 'dashboard', label: '대시보드' },
  { id: 'chart', label: '차트·시뮬' },
  { id: 'strategy', label: '전략 관리' },
  { id: 'backtest', label: '백테스트' },
]

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [mode, setMode] = useState(null)

  useEffect(() => {
    api.get('/health')
      .then((d) => setMode(d.mode))
      .catch(() => setMode('UNKNOWN'))
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* 헤더 */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold">⚡ AutoTrade</span>
            {mode && (
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${mode === 'PAPER' ? 'border-yellow-500 text-yellow-400 bg-yellow-900/30' : 'border-green-500 text-green-400 bg-green-900/30'}`}>
                {mode}
              </span>
            )}
          </div>

          {/* 탭 네비게이션 */}
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${tab === t.id ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700/50'}`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* 메인 콘텐츠 */}
      <main className="max-w-6xl mx-auto px-6 py-6">
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'chart' && <ChartView />}
        {tab === 'strategy' && <StrategyControl />}
        {tab === 'backtest' && <TradeHistory />}
      </main>
    </div>
  )
}
