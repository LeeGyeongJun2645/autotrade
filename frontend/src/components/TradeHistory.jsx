import { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

const STRATEGIES = [
  { value: 'volatility_breakout', label: '변동성 돌파' },
  { value: 'moving_average', label: '이동평균 크로스' },
  { value: 'rsi', label: 'RSI 역추세' },
]

function ResultCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-gray-700 rounded-lg p-4 text-center">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-xl font-bold font-mono ${color}`}>{value}</p>
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      {children}
    </div>
  )
}

const inputCls = 'w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500'

function BacktestPanel() {
  const [form, setForm] = useState({
    strategy: 'volatility_breakout',
    symbol: '',
    ticker: '',
    count: 120,
    initial_cash: 10000000,
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [useUpbit, setUseUpbit] = useState(false)

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }))

  const run = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const body = {
        strategy: form.strategy,
        count: Number(form.count),
        initial_cash: Number(form.initial_cash),
        ...(useUpbit ? { ticker: form.ticker } : { symbol: form.symbol }),
      }
      const data = await api.post('/backtest', body)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const fmtPct = (v, sign = true) => v == null ? '—' : `${sign && v > 0 ? '+' : ''}${v.toFixed(2)}%`
  const fmtKrw = (v) => `${Number(v).toLocaleString('ko-KR')}원`

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-5 space-y-4">
        <h3 className="font-semibold">백테스트</h3>

        <Field label="전략">
          <select value={form.strategy} onChange={(e) => set('strategy', e.target.value)} className={inputCls}>
            {STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
        </Field>

        <div className="flex gap-3">
          <button
            onClick={() => setUseUpbit(false)}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${!useUpbit ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
          >
            KIS (주식)
          </button>
          <button
            onClick={() => setUseUpbit(true)}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${useUpbit ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
          >
            Upbit (코인)
          </button>
        </div>

        {!useUpbit ? (
          <Field label="종목코드 (KIS)">
            <input
              value={form.symbol}
              onChange={(e) => set('symbol', e.target.value.toUpperCase())}
              placeholder="005930"
              className={inputCls}
            />
          </Field>
        ) : (
          <Field label="마켓 코드 (Upbit)">
            <input
              value={form.ticker}
              onChange={(e) => set('ticker', e.target.value.toUpperCase())}
              placeholder="KRW-BTC"
              className={inputCls}
            />
          </Field>
        )}

        <div className="grid grid-cols-2 gap-3">
          <Field label="데이터 봉 수">
            <input
              type="number"
              value={form.count}
              onChange={(e) => set('count', e.target.value)}
              min={15}
              max={200}
              className={inputCls}
            />
          </Field>
          <Field label="초기 자본금 (원)">
            <input
              type="number"
              value={form.initial_cash}
              onChange={(e) => set('initial_cash', e.target.value)}
              step={1000000}
              className={inputCls}
            />
          </Field>
        </div>

        {error && (
          <p className="text-red-400 text-sm bg-red-900/30 border border-red-800 rounded-lg px-3 py-2">{error}</p>
        )}

        <button
          onClick={run}
          disabled={loading || (!form.symbol && !form.ticker)}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 py-2.5 rounded-lg font-medium transition-colors"
        >
          {loading ? '백테스트 실행 중...' : '백테스트 실행'}
        </button>
      </div>

      {result && (
        <div className="bg-gray-800 rounded-xl border border-gray-700 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">결과</h3>
            <span className="text-xs text-gray-400">
              {fmtKrw(result.initial_cash)} → {fmtKrw(result.final_value)}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <ResultCard
              label="총 수익률"
              value={fmtPct(result.total_return_pct)}
              color={result.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <ResultCard
              label="최대 낙폭"
              value={`-${result.max_drawdown_pct.toFixed(2)}%`}
              color="text-red-400"
            />
            <ResultCard
              label="샤프 비율"
              value={result.sharpe_ratio.toFixed(3)}
              color={result.sharpe_ratio >= 1 ? 'text-green-400' : 'text-yellow-400'}
            />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <ResultCard label="총 거래" value={`${result.trade_count}회`} />
            <ResultCard
              label="승률"
              value={`${result.win_rate.toFixed(1)}%`}
              color={result.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}
            />
            <ResultCard label="손익비" value={`${result.win_count}W / ${result.loss_count}L`} />
          </div>
        </div>
      )}
    </div>
  )
}

const SIDE_COLORS = {
  BUY: 'text-green-400 bg-green-900/40',
  SELL: 'text-red-400 bg-red-900/40',
}

const REASON_LABELS = {
  STOP_LOSS: '손절',
  TAKE_PROFIT: '익절',
  TRAILING_STOP: '트레일링',
  STRATEGY_SELL: '전략 청산',
  EOD_VOLATILITY_BREAKOUT: 'EOD',
}

function TradeLogPanel() {
  const [trades, setTrades] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchTrades = useCallback(async () => {
    try {
      const data = await api.get('/trades?limit=50')
      setTrades(data)
    } catch {
      // 무시
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchTrades()
    const id = setInterval(fetchTrades, 30_000)
    return () => clearInterval(id)
  }, [fetchTrades])

  const fmt = (n) => Number(n).toLocaleString('ko-KR')
  const fmtPct = (v) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700">
        <h3 className="font-semibold">거래 기록</h3>
        <button
          onClick={fetchTrades}
          className="text-xs text-gray-400 hover:text-white transition-colors"
        >
          새로고침
        </button>
      </div>

      {loading ? (
        <div className="text-center py-10 text-gray-500 text-sm">불러오는 중...</div>
      ) : trades.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="text-2xl mb-2">📋</p>
          <p className="text-sm">거래 기록 없음</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-xs border-b border-gray-700">
                {['시각', '거래소', '종목', '구분', '전략', '수량', '가격', '수익률', '사유'].map((h) => (
                  <th key={h} className="text-left px-4 py-3 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.id} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                  <td className="px-4 py-3 text-gray-400 text-xs whitespace-nowrap">
                    {t.created_at?.slice(0, 16).replace('T', ' ')}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs bg-gray-700 px-2 py-0.5 rounded font-mono">{t.exchange}</span>
                  </td>
                  <td className="px-4 py-3 font-mono font-semibold">{t.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${SIDE_COLORS[t.side] ?? 'text-gray-400'}`}>
                      {t.side}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-300">
                    {t.strategy?.replace('_', ' ') ?? '—'}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs">
                    {t.exchange === 'UPBIT' ? (t.qty ?? 0).toFixed(6) : `${t.qty ?? 0}주`}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs">{fmt(Math.round(t.price))}원</td>
                  <td className={`px-4 py-3 font-mono text-xs ${t.profit_rate == null ? 'text-gray-500' : t.profit_rate >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {fmtPct(t.profit_rate)}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-400">
                    {REASON_LABELS[t.reason] ?? t.reason ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function TradeHistory() {
  const [view, setView] = useState('log')

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button
          onClick={() => setView('log')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${view === 'log' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-700/50'}`}
        >
          거래 기록
        </button>
        <button
          onClick={() => setView('backtest')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${view === 'backtest' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-700/50'}`}
        >
          백테스트
        </button>
      </div>

      {view === 'log' ? <TradeLogPanel /> : <BacktestPanel />}
    </div>
  )
}
