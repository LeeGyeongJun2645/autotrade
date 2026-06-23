import { useState } from 'react'
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

export default function TradeHistory() {
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

  const fmtPct = (v, sign = true) => `${sign && v > 0 ? '+' : ''}${v.toFixed(2)}%`
  const fmtKrw = (v) => `${Number(v).toLocaleString('ko-KR')}원`

  return (
    <div className="space-y-6">
      {/* 백테스트 폼 */}
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

      {/* 결과 */}
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
