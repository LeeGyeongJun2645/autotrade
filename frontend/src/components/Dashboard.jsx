import { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

function BalanceCard({ title, children }) {
  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">{title}</h3>
      {children}
    </div>
  )
}

function Stat({ label, value, color = 'text-white' }) {
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-gray-400 text-sm">{label}</span>
      <span className={`font-mono text-sm font-medium ${color}`}>{value}</span>
    </div>
  )
}

function ProfitBadge({ rate }) {
  const pct = (rate * 100).toFixed(2)
  const positive = rate >= 0
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded ${positive ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
      {positive ? '+' : ''}{pct}%
    </span>
  )
}

export default function Dashboard({ sse }) {
  const { positions, connected, error } = sse
  const [kisBalance, setKisBalance] = useState(null)
  const [upbitBalance, setUpbitBalance] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchBalances = useCallback(async () => {
    try {
      const kis = await api.get('/balance/kis')
      setKisBalance(kis)
    } catch {
      // KIS 키 미설정 또는 API 오류 시 무시
    }
    try {
      const upbit = await api.get('/balance/upbit')
      setUpbitBalance(upbit)
    } catch {
      // Upbit 키 미설정 시 무시
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchBalances()
    const id = setInterval(fetchBalances, 30_000)
    return () => clearInterval(id)
  }, [fetchBalances])

  const fmt = (n) => Number(n).toLocaleString('ko-KR')
  const posEntries = Object.entries(positions)

  return (
    <div className="space-y-6">
      {/* SSE 상태 배너 */}
      {error && (
        <div className="bg-yellow-900/50 border border-yellow-600 text-yellow-300 text-sm px-4 py-2 rounded-lg">
          ⚠ {error}
        </div>
      )}

      {/* 잔고 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <BalanceCard title="KIS 계좌">
          {loading ? (
            <p className="text-gray-500 text-sm">불러오는 중...</p>
          ) : kisBalance ? (
            <>
              <Stat label="주문 가능 예수금" value={`${fmt(kisBalance.cash)}원`} color="text-blue-300" />
              <Stat label="총 평가금액" value={`${fmt(kisBalance.total_eval)}원`} />
              <Stat label="보유 종목" value={`${kisBalance.holdings?.length ?? 0}개`} />
            </>
          ) : (
            <p className="text-gray-500 text-sm">KIS API 키 미설정</p>
          )}
        </BalanceCard>

        <BalanceCard title="업비트 계좌">
          {loading ? (
            <p className="text-gray-500 text-sm">불러오는 중...</p>
          ) : upbitBalance ? (
            <>
              <Stat label="KRW 잔고" value={`${fmt(upbitBalance.krw)}원`} color="text-blue-300" />
              <Stat label="보유 코인" value={`${upbitBalance.holdings?.length ?? 0}개`} />
              {upbitBalance.holdings?.slice(0, 3).map((h) => (
                <Stat key={h.ticker} label={h.ticker} value={`${fmt(Math.round(h.current_value))}원`} />
              ))}
            </>
          ) : (
            <p className="text-gray-500 text-sm">Upbit API 키 미설정</p>
          )}
        </BalanceCard>
      </div>

      {/* 현재 포지션 */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700">
          <h3 className="font-semibold">현재 포지션</h3>
          <div className="flex items-center gap-2 text-xs">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
            <span className="text-gray-400">{connected ? 'SSE 실시간' : '연결 중...'}</span>
          </div>
        </div>

        {posEntries.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p className="text-2xl mb-2">📭</p>
            <p className="text-sm">보유 포지션 없음</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-xs border-b border-gray-700">
                  {['종목', '전략', '수량', '매수가', '손절가', '익절가', '개설일'].map((h) => (
                    <th key={h} className="text-left px-4 py-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {posEntries.map(([symbol, pos]) => {
                  const profitRate = (pos.highest_price - pos.entry_price) / pos.entry_price
                  return (
                    <tr key={symbol} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                      <td className="px-4 py-3 font-mono font-semibold text-white">{symbol}</td>
                      <td className="px-4 py-3">
                        <span className="bg-indigo-900/60 text-indigo-300 text-xs px-2 py-0.5 rounded">
                          {pos.strategy.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono">
                        {pos.is_crypto ? pos.qty.toFixed(8) : `${pos.qty}주`}
                      </td>
                      <td className="px-4 py-3 font-mono">{fmt(Math.round(pos.entry_price))}원</td>
                      <td className="px-4 py-3 font-mono text-red-400">{fmt(Math.round(pos.stop_loss_price))}원</td>
                      <td className="px-4 py-3 font-mono text-green-400">{fmt(Math.round(pos.take_profit_price))}원</td>
                      <td className="px-4 py-3 text-gray-400 text-xs">{pos.opened_at?.slice(0, 16).replace('T', ' ')}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
