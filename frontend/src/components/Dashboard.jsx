import { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

const fmt = (n) => Number(Math.round(n)).toLocaleString('ko-KR')
const sign = (n) => n >= 0 ? '+' : ''

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

// ── 포트폴리오 히스토리 ───────────────────────────────────────────
function PortfolioHistory() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const res = await api.get('/portfolio/history?days=30')
      setData(res)
    } catch {
      setData(null)
    }
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load])

  const triggerSnapshot = async () => {
    try {
      await api.post('/portfolio/snapshot')
      await load()
    } catch { /* ignore */ }
  }

  if (loading) return null
  // data.history가 null/undefined인 경우(백엔드 응답 이상) 안전 처리
  if (!data || !Array.isArray(data.history) || data.history.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-5">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-white">포트폴리오 손익 히스토리</h3>
          <button onClick={triggerSnapshot}
            className="text-xs bg-indigo-700 hover:bg-indigo-600 text-white px-3 py-1.5 rounded">
            지금 스냅샷 저장
          </button>
        </div>
        <p className="text-gray-500 text-sm text-center py-4">
          아직 스냅샷 없음 — 매일 16:00 KST 자동 저장 (또는 위 버튼으로 즉시 저장)
        </p>
      </div>
    )
  }

  const initial = data.initial_capital
  const latest  = data.history[data.history.length - 1]
  // NaN/null 모두 0으로 처리 (isFinite 로 NaN·Infinity 방어)
  const pnlAmt  = Number.isFinite(latest?.pnl_amount) ? latest.pnl_amount : 0
  const pnlPct  = Number.isFinite(latest?.pnl_pct)    ? latest.pnl_pct    : 0

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="font-semibold text-white">포트폴리오 손익 히스토리</h3>
          <span className="text-xs text-gray-500">매일 16:00 자동 갱신</span>
        </div>
        <button onClick={triggerSnapshot}
          className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-1.5 rounded">
          지금 저장
        </button>
      </div>

      {/* 요약 카드 */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">원금</p>
          <p className="font-mono text-sm font-bold text-white">
            {initial > 0 ? `${fmt(initial)}원` : '—'}
          </p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">현재 총평가</p>
          <p className="font-mono text-sm font-bold text-white">{fmt(latest.total_value)}원</p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">원금 대비 손익</p>
          {initial > 0 ? (
            <>
              <p className={`font-mono text-sm font-bold ${pnlAmt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {sign(pnlAmt)}{fmt(Math.abs(pnlAmt))}원
              </p>
              <p className={`text-xs font-mono ${pnlPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {sign(pnlPct)}{pnlPct.toFixed(2)}%
              </p>
            </>
          ) : (
            <p className="text-gray-500 text-xs">원금 미설정</p>
          )}
        </div>
      </div>

      {/* 날짜별 테이블 (최근 14일) */}
      <div className="overflow-x-auto max-h-60 overflow-y-auto rounded-lg border border-gray-700/50">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-gray-800 border-b border-gray-700">
            <tr className="text-gray-400">
              {['날짜', 'KIS 현금', 'KIS 주식', '업비트 KRW', '업비트 코인', '총평가', '원금 대비'].map(h => (
                <th key={h} className="px-3 py-2 text-left font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...data.history].reverse().map((row, i) => {
              const pa = Number.isFinite(row.pnl_amount) ? row.pnl_amount : null
              const pp = Number.isFinite(row.pnl_pct)    ? row.pnl_pct    : null
              return (
                <tr key={i} className="border-b border-gray-700/40 hover:bg-gray-700/20">
                  <td className="px-3 py-1.5 text-gray-400 whitespace-nowrap">{row.snapshot_at?.slice(0, 16)}</td>
                  <td className="px-3 py-1.5 font-mono text-gray-300">{fmt(row.kis_cash)}</td>
                  <td className="px-3 py-1.5 font-mono text-gray-300">{fmt(row.kis_stocks)}</td>
                  <td className="px-3 py-1.5 font-mono text-gray-300">{fmt(row.upbit_krw)}</td>
                  <td className="px-3 py-1.5 font-mono text-gray-300">{fmt(row.upbit_coins)}</td>
                  <td className="px-3 py-1.5 font-mono font-semibold text-white">{fmt(row.total_value)}</td>
                  <td className={`px-3 py-1.5 font-mono font-semibold whitespace-nowrap ${
                    pa == null ? 'text-gray-600'
                    : pa >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {pa != null && pp != null
                      ? `${sign(pa)}${fmt(Math.abs(pa))}원 (${sign(pp)}${pp.toFixed(2)}%)`
                      : '원금 미설정'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── ML 신호 + 뉴스 헤드라인 ──────────────────────────────────────
function NewsPanel({ symbol }) {
  const [headlines, setHeadlines] = useState(null)
  const [loading, setLoading]     = useState(false)

  const toggle = async () => {
    if (loading) return                          // 로딩 중 재클릭 레이스 컨디션 방지
    if (headlines !== null) { setHeadlines(null); return }
    setLoading(true)
    try {
      const data = await api.get(`/news/${symbol}/headlines`)
      setHeadlines(data)
    } catch {
      setHeadlines({ headlines: [], news_score: null })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <button onClick={toggle} disabled={loading}
        className="text-xs text-indigo-400 hover:text-indigo-300 underline underline-offset-2 disabled:opacity-50">
        {loading ? '…' : headlines ? '닫기' : '뉴스 보기'}
      </button>
      {headlines && (
        <div className="mt-1 bg-gray-900/80 border border-gray-700 rounded-lg p-2 max-w-xs">
          {headlines.headlines.length === 0 ? (
            <p className="text-gray-600 text-xs">캐시 없음 (ML 신호 수신 후 1시간 유효)</p>
          ) : (
            <ul className="space-y-1">
              {headlines.headlines.slice(0, 8).map((h, i) => (
                <li key={i} className="text-xs text-gray-300 leading-snug">• {h}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

const SIGNAL_BADGE = {
  strong_buy:  { label: '강력매수', cls: 'bg-green-800 text-green-300 border-green-600' },
  buy:         { label: '매수',     cls: 'bg-green-900/60 text-green-400 border-green-700' },
  hold:        { label: '관망',     cls: 'bg-gray-700 text-gray-400 border-gray-600' },
  sell:        { label: '매도',     cls: 'bg-red-900/60 text-red-400 border-red-700' },
  strong_sell: { label: '강력매도', cls: 'bg-red-800 text-red-300 border-red-600' },
}

const FG_LABEL = (v) => {
  if (v === undefined || v === null) return null
  if (v <= -0.5) return { text: '극도공포', cls: 'text-blue-400' }
  if (v <= -0.1) return { text: '공포', cls: 'text-blue-300' }
  if (v < 0.1)   return { text: '중립', cls: 'text-gray-400' }
  if (v < 0.5)   return { text: '탐욕', cls: 'text-orange-300' }
  return { text: '극도탐욕', cls: 'text-orange-400' }
}

function MLSignalRow({ symbol, data }) {
  const b = SIGNAL_BADGE[data.signal] ?? SIGNAL_BADGE.hold
  const fg = FG_LABEL(data.fear_greed)
  return (
    <tr className="border-b border-gray-700/50 hover:bg-gray-700/20 align-top">
      <td className="px-4 py-2 font-mono text-sm text-white">{symbol}</td>
      <td className="px-4 py-2">
        <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${b.cls}`}>{b.label}</span>
      </td>
      <td className="px-4 py-2 font-mono text-sm">{data.buy_prob != null ? `${(data.buy_prob * 100).toFixed(1)}%` : '—'}</td>
      <td className={`px-4 py-2 font-mono text-sm ${(data.news_score ?? 0) > 0.05 ? 'text-green-400' : (data.news_score ?? 0) < -0.05 ? 'text-red-400' : 'text-gray-400'}`}>
        {data.news_score != null ? `${data.news_score > 0 ? '+' : ''}${data.news_score.toFixed(3)}` : '—'}
      </td>
      <td className={`px-4 py-2 text-xs ${fg ? fg.cls : 'text-gray-600'}`}>
        {fg ? fg.text : '—'}
      </td>
      <td className="px-4 py-2 text-xs text-gray-500">{data.checked_at ?? '-'}</td>
      <td className="px-4 py-2">
        <NewsPanel symbol={symbol} />
      </td>
    </tr>
  )
}

function RetrainButton() {
  const [status, setStatus] = useState(null)
  const [busy, setBusy]     = useState(false)

  const trigger = async () => {
    if (busy) return
    setBusy(true)
    setStatus(null)
    try {
      const res = await api.post('/agents/retrain')
      setStatus({ ok: true, msg: res.message })
    } catch {
      setStatus({ ok: false, msg: '재학습 요청 실패' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex items-center gap-3">
      <button onClick={trigger} disabled={busy}
        className="text-xs px-3 py-1.5 rounded-lg bg-indigo-700 hover:bg-indigo-600 disabled:opacity-40 text-white font-medium transition-colors">
        {busy ? '재학습 중…' : '🔄 에이전트 즉시 재학습'}
      </button>
      {status && (
        <span className={`text-xs ${status.ok ? 'text-green-400' : 'text-red-400'}`}>
          {status.msg}
        </span>
      )}
    </div>
  )
}

export default function Dashboard({ sse }) {
  const { positions, mlSignals, connected, error } = sse
  const [kisBalance, setKisBalance] = useState(null)
  const [upbitBalance, setUpbitBalance] = useState(null)
  const [upbitError, setUpbitError] = useState(null)
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
      setUpbitError(null)
    } catch (e) {
      setUpbitError(e?.message || 'API 오류')
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchBalances()
    const id = setInterval(fetchBalances, 30_000)
    return () => clearInterval(id)
  }, [fetchBalances])

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
            <p className="text-gray-500 text-sm">{upbitError ?? 'Upbit API 키 미설정'}</p>
          )}
        </BalanceCard>
      </div>

      {/* 에이전트 재학습 */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 px-5 py-3 flex items-center justify-between">
        <div>
          <span className="text-sm font-semibold">AI 에이전트 학습</span>
          <span className="text-xs text-gray-500 ml-2">매일 18:00 자동 재학습 | 수동 트리거 가능</span>
        </div>
        <RetrainButton />
      </div>

      {/* 포트폴리오 손익 히스토리 */}
      <PortfolioHistory />

      {/* ML 신호 현황 */}
      {Object.keys(mlSignals).length > 0 && (
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-700 flex items-center gap-2">
            <span className="text-sm font-semibold">ML 신호 현황</span>
            <span className="text-xs text-gray-500">매 정각 자동 갱신</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-xs border-b border-gray-700">
                  {['종목', '신호', '매수확률', '뉴스감성', '공포탐욕', '갱신', '뉴스 헤드라인'].map((h) => (
                    <th key={h} className="text-left px-4 py-2 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(mlSignals).map(([symbol, data]) => (
                  <MLSignalRow key={symbol} symbol={symbol} data={data} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

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
                  return (
                    <tr key={symbol} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                      <td className="px-4 py-3 font-mono font-semibold text-white">{symbol}</td>
                      <td className="px-4 py-3">
                        <span className="bg-indigo-900/60 text-indigo-300 text-xs px-2 py-0.5 rounded">
                          {(pos.strategy ?? '—').replace('_', ' ')}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono">
                        {pos.is_crypto ? (pos.qty ?? 0).toFixed(8) : `${pos.qty ?? 0}주`}
                      </td>
                      <td className="px-4 py-3 font-mono">{fmt(Math.round(pos.entry_price))}원</td>
                      <td className="px-4 py-3 font-mono text-red-400">{fmt(Math.round(pos.stop_loss_price))}원</td>
                      <td className="px-4 py-3 font-mono text-green-400">{fmt(Math.round(pos.take_profit_price))}원</td>
                      <td className="px-4 py-3 text-gray-400 text-xs">{pos.opened_at?.slice(0, 16)?.replace('T', ' ')}</td>
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
