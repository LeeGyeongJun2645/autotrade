import { useState, useEffect } from 'react'
import { api } from '../api.js'
import HoldingChart from './HoldingChart.jsx'

// ── 상수 ────────────────────────────────────────────────────────
const FEAT_COLOR = {
  all:      'bg-indigo-900/60 text-indigo-300',
  momentum: 'bg-green-900/60 text-green-300',
  trend:    'bg-blue-900/60 text-blue-300',
  volume:   'bg-orange-900/60 text-orange-300',
}
const FEAT_KR = {
  all:      '복합전략',
  momentum: '모멘텀',
  trend:    '추세전략',
  volume:   '거래량',
}
const FEAT_DESC = {
  all:      '모멘텀+추세+거래량+캔들패턴 48개 지표 전체 사용. 가장 많은 정보로 판단.',
  momentum: 'RSI·MACD·스토캐스틱 등 가격이 얼마나 빠르게 움직이는지로 단기 방향을 예측. 과매수/과매도 구간 탐지.',
  trend:    '이동평균(MA5/20/60)·ADX·EMA9/21·VWAP 기반. 추세가 형성된 방향으로 따라가는 전략.',
  volume:   'OBV·CMF·MFI 등 거래량 흐름 분석. 큰손(기관)의 매집·분산 움직임을 탐지.',
}
const INITIAL = 1_000_000

// ── 유틸 ────────────────────────────────────────────────────────
const fmt  = n => (n == null || !Number.isFinite(Number(n))) ? '—' : Number(Math.round(n)).toLocaleString('ko-KR')
const sign = n => n >= 0 ? '+' : ''
const label = a => `${a.agent_id}(${FEAT_KR[a.feature_set] ?? a.feature_set})`

// ── 공통 배지 ────────────────────────────────────────────────────
function ReturnBadge({ pct, size = 'sm' }) {
  if (pct == null || !Number.isFinite(pct)) return <span className="text-gray-600 text-xs">—</span>
  const pos = pct >= 0
  const sz  = size === 'lg' ? 'text-xl' : 'text-sm'
  return (
    <span className={`font-mono font-semibold ${sz} ${pos ? 'text-green-400' : 'text-red-400'}`}>
      {sign(pct)}{pct.toFixed(2)}%
    </span>
  )
}

function WinBadge({ rate, wins = null, total = null, size = 'sm' }) {
  if (rate == null || !Number.isFinite(rate)) return <span className="text-gray-600 text-xs">—</span>
  const color = rate >= 60 ? 'text-green-400' : rate >= 50 ? 'text-yellow-400' : 'text-red-400'
  const sz    = size === 'lg' ? 'text-2xl' : 'text-sm'
  return (
    <span className={`font-mono font-bold ${color} ${sz}`}>
      {rate.toFixed(1)}%
      {total !== null && (
        <span className="text-gray-500 font-normal text-xs ml-1">({wins}/{total})</span>
      )}
    </span>
  )
}

// ── 전략 설명 블록 ───────────────────────────────────────────────
function StrategyInfo({ featureSet }) {
  return (
    <div className={`text-xs px-2 py-1 rounded mt-1 ${FEAT_COLOR[featureSet] ?? ''} bg-opacity-30`}>
      <span className="font-semibold">{FEAT_KR[featureSet]}</span>
      {' — '}
      <span className="opacity-80">{FEAT_DESC[featureSet]}</span>
    </div>
  )
}

// ── 거래 행 ──────────────────────────────────────────────────────
function TradeRow({ t, stockNames = {} }) {
  const isBuy  = t.action === 'BUY'
  const pr     = t.profit_rate != null ? t.profit_rate * 100 : null
  const invest = isBuy
    ? t.price * t.qty
    : t.entry_price != null ? t.entry_price * t.qty : null
  const pnl    = !isBuy && t.entry_price != null
    ? (t.price - t.entry_price) * t.qty
    : null

  return (
    <tr className="border-b border-gray-700/40 text-xs hover:bg-gray-700/20">
      <td className="px-2 py-1.5 font-semibold text-gray-200">
        {stockNames[t.ticker]
          ? <><span>{stockNames[t.ticker]}</span><span className="text-gray-600 ml-1 font-normal">{t.ticker}</span></>
          : <span className="font-mono">{t.ticker?.replace('KRW-', '')}</span>}
      </td>
      <td className="px-2 py-1.5">
        <span className={`font-semibold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>
          {isBuy ? '매수' : '매도'}
        </span>
      </td>
      <td className="px-2 py-1.5 font-mono text-white text-right">
        {fmt(t.price)}원
      </td>
      <td className="px-2 py-1.5 font-mono text-gray-400 text-right">
        {invest != null ? `${fmt(invest)}원` : '—'}
      </td>
      <td className={`px-2 py-1.5 font-mono font-semibold text-right ${
        pnl == null ? 'text-gray-600'
        : pnl >= 0  ? 'text-green-400' : 'text-red-400'
      }`}>
        {isBuy ? '—'
          : pnl != null ? `${sign(pnl)}${fmt(pnl)}원` : '—'}
      </td>
      <td className={`px-2 py-1.5 font-mono font-semibold text-right ${
        isBuy ? 'text-gray-600'
        : pr == null ? 'text-gray-600'
        : pr >= 0 ? 'text-green-400' : 'text-red-400'
      }`}>
        {isBuy ? '—' : pr != null ? `${sign(pr)}${pr.toFixed(2)}%` : '—'}
      </td>
      {/* 매도 후 잔액 — 팔고 나서 현재 내 잔액이 얼마인지 */}
      <td className="px-2 py-1.5 font-mono text-right">
        {!isBuy && t.balance != null
          ? <span className="text-gray-300">{fmt(t.balance)}원</span>
          : <span className="text-gray-700">—</span>}
      </td>
      <td className="px-2 py-1.5 text-gray-500 whitespace-nowrap">
        {t.traded_at?.slice(11, 16)}
      </td>
    </tr>
  )
}

// ── 포지션 카드 ──────────────────────────────────────────────────
function PosCard({ ticker, pos, isCoin }) {
  const pnl      = pos.unrealized_pnl_pct ?? 0
  const invested = (pos.entry_price ?? 0) * (pos.qty ?? 0)
  const curVal   = pos.current_value ?? invested

  return (
    <div className={`rounded-lg border p-3 text-xs space-y-1.5
      ${isCoin
        ? 'bg-yellow-900/15 border-yellow-800/40'
        : 'bg-blue-900/15 border-blue-800/40'}`}>
      {/* 종목명 + 미실현 손익 */}
      <div className="flex items-center justify-between">
        <span className="font-mono font-bold text-white text-sm">
          {ticker.replace('KRW-', '')}
        </span>
        <span className={`font-bold text-sm ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {sign(pnl)}{pnl.toFixed(2)}%
        </span>
      </div>
      {/* 매수가 */}
      <div className="flex justify-between text-gray-400">
        <span>매수가</span>
        <span className="font-mono text-gray-200">{fmt(pos.entry_price)}원</span>
      </div>
      {/* 수량 */}
      <div className="flex justify-between text-gray-400">
        <span>수량</span>
        <span className="font-mono text-gray-200">
          {(pos.qty ?? 0) < 1 ? (pos.qty ?? 0).toFixed(6) : fmt(pos.qty ?? 0)}
        </span>
      </div>
      {/* 투자금액 */}
      <div className="flex justify-between text-gray-400">
        <span>투자금</span>
        <span className="font-mono text-gray-200">{fmt(invested)}원</span>
      </div>
      {/* 현재 평가금 */}
      <div className={`flex justify-between font-semibold ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        <span>현재 평가</span>
        <span className="font-mono">{fmt(curVal)}원</span>
      </div>
      {/* 매수 시각 */}
      {pos.entered_at && (
        <div className="text-gray-600 text-right">{pos.entered_at.slice(11, 16)} 매수</div>
      )}
    </div>
  )
}

// ── 에이전트 상세 패널 ────────────────────────────────────────────
function AgentDetail({ agent, stockNames = {} }) {
  const [dbTrades, setDbTrades] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const isCoin   = agent.market === 'coin'
  const posEnt   = Object.entries(agent.positions ?? {})
  const totalVal = agent.total_value ?? agent.balance ?? INITIAL
  const pnlAmt   = totalVal - INITIAL

  const loadTrades = async () => {
    if (dbTrades) { setDbTrades(null); return }
    setLoading(true)
    let mounted = true
    try {
      const data = await api.get(`/agents/${agent.agent_id}/trades?limit=200`)
      if (mounted) setDbTrades(data)
    } catch { if (mounted) setDbTrades([]) }
    if (mounted) setLoading(false)
    return () => { mounted = false }
  }

  const trades = (dbTrades ?? agent.recent_trades ?? []).slice(0, 20)

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-5 space-y-4">

      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <h3 className="font-bold text-white text-lg">{label(agent)}</h3>
          {agent.is_champion && (
            <span className="text-xs bg-yellow-700/60 text-yellow-300 px-2 py-0.5 rounded">
              🏆 게이트 챔피언 (총자산 1위)
            </span>
          )}
          <span className={`text-xs px-2 py-0.5 rounded ${FEAT_COLOR[agent.feature_set] ?? ''}`}>
            {agent.feature_set}
          </span>
          <span className="text-xs text-gray-500">
            {isCoin ? '코인 24/7' : '주식 장중 09:00~15:30'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {agent.trained_at && (
            <span className="text-xs text-gray-600">학습: {agent.trained_at.slice(5, 16)}</span>
          )}
          <button onClick={loadTrades}
            className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-1.5 rounded">
            {loading ? '로딩 중...' : dbTrades ? 'DB 닫기' : 'DB 거래 기록'}
          </button>
        </div>
      </div>

      {/* 전략 설명 */}
      <StrategyInfo featureSet={agent.feature_set} />

      {/* 전략 설정 */}
      <div className="grid grid-cols-4 gap-2">
        {[
          ['인터벌', `${agent.interval_min ?? 5}분봉`],
          ['레이블 기준', agent.label_threshold != null ? `+${(agent.label_threshold * 100).toFixed(1)}%` : '—'],
          ['매수 임계값', agent.buy_threshold != null ? `${(agent.buy_threshold * 100).toFixed(0)}%` : '—'],
          ['챔피언 선정', '총자산 기준'],
        ].map(([k, v]) => (
          <div key={k} className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">{k}</p>
            <p className="font-mono text-sm text-white">{v}</p>
          </div>
        ))}
      </div>

      {/* 성과 4박스 */}
      <div className="grid grid-cols-5 gap-2">
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">승률</p>
          <WinBadge rate={agent.win_rate} wins={agent.win_trades} total={agent.total_trades} />
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">누적 수익률</p>
          <ReturnBadge pct={agent.total_return_pct} />
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">오늘 수익률</p>
          <ReturnBadge pct={agent.today_return_pct ?? 0} />
          <p className="text-xs text-gray-500 mt-0.5">일 {agent.today_buy_count ?? 0}건</p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">총평가액 (챔피언 기준)</p>
          <p className="font-mono text-sm text-white">{fmt(totalVal)}원</p>
          <p className={`text-xs font-mono mt-0.5 ${pnlAmt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            원금 대비 {sign(pnlAmt)}{fmt(Math.abs(pnlAmt))}원
          </p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <p className="text-xs text-gray-400 mb-1">현금 / 포지션평가</p>
          <p className="font-mono text-xs text-gray-300">{fmt(agent.balance)}원</p>
          <p className="font-mono text-xs text-gray-500">+ {fmt(agent.position_value ?? 0)}원</p>
          {posEnt.length > 0 && (
            <p className={`text-xs mt-0.5 font-semibold
              ${isCoin ? 'text-yellow-300' : 'text-blue-300'}`}>
              {posEnt.length}종목 보유 중
            </p>
          )}
        </div>
      </div>

      {/* 현재 포지션 */}
      {posEnt.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-2">
            현재 보유 포지션 —{' '}
            <span className="text-white font-bold">{posEnt.length}종목</span>
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {posEnt.map(([ticker, pos]) => (
              <PosCard key={ticker} ticker={ticker} pos={pos} isCoin={isCoin} />
            ))}
          </div>
        </div>
      )}

      {/* 거래 기록 */}
      <div>
        <p className="text-xs text-gray-400 mb-2">
          거래 기록 최근{' '}
          <span className="text-white font-bold">{trades.length}건</span>
          {dbTrades && <span className="text-indigo-400 ml-1">(DB 전체)</span>}
        </p>
        {trades.length === 0 ? (
          <div className="text-center py-6 text-gray-600 text-sm border border-gray-700/50 rounded-lg">
            {agent.total_trades === 0
              ? `신호 대기 중 — 매수 임계값 ${agent.buy_threshold != null ? (agent.buy_threshold * 100).toFixed(0) : '?'}% 이상 신호 없음 (ADX 레짐 필터 적용 중)`
              : '기록 없음'}
          </div>
        ) : (
          <div className="overflow-x-auto max-h-80 overflow-y-auto rounded-lg border border-gray-700/50">
            <table className="w-full">
              <thead className="sticky top-0 bg-gray-800 border-b border-gray-700">
                <tr className="text-gray-400 text-xs">
                  {[
                    ['종목', 'text-left'],
                    ['구분', 'text-left'],
                    ['체결가', 'text-right'],
                    ['투자금', 'text-right'],
                    ['손익금액', 'text-right'],
                    ['수익률', 'text-right'],
                    ['매도 후 잔액', 'text-right'],
                    ['시각', 'text-left'],
                  ].map(([h, align]) => (
                    <th key={h} className={`${align} px-2 py-2 font-medium whitespace-nowrap`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trades.map((t, idx) => <TradeRow key={`${t.id ?? t.traded_at}-${t.ticker}-${idx}`} t={t} stockNames={stockNames} />)}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── 챔피언 카드 ──────────────────────────────────────────────────
function ChampionCard({ agent, onClick, selected, market }) {
  const isCoin   = market === 'coin'
  const posCount = Object.keys(agent.positions ?? {}).length
  const totalVal = agent.total_value ?? agent.balance ?? INITIAL
  const pnlAmt   = totalVal - INITIAL

  return (
    <div
      onClick={onClick}
      className={`rounded-2xl border-2 cursor-pointer transition-all p-5
        border-yellow-500 shadow-lg shadow-yellow-900/30
        ${isCoin ? 'bg-yellow-950/25' : 'bg-blue-950/25'}
        ${selected ? 'ring-2 ring-yellow-400/50' : 'hover:shadow-yellow-900/50'}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold bg-yellow-700/60 text-yellow-300 px-2 py-1 rounded-full">
            🏆 총자산 1위 (앙상블 최다 발언권)
          </span>
          <span className="text-lg font-extrabold text-white">{label(agent)}</span>
        </div>
        <span className="text-xs text-gray-500">실매매 게이트: 전 에이전트 앙상블 투표 중</span>
      </div>

      {/* 전략 설명 */}
      <div className="mb-3">
        <StrategyInfo featureSet={agent.feature_set} />
      </div>

      <div className="grid grid-cols-4 gap-3 mb-3">
        {/* 총평가액 — 챔피언 선정 기준 */}
        <div className={`rounded-xl p-3 col-span-1 ${isCoin ? 'bg-yellow-900/30 border border-yellow-700/50' : 'bg-blue-900/30 border border-blue-700/50'}`}>
          <p className="text-xs text-gray-400 mb-1">총평가액 <span className="text-yellow-500">★선정기준</span></p>
          <p className="font-mono text-base font-bold text-white">{fmt(totalVal)}원</p>
          <p className={`text-xs mt-0.5 font-mono font-semibold ${pnlAmt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {sign(pnlAmt)}{fmt(Math.abs(pnlAmt))}원
          </p>
        </div>
        <div className={`rounded-xl p-3 ${isCoin ? 'bg-yellow-900/20' : 'bg-blue-900/20'}`}>
          <p className="text-xs text-gray-400 mb-1">수익률</p>
          <ReturnBadge pct={agent.total_return_pct} size="lg" />
        </div>
        <div className={`rounded-xl p-3 ${isCoin ? 'bg-yellow-900/20' : 'bg-blue-900/20'}`}>
          <p className="text-xs text-gray-400 mb-1">승률</p>
          <WinBadge rate={agent.win_rate} wins={agent.win_trades} total={agent.total_trades} size="lg" />
        </div>
        <div className={`rounded-xl p-3 ${isCoin ? 'bg-yellow-900/20' : 'bg-blue-900/20'}`}>
          <p className="text-xs text-gray-400 mb-1">현금 / 포지션</p>
          <p className="font-mono text-xs text-gray-300">{fmt(agent.balance)}원</p>
          <p className="font-mono text-xs text-gray-500">+ {fmt(agent.position_value ?? 0)}원</p>
          {posCount > 0 && (
            <p className={`text-xs mt-1 font-semibold ${isCoin ? 'text-yellow-300' : 'text-blue-300'}`}>
              {posCount}종목 보유
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-4 text-xs text-gray-600">
        <span>레이블 +{agent.label_threshold != null ? (agent.label_threshold * 100).toFixed(1) : '?'}% · 임계 {agent.buy_threshold != null ? (agent.buy_threshold * 100).toFixed(0) : '?'}%</span>
        {agent.trained_at && <span>최근 학습: {agent.trained_at.slice(5, 16)}</span>}
      </div>
    </div>
  )
}

// ── 일반 에이전트 카드 ────────────────────────────────────────────
function AgentCard({ agent, onClick, selected }) {
  const isCoin   = agent.market === 'coin'
  const posEnt   = Object.entries(agent.positions ?? {})
  const noTrades = agent.total_trades === 0
  const totalVal = agent.total_value ?? agent.balance ?? INITIAL
  const pnlAmt   = totalVal - INITIAL

  // 보유 포지션 투자 원금 (고정값 — 시장가와 무관)
  const invested = posEnt.reduce((s, [, pos]) => s + (pos.entry_price ?? 0) * (pos.qty ?? 0), 0)
  const posVal   = agent.position_value ?? 0
  const unrealized = posVal - invested  // 미실현 손익

  // 최근 매도 거래의 실현 손익
  const lastSell   = Array.isArray(agent.recent_trades) ? agent.recent_trades.find(t => t.action === 'SELL') : undefined
  const lastPnlAmt = lastSell?.entry_price != null
    ? (lastSell.price - lastSell.entry_price) * lastSell.qty
    : null
  const lastPnlPct = lastSell?.profit_rate != null ? lastSell.profit_rate * 100 : null

  return (
    <div
      onClick={onClick}
      className={`rounded-xl p-3 border cursor-pointer transition-all flex flex-col
        ${noTrades
          ? 'bg-gray-800/40 border-gray-700/40'
          : 'bg-gray-800 border-gray-700 hover:border-gray-500'}
        ${selected ? 'border-indigo-500 ring-1 ring-indigo-500/30' : ''}`}
    >
      {/* 이름 + 포지션 뱃지 */}
      <div className="flex items-center justify-between mb-0.5">
        <span className={`font-bold text-sm leading-tight ${noTrades ? 'text-gray-500' : 'text-white'}`}>
          {label(agent)}
        </span>
        {posEnt.length > 0 && (
          <span className={`text-xs px-1.5 py-0.5 rounded font-semibold
            ${isCoin ? 'bg-yellow-900/40 text-yellow-300' : 'bg-blue-900/40 text-blue-300'}`}>
            {posEnt.length}종목
          </span>
        )}
      </div>

      {/* 전략 한줄 설명 */}
      <p className="text-xs text-gray-600 leading-tight mb-2">
        {FEAT_DESC[agent.feature_set]?.slice(0, 30)}…
      </p>

      <div className="text-xs text-gray-600 mb-2">
        레이블 +{agent.label_threshold != null ? (agent.label_threshold * 100).toFixed(1) : '?'}% · 임계 {agent.buy_threshold != null ? (agent.buy_threshold * 100).toFixed(0) : '?'}%
      </div>

      {noTrades ? (
        <p className="text-xs text-gray-600 text-center py-2 border border-gray-700/30 rounded">
          신호 대기 중
        </p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-1.5 mb-2">
            <div>
              <p className="text-xs text-gray-500">승률</p>
              <WinBadge rate={agent.win_rate} wins={agent.win_trades} total={agent.total_trades} />
            </div>
            <div>
              <p className="text-xs text-gray-500">수익률</p>
              <ReturnBadge pct={agent.total_return_pct} />
            </div>
          </div>

          <div className="text-xs border-t border-gray-700/50 pt-1.5 space-y-0.5">
            {/* 잔액 (현금) */}
            <div className="flex justify-between">
              <span className="text-gray-500">잔액</span>
              <span className="font-mono text-gray-300">{fmt(agent.balance)}원</span>
            </div>

            {/* 보유 중일 때: 투자금(고정) / 현재 평가(변동) / 미실현 손익 */}
            {invested > 0 && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-500">투자금</span>
                  <span className="font-mono text-gray-400">{fmt(invested)}원</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">현재 평가</span>
                  <span className="font-mono text-gray-300">{fmt(posVal)}원</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">미실현</span>
                  <span className={`font-mono font-semibold ${unrealized >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {sign(unrealized)}{fmt(Math.abs(unrealized))}원
                  </span>
                </div>
              </>
            )}

            {/* 최근 실현 손익 (마지막 매도) */}
            {lastPnlAmt != null && (
              <div className="flex justify-between border-t border-gray-700/30 pt-0.5 mt-0.5">
                <span className="text-gray-500">최근 실현</span>
                <span className={`font-mono font-semibold ${lastPnlAmt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {sign(lastPnlAmt)}{fmt(Math.abs(lastPnlAmt))}원
                  {lastPnlPct != null && (
                    <span className="text-xs opacity-70 ml-1">({sign(lastPnlPct)}{lastPnlPct.toFixed(2)}%)</span>
                  )}
                </span>
              </div>
            )}

            {/* 총평가 + 원금 대비 */}
            <div className="flex justify-between border-t border-gray-700/30 pt-0.5 mt-0.5">
              <span className="text-gray-500">총평가</span>
              <span className="font-mono text-white font-semibold">{fmt(totalVal)}원</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">원금 대비</span>
              <span className={`font-mono font-semibold ${pnlAmt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {sign(pnlAmt)}{fmt(Math.abs(pnlAmt))}원
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ── 섹션 요약 ─────────────────────────────────────────────────────
function SectionSummary({ agents, market }) {
  const active    = agents.filter(a => a.total_trades > 0)
  const bestVal   = active.length
    ? (Math.max(...active.map(a => a.total_value ?? a.balance ?? INITIAL)) || INITIAL)
    : INITIAL
  const champ     = agents.find(a => a.is_champion)
  const isCoin    = market === 'coin'

  // total_trades = 매도 완료 건수, 미청산 포지션 = 매수 후 미매도
  const totalSells = agents.reduce((s, a) => s + a.total_trades, 0)
  const totalBuys  = agents.reduce((s, a) => s + a.total_trades + Object.keys(a.positions ?? {}).length, 0)

  return (
    <div className={`grid grid-cols-5 gap-3 mb-4 p-3 rounded-lg
      ${isCoin
        ? 'bg-yellow-900/10 border border-yellow-800/30'
        : 'bg-blue-900/10 border border-blue-800/30'}`}>
      <div>
        <p className="text-xs text-gray-500">경쟁 에이전트</p>
        <p className="font-bold text-white">
          {agents.length}개
          <span className="text-xs text-gray-500 font-normal ml-1">({active.length}개 활성)</span>
        </p>
      </div>
      <div>
        <p className="text-xs text-gray-500">총 매수</p>
        <p className="font-bold text-green-400">{totalBuys}건</p>
      </div>
      <div>
        <p className="text-xs text-gray-500">총 매도</p>
        <p className="font-bold text-red-400">{totalSells}건</p>
      </div>
      <div>
        <p className="text-xs text-gray-500">최고 총평가액</p>
        <p className={`font-bold font-mono text-sm ${bestVal > INITIAL ? 'text-green-400' : bestVal < INITIAL ? 'text-red-400' : 'text-gray-400'}`}>
          {fmt(bestVal)}원
        </p>
      </div>
      <div>
        <p className="text-xs text-gray-500">현재 챔피언</p>
        <p className="font-bold text-yellow-300 text-sm">
          {champ ? label(champ) : '아직 거래 없음'}
        </p>
      </div>
    </div>
  )
}

// ── 마켓 섹션 ─────────────────────────────────────────────────────
function MarketSection({ agents, market, selected, onSelect, stockNames }) {
  const isCoin       = market === 'coin'
  const champion     = agents.find(a => a.is_champion)
  const rest         = agents.filter(a => !a.is_champion)
  const selectedAgent = agents.find(a => a.agent_id === selected)

  if (agents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-600 text-sm border border-gray-700 rounded-xl">
        서버 시작 후 5분 뒤 첫 실행
      </div>
    )
  }

  return (
    <>
      <SectionSummary agents={agents} market={market} />

      {/* 챔피언 카드 (클릭 시 상세 토글) */}
      {champion ? (
        <div className="mb-4">
          <ChampionCard
            agent={champion}
            market={market}
            selected={selected === champion.agent_id}
            onClick={() => onSelect(champion.agent_id)}
          />
        </div>
      ) : (
        <div className={`mb-4 rounded-2xl border-2 border-dashed p-4 text-center text-sm
          ${isCoin
            ? 'border-yellow-800/40 text-yellow-700'
            : 'border-blue-800/40 text-blue-700'}`}>
          앙상블 게이트 실행 중 — 총자산 1위 에이전트가 최다 발언권 보유 (10회 이상 거래 필요)
        </div>
      )}

      {/* 선택된 에이전트 상세 — 카드 그리드 위에 표시 */}
      {selectedAgent && (
        <div className="mb-4">
          <AgentDetail key={selectedAgent.agent_id} agent={selectedAgent} stockNames={stockNames} />
        </div>
      )}

      {/* 나머지 에이전트 카드 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 items-start">
        {rest.map(agent => (
          <AgentCard
            key={agent.agent_id}
            agent={agent}
            selected={selected === agent.agent_id}
            onClick={() => onSelect(agent.agent_id)}
          />
        ))}
      </div>
    </>
  )
}

// ── 분석 패널 ─────────────────────────────────────────────────────
function AnalysisPanel() {
  const [data,   setData]   = useState(null)
  const [loading,setLoading]= useState(false)
  const [open,   setOpen]   = useState(false)

  const load = async () => {
    if (loading) return
    setLoading(true)
    try {
      const r = await api.get('/agents/analysis')
      setData(r)
      setOpen(true)
    } catch { setData(null) }
    finally { setLoading(false) }
  }

  const s = data?.summary ?? {}
  const byTicker   = data?.by_ticker   ?? []
  const byHour     = data?.by_hour     ?? []
  const byStrategy = data?.by_strategy ?? []

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="font-bold text-white text-sm">📊 거래 분석</span>
        <button onClick={open ? () => setOpen(false) : load}
          className="text-xs px-3 py-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
          {loading ? '분석 중…' : open ? '접기' : '분석 보기'}
        </button>
      </div>
      {open && data && (
        <div className="space-y-4">
          {/* 전체 요약 */}
          <div className="grid grid-cols-4 gap-3 text-center">
            {[
              ['총 매도', s.total_sells ?? 0, 'text-white'],
              ['전체 승률', s.overall_win_rate != null ? `${s.overall_win_rate}%` : '—',
               s.overall_win_rate >= 55 ? 'text-green-400' : s.overall_win_rate >= 50 ? 'text-yellow-400' : 'text-red-400'],
              ['평균 손익', s.avg_pnl_pct != null ? `${s.avg_pnl_pct > 0 ? '+' : ''}${s.avg_pnl_pct}%` : '—',
               s.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'],
              ['종목 수', s.unique_tickers ?? 0, 'text-gray-300'],
            ].map(([k, v, cls]) => (
              <div key={k} className="bg-gray-700/50 rounded-lg p-2">
                <p className="text-xs text-gray-400">{k}</p>
                <p className={`font-bold font-mono text-sm ${cls}`}>{v}</p>
              </div>
            ))}
          </div>

          {/* 전략별 */}
          <div>
            <p className="text-xs text-gray-400 mb-1.5 font-semibold">전략별 성과</p>
            <div className="grid grid-cols-2 gap-2">
              {byStrategy.map(s => (
                <div key={`${s.feature_set}_${s.market}`}
                  className="bg-gray-700/40 rounded-lg px-3 py-2 flex justify-between items-center">
                  <span className="text-xs text-gray-300">
                    {FEAT_KR[s.feature_set] ?? s.feature_set}
                    <span className="text-gray-500 ml-1">({s.market === 'coin' ? '코인' : '주식'})</span>
                  </span>
                  <span className="text-xs font-mono space-x-2">
                    <span className={s.win_rate_pct >= 50 ? 'text-green-400' : 'text-red-400'}>{s.win_rate_pct}%</span>
                    <span className="text-gray-500">{s.trades}건</span>
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* 시간대별 */}
          <div>
            <p className="text-xs text-gray-400 mb-1.5 font-semibold">시간대별 승률 (KST)</p>
            <div className="flex flex-wrap gap-1.5">
              {byHour.map(h => (
                <div key={h.hour_kst}
                  className={`rounded px-2 py-1 text-xs font-mono
                    ${h.win_rate_pct >= 55 ? 'bg-green-900/50 text-green-300'
                    : h.win_rate_pct >= 50 ? 'bg-yellow-900/50 text-yellow-300'
                    : 'bg-red-900/30 text-red-400'}`}>
                  {String(h.hour_kst).padStart(2,'0')}시 {h.win_rate_pct}% ({h.trades}건)
                </div>
              ))}
            </div>
          </div>

          {/* 종목별 (상위/하위) */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-400 mb-1.5 font-semibold">수익 상위 종목</p>
              {byTicker.slice(0, 8).map(t => (
                <div key={t.ticker} className="flex justify-between text-xs py-0.5">
                  <span className="font-mono text-gray-300">{t.ticker.replace('KRW-','')}</span>
                  <span className="space-x-2">
                    <span className={t.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {t.avg_pnl_pct > 0 ? '+' : ''}{t.avg_pnl_pct}%
                    </span>
                    <span className="text-gray-500">{t.win_rate_pct}% ({t.trades}건)</span>
                  </span>
                </div>
              ))}
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1.5 font-semibold">손실 하위 종목</p>
              {[...byTicker].sort((a,b) => a.avg_pnl_pct - b.avg_pnl_pct).slice(0, 8).map(t => (
                <div key={t.ticker} className="flex justify-between text-xs py-0.5">
                  <span className="font-mono text-gray-300">{t.ticker.replace('KRW-','')}</span>
                  <span className="space-x-2">
                    <span className={t.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {t.avg_pnl_pct > 0 ? '+' : ''}{t.avg_pnl_pct}%
                    </span>
                    <span className="text-gray-500">{t.win_rate_pct}% ({t.trades}건)</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


// ── 전체 포지션 한눈에 보기 ─────────────────────────────────────
function PositionsOverview({ agents }) {
  const holding = (agents ?? []).filter(a => Object.keys(a.positions ?? {}).length > 0)
  if (holding.length === 0) return null
  const totalPositions = holding.reduce((s, a) => s + Object.keys(a.positions).length, 0)

  return (
    <div className="bg-gray-800 rounded-xl border border-yellow-700/40 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="font-bold text-white text-sm">
          📍 현재 보유 포지션 &mdash; {holding.length}개 에이전트 &middot; {totalPositions}종목
        </span>
        <span className="text-xs text-gray-500">SSE 실시간 갱신</span>
      </div>
      <div className="space-y-2">
        {holding.map(agent => (
          <div key={agent.agent_id} className="flex items-start gap-3 text-xs">
            <span className="w-20 font-bold text-gray-300 pt-0.5 shrink-0">{agent.agent_id}</span>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(agent.positions).map(([ticker, pos]) => {
                const pnl = pos.unrealized_pnl_pct ?? 0
                const entMin = pos.entered_at ? pos.entered_at.slice(11, 16) : ''
                return (
                  <div key={ticker}
                    className={`flex items-center gap-1 px-2 py-0.5 rounded-full border font-mono text-xs
                      ${pnl >= 0.3 ? 'bg-green-900/50 border-green-700/60 text-green-300'
                      : pnl <= -0.3 ? 'bg-red-900/50 border-red-700/60 text-red-300'
                      : 'bg-gray-700/60 border-gray-600/60 text-gray-300'}`}>
                    <span className="font-bold">{ticker.replace('KRW-', '')}</span>
                    <span className={`ml-0.5 ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pnl >= 0 ? '+' : ''}{pnl.toFixed(1)}%
                    </span>
                    {entMin && <span className="text-gray-500 text-[10px] ml-0.5">{entMin}</span>}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── 리셋 버튼 ─────────────────────────────────────────────────────
function ResetButton() {
  const [busy,   setBusy]   = useState(false)
  const [status, setStatus] = useState(null)

  const reset = async () => {
    if (busy) return
    if (!window.confirm('⚠️ 모든 에이전트를 1,000,000원으로 리셋하고 재학습을 시작합니다.\n기존 거래 기록은 아카이브로 보존됩니다.\n\n계속하시겠습니까?')) return
    setBusy(true)
    setStatus(null)
    try {
      const r = await api.post('/agents/reset?retrain=true')
      setStatus({ ok: true, msg: r.message })
    } catch {
      setStatus({ ok: false, msg: '리셋 실패 — 서버 로그 확인' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex items-center gap-3">
      <button onClick={reset} disabled={busy}
        className="text-xs px-3 py-1.5 rounded-lg bg-red-900/70 hover:bg-red-800 disabled:opacity-40 text-red-200 font-medium transition-colors border border-red-700/60">
        {busy ? '리셋 중…' : '🔄 전체 리셋 + 재학습 (1M)'}
      </button>
      {status && (
        <span className={`text-xs ${status.ok ? 'text-green-400' : 'text-red-400'}`}>{status.msg}</span>
      )}
    </div>
  )
}

// ── 메인 ──────────────────────────────────────────────────────────
export default function AgentDashboard({ agents }) {
  const [selected,         setSelected]         = useState(null)
  const [selectedEnsemble, setSelectedEnsemble] = useState(null)
  const [stockNames,       setStockNames]       = useState({})

  useEffect(() => {
    api.get('/stock-names').then(setStockNames).catch(() => {})
  }, [])

  const ensembleAgents = agents.filter(a => a.agent_id.startsWith('ENSEMBLE_'))
  const coinAgents    = agents.filter(a => a.market === 'coin' && !a.agent_id.startsWith('ENSEMBLE_'))
  const stockAgents   = agents.filter(a => a.market === 'stock' && !a.agent_id.startsWith('ENSEMBLE_'))
  const selectedAgent = agents.find(a => a.agent_id === selected)
  const handleClick   = id => setSelected(selected === id ? null : id)

  const allSim       = [...coinAgents, ...stockAgents]
  const totalTrades  = allSim.reduce((s, a) => s + a.total_trades, 0)
  const activeAgents = allSim.filter(a => a.total_trades > 0)
  const bestTotal    = activeAgents.length
    ? (Math.max(...activeAgents.map(a => a.total_value ?? a.balance ?? INITIAL)) || INITIAL)
    : INITIAL
  const champions    = allSim.filter(a => a.is_champion)

  return (
    <div className="space-y-6">

      {/* 보유 종목 차트 */}
      <HoldingChart agents={agents} />

      {/* 전체 요약 헤더 */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-bold text-white text-lg">AI 에이전트 경쟁 현황</h2>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">5분마다 자동 갱신 · 매일 06:07 재학습</span>
            <ResetButton />
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">총 에이전트</p>
            <p className="text-2xl font-bold text-white">{agents.length}</p>
            <p className="text-xs text-gray-500">코인 10 + 주식 10 · 활성 {activeAgents.length}개</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">총 가상 거래</p>
            <p className="text-2xl font-bold text-white">{totalTrades}</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">최고 총평가액</p>
            <p className={`text-xl font-bold font-mono ${bestTotal > INITIAL ? 'text-green-400' : 'text-gray-400'}`}>
              {fmt(bestTotal)}원
            </p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-1">현재 수익률 1위</p>
            {champions.length > 0
              ? champions.map(c => (
                  <p key={c.agent_id} className="text-sm font-bold text-yellow-300">
                    {c.market === 'coin' ? '코인' : '주식'}: {label(c)}
                  </p>
                ))
              : <p className="text-sm text-gray-500">아직 거래 없음</p>
            }
          </div>
        </div>
      </div>

      {/* 전체 포지션 한눈에 보기 */}
      <PositionsOverview agents={agents ?? []} />

      {/* 앙상블 실매매 신호 추적 — 맨 위 */}
      {ensembleAgents.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-3 pb-2 border-b border-purple-800/40">
            <span className="text-lg font-bold text-purple-300">앙상블 실매매 신호 추적</span>
            <span className="text-xs bg-purple-900/60 text-purple-300 px-2 py-0.5 rounded">가상 10M</span>
            <span className="text-xs text-gray-500">실제 앙상블 신호를 가상 포트폴리오로 추적 — 클릭하면 상세 보기</span>
          </div>

          {/* 요약 카드 2개 — 클릭 시 상세 토글 */}
          <div className="grid grid-cols-2 gap-4">
            {ensembleAgents.map(ea => {
              const totalVal  = ea.total_value ?? ea.balance ?? INITIAL
              const pnlAmt    = totalVal - INITIAL
              const posEnt    = Object.entries(ea.positions ?? {})
              const isSelected = selectedEnsemble === ea.agent_id
              return (
                <div key={ea.agent_id}
                  onClick={() => setSelectedEnsemble(isSelected ? null : ea.agent_id)}
                  className={`bg-purple-950/30 border rounded-xl p-4 cursor-pointer transition-all space-y-3
                    ${isSelected
                      ? 'border-purple-500 ring-2 ring-purple-500/30'
                      : 'border-purple-800/40 hover:border-purple-600/60'}`}>
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-purple-200">{ea.agent_id}</span>
                    <span className="text-xs text-gray-400">{ea.market === 'coin' ? '🪙 코인' : '📈 주식'} 앙상블</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-center">
                    <div className="bg-purple-900/30 rounded-lg p-2">
                      <p className="text-xs text-gray-400">총평가액</p>
                      <p className={`font-mono font-bold text-sm ${pnlAmt >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {fmt(totalVal)}원
                      </p>
                      <p className={`text-xs font-mono ${pnlAmt >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {pnlAmt >= 0 ? '+' : ''}{fmt(pnlAmt)}원
                      </p>
                    </div>
                    <div className="bg-purple-900/30 rounded-lg p-2">
                      <p className="text-xs text-gray-400">수익률</p>
                      <ReturnBadge pct={ea.total_return_pct} size="lg" />
                    </div>
                    <div className="bg-purple-900/30 rounded-lg p-2">
                      <p className="text-xs text-gray-400">승률</p>
                      <WinBadge rate={ea.win_rate} wins={ea.win_trades} total={ea.total_trades} size="lg" />
                    </div>
                    <div className="bg-purple-900/30 rounded-lg p-2">
                      <p className="text-xs text-gray-400">보유</p>
                      <p className="font-bold text-purple-300 text-sm">{posEnt.length}종목</p>
                      {posEnt.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1 justify-center">
                          {posEnt.slice(0,3).map(([ticker, pos]) => {
                            const pnl = pos.unrealized_pnl_pct ?? 0
                            return (
                              <span key={ticker} className={`text-xs font-mono ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {ticker.replace('KRW-','')}
                              </span>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* 선택된 앙상블 에이전트 상세 — 카드 바로 아래 */}
          {selectedEnsemble && (() => {
            const ea = ensembleAgents.find(a => a.agent_id === selectedEnsemble)
            return ea ? <AgentDetail key={ea.agent_id} agent={ea} stockNames={stockNames} /> : null
          })()}
        </div>
      )}

      {/* 암호화폐 섹션 */}
      <div>
        <div className="flex items-center gap-3 mb-3 pb-2 border-b border-yellow-800/40">
          <span className="text-lg font-bold text-yellow-300">암호화폐 경쟁</span>
          <span className="text-xs bg-yellow-900/60 text-yellow-300 px-2 py-0.5 rounded">코인 24/7</span>
          <span className="text-xs text-gray-500">업비트 거래대금 상위 50개 · 5분봉 · AI01~AI10</span>
          <span className="text-xs bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded font-mono ml-auto">
            {coinAgents.reduce((s, a) => s + a.total_trades, 0)}회
          </span>
        </div>
        <MarketSection agents={coinAgents} market="coin" selected={selected} onSelect={handleClick} stockNames={stockNames} />
      </div>

      {/* 구분선 */}
      <div className="border-t border-gray-700" />

      {/* 주식 섹션 */}
      <div>
        <div className="flex items-center gap-3 mb-3 pb-2 border-b border-blue-800/40">
          <span className="text-lg font-bold text-blue-300">주식 경쟁</span>
          <span className="text-xs bg-blue-900/60 text-blue-300 px-2 py-0.5 rounded">장중 09:00~15:30</span>
          <span className="text-xs text-gray-500">KIS 거래량 상위 50개 · 5분봉 · AI11~AI20</span>
          <span className="text-xs bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded font-mono ml-auto">
            {stockAgents.reduce((s, a) => s + a.total_trades, 0)}회
          </span>
        </div>
        <MarketSection agents={stockAgents} market="stock" selected={selected} onSelect={handleClick} stockNames={stockNames} />
      </div>

      {/* 분석 패널 */}
      <AnalysisPanel />

    </div>
  )
}
