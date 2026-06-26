import { useState } from 'react'
import { api } from '../api.js'

const INTERVAL_LABEL = { 1: '1분봉', 5: '5분봉', 15: '15분봉' }
const MARKET_BADGE = {
  coin:  { label: '코인 24/7', cls: 'bg-yellow-900/60 text-yellow-300' },
  stock: { label: '주식 장중', cls: 'bg-blue-900/60 text-blue-300' },
}
const FEAT_COLOR = {
  all:      'bg-indigo-900/60 text-indigo-300',
  momentum: 'bg-green-900/60 text-green-300',
  trend:    'bg-blue-900/60 text-blue-300',
  volume:   'bg-orange-900/60 text-orange-300',
}

function ReturnBadge({ pct }) {
  const pos = pct >= 0
  return (
    <span className={`font-mono text-sm font-semibold ${pos ? 'text-green-400' : 'text-red-400'}`}>
      {pos ? '+' : ''}{pct.toFixed(2)}%
    </span>
  )
}

function WinRate({ rate }) {
  const color = rate >= 60 ? 'text-green-400' : rate >= 50 ? 'text-yellow-400' : 'text-red-400'
  return <span className={`font-mono text-sm font-bold ${color}`}>{rate.toFixed(1)}%</span>
}

function TradeRow({ t }) {
  const isBuy = t.action === 'BUY'
  const pr = t.profit_rate != null ? t.profit_rate * 100 : null
  return (
    <tr className="border-b border-gray-700/40 text-xs hover:bg-gray-700/20">
      <td className="px-3 py-1.5 font-mono text-gray-300">{t.ticker?.replace('KRW-', '')}</td>
      <td className="px-3 py-1.5">
        <span className={`font-semibold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>
          {isBuy ? '매수' : '매도'}
        </span>
      </td>
      <td className="px-3 py-1.5 font-mono text-white">{Number(t.price).toLocaleString('ko-KR')}원</td>
      {!isBuy && (
        <td className={`px-3 py-1.5 font-mono font-semibold ${pr >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {pr != null ? `${pr >= 0 ? '+' : ''}${pr.toFixed(2)}%` : '—'}
        </td>
      )}
      {isBuy && <td className="px-3 py-1.5 text-gray-500">—</td>}
      <td className="px-3 py-1.5 text-gray-500">{t.traded_at?.slice(11, 16)}</td>
    </tr>
  )
}

function AgentCard({ agent, onClick, selected }) {
  const hasPos = Object.keys(agent.positions ?? {}).length > 0
  return (
    <div
      onClick={onClick}
      className={`bg-gray-800 rounded-xl p-4 border cursor-pointer transition-all ${
        agent.is_champion
          ? 'border-yellow-500 shadow-lg shadow-yellow-900/30'
          : selected
          ? 'border-indigo-500'
          : 'border-gray-700 hover:border-gray-500'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-bold text-white text-sm">{agent.agent_id}</span>
          {agent.is_champion && (
            <span className="text-xs bg-yellow-700/60 text-yellow-300 px-1.5 py-0.5 rounded font-semibold">챔피언</span>
          )}
          <span className={`text-xs px-1.5 py-0.5 rounded ${MARKET_BADGE[agent.market]?.cls ?? ''}`}>
            {MARKET_BADGE[agent.market]?.label ?? agent.market}
          </span>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded ${FEAT_COLOR[agent.feature_set] ?? ''}`}>
          {agent.feature_set}
        </span>
      </div>

      <div className="text-xs text-gray-400 mb-3">
        {INTERVAL_LABEL[agent.interval_min]} · 레이블 +{(agent.label_threshold * 100).toFixed(1)}% · 임계 {(agent.buy_threshold * 100).toFixed(0)}%
      </div>

      <div className="grid grid-cols-2 gap-2 mb-2">
        <div>
          <p className="text-xs text-gray-500">승률</p>
          <WinRate rate={agent.win_rate} />
          <span className="text-xs text-gray-600 ml-1">({agent.win_trades}/{agent.total_trades})</span>
        </div>
        <div>
          <p className="text-xs text-gray-500">수익률</p>
          <ReturnBadge pct={agent.total_return_pct} />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">
          잔액 {Number(agent.balance).toLocaleString('ko-KR')}원
        </span>
        {hasPos && (
          <span className="text-xs bg-blue-900/50 text-blue-300 px-1.5 py-0.5 rounded">
            포지션 {Object.keys(agent.positions).length}개
          </span>
        )}
      </div>
    </div>
  )
}

function AgentDetail({ agent }) {
  const [dbTrades, setDbTrades] = useState(null)
  const [loading, setLoading] = useState(false)

  const loadTrades = async () => {
    if (dbTrades) return
    setLoading(true)
    try {
      const data = await api.get(`/agents/${agent.agent_id}/trades?limit=50`)
      setDbTrades(data)
    } catch {
      setDbTrades([])
    }
    setLoading(false)
  }

  const trades = dbTrades ?? agent.recent_trades ?? []

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="font-bold text-white text-lg">{agent.agent_id}</h3>
          {agent.is_champion && (
            <span className="text-sm bg-yellow-700/60 text-yellow-300 px-2 py-0.5 rounded font-semibold">🏆 챔피언</span>
          )}
          <span className={`text-xs px-2 py-0.5 rounded ${FEAT_COLOR[agent.feature_set] ?? ''}`}>
            {agent.feature_set}
          </span>
        </div>
        <button
          onClick={loadTrades}
          className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-1.5 rounded transition-colors"
        >
          {loading ? '로딩 중...' : 'DB 거래 기록 더 보기'}
        </button>
      </div>

      {/* 설정 정보 */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          ['인터벌', INTERVAL_LABEL[agent.interval_min]],
          ['레이블 기준', `+${(agent.label_threshold * 100).toFixed(1)}%`],
          ['매수 임계값', `${(agent.buy_threshold * 100).toFixed(0)}%`],
          ['피처 세트', agent.feature_set],
        ].map(([k, v]) => (
          <div key={k} className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">{k}</p>
            <p className="font-mono text-sm text-white">{v}</p>
          </div>
        ))}
      </div>

      {/* 성과 지표 */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          ['승률', <WinRate rate={agent.win_rate} />],
          ['수익률', <ReturnBadge pct={agent.total_return_pct} />],
          ['총 거래', `${agent.total_trades}회`],
          ['현재 잔액', `${Number(agent.balance).toLocaleString('ko-KR')}원`],
        ].map(([k, v]) => (
          <div key={k} className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-1">{k}</p>
            <div className="font-mono text-sm">{v}</div>
          </div>
        ))}
      </div>

      {/* 현재 포지션 */}
      {Object.keys(agent.positions ?? {}).length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-400 mb-2">현재 보유 포지션</p>
          <div className="flex gap-2 flex-wrap">
            {Object.entries(agent.positions).map(([ticker, pos]) => (
              <div key={ticker} className="bg-blue-900/30 border border-blue-700/50 rounded-lg px-3 py-2 text-xs">
                <span className="font-mono text-blue-300 font-semibold">{ticker.replace('KRW-', '')}</span>
                <span className="text-gray-400 ml-2">매수가 {Number(pos.entry_price).toLocaleString('ko-KR')}원</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 거래 기록 */}
      <div>
        <p className="text-xs text-gray-400 mb-2">거래 기록 ({trades.length}건)</p>
        {trades.length === 0 ? (
          <p className="text-gray-600 text-sm text-center py-4">거래 없음 (5분 후 첫 실행)</p>
        ) : (
          <div className="overflow-x-auto max-h-60 overflow-y-auto">
            <table className="w-full">
              <thead>
                <tr className="text-gray-400 text-xs border-b border-gray-700">
                  {['종목', '구분', '가격', '수익률', '시각'].map(h => (
                    <th key={h} className="text-left px-3 py-1.5 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => <TradeRow key={i} t={t} />)}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default function AgentDashboard({ agents }) {
  const [selected, setSelected] = useState(null)

  const sorted = [...agents].sort((a, b) => b.win_rate - a.win_rate)
  const champion = sorted.find(a => a.is_champion)
  const selectedAgent = agents.find(a => a.agent_id === selected)

  return (
    <div className="space-y-6">
      {/* 헤더 요약 */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-bold text-white">AI 에이전트 경쟁 현황</h2>
          <span className="text-xs text-gray-500">5분마다 자동 갱신 · 매일 자정 챔피언 선정</span>
        </div>
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">총 에이전트</p>
            <p className="text-2xl font-bold text-white">{agents.length}</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">총 가상 거래</p>
            <p className="text-2xl font-bold text-white">{agents.reduce((s, a) => s + a.total_trades, 0)}</p>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-3">
            <p className="text-xs text-gray-400">최고 승률</p>
            <p className="text-2xl font-bold text-green-400">
              {agents.length ? Math.max(...agents.map(a => a.win_rate)).toFixed(1) : '0.0'}%
            </p>
          </div>
          <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-3">
            <p className="text-xs text-yellow-400">현재 챔피언</p>
            <p className="text-xl font-bold text-yellow-300">{champion?.agent_id ?? '미선정'}</p>
            {champion && <p className="text-xs text-yellow-500">{INTERVAL_LABEL[champion.interval_min]} / {champion.feature_set}</p>}
          </div>
        </div>
      </div>

      {/* 선택된 에이전트 상세 */}
      {selectedAgent && (
        <AgentDetail agent={selectedAgent} />
      )}

      {/* 에이전트 카드 그리드 */}
      <div>
        <p className="text-xs text-gray-500 mb-3">카드를 클릭하면 상세 정보를 볼 수 있습니다</p>
        {agents.length === 0 ? (
          <div className="text-center py-16 text-gray-500">
            <p className="text-3xl mb-3">🤖</p>
            <p className="text-sm">에이전트 초기화 중... 서버 시작 후 5분 후 첫 실행</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {sorted.map(agent => (
              <AgentCard
                key={agent.agent_id}
                agent={agent}
                selected={selected === agent.agent_id}
                onClick={() => setSelected(selected === agent.agent_id ? null : agent.agent_id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
