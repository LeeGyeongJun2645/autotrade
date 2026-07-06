import { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

const CAT_TABS = [
  { id: null,    label: '전체' },
  { id: 'coin',  label: '코인' },
  { id: 'stock', label: '주식' },
  { id: 'dart',  label: 'DART공시' },
]

const CAT_COLOR = {
  coin:  'text-yellow-400 bg-yellow-900/30 border-yellow-700/50',
  stock: 'text-blue-400 bg-blue-900/30 border-blue-700/50',
  dart:  'text-green-400 bg-green-900/30 border-green-700/50',
}

const CAT_DOT = {
  coin:  'bg-yellow-400',
  stock: 'bg-blue-400',
  dart:  'bg-green-400',
}

function timeAgo(published) {
  if (!published) return ''
  try {
    const d = new Date(published)
    if (isNaN(d)) return published.slice(0, 16)
    const diff = (Date.now() - d.getTime()) / 1000
    if (diff < 60)   return `${Math.floor(diff)}초 전`
    if (diff < 3600) return `${Math.floor(diff/60)}분 전`
    if (diff < 86400) return `${Math.floor(diff/3600)}시간 전`
    return `${Math.floor(diff/86400)}일 전`
  } catch {
    return published.slice(0, 16)
  }
}

export default function NewsBanner({ side = 'right', defaultCategory = null }) {
  const [items, setItems]   = useState([])
  const [cat, setCat]       = useState(defaultCategory)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(null)

  const load = useCallback(async () => {
    try {
      const url = cat ? `/news/feed?category=${cat}` : '/news/feed'
      const data = await api.get(url)
      setItems(Array.isArray(data) ? data : [])
      setLastUpdated(new Date())
    } catch {
      /* 무시 */
    } finally {
      setLoading(false)
    }
  }, [cat])

  useEffect(() => {
    setLoading(true)
    load()
    const id = setInterval(load, 3 * 60 * 1000) // 3분마다 자동 갱신
    return () => clearInterval(id)
  }, [load])

  const visible = cat ? items : items.slice(0, 60)

  return (
    <div className={`flex flex-col bg-gray-900 border border-gray-700 rounded-xl overflow-hidden h-full
      ${side === 'left' ? 'border-l-0 rounded-l-none' : 'border-r-0 rounded-r-none'}`}>

      {/* 헤더 */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-700 shrink-0 bg-gray-800/60">
        <span className="text-xs font-bold text-gray-200 tracking-wide">📰 뉴스·공시</span>
        {lastUpdated && (
          <span className="text-xs text-gray-500">{lastUpdated.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })} 갱신</span>
        )}
      </div>

      {/* 탭 */}
      <div className="flex border-b border-gray-700 shrink-0">
        {CAT_TABS.map(t => (
          <button
            key={String(t.id)}
            onClick={() => setCat(t.id)}
            className={`flex-1 py-1.5 text-xs font-medium transition-colors
              ${cat === t.id
                ? 'text-white bg-gray-700 border-b-2 border-indigo-500'
                : 'text-gray-500 hover:text-gray-300'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* 뉴스 목록 */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-20">
            <span className="text-xs text-gray-500 animate-pulse">불러오는 중...</span>
          </div>
        ) : visible.length === 0 ? (
          <div className="text-center text-gray-600 text-xs py-8">뉴스 없음</div>
        ) : (
          <ul className="divide-y divide-gray-800">
            {visible.map((item, i) => (
              <li key={i} className="px-3 py-2.5 hover:bg-gray-800/50 transition-colors group">
                <a
                  href={item.url || '#'}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block"
                >
                  {/* 출처 배지 + 시간 */}
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${CAT_DOT[item.category] ?? 'bg-gray-500'}`} />
                    <span className="text-gray-500 text-xs truncate">{item.source}</span>
                    <span className="text-gray-600 text-xs ml-auto shrink-0">{timeAgo(item.published)}</span>
                  </div>
                  {/* 제목 */}
                  <p className="text-xs leading-relaxed text-gray-300 group-hover:text-white transition-colors line-clamp-2">
                    {item.title}
                  </p>
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* 하단: 카테고리별 건수 */}
      <div className="flex border-t border-gray-700 shrink-0 bg-gray-800/40">
        {['coin','stock','dart'].map(c => {
          const cnt = items.filter(x => x.category === c).length
          return (
            <div key={c} className="flex-1 text-center py-1.5">
              <span className={`text-xs px-1.5 py-0.5 rounded border ${CAT_COLOR[c]}`}>
                {c === 'coin' ? '코인' : c === 'stock' ? '주식' : 'DART'} {cnt}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
