import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, LabelList,
  ReferenceLine
} from 'recharts'
import { AlertTriangle } from 'lucide-react'

import { API_BASE } from '../config'
import { MetricsResponse } from '../types/api'

const METRIC_KEYS = ['rouge1', 'rouge2', 'rougeL'] as const
const METRIC_LABELS: Record<string, string> = { rouge1: 'ROUGE-1', rouge2: 'ROUGE-2', rougeL: 'ROUGE-L' }

// Custom tooltip
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(7,7,26,0.95)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: 10,
      padding: '10px 14px',
      fontSize: 13,
    }}>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 600 }}>{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color, fontFamily: 'JetBrains Mono, monospace' }}>
          {p.name}: <strong>{((p.value as number) * 100).toFixed(1)}%</strong>
        </p>
      ))}
    </div>
  )
}

export default function MetricsChart() {
  const [data, setData]       = useState<MetricsResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API_BASE}/metrics`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div className="glass-card chart-panel">
      <div className="chart-title">Evaluation Metrics</div>
      <div className="chart-subtitle">ROUGE scores across held-out prompts</div>
      {[1, 2, 3].map(i => (
        <div key={i} className="shimmer" style={{ height: 30, marginBottom: 12, borderRadius: 6 }} />
      ))}
    </div>
  )

  if (error || !data?.results) return (
    <div className="glass-card chart-panel">
      <div className="chart-title">Evaluation Metrics</div>
      <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
        <AlertTriangle size={16} /> Could not load metrics. Start the server: <code>USE_MOCK=true uvicorn main:app</code>
      </p>
    </div>
  )

  const base: any = data.results.find(r => r.model === 'base') || {}
  const ft: any   = data.results.find(r => r.model === 'finetuned') || {}

  // Chart data: one bar group per ROUGE metric
  const chartData = METRIC_KEYS.map((k: string) => ({
    metric: METRIC_LABELS[k],
    Base:   parseFloat(((base[k] as number) || 0).toFixed(4)),
    'Fine-tuned': parseFloat(((ft[k] as number) || 0).toFixed(4)),
  }))

  const isNote = !!data.note

  return (
    <div className="glass-card chart-panel">
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
        <div>
          <div className="chart-title">Evaluation Metrics</div>
          <div className="chart-subtitle">
            ROUGE F1 scores · {data.eval_samples || 50} held-out prompts
            {isNote && <span style={{ color:'#f59e0b', marginLeft:8 }}>• Sample data</span>}
          </div>
        </div>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-dot" style={{ background:'var(--accent-base)' }} />
            <span>Base</span>
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background:'var(--accent-ft)' }} />
            <span>Fine-tuned</span>
          </div>
        </div>
      </div>

      {/* Stat cards: perplexity + avg length */}
      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-card-label">Perplexity ↓</div>
          <div className="stat-card-values">
            <span className="stat-val base">{base.perplexity ?? '—'}</span>
            <span className="stat-sep">vs</span>
            <span className="stat-val ft">{ft.perplexity ?? '—'}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-card-label">Avg Length (words) ↑</div>
          <div className="stat-card-values">
            <span className="stat-val base">{base.avg_response_length ?? '—'}</span>
            <span className="stat-sep">vs</span>
            <span className="stat-val ft">{ft.avg_response_length ?? '—'}</span>
          </div>
        </div>
      </div>

      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} barCategoryGap="30%" barGap={4}
          margin={{ top: 20, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis
            dataKey="metric"
            tick={{ fill: 'var(--text-secondary)', fontSize: 12, fontFamily: 'Inter' }}
            axisLine={false} tickLine={false}
          />
          <YAxis
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={false} tickLine={false} domain={[0, 0.7]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          
          <ReferenceLine y={0.5} stroke="rgba(16, 185, 129, 0.4)" strokeDasharray="3 3" label={{ position: 'insideTopRight', value: '50% Target', fill: 'rgba(16, 185, 129, 0.8)', fontSize: 10 }} />

          <Bar dataKey="Base" fill="#06b6d4" radius={[4,4,0,0]}>
            <LabelList
              dataKey="Base"
              position="top"
              formatter={(v: any) => `${(Number(v)*100).toFixed(0)}%`}
              style={{ fill:'var(--accent-base)', fontSize:11, fontFamily:'JetBrains Mono,monospace', fontWeight:600 }}
            />
          </Bar>
          <Bar dataKey="Fine-tuned" fill="#f59e0b" radius={[4,4,0,0]}>
            <LabelList
              dataKey="Fine-tuned"
              position="top"
              formatter={(v: any) => `${(Number(v)*100).toFixed(0)}%`}
              style={{ fill:'var(--accent-ft)', fontSize:11, fontFamily:'JetBrains Mono,monospace', fontWeight:600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
