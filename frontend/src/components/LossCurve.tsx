import { useEffect, useState } from 'react'
import {
  XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
  Area, AreaChart
} from 'recharts'
import { AlertTriangle } from 'lucide-react'

import { API_BASE } from '../config'
import { LossData, LossDataResponse } from '../types/api'

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  const d = payload[0]
  const dataPayload = d.payload as LossData
  return (
    <div style={{
      background: 'rgba(7,7,26,0.95)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: 10,
      padding: '10px 14px',
      fontSize: 13,
    }}>
      <p style={{ color: 'var(--text-muted)', marginBottom: 4, fontSize: 11 }}>
        Step {label} {dataPayload.epoch ? `· Epoch ${dataPayload.epoch.toFixed(1)}` : ''}
      </p>
      <p style={{ color: '#a78bfa', fontFamily: 'JetBrains Mono, monospace', fontWeight: 700 }}>
        Loss: {d.value?.toFixed(4)}
      </p>
      {dataPayload.learning_rate && (
        <p style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          LR: {parseFloat(dataPayload.learning_rate as string).toExponential(2)}
        </p>
      )}
    </div>
  )
}

export default function LossCurve() {
  const [data, setData]       = useState<LossDataResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API_BASE}/training-loss`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div className="glass-card chart-panel">
      <div className="chart-title">Training Loss Curve</div>
      <div className="chart-subtitle">Loss per step over training</div>
      <div className="shimmer" style={{ height: 200, borderRadius: 8, marginTop: 12 }} />
    </div>
  )

  if (error || !data?.data) return (
    <div className="glass-card chart-panel">
      <div className="chart-title">Training Loss Curve</div>
      <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
        <AlertTriangle size={16} /> Could not load loss data. Start the server: <code>USE_MOCK=true uvicorn main:app</code>
      </p>
    </div>
  )

  const rows    = data.data.filter(r => r.loss !== null) as (LossData & { loss: number })[]
  if (rows.length === 0) return null
  
  const minLoss = Math.min(...rows.map(r => r.loss))
  const isNote  = !!data.note

  // Find steps where a new epoch begins (integer epochs)
  const epochBoundaries: number[] = [];
  let currentEpoch = 0;
  for (const row of rows) {
    if (row.epoch && Math.floor(row.epoch) > currentEpoch) {
      currentEpoch = Math.floor(row.epoch);
      epochBoundaries.push(row.step);
    }
  }

  return (
    <div className="glass-card chart-panel">
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:4 }}>
        <div>
          <div className="chart-title">Training Loss Curve</div>
          <div className="chart-subtitle">
            Cross-entropy loss per step · QLoRA fine-tuning
            {isNote && <span style={{ color:'#f59e0b', marginLeft:8 }}>• Simulated data</span>}
          </div>
        </div>
        <div style={{
          fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--text-muted)', textAlign: 'right',
        }}>
          <div>Min loss</div>
          <div style={{ color:'#a78bfa', fontWeight:700, fontSize:15 }}>
            {minLoss.toFixed(4)}
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={rows} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#8b5cf6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="step"
            tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={false} tickLine={false}
            label={{ value: 'Step', position: 'insideBottomRight', offset: -4, fill: 'var(--text-muted)', fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={false} tickLine={false}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          
          <ReferenceLine
            y={minLoss} stroke="rgba(139,92,246,0.3)"
            strokeDasharray="4 4"
            label={{ value: 'Best', position: 'right', fill: '#a78bfa', fontSize: 10 }}
          />
          
          {epochBoundaries.map(step => (
            <ReferenceLine
              key={step}
              x={step}
              stroke="rgba(255,255,255,0.1)"
              strokeDasharray="3 3"
              label={{ value: 'Epoch', position: 'insideTopLeft', fill: 'var(--text-muted)', fontSize: 10 }}
            />
          ))}

          <Area
            type="monotone"
            dataKey="loss"
            stroke="#8b5cf6"
            strokeWidth={2.5}
            fill="url(#lossGrad)"
            dot={false}
            activeDot={{ r: 5, fill: '#a78bfa', stroke: '#07071a', strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
