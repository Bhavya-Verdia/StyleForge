import { useState, useCallback, useEffect } from 'react'
import './index.css'
import PromptControls  from './components/PromptControls'
import ComparisonPanel from './components/ComparisonPanel'
import MetricsChart    from './components/MetricsChart'
import LossCurve       from './components/LossCurve'
import { Code, Terminal, Sun, Moon, Database, Wrench, Zap, Layers, Target } from 'lucide-react'

import { API_BASE } from './config'

import { GenerateRequest } from './types/api'

export default function App() {
  const [trigger, setTrigger]         = useState<(GenerateRequest & { _ts: number }) | null>(null)
  const [isGenerating, setGenerating] = useState<boolean>(false)
  const [adapters, setAdapters]       = useState<string[]>([])
  const [activeAdapter, setActiveAdapter] = useState<string>('')
  
  // Theme toggle
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    fetch(`${API_BASE}/adapters`)
      .then(res => res.json())
      .then(data => {
        setAdapters(data.adapters)
        if (data.adapters.length > 0) setActiveAdapter(data.adapters[0])
      })
      .catch(console.error)
  }, [])

  const handleAdapterChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const adapterName = e.target.value
    setActiveAdapter(adapterName)
    try {
      await fetch(`${API_BASE}/adapter/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ adapter_name: adapterName })
      })
    } catch (err) {
      console.error("Failed to load adapter", err)
    }
  }

  const handleGenerate = useCallback((params: GenerateRequest) => {
    // Wrap in new object to force ComparisonPanel useEffect even with same params
    setTrigger({ ...params, _ts: Date.now() })
  }, [])

  return (
    <>
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="header">
        <div className="container header-inner">
          <a href="/" className="logo">
            <div className="logo-icon"><Code size={24} /></div>
            <div>
              <div className="logo-text">StyleForge LLM</div>
              <div className="logo-subtitle">Fine-Tuning Comparison Platform</div>
            </div>
          </a>

          <div style={{ display:'flex', gap:10, alignItems:'center' }}>
            <div className="header-badge">
              <div className="dot" />
              <span>Mistral-7B Base</span>
            </div>
            
            {adapters.length > 0 && (
              <select 
                className="adapter-selector" 
                value={activeAdapter}
                onChange={handleAdapterChange}
                title="Select active adapter"
              >
                {adapters.map(a => (
                  <option key={a} value={a}>Adapter: {a}</option>
                ))}
              </select>
            )}

            <div className="header-badge" style={{
              background:'rgba(6,182,212,0.08)', borderColor:'rgba(6,182,212,0.2)', color:'#67e8f9'
            }}>
              <Terminal size={14} />
              <span>Coding Domain</span>
            </div>
            
            <button 
              onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                padding: '6px 10px',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              title="Toggle Theme"
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ──────────────────────────────────────────── */}
      <main>
        <div className="container main-content">

          {/* Hero summary strip */}
          <div style={{
            padding: '18px 24px',
            background: 'rgba(99,102,241,0.06)',
            border: '1px solid rgba(99,102,241,0.15)',
            borderRadius: 'var(--radius-lg)',
            display: 'flex',
            gap: 32,
            flexWrap: 'wrap',
          }}>
            {[
              { icon: <Database size={18} />, label:'Dataset',   value:'python_code_instructions_18k' },
              { icon: <Wrench size={18} />, label:'LoRA rank',  value:'r=16 · α=32' },
              { icon: <Zap size={18} />, label:'Quant',      value:'4-bit NF4 (BnB)' },
              { icon: <Layers size={18} />, label:'Epochs',     value:'3 · cosine LR' },
              { icon: <Target size={18} />, label:'Targets',    value:'q k v o gate up down' },
            ].map(({ icon, label, value }) => (
              <div key={label} style={{ display:'flex', alignItems:'center', gap:10 }}>
                <span style={{ display:'flex', alignItems:'center', color: 'var(--text-secondary)' }}>{icon}</span>
                <div>
                  <div style={{ fontSize:10, color:'var(--text-muted)', textTransform:'uppercase', letterSpacing:'0.08em', fontWeight:600 }}>{label}</div>
                  <div style={{ fontSize:13, fontWeight:600, color:'var(--text-secondary)', fontFamily:'JetBrains Mono, monospace' }}>{value}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Prompt controls */}
          <div>
            <p className="section-title">Prompt Configuration</p>
            <PromptControls onGenerate={handleGenerate} isGenerating={isGenerating} />
          </div>

          <ComparisonPanel
            trigger={trigger}
            isGenerating={isGenerating}
            setIsGenerating={setGenerating}
            theme={theme}
          />

          {/* Metrics + Loss curve */}
          <div>
            <p className="section-title">Training & Evaluation Analytics</p>
            <div className="metrics-section">
              <MetricsChart />
              <LossCurve />
            </div>
          </div>

        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer className="footer">
        <div className="container">
          StyleForge LLM &nbsp;·&nbsp;
          Mistral-7B-Instruct-v0.3 &nbsp;·&nbsp;
          QLoRA (PEFT + bitsandbytes) &nbsp;·&nbsp;
          SFTTrainer (trl) &nbsp;·&nbsp;
          FastAPI + SSE &nbsp;·&nbsp;
          React + recharts
        </div>
      </footer>
    </>
  )
}
