import { useState, useEffect } from 'react'
import { GenerateRequest } from '../types/api'

const PRELOADED_EXAMPLES = [
  {
    label: '🔄 Fibonacci (DP)',
    prompt: 'Write a Python function to compute the nth Fibonacci number using dynamic programming with O(1) space complexity. Include type hints and a full docstring.',
  },
  {
    label: '🌲 Binary Search Tree',
    prompt: 'Implement a Python class for a Binary Search Tree with insert, search, and in-order traversal methods. Include proper type hints and docstrings.',
  },
  {
    label: '🔗 REST API client',
    prompt: 'Write a Python class that wraps the requests library to build a robust REST API client with retry logic, timeout handling, and JSON response parsing.',
  },
  {
    label: '🔄 Async web scraper',
    prompt: 'Create an async Python function using aiohttp to scrape multiple URLs concurrently and return a dictionary mapping URL to page title.',
  },
  {
    label: '🧮 Matrix operations',
    prompt: 'Implement matrix multiplication in pure Python (no numpy) with proper validation, type hints, and O(n³) algorithm documentation.',
  },
]

interface PromptControlsProps {
  onGenerate: (req: GenerateRequest) => void;
  isGenerating: boolean;
}

export default function PromptControls({ onGenerate, isGenerating }: PromptControlsProps) {
  const [prompt, setPrompt]         = useState('')
  const [temperature, setTemp]      = useState(0.7)
  const [maxTokens, setMaxTokens]   = useState(2048)
  const [selectedEx, setSelectedEx] = useState('')
  const [history, setHistory]       = useState<string[]>([])

  useEffect(() => {
    const saved = localStorage.getItem('prompt_history')
    if (saved) {
      try {
        setHistory(JSON.parse(saved))
      } catch {
        // ignore
      }
    }
  }, [])

  const saveToHistory = (newPrompt: string) => {
    const updated = [newPrompt, ...history.filter(h => h !== newPrompt)].slice(0, 5)
    setHistory(updated)
    localStorage.setItem('prompt_history', JSON.stringify(updated))
  }

  const handleExample = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const idx = parseInt(e.target.value, 10)
    if (!isNaN(idx)) {
      setPrompt(PRELOADED_EXAMPLES[idx].prompt)
      setSelectedEx(e.target.value)
    } else {
      setSelectedEx('')
    }
  }

  const handleGenerate = () => {
    if (!prompt.trim()) return
    const currentPrompt = prompt.trim()
    saveToHistory(currentPrompt)
    onGenerate({ prompt: currentPrompt, temperature, max_tokens: maxTokens })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleGenerate()
    }
  }

  return (
    <div className="glass-card controls-panel">
      <div className="controls-grid">
        {/* Left: prompt area */}
        <div className="prompt-area">
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
            <span className="prompt-label">PROMPT</span>
            <span style={{ display:'flex', gap:8, alignItems:'center' }}>
              <select
                id="examples-dropdown"
                className="examples-select"
                style={{ width:'auto', fontSize:12 }}
                value={selectedEx}
                onChange={handleExample}
              >
                <option value="">⚡ Preloaded examples…</option>
                {PRELOADED_EXAMPLES.map((ex, i) => (
                  <option key={i} value={i}>{ex.label}</option>
                ))}
              </select>
              <span className="char-count">{prompt.length} / 4096</span>
            </span>
          </div>

          <textarea
            id="prompt-input"
            className="prompt-textarea"
            value={prompt}
            onChange={e => { setPrompt(e.target.value); setSelectedEx('') }}
            onKeyDown={handleKeyDown}
            placeholder="Describe a Python coding task… (⌘+Enter to generate)"
            rows={5}
            maxLength={4096}
          />
          
          {history.length > 0 && (
            <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>History:</span>
              {history.map((h, i) => (
                <button
                  key={i}
                  className="history-pill"
                  onClick={() => { setPrompt(h); setSelectedEx(''); }}
                  title={h}
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 12,
                    padding: '2px 8px',
                    fontSize: 11,
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    maxWidth: 150
                  }}
                >
                  {h}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right: sliders + button */}
        <div className="controls-right">
          <div className="slider-group">
            <div className="slider-row">
              <span className="slider-label">TEMPERATURE</span>
              <span className="slider-value">{temperature.toFixed(2)}</span>
            </div>
            <input
              id="temperature-slider"
              aria-label="Temperature"
              type="range" min="0.1" max="1.5" step="0.05"
              value={temperature}
              onChange={e => setTemp(parseFloat(e.target.value))}
            />
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:10, color:'var(--text-muted)' }}>
              <span>Precise</span><span>Creative</span>
            </div>
          </div>

          <div className="slider-group">
            <div className="slider-row">
              <span className="slider-label">MAX TOKENS</span>
              <span className="slider-value">{maxTokens}</span>
            </div>
            <input
              id="max-tokens-slider"
              aria-label="Maximum tokens"
              type="range" min="64" max="2048" step="64"
              value={maxTokens}
              onChange={e => setMaxTokens(parseInt(e.target.value, 10))}
            />
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:10, color:'var(--text-muted)' }}>
              <span>64</span><span>2048</span>
            </div>
          </div>

          <button
            id="generate-btn"
            className="generate-btn"
            onClick={handleGenerate}
            disabled={isGenerating || !prompt.trim()}
          >
            {isGenerating ? (
              <><span className="spinner" /><span>Generating…</span></>
            ) : (
              <><span>⚡</span><span>Generate</span></>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
