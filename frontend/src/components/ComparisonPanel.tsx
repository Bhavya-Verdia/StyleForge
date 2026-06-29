import { useState, useEffect, useRef } from 'react'
import useSSEStream, { SSEStreamResult } from '../hooks/useSSEStream'
import { GenerateRequest } from '../types/api'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Bot, Sparkles, Check, Copy, CheckCircle, AlertCircle } from 'lucide-react'
import { API_BASE } from '../config'

interface PanelProps {
  label: string;
  modelKey: 'base' | 'ft';
  stream: SSEStreamResult;
  onCopy: (text: string) => void;
  copied: boolean;
  theme: string;
}

function Panel({ label, modelKey, stream, onCopy, copied, theme }: PanelProps) {
  const outputRef = useRef<HTMLDivElement>(null)
  const [syntaxStatus, setSyntaxStatus] = useState<'idle' | 'checking' | 'valid' | 'invalid'>('idle')
  const [syntaxError, setSyntaxError] = useState<string>('')

  // Auto-scroll while streaming
  useEffect(() => {
    if (stream.isStreaming && outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [stream.text, stream.isStreaming])

  // Validate syntax when stream finishes
  useEffect(() => {
    if (!stream.isStreaming && stream.text && stream.text.length > 5) {
      setSyntaxStatus('checking')
      fetch(`${API_BASE}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: stream.text })
      })
      .then(res => res.json())
      .then(data => {
        if (data.valid) {
          setSyntaxStatus('valid')
        } else {
          setSyntaxStatus('invalid')
          setSyntaxError(`Line ${data.errorLine}: ${data.error}`)
        }
      })
      .catch(() => setSyntaxStatus('idle'))
    } else if (stream.isStreaming) {
      setSyntaxStatus('idle')
    }
  }, [stream.isStreaming, stream.text])

  const isBase = modelKey === 'base'

  return (
    <div className={`glass-card model-panel ${stream.isStreaming ? `generating-${modelKey}` : ''}`}>
      {/* Header */}
      <div className="panel-header">
        <div className="model-label">
          <div className={`model-dot ${isBase ? 'base' : 'ft'}`} />
          <div>
            <div className={`model-name ${isBase ? 'base' : 'ft'}`}>{label}</div>
            <div className="model-tag">
              {isBase ? 'Mistral-7B-Instruct-v0.3' : 'QLoRA · r=16 · NF4'}
            </div>
          </div>
        </div>

        <div className="panel-meta">
          {stream.tokensPerSec !== null && (
            <span className="token-speed">
              {stream.tokensPerSec} tok/s
            </span>
          )}
          {stream.tokenCount > 0 && (
            <span className="token-speed" style={{ color: 'var(--text-muted)' }}>
              {stream.tokenCount} tokens
            </span>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginLeft: 8 }}>
            {syntaxStatus === 'checking' && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Checking...</span>}
            {syntaxStatus === 'valid' && <span style={{ fontSize: 11, color: '#10b981', display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle size={14}/> Valid</span>}
            {syntaxStatus === 'invalid' && <span style={{ fontSize: 11, color: '#ef4444', display: 'flex', alignItems: 'center', gap: 4 }} title={syntaxError}><AlertCircle size={14}/> Error</span>}
            
            <button
              id={`copy-btn-${modelKey}`}
              className={`copy-btn${copied ? ' copied' : ''}`}
              aria-label="Copy code"
              onClick={() => onCopy(stream.text)}
              disabled={!stream.text}
              style={{ display: 'flex', alignItems: 'center', gap: 4 }}
            >
              {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy</>}
            </button>
          </div>
        </div>
      </div>

      {/* Output */}
      <div className="panel-output" ref={outputRef}>
        {!stream.text && !stream.isStreaming ? (
          <div className="output-placeholder">
            <div className="output-placeholder-icon">
              {isBase ? <Bot size={40} /> : <Sparkles size={40} />}
            </div>
            <div className="output-placeholder-text">
              {isBase ? 'Base model output will appear here' : 'Fine-tuned model output will appear here'}
            </div>
          </div>
        ) : (
          <div style={{ position: 'relative' }}>
            <SyntaxHighlighter
              language="python"
              style={theme === 'light' ? vs : vscDarkPlus}
              customStyle={{ background: 'transparent', padding: 0, margin: 0, fontSize: '14px' }}
              wrapLines={true}
            >
              {stream.text + (stream.isStreaming ? '▍' : '')}
            </SyntaxHighlighter>
          </div>
        )}
      </div>
    </div>
  )
}

interface ComparisonPanelProps {
  trigger: (GenerateRequest & { _ts: number }) | null;
  isGenerating: boolean;
  setIsGenerating: (generating: boolean) => void;
  theme: string;
}

export default function ComparisonPanel({ trigger, setIsGenerating, theme }: ComparisonPanelProps) {
  const [copiedBase, setCopiedBase] = useState(false)
  const [copiedFT,   setCopiedFT]   = useState(false)

  const base = useSSEStream('/generate/base')
  const ft   = useSSEStream('/generate/finetuned')

  // Fire both streams when trigger fires
  useEffect(() => {
    if (!trigger) return
    const body = {
      prompt:      trigger.prompt,
      max_tokens:  trigger.max_tokens,
      temperature: trigger.temperature,
    }
    base.start(body)
    ft.start(body)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trigger])

  // Track combined streaming state
  useEffect(() => {
    setIsGenerating(base.isStreaming || ft.isStreaming)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base.isStreaming, ft.isStreaming])

  const copyText = (text: string, setter: (val: boolean) => void) => {
    if (!text) return
    navigator.clipboard.writeText(text).then(() => {
      setter(true)
      setTimeout(() => setter(false), 2000)
    })
  }

  return (
    <div>
      <p className="section-title">Side-by-side comparison</p>
      <div className="comparison-grid">
        <Panel
          label="Base Model"
          modelKey="base"
          stream={base}
          onCopy={(t) => copyText(t, setCopiedBase)}
          copied={copiedBase}
          theme={theme}
        />
        <Panel
          label="Fine-tuned Model"
          modelKey="ft"
          stream={ft}
          onCopy={(t) => copyText(t, setCopiedFT)}
          copied={copiedFT}
          theme={theme}
        />
      </div>
    </div>
  )
}
