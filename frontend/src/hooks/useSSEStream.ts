import { useState, useRef } from 'react'
import { API_BASE } from '../config'

export interface SSEStreamResult {
  text: string;
  isStreaming: boolean;
  tokenCount: number;
  tokensPerSec: number | null;
  start: (body: any) => Promise<void>;
  stop: () => void;
}

export default function useSSEStream(endpoint: string): SSEStreamResult {
  const [text, setText]           = useState('')
  const [isStreaming, setStream]  = useState(false)
  const [tokenCount, setCount]    = useState(0)
  const [tokensPerSec, setTPS]    = useState<number | null>(null)
  
  const abortRef                  = useRef<AbortController | null>(null)
  const startTimeRef              = useRef<number | null>(null)
  const countRef                  = useRef<number>(0)

  const start = async (body: any) => {
    // Cancel any existing stream
    if (abortRef.current) abortRef.current.abort()

    setText('')
    setStream(true)
    setCount(0)
    setTPS(null)
    countRef.current   = 0
    startTimeRef.current = performance.now()

    const controller = new AbortController()
    abortRef.current  = controller

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal:  controller.signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        setText(`⚠ Error ${res.status}: ${err.detail || res.statusText}`)
        setStream(false)
        return
      }

      if (!res.body) {
        setText(`⚠ Error: Response body is null`)
        setStream(false)
        return
      }

      const reader   = res.body.getReader()
      const decoder  = new TextDecoder()
      let   buffer   = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // keep partial line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.done) { setStream(false); return }
            if (data.token) {
              setText(prev => prev + data.token)
              countRef.current += 1
              setCount(countRef.current)
              const elapsed = (performance.now() - (startTimeRef.current || 0)) / 1000
              if (elapsed > 0.5) {
                setTPS(Math.round(countRef.current / elapsed))
              }
            }
          } catch {/* skip malformed lines */}
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setText(prev => prev + `\n\n⚠ Connection error: ${err.message}`)
      }
    } finally {
      setStream(false)
    }
  }

  const stop = () => {
    if (abortRef.current) abortRef.current.abort()
    setStream(false)
  }

  return { text, isStreaming, tokenCount, tokensPerSec, start, stop }
}
