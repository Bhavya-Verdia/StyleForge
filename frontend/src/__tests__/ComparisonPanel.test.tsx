import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ComparisonPanel from '../components/ComparisonPanel'

// Mock the hook
vi.mock('../hooks/useSSEStream', () => ({
  default: () => ({
    text: 'def test(): pass',
    isStreaming: false,
    tokenCount: 4,
    tokensPerSec: 10,
    start: vi.fn(),
    stop: vi.fn()
  })
}))

describe('ComparisonPanel Component', () => {
  it('renders correctly with mocked streams', () => {
    const mockSetIsGenerating = vi.fn()
    
    render(
      <ComparisonPanel 
        trigger={null}
        isGenerating={false}
        setIsGenerating={mockSetIsGenerating}
      />
    )
    
    expect(screen.getByText('Base Model')).toBeInTheDocument()
    expect(screen.getByText('Fine-tuned Model')).toBeInTheDocument()
    // Since both streams return 'def test(): pass', it will be in the document twice
    expect(screen.getAllByText('def test(): pass').length).toBe(2)
    expect(mockSetIsGenerating).toHaveBeenCalledWith(false)
  })

  it('calls start when trigger changes', () => {
    const mockSetIsGenerating = vi.fn()
    const trigger = { prompt: "hello", max_tokens: 10, temperature: 0.7, _ts: 123 }
    
    render(
      <ComparisonPanel 
        trigger={trigger}
        isGenerating={false}
        setIsGenerating={mockSetIsGenerating}
      />
    )

    expect(screen.getAllByText('def test(): pass').length).toBe(2)
  })
})
