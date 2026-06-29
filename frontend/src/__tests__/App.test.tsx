import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import App from '../App'
import ErrorBoundary from '../components/ErrorBoundary'

describe('App Component', () => {
  it('renders the header and main panels', () => {
    render(
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    )
    // Check for main title
    expect(screen.getByText('StyleForge LLM')).toBeInTheDocument()
    // Check for the panels
    expect(screen.getByText('Base Model')).toBeInTheDocument()
    expect(screen.getByText('Fine-tuned Model')).toBeInTheDocument()
  })
})
