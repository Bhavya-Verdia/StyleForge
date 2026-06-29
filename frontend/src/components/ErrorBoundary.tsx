import { Component, ReactNode } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: any;
  errorInfo: any;
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error }
  }

  componentDidCatch(error: any, errorInfo: any) {
    this.setState({ errorInfo })
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg-primary, #07071a)',
            color: 'var(--text-primary, #f1f5f9)',
            fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
            padding: '24px',
          }}
        >
          <div
            style={{
              maxWidth: 520,
              textAlign: 'center',
              padding: '48px 40px',
              background: 'rgba(255, 255, 255, 0.035)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: 20,
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
            }}
          >
            <div style={{ fontSize: 48, marginBottom: 16 }}>⚠️</div>
            <h1
              style={{
                fontSize: 22,
                fontWeight: 700,
                marginBottom: 12,
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Something went wrong
            </h1>
            <p
              style={{
                fontSize: 14,
                color: '#94a3b8',
                lineHeight: 1.6,
                marginBottom: 24,
              }}
            >
              An unexpected error occurred while rendering the application.
              Please try reloading the page.
            </p>

            {this.state.error && (
              <pre
                style={{
                  textAlign: 'left',
                  fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: '#f87171',
                  background: 'rgba(248, 113, 113, 0.08)',
                  border: '1px solid rgba(248, 113, 113, 0.15)',
                  borderRadius: 10,
                  padding: '14px 16px',
                  marginBottom: 24,
                  overflow: 'auto',
                  maxHeight: 160,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {this.state.error.toString()}
              </pre>
            )}

            <button
              onClick={this.handleReload}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '12px 28px',
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa)',
                border: 'none',
                borderRadius: 12,
                color: 'white',
                fontSize: 14,
                fontWeight: 700,
                cursor: 'pointer',
                boxShadow: '0 4px 20px rgba(99,102,241,0.4)',
                transition: 'transform 0.15s, box-shadow 0.15s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.boxShadow = '0 8px 28px rgba(99,102,241,0.5)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = '0 4px 20px rgba(99,102,241,0.4)'
              }}
            >
              🔄 Reload Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
