import { FormEvent, useState } from 'react'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type ApiSource = {
  source?: string | null
  section?: string | null
  page?: number | null
  chunk_id?: number | null
  preview?: string | null
}

type ChatReply = {
  answer: string
  raw_answer?: string | null
  status: string
  expanded_query?: string | null
  sources: ApiSource[]
  session_id: string
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        'Ask a question to validate an update in the rulebook. For example, `Validate this update: A new vehicle must have a newly manufactured chassis with significant changes in the Primary Structure compared to its predecessor.`.',
    },
  ])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem('curt-session-id') ?? crypto.randomUUID(),
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function sendMessage(content: string) {
    const trimmed = content.trim()
    if (!trimmed || loading) return

    setError(null)
    setLoading(true)

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: 'user', content: trimmed },
    ]

    setMessages(nextMessages)
    setInput('')

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: trimmed,
          session_id: sessionId,
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = (await response.json()) as ChatReply
      localStorage.setItem('curt-session-id', data.session_id)
      setSessionId(data.session_id)

      setMessages((current) => [
        ...current,
        { role: 'assistant', content: data.answer },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content:
            'I could not reach the backend. Check that FastAPI is running on port 8000.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void sendMessage(input)
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-visual">
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdjQXjmmOJYGpvcBrPxJxy53ljECUnM1yjRQ&s"
            alt="CURT logo"
          />
        </div>

        <div className="sidebar-copy-block">
          <h1>CURT Internal Chatbot</h1>
          <p className="sidebar-copy">
            Ask questions about care updates over the CURT knowledge base and the chatbot will verify them.
          </p>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="hero">
          <p className="eyebrow">Knowledge assistant</p>
          <h2>Ask questions over the CURT knowledge base</h2>
        </header>

        <section className="conversation" aria-live="polite">
          {messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`message ${message.role}`}
            >
              <span className="role">
                {message.role === 'user' ? 'You' : 'Assistant'}
              </span>
              <p>{message.content}</p>
            </article>
          ))}
        </section>

        {error && <p className="error">{error}</p>}

        <form className="composer" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Ask about the rulebook..."
            aria-label="Chat message"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App
