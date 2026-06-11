import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ChatSocket } from '../chatSocket'
import type { ChatSocketStatus } from '../chatSocket'
import type { ChatItem, PermissionRequest, ServerFrame, ToolCallState } from '../types'
import { ShieldIcon } from './icons'

function ToolCard({ tc }: { tc: ToolCallState }) {
  const statusClass =
    tc.status === 'completed' ? 'ok' : tc.status === 'failed' ? 'fail' : 'busy'
  return (
    <div className="tool-card">
      <div className="tool-card-head">
        <span className={`status-dot ${statusClass}`} />
        <span className="tool-title">{tc.title}</span>
        <span className="tool-status">{tc.status}</span>
      </div>
      {tc.rawInput != null && (
        <details>
          <summary>input</summary>
          <pre>{JSON.stringify(tc.rawInput, null, 2)}</pre>
        </details>
      )}
      {tc.output && (
        <details>
          <summary>output</summary>
          <pre>{tc.output}</pre>
        </details>
      )}
    </div>
  )
}

function PermissionCard({
  perm,
  onDecide,
}: {
  perm: PermissionRequest
  onDecide: (optionId: string | null) => void
}) {
  return (
    <div className="approval-box">
      <div className="approval-header">
        <ShieldIcon size={13} strokeWidth={2.5} />
        Action requires your approval
      </div>
      <div className="approval-title">{perm.toolCall.title ?? 'tool call'}</div>
      {perm.explanation && <div className="approval-explanation">{perm.explanation}</div>}
      {perm.risks && perm.risks.length > 0 && (
        <div className="risk-chips">
          {perm.risks.map((r) => (
            <span key={r.key} className={`risk-chip risk-${r.severity}`} title={r.key}>
              {r.label}
            </span>
          ))}
        </div>
      )}
      {perm.toolCall.rawInput != null && (
        <pre className="approval-input">{JSON.stringify(perm.toolCall.rawInput, null, 2)}</pre>
      )}
      <div className="approval-btns">
        {perm.options.map((opt) => (
          <button
            key={opt.optionId}
            className={opt.kind?.includes('reject') ? 'abtn-reject' : 'abtn-approve'}
            onClick={() => onDecide(opt.optionId)}
          >
            {opt.kind?.includes('reject') ? '' : '✓ '}
            {opt.name ?? opt.optionId}
          </button>
        ))}
        <button className="btn-plain" onClick={() => onDecide(null)}>
          Cancel
        </button>
      </div>
    </div>
  )
}

/** The agent chat: streaming transcript, tool-call cards, permission prompts. */
export function Chat() {
  const [items, setItems] = useState<ChatItem[]>([])
  const [toolCalls, setToolCalls] = useState<Record<string, ToolCallState>>({})
  const [status, setStatus] = useState<ChatSocketStatus>('connecting')
  const [busy, setBusy] = useState(false)
  const [usage, setUsage] = useState<number | null>(null)
  const [permission, setPermission] = useState<PermissionRequest | null>(null)
  const [input, setInput] = useState('')
  const socketRef = useRef<ChatSocket | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onFrame = (frame: ServerFrame) => {
      switch (frame.type) {
        case 'chunk':
          setItems((prev) => {
            const last = prev[prev.length - 1]
            if (last && last.kind === 'assistant') {
              return [...prev.slice(0, -1), { kind: 'assistant', text: last.text + frame.text }]
            }
            return [...prev, { kind: 'assistant', text: frame.text }]
          })
          break
        case 'tool_call':
          setToolCalls((prev) => {
            if (!(frame.toolCallId in prev)) {
              setItems((items) => [...items, { kind: 'tool', toolCallId: frame.toolCallId }])
            }
            return {
              ...prev,
              [frame.toolCallId]: {
                toolCallId: frame.toolCallId,
                title: frame.title,
                status: frame.status,
                kind: frame.kind,
                rawInput: frame.rawInput,
                output: prev[frame.toolCallId]?.output,
              },
            }
          })
          break
        case 'tool_call_update':
          setToolCalls((prev) => {
            const tc = prev[frame.toolCallId]
            if (!tc) return prev
            return {
              ...prev,
              [frame.toolCallId]: {
                ...tc,
                status: frame.status ?? tc.status,
                output: frame.output ?? tc.output,
              },
            }
          })
          break
        case 'usage':
          setUsage(frame.used)
          break
        case 'permission_request':
          setPermission({
            requestId: frame.requestId,
            toolCall: frame.toolCall,
            options: frame.options,
            explanation: frame.explanation,
            risks: frame.risks,
          })
          break
        case 'done':
          setBusy(false)
          setPermission(null)
          break
        case 'error':
          setItems((prev) => [...prev, { kind: 'error', text: frame.message }])
          break
        case 'ready':
          break
      }
    }
    const socket = new ChatSocket({ onFrame, onStatusChange: setStatus })
    socketRef.current = socket
    return () => socket.close()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [items, permission])

  const send = () => {
    const text = input.trim()
    if (!text || status !== 'open') return
    setItems((prev) => [...prev, { kind: 'user', text }])
    setInput('')
    setBusy(true)
    socketRef.current?.sendPrompt(text)
  }

  const decide = (optionId: string | null) => {
    if (permission) {
      socketRef.current?.sendPermission(permission.requestId, optionId)
      setPermission(null)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Chat</span>
        <span className="panel-actions chat-meta">
          {usage != null && <span className="usage">{usage.toLocaleString()} tokens</span>}
          <span className={`conn conn-${status}`}>{status}</span>
        </span>
      </div>
      <div className="panel-body chat-scroll">
        {items.length === 0 && (
          <div className="viewer-message">Ask the MedMCP agent anything about your workspace.</div>
        )}
        {items.map((item, i) => {
          if (item.kind === 'tool') {
            const tc = toolCalls[item.toolCallId]
            return tc ? <ToolCard key={i} tc={tc} /> : null
          }
          if (item.kind === 'user') {
            return (
              <div key={i} className="msg msg-user">
                <div className="msg-avatar avatar-user">You</div>
                <div className="msg-bubble bubble-user">{item.text}</div>
              </div>
            )
          }
          if (item.kind === 'error') {
            return (
              <div key={i} className="msg-error">
                {item.text}
              </div>
            )
          }
          return (
            <div key={i} className="msg msg-ai">
              <div className="msg-avatar avatar-ai">AI</div>
              <div className="msg-bubble bubble-ai">
                <ReactMarkdown>{item.text}</ReactMarkdown>
              </div>
            </div>
          )
        })}
        {permission && <PermissionCard perm={permission} onDecide={decide} />}
        {busy && !permission && <div className="thinking">●●●</div>}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <textarea
          value={input}
          placeholder="Message the agent… (Enter to send, Shift+Enter for newline)"
          rows={2}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
        />
        {busy ? (
          <button className="btn-danger" onClick={() => socketRef.current?.cancel()}>
            Stop
          </button>
        ) : (
          <button className="btn-primary" onClick={send} disabled={status !== 'open'}>
            Send
          </button>
        )}
      </div>
    </div>
  )
}
