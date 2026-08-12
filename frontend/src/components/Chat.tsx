import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { ChatSocket } from '../chatSocket'
import type { ChatSocketStatus } from '../chatSocket'
import { previewRewind, rewindTo } from '../api'
import type { ChatItem, PermissionRequest, ServerFrame, ToolCallState } from '../types'
import { ChatsMenu } from './ChatsMenu'
import { ShieldIcon } from './icons'

// vibe-acp >= 2.14 sends a tool call's rawInput as a JSON-encoded *string*
// (read/edit/write tools) rather than an object. Parse it back so the card
// shows a pretty object instead of an escaped-JSON blob; non-JSON strings are
// shown as-is and objects are pretty-printed as before.
function formatToolInput(raw: unknown): string {
  if (typeof raw === 'string') {
    try {
      const parsed: unknown = JSON.parse(raw)
      if (parsed !== null && typeof parsed === 'object') {
        return JSON.stringify(parsed, null, 2)
      }
    } catch {
      // not JSON — fall through and show the raw string
    }
    return raw
  }
  return JSON.stringify(raw, null, 2)
}

// Memoized so a streaming update to the newest message doesn't re-render
// (and re-parse the markdown of) every earlier row in the transcript.
const ToolCard = memo(function ToolCard({ tc }: { tc: ToolCallState }) {
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
          <pre>{formatToolInput(tc.rawInput)}</pre>
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
})

/** Render a token count compactly, e.g. 131072 → "131k". */
function fmtTokens(n: number): string {
  if (n < 1000) return String(n)
  const k = n / 1000
  return `${k >= 100 ? Math.round(k) : k.toFixed(1)}k`
}

/** Compact context meter: fill bar + "used / window" label. */
function ContextMeter({ used, size }: { used: number; size: number | null }) {
  if (size == null || size <= 0) {
    return <span className="usage">{used.toLocaleString()} tokens</span>
  }
  const frac = Math.min(used / size, 1)
  const level = frac > 0.9 ? 'high' : frac > 0.7 ? 'mid' : 'low'
  return (
    <span
      className="ctx-meter"
      title={`Context: ${used.toLocaleString()} of ${size.toLocaleString()} tokens (${Math.round(frac * 100)}%)`}
    >
      <span className="ctx-bar">
        <span className={`ctx-fill ctx-${level}`} style={{ width: `${frac * 100}%` }} />
      </span>
      <span className="usage">
        {fmtTokens(used)} / {fmtTokens(size)}
      </span>
    </span>
  )
}

// Split markdown into blocks at blank lines, but never inside a fenced code
// block (``` / ~~~), so a code block containing blank lines stays one block.
// Used to keep settled blocks stable while a message streams in.
function splitMarkdownBlocks(text: string): string[] {
  const blocks: string[] = []
  let current: string[] = []
  let inFence = false
  const flush = () => {
    if (current.length > 0) {
      blocks.push(current.join('\n'))
      current = []
    }
  }
  for (const line of text.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence
      current.push(line)
    } else if (!inFence && line.trim() === '') {
      flush()
    } else {
      current.push(line)
    }
  }
  flush()
  return blocks
}

// remark-math only renders $$…$$ as a centered *display* block when the fences
// sit on their own lines; a single-line $$…$$ (which models routinely emit)
// parses as inline math — no displaystyle, no .katex-display scroll container.
// Promote a standalone single-line display equation to block form. Lines inside
// fenced code are skipped so literal $$ in a code sample is never touched, and a
// line carrying more than one $$…$$ pair is left as inline (ambiguous).
function promoteDisplayMath(text: string): string {
  if (!text.includes('$$')) return text
  let inFence = false
  return text
    .split('\n')
    .map((line) => {
      if (/^\s*(```|~~~)/.test(line)) {
        inFence = !inFence
        return line
      }
      if (inFence) return line
      const m = /^\s*\$\$(.+?)\$\$\s*$/.exec(line)
      if (m && !m[1].includes('$$')) return `$$\n${m[1].trim()}\n$$`
      return line
    })
    .join('\n')
}

// One markdown block, memoized on its source: a settled block never re-parses
// once its text stops changing. remark-gfm adds tables, strikethrough, task
// lists, and bare-URL autolinks on top of CommonMark; remark-math + rehype-katex
// render LaTeX ($…$ inline, $$…$$ display).
const REMARK_PLUGINS = [remarkGfm, remarkMath]
const REHYPE_PLUGINS = [rehypeKatex]
const MarkdownBlock = memo(function MarkdownBlock({ source }: { source: string }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>
      {source}
    </ReactMarkdown>
  )
})

const AssistantMessage = memo(function AssistantMessage({
  text,
  streaming,
}: {
  text: string
  streaming: boolean
}) {
  // Re-parsing the whole (growing) message through react-markdown on every
  // 50ms chunk flush is ~O(n²) and is what makes a long answer feel heavy. While
  // streaming, split off the settled blocks (rendered as one memoized document
  // so list numbering etc. stay correct) from the still-growing trailing block,
  // so only that small tail re-parses per flush. A settled message renders as a
  // single document — fully correct, parsed once.
  const { head, tail } = useMemo(() => {
    const md = promoteDisplayMath(text)
    if (!streaming) return { head: md, tail: '' }
    const blocks = splitMarkdownBlocks(md)
    return {
      head: blocks.slice(0, -1).join('\n\n'),
      tail: blocks.length > 0 ? blocks[blocks.length - 1] : '',
    }
  }, [text, streaming])
  return (
    <div className="msg msg-ai">
      <div className="msg-avatar avatar-ai">AI</div>
      <div className="msg-bubble bubble-ai">
        {head && <MarkdownBlock source={head} />}
        {tail && <MarkdownBlock source={tail} />}
      </div>
    </div>
  )
})

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
      {perm.explanation ? (
        <div className="approval-explanation">{perm.explanation}</div>
      ) : perm.explaining ? (
        <div className="approval-pending">Generating explanation…</div>
      ) : null}
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
        <pre className="approval-input">{formatToolInput(perm.toolCall.rawInput)}</pre>
      )}
      {perm.paths && perm.paths.length > 0 && (
        // Sits directly under the arguments it describes. Every checked path is
        // listed, not just the problems: confirming that an input really is there
        // is worth as much as flagging one that isn't.
        <ul className="approval-paths">
          {perm.paths.map((p, i) => (
            <li key={`${p.param}-${i}`} className={`path-${p.severity}`} title={p.value}>
              <span className="path-mark" aria-hidden="true">
                {p.severity === 'error' ? '✕' : p.severity === 'warning' ? '!' : '✓'}
              </span>
              <code className="path-param">{p.param}</code>
              <span className="path-note">{p.note}</span>
            </li>
          ))}
        </ul>
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

// The message composer, split out (with its own draft state) so a keystroke
// re-renders only this textarea — not the whole transcript above it. Memoized,
// so a streaming flush in the parent doesn't re-render it either. Enter sends
// even while busy (the server supersedes the in-flight prompt).
const ChatInput = memo(function ChatInput({
  busy,
  canSend,
  onSend,
  onCancel,
}: {
  busy: boolean
  canSend: boolean
  onSend: (text: string) => void
  onCancel: () => void
}) {
  const [input, setInput] = useState('')
  const submit = () => {
    const text = input.trim()
    if (!text || !canSend) return
    onSend(text)
    setInput('')
  }
  return (
    <div className="chat-input-row">
      <textarea
        value={input}
        placeholder="Message the agent… (Enter to send, Shift+Enter for newline)"
        rows={2}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
      />
      {busy ? (
        <button className="btn-danger" onClick={onCancel}>
          Stop
        </button>
      ) : (
        <button className="btn-primary" onClick={submit} disabled={!canSend}>
          Send
        </button>
      )}
    </div>
  )
})

/** The agent chat: streaming transcript, tool-call cards, permission prompts. */
// Memoized: App bumps `fsVersion` on every completed tool call to refresh the
// explorer/viewer, which would otherwise re-render the whole transcript here on
// each one. Chat's props from App are all stable refs, so memo skips those.
export const Chat = memo(function Chat({
  onPromptedSession,
  viewedPath,
  onToolActivity,
  resumeSessionId,
  onSessionEstablished,
  onNewChat,
  currentSessionId,
  onSelectSession,
}: {
  /** Called with the vibe session id whenever a prompt is sent into it. */
  onPromptedSession?: (id: string) => void
  /** Workspace-relative file open in the viewer, sent as prompt context. */
  viewedPath?: string | null
  /** Called when a tool call completes / a turn ends (may have written files). */
  onToolActivity?: () => void
  /** Session to resume on connect; the server reloads it and replays history. */
  resumeSessionId?: string | null
  /** Called with the live session id once the server reports `ready`. */
  onSessionEstablished?: (id: string) => void
  /** Start a fresh chat (parent remounts this component with no resume id). */
  onNewChat?: () => void
  /** The session currently shown — highlighted in the Chats menu. */
  currentSessionId?: string | null
  /** Open a previous session (parent remounts this component to resume it). */
  onSelectSession?: (id: string) => void
}) {
  const [items, setItems] = useState<ChatItem[]>([])
  const [toolCalls, setToolCalls] = useState<Record<string, ToolCallState>>({})
  const [status, setStatus] = useState<ChatSocketStatus>('connecting')
  const [busy, setBusy] = useState(false)
  const [model, setModel] = useState<string | null>(null)
  const [usage, setUsage] = useState<{ used: number; size: number | null } | null>(null)
  const [permission, setPermission] = useState<PermissionRequest | null>(null)
  // Pending rewind confirmation: set after the preview call, cleared on
  // confirm/cancel/reconnect. `paths` are the workspace files a rewind restores.
  const [rewind, setRewind] = useState<{
    messageId: string
    paths: string[]
    busy: boolean
  } | null>(null)
  const socketRef = useRef<ChatSocket | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  // The frame handler below is a mount-once closure; route the callback
  // through a ref so it always sees the latest prop.
  const onToolActivityRef = useRef(onToolActivity)
  useEffect(() => {
    onToolActivityRef.current = onToolActivity
  }, [onToolActivity])
  const onSessionEstablishedRef = useRef(onSessionEstablished)
  useEffect(() => {
    onSessionEstablishedRef.current = onSessionEstablished
  }, [onSessionEstablished])
  const onPromptedSessionRef = useRef(onPromptedSession)
  useEffect(() => {
    onPromptedSessionRef.current = onPromptedSession
  }, [onPromptedSession])
  // Captured once: the socket is created in a mount-only effect, and the parent
  // remounts this component (new key) when switching/starting sessions.
  const resumeIdRef = useRef(resumeSessionId)

  useEffect(() => {
    // Tool-call ids that already have a transcript row (ids are unique across
    // sessions, so this only ever grows by one entry per tool call).
    const seenToolIds = new Set<string>()
    // Chunks arrive near per-token; buffer them and flush into state on a
    // short timer so a long answer doesn't trigger thousands of re-renders
    // (each re-parsing the accumulated markdown).
    let chunkBuffer = ''
    let flushTimer: number | null = null
    const flushChunks = () => {
      if (flushTimer != null) {
        clearTimeout(flushTimer)
        flushTimer = null
      }
      if (!chunkBuffer) return
      const text = chunkBuffer
      chunkBuffer = ''
      setItems((prev) => {
        const last = prev[prev.length - 1]
        if (last && last.kind === 'assistant') {
          return [...prev.slice(0, -1), { kind: 'assistant', text: last.text + text }]
        }
        return [...prev, { kind: 'assistant', text }]
      })
    }
    const onFrame = (frame: ServerFrame) => {
      // Any non-chunk frame that appends to the transcript must flush first,
      // or buffered text would land after the item it preceded.
      if (frame.type !== 'chunk') flushChunks()
      switch (frame.type) {
        case 'chunk':
          chunkBuffer += frame.text
          if (flushTimer == null) {
            flushTimer = window.setTimeout(flushChunks, 50)
          }
          break
        case 'tool_call':
          // Track first-seen ids outside the updater — calling setItems from
          // inside the setToolCalls updater would make it impure, and React
          // may re-invoke updaters (appending duplicate transcript rows).
          if (!seenToolIds.has(frame.toolCallId)) {
            seenToolIds.add(frame.toolCallId)
            setItems((items) => [...items, { kind: 'tool', toolCallId: frame.toolCallId }])
          }
          setToolCalls((prev) => ({
            ...prev,
            [frame.toolCallId]: {
              toolCallId: frame.toolCallId,
              title: frame.title,
              status: frame.status,
              kind: frame.kind,
              rawInput: frame.rawInput,
              output: prev[frame.toolCallId]?.output,
            },
          }))
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
                // Sent again once the model finished streaming the arguments;
                // the opening frame's copy can be empty or partial.
                rawInput: frame.rawInput ?? tc.rawInput,
              },
            }
          })
          if (frame.status === 'completed') onToolActivityRef.current?.()
          break
        case 'usage':
          setUsage({ used: frame.used, size: frame.size ?? null })
          break
        case 'permission_request':
          setPermission({
            requestId: frame.requestId,
            toolCall: frame.toolCall,
            options: frame.options,
            explanation: frame.explanation,
            explaining: frame.explaining,
            risks: frame.risks,
          })
          break
        case 'permission_update':
          // The explanation arrives after the box is shown; ignore it if the
          // user already decided (the box for that requestId is gone).
          setPermission((p) =>
            p && p.requestId === frame.requestId
              ? { ...p, explanation: frame.explanation, risks: frame.risks, explaining: false }
              : p,
          )
          break
        case 'done':
          setBusy(false)
          setPermission(null)
          onToolActivityRef.current?.()
          break
        case 'error':
          setItems((prev) => [...prev, { kind: 'error', text: frame.message }])
          break
        case 'user':
          // Two sources: history replay on resume, and vibe ≥2.23's echo of a
          // *live* prompt. Live sends are already rendered locally in send(),
          // so when the echo matches the newest id-less user bubble, merge its
          // messageId into that bubble instead of appending a duplicate — this
          // is also what gives live turns a rewind anchor without a reload.
          setItems((prev) => {
            for (let j = prev.length - 1; j >= 0; j--) {
              const it = prev[j]
              if (it.kind !== 'user') continue
              if (!it.messageId && it.text === frame.text) {
                const next = prev.slice()
                next[j] = { ...it, messageId: frame.messageId }
                return next
              }
              break // newest user turn differs — a genuine replayed frame
            }
            return [...prev, { kind: 'user', text: frame.text, messageId: frame.messageId }]
          })
          break
        case 'ready':
          // (Re)connect: the server is the transcript's source of truth — a
          // resume replays history, a reconnect reloads it — so rebuild from
          // scratch and drop any buffered/partial state from the old socket.
          seenToolIds.clear()
          chunkBuffer = ''
          if (flushTimer != null) {
            clearTimeout(flushTimer)
            flushTimer = null
          }
          sessionIdRef.current = frame.sessionId
          setItems([])
          setToolCalls({})
          setBusy(false)
          setPermission(null)
          setRewind(null)
          if (frame.model) setModel(frame.model)
          onSessionEstablishedRef.current?.(frame.sessionId)
          break
      }
    }
    const onStatusChange = (s: ChatSocketStatus) => {
      setStatus(s)
      if (s === 'closed') {
        // The in-flight prompt and any pending approval died with the socket
        // (the server's final `done` frame is lost on close, and the next
        // session knows nothing about the old requestId) — reset, or the UI
        // is stuck on a dead Stop button after every reconnect.
        setBusy(false)
        setPermission(null)
      }
    }
    const socket = new ChatSocket({ onFrame, onStatusChange }, resumeIdRef.current)
    socketRef.current = socket
    return () => {
      socket.close()
      if (flushTimer != null) clearTimeout(flushTimer)
    }
  }, [])

  useEffect(() => {
    // Smooth scrolling per streamed flush is janky; only ease when idle.
    bottomRef.current?.scrollIntoView({ behavior: busy ? 'auto' : 'smooth' })
  }, [items, permission, busy])

  // Stable so the memoized ChatInput doesn't re-render on every streaming flush;
  // the trim/empty + open-socket guards live in ChatInput now.
  const send = useCallback(
    (text: string) => {
      setItems((prev) => [...prev, { kind: 'user', text }])
      setBusy(true)
      socketRef.current?.sendPrompt(text, viewedPath)
      if (sessionIdRef.current) onPromptedSession?.(sessionIdRef.current)
    },
    [viewedPath, onPromptedSession],
  )

  const cancel = useCallback(() => socketRef.current?.cancel(), [])

  const decide = (optionId: string | null) => {
    if (permission) {
      socketRef.current?.sendPermission(permission.requestId, optionId)
      setPermission(null)
    }
  }

  const askRewind = async (messageId: string) => {
    const sid = sessionIdRef.current
    if (!sid) return
    try {
      const res = await previewRewind(sid, messageId)
      setRewind({ messageId, paths: res.paths ?? [], busy: false })
    } catch (e) {
      setItems((prev) => [
        ...prev,
        { kind: 'error', text: `Rewind preview failed: ${(e as Error).message}` },
      ])
    }
  }

  const doRewind = async () => {
    const sid = sessionIdRef.current
    if (!rewind || !sid) return
    setRewind({ ...rewind, busy: true })
    try {
      await rewindTo(sid, rewind.messageId)
      // The transcript is truncated server-side; the reconnect's history
      // replay is the clean rebuild (ready clears items and rewind state).
      socketRef.current?.reconnect()
    } catch (e) {
      setRewind(null)
      setItems((prev) => [
        ...prev,
        { kind: 'error', text: `Rewind failed: ${(e as Error).message}` },
      ])
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="chat-head-left">
          <span>Chat</span>
          {onNewChat && (
            <button className="btn-plain chat-new-btn" onClick={onNewChat} title="Start a new chat">
              ＋ New chat
            </button>
          )}
          {onSelectSession && (
            <ChatsMenu
              currentSessionId={currentSessionId ?? null}
              onSelectSession={onSelectSession}
              onCurrentDeleted={onNewChat ?? (() => undefined)}
            />
          )}
        </span>
        <span className="panel-actions chat-meta">
          {model != null && <span className="model-name">{model}</span>}
          {usage != null && <ContextMeter used={usage.used} size={usage.size} />}
          <span className={`conn conn-${status}`}>{status === 'open' ? 'running' : status}</span>
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
            const mid = item.messageId
            return (
              <div key={i} className="user-turn">
                <div className="msg msg-user">
                  {mid && !busy && (
                    <button
                      className="btn-icon rewind-btn"
                      title="Rewind the chat to before this message"
                      onClick={() => void askRewind(mid)}
                    >
                      ↺
                    </button>
                  )}
                  <div className="msg-avatar avatar-user">You</div>
                  <div className="msg-bubble bubble-user">{item.text}</div>
                </div>
                {rewind && rewind.messageId === mid && (
                  <div className="rewind-confirm">
                    <div className="rewind-confirm-title">Rewind to before this message?</div>
                    <div>
                      The conversation from here on is discarded.
                      {rewind.paths.length > 0
                        ? ` ${rewind.paths.length} workspace file(s) will be restored:`
                        : ' No workspace files need restoring.'}
                    </div>
                    {rewind.paths.length > 0 && (
                      <ul className="rewind-paths">
                        {rewind.paths.map((p) => (
                          <li key={p}>{p}</li>
                        ))}
                      </ul>
                    )}
                    <div className="rewind-actions">
                      <button
                        className="btn-plain"
                        disabled={rewind.busy}
                        onClick={() => setRewind(null)}
                      >
                        Cancel
                      </button>
                      <button
                        className="btn-danger"
                        disabled={rewind.busy}
                        onClick={() => void doRewind()}
                      >
                        {rewind.busy ? 'Rewinding…' : 'Rewind'}
                      </button>
                    </div>
                  </div>
                )}
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
          // The actively-streaming message is the last item while busy; it
          // renders block-split. Everything else renders as a settled document.
          return (
            <AssistantMessage key={i} text={item.text} streaming={busy && i === items.length - 1} />
          )
        })}
        {permission && <PermissionCard perm={permission} onDecide={decide} />}
        {busy && !permission && <div className="thinking">●●●</div>}
        <div ref={bottomRef} />
      </div>
      <ChatInput busy={busy} canSend={status === 'open'} onSend={send} onCancel={cancel} />
    </div>
  )
})
