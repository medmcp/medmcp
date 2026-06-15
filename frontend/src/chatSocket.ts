import type { ServerFrame } from './types'

/** Callbacks the UI registers for socket lifecycle + frames. */
export interface ChatSocketHandlers {
  onFrame: (frame: ServerFrame) => void
  onStatusChange: (status: ChatSocketStatus) => void
}

export type ChatSocketStatus = 'connecting' | 'open' | 'closed'

/**
 * WebSocket wrapper for /ws/chat with automatic reconnect.
 *
 * A connection resumes an existing session when given a `resumeId` (the server
 * reloads it and replays its transcript); otherwise it opens a fresh one. Once
 * the server reports the live session id in `ready`, that id becomes the resume
 * target, so a dropped connection reattaches to the same session rather than
 * starting over — the server is the source of truth for the transcript.
 */
export class ChatSocket {
  private ws: WebSocket | null = null
  private handlers: ChatSocketHandlers
  private closedByUser = false
  private retryDelay = 1000
  private retryTimer: number | null = null
  private resumeId: string | null

  constructor(handlers: ChatSocketHandlers, resumeId?: string | null) {
    this.handlers = handlers
    this.resumeId = resumeId ?? null
    this.connect()
  }

  private connect(): void {
    // A pending reconnect timer can fire after close(); without this guard it
    // would open a zombie socket nothing ever closes (and the server would
    // allocate a session for it).
    if (this.closedByUser) return
    this.handlers.onStatusChange('connecting')
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const query = this.resumeId ? `?resume=${encodeURIComponent(this.resumeId)}` : ''
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat${query}`)
    this.ws = ws
    ws.onopen = () => {
      this.handlers.onStatusChange('open')
    }
    ws.onmessage = (ev: MessageEvent<string>) => {
      try {
        const frame = JSON.parse(ev.data) as ServerFrame
        // Reset the backoff only once the server confirms a working session;
        // the socket is accepted before session setup, so onopen fires even
        // when setup is about to fail and would otherwise defeat the backoff.
        if (frame.type === 'ready') {
          this.retryDelay = 1000
          // Reattach to this exact session if the socket later drops.
          this.resumeId = frame.sessionId
        } else if (frame.type === 'session_migrated') {
          // The session forked on continue; reconnect to the fork, not the
          // (now-deleted) original, or a drop would lose the conversation.
          this.resumeId = frame.sessionId
        }
        this.handlers.onFrame(frame)
      } catch {
        /* ignore malformed frames */
      }
    }
    ws.onclose = () => {
      this.handlers.onStatusChange('closed')
      if (!this.closedByUser) {
        this.retryTimer = window.setTimeout(() => this.connect(), this.retryDelay)
        this.retryDelay = Math.min(this.retryDelay * 2, 15000)
      }
    }
  }

  private send(msg: object): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg))
    }
  }

  sendPrompt(text: string, viewedPath?: string | null): void {
    this.send({ type: 'prompt', text, viewedPath: viewedPath ?? null })
  }

  sendPermission(requestId: number, optionId: string | null): void {
    this.send({ type: 'permission', requestId, optionId })
  }

  cancel(): void {
    this.send({ type: 'cancel' })
  }

  close(): void {
    this.closedByUser = true
    if (this.retryTimer != null) {
      clearTimeout(this.retryTimer)
      this.retryTimer = null
    }
    this.ws?.close()
  }
}
