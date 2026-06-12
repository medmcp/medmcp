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
 * Each (re)connection creates a fresh vibe-acp session server-side; the UI
 * keeps its transcript client-side across reconnects.
 */
export class ChatSocket {
  private ws: WebSocket | null = null
  private handlers: ChatSocketHandlers
  private closedByUser = false
  private retryDelay = 1000
  private retryTimer: number | null = null

  constructor(handlers: ChatSocketHandlers) {
    this.handlers = handlers
    this.connect()
  }

  private connect(): void {
    // A pending reconnect timer can fire after close(); without this guard it
    // would open a zombie socket nothing ever closes (and the server would
    // allocate a session for it).
    if (this.closedByUser) return
    this.handlers.onStatusChange('connecting')
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat`)
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
