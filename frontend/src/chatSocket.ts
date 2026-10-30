import { canUseApi } from './api'
import type { ServerFrame } from './types'

/** Callbacks the UI registers for socket lifecycle + frames. */
export interface ChatSocketHandlers {
  onFrame: (frame: ServerFrame) => void
  onStatusChange: (status: ChatSocketStatus) => void
  /** The server is refusing this browser outright, with its reason. Reconnecting
   *  will not fix it, so the reason belongs on screen instead of "closed". */
  onRefused?: (detail: string) => void
}

// Two failed connections in a row is late enough not to fire on an ordinary
// restart and early enough to beat the backoff reaching its 15 s ceiling.
const REFUSAL_PROBE_AFTER = 2

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
  private failures = 0
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
          this.failures = 0
          // Reattach to this exact session if the socket later drops.
          this.resumeId = frame.sessionId
        }
        this.handlers.onFrame(frame)
      } catch {
        /* ignore malformed frames */
      }
    }
    ws.onclose = () => {
      this.handlers.onStatusChange('closed')
      if (this.closedByUser) return
      this.failures += 1
      if (this.failures === REFUSAL_PROBE_AFTER) void this.probeRefusal()
      this.retryTimer = window.setTimeout(() => this.connect(), this.retryDelay)
      this.retryDelay = Math.min(this.retryDelay * 2, 15000)
    }
  }

  /** Ask over HTTP why the socket will not stay up.
   *
   * The upgrade is refused *before* `accept` (OriginGuard / AccessGuard are pure
   * ASGI for that reason), so the browser is handed a close event with code 1006
   * and no reason at all — there is nothing to read off the socket. The same
   * refusal over HTTP carries the server's own wording. A 401 is left alone: that
   * is the login path, and `api.ts` already reloads onto the login screen.
   */
  private async probeRefusal(): Promise<void> {
    const access = await canUseApi()
    if (access.allowed || access.status === 401 || !access.detail) return
    this.handlers.onRefused?.(access.detail)
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

  /** Drop the connection and reattach to the same session (server-side state
   * changed under us — e.g. a rewind truncated the transcript — and the
   * reconnect's history replay is the clean way to rebuild it). */
  reconnect(): void {
    this.ws?.close()
  }
}
