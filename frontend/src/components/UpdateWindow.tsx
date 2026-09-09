import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { UpdateState } from '../types'
import { XIcon } from './icons'

interface UpdateWindowProps {
  open: boolean
  onClose: () => void
  state: UpdateState | null
  /** "Later": hide the header notice for this release. */
  onDismiss: () => void
}

type Mode = 'info' | 'confirm' | 'pulling' | 'restarting' | 'failed' | 'timeout'

/** How long to wait for the new release to answer before giving the host command. */
const RESTART_DEADLINE_MS = 6 * 60 * 1000
const HEALTH_POLL_MS = 2000
const REMARK_PLUGINS = [remarkGfm]
const NO_LOADING_ELEMENTS = ['img']

function formatDate(iso: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString()
}

/**
 * What a new release changes, and the button that applies it.
 *
 * The update itself is the one operation in the product that replaces the
 * software you are looking at, so the window narrates it in stages rather than
 * spinning: the image pull streams while this server is still up (a failed pull
 * leaves everything as it was), then the server hands over to the helper and
 * this page waits for the new version to answer before reloading. Every failure
 * path ends with the host command, never with a page that just stopped.
 */
export function UpdateWindow({ open, onClose, state, onDismiss }: UpdateWindowProps) {
  const [mode, setMode] = useState<Mode>('info')
  const [log, setLog] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<number | null>(null)

  // Unmount cleanup only: a socket or poll left running would keep acting on
  // a window that is gone.
  useEffect(
    () => () => {
      socketRef.current?.close()
      socketRef.current = null
      if (pollRef.current != null) window.clearInterval(pollRef.current)
      pollRef.current = null
    },
    [],
  )

  // Closing starts the next opening over. A window closed mid-pull leaves the
  // pull to finish on the server (harmless to abandon) and forgets the log.
  const reset = () => {
    setMode('info')
    setLog([])
    setError(null)
  }

  if (!open || !state?.latest) return null
  const latest = state.latest
  const busy = mode === 'pulling' || mode === 'restarting'
  const commands = state.host_commands

  const waitForRestart = (target: string) => {
    const deadline = Date.now() + RESTART_DEADLINE_MS
    pollRef.current = window.setInterval(() => {
      if (Date.now() > deadline) {
        if (pollRef.current != null) window.clearInterval(pollRef.current)
        pollRef.current = null
        setMode('timeout')
        return
      }
      fetch('/healthz', { cache: 'no-store' })
        .then((r) => (r.ok ? r.json() : null))
        .then((body: { version?: string } | null) => {
          if (body?.version === target) location.reload()
        })
        .catch(() => {
          /* the old server is gone and the new one is not up yet */
        })
    }, HEALTH_POLL_MS)
  }

  const start = () => {
    setMode('pulling')
    setLog(['Pulling the new image…'])
    setError(null)
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/update`)
    socketRef.current = ws
    ws.onopen = () => ws.send(JSON.stringify({ version: latest.version }))
    ws.onmessage = (ev: MessageEvent<string>) => {
      const m = JSON.parse(ev.data) as {
        type: 'progress' | 'restarting' | 'error'
        line?: string
        message?: string
        version?: string
      }
      if (m.type === 'progress') {
        setLog((prev) => [...prev.slice(-40), m.line ?? ''])
        return
      }
      ws.close()
      socketRef.current = null
      if (m.type === 'restarting') {
        setMode('restarting')
        waitForRestart(m.version ?? latest.version)
      } else {
        setError(m.message ?? 'update failed')
        setMode('failed')
      }
    }
    ws.onerror = () => {
      if (socketRef.current !== ws) return
      socketRef.current = null
      setError('The connection to the workspace dropped before the update started.')
      setMode('failed')
    }
  }

  const close = () => {
    if (busy) return
    reset()
    onClose()
  }
  const dismiss = () => {
    reset()
    onDismiss()
  }

  return (
    <>
      <div className="modal-backdrop" onClick={close} />
      <div className="upwin" role="dialog" aria-label="Software update">
        <div className="panel-header">
          <span>Software update</span>
          <span className="panel-actions">
            <button className="btn-icon" title="Close" disabled={busy} onClick={close}>
              <XIcon />
            </button>
          </span>
        </div>
        <div className="upwin-body">
          <div className="upwin-versions">
            <span className="upwin-ver">v{state.current.version}</span>
            <span className="upwin-arrow">→</span>
            <span className="upwin-ver new">v{latest.version}</span>
            {latest.published_at && (
              <span className="upwin-date">released {formatDate(latest.published_at)}</span>
            )}
          </div>

          {mode === 'info' && (
            <>
              {latest.notes ? (
                <div className="upwin-notes">
                  {/* No images: the notes must not make the browser fetch anything.
                      Raw HTML is already off by default in react-markdown. */}
                  <ReactMarkdown
                    remarkPlugins={REMARK_PLUGINS}
                    disallowedElements={NO_LOADING_ELEMENTS}
                    unwrapDisallowed
                  >
                    {latest.notes}
                  </ReactMarkdown>
                </div>
              ) : (
                <p className="upwin-muted">This release carries no notes.</p>
              )}
              {latest.url && (
                <a className="upwin-link" href={latest.url} target="_blank" rel="noreferrer">
                  Open the release page ↗
                </a>
              )}
              {!state.can_apply && (
                <div className="upwin-reason">
                  <p>{state.apply_reason ?? 'This install is updated from the host.'}</p>
                  {commands && <pre className="upwin-cmd">{commands.update}</pre>}
                </div>
              )}
              <div className="modal-actions">
                <button className="btn-plain" onClick={dismiss}>
                  Later
                </button>
                {state.can_apply && (
                  <button className="btn-primary" onClick={() => setMode('confirm')}>
                    Update now
                  </button>
                )}
              </div>
            </>
          )}

          {mode === 'confirm' && (
            <>
              <p className="upwin-status">
                MedMCP downloads the new release and restarts. Open chats reconnect once it is
                back. Your workspace files, chats, workflows and installed stacks are kept; the
                GPU choice and which stacks are enabled go back to their defaults. Expect a few
                minutes; running tool calls are interrupted.
              </p>
              <p className="upwin-muted">
                If the new release does not start, the previous one is put back automatically.
              </p>
              <div className="modal-actions">
                <button className="btn-plain" onClick={() => setMode('info')}>
                  Back
                </button>
                <button className="btn-primary" onClick={start}>
                  Update to v{latest.version}
                </button>
              </div>
            </>
          )}

          {mode === 'pulling' && (
            <>
              <p className="upwin-status">Downloading v{latest.version}…</p>
              <pre className="upwin-log">{log.slice(-8).join('\n')}</pre>
            </>
          )}

          {mode === 'restarting' && (
            <>
              <p className="upwin-status">
                Restarting on v{latest.version}. This page reloads when the new version answers.
              </p>
              <p className="upwin-muted">
                The connection drops for a moment while the container is replaced.
              </p>
            </>
          )}

          {mode === 'failed' && (
            <>
              <p className="upwin-status upwin-error">{error}</p>
              <p className="upwin-muted">Nothing was changed. To update from the host instead:</p>
              {commands && <pre className="upwin-cmd">{commands.update}</pre>}
              <div className="modal-actions">
                <button className="btn-plain" onClick={close}>
                  Close
                </button>
              </div>
            </>
          )}

          {mode === 'timeout' && (
            <>
              <p className="upwin-status">
                The new version has not answered yet. It may still be starting, or the previous
                release may have been put back.
              </p>
              <p className="upwin-muted">
                Check with <code>docker logs medmcp-updater</code> on the host, or run the update
                there:
              </p>
              {commands && <pre className="upwin-cmd">{commands.update}</pre>}
              <p className="upwin-muted">To return to the previous release:</p>
              {commands && <pre className="upwin-cmd">{commands.rollback}</pre>}
              <div className="modal-actions">
                <button className="btn-plain" onClick={() => location.reload()}>
                  Reload
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  )
}
