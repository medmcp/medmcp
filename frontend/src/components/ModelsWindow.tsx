import { useCallback, useEffect, useRef, useState } from 'react'
import { useConfirm } from '../confirm'
import { createPortal } from 'react-dom'
import { deleteLocalModel, fetchCloudModel, fetchLocalModels } from '../api'
import { MODEL_LOGOS } from '../modelLogos'
import type { LocalModelRow, LocalModelsState } from '../types'
import { XIcon } from './icons'

interface ModelsWindowProps {
  open: boolean
  onClose: () => void
}

/** A switch in flight: which model, and how far along. */
interface Switching {
  id: string
  stage: 'download' | 'prepare' | 'switch'
  text: string
  completed: number
  total: number
}

type SelectFrame =
  | { type: 'progress'; stage: Switching['stage']; text?: string; completed?: number; total?: number }
  | { type: 'done'; id: string; model: string }
  | { type: 'needs_license'; id: string; label: string; license: string }
  | { type: 'error'; message: string }

const gb = (bytes: number) => (bytes / 1e9).toFixed(1)

/**
 * Choose which model on this machine answers chats.
 *
 * One model is in use at a time. Picking another downloads it if it is not
 * there yet (the progress shows in its row), prepares it, and restarts the
 * agent on it; downloaded models that are not in use can be removed again.
 * Muse Glimmer is the default and the one model that cannot be removed — it is
 * what everything falls back to.
 */
export function ModelsWindow({ open, onClose }: ModelsWindowProps) {
  const [state, setState] = useState<LocalModelsState | null>(null)
  // Whether a cloud model is answering chats right now; the choice made here
  // then only governs explanations, titles, and what chats return to.
  const [cloudModel, setCloudModel] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [switching, setSwitching] = useState<Switching | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  // A model whose licence has to be accepted before its download starts.
  const [consent, setConsent] = useState<{ id: string; label: string; license: string } | null>(
    null,
  )
  const socket = useRef<WebSocket | null>(null)

  const reload = useCallback(async () => {
    setState(await fetchLocalModels())
  }, [])

  useEffect(() => {
    if (!open) return
    fetchLocalModels()
      .then(setState)
      .catch((e: unknown) => setError(String(e)))
    fetchCloudModel()
      .then((c) => setCloudModel(c.active && c.model ? c.model.model : null))
      .catch(() => setCloudModel(null)) // the note hides rather than guesses
  }, [open])

  // A switch belongs to the window that started it: leaving cancels it, which
  // the server treats as safe at every point (a partial download is resumed).
  useEffect(() => () => socket.current?.close(), [])

  const use = (model: LocalModelRow, acceptLicense = false) => {
    if (switching) return
    setError(null)
    setNotice(null)
    setSwitching({ id: model.id, stage: 'download', text: 'Starting…', completed: 0, total: 0 })
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/models/select`)
    socket.current = ws
    let finished = false
    const finish = () => {
      finished = true
      socket.current = null
      ws.close()
      setSwitching(null)
    }
    ws.onopen = () => ws.send(JSON.stringify({ id: model.id, accept_license: acceptLicense }))
    ws.onmessage = (ev: MessageEvent<string>) => {
      const m = JSON.parse(ev.data) as SelectFrame
      if (m.type === 'progress') {
        setSwitching({
          id: model.id,
          stage: m.stage,
          text: m.text ?? '',
          completed: m.completed ?? 0,
          total: m.total ?? 0,
        })
        return
      }
      finish()
      if (m.type === 'needs_license') {
        setConsent({ id: m.id, label: m.label, license: m.license })
        return
      }
      if (m.type === 'done') {
        setNotice(`Now using ${model.label}.`)
      } else {
        setError(m.message)
      }
      reload().catch((e: unknown) => setError(String(e)))
    }
    ws.onclose = () => {
      if (finished) return
      // Closed without a final frame: cancelled here, or the server went away.
      socket.current = null
      setSwitching(null)
      reload().catch(() => undefined)
    }
  }

  const cancel = () => socket.current?.close()

  const confirmDialog = useConfirm()
  const remove = (model: LocalModelRow) =>
    void confirmDialog(`Delete ${model.label}?`).then((ok) => ok && doRemove(model))
  const doRemove = (model: LocalModelRow) => {
    setError(null)
    setNotice(null)
    setDeleting(model.id)
    deleteLocalModel(model.id)
      .then(() => setNotice(`Deleted ${model.label}.`))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setDeleting(null)
        reload().catch(() => undefined)
      })
  }

  if (!open) return null

  const busy = switching !== null || deleting !== null

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="extwin" role="dialog">
        <div className="panel-header">
          <span>Models</span>
          <span className="panel-actions">
            <button className="btn-icon" aria-label="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>
        <div className="extwin-body">
          {error && <div className="panel-error">{error}</div>}
          {notice && <div className="settings-row-hint models-notice">{notice}</div>}
          {state && !state.reachable && (
            <div className="settings-row-hint ext-mcp-missing-token">
              The model server is not answering.
            </div>
          )}
          {cloudModel && (
            <div className="settings-row-hint ext-mcp-missing-token">
              The cloud model {cloudModel} is answering chats. This choice applies again
              when it is switched off.
            </div>
          )}

          <div className="ext-mcp models-list">
            {!state && !error && <div className="settings-row-hint">Loading…</div>}
            {state?.models.map((m) => {
              const run = switching?.id === m.id ? switching : null
              return (
                <div key={m.id} className="models-row">
                  <div className="ext-mcp-server">
                    <ModelLogo id={m.id} vendor={m.vendor} />
                    <div className="ext-mcp-server-text models-text">
                      <div className="settings-row-label">
                        {m.label}
                        {m.active && <span className="market-state on models-state">In use</span>}
                      </div>
                      <div className="settings-row-hint">
                        {m.vendor} · {m.size_gb} GB ·{' '}
                        <span className={m.license_ack ? 'models-license-other' : undefined}>
                          {m.license}
                        </span>
                      </div>
                    </div>
                    <div className="ext-mcp-server-actions">
                      {!m.active && !run && (
                        <button
                          className="btn-plain"
                          disabled={busy || !state.reachable}
                          onClick={() => use(m)}
                        >
                          {m.downloaded ? 'Use' : 'Download'}
                        </button>
                      )}
                      {m.deletable && !run && (
                        <button
                          className="btn-text"
                          disabled={busy}
                          onClick={() => remove(m)}
                        >
                          {deleting === m.id ? 'Deleting…' : 'Delete'}
                        </button>
                      )}
                      {run && (
                        <button className="btn-text" onClick={cancel}>
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>
                  {run && <SwitchProgress run={run} />}
                </div>
              )
            })}
          </div>

          <div className="settings-row-hint models-foot">
            Downloaded from the Ollama library. Your data stays on this machine.
          </div>
        </div>
      </div>

      {/* Portalled and lifted `over-window`, like the other dialogs opened from a
          window: rendered in place it could not cover the list behind it. */}
      {consent &&
        createPortal(
          <div className="modal-backdrop over-window" onClick={() => setConsent(null)}>
            <div
              className="modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="models-license-title"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 id="models-license-title" className="models-license-title">
                Download {consent.label}?
              </h3>
              <p className="settings-row-hint">
                {consent.label} is released under the <strong>{consent.license}</strong>, not
                Apache 2.0. Continue only if its terms fit your use.
              </p>
              <div className="modal-actions">
                <button className="btn-text" onClick={() => setConsent(null)}>
                  Cancel
                </button>
                <button
                  className="btn-primary"
                  onClick={() => {
                    const model = state?.models.find((m) => m.id === consent.id)
                    setConsent(null)
                    if (model) use(model, true)
                  }}
                >
                  Accept and download
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </>
  )
}

/**
 * The model's mark, in its own colours. A single-colour mark is drawn as a mask
 * instead, so it takes a light tone from the UI: as an image it would render
 * black and vanish on the navy surface. The slot is kept when there is no mark,
 * so the names stay aligned.
 */
function ModelLogo({ id, vendor }: { id: string; vendor: string }) {
  const logo = MODEL_LOGOS[id]
  if (!logo) return <span className="models-logo" aria-hidden="true" />
  if (!logo.mono) return <img className="models-logo" src={logo.src} alt={vendor} />
  const mask = `url("${logo.src}")`
  return (
    <span
      className="models-logo mono"
      role="img"
      style={{ maskImage: mask, WebkitMaskImage: mask }}
    />
  )
}

/** The bar under the row being switched: bytes while downloading, a sweep after. */
function SwitchProgress({ run }: { run: Switching }) {
  const downloading = run.stage === 'download' && run.total > 0
  const pct = downloading ? Math.min(100, (run.completed / run.total) * 100) : 0
  const label = downloading
    ? `${gb(run.completed)} / ${gb(run.total)} GB`
    : run.stage === 'download'
      ? run.text || 'Starting…'
      : run.stage === 'prepare'
        ? 'Preparing…'
        : 'Switching…'
  return (
    <div className="wf-progress models-progress">
      <div className={`wf-progress-bar${downloading ? '' : ' active'}`}>
        <span className="wf-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="wf-progress-label">{label}</span>
    </div>
  )
}
