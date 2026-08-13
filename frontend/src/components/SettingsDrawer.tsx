import { useEffect, useState } from 'react'
import { fetchGpus, fetchSettings, saveSettings } from '../api'
import type { GpuInfo, SettingsState } from '../types'
import { ChevronRightIcon, XIcon } from './icons'

interface SettingsDrawerProps {
  open: boolean
  onClose: () => void
}

function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      className={`toggle${checked ? ' on' : ''}`}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span className="toggle-knob" />
    </button>
  )
}

function Row({
  label,
  hint,
  checked,
  onChange,
  disabled,
}: {
  label: string
  hint?: string
  checked: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
}) {
  return (
    <div className={`settings-row${disabled ? ' disabled' : ''}`}>
      <div className="settings-row-text">
        <div className="settings-row-label">{label}</div>
        {hint && <div className="settings-row-hint">{hint}</div>}
      </div>
      <Toggle checked={checked} onChange={onChange} disabled={disabled} />
    </div>
  )
}

/**
 * Right-side drawer with the chat control panels: feature toggles, MCP stack
 * switches, and the stack GPU. Every change is saved immediately; stack and GPU
 * changes restart the agent (the chat reconnects into a fresh session).
 */
export function SettingsDrawer({ open, onClose }: SettingsDrawerProps) {
  const [state, setState] = useState<SettingsState | null>(null)
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [advanced, setAdvanced] = useState(false)

  // Clear the toast on its own: it confirms something that already happened, so
  // leaving it up implies a state that still needs attention.
  useEffect(() => {
    if (!notice) return
    const t = window.setTimeout(() => setNotice(null), 4000)
    return () => window.clearTimeout(t)
  }, [notice])

  useEffect(() => {
    if (!open) return
    // Settings gates on its own state only; the GPU list does not. Enumerating
    // GPUs costs a container spawn on the server, and awaiting it alongside made
    // opening the drawer take about a second even though everything else was
    // already there. It fills the picker in when it lands.
    fetchSettings()
      .then((s) => {
        setState(s)
        setError(null)
        setNotice(null)
      })
      .catch((e: unknown) => setError(String(e)))
    fetchGpus()
      .then(setGpus)
      .catch(() => setGpus([])) // best-effort: the field falls back to free text
  }, [open])

  const apply = (next: SettingsState) => {
    setState(next)
    setSaving(true)
    saveSettings(next)
      .then((restarted) => {
        setError(null)
        setNotice(restarted ? 'Agent restarted — the chat starts a fresh session.' : null)
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setSaving(false))
  }

  if (!open) return null


  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer">
        <div className="panel-header">
          <span>Settings</span>
          <span className="panel-actions">
            <button className="btn-icon" title="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>
        <div className="drawer-body">
          {error && <div className="panel-error">{error}</div>}
          {!state ? (
            <div className="viewer-message">Loading…</div>
          ) : (
            <>
              <div className="settings-section">General</div>
              <Row
                label="Explain tool calls"
                hint="Adds a plain-language explanation and risk tags to each permission prompt."
                checked={state.explain_tools}
                onChange={(v) => apply({ ...state, explain_tools: v })}
              />
              <div className="settings-row">
                <div className="settings-row-text">
                  <div className="settings-row-label">GPU</div>
                  <div className="settings-row-hint">Imaging stacks; chat model set at startup.</div>
                </div>
                <select
                  className="wf-input gpu-select"
                  value={state.gpu}
                  disabled={saving}
                  onChange={(e) => apply({ ...state, gpu: e.target.value })}
                >
                  <option value="all">All GPUs</option>
                  {gpus.map((g) => (
                    <option key={g.uuid} value={g.index}>
                      GPU {g.index}
                    </option>
                  ))}
                  {state.gpu !== 'all' && !gpus.some((g) => g.index === state.gpu) && (
                    <option value={state.gpu}>{state.gpu}</option>
                  )}
                </select>
              </div>

              {/* Provenance is on, and meant to stay on — it is the record of what
                  the agent did to the data. The switch survives for the rare case
                  that needs it, one disclosure away from being reached by accident. */}
              <button
                type="button"
                className="settings-advanced-toggle"
                onClick={() => setAdvanced((v) => !v)}
              >
                <ChevronRightIcon
                  size={12}
                  className={advanced ? 'settings-chevron open' : 'settings-chevron'}
                />
                Advanced
              </button>
              {advanced && (
                <Row
                  label="Record provenance"
                  hint="Keeps a replayable record of what each chat did (manifest, tool log, permissions). Turning this off means a chat leaves no audit trail and cannot be distilled into a workflow."
                  checked={state.record_provenance}
                  onChange={(v) => apply({ ...state, record_provenance: v })}
                />
              )}

              <div className="drawer-footnote">
                Stack and GPU changes restart the agent; open chats reconnect into a fresh session.
                {saving ? ' Saving…' : ''}
              </div>
            </>
          )}
        </div>
        {/* Pinned over the drawer rather than placed in the flow. As the first
            child of the body it pushed every control down the moment a setting
            was saved — so the row you had just clicked moved out from under the
            pointer, which is both disorienting and a way to mis-click the next
            toggle. Overlaying costs no layout. */}
        {notice && (
          <div className="drawer-toast" role="status">
            {notice}
          </div>
        )}
      </aside>
    </>
  )
}
