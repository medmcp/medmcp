import { useEffect, useState } from 'react'
import { fetchExternalMcp, fetchGpus, fetchSettings, saveSettings } from '../api'
import type { ExternalMcpState, GpuInfo, SettingsState } from '../types'
import { Row } from './SettingsControls'
import { ChevronRightIcon, XIcon } from './icons'

interface SettingsDrawerProps {
  open: boolean
  onClose: () => void
  /** Whether the Advanced disclosure is expanded (owned by the caller, so the
   *  warning banner can open the drawer straight onto the control it names). */
  advancedOpen: boolean
  /** Expand or collapse Advanced. */
  onAdvancedToggle: (open: boolean) => void
  /** Open the external-MCP window. */
  onManageExternal: () => void
  /** Bumped when external-MCP state changed, so the summary re-reads it. */
  externalVersion: number
}

/**
 * Right-side drawer with the chat control panels: feature toggles, MCP stack
 * switches, and the stack GPU. Every change is saved immediately; stack and GPU
 * changes restart the agent (the chat reconnects into a fresh session).
 */
export function SettingsDrawer({
  open,
  onClose,
  advancedOpen,
  onAdvancedToggle,
  onManageExternal,
  externalVersion,
}: SettingsDrawerProps) {
  const [state, setState] = useState<SettingsState | null>(null)
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  // Just enough external-MCP state to say what is connected; the window owns
  // the rest.
  const [external, setExternal] = useState<ExternalMcpState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

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

  useEffect(() => {
    if (!open) return
    fetchExternalMcp()
      .then(setExternal)
      .catch(() => setExternal(null)) // summary hides rather than guesses
  }, [open, externalVersion])

  const apply = (next: SettingsState) => {
    setState(next)
    setSaving(true)
    saveSettings(next)
      .then((restarted) => {
        setError(null)
        setNotice(restarted ? 'Agent restarted. The chat starts a fresh session.' : null)
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setSaving(false))
  }

  // What the row says at rest. "Connected" counts servers the agent can actually
  // reach: the feature on and the server switched on.
  const connectedServers = external?.enabled ? external.servers.filter((s) => s.active) : []
  const connected = connectedServers.length > 0
  const externalSummary = !external
    ? 'Tools hosted outside this machine.'
    : connected
      ? `${connectedServers.length} connected: ${connectedServers.map((s) => s.name).join(', ')}`
      : external.enabled
        ? 'On, with nothing connected.'
        : 'Off. Nothing is sent outside this machine.'

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
                  <div className="settings-row-hint">Used by imaging stacks. The chat model is set at startup.</div>
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
                className={`settings-advanced-toggle${advancedOpen ? ' open' : ''}`}
                aria-expanded={advancedOpen}
                onClick={() => onAdvancedToggle(!advancedOpen)}
              >
                Advanced
                <ChevronRightIcon
                  size={12}
                  className={advancedOpen ? 'settings-chevron open' : 'settings-chevron'}
                />
                {!advancedOpen && (
                  <span className="settings-advanced-peek">provenance, external servers</span>
                )}
              </button>
              {advancedOpen && (
                <div className="settings-advanced-body">
                  <Row
                    label="Record provenance"
                    hint="Keeps a record of what each chat did, so you can review it later or turn it into a workflow. With this off, a chat leaves no trail."
                    checked={state.record_provenance}
                    onChange={(v) => apply({ ...state, record_provenance: v })}
                  />
                  <div className="settings-row">
                    <div className="settings-row-text">
                      <div className="settings-row-label">External MCP servers</div>
                      <div className={`settings-row-hint${connected ? ' ext-summary-on' : ''}`}>
                        {externalSummary}
                      </div>
                    </div>
                    <button className="btn-plain" onClick={onManageExternal}>
                      Manage…
                    </button>
                  </div>
                </div>
              )}

              <div className="drawer-footnote">
                GPU and external server changes restart the agent. Open chats reconnect into a fresh session.
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
