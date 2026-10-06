import { useEffect, useState } from 'react'
import { fetchCloudModel, fetchExternalMcp, fetchGpus, fetchSettings, saveSettings } from '../api'
import type {
  CloudModelState,
  ExternalMcpState,
  GpuInfo,
  SettingsState,
  UpdateState,
} from '../types'
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
  /** Open the cloud-model window. */
  onManageCloud: () => void
  /** Open the Models window (which local model answers chats). */
  onManageModels: () => void
  /** Bumped when cloud-model state changed, so the summary re-reads it. */
  cloudVersion: number
  /** The release record (owned by App, which also polls it). */
  update: UpdateState | null
  /** Ask the server to look for a release now. */
  onCheckUpdate: () => Promise<void>
  /** Open the update window. */
  onOpenUpdate: () => void
  /** Switch the daily release check on or off. */
  onSetAutoCheck: (enabled: boolean) => Promise<void>
}

/** "2 h ago" for the version row; empty when unknown. */
function ago(iso: string | null): string {
  if (!iso) return ''
  const ms = Date.now() - new Date(iso).getTime()
  if (!Number.isFinite(ms) || ms < 0) return ''
  const min = Math.round(ms / 60000)
  if (min < 2) return 'just now'
  if (min < 60) return `${min} min ago`
  const h = Math.round(min / 60)
  if (h < 36) return `${h} h ago`
  return `${Math.round(h / 24)} d ago`
}

function versionHint(u: UpdateState | null, checking: boolean): string {
  if (!u) return ''
  if (u.available) return `v${u.latest?.version} is available.`
  if (!u.enabled) return 'Release checks are off (MEDMCP_UPDATE_CHECK=0).'
  if (checking) return 'Checking…'
  if (u.error && !u.latest) return `Could not check for releases: ${u.error}`
  if (!u.checked_at) return u.auto_check ? 'Not checked yet.' : 'Automatic checks are off.'
  const when = ago(u.checked_at)
  const status = `Up to date${when ? ` · checked ${when}` : ''}.`
  return u.auto_check ? status : `${status} Automatic checks are off.`
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
  onManageCloud,
  onManageModels,
  cloudVersion,
  update,
  onCheckUpdate,
  onOpenUpdate,
  onSetAutoCheck,
}: SettingsDrawerProps) {
  const [checking, setChecking] = useState(false)
  const [state, setState] = useState<SettingsState | null>(null)
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  // Just enough external-MCP state to say what is connected; the window owns
  // the rest.
  const [external, setExternal] = useState<ExternalMcpState | null>(null)
  const [cloud, setCloud] = useState<CloudModelState | null>(null)
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

  useEffect(() => {
    if (!open) return
    fetchCloudModel()
      .then(setCloud)
      .catch(() => setCloud(null)) // summary hides rather than guesses
  }, [open, cloudVersion])

  const apply = (next: SettingsState) => {
    setState(next)
    setSaving(true)
    saveSettings(next)
      .then((restarted) => {
        setError(null)
        setNotice(restarted ? 'Agent restarted.' : null)
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
        ? 'On, nothing connected.'
        : 'Off.'

  // "In use" means chats are actually sent there: on, and the key available.
  const cloudSummary = !cloud
    ? 'A model hosted outside this machine.'
    : cloud.active && cloud.model
      ? `In use: ${cloud.model.model}`
      : cloud.enabled
        ? 'On, but the API key is missing.'
        : 'Off.'

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
                hint="Explains each permission prompt in plain language."
                checked={state.explain_tools}
                onChange={(v) => apply({ ...state, explain_tools: v })}
              />
              <div className="settings-row">
                <div className="settings-row-text">
                  <div className="settings-row-label">Model</div>
                  <div className="settings-row-hint">The local model that answers chats.</div>
                </div>
                <button className="btn-plain" onClick={onManageModels}>
                  Change…
                </button>
              </div>
              <div className="settings-row">
                <div className="settings-row-text">
                  <div className="settings-row-label">GPU</div>
                  <div className="settings-row-hint">Used by imaging stacks.</div>
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
                  <span className="settings-advanced-peek">provenance, updates, external servers, cloud model</span>
                )}
              </button>
              {advancedOpen && (
                <div className="settings-advanced-body">
                  <Row
                    label="Record provenance"
                    hint="Keeps a record of what each chat did."
                    checked={state.record_provenance}
                    onChange={(v) => apply({ ...state, record_provenance: v })}
                  />
                  {update && (
                    <Row
                      label="Check for updates automatically"
                      hint={
                        update.enabled
                          ? 'Asks github.com once a day. No workspace data is sent.'
                          : 'Turned off for this deployment.'
                      }
                      checked={update.auto_check}
                      disabled={!update.enabled}
                      onChange={(v) => {
                        onSetAutoCheck(v).catch(() => {})
                      }}
                    />
                  )}
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
                  <div className="settings-row">
                    <div className="settings-row-text">
                      <div className="settings-row-label">Cloud model</div>
                      <div className={`settings-row-hint${cloud?.active ? ' ext-summary-on' : ''}`}>
                        {cloudSummary}
                      </div>
                    </div>
                    <button className="btn-plain" onClick={onManageCloud}>
                      Manage…
                    </button>
                  </div>
                </div>
              )}

              <div className="drawer-footnote">
                Model, GPU and external access changes restart the agent.
                {saving ? ' Saving…' : ''}
              </div>
            </>
          )}
        </div>
        {/* The version lives at the foot of the drawer rather than among the
            settings: it is not a choice, and a release check is something you
            reach for occasionally, from the same spot every time. */}
        {update && (
          <div className="drawer-footer">
            <div className="drawer-footer-text">
              <span className="drawer-footer-version">MedMCP v{update.current.version}</span>
              <span className={`drawer-footer-hint${update.available ? ' available' : ''}`}>
                {versionHint(update, checking)}
              </span>
            </div>
            {update.available ? (
              <button className="btn-primary" onClick={onOpenUpdate}>
                Update…
              </button>
            ) : (
              update.enabled && (
                <button
                  className="btn-plain"
                  disabled={checking}
                  onClick={() => {
                    setChecking(true)
                    onCheckUpdate().finally(() => setChecking(false))
                  }}
                >
                  Check for update
                </button>
              )
            )}
          </div>
        )}
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
