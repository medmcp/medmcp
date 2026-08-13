import { useEffect, useState } from 'react'
import {
  fetchCatalog,
  fetchGpus,
  fetchInstalledStacks,
  fetchSettings,
  saveSettings,
  uninstallStack,
} from '../api'
import type { CatalogEntry, GpuInfo, InstalledStack, SettingsState } from '../types'
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

/** Indeterminate progress bar + live status line, shown inline where an install runs. */
function InstallProgress({ line }: { line: string | null }) {
  return (
    <div className="install-inline">
      <div className="install-bar">
        <span className="install-bar-fill" />
      </div>
      <div className="install-line">{line ?? 'Starting…'}</div>
    </div>
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
  const [installed, setInstalled] = useState<InstalledStack[]>([])
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  const [installImage, setInstallImage] = useState('')
  const [installing, setInstalling] = useState(false)
  const [installingImage, setInstallingImage] = useState<string | null>(null)
  const [installProgress, setInstallProgress] = useState<string | null>(null)
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
    // The three cheap calls gate the panel; the GPU list does not. Enumerating
    // GPUs costs a container spawn on the server, and waiting on it inside a
    // Promise.all made opening the drawer take about a second even though
    // everything else was already there. It fills the picker in when it lands.
    Promise.all([fetchSettings(), fetchInstalledStacks(), fetchCatalog()])
      .then(([s, inst, cat]) => {
        setState(s)
        setInstalled(inst)
        setCatalog(cat)
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

  const reloadStacks = async () => {
    const [s, inst, cat] = await Promise.all([
      fetchSettings(),
      fetchInstalledStacks(),
      fetchCatalog(),
    ])
    setState(s)
    setInstalled(inst)
    setCatalog(cat)
  }

  const performInstall = (image: string, after?: () => void) => {
    if (!image || installing) return
    setInstalling(true)
    setInstallingImage(image)
    setError(null)
    setNotice(null)
    setInstallProgress('Starting…')
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/stacks/install`)
    ws.onopen = () => ws.send(JSON.stringify({ image }))
    ws.onmessage = (ev: MessageEvent<string>) => {
      const m = JSON.parse(ev.data) as {
        type: 'progress' | 'done' | 'error'
        line?: string
        name?: string
        message?: string
      }
      if (m.type === 'progress') {
        setInstallProgress(m.line ?? null)
      } else if (m.type === 'done') {
        ws.close()
        after?.()
        setInstalling(false)
        setInstallingImage(null)
        setInstallProgress(null)
        reloadStacks().then(() =>
          setNotice(`Installed ${m.name} — agent restarted into a fresh session.`),
        )
      } else {
        ws.close()
        setInstalling(false)
        setInstallingImage(null)
        setInstallProgress(null)
        setError(m.message ?? 'install failed')
      }
    }
    ws.onerror = () => {
      setInstalling(false)
      setInstallingImage(null)
      setInstallProgress(null)
      setError('install connection failed')
    }
  }

  const doInstall = () => performInstall(installImage.trim(), () => setInstallImage(''))

  const doUninstall = (name: string) => {
    if (installing) return
    setInstalling(true)
    setError(null)
    setNotice(null)
    uninstallStack(name)
      .then(async () => {
        await reloadStacks()
        setNotice(`Uninstalled ${name} — agent restarted into a fresh session.`)
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setInstalling(false))
  }

  if (!open) return null

  const installedNames = new Set(installed.map((i) => i.name))

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

              <div className="settings-section">Stacks</div>
              {state.stacks.length === 0 && (
                <div className="settings-row-hint">No stacks installed yet.</div>
              )}
              {state.stacks.map((s) => {
                const container = installedNames.has(s.name)
                return (
                  <div className="settings-row" key={s.name}>
                    <div className="settings-row-text">
                      <div className="settings-row-label">{s.name}</div>
                      {(s.version || container) && (
                        <div className="settings-row-hint">
                          {s.version ? `v${s.version}` : 'container image'}
                        </div>
                      )}
                    </div>
                    <div className="stack-row-actions">
                      <Toggle
                        checked={s.active}
                        onChange={(v) =>
                          apply({
                            ...state,
                            stacks: state.stacks.map((x) =>
                              x.name === s.name ? { ...x, active: v } : x,
                            ),
                          })
                        }
                      />
                      {container && (
                        <button
                          className="btn-plain stack-uninstall"
                          disabled={installing}
                          onClick={() => doUninstall(s.name)}
                        >
                          Uninstall
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}

              <div className="settings-section">Available</div>
              {catalog.length > 0 && catalog.every((c) => c.installed) && (
                <div className="settings-row-hint">All available stacks are installed.</div>
              )}
              {catalog
                .filter((c) => !c.installed)
                .map((c) => (
                  <div className="settings-row" key={`cat-${c.name}`}>
                    <div className="settings-row-text">
                      <div className="settings-row-label">
                        {c.name}
                        {c.gpu && <span className="stack-badge">GPU</span>}
                      </div>
                      {c.description && <div className="settings-row-hint">{c.description}</div>}
                    </div>
                    <div className="stack-row-actions">
                      {installingImage === c.image ? (
                        <InstallProgress line={installProgress} />
                      ) : (
                        <button
                          className="btn-primary btn-sm"
                          disabled={installing}
                          onClick={() => performInstall(c.image)}
                        >
                          Install
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              <div className="stack-install">
                <input
                  className="wf-input"
                  placeholder="Or install from an image, e.g. ghcr.io/medmcp/neuro:dev"
                  value={installImage}
                  disabled={installing}
                  onChange={(e) => setInstallImage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') doInstall()
                  }}
                />
                <button
                  className="btn-primary"
                  disabled={installing || !installImage.trim()}
                  onClick={doInstall}
                >
                  {installing ? 'Working…' : 'Install'}
                </button>
              </div>
              {installingImage !== null && !catalog.some((c) => c.image === installingImage) && (
                <InstallProgress line={installProgress} />
              )}

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
