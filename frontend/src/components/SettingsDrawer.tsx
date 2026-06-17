import { useEffect, useState } from 'react'
import {
  fetchCatalog,
  fetchInstalledStacks,
  fetchSettings,
  installStack,
  saveSettings,
  uninstallStack,
} from '../api'
import type { CatalogEntry, InstalledStack, SettingsState } from '../types'
import { XIcon } from './icons'

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
 * switches, and personal-workflow switches. Every change is saved
 * immediately; stack/workflow changes restart the agent (the chat reconnects
 * into a fresh session).
 */
export function SettingsDrawer({ open, onClose }: SettingsDrawerProps) {
  const [state, setState] = useState<SettingsState | null>(null)
  const [installed, setInstalled] = useState<InstalledStack[]>([])
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [installImage, setInstallImage] = useState('')
  const [installing, setInstalling] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    Promise.all([fetchSettings(), fetchInstalledStacks(), fetchCatalog()])
      .then(([s, inst, cat]) => {
        setState(s)
        setInstalled(inst)
        setCatalog(cat)
        setError(null)
        setNotice(null)
      })
      .catch((e: unknown) => setError(String(e)))
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
    setError(null)
    setNotice(null)
    installStack(image)
      .then(async (name) => {
        after?.()
        await reloadStacks()
        setNotice(`Installed ${name} — agent restarted into a fresh session.`)
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setInstalling(false))
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
          {notice && <div className="drawer-notice">{notice}</div>}
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
              <Row
                label="Record provenance"
                hint="Keeps a replayable record of what each chat did (manifest, tool log, permissions)."
                checked={state.record_provenance}
                onChange={(v) => apply({ ...state, record_provenance: v })}
              />
              <Row
                label="Personal workflows"
                hint="Master switch for distilled workflows; off hides and unloads all of them."
                checked={state.workflows_enabled}
                onChange={(v) => apply({ ...state, workflows_enabled: v })}
              />

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
              {catalog.map((c) => (
                <div className="settings-row" key={`cat-${c.name}`}>
                  <div className="settings-row-text">
                    <div className="settings-row-label">
                      {c.name}
                      {c.gpu && <span className="stack-badge">GPU</span>}
                    </div>
                    {c.description && <div className="settings-row-hint">{c.description}</div>}
                  </div>
                  <div className="stack-row-actions">
                    {c.installed ? (
                      <span className="settings-row-hint">Installed</span>
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

              {state.workflows_enabled && (
                <>
                  <div className="settings-section">Workflows</div>
                  {state.workflows.length === 0 && (
                    <div className="settings-row-hint">No workflows saved yet.</div>
                  )}
                  {state.workflows.map((w) => (
                    <Row
                      key={w.name}
                      label={w.name}
                      hint={w.description || (w.kind === 'draft' ? 'draft' : undefined)}
                      checked={w.active}
                      onChange={(v) =>
                        apply({
                          ...state,
                          workflows: state.workflows.map((x) =>
                            x.name === w.name ? { ...x, active: v } : x,
                          ),
                        })
                      }
                    />
                  ))}
                </>
              )}

              <div className="drawer-footnote">
                Stack and workflow changes restart the agent; open chats reconnect into a fresh
                session.{saving ? ' Saving…' : ''}
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  )
}
