import { useEffect, useState } from 'react'
import { fetchSettings, saveSettings } from '../api'
import type { SettingsState } from '../types'
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
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    fetchSettings()
      .then((s) => {
        setState(s)
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
                <div className="settings-row-hint">No stacks installed.</div>
              )}
              {state.stacks.map((s) => (
                <Row
                  key={s.name}
                  label={s.name}
                  hint={s.version ? `v${s.version}` : undefined}
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
              ))}

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
