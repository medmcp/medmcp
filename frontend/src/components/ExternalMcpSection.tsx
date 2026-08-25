import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  acknowledgeExternalMcp,
  addExternalServer,
  fetchExternalMcp,
  removeExternalServer,
  replaceExternalToken,
  setExternalMcpEnabled,
  setExternalServerActive,
} from '../api'
import type { ExternalMcpState, ExternalServer } from '../types'
import { Row, Toggle } from './SettingsControls'

/**
 * Advanced-settings section for wiring third-party MCP services into the
 * workspace.
 *
 * The whole product guarantees that data stays on-premise, and this is the one
 * control that breaks that guarantee, so the switch does not act on its own:
 * every time it is turned on it opens a consent dialog that has to be read and
 * accepted. The acknowledgement is recorded server-side and is a precondition
 * there too — the dialog is the explanation, not the enforcement — and turning
 * the feature off clears it again, so re-arming it can never happen in silence.
 */
interface ExternalMcpSectionProps {
  /** Called after any change that may have altered what is connected. */
  onChanged?: () => void
}

export function ExternalMcpSection({ onChanged }: ExternalMcpSectionProps) {
  const [state, setState] = useState<ExternalMcpState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [consenting, setConsenting] = useState(false)
  const [understood, setUnderstood] = useState(false)
  const [adding, setAdding] = useState(false)
  // Name of the server whose token is being replaced, if any.
  const [replacing, setReplacing] = useState<string | null>(null)

  const reload = useCallback(async () => {
    setState(await fetchExternalMcp())
  }, [])

  useEffect(() => {
    fetchExternalMcp()
      .then(setState)
      .catch((e: unknown) => setError(String(e)))
  }, [])

  /** Run a mutation, then refetch — every one of them restarts the agent. */
  const run = useCallback(
    (action: () => Promise<void>) => {
      setBusy(true)
      setError(null)
      action()
        .then(reload)
        .then(() => onChanged?.())
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setBusy(false))
    },
    [reload, onChanged],
  )

  if (!state) return null

  const onToggle = (next: boolean) => {
    // Turning it on explains itself before it acts, every time: the server
    // clears the acknowledgement on disable, so `acknowledged` is only ever true
    // for a feature that is already on.
    if (next && !state.acknowledged) {
      setUnderstood(false)
      setConsenting(true)
      return
    }
    run(() => setExternalMcpEnabled(next))
  }

  const accept = () =>
    run(async () => {
      await acknowledgeExternalMcp()
      await setExternalMcpEnabled(true)
      setConsenting(false)
    })

  return (
    <>
      <Row
        label="External MCP servers"
        hint="Lets the agent use tools hosted outside this machine. Off by default — anything sent to such a server leaves your infrastructure."
        checked={state.enabled}
        onChange={onToggle}
        disabled={busy}
      />

      {error && <div className="panel-error">{error}</div>}

      {/* Someone who turns this off to be sure nothing is leaving gets told so.
          An empty space is not an answer to that question. */}
      {!state.enabled && state.servers.length > 0 && (
        <div className="settings-row-hint ext-mcp-disconnected">
          {state.servers.length} server{state.servers.length === 1 ? '' : 's'} configured, none
          enabled. Nothing is sent outside this machine while this is off.
        </div>
      )}

      {state.enabled && (
        <div className="ext-mcp">
          {state.servers.length === 0 && (
            <div className="settings-row-hint ext-mcp-empty">No external servers yet.</div>
          )}
          {state.servers.map((s) => (
            <div key={s.name} className="ext-mcp-server">
              <div className="ext-mcp-server-text">
                <div className="settings-row-label">{s.name}</div>
                <div className="settings-row-hint ext-mcp-url">{s.url}</div>
                <div className="settings-row-hint">
                  {s.transport}
                  {tokenSource(s)}
                  {s.api_key_env && s.api_key_header ? ` · ${s.api_key_header}` : ''}
                </div>
                {/* Only reachable on the env-var path. Without it the sole symptom
                    is the remote service's 401: vibe sends the request
                    unauthenticated rather than failing where the cause is. */}
                {s.api_key_env && s.token_present === false && (
                  <div className="settings-row-hint ext-mcp-missing-token">
                    ${s.api_key_env} is not set where the agent runs — requests go out with no
                    credential. Add it to medmcp.env and restart, or replace it with a stored
                    token.
                  </div>
                )}
                {replacing === s.name ? (
                  <ReplaceTokenForm
                    busy={busy}
                    onCancel={() => setReplacing(null)}
                    onSubmit={(value) => {
                      run(() => replaceExternalToken(s.name, value))
                      setReplacing(null)
                    }}
                  />
                ) : (
                  <button
                    type="button"
                    className="btn-plain ext-mcp-replace"
                    disabled={busy}
                    onClick={() => setReplacing(s.name)}
                  >
                    {s.token_managed ? 'Replace token' : 'Store a token here instead'}
                  </button>
                )}
              </div>
              <div className="ext-mcp-server-actions">
                <Toggle
                  checked={s.active}
                  disabled={busy}
                  onChange={(v) => run(() => setExternalServerActive(s.name, v))}
                />
                <button
                  className="btn-text"
                  disabled={busy}
                  onClick={() => run(() => removeExternalServer(s.name))}
                >
                  Remove
                </button>
              </div>
            </div>
          ))}

          {adding ? (
            <AddServerForm
              transports={state.transports}
              busy={busy}
              onCancel={() => setAdding(false)}
              onSubmit={(server) =>
                run(async () => {
                  await addExternalServer(server)
                  setAdding(false)
                })
              }
            />
          ) : (
            <button className="btn-text" disabled={busy} onClick={() => setAdding(true)}>
              + Add server
            </button>
          )}
        </div>
      )}

      {/* Portalled to the body: this section renders inside .drawer, which is a
          stacking context (position: fixed + z-index), so a backdrop rendered in
          place cannot paint above the drawer no matter how high its z-index —
          the settings behind the dialog would stay visible and clickable. */}
      {consenting &&
        createPortal(
          <div className="modal-backdrop" onClick={() => setConsenting(false)}>
            <div
              className="modal ext-mcp-consent"
              role="dialog"
              aria-modal="true"
              aria-labelledby="ext-mcp-consent-title"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 id="ext-mcp-consent-title">Connect tools outside this machine?</h3>
              <p>
                MedMCP runs on-premise: today, no imaging data, patient metadata, or results leave
                your infrastructure. Enabling external MCP servers ends that guarantee for the tools
                you add.
              </p>
              <ul>
                <li>
                  Anything the agent passes to an external tool — file contents, paths, patient
                  identifiers, results — is sent to whoever operates that server.
                </li>
                <li>
                  You are responsible for what those services receive, for their terms and security,
                  and for whether that is lawful for your data.
                </li>
                <li>
                  The agent will no longer be told to keep everything on-premise, so it may choose
                  these tools on its own. Each call still needs your approval, and the approval
                  prompt is where you see what is being sent.
                </li>
                <li>
                  Turning this off at any time removes those servers from the agent, and you
                  will be asked to accept this again before it can be switched back on.
                </li>
              </ul>
              <label className="ext-mcp-understood">
                <input
                  type="checkbox"
                  checked={understood}
                  onChange={(e) => setUnderstood(e.target.checked)}
                />
                I understand and accept responsibility for data sent to external servers.
              </label>
              <div className="modal-actions">
                <button className="btn-text" onClick={() => setConsenting(false)}>
                  Cancel
                </button>
                <button className="btn-primary" disabled={!understood || busy} onClick={accept}>
                  Enable
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </>
  )
}

/** Where a server's credential comes from, for the one-line summary. */
function tokenSource(s: ExternalServer): string {
  if (s.token_managed) return ' · token stored here'
  if (s.api_key_env) return ` · token from $${s.api_key_env}`
  return ' · no auth'
}

/** Write-only token entry: the current value is never available to prefill. */
function ReplaceTokenForm({
  busy,
  onSubmit,
  onCancel,
}: {
  busy: boolean
  onSubmit: (token: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState('')
  return (
    <form
      className="ext-mcp-replace-form"
      onSubmit={(e) => {
        e.preventDefault()
        if (value.trim()) onSubmit(value)
      }}
    >
      <input
        className="wf-input"
        type="password"
        value={value}
        autoComplete="off"
        placeholder="new token"
        onChange={(e) => setValue(e.target.value)}
      />
      <button className="btn-plain" type="submit" disabled={busy || !value.trim()}>
        Save
      </button>
      <button className="btn-plain" type="button" onClick={onCancel}>
        Cancel
      </button>
    </form>
  )
}

function AddServerForm({
  transports,
  busy,
  onSubmit,
  onCancel,
}: {
  transports: string[]
  busy: boolean
  onSubmit: (server: {
    name: string
    transport: string
    url: string
    token: string
    api_key_env: string
    api_key_header: string
    api_key_format: string
  }) => void
  onCancel: () => void
}) {
  const [name, setName] = useState('')
  const [transport, setTransport] = useState(transports[0] ?? 'streamable-http')
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [envVar, setEnvVar] = useState('')
  // The token goes in directly; naming a variable the deployment already sets is
  // the path for a site that manages its own secrets, so it folds away.
  const [useEnvVar, setUseEnvVar] = useState(false)
  const [header, setHeader] = useState('')
  const [format, setFormat] = useState('')
  // Most services take a bearer token, so the scheme fields stay folded away
  // until someone needs them.
  const [customScheme, setCustomScheme] = useState(false)

  return (
    <form
      className="ext-mcp-form"
      onSubmit={(e) => {
        e.preventDefault()
        onSubmit({
          name,
          transport,
          url,
          token: useEnvVar ? '' : token,
          api_key_env: useEnvVar ? envVar : '',
          api_key_header: header,
          api_key_format: format,
        })
      }}
    >
      <label>
        Name
        <input
          className="wf-input"
          value={name}
          placeholder="pubmed"
          onChange={(e) => setName(e.target.value)}
        />
      </label>
      <label>
        Transport
        <select
          className="wf-input"
          value={transport}
          onChange={(e) => setTransport(e.target.value)}
        >
          {transports.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>
      <label>
        URL
        <input
          className="wf-input"
          value={url}
          placeholder="https://example.org/mcp"
          onChange={(e) => setUrl(e.target.value)}
        />
      </label>
      {useEnvVar ? (
        <label>
          Token env var (optional)
          <input
            className="wf-input"
            value={envVar}
            placeholder="PUBMED_TOKEN"
            onChange={(e) => setEnvVar(e.target.value)}
          />
          <span className="settings-row-hint">
            The name of a variable already set where the agent runs — for a deployment that
            manages its own secrets. In the container install that means an entry in{' '}
            <code>medmcp.env</code> and a restart.
          </span>
        </label>
      ) : (
        <label>
          Token (optional)
          <input
            className="wf-input"
            type="password"
            value={token}
            autoComplete="off"
            placeholder="paste the service's token"
            onChange={(e) => setToken(e.target.value)}
          />
          {/* Kept out of config.toml and out of every response body; handed to the
              agent process at startup. See settings.load_external_secrets. */}
          <span className="settings-row-hint">
            Stored on this machine and given to the agent when it starts. It is never written to
            the agent's config and never sent back to this page.
          </span>
        </label>
      )}
      <button
        type="button"
        className="btn-plain ext-mcp-token-mode"
        onClick={() => setUseEnvVar((v) => !v)}
      >
        {useEnvVar ? 'Enter a token instead' : 'Use an environment variable instead'}
      </button>

      {/* Server-side these are refused without a token, since they would have
          nothing to send — so only offer them once one is named. */}
      {(useEnvVar ? envVar : token) &&
        (customScheme ? (
          <>
            <label>
              Header
              <input
                className="wf-input"
                value={header}
                placeholder="Authorization"
                onChange={(e) => setHeader(e.target.value)}
              />
            </label>
            <label>
              Value format
              <input
                className="wf-input"
                value={format}
                placeholder="Bearer {token}"
                onChange={(e) => setFormat(e.target.value)}
              />
              <span className="settings-row-hint">
                Must contain <code>{'{token}'}</code>. For an API-key service, try header{' '}
                <code>X-API-Key</code> with format <code>{'{token}'}</code>.
              </span>
            </label>
          </>
        ) : (
          <button type="button" className="btn-text" onClick={() => setCustomScheme(true)}>
            Not a bearer token?
          </button>
        ))}

      <div className="ext-mcp-form-actions">
        <button type="button" className="btn-text" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="btn-primary" disabled={busy || !name || !url}>
          Add
        </button>
      </div>
    </form>
  )
}
