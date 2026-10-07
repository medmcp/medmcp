import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  acknowledgeCloudModel,
  fetchCloudModel,
  removeCloudModelConfig,
  saveCloudModelConfig,
  setCloudModelEnabled,
} from '../api'
import type { CloudModelConfig, CloudModelState, CloudProviderPreset } from '../types'
import { Row } from './SettingsControls'
import { XIcon } from './icons'

interface CloudModelWindowProps {
  open: boolean
  onClose: () => void
  /** Called when the model in force may have changed, so the standing warning catches up. */
  onChanged?: () => void
}

/**
 * Run chats on a model hosted outside this machine.
 *
 * A window for the reason the external-MCP one is: the decision it asks for ends
 * the on-premise guarantee, and the consent text needs the room.
 */
export function CloudModelWindow({ open, onClose, onChanged }: CloudModelWindowProps) {
  if (!open) return null
  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="extwin" role="dialog" aria-label="Cloud model">
        <div className="panel-header">
          <span>Cloud model</span>
          <span className="panel-actions">
            <button className="btn-icon" aria-label="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>
        <div className="extwin-body">
          <CloudModelSection onChanged={onChanged} />
        </div>
      </div>
    </>
  )
}

/** The host a configuration sends conversations to, for the places that name it. */
function hostOf(apiBase: string): string {
  try {
    return new URL(apiBase).host
  } catch {
    return apiBase
  }
}

/**
 * The body of the cloud-model window.
 *
 * External MCP sends what one approved tool call passes along; this sends the
 * conversation itself, on every turn, with no prompt in between. So the switch
 * follows the same rules and states the wider consequence: it never acts on its
 * own, the acknowledgement is recorded and enforced server-side, and turning it
 * off clears it so re-arming can never happen in silence.
 */
function CloudModelSection({ onChanged }: { onChanged?: () => void }) {
  const [state, setState] = useState<CloudModelState | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Changes that switch the model in force restart the agent, which takes about
  // a second and a half — long enough to be felt, so it is counted and shown.
  const [inFlight, setInFlight] = useState(0)
  // Chained rather than concurrent: two overlapping restarts would race.
  const queue = useRef<Promise<unknown>>(Promise.resolve())
  const [consenting, setConsenting] = useState(false)
  const [understood, setUnderstood] = useState(false)
  const [editing, setEditing] = useState(false)

  const reload = useCallback(async () => {
    setState(await fetchCloudModel())
  }, [])

  useEffect(() => {
    fetchCloudModel()
      .then(setState)
      .catch((e: unknown) => setError(String(e)))
  }, [])

  /** Queue a change, refetch, tell the caller. The refetch is the authority,
   *  including after a failure, so a rejected change snaps back. */
  const run = useCallback(
    (action: () => Promise<void>, optimistic?: (s: CloudModelState) => CloudModelState) => {
      setError(null)
      if (optimistic) setState((s) => (s ? optimistic(s) : s))
      setInFlight((n) => n + 1)
      queue.current = queue.current
        .then(action)
        .then(reload)
        .then(() => onChanged?.())
        .catch((e: unknown) => {
          setError(e instanceof Error ? e.message : String(e))
          return reload().catch(() => undefined)
        })
        .finally(() => setInFlight((n) => n - 1))
    },
    [reload, onChanged],
  )

  const busy = inFlight > 0

  if (!state) return error ? <div className="panel-error">{error}</div> : null

  const model = state.model
  const preset = model ? state.providers.find((p) => p.id === model.provider) : undefined

  const onToggle = (next: boolean) => {
    // Turning it on explains itself before it acts, every time: the server
    // clears the acknowledgement on disable, so `acknowledged` is only ever true
    // for a feature that is already on.
    if (next && !state.acknowledged) {
      setUnderstood(false)
      setConsenting(true)
      return
    }
    run(
      () => setCloudModelEnabled(next),
      (s) => ({ ...s, enabled: next, active: next && s.active, acknowledged: next }),
    )
  }

  const accept = () =>
    run(async () => {
      await acknowledgeCloudModel()
      await setCloudModelEnabled(true)
      setConsenting(false)
    })

  return (
    <>
      <Row
        label="Use a cloud model"
        hint={
          model
            ? 'Sends chats to the model below instead of the local one.'
            : 'Configure a model below first.'
        }
        checked={state.enabled}
        disabled={!model}
        onChange={onToggle}
      />

      {error && <div className="panel-error">{error}</div>}

      {busy && (
        <div className="settings-row-hint ext-mcp-applying">
          Applying…
        </div>
      )}

      {/* Someone who turns this off to be sure nothing is leaving gets told so. */}
      {!state.enabled && (
        <div className="settings-row-hint ext-mcp-disconnected">
          Off. Nothing is sent outside this machine.
        </div>
      )}
      {state.enabled && model && !state.active && (
        <div className="settings-row-hint ext-mcp-missing-token">
          On, but the API key is missing
          {model.api_key_env && !model.key_managed ? ` ($${model.api_key_env} is not set)` : ''}, so
          chats still run on the local model.
        </div>
      )}

      <div className={`ext-mcp${state.enabled ? '' : ' is-off'}`}>
        {model && !editing ? (
          <div className="ext-mcp-server">
            <div className="ext-mcp-server-text">
              <div className="settings-row-label">{model.model}</div>
              <div className="settings-row-hint ext-mcp-url">{model.api_base}</div>
              <div className="settings-row-hint">
                {preset?.label ?? model.provider}
                {keySource(model)}
              </div>
            </div>
            <div className="ext-mcp-server-actions">
              <button className="btn-text" disabled={busy} onClick={() => setEditing(true)}>
                Edit
              </button>
              <button
                className="btn-text"
                disabled={busy}
                onClick={() =>
                  run(removeCloudModelConfig, (s) => ({
                    ...s,
                    enabled: false,
                    active: false,
                    acknowledged: false,
                    model: null,
                  }))
                }
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <ModelForm
            providers={state.providers}
            current={model}
            live={state.enabled}
            busy={busy}
            onCancel={model ? () => setEditing(false) : undefined}
            onSubmit={(config) =>
              run(async () => {
                await saveCloudModelConfig(config)
                setEditing(false)
              })
            }
          />
        )}
      </div>

      {/* Portalled to the body and lifted `over-window`: this section renders
          inside a window that is its own stacking context, so a backdrop rendered
          in place could not cover the controls behind the dialog. */}
      {consenting &&
        model &&
        createPortal(
          <div className="modal-backdrop over-window" onClick={() => setConsenting(false)}>
            <div
              className="modal ext-mcp-consent"
              role="dialog"
              aria-modal="true"
              aria-labelledby="cloud-model-consent-title"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 id="cloud-model-consent-title">Run chats on a cloud model?</h3>
              <p>
                MedMCP runs on-premise: today the model that reads your chats runs on this
                machine. With this on, <strong>{model.model}</strong> at{' '}
                <strong>{hostOf(model.api_base)}</strong> reads them instead.
              </p>
              <ul>
                <li>
                  Everything in a chat is sent to that provider on every turn: what you type,
                  the files the agent reads, tool results, and the paths and metadata in
                  them, including patient identifiers wherever they appear. No approval
                  prompt covers this; it is not limited to single tool calls.
                </li>
                <li>
                  You are responsible for whether that is lawful for your data, and for the
                  provider's terms, its data retention and any data-processing agreement it
                  requires.
                </li>
                <li>
                  The tool stacks keep running on this machine, and so do tool-call
                  explanations and chat titles. Tool calls still need your approval.
                </li>
                <li>Every turn is billed to the owner of the API key.</li>
                <li>
                  Turning this off returns chats to the local model, and you will be asked to
                  accept this again before it can be switched back on.
                </li>
              </ul>
              <label className="ext-mcp-understood">
                <input
                  type="checkbox"
                  checked={understood}
                  onChange={(e) => setUnderstood(e.target.checked)}
                />
                I understand and accept responsibility for data sent to this provider.
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

/** Where the key comes from, for the one-line summary. */
function keySource(m: CloudModelConfig): string {
  if (m.key_managed) return ' · key stored here'
  if (m.api_key_env) return ` · key from $${m.api_key_env}`
  return ' · no key'
}

function ModelForm({
  providers,
  current,
  live,
  busy,
  onSubmit,
  onCancel,
}: {
  providers: CloudProviderPreset[]
  current: CloudModelConfig | null
  /** The feature is on, so saving changes where open chats go. */
  live: boolean
  busy: boolean
  onSubmit: (config: {
    provider: string
    model: string
    api_base: string
    api_key: string
    api_key_env: string
    compact_threshold: number | null
  }) => void
  onCancel?: () => void
}) {
  const [providerId, setProviderId] = useState(current?.provider ?? providers[0]?.id ?? '')
  const [model, setModel] = useState(current?.model ?? '')
  const [apiBase, setApiBase] = useState(current?.api_base ?? '')
  const [apiKey, setApiKey] = useState('')
  const [envVar, setEnvVar] = useState(
    current && !current.key_managed ? current.api_key_env : '',
  )
  // The key goes in directly; naming a variable the deployment already sets is
  // the path for a site that manages its own secrets, so it folds away.
  const [useEnvVar, setUseEnvVar] = useState(Boolean(current?.api_key_env) && !current?.key_managed)
  // Only sent when changed: left alone, the server applies the provider's default.
  const [budget, setBudget] = useState('')
  const [showBudget, setShowBudget] = useState(false)

  const preset = providers.find((p) => p.id === providerId)
  const fixedEndpoint = preset?.api_base ?? ''
  const endpoint = fixedEndpoint || apiBase.trim().replace(/\/+$/, '')
  // The server keeps a stored key only while the endpoint is unchanged; say so
  // in the field rather than let an empty box look like "no key".
  const keepsKey = Boolean(current?.key_managed) && current?.api_base === endpoint
  const hasKey = useEnvVar ? envVar.trim() !== '' : apiKey.trim() !== '' || keepsKey
  const ready =
    model.trim() !== '' && endpoint !== '' && (hasKey || preset?.key_required === false)

  return (
    <form
      className="ext-mcp-form"
      onSubmit={(e) => {
        e.preventDefault()
        const parsed = Number.parseInt(budget, 10)
        onSubmit({
          provider: providerId,
          model: model.trim(),
          api_base: fixedEndpoint ? '' : apiBase.trim(),
          api_key: useEnvVar ? '' : apiKey,
          api_key_env: useEnvVar ? envVar.trim() : '',
          compact_threshold: showBudget && Number.isFinite(parsed) ? parsed : null,
        })
      }}
    >
      <label>
        Provider
        <select
          className="wf-input"
          value={providerId}
          onChange={(e) => setProviderId(e.target.value)}
        >
          {providers.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      {fixedEndpoint ? (
        <span className="settings-row-hint ext-mcp-url">{fixedEndpoint}</span>
      ) : (
        <label>
          Endpoint
          <input
            className="wf-input"
            value={apiBase}
            placeholder="https://llm-gateway.example.org/v1"
            onChange={(e) => setApiBase(e.target.value)}
          />
          <span className="settings-row-hint">Base URL of an OpenAI-compatible API.</span>
        </label>
      )}
      <label>
        Model
        <input
          className="wf-input"
          value={model}
          list="cloud-model-suggestions"
          placeholder={preset?.models[0] ?? "the provider's model id"}
          onChange={(e) => setModel(e.target.value)}
        />
        <datalist id="cloud-model-suggestions">
          {(preset?.models ?? []).map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </label>
      {useEnvVar ? (
        <label>
          API key env var
          <input
            className="wf-input"
            value={envVar}
            placeholder="LLM_API_KEY"
            onChange={(e) => setEnvVar(e.target.value)}
          />
          <span className="settings-row-hint">
            A variable already set where the agent runs (<code>medmcp.env</code>).
          </span>
        </label>
      ) : (
        <label>
          API key{preset?.key_required === false ? ' (optional)' : ''}
          <input
            className="wf-input"
            type="password"
            value={apiKey}
            autoComplete="off"
            placeholder={keepsKey ? 'leave empty to keep the stored key' : "paste the provider's API key"}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <span className="settings-row-hint">Stored on this machine; never shown again.</span>
        </label>
      )}
      <button
        type="button"
        className="btn-plain ext-mcp-token-mode"
        onClick={() => setUseEnvVar((v) => !v)}
      >
        {useEnvVar ? 'Enter a key instead' : 'Use an environment variable instead'}
      </button>

      {showBudget ? (
        <label>
          Context budget (tokens)
          <input
            className="wf-input"
            type="number"
            min={8000}
            step={1000}
            value={budget}
            placeholder={String(current?.compact_threshold ?? preset?.compact_threshold ?? '')}
            onChange={(e) => setBudget(e.target.value)}
          />
          <span className="settings-row-hint">
            History kept before the agent summarises it. Keep it below the model's context.
          </span>
        </label>
      ) : (
        <button
          type="button"
          className="btn-plain ext-mcp-token-mode"
          onClick={() => setShowBudget(true)}
        >
          Change the context budget
        </button>
      )}

      {live && (
        <span className="settings-row-hint ext-mcp-missing-token">
          The cloud model is on: saving takes effect immediately.
        </span>
      )}
      <div className="ext-mcp-form-actions">
        {onCancel && (
          <button type="button" className="btn-text" onClick={onCancel}>
            Cancel
          </button>
        )}
        <button type="submit" className="btn-primary" disabled={busy || !ready}>
          Save
        </button>
      </div>
    </form>
  )
}
