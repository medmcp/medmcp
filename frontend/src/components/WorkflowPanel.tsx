import { useCallback, useEffect, useRef, useState } from 'react'
import {
  deleteWorkflow,
  distillSession,
  fetchWorkflowDetail,
  fetchWorkflows,
  promoteWorkflow,
  refineWorkflow,
  renameWorkflow,
  replayPreview,
  unpromoteWorkflow,
} from '../api'
import { getDraggedFilePath } from '../dragState'
import type {
  ReplayFrame,
  ReplayPreviewStep,
  WorkflowDetail,
  WorkflowListEntry,
} from '../types'
import { DRAG_PATH_MIME } from '../types'
import {
  BookmarkPlusIcon,
  ChevronRightIcon,
  PlayIcon,
  RefreshIcon,
  StopSquareIcon,
} from './icons'

type ReplayStepFrame = Extract<ReplayFrame, { type: 'step' }>

/** What the expanded workflow's detail area is currently showing. */
type Mode =
  | { kind: 'view' }
  | { kind: 'rename'; value: string }
  | { kind: 'refine'; value: string }
  | { kind: 'inputs'; values: Record<string, string> }
  | { kind: 'preview'; values: Record<string, string>; steps: ReplayPreviewStep[] }
  | { kind: 'running'; log: ReplayStepFrame[] }
  | { kind: 'done'; log: ReplayStepFrame[]; ok: boolean; error?: string | null; outputs: string[] }

/** One replay input field; accepts a file drag from the explorer. */
function InputField({
  name,
  example,
  description,
  value,
  onChange,
}: {
  name: string
  example: string
  description: string
  value: string
  onChange: (v: string) => void
}) {
  const [dropReady, setDropReady] = useState(false)
  return (
    <label className="wf-input-row">
      <span className="wf-input-label">
        <code>{name}</code>
        {description && <span className="wf-input-desc">{description}</span>}
      </span>
      <input
        className={dropReady ? 'wf-input drop-ready' : 'wf-input'}
        value={value}
        placeholder={example ? `e.g. ${example}` : 'value'}
        onChange={(e) => onChange(e.target.value)}
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes(DRAG_PATH_MIME) || getDraggedFilePath() !== null) {
            e.preventDefault()
            setDropReady(true)
          }
        }}
        onDragLeave={() => setDropReady(false)}
        onDrop={(e) => {
          const path = e.dataTransfer.getData(DRAG_PATH_MIME) || getDraggedFilePath()
          if (path) {
            e.preventDefault()
            onChange(path)
          }
          setDropReady(false)
        }}
      />
    </label>
  )
}

function StepList({ steps }: { steps: { server: string; tool: string }[] }) {
  return (
    <ol className="wf-steps">
      {steps.map((s, i) => (
        <li key={i}>
          <code>
            {s.server}:{s.tool}
          </code>
        </li>
      ))}
    </ol>
  )
}

function RunLog({ log }: { log: ReplayStepFrame[] }) {
  return (
    <div className="wf-runlog">
      {log.map((s) => (
        <div key={s.index} className={s.ok ? 'wf-runstep ok' : 'wf-runstep fail'}>
          <span className={`status-dot ${s.ok ? 'ok' : 'fail'}`} />
          <span>
            {s.index}.{' '}
            <code>
              {s.server}:{s.tool}
            </code>
            {Object.values(s.produced).length > 0 && (
              <span className="wf-produced"> → {Object.values(s.produced).join(', ')}</span>
            )}
            {!s.ok && s.error && <span className="wf-step-error"> {s.error}</span>}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * Personal-workflows panel: save the current chat as a reusable workflow,
 * review/promote/refine drafts, and replay a recipe deterministically (no
 * LLM) on new inputs with a preview + explicit confirmation.
 */
export function WorkflowPanel({
  distillSessionId,
  onWorkspaceChanged,
}: {
  distillSessionId: string | null
  /** Called when a replay step may have written files into the workspace. */
  onWorkspaceChanged?: () => void
}) {
  const [workflows, setWorkflows] = useState<WorkflowListEntry[] | null>(null)
  const [enabled, setEnabled] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [detail, setDetail] = useState<WorkflowDetail | null>(null)
  const [mode, setMode] = useState<Mode>({ kind: 'view' })
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const runWs = useRef<WebSocket | null>(null)

  const reload = useCallback(
    () =>
      fetchWorkflows()
        .then((res) => {
          setEnabled(res.enabled)
          setWorkflows(res.workflows)
          setError(null)
        })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e))),
    [],
  )

  useEffect(() => {
    void reload()
    return () => runWs.current?.close()
  }, [reload])

  const openDetail = async (name: string) => {
    runWs.current?.close()
    setMode({ kind: 'view' })
    if (expanded === name) {
      setExpanded(null)
      setDetail(null)
      return
    }
    setExpanded(name)
    setDetail(null)
    try {
      setDetail(await fetchWorkflowDetail(name))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setExpanded(null)
    }
  }

  const withBusy = async (label: string, fn: () => Promise<void>) => {
    setBusy(label)
    setError(null)
    try {
      await fn()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const saveChat = () =>
    withBusy('Distilling the chat into a workflow (local model)…', async () => {
      if (!distillSessionId) return
      const draft = await distillSession(distillSessionId)
      await reload()
      setExpanded(draft.name)
      setDetail(draft)
      setMode({ kind: 'view' })
    })

  const promote = (name: string) =>
    withBusy('Promoting…', async () => {
      await promoteWorkflow(name)
      await reload()
      setDetail(await fetchWorkflowDetail(name))
    })

  const unpromote = (name: string) =>
    withBusy('Moving back to draft…', async () => {
      await unpromoteWorkflow(name)
      await reload()
      setDetail(await fetchWorkflowDetail(name))
    })

  const remove = (name: string) => {
    if (!window.confirm(`Delete workflow "${name}"? This cannot be undone.`)) return
    void withBusy('Deleting…', async () => {
      await deleteWorkflow(name)
      setExpanded(null)
      setDetail(null)
      await reload()
    })
  }

  const rename = (name: string, newName: string) =>
    withBusy('Renaming…', async () => {
      const slug = await renameWorkflow(name, newName)
      await reload()
      setExpanded(slug)
      setDetail(await fetchWorkflowDetail(slug))
      setMode({ kind: 'view' })
    })

  const refine = (name: string, instruction: string) =>
    withBusy('Refining the description (local model)…', async () => {
      await refineWorkflow(name, instruction)
      await reload()
      setDetail(await fetchWorkflowDetail(name))
      setMode({ kind: 'view' })
    })

  const toPreview = (name: string, values: Record<string, string>) =>
    withBusy('Resolving steps…', async () => {
      const res = await replayPreview(name, values)
      if (!res.ok) {
        setError(res.error ?? 'this workflow cannot be replayed')
        return
      }
      setMode({ kind: 'preview', values, steps: res.steps })
    })

  const startRun = (name: string, values: Record<string, string>) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/replay`)
    runWs.current = ws
    setMode({ kind: 'running', log: [] })
    let finished = false
    ws.onopen = () => ws.send(JSON.stringify({ name, inputs: values }))
    ws.onmessage = (ev: MessageEvent<string>) => {
      let frame: ReplayFrame
      try {
        frame = JSON.parse(ev.data) as ReplayFrame
      } catch {
        return
      }
      onWorkspaceChanged?.()
      if (frame.type === 'step') {
        const step = frame
        setMode((m) => (m.kind === 'running' ? { ...m, log: [...m.log, step] } : m))
      } else {
        finished = true
        const result = frame
        setMode((m) => ({
          kind: 'done',
          log: m.kind === 'running' ? m.log : [],
          ok: result.ok,
          error: result.error,
          outputs: result.outputs ?? [],
        }))
      }
    }
    ws.onclose = () => {
      if (!finished) {
        setMode((m) =>
          m.kind === 'running'
            ? { kind: 'done', log: m.log, ok: false, error: 'run aborted', outputs: [] }
            : m,
        )
      }
    }
  }

  const beginRun = (d: WorkflowDetail) => {
    if (d.inputs.length === 0) {
      void toPreview(d.name, {})
      return
    }
    setMode({
      kind: 'inputs',
      values: Object.fromEntries(d.inputs.map((i) => [i.name, ''])),
    })
  }

  const renderDetail = (d: WorkflowDetail) => (
    <div className="wf-detail">
      {d.description && <div className="wf-desc">{d.description}</div>}
      {mode.kind === 'view' && (
        <>
          {d.steps.length > 0 ? (
            <StepList steps={d.steps} />
          ) : (
            <div className="wf-notice">
              This chat produced no replayable tool steps — the workflow is empty.
            </div>
          )}
          {!d.replayable && d.replay_error && d.steps.length > 0 && (
            <div className="wf-notice">Can't replay: {d.replay_error}</div>
          )}
          <div className="wf-actions">
            <button
              className="btn-primary wf-run-btn"
              disabled={!d.replayable}
              title={
                d.replayable
                  ? 'Replay these exact steps on new inputs — no LLM involved'
                  : (d.replay_error ?? '')
              }
              onClick={() => beginRun(d)}
            >
              <PlayIcon size={12} /> Run
            </button>
            {d.kind === 'draft' ? (
              <>
                <button
                  className="btn-plain"
                  title="Keep permanently; loads as a skill in new chat sessions"
                  onClick={() => void promote(d.name)}
                >
                  Promote
                </button>
                <button
                  className="btn-plain"
                  onClick={() => setMode({ kind: 'rename', value: d.name })}
                >
                  Rename
                </button>
                <button
                  className="btn-plain"
                  title="Rewrite the description with a plain-language instruction"
                  onClick={() => setMode({ kind: 'refine', value: '' })}
                >
                  Refine
                </button>
              </>
            ) : (
              <button
                className="btn-plain"
                title="Move back to draft for renaming/refining"
                onClick={() => void unpromote(d.name)}
              >
                Edit
              </button>
            )}
            <button className="btn-plain wf-delete" onClick={() => remove(d.name)}>
              Delete
            </button>
          </div>
        </>
      )}

      {mode.kind === 'rename' && (
        <form
          className="wf-form"
          onSubmit={(e) => {
            e.preventDefault()
            if (mode.value.trim()) void rename(d.name, mode.value.trim())
          }}
        >
          <input
            className="wf-input"
            autoFocus
            value={mode.value}
            onChange={(e) => setMode({ kind: 'rename', value: e.target.value })}
          />
          <div className="wf-actions">
            <button className="btn-primary" type="submit">
              Rename
            </button>
            <button className="btn-plain" type="button" onClick={() => setMode({ kind: 'view' })}>
              Cancel
            </button>
          </div>
        </form>
      )}

      {mode.kind === 'refine' && (
        <form
          className="wf-form"
          onSubmit={(e) => {
            e.preventDefault()
            if (mode.value.trim()) void refine(d.name, mode.value.trim())
          }}
        >
          <textarea
            className="wf-input"
            autoFocus
            rows={3}
            placeholder="e.g. Mention that the input must be a T1-weighted scan"
            value={mode.value}
            onChange={(e) => setMode({ kind: 'refine', value: e.target.value })}
          />
          <div className="wf-actions">
            <button className="btn-primary" type="submit">
              Refine
            </button>
            <button className="btn-plain" type="button" onClick={() => setMode({ kind: 'view' })}>
              Cancel
            </button>
          </div>
        </form>
      )}

      {mode.kind === 'inputs' && (
        <form
          className="wf-form"
          onSubmit={(e) => {
            e.preventDefault()
            void toPreview(d.name, mode.values)
          }}
        >
          <div className="wf-hint">
            Provide a value for each input — type it or drag a file in from the explorer.
          </div>
          {d.inputs.map((i) => (
            <InputField
              key={i.name}
              name={i.name}
              example={i.example}
              description={i.description}
              value={mode.values[i.name] ?? ''}
              onChange={(v) => setMode({ kind: 'inputs', values: { ...mode.values, [i.name]: v } })}
            />
          ))}
          <div className="wf-actions">
            <button
              className="btn-primary"
              type="submit"
              disabled={d.inputs.some((i) => !(mode.values[i.name] ?? '').trim())}
            >
              Preview steps
            </button>
            <button className="btn-plain" type="button" onClick={() => setMode({ kind: 'view' })}>
              Cancel
            </button>
          </div>
        </form>
      )}

      {mode.kind === 'preview' && (
        <div className="wf-form">
          <div className="wf-hint">
            Replay will run these exact steps — no LLM, no permission prompts. Review before
            running.
          </div>
          <ol className="wf-steps">
            {mode.steps.map((s) => (
              <li key={s.index}>
                <code>
                  {s.server}:{s.tool}
                </code>
                <pre className="wf-args">{JSON.stringify(s.arguments)}</pre>
              </li>
            ))}
          </ol>
          <div className="wf-actions">
            <button className="btn-primary" onClick={() => startRun(d.name, mode.values)}>
              <PlayIcon size={12} /> Run now
            </button>
            <button
              className="btn-plain"
              onClick={() =>
                d.inputs.length > 0
                  ? setMode({ kind: 'inputs', values: mode.values })
                  : setMode({ kind: 'view' })
              }
            >
              Back
            </button>
          </div>
        </div>
      )}

      {mode.kind === 'running' && (
        <div className="wf-form">
          <div className="wf-hint">Replaying… step results appear as they finish.</div>
          <RunLog log={mode.log} />
          <div className="wf-actions">
            <button className="btn-danger" onClick={() => runWs.current?.close()}>
              <StopSquareIcon size={12} /> Stop
            </button>
          </div>
        </div>
      )}

      {mode.kind === 'done' && (
        <div className="wf-form">
          <RunLog log={mode.log} />
          {mode.ok ? (
            <div className="wf-result ok">
              Replay complete — {mode.log.length} step(s) ran.
              {mode.outputs.length > 0 && (
                <div className="wf-outputs">
                  Outputs:
                  {mode.outputs.map((o) => (
                    <code key={o}>{o}</code>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="wf-result fail">Replay failed. {mode.error ?? ''}</div>
          )}
          <div className="wf-actions">
            <button className="btn-plain" onClick={() => setMode({ kind: 'view' })}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Workflows</span>
        <span className="panel-actions">
          <button
            className="btn-icon"
            title={
              distillSessionId
                ? 'Save the current chat as a reusable workflow'
                : 'Send a message in the chat first'
            }
            disabled={!distillSessionId || busy !== null}
            onClick={() => void saveChat()}
          >
            <BookmarkPlusIcon />
          </button>
          <button className="btn-icon" title="Refresh" onClick={() => void reload()}>
            <RefreshIcon />
          </button>
        </span>
      </div>
      <div className="panel-body wf-body">
        {!enabled && (
          <div className="viewer-message">
            Personal workflows are turned off — enable them in Settings.
          </div>
        )}
        {error && <div className="panel-error">{error}</div>}
        {busy && (
          <div className="wf-busy">
            <span className="status-dot busy" /> {busy}
          </div>
        )}
        {enabled && workflows !== null && workflows.length === 0 && !busy && (
          <div className="viewer-message">
            No saved workflows yet. Run an analysis in the chat, then click{' '}
            <BookmarkPlusIcon size={12} /> to turn it into a reusable, replayable workflow.
          </div>
        )}
        {enabled &&
          workflows?.map((w) => (
            <div key={w.name} className="wf-item">
              <button className="wf-row" onClick={() => void openDetail(w.name)}>
                <ChevronRightIcon
                  size={12}
                  className={expanded === w.name ? 'wf-chevron open' : 'wf-chevron'}
                />
                <span className="wf-name">{w.name}</span>
                <span className={`wf-chip wf-chip-${w.kind}`}>{w.kind}</span>
              </button>
              {expanded === w.name &&
                (detail ? (
                  renderDetail(detail)
                ) : (
                  <div className="wf-busy">
                    <span className="status-dot busy" /> Loading…
                  </div>
                ))}
            </div>
          ))}
      </div>
    </div>
  )
}
