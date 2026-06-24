import { memo, useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
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

/** Outcome of one batch item; `steps` counts its logged step frames. */
type ItemResult = { ok: boolean; error?: string | null; outputs: string[]; steps: number }

/** What the expanded workflow's detail area is currently showing. */
type Mode =
  | { kind: 'view' }
  | { kind: 'rename'; value: string }
  | { kind: 'refine'; value: string }
  | { kind: 'inputs'; values: Record<string, string>; batchInput: string | null }
  | {
      kind: 'preview'
      values: Record<string, string>
      batchInput: string | null
      batchValues: string[]
      steps: ReplayPreviewStep[]
    }
  | {
      kind: 'running'
      name: string
      runs: Record<string, string>[]
      batchInput: string | null
      total: number
      stepsPerItem: number
      startedAt: number
      lastStepAt: number
      items: ItemResult[]
      log: ReplayStepFrame[]
    }
  | {
      kind: 'done'
      name: string
      runs: Record<string, string>[]
      batchInput: string | null
      total: number
      stepsPerItem: number
      startedAt: number
      finishedAt: number
      items: ItemResult[]
      log: ReplayStepFrame[]
      ok: boolean
      error?: string | null
    }

/** Fraction complete: finished items plus the current item's step fraction. */
function runProgress(m: Extract<Mode, { kind: 'running' }>): number {
  if (m.total === 0) return 0
  const prevSteps = m.items.reduce((acc, it) => acc + it.steps, 0)
  const currentFrac =
    m.stepsPerItem > 0 ? Math.min((m.log.length - prevSteps) / m.stepsPerItem, 1) : 0
  return (m.items.length + currentFrac) / m.total
}

/** One run's input bindings per batch item (a single run is a batch of one). */
function buildRuns(
  values: Record<string, string>,
  batchInput: string | null,
  batchValues: string[],
): Record<string, string>[] {
  if (batchInput && batchValues.length > 0) {
    return batchValues.map((v) => ({ ...values, [batchInput]: v }))
  }
  return [values]
}

/** "1m 23s" / "45s" from a millisecond duration. */
function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`
}

function basename(p: string): string {
  const i = p.lastIndexOf('/')
  return i >= 0 ? p.slice(i + 1) : p
}

type Input = { name: string; example: string; description: string }

/** Dropdown label for an input: its name plus whatever hints what it is —
 *  the human description and/or the example filename (e.g. "...T1w.nii.gz"). */
function inputOptionLabel(i: Input): string {
  const bits: string[] = []
  if (i.description) bits.push(i.description)
  if (i.example) bits.push(`e.g. ${basename(i.example)}`)
  return bits.length > 0 ? `${i.name} — ${bits.join(' · ')}` : i.name
}

/** Short label for batch item *i* — its batched input's filename, else "Item N". */
function itemLabel(runs: Record<string, string>[], batchInput: string | null, i: number): string {
  const v = batchInput ? runs[i]?.[batchInput] : undefined
  return v ? basename(v) : `Item ${i + 1}`
}

/** 1-based step number currently in progress for the active item. */
function currentStep(m: Extract<Mode, { kind: 'running' }>): number {
  const prevSteps = m.items.reduce((acc, it) => acc + it.steps, 0)
  return Math.min(m.log.length - prevSteps + 1, m.stepsPerItem)
}

/** Elapsed time, plus a counting-down ETA extrapolated from finished steps. */
function runTiming(m: Extract<Mode, { kind: 'running' }>, now: number): string {
  const elapsed = now - m.startedAt
  let label = `elapsed ${formatDuration(elapsed)}`
  const totalSteps = m.total * m.stepsPerItem
  const doneSteps = m.log.length
  // Average over *finished* step time only (lastStepAt), not total elapsed —
  // otherwise a long in-flight step inflates the average and the ETA climbs.
  // Then count down: remaining = estimated total − elapsed. Rough, hence "~".
  if (doneSteps > 0 && doneSteps < totalSteps) {
    const avgPerStep = (m.lastStepAt - m.startedAt) / doneSteps
    const remaining = avgPerStep * totalSteps - elapsed
    if (remaining > 0) label += ` · ~${formatDuration(remaining)} left`
  }
  return label
}

/** Produced file paths, clickable to open in the viewer when a handler is given. */
function OutputLinks({
  paths,
  onOpenFile,
}: {
  paths: string[]
  onOpenFile?: (path: string) => void
}) {
  return (
    <span className="wf-output-list">
      {paths.map((p) =>
        onOpenFile ? (
          <button
            key={p}
            type="button"
            className="wf-output-link"
            title="Open in viewer"
            onClick={() => onOpenFile(p)}
          >
            {p}
          </button>
        ) : (
          <code key={p}>{p}</code>
        ),
      )}
    </span>
  )
}

/** One replay input field; accepts a file drag from the explorer (and is
 *  auto-filled from the explorer selection by the parent form). */
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

function RunLog({
  log,
  total = 1,
  runs,
  batchInput,
}: {
  log: ReplayStepFrame[]
  total?: number
  runs?: Record<string, string>[]
  batchInput?: string | null
}) {
  const rows: ReactNode[] = []
  let lastItem = -1
  for (const s of log) {
    const item = s.item ?? 0
    if (total > 1 && item !== lastItem) {
      const file = runs && batchInput ? basename(runs[item]?.[batchInput] ?? '') : ''
      rows.push(
        <div key={`item-${item}`} className="wf-runitem">
          Item {item + 1} of {total}
          {file && ` · ${file}`}
        </div>,
      )
      lastItem = item
    }
    rows.push(
      <div key={`${item}-${s.index}`} className={s.ok ? 'wf-runstep ok' : 'wf-runstep fail'}>
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
      </div>,
    )
  }
  return <div className="wf-runlog">{rows}</div>
}

function ProgressBar({ frac, label, active }: { frac: number; label: string; active?: boolean }) {
  const pct = Math.round(Math.min(Math.max(frac, 0), 1) * 100)
  return (
    <div className="wf-progress" title={`${pct}%`}>
      <span className={active ? 'wf-progress-bar active' : 'wf-progress-bar'}>
        <span className="wf-progress-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="wf-progress-label">{label}</span>
    </div>
  )
}

/** Read-only view of the explorer selection feeding the batched input. */
function BatchSelection({ paths }: { paths: string[] }) {
  if (paths.length === 0) {
    return (
      <div className="wf-hint">
        Select the files to process in the explorer — ctrl/shift-click for multiple.
      </div>
    )
  }
  return (
    <div className="wf-batch-selection">
      <div className="wf-hint">
        Using the explorer selection ({paths.length} file{paths.length === 1 ? '' : 's'}):
      </div>
      <div className="wf-batch-chips">
        {paths.map((p) => (
          <code key={p}>{p}</code>
        ))}
      </div>
    </div>
  )
}

/**
 * Personal-workflows panel: save the current chat as a reusable workflow,
 * review/promote/refine drafts, and replay a recipe deterministically (no
 * LLM) on new inputs with a preview + explicit confirmation.
 */
// Memoized: App bumps `fsVersion` on every completed tool call / turn end to
// refresh the explorer and viewer, but this panel doesn't read it — memo keeps
// those bumps (frequent during an agent turn) from re-rendering it, and keeps
// it from re-rendering each frame while a separator is dragged. Its props from
// App are stable refs.
export const WorkflowPanel = memo(function WorkflowPanel({
  distillSessionId,
  onWorkspaceChanged,
  onOpenFile,
  selectedPaths = [],
}: {
  distillSessionId: string | null
  /** Called when a replay step may have written files into the workspace. */
  onWorkspaceChanged?: () => void
  /** Open a produced file in the viewer. */
  onOpenFile?: (path: string) => void
  /** Files multi-selected in the explorer — offered to the batch editor. */
  selectedPaths?: string[]
}) {
  const [workflows, setWorkflows] = useState<WorkflowListEntry[] | null>(null)
  const [enabled, setEnabled] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [detail, setDetail] = useState<WorkflowDetail | null>(null)
  const [mode, setMode] = useState<Mode>({ kind: 'view' })
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [now, setNow] = useState(0)
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

  // Tick once a second while a run is live so the elapsed/ETA readout updates.
  useEffect(() => {
    if (mode.kind !== 'running') return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [mode.kind])

  // Auto-fill the single-run inputs form from the explorer selection — no extra
  // click, mirroring how the batch flow binds the selection. Done by adjusting
  // state during render on a selection change (tracked via prevSel) rather than
  // in an effect — the React-recommended pattern for "adjust state when a prop
  // changes" (https://react.dev/learn/you-might-not-need-an-effect). One input:
  // take the latest selected file (re-selecting updates it). Several inputs:
  // fill the still-empty ones in order. Skipped while batching (the selection is
  // reserved for the batch input then) and when nothing is selected; the no-op
  // guard avoids a render loop. Typing/dragging still overrides.
  const [prevSel, setPrevSel] = useState<string[]>(selectedPaths)
  if (selectedPaths !== prevSel) {
    setPrevSel(selectedPaths)
    if (mode.kind === 'inputs' && mode.batchInput === null && selectedPaths.length > 0) {
      setMode((m) => {
        if (m.kind !== 'inputs' || m.batchInput !== null) return m
        const fields = Object.keys(m.values)
        const values = { ...m.values }
        if (fields.length === 1) {
          values[fields[0]] = selectedPaths[selectedPaths.length - 1]
        } else {
          const empty = fields.filter((n) => !(values[n] ?? '').trim())
          empty.forEach((n, idx) => {
            if (selectedPaths[idx]) values[n] = selectedPaths[idx]
          })
        }
        return fields.some((n) => values[n] !== m.values[n]) ? { ...m, values } : m
      })
    }
  }

  // The workflow whose detail the panel currently wants. Guards against a
  // slow response for a previously clicked row overwriting the current one —
  // the detail's action buttons (Run/Promote/Delete) act on detail.name, so a
  // stale detail under another row's header would target the wrong workflow.
  const detailForRef = useRef<string | null>(null)

  const openDetail = async (name: string) => {
    // Leave an in-flight or finished run alone when toggling rows — its card is
    // pinned at the panel top, so collapsing/switching no longer aborts or hides it.
    if (mode.kind !== 'running' && mode.kind !== 'done') {
      runWs.current?.close()
      setMode({ kind: 'view' })
    }
    if (expanded === name) {
      detailForRef.current = null
      setExpanded(null)
      setDetail(null)
      return
    }
    detailForRef.current = name
    setExpanded(name)
    setDetail(null)
    try {
      const d = await fetchWorkflowDetail(name)
      if (detailForRef.current === name) setDetail(d)
    } catch (e) {
      if (detailForRef.current === name) {
        setError(e instanceof Error ? e.message : String(e))
        setExpanded(null)
      }
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
      detailForRef.current = draft.name
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
      detailForRef.current = null
      setExpanded(null)
      setDetail(null)
      await reload()
    })
  }

  const rename = (name: string, newName: string) =>
    withBusy('Renaming…', async () => {
      const slug = await renameWorkflow(name, newName)
      await reload()
      detailForRef.current = slug
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

  const toPreview = (
    name: string,
    values: Record<string, string>,
    batchInput: string | null,
    batchValues: string[],
  ) =>
    withBusy('Resolving steps…', async () => {
      // Preview resolves the first batch item; the others differ only in the
      // batched input's value, which the preview screen lists alongside.
      const res = await replayPreview(name, buildRuns(values, batchInput, batchValues)[0])
      if (!res.ok) {
        setError(res.error ?? 'this workflow cannot be replayed')
        return
      }
      setMode({ kind: 'preview', values, batchInput, batchValues, steps: res.steps })
    })

  const startRun = (
    name: string,
    runs: Record<string, string>[],
    stepsPerItem: number,
    batchInput: string | null,
    startedAt: number,
  ) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/replay`)
    runWs.current = ws
    setNow(startedAt)
    setMode({
      kind: 'running',
      name,
      runs,
      batchInput,
      total: runs.length,
      stepsPerItem,
      startedAt,
      lastStepAt: startedAt,
      items: [],
      log: [],
    })
    let finished = false
    ws.onopen = () => ws.send(JSON.stringify({ name, runs }))
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
        setMode((m) =>
          m.kind === 'running' ? { ...m, log: [...m.log, step], lastStepAt: Date.now() } : m,
        )
      } else if (frame.type === 'item_result') {
        const res = frame
        setMode((m) => {
          if (m.kind !== 'running') return m
          const prevSteps = m.items.reduce((acc, it) => acc + it.steps, 0)
          const item: ItemResult = {
            ok: res.ok,
            error: res.error,
            outputs: res.outputs,
            steps: m.log.length - prevSteps,
          }
          return { ...m, items: [...m.items, item] }
        })
      } else {
        finished = true
        const result = frame
        setMode((m) =>
          m.kind === 'running'
            ? {
                kind: 'done',
                name: m.name,
                runs: m.runs,
                batchInput: m.batchInput,
                total: m.total,
                stepsPerItem: m.stepsPerItem,
                startedAt: m.startedAt,
                finishedAt: Date.now(),
                items: m.items,
                log: m.log,
                ok: result.ok,
                error: result.error,
              }
            : m,
        )
      }
    }
    ws.onclose = () => {
      if (!finished) {
        setMode((m) =>
          m.kind === 'running'
            ? {
                kind: 'done',
                name: m.name,
                runs: m.runs,
                batchInput: m.batchInput,
                total: m.total,
                stepsPerItem: m.stepsPerItem,
                startedAt: m.startedAt,
                finishedAt: Date.now(),
                items: m.items,
                log: m.log,
                ok: false,
                error: 'run aborted',
              }
            : m,
        )
      }
    }
  }

  const beginRun = (d: WorkflowDetail) => {
    if (d.inputs.length === 0) {
      void toPreview(d.name, {}, null, [])
      return
    }
    setMode({
      kind: 'inputs',
      values: Object.fromEntries(d.inputs.map((i) => [i.name, ''])),
      batchInput: null,
    })
  }

  const renderRunning = (m: Extract<Mode, { kind: 'running' }>) => (
    <div className="wf-form">
      <div className="wf-hint">Replaying… step results appear as they finish.</div>
      <ProgressBar
        frac={runProgress(m)}
        active
        label={
          m.total > 1
            ? `item ${Math.min(m.items.length + 1, m.total)} of ${m.total}`
            : `step ${Math.min(m.log.length + 1, m.stepsPerItem)} of ${m.stepsPerItem}`
        }
      />
      <div className="wf-active">
        {m.items.length < m.total ? (
          m.batchInput ? (
            <span>
              Processing <strong>{basename(m.runs[m.items.length]?.[m.batchInput] ?? '')}</strong> —
              item {m.items.length + 1} of {m.total} · step {currentStep(m)} of {m.stepsPerItem}
            </span>
          ) : (
            <span>
              Running step {currentStep(m)} of {m.stepsPerItem}
            </span>
          )
        ) : (
          <span>Finishing…</span>
        )}
      </div>
      <div className="wf-timing">{runTiming(m, now)}</div>
      <RunLog log={m.log} total={m.total} runs={m.runs} batchInput={m.batchInput} />
      <div className="wf-actions">
        <button className="btn-danger" onClick={() => runWs.current?.close()}>
          <StopSquareIcon size={12} /> Stop
        </button>
      </div>
    </div>
  )

  const renderDone = (m: Extract<Mode, { kind: 'done' }>) => {
    const succeeded = m.items.filter((it) => it.ok).length
    const failedRan = m.items.filter((it) => !it.ok).length
    const notRun = m.total - m.items.length
    const retryRuns = m.runs.filter((_, i) => !m.items[i]?.ok)
    const duration = m.finishedAt - m.startedAt
    return (
      <div className="wf-form">
        {m.total > 1 && (
          <div className="wf-run-summary">
            <span className="tally-ok">{succeeded} done</span>
            {failedRan > 0 && <span className="tally-fail">{failedRan} failed</span>}
            {notRun > 0 && <span className="tally-muted">{notRun} not run</span>}
            <span className="tally-muted">in {formatDuration(duration)}</span>
          </div>
        )}
        <RunLog log={m.log} total={m.total} runs={m.runs} batchInput={m.batchInput} />
        {m.total > 1 && (
          <div className="wf-batch-summary">
            {m.items.map((it, i) => (
              <div key={i} className={it.ok ? 'wf-runstep ok' : 'wf-runstep fail'}>
                <span className={`status-dot ${it.ok ? 'ok' : 'fail'}`} />
                <span>
                  {itemLabel(m.runs, m.batchInput, i)}:{' '}
                  {it.ok ? (
                    it.outputs.length > 0 ? (
                      <>
                        done → <OutputLinks paths={it.outputs} onOpenFile={onOpenFile} />
                      </>
                    ) : (
                      'done'
                    )
                  ) : (
                    (it.error ?? 'failed')
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
        {m.ok ? (
          <div className="wf-result ok">
            {m.total > 1
              ? `Batch complete — all ${m.items.length} item(s) succeeded in ${formatDuration(duration)}.`
              : `Replay complete in ${formatDuration(duration)} — ${m.log.length} step(s) ran.`}
            {m.total === 1 && m.items.some((it) => it.outputs.length > 0) && (
              <div className="wf-outputs">
                Outputs:
                <OutputLinks paths={m.items.flatMap((it) => it.outputs)} onOpenFile={onOpenFile} />
              </div>
            )}
          </div>
        ) : (
          <div className="wf-result fail">
            {m.total > 1 && succeeded > 0
              ? `${succeeded} of ${m.total} item(s) succeeded — ${retryRuns.length} need a retry.`
              : `Replay failed. ${m.error ?? ''}`}
          </div>
        )}
        <div className="wf-actions">
          {retryRuns.length > 0 && (
            <button
              className="btn-primary"
              title="Re-run only the items that failed or didn't finish"
              onClick={() => startRun(m.name, retryRuns, m.stepsPerItem, m.batchInput, Date.now())}
            >
              <RefreshIcon size={12} /> Retry {retryRuns.length}
            </button>
          )}
          <button className="btn-plain" onClick={() => setMode({ kind: 'view' })}>
            Close
          </button>
        </div>
      </div>
    )
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
            // Snapshot the explorer selection now so later clicks in the
            // explorer can't change what the preview confirmed.
            void toPreview(d.name, mode.values, mode.batchInput, selectedPaths)
          }}
        >
          <div className="wf-hint">
            Provide a value for each input — select a file in the explorer to fill it, or type /
            drag a path.
          </div>
          {d.inputs.map((i) =>
            i.name === mode.batchInput ? null : (
              <InputField
                key={i.name}
                name={i.name}
                example={i.example}
                description={i.description}
                value={mode.values[i.name] ?? ''}
                onChange={(v) => setMode({ ...mode, values: { ...mode.values, [i.name]: v } })}
              />
            ),
          )}
          <label className="wf-input-row">
            <span className="wf-input-label">
              Batch over
              <span className="wf-input-desc">
                run once per file selected in the explorer, bound to this input
              </span>
            </span>
            <select
              className="wf-input"
              value={mode.batchInput ?? ''}
              onChange={(e) => setMode({ ...mode, batchInput: e.target.value || null })}
            >
              <option value="">— single run —</option>
              {d.inputs.map((i) => (
                <option key={i.name} value={i.name} title={i.example || undefined}>
                  {inputOptionLabel(i)}
                </option>
              ))}
            </select>
          </label>
          {mode.batchInput &&
            (() => {
              const bi = d.inputs.find((i) => i.name === mode.batchInput)
              return bi && (bi.description || bi.example) ? (
                <div className="wf-hint">
                  Batching <code>{bi.name}</code>
                  {bi.description ? ` — ${bi.description}` : ''}
                  {bi.example ? ` (e.g. ${basename(bi.example)})` : ''}
                </div>
              ) : null
            })()}
          {mode.batchInput && <BatchSelection paths={selectedPaths} />}
          <div className="wf-actions">
            <button
              className="btn-primary"
              type="submit"
              disabled={
                d.inputs.some(
                  (i) => i.name !== mode.batchInput && !(mode.values[i.name] ?? '').trim(),
                ) ||
                (mode.batchInput !== null && selectedPaths.length === 0)
              }
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
          {mode.batchInput && mode.batchValues.length > 0 && (
            <div className="wf-hint">
              Batch: runs {mode.batchValues.length}×, with <code>{mode.batchInput}</code> set to
              each of:
              <span className="wf-batch-chips">
                {mode.batchValues.map((v) => (
                  <code key={v}>{v}</code>
                ))}
              </span>
              The steps below show the first item.
            </div>
          )}
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
            <button
              className="btn-primary"
              onClick={() =>
                startRun(
                  d.name,
                  buildRuns(mode.values, mode.batchInput, mode.batchValues),
                  d.steps.length,
                  mode.batchInput,
                  Date.now(),
                )
              }
            >
              <PlayIcon size={12} /> Run now
            </button>
            <button
              className="btn-plain"
              onClick={() =>
                d.inputs.length > 0
                  ? setMode({ kind: 'inputs', values: mode.values, batchInput: mode.batchInput })
                  : setMode({ kind: 'view' })
              }
            >
              Back
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
        {(mode.kind === 'running' || mode.kind === 'done') && (
          <div className="wf-run-card">
            <div className="wf-run-card-head">
              <PlayIcon size={12} /> {mode.name}
            </div>
            {mode.kind === 'running' ? renderRunning(mode) : renderDone(mode)}
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
})
