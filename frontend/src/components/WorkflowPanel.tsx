import { memo, useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  batchFromPlan,
  deleteWorkflow,
  distillSession,
  exportWorkflow,
  fetchWorkflowDetail,
  fetchWorkflows,
  importWorkflow,
  promoteWorkflow,
  refineWorkflow,
  renameWorkflow,
  replayPreview,
  unpromoteWorkflow,
} from '../api'
import { getDraggedFilePath } from '../dragState'
import type {
  BatchPlanSkip,
  ReplayFrame,
  ReplayPreviewStep,
  StackRequirement,
  WorkflowDetail,
  WorkflowListEntry,
} from '../types'
import { DRAG_PATH_MIME } from '../types'
import {
  BookmarkPlusIcon,
  ChevronRightIcon,
  DownloadIcon,
  PlayIcon,
  RefreshIcon,
  StopSquareIcon,
  UploadIcon,
  XIcon,
} from './icons'

type ReplayStepFrame = Extract<ReplayFrame, { type: 'step' }>

/** Outcome of one batch item; `steps` counts its logged step frames. */
type ItemResult = { ok: boolean; error?: string | null; outputs: string[]; steps: number }

/** What the expanded workflow's detail area is currently showing. */
type Mode =
  | { kind: 'view' }
  | { kind: 'rename'; value: string }
  | { kind: 'refine'; value: string }
  | { kind: 'inputs'; batch: boolean; values: Record<string, string>; rows: Record<string, string>[] }
  | {
      kind: 'preview'
      batch: boolean
      values: Record<string, string>
      rows: Record<string, string>[]
      runs: Record<string, string>[]
      steps: ReplayPreviewStep[]
    }
  | {
      kind: 'running'
      name: string
      runs: Record<string, string>[]
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

/** The runs to execute: a single binding, or every fully-filled table row. Each
 *  run is an independent full input binding ({in_1, in_2, …}). */
function buildRuns(
  st: { batch: boolean; values: Record<string, string>; rows: Record<string, string>[] },
  inputNames: string[],
): Record<string, string>[] {
  if (!st.batch) return [st.values]
  return st.rows.filter((r) => inputNames.every((n) => (r[n] ?? '').trim() !== ''))
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

/** The file basenames of a run's input values, joined (e.g. "subjA.nii + atlas.nii"). */
function runFiles(run: Record<string, string>): string {
  return Object.values(run)
    .filter((v) => v.trim() !== '')
    .map(basename)
    .join(' + ')
}

/** Short label for batch item *i* — its input filenames, else "Item N". */
function itemLabel(runs: Record<string, string>[], i: number): string {
  return runFiles(runs[i] ?? {}) || `Item ${i + 1}`
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

/** The stacks a workflow needs, pinned by image(+digest) or version. */
const REQ_STATUS: Record<
  NonNullable<StackRequirement['status']>,
  { icon: string; cls: string; title: string }
> = {
  ok: { icon: '✓', cls: 'ok', title: 'installed' },
  missing: { icon: '✗', cls: 'missing', title: 'not installed' },
  mismatch: { icon: '⚠', cls: 'mismatch', title: 'installed, but a different version' },
}

function Requirements({ requires }: { requires: StackRequirement[] }) {
  if (requires.length === 0) return null
  const mismatched = requires.filter((r) => r.status === 'mismatch')
  return (
    <div className="wf-requires">
      <div className="wf-requires-title">Requires</div>
      <ul className="wf-requires-list">
        {requires.map((r) => {
          const pin = r.image
            ? `${r.image}${r.digest ? ` @ ${r.digest.slice(0, 19)}…` : ''}`
            : r.version
              ? `v${r.version}`
              : ''
          const s = r.status ? REQ_STATUS[r.status] : null
          return (
            <li key={r.stack}>
              {s && (
                <span className={`wf-req-status ${s.cls}`} title={s.title}>
                  {s.icon}
                </span>
              )}
              <code>{r.stack}</code>
              {pin && <span className="wf-req-pin">{pin}</span>}
            </li>
          )
        })}
      </ul>
      {mismatched.length > 0 && (
        <div className="wf-req-warn">
          A different version of {mismatched.map((r) => r.stack).join(', ')} is installed than this
          workflow was built with — results may not reproduce exactly.
        </div>
      )}
    </div>
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
}: {
  log: ReplayStepFrame[]
  total?: number
  runs?: Record<string, string>[]
}) {
  const rows: ReactNode[] = []
  let lastItem = -1
  for (const s of log) {
    const item = s.item ?? 0
    if (total > 1 && item !== lastItem) {
      const file = runs ? runFiles(runs[item] ?? {}) : ''
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

/** One batch-table cell: a compact path input that also accepts a file drag. */
function BatchCell({
  value,
  example,
  onChange,
}: {
  value: string
  example: string
  onChange: (v: string) => void
}) {
  const [dropReady, setDropReady] = useState(false)
  return (
    <input
      className={dropReady ? 'wf-batch-cell drop-ready' : 'wf-batch-cell'}
      value={value}
      placeholder={example ? `e.g. ${basename(example)}` : 'path'}
      title={value || undefined}
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
  const [expanded, setExpanded] = useState<string | null>(null)
  const [detail, setDetail] = useState<WorkflowDetail | null>(null)
  const [mode, setMode] = useState<Mode>({ kind: 'view' })
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [now, setNow] = useState(0)
  const [planCsv, setPlanCsv] = useState('')
  const [planSkipped, setPlanSkipped] = useState<BatchPlanSkip[] | null>(null)
  const runWs = useRef<WebSocket | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const reload = useCallback(
    () =>
      fetchWorkflows()
        .then((list) => {
          setWorkflows(list)
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
    if (mode.kind === 'inputs' && !mode.batch && selectedPaths.length > 0) {
      setMode((m) => {
        if (m.kind !== 'inputs' || m.batch) return m
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

  const exportFile = (name: string) => withBusy('Exporting…', () => exportWorkflow(name))

  const importFile = (file: File) =>
    withBusy('Importing workflow…', async () => {
      const draft = await importWorkflow(await file.text())
      await reload()
      detailForRef.current = draft.name
      setExpanded(draft.name)
      setDetail(draft)
      setMode({ kind: 'view' })
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

  // Fill the batch table from a plan_batch manifest — the "binding name list"
  // built once from a single-subject prototype, so the roll-out over the whole
  // cohort is a review-and-run, not a row-by-row drag. Flagged (missing/ambiguous)
  // items are surfaced, never silently run.
  const fillFromPlan = (name: string) =>
    withBusy('Building the binding list from the cohort plan…', async () => {
      const res = await batchFromPlan(name, planCsv.trim())
      if (!res.ok) {
        setPlanSkipped(null)
        setError(res.error ?? 'could not read the cohort plan')
        return
      }
      setPlanSkipped(res.skipped)
      if (res.runs.length === 0) {
        setError('no ready (ok) items in the plan — resolve the flagged rows first')
        return
      }
      setMode((m) =>
        m.kind === 'inputs' ? { ...m, batch: true, rows: res.runs } : m,
      )
    })

  const toPreview = (
    name: string,
    st: { batch: boolean; values: Record<string, string>; rows: Record<string, string>[] },
    inputNames: string[],
  ) =>
    withBusy('Resolving steps…', async () => {
      const runs = buildRuns(st, inputNames)
      if (runs.length === 0) {
        setError('add at least one run with every input filled')
        return
      }
      // Preview resolves the first run; a batch's other runs differ only in their
      // input values, which the preview screen lists alongside.
      const res = await replayPreview(name, runs[0])
      if (!res.ok) {
        setError(res.error ?? 'this workflow cannot be replayed')
        return
      }
      setMode({ kind: 'preview', batch: st.batch, values: st.values, rows: st.rows, runs, steps: res.steps })
    })

  const startRun = (
    name: string,
    runs: Record<string, string>[],
    stepsPerItem: number,
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
      void toPreview(d.name, { batch: false, values: {}, rows: [] }, [])
      return
    }
    const empty = Object.fromEntries(d.inputs.map((i) => [i.name, '']))
    setMode({ kind: 'inputs', batch: false, values: empty, rows: [{ ...empty }] })
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
          m.total > 1 ? (
            <span>
              Processing <strong>{itemLabel(m.runs, m.items.length)}</strong> — item{' '}
              {m.items.length + 1} of {m.total} · step {currentStep(m)} of {m.stepsPerItem}
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
      <RunLog log={m.log} total={m.total} runs={m.runs} />
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
        <RunLog log={m.log} total={m.total} runs={m.runs} />
        {m.total > 1 && (
          <div className="wf-batch-summary">
            {m.items.map((it, i) => (
              <div key={i} className={it.ok ? 'wf-runstep ok' : 'wf-runstep fail'}>
                <span className={`status-dot ${it.ok ? 'ok' : 'fail'}`} />
                <span>
                  {itemLabel(m.runs, i)}:{' '}
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
              onClick={() => startRun(m.name, retryRuns, m.stepsPerItem, Date.now())}
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
          <Requirements requires={d.requires} />
          {d.manual_steps.length > 0 && (
            <div className="wf-notice">
              Includes manual step(s) replay can't run — do these by hand:
              <ul className="wf-manual-list">
                {d.manual_steps.map((m, i) => (
                  <li key={i}>
                    <code>{m}</code>
                  </li>
                ))}
              </ul>
            </div>
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
                  title="Mark as reviewed and keep permanently"
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
            <button
              className="btn-plain"
              title="Export as a shareable .workflow.yaml file"
              onClick={() => void exportFile(d.name)}
            >
              <UploadIcon size={12} /> Export
            </button>
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

      {mode.kind === 'inputs' &&
        ((m: Extract<Mode, { kind: 'inputs' }>) => {
          const inputNames = d.inputs.map((i) => i.name)
          const emptyRow = (): Record<string, string> =>
            Object.fromEntries(inputNames.map((n) => [n, '']))
          // Distribute the selected files across rows: every N files (N = number
          // of inputs) fill one row's inputs in order. So a 2-input workflow turns
          // 2 selected files into one run (in_1, in_2), 4 files into two runs, etc.
          // A single-input workflow gets one row per file. Drops blank rows.
          const addFromSelection = () => {
            const k = inputNames.length
            if (k === 0 || selectedPaths.length === 0) return
            const added: Record<string, string>[] = []
            for (let i = 0; i < selectedPaths.length; i += k) {
              const row = emptyRow()
              inputNames.forEach((n, j) => {
                const file = selectedPaths[i + j]
                if (file) row[n] = file
              })
              added.push(row)
            }
            const filled = m.rows.filter((r) => Object.values(r).some((v) => v.trim()))
            setMode({ ...m, rows: [...filled, ...added] })
          }
          const singleIncomplete = inputNames.some((n) => !(m.values[n] ?? '').trim())
          const completeRows = m.rows.filter((r) => inputNames.every((n) => (r[n] ?? '').trim()))
          const previewDisabled = m.batch ? completeRows.length === 0 : singleIncomplete
          return (
            <form
              className="wf-form"
              onSubmit={(e) => {
                e.preventDefault()
                void toPreview(d.name, { batch: m.batch, values: m.values, rows: m.rows }, inputNames)
              }}
            >
              <div className="wf-mode-toggle">
                <button
                  type="button"
                  className={!m.batch ? 'active' : ''}
                  onClick={() => setMode({ ...m, batch: false })}
                >
                  Single run
                </button>
                <button
                  type="button"
                  className={m.batch ? 'active' : ''}
                  onClick={() =>
                    setMode({
                      ...m,
                      batch: true,
                      rows: m.rows.some((r) => Object.values(r).some((v) => v.trim()))
                        ? m.rows
                        : [{ ...m.values }],
                    })
                  }
                >
                  Batch
                </button>
              </div>

              {!m.batch ? (
                <>
                  <div className="wf-hint">
                    Provide a value for each input — select a file in the explorer to fill it, or
                    type / drag a path.
                  </div>
                  {d.inputs.map((i) => (
                    <InputField
                      key={i.name}
                      name={i.name}
                      example={i.example}
                      description={i.description}
                      value={m.values[i.name] ?? ''}
                      onChange={(v) => setMode({ ...m, values: { ...m.values, [i.name]: v } })}
                    />
                  ))}
                </>
              ) : (
                <>
                  <div className="wf-hint">
                    One row per run — each runs the whole workflow on its own inputs. Drag a file
                    into any cell, or select files in the explorer and “Add from selection” (every{' '}
                    {d.inputs.length} selected file{d.inputs.length === 1 ? '' : 's'} fill one row).
                  </div>
                  <div className="wf-batch-table">
                    <div className="wf-batch-row wf-batch-head">
                      <span className="wf-batch-idx">#</span>
                      {d.inputs.map((i) => (
                        <span key={i.name} className="wf-batch-col" title={inputOptionLabel(i)}>
                          {i.name}
                        </span>
                      ))}
                      <span className="wf-batch-rm" />
                    </div>
                    {m.rows.map((row, ri) => (
                      <div key={ri} className="wf-batch-row">
                        <span className="wf-batch-idx">{ri + 1}</span>
                        {d.inputs.map((i) => (
                          <BatchCell
                            key={i.name}
                            value={row[i.name] ?? ''}
                            example={i.example}
                            onChange={(v) =>
                              setMode({
                                ...m,
                                rows: m.rows.map((r, j) =>
                                  j === ri ? { ...r, [i.name]: v } : r,
                                ),
                              })
                            }
                          />
                        ))}
                        <button
                          type="button"
                          className="btn-icon wf-batch-rm"
                          title="Remove row"
                          onClick={() =>
                            setMode({ ...m, rows: m.rows.filter((_, j) => j !== ri) })
                          }
                        >
                          <XIcon size={12} />
                        </button>
                      </div>
                    ))}
                    <div className="wf-batch-tools">
                      <button
                        type="button"
                        className="btn-plain"
                        onClick={() => setMode({ ...m, rows: [...m.rows, emptyRow()] })}
                      >
                        + Add row
                      </button>
                      <button
                        type="button"
                        className="btn-plain"
                        disabled={selectedPaths.length === 0}
                        title={
                          selectedPaths.length === 0
                            ? 'Select files in the explorer first'
                            : `Group the ${selectedPaths.length} selected file(s) into rows of ${d.inputs.length} (one per input)`
                        }
                        onClick={addFromSelection}
                      >
                        Add from selection ({selectedPaths.length})
                      </button>
                    </div>
                    <div className="wf-batch-tools wf-batch-plan">
                      <input
                        className="wf-batch-cell"
                        value={planCsv}
                        placeholder="…/batch_plan.csv (from the cohort plan_batch step)"
                        title={planCsv || undefined}
                        onChange={(e) => setPlanCsv(e.target.value)}
                        onDragOver={(e) => {
                          if (
                            e.dataTransfer.types.includes(DRAG_PATH_MIME) ||
                            getDraggedFilePath() !== null
                          )
                            e.preventDefault()
                        }}
                        onDrop={(e) => {
                          const path =
                            e.dataTransfer.getData(DRAG_PATH_MIME) || getDraggedFilePath()
                          if (path) {
                            e.preventDefault()
                            setPlanCsv(path)
                          }
                        }}
                      />
                      <button
                        type="button"
                        className="btn-plain"
                        disabled={!planCsv.trim()}
                        title="Build one row per cohort subject from a plan_batch manifest"
                        onClick={() => void fillFromPlan(d.name)}
                      >
                        Fill from cohort plan
                      </button>
                    </div>
                  </div>
                  {planSkipped && planSkipped.length > 0 && (
                    <div className="wf-hint wf-plan-skipped">
                      {planSkipped.length} item(s) flagged and left out — resolve them in the plan,
                      then re-fill:
                      <ul className="wf-manual-list">
                        {planSkipped.map((s, i) => (
                          <li key={i}>
                            <code>
                              {s.subject}
                              {s.session ? `/${s.session}` : ''}
                            </code>{' '}
                            — {s.status}: {s.reason}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {completeRows.length > 0 && (
                    <div className="wf-hint">
                      {completeRows.length} run{completeRows.length === 1 ? '' : 's'} ready
                      {m.rows.length > completeRows.length
                        ? ` (${m.rows.length - completeRows.length} incomplete row(s) skipped)`
                        : ''}
                      .
                    </div>
                  )}
                </>
              )}

              <div className="wf-actions">
                <button className="btn-primary" type="submit" disabled={previewDisabled}>
                  Preview steps
                </button>
                <button
                  className="btn-plain"
                  type="button"
                  onClick={() => setMode({ kind: 'view' })}
                >
                  Cancel
                </button>
              </div>
            </form>
          )
        })(mode)}

      {mode.kind === 'preview' && (
        <div className="wf-form">
          <div className="wf-hint">
            Replay will run these exact steps — no LLM, no permission prompts. Review before
            running.
          </div>
          {mode.runs.length > 1 && (
            <div className="wf-hint">
              Batch: {mode.runs.length} runs — the steps below show the first. All runs:
              <span className="wf-batch-chips">
                {mode.runs.map((_run, i) => (
                  <code key={i}>{itemLabel(mode.runs, i)}</code>
                ))}
              </span>
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
              onClick={() => startRun(d.name, mode.runs, d.steps.length, Date.now())}
            >
              <PlayIcon size={12} /> Run now
            </button>
            <button
              className="btn-plain"
              onClick={() =>
                d.inputs.length > 0
                  ? setMode({
                      kind: 'inputs',
                      batch: mode.batch,
                      values: mode.values,
                      rows: mode.rows,
                    })
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
          <button
            className="btn-icon"
            title="Import a shared workflow (.workflow.yaml)"
            disabled={busy !== null}
            onClick={() => fileInputRef.current?.click()}
          >
            <DownloadIcon />
          </button>
          <button className="btn-icon" title="Refresh" onClick={() => void reload()}>
            <RefreshIcon />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".yaml,.yml"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]
              e.target.value = '' // allow re-importing the same file
              if (file) void importFile(file)
            }}
          />
        </span>
      </div>
      <div className="panel-body wf-body">
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
        {workflows !== null && workflows.length === 0 && !busy && (
          <div className="viewer-message">
            No saved workflows yet. Run an analysis in the chat, then click{' '}
            <BookmarkPlusIcon size={12} /> to turn it into a reusable, replayable workflow.
          </div>
        )}
        {workflows?.map((w) => (
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
