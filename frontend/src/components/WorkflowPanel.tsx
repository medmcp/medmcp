import { memo, useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  batchFromPlan,
  deleteWorkflow,
  distillSession,
  exportWorkflow,
  fetchRuns,
  fetchWorkflowDetail,
  fetchWorkflows,
  fetchWorkspaceRoot,
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
  ReplayPreviewResult,
  RunStatus,
    StackRequirement,
  WorkflowDetail,
  WorkflowListEntry,
} from '../types'
import { DRAG_PATH_MIME } from '../types'
import {
  BookmarkPlusIcon,
  ChevronRightIcon,
  DownloadIcon,
  FileIcon,
  PlayIcon,
  RefreshIcon,
  StopSquareIcon,
  TableIcon,
  XIcon,
} from './icons'

type ReplayStepFrame = Extract<ReplayFrame, { type: 'step' }>

/** localStorage key holding the run the page is following (reattached on reload). */
const ACTIVE_RUN_KEY = 'medmcp.activeRun'

/** Outcome of one batch item; `steps` counts its logged step frames. */
type ItemResult = { ok: boolean; error?: string | null; outputs: string[]; steps: number }

/** One row of the run editor: one value per required input. */
type Row = { values: Record<string, string> }

/** What the expanded workflow's detail area is currently showing. */
type Mode =
  | { kind: 'view' }
  | { kind: 'rename'; value: string }
  | { kind: 'refine'; value: string }
  | { kind: 'run'; source: RunSource; rows: Row[]; resultsDir: string }

/** Where a run's inputs come from: files picked by hand (the default, prefilled
 *  from the explorer selection), or a cohort plan. The two are different jobs
 *  (a few scans vs. a reviewed roll-out), so only the chosen editor is shown. */
type RunSource = 'files' | 'plan'

const SOURCES: { value: RunSource; icon: ReactNode; title: string; hint: string }[] = [
  { value: 'files', icon: <FileIcon size={14} />, title: 'Files', hint: 'Pick scans by hand' },
  {
    value: 'plan',
    icon: <TableIcon size={14} />,
    title: 'Cohort plan',
    hint: 'One row per subject',
  },
]

/** The run the panel is following: live, or finished and still on screen. */
type RunState = {
  runId: string
  workflow: string
  runs: Record<string, string>[]
  total: number
  stepsPerItem: number
  startedAt: number
  lastStepAt: number
  items: ItemResult[]
  log: ReplayStepFrame[]
  /** The tool in flight right now, with when it started. */
  current: { item: number; index: number; server: string; tool: string; since: number } | null
  status: RunStatus
  error?: string | null
  finishedAt?: number
  /** False while the socket is down and the panel is trying to get back on. */
  attached: boolean
}

/** Fraction complete: finished items plus the current item's step fraction. */
function runProgress(r: RunState): number {
  if (r.total === 0) return 0
  const prevSteps = r.items.reduce((acc, it) => acc + it.steps, 0)
  const currentFrac =
    r.stepsPerItem > 0 ? Math.min((r.log.length - prevSteps) / r.stepsPerItem, 1) : 0
  return (r.items.length + currentFrac) / r.total
}

/** "1m 23s" / "45s" from a millisecond duration. */
function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`
}

function basename(p: string): string {
  const i = p.lastIndexOf('/')
  return i >= 0 ? p.slice(i + 1) : p
}

/** A tool's absolute output path as a workspace path (what the viewer opens),
 *  or null when it lies outside the workspace. */
function workspacePath(p: string, root: string): string | null {
  if (!root) return null
  const prefix = root.endsWith('/') ? root : `${root}/`
  return p.startsWith(prefix) ? p.slice(prefix.length) : null
}

/** Shorten a value for display: absolute workspace paths become relative. */
function display(v: unknown, root: string): string {
  if (typeof v === 'string') return workspacePath(v, root) ?? v
  return JSON.stringify(v)
}

/** The first line of a tool error — the rest is stack/validation detail. */
function firstLine(err: string): string {
  const line = err.split('\n').find((l) => l.trim() !== '') ?? err
  return line.length > 160 ? `${line.slice(0, 157)}…` : line
}

/** How long a finished step took, from the timestamps on its frame. */
function stepDuration(s: ReplayStepFrame): number | null {
  if (!s.started_at || !s.finished_at) return null
  const a = Date.parse(s.started_at)
  const b = Date.parse(s.finished_at)
  return Number.isNaN(a) || Number.isNaN(b) ? null : Math.max(0, b - a)
}

type Input = { name: string; example: string; description: string; default?: string }

/** A readable column title for an input: the argument it feeds (from the
 *  distilled "the X for server:tool" description), a short description the user
 *  wrote, else the bare placeholder name. */
function inputTitle(i: Input): string {
  const m = /^the (\S+) for /.exec(i.description)
  if (m) return m[1]
  const d = i.description.trim()
  return d && d.length <= 32 ? d : i.name
}

/** The directory prefix every path in *paths* shares (with trailing slash), or ''. */
function commonDir(paths: string[]): string {
  if (paths.length === 0) return ''
  let prefix = paths[0].slice(0, paths[0].lastIndexOf('/') + 1)
  for (const p of paths) {
    while (prefix && !p.startsWith(prefix)) {
      prefix = prefix.slice(0, prefix.lastIndexOf('/', prefix.length - 2) + 1)
    }
  }
  return prefix
}

/** Labels that tell the items of a batch apart: each item's input paths,
 *  workspace-relative, minus whatever directory every item shares. A cohort of
 *  `patient_NN/visit_01/t1n_3d.nii.gz` reads as `patient_01/visit_01/t1n_3d.nii.gz`,
 *  and files that all sit in one folder read as bare names. */
function itemLabels(runs: Record<string, string>[], root: string): string[] {
  const rel = runs.map((run) =>
    Object.values(run)
      .filter((v) => v.trim() !== '')
      .map((v) => workspacePath(v, root) ?? v),
  )
  const shared = commonDir(rel.flat())
  return rel.map((paths, i) =>
    paths.length > 0
      ? paths.map((p) => (p.startsWith(shared) ? p.slice(shared.length) : p)).join(' + ')
      : `Item ${i + 1}`,
  )
}

/** Elapsed time, plus a counting-down ETA extrapolated from finished steps. */
function runTiming(r: RunState, now: number): string {
  const elapsed = now - r.startedAt
  let label = `elapsed ${formatDuration(elapsed)}`
  const totalSteps = r.total * r.stepsPerItem
  const doneSteps = r.log.length
  // Average over *finished* step time only (lastStepAt), not total elapsed —
  // otherwise a long in-flight step inflates the average and the ETA climbs.
  if (doneSteps > 0 && doneSteps < totalSteps) {
    const avgPerStep = (r.lastStepAt - r.startedAt) / doneSteps
    const remaining = avgPerStep * totalSteps - elapsed
    if (remaining > 0) label += ` · ~${formatDuration(remaining)} left`
  }
  return label
}

/** Produced files as small chips: the file name, full path on hover, and a
 *  click opens it in the viewer when it lives inside the workspace. */
function FileChips({
  paths,
  root,
  onOpenFile,
}: {
  paths: string[]
  root: string
  onOpenFile?: (path: string) => void
}) {
  if (paths.length === 0) return null
  return (
    <span className="wf-chips">
      {paths.map((p) => {
        const rel = workspacePath(p, root)
        const openable = rel !== null && !!onOpenFile
        return (
          <button
            key={p}
            type="button"
            className={openable ? 'wf-file-chip' : 'wf-file-chip static'}
            title={openable ? `${rel} — open in viewer` : p}
            disabled={!openable}
            onClick={() => rel !== null && onOpenFile?.(rel)}
          >
            {basename(p)}
          </button>
        )
      })}
    </span>
  )
}

/** A tool error: its first line, with the rest behind a toggle. */
function ErrorText({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const head = firstLine(text)
  const more = text.trim() !== head.replace(/…$/, '') && text.includes('\n')
  return (
    <span className="wf-step-error">
      {head}
      {(more || head.endsWith('…')) && (
        <>
          {' '}
          <button type="button" className="btn-text wf-more" onClick={() => setOpen((v) => !v)}>
            {open ? 'less' : 'more'}
          </button>
          {open && <pre className="wf-error-full">{text}</pre>}
        </>
      )}
    </span>
  )
}

/** A path input that also accepts a file drag from the explorer. */
function PathCell({
  value,
  placeholder,
  onChange,
  onFocus,
  className = 'wf-batch-cell',
}: {
  value: string
  placeholder: string
  onChange: (v: string) => void
  onFocus?: () => void
  className?: string
}) {
  const [dropReady, setDropReady] = useState(false)
  return (
    <input
      className={dropReady ? `${className} drop-ready` : className}
      value={value}
      placeholder={placeholder}
      title={value || undefined}
      onFocus={onFocus}
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
      <span className="wf-requires-title">Requires</span>
      <ul className="wf-requires-list">
        {requires.map((r) => {
          const pin = r.image
            ? `${r.image}${r.digest ? ` @ ${r.digest.slice(0, 19)}…` : ''}`
            : r.version
              ? `v${r.version}`
              : ''
          const s = r.status ? REQ_STATUS[r.status] : null
          return (
            <li key={r.stack} title={pin ? `${s?.title ?? ''} · ${pin}`.replace(/^ · /, '') : s?.title}>
              {s && <span className={`wf-req-status ${s.cls}`}>{s.icon}</span>}
              <code>{r.stack}</code>
            </li>
          )
        })}
      </ul>
      {mismatched.length > 0 && (
        <div className="wf-req-warn">
          A different version of {mismatched.map((r) => r.stack).join(', ')} is installed than this
          workflow was built with. Results may not reproduce exactly.
        </div>
      )}
    </div>
  )
}

function StepList({ steps }: { steps: { server: string; tool: string }[] }) {
  return (
    <ol className="wf-steps">
      {steps.map((s, i) => (
        <li key={i} title={`${s.server}:${s.tool}`}>
          <code>{s.tool}</code>
        </li>
      ))}
    </ol>
  )
}

/** The steps of one item, one line each: tick, tool, how long. */
function StepRows({
  steps,
  current,
  now,
  stepsPerItem,
}: {
  steps: ReplayStepFrame[]
  current?: RunState['current']
  now?: number
  stepsPerItem?: number
}) {
  const rows = steps.map((s) => {
    const dur = stepDuration(s)
    return (
      <div key={s.index} className={s.ok ? 'wf-steprow' : 'wf-steprow fail'}>
        <span className={`status-dot ${s.ok ? 'ok' : 'fail'}`} />
        <code>{s.tool}</code>
        <span className="wf-recent-muted">
          {dur !== null ? formatDuration(dur) : ''}
        </span>
        {!s.ok && s.error && <ErrorText text={s.error} />}
      </div>
    )
  })
  if (current) {
    rows.push(
      <div key={`live-${current.index}`} className="wf-steprow live">
        <span className="status-dot busy" />
        <code>{current.tool}</code>
        <span className="wf-recent-muted">
          {now !== undefined ? formatDuration(now - current.since) : 'running'}
        </span>
      </div>,
    )
  }
  const remaining = (stepsPerItem ?? 0) - steps.length - (current ? 1 : 0)
  if (remaining > 0 && current) {
    rows.push(
      <div key="queued" className="wf-steprow queued">
        <span className="status-dot" />
        <span className="wf-recent-muted">
          {remaining} more step{remaining === 1 ? '' : 's'}
        </span>
      </div>,
    )
  }
  return <div className="wf-steprows">{rows}</div>
}

/** One line per batch item — finished ones with outputs, the current one with
 *  its step, the rest folded into a count. Click a line for its steps. */
function ItemRows({
  r,
  now,
  root,
  onOpenFile,
}: {
  r: RunState
  now: number
  root: string
  onOpenFile?: (path: string) => void
}) {
  const [open, setOpen] = useState<number | null>(null)
  const labels = itemLabels(r.runs, root)
  const rows: ReactNode[] = []
  let queued = 0
  for (let i = 0; i < r.total; i++) {
    const it = r.items[i]
    const steps = r.log.filter((s) => (s.item ?? 0) === i)
    const isCurrent = !it && (r.current?.item === i || (steps.length > 0 && r.status === 'running'))
    if (!it && !isCurrent) {
      queued += 1
      continue
    }
    const dot = it ? (it.ok ? 'ok' : 'fail') : 'busy'
    const expanded = open === i
    rows.push(
      <div key={i} className="wf-itemrow-wrap">
        <button
          type="button"
          className={`wf-itemrow ${dot}`}
          onClick={() => setOpen(expanded ? null : i)}
          title={expanded ? 'Hide steps' : 'Show steps'}
        >
          <span className={`status-dot ${dot}`} />
          <span className="wf-itemrow-label">{labels[i]}</span>
          <span className="wf-recent-muted wf-itemrow-note">
            {it
              ? it.ok
                ? it.outputs.length > 0
                  ? `${it.outputs.length} file${it.outputs.length === 1 ? '' : 's'}`
                  : ''
                : expanded
                  ? ''
                  : firstLine(it.error ?? 'failed')
              : r.current && r.current.item === i
                ? `step ${r.current.index} of ${r.stepsPerItem} · ${r.current.tool}`
                : 'starting'}
          </span>
        </button>
        {expanded && it?.ok && (
          <FileChips paths={it.outputs} root={root} onOpenFile={onOpenFile} />
        )}
        {expanded && (
          <StepRows
            steps={steps}
            current={!it && r.current?.item === i ? r.current : null}
            now={now}
            stepsPerItem={r.stepsPerItem}
          />
        )}
      </div>,
    )
  }
  if (queued > 0) {
    rows.push(
      <div key="queued" className="wf-itemrow queued">
        <span className="status-dot" />
        <span className="wf-recent-muted">
          {queued} more item{queued === 1 ? '' : 's'} {r.status === 'running' ? 'queued' : 'not run'}
        </span>
      </div>,
    )
  }
  return <div className="wf-itemrows">{rows}</div>
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

function emptyRow(names: string[]): Row {
  return { values: Object.fromEntries(names.map((n) => [n, ''])) }
}

/** Put newly selected files into the editor: the first one into the cell the
 *  user last focused (if it is still there), the rest into the empty cells in
 *  row order, appending rows when none are left. Rows are never removed or
 *  reordered by a selection — "add a row, then click the file" must land the
 *  file in that row. */
function placeFiles(
  rows: Row[],
  names: string[],
  files: string[],
  focused: { row: number; name: string } | null,
): Row[] {
  if (names.length === 0 || files.length === 0) return rows
  const next = rows.map((r) => ({ values: { ...r.values } }))
  const queue = [...files]
  if (focused && next[focused.row] && names.includes(focused.name)) {
    next[focused.row].values[focused.name] = queue.shift() ?? ''
  }
  for (const row of next) {
    for (const n of names) {
      if (queue.length === 0) return next
      if (!(row.values[n] ?? '').trim()) row.values[n] = queue.shift() ?? ''
    }
  }
  while (queue.length > 0) {
    const row = emptyRow(names)
    for (const n of names) {
      if (queue.length === 0) break
      row.values[n] = queue.shift() ?? ''
    }
    next.push(row)
  }
  return next
}

/**
 * Personal-workflows panel: save the current chat as a reusable workflow,
 * review/promote/refine drafts, and run a recipe deterministically (no LLM)
 * on new inputs. A run is a server-side job: it keeps going if this page goes
 * away, and the page reattaches to it on reload.
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
  /** Files multi-selected in the explorer — offered to the run editor. */
  selectedPaths?: string[]
}) {
  const [workflows, setWorkflows] = useState<WorkflowListEntry[] | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [detail, setDetail] = useState<WorkflowDetail | null>(null)
  const [mode, setMode] = useState<Mode>({ kind: 'view' })
  const [run, setRun] = useState<RunState | null>(null)
  const [preview, setPreview] = useState<{ key: string; result: ReplayPreviewResult } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const [planCsv, setPlanCsv] = useState('')
  const [planSkipped, setPlanSkipped] = useState<BatchPlanSkip[] | null>(null)
  const [showSteps, setShowSteps] = useState(false)
  const [showResultsDir, setShowResultsDir] = useState(false)
  // Absolute workspace root: tool outputs come back absolute, the viewer wants
  // workspace-relative, and nobody wants to read the host prefix on every chip.
  const [root, setRoot] = useState('')
  // The editor cell the user focused last: a file clicked in the explorer goes
  // there first. State rather than a ref because it is read while adjusting
  // state during render (below), where refs are off limits.
  const [focusedCell, setFocusedCell] = useState<{ row: number; name: string } | null>(null)
  const runWs = useRef<WebSocket | null>(null)
  // A dropped socket on a live run: the id to get back on, and a counter so a
  // second drop re-arms the timer even for the same id.
  const [reattach, setReattach] = useState<{ id: string; attempt: number } | null>(null)
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

  // ── The run socket ──
  //
  // One socket per followed run. `first` is either a start request or an attach;
  // the server answers both with the same frame stream. Closing the socket never
  // stops the run — Stop is an explicit cancel message — so a drop while the run
  // is live is followed by a reattach.
  const followRun = useCallback(
    (first: Record<string, unknown>) => {
      runWs.current?.close()
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/replay`)
      runWs.current = ws
      let finished = false
      let runId = typeof first.attach === 'string' ? first.attach : ''
      ws.onopen = () => ws.send(JSON.stringify(first))
      ws.onmessage = (ev: MessageEvent<string>) => {
        let frame: ReplayFrame
        try {
          frame = JSON.parse(ev.data) as ReplayFrame
        } catch {
          return
        }
        if (frame.type === 'started') {
          runId = frame.run_id
          localStorage.setItem(ACTIVE_RUN_KEY, frame.run_id)
          const startedAt = Date.parse(frame.started_at) || Date.now()
          setNow(Date.now())
          setReattach(null)
          setRun({
            runId: frame.run_id,
            workflow: frame.workflow,
            runs: frame.runs,
            total: frame.total,
            stepsPerItem: frame.steps_per_item,
            startedAt,
            lastStepAt: startedAt,
            items: [],
            log: [],
            current: null,
            status: 'running',
            attached: true,
          })
          return
        }
        if (frame.type === 'step_started') {
          const f = frame
          const since = (f.started_at && Date.parse(f.started_at)) || Date.now()
          setRun((r) => (r ? { ...r, current: { ...f, since }, attached: true } : r))
          return
        }
        onWorkspaceChanged?.()
        if (frame.type === 'step') {
          const step = frame
          setRun((r) =>
            r
              ? { ...r, log: [...r.log, step], current: null, lastStepAt: Date.now(), attached: true }
              : r,
          )
        } else if (frame.type === 'item_result') {
          const res = frame
          setRun((r) => {
            if (!r) return r
            const prevSteps = r.items.reduce((acc, it) => acc + it.steps, 0)
            const item: ItemResult = {
              ok: res.ok,
              error: res.error,
              outputs: res.outputs,
              steps: r.log.length - prevSteps,
            }
            return { ...r, items: [...r.items, item] }
          })
        } else {
          finished = true
          const result = frame
          localStorage.removeItem(ACTIVE_RUN_KEY)
          setRun((r) => {
            if (!r) {
              // An attach that failed outright (unknown run): nothing to show.
              if (!result.ok && result.error) setError(result.error)
              return r
            }
            return {
              ...r,
              current: null,
              status: result.status ?? (result.ok ? 'done' : 'failed'),
              error: result.error,
              finishedAt: (result.finished_at && Date.parse(result.finished_at)) || Date.now(),
              attached: true,
            }
          })
        }
      }
      ws.onclose = () => {
        if (finished || runWs.current !== ws) return
        // The run is still going on the server; get back on it (see the
        // reattach effect below).
        setRun((r) => (r && r.status === 'running' ? { ...r, attached: false } : r))
        if (runId) setReattach((prev) => ({ id: runId, attempt: (prev?.attempt ?? 0) + 1 }))
      }
    },
    [onWorkspaceChanged],
  )

  useEffect(() => {
    if (!reattach) return
    const id = window.setTimeout(() => followRun({ attach: reattach.id }), 2000)
    return () => clearTimeout(id)
  }, [reattach, followRun])

  useEffect(() => {
    void reload()
    fetchWorkspaceRoot()
      .then(setRoot)
      .catch(() => undefined)
    // Pick the run up again after a reload: the one this page was following,
    // else whatever is live on the server (started from another tab).
    const remembered = localStorage.getItem(ACTIVE_RUN_KEY)
    if (remembered) {
      followRun({ attach: remembered })
    } else {
      fetchRuns(undefined, 1)
        .then((r) => {
          if (r.live[0]) followRun({ attach: r.live[0] })
        })
        .catch(() => undefined)
    }
    return () => runWs.current?.close()
    // Mount only: reattaching is a one-time decision, not something to redo
    // whenever a callback identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Tick once a second while a run is live so the elapsed/ETA readout updates;
  // slower otherwise so "5 min ago" labels stay honest.
  useEffect(() => {
    const live = run?.status === 'running'
    const id = setInterval(() => setNow(Date.now()), live ? 1000 : 30000)
    return () => clearInterval(id)
  }, [run?.status])

  // ── Explorer selection feeds the run editor ──
  // Adjusting state during render on a prop change (tracked via prevSel), the
  // React-recommended pattern for "derive state from a changed prop". Only the
  // files that were *added* to the selection are placed (see placeFiles), so a
  // deselect or a re-click changes nothing and no row is ever taken away.
  const [prevSel, setPrevSel] = useState<string[]>(selectedPaths)
  if (selectedPaths !== prevSel) {
    setPrevSel(selectedPaths)
    const added = selectedPaths.filter((p) => !prevSel.includes(p))
    if (mode.kind === 'run' && detail && added.length > 0) {
      if (mode.source === 'files') {
        const names = detail.inputs.filter((i) => !i.default).map((i) => i.name)
        setMode({ ...mode, rows: placeFiles(mode.rows, names, added, focusedCell) })
        setFocusedCell(null)
      } else if (mode.source === 'plan' && mode.rows.length === 0) {
        setPlanCsv(added[0])
      }
    }
  }

  // ── Pre-flight preview, debounced on every edit ──
  // The run button is only enabled once the preview for exactly these rows has
  // come back: the resolved steps and per-row checks are the confirmation the
  // engine's no-permission-prompt design requires.
  const requiredNames = detail ? detail.inputs.filter((i) => !i.default).map((i) => i.name) : []
  const derivedNames = detail ? detail.inputs.filter((i) => i.default).map((i) => i.name) : []
  const runBindings: Record<string, string>[] =
    mode.kind === 'run'
      ? mode.rows
          .filter((r) => requiredNames.every((n) => (r.values[n] ?? '').trim() !== ''))
          .map((r) => {
            const values: Record<string, string> = {}
            requiredNames.forEach((n) => {
              values[n] = r.values[n].trim()
            })
            if (mode.resultsDir.trim()) {
              derivedNames.forEach((n) => {
                values[n] = mode.resultsDir.trim()
              })
            }
            return values
          })
      : []
  const previewKey =
    mode.kind === 'run' && detail
      ? JSON.stringify([detail.name, runBindings])
      : ''
  useEffect(() => {
    if (!previewKey || !detail) return
    if (runBindings.length === 0) return
    const name = detail.name
    const bindings = runBindings
    const id = window.setTimeout(() => {
      replayPreview(name, bindings)
        .then((result) => setPreview({ key: previewKey, result }))
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
    }, 350)
    return () => clearTimeout(id)
    // previewKey encodes everything the request depends on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey])

  // The workflow whose detail the panel currently wants. Guards against a
  // slow response for a previously clicked row overwriting the current one —
  // the detail's action buttons act on detail.name, so a stale detail under
  // another row's header would target the wrong workflow.
  const detailForRef = useRef<string | null>(null)

  const openDetail = async (name: string) => {
    setMode({ kind: 'view' })
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

  const showDraft = (draft: WorkflowDetail) => {
    detailForRef.current = draft.name
    setExpanded(draft.name)
    setDetail(draft)
    setMode({ kind: 'view' })
  }

  const saveChat = () =>
    withBusy('Distilling the chat into a workflow (local model)…', async () => {
      if (!distillSessionId) return
      const draft = await distillSession(distillSessionId)
      await reload()
      showDraft(draft)
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
      showDraft(draft)
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

  // Fill the editor from a plan_batch manifest — the binding list built once
  // from a single-subject prototype, so the cohort roll-out is a review-and-run.
  // Flagged (missing/ambiguous) items are surfaced, never silently run.
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
        setError('No ready items in the plan. Resolve the flagged rows first.')
        return
      }
      setMode((m) =>
        m.kind === 'run'
          ? { ...m, source: 'plan', rows: res.runs.map((values) => ({ values })) }
          : m,
      )
    })

  const beginRun = (d: WorkflowDetail) => {
    setPreview(null)
    setPlanSkipped(null)
    setPlanCsv('')
    setShowSteps(false)
    setShowResultsDir(false)
    setFocusedCell(null)
    // Files is the default, prefilled from whatever is selected in the explorer.
    const names = d.inputs.filter((i) => !i.default).map((i) => i.name)
    setMode({
      kind: 'run',
      source: 'files',
      rows: placeFiles([emptyRow(names)], names, selectedPaths, null),
      resultsDir: '',
    })
  }

  /** Pick (or switch) where the inputs come from; switching starts over. */
  const chooseSource = (
    d: WorkflowDetail,
    m: Extract<Mode, { kind: 'run' }>,
    source: RunSource,
  ) => {
    if (source === m.source) return
    const names = d.inputs.filter((i) => !i.default).map((i) => i.name)
    setPreview(null)
    setPlanSkipped(null)
    setPlanCsv('')
    setFocusedCell(null)
    setMode({
      ...m,
      source,
      rows: source === 'files' ? placeFiles([emptyRow(names)], names, selectedPaths, null) : [],
    })
  }

  const startRun = (name: string, runs: Record<string, string>[]) => {
    setError(null)
    followRun({ name, runs })
    setMode({ kind: 'view' })
  }

  const stopRun = () => {
    runWs.current?.send(JSON.stringify({ type: 'cancel' }))
  }

  const closeRun = () => {
    runWs.current?.close()
    runWs.current = null
    localStorage.removeItem(ACTIVE_RUN_KEY)
    setRun(null)
  }

  const renderRun = (r: RunState) => {
    const live = r.status === 'running'
    const succeeded = r.items.filter((it) => it.ok).length
    const failedRan = r.items.filter((it) => !it.ok).length
    const notRun = r.total - r.items.length
    const retryRuns = r.runs.filter((_, i) => !r.items[i]?.ok)
    const duration = (r.finishedAt ?? now) - r.startedAt
    const single = r.total === 1
    const failedStep = single ? r.log.find((s) => !s.ok) : undefined
    return (
      <div className="wf-form">
        {live ? (
          <>
            <ProgressBar
              frac={runProgress(r)}
              active
              label={
                single
                  ? `step ${Math.min(r.log.length + 1, r.stepsPerItem)} of ${r.stepsPerItem}`
                  : `${r.items.length} of ${r.total}`
              }
            />
            <div className="wf-timing">
              {!r.attached
                ? 'Connection lost — reconnecting. The run continues on the server.'
                : runTiming(r, now)}
            </div>
          </>
        ) : (
          <div className="wf-run-summary">
            {single ? (
              <span className={r.status === 'done' ? 'tally-ok' : 'tally-fail'}>
                {r.status === 'done'
                  ? 'Done'
                  : r.status === 'cancelled'
                    ? 'Stopped'
                    : failedStep
                      ? `Failed at step ${failedStep.index}`
                      : 'Failed'}
              </span>
            ) : (
              <>
                <span className="tally-ok">{succeeded} done</span>
                {failedRan > 0 && <span className="tally-fail">{failedRan} failed</span>}
                {notRun > 0 && <span className="tally-muted">{notRun} not run</span>}
              </>
            )}
            <span className="tally-muted">{formatDuration(duration)}</span>
          </div>
        )}
        {single ? (
          <>
            <StepRows
              steps={r.log}
              current={live ? r.current : null}
              now={now}
              stepsPerItem={r.stepsPerItem}
            />
            {!live && r.items[0]?.outputs.length ? (
              <FileChips paths={r.items[0].outputs} root={root} onOpenFile={onOpenFile} />
            ) : null}
          </>
        ) : (
          <ItemRows r={r} now={now} root={root} onOpenFile={onOpenFile} />
        )}
        {!live && !single && r.error && r.status === 'failed' && failedRan === 0 && (
          <div className="wf-step-error">{firstLine(r.error)}</div>
        )}
        <div className="wf-actions">
          {live ? (
            <button className="btn-danger" onClick={stopRun} disabled={!r.attached}>
              <StopSquareIcon size={12} /> Stop
            </button>
          ) : (
            <>
              {retryRuns.length > 0 && (
                <button
                  className="btn-primary"
                  title={
                    r.status === 'cancelled'
                      ? 'Continue with the items that did not finish.'
                      : 'Run only what failed.'
                  }
                  onClick={() => startRun(r.workflow, retryRuns)}
                >
                  {r.status === 'cancelled' ? <PlayIcon size={12} /> : <RefreshIcon size={12} />}{' '}
                  {r.status === 'cancelled' ? 'Resume' : 'Retry'}
                  {single ? '' : ` ${retryRuns.length}`}
                </button>
              )}
              <button className="btn-plain" onClick={closeRun}>
                Close
              </button>
            </>
          )}
        </div>
      </div>
    )
  }

  const renderEditor = (d: WorkflowDetail, m: Extract<Mode, { kind: 'run' }>) => {
    const required = d.inputs.filter((i) => !i.default)
    const derived = d.inputs.filter((i) => i.default)
    const ready = preview && preview.key === previewKey ? preview.result : null
    const readyItems = ready ? ready.items.filter((it) => it.ok) : []
    const complete = runBindings.length
    // Row i of the editor maps to binding j only when it is complete; keep the
    // preview verdicts aligned with the rows they were computed for.
    let bindingIdx = -1
    const rowVerdict = m.rows.map((r) => {
      const isComplete = requiredNames.every((n) => (r.values[n] ?? '').trim() !== '')
      if (!isComplete) return null
      bindingIdx += 1
      return ready?.items[bindingIdx] ?? null
    })
    const canRun = !!ready && ready.ok && readyItems.length > 0 && run?.status !== 'running'
    // The run controls appear once there is something to run: right away for
    // hand-picked files, after the plan has been loaded for a cohort.
    const showControls = m.source === 'files' || (m.source === 'plan' && m.rows.length > 0)
    const runnable = ready
      ? runBindings.filter((_, i) => ready.items[i]?.ok)
      : []
    return (
      <form
        className="wf-form"
        onSubmit={(e) => {
          e.preventDefault()
          if (canRun) startRun(d.name, runnable)
        }}
      >
        {required.length > 0 && (
          <div className="wf-source-cards" role="radiogroup" aria-label="Where the inputs come from">
            {SOURCES.map((o) => (
              <button
                key={o.value}
                type="button"
                role="radio"
                aria-checked={m.source === o.value}
                className={m.source === o.value ? 'wf-source-card active' : 'wf-source-card'}
                onClick={() => chooseSource(d, m, o.value)}
              >
                <span className="wf-source-icon">{o.icon}</span>
                <span className="wf-source-text">
                  <span className="wf-source-title">{o.title}</span>
                  <span className="wf-source-hint">{o.hint}</span>
                </span>
              </button>
            ))}
          </div>
        )}
        <div className="wf-hint">
          {required.length === 0
            ? 'This workflow takes no inputs.'
            : m.source === 'files'
              ? 'One row per run. Click a cell, then a file in the explorer, or drag a file into it.'
              : m.rows.length === 0
                ? 'Point at the batch_plan.csv written by the cohort plan_batch step: click the field, then the file in the explorer, or drag it in.'
                : 'One row per cohort item, as resolved by the plan. Fix a path here if one is wrong.'}
        </div>
        {m.source === 'plan' && m.rows.length === 0 && (
          <div className="wf-batch-tools">
            <PathCell value={planCsv} placeholder="…/batch_plan.csv" onChange={setPlanCsv} />
            <button
              type="button"
              className="btn-primary"
              disabled={!planCsv.trim()}
              onClick={() => void fillFromPlan(d.name)}
            >
              Load plan
            </button>
          </div>
        )}
        {required.length > 0 && m.rows.length > 0 && (
          <div className="wf-batch-table">
            <div className="wf-batch-row wf-batch-head">
              <span className="wf-batch-idx">#</span>
              {required.map((i) => (
                <span key={i.name} className="wf-batch-col" title={`${i.name}${i.description ? ` — ${i.description}` : ''}`}>
                  {inputTitle(i)}
                </span>
              ))}
              <span className="wf-batch-status" />
              <span className="wf-batch-rm" />
            </div>
            {m.rows.map((row, ri) => {
              const verdict = rowVerdict[ri]
              const warn = verdict?.findings.find((f) => f.severity === 'warning')
              return (
                <div key={ri} className="wf-batch-row">
                  <span className="wf-batch-idx">{ri + 1}</span>
                  {required.map((i) => (
                    <PathCell
                      key={i.name}
                      value={row.values[i.name] ?? ''}
                      placeholder={i.example ? `e.g. ${basename(i.example)}` : 'path'}
                      onFocus={() => setFocusedCell({ row: ri, name: i.name })}
                      onChange={(v) =>
                        setMode({
                          ...m,
                          rows: m.rows.map((r, j) =>
                            j === ri ? { values: { ...r.values, [i.name]: v } } : r,
                          ),
                        })
                      }
                    />
                  ))}
                  <span
                    className={`wf-batch-status ${verdict ? (verdict.ok ? (warn ? 'warn' : 'ok') : 'fail') : ''}`}
                    title={
                      verdict
                        ? verdict.ok
                          ? warn
                            ? warn.note
                              : 'ready'
                          : (verdict.error ?? 'cannot run')
                        : ''
                    }
                  >
                    {verdict ? (verdict.ok ? (warn ? '⚠' : '✓') : '✗') : ''}
                  </span>
                  <button
                    type="button"
                    className="btn-icon wf-batch-rm"
                    title="Remove row"
                    onClick={() => setMode({ ...m, rows: m.rows.filter((_, j) => j !== ri) })}
                  >
                    <XIcon size={12} />
                  </button>
                </div>
              )
            })}
            {m.rows.some((_, ri) => rowVerdict[ri] && !rowVerdict[ri].ok) && (
              <div className="wf-row-problems">
                {m.rows.map((_, ri) => {
                  const v = rowVerdict[ri]
                  if (!v || v.ok) return null
                  const f = v.findings.find((x) => x.severity === 'error')
                  return (
                    <div key={ri} className="wf-step-error">
                      Row {ri + 1}: {v.error ?? 'cannot run'}
                      {f && f.entries.length > 0 && (
                        <span className="wf-nearest">
                          {' '}
                          — in <code>{f.nearest}</code>: {f.entries.slice(0, 4).join(', ')}
                          {f.entry_total > 4 ? ', …' : ''}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
            <div className="wf-batch-tools">
              {m.source === 'files' ? (
                <button
                  type="button"
                  className="btn-plain"
                  onClick={() => setMode({ ...m, rows: [...m.rows, emptyRow(requiredNames)] })}
                >
                  + Add row
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-text"
                  onClick={() => {
                    setPlanSkipped(null)
                    setPreview(null)
                    setMode({ ...m, rows: [] })
                  }}
                >
                  Load a different plan
                </button>
              )}
            </div>
          </div>
        )}
        {planSkipped && planSkipped.length > 0 && (
          <div className="wf-hint wf-plan-skipped">
            {planSkipped.length} item(s) flagged and left out. Resolve them in the plan, then
            re-fill:
            <ul className="wf-manual-list">
              {planSkipped.map((s, i) => (
                <li key={i}>
                  <code>
                    {s.subject}
                    {s.session ? `/${s.session}` : ''}
                  </code>{' '}
                  · {s.status}: {s.reason}
                </li>
              ))}
            </ul>
          </div>
        )}
        {derived.length > 0 && (
          <>
            <button
              type="button"
              className="btn-text wf-derived-toggle"
              onClick={() => setShowResultsDir((v) => !v)}
            >
              {showResultsDir ? 'Hide' : 'Change'} where results are written
            </button>
            {showResultsDir && (
              <label className="wf-input-row">
                <span className="wf-input-label">
                  <span className="wf-input-desc">
                    Leave blank to write beside each input file. Applies to every row.
                  </span>
                </span>
                <PathCell
                  className="wf-input"
                  value={m.resultsDir}
                  placeholder="results folder"
                  onChange={(v) => setMode({ ...m, resultsDir: v })}
                />
              </label>
            )}
          </>
        )}
        {complete > 0 && ready && ready.steps.length > 0 && (
          <div className="wf-preview">
            <button
              type="button"
              className="btn-text wf-derived-toggle"
              onClick={() => setShowSteps((v) => !v)}
            >
              {showSteps ? 'Hide' : 'Show'} what will run
              {complete > 1 ? ' (first row)' : ''}
            </button>
            {showSteps && (
              <ol className="wf-steps">
                {ready.steps.map((s) => (
                  <li key={s.index}>
                    <code>{s.tool}</code>
                    <span className="wf-args-inline">
                      {Object.entries(s.arguments).map(([k, v]) => (
                        <span key={k} title={typeof v === 'string' ? v : undefined}>
                          {k}: {display(v, root)}
                        </span>
                      ))}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
        <div className="wf-actions">
          {showControls && (
            <button
              className="btn-primary"
              type="submit"
              disabled={!canRun}
            title={
              canRun
                ? 'Run these exact steps. No LLM, no permission prompts.'
                : complete === 0
                  ? 'Fill in at least one row'
                  : ready
                    ? (ready.error ?? 'nothing to run')
                    : 'Checking…'
            }
          >
            <PlayIcon size={12} />{' '}
            {ready && complete > readyItems.length && readyItems.length > 0
              ? `Run ${readyItems.length} of ${complete} rows`
              : readyItems.length > 1
                  ? `Run ${readyItems.length} rows`
                  : 'Run'}
            </button>
          )}
          <button className="btn-plain" type="button" onClick={() => setMode({ kind: 'view' })}>
            Cancel
          </button>
          {complete > 0 && !ready && <span className="wf-recent-muted">Checking inputs…</span>}
        </div>
      </form>
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
              This chat produced no replayable tool steps, so the workflow is empty.
            </div>
          )}
          {!d.replayable && d.replay_error && d.steps.length > 0 && (
            <div className="wf-notice">Can't run: {d.replay_error}</div>
          )}
          <Requirements requires={d.requires} />
          {d.manual_steps.length > 0 && (
            <div className="wf-notice">
              Includes manual step(s) a run can't do. Do these by hand:
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
                  ? 'Run these exact steps on new inputs. No LLM involved.'
                  : (d.replay_error ?? '')
              }
              onClick={() => beginRun(d)}
            >
              <PlayIcon size={12} /> Run
            </button>
            {d.kind === 'draft' && (
              <button
                className="btn-plain"
                title="Mark as reviewed and keep permanently"
                onClick={() => void promote(d.name)}
              >
                Promote
              </button>
            )}
            <span className="wf-actions-secondary">
              {d.kind === 'draft' ? (
                <>
                  <button
                    className="btn-text"
                    onClick={() => setMode({ kind: 'rename', value: d.name })}
                  >
                    Rename
                  </button>
                  <button
                    className="btn-text"
                    title="Rewrite the description with a plain-language instruction"
                    onClick={() => setMode({ kind: 'refine', value: '' })}
                  >
                    Refine
                  </button>
                </>
              ) : (
                <button
                  className="btn-text"
                  title="Move back to draft for renaming/refining"
                  onClick={() => void unpromote(d.name)}
                >
                  Edit
                </button>
              )}
              <button
                className="btn-text"
                title="Export as a shareable .workflow.yaml file"
                onClick={() => void exportFile(d.name)}
              >
                Export
              </button>
              <button className="btn-text wf-delete" onClick={() => remove(d.name)}>
                Delete
              </button>
            </span>
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

      {mode.kind === 'run' && renderEditor(d, mode)}
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
        {run && (
          <div className="wf-run-card">
            <div className="wf-run-card-head">
              <PlayIcon size={12} /> {run.workflow}
              <span className="wf-recent-muted wf-run-card-id" title={run.runId}>
                {run.total > 1 ? `${run.total} items` : 'single run'}
              </span>
            </div>
            {renderRun(run)}
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
