import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { cancelTask, fetchJobs, fetchTasks, stopJob } from '../api'
import type { JobInfo, TaskInfo } from '../types'

const POLL_MS = 10_000
// While a task runs: its status line ("Waiting for the GPU (1 ahead)") moves faster.
const TASK_POLL_MS = 3_000
const POP_WIDTH = 360
const POP_GAP = 6
const VIEWPORT_MARGIN = 8

interface PopPlace {
  bottom: number
  right: number
  width: number
  maxHeight: number
}

/**
 * The list opens upward from the chip, right edges aligned. It is portalled to
 * the document because the chat panel clips its overflow and is a containment
 * root, which cuts off even a `position: fixed` child at the panel edge.
 */
function placePop(rect: DOMRect): PopPlace {
  const vw = window.innerWidth
  const width = Math.min(POP_WIDTH, vw - 2 * VIEWPORT_MARGIN)
  const right = Math.max(VIEWPORT_MARGIN, Math.min(vw - rect.right, vw - width - VIEWPORT_MARGIN))
  return {
    bottom: window.innerHeight - rect.top + POP_GAP,
    right,
    width,
    maxHeight: Math.max(120, rect.top - POP_GAP - VIEWPORT_MARGIN),
  }
}

/** A file the call works on: its argument's name ("input") and the file's name. */
interface FileArg {
  role: string
  name: string
}

/** A running job or task, as one entry of the list. */
interface Row {
  id: string
  tool: string
  stack: string
  files: FileArg[]
  /** "Waiting for the GPU (1 ahead)", a job's progress line, or "Running for 3 min". */
  status: string
  stop: () => Promise<void>
}

function duration(sec: number): string {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  if (h) return `${h} h ${m} min`
  return m ? `${m} min` : `${Math.floor(sec)} s`
}

/** The path arguments of a call: `input_path` → input, `output_dir` → output. */
function fileArgs(args: Record<string, unknown> | undefined): FileArg[] {
  return Object.entries(args ?? {})
    .filter((e): e is [string, string] => typeof e[1] === 'string' && e[1].includes('/'))
    .map(([key, value]) => ({
      role: key.replace(/_(path|dir|file|folder)$/, '').replace(/_/g, ' '),
      name: value.replace(/\/+$/, '').split('/').pop() ?? value,
    }))
}

function jobRow(job: JobInfo): Row {
  const sec = job.started_at ? Math.max(0, (Date.now() - Date.parse(job.started_at)) / 1000) : 0
  return {
    id: job.id,
    tool: job.name || job.tool,
    stack: job.stack.replace(/^medmcp-/, ''),
    files: fileArgs(job.arguments),
    status: job.progress?.text || `Running for ${duration(sec)}`,
    stop: () => stopJob(job.id),
  }
}

function taskRow(task: TaskInfo): Row {
  return {
    id: `task:${task.taskId}`,
    tool: task.tool,
    stack: task.stack.replace(/^medmcp-/, ''),
    files: fileArgs(task.arguments),
    // A queue position says more than a duration; "Running" says nothing.
    status:
      task.statusMessage && task.statusMessage !== 'Running'
        ? task.statusMessage
        : `Running for ${duration(task.elapsedSec)}`,
    stop: () => cancelTask(task.taskId),
  }
}

/**
 * Background work at a glance, in the message box's bar next to send: "1 running
 * task" while jobs or tasks run, nothing otherwise — the chat's own notice says
 * when one has finished. Click for the list: tool, stack, the files it works on,
 * its status, and Stop.
 * `refreshSignal` changes when a chat's tool call settles, so a call that just
 * became a task shows at once.
 */
export function JobPill({ refreshSignal = 0 }: { refreshSignal?: number }) {
  const [rows, setRows] = useState<Row[]>([])
  const [open, setOpen] = useState(false)
  const [place, setPlace] = useState<PopPlace | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const pop = useRef<HTMLDivElement>(null)

  const refresh = useCallback(() => {
    void Promise.all([
      fetchTasks().catch(() => [] as TaskInfo[]),
      fetchJobs().catch(() => [] as JobInfo[]),
    ]).then(([tasks, jobs]) =>
      setRows([
        ...tasks.filter((t) => t.status === 'working').map(taskRow),
        ...jobs.filter((j) => j.status === 'running').map(jobRow),
      ]),
    )
  }, [])

  const taskRunning = rows.some((r) => r.id.startsWith('task:'))
  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, taskRunning ? TASK_POLL_MS : POLL_MS)
    return () => window.clearInterval(timer)
  }, [refresh, taskRunning, refreshSignal])

  const shown = open && rows.length > 0
  useLayoutEffect(() => {
    if (!shown) return
    const update = () => {
      if (box.current) setPlace(placePop(box.current.getBoundingClientRect()))
    }
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [shown])

  useEffect(() => {
    if (!shown) return
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node
      if (!box.current?.contains(target) && !pop.current?.contains(target)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [shown])

  if (!rows.length) return null

  const onStop = async (row: Row) => {
    try {
      await row.stop()
    } finally {
      refresh()
    }
  }

  return (
    <div className="job-pill-box" ref={box}>
      <button className="job-pill" onClick={() => setOpen((o) => !o)} aria-expanded={shown}>
        <span className="job-pill-spinner" />
        {rows.length} running {rows.length === 1 ? 'task' : 'tasks'}
      </button>
      {shown &&
        place &&
        createPortal(
          <div
            ref={pop}
            className="job-pop"
            role="dialog"
            aria-label="Running tasks"
            style={{
              bottom: place.bottom,
              right: place.right,
              width: place.width,
              maxHeight: place.maxHeight,
            }}
          >
            {rows.map((row) => (
              <div key={row.id} className="job-row">
                <div className="job-row-head">
                  <span className="job-row-tool">{row.tool}</span>
                  <span className="job-row-stack">{row.stack}</span>
                  <button className="btn-text" onClick={() => void onStop(row)}>
                    Stop
                  </button>
                </div>
                {row.files.map((f) => (
                  <div key={f.role} className="job-row-line">
                    <span className="job-row-key">{f.role}</span>
                    <span className="job-row-file">{f.name}</span>
                  </div>
                ))}
                <div className="job-row-line">
                  <span className="job-row-key">status</span>
                  <span>{row.status}</span>
                </div>
              </div>
            ))}
          </div>,
          document.body,
        )}
    </div>
  )
}
