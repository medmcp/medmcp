import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJobs, stopJob } from '../api'
import type { JobInfo } from '../types'

const POLL_MS = 10_000
const SEEN_KEY = 'medmcp.jobs.seen'

/** Ids of finished jobs the person has already looked at (per browser). */
function loadSeen(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(SEEN_KEY) ?? '[]') as string[])
  } catch {
    return new Set()
  }
}

function saveSeen(seen: Set<string>): void {
  try {
    localStorage.setItem(SEEN_KEY, JSON.stringify([...seen].slice(-100)))
  } catch {
    /* private mode: the pill just shows finished jobs again next time */
  }
}

function label(job: JobInfo): string {
  return job.name || `${job.stack.replace(/^medmcp-/, '')} · ${job.tool}`
}

/** How long the job ran: until now while running, until it ended otherwise. */
function elapsed(job: JobInfo): string {
  if (!job.started_at) return ''
  const end = job.status === 'running' || !job.finished_at ? Date.now() : Date.parse(job.finished_at)
  const sec = Math.max(0, (end - Date.parse(job.started_at)) / 1000)
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  return h ? `${h} h ${m} min` : `${m} min`
}

/**
 * Background jobs at a glance: a pill in the header while a job runs (with its
 * progress line, e.g. "epoch 41/100 · pseudo-Dice 0.88"), and after one ends
 * until the person has looked at it. Click for the list with Stop.
 */
export function JobPill() {
  const [jobs, setJobs] = useState<JobInfo[]>([])
  const [open, setOpen] = useState(false)
  const [seen, setSeen] = useState<Set<string>>(loadSeen)
  const box = useRef<HTMLDivElement>(null)

  const refresh = useCallback(() => {
    fetchJobs()
      .then(setJobs)
      .catch(() => setJobs([]))
  }, [])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, POLL_MS)
    return () => window.clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const running = jobs.filter((j) => j.status === 'running')
  const unseen = jobs.filter((j) => j.status !== 'running' && !seen.has(j.id))
  // Opening the list marks finished jobs as seen; the pill stays while the list
  // is open, so the list does not vanish with it.
  const first = running[0] ?? unseen[0] ?? (open ? jobs[0] : undefined)
  if (!first) return null

  const failed = unseen.some((j) => j.status === 'failed')
  const state = running.length ? 'running' : failed ? 'failed' : 'done'
  const progress = first.progress
  // The state goes in its own span so a long job name never hides it.
  const status = running.length
    ? [progress?.text || 'running', running.length > 1 ? `+${running.length - 1}` : '']
        .filter(Boolean)
        .join(' · ')
    : first.status === 'failed'
      ? 'failed'
      : 'finished'

  const toggle = () => {
    setOpen((o) => !o)
    if (!running.length) {
      const next = new Set(seen)
      unseen.forEach((j) => next.add(j.id))
      setSeen(next)
      saveSeen(next)
    }
  }

  const onStop = async (job: JobInfo) => {
    try {
      await stopJob(job.id)
    } finally {
      refresh()
    }
  }

  return (
    <div className="job-pill-box" ref={box}>
      <button
        className={`job-pill job-pill-${state}`}
        onClick={toggle}
        aria-label="Background jobs"
        title="Background jobs"
      >
        <span className="job-pill-dot" />
        <span className="job-pill-text">{label(first)}</span>
        <span className="job-pill-state">{status}</span>
        {running.length > 0 && typeof progress?.fraction === 'number' && (
          <span className="job-pill-bar">
            <span style={{ width: `${Math.round(progress.fraction * 100)}%` }} />
          </span>
        )}
      </button>
      {open && (
        <div className="job-pop" role="dialog" aria-label="Background jobs">
          {jobs.slice(0, 8).map((job) => (
            <div key={job.id} className="job-row">
              <div className="job-row-main">
                <span className={`job-status job-status-${job.status}`}>{job.status}</span>
                <span className="job-row-name">{label(job)}</span>
              </div>
              <div className="job-row-detail">
                {job.progress?.text && <span>{job.progress.text}</span>}
                <span>{elapsed(job)}</span>
                {job.session_title && <span>from “{job.session_title}”</span>}
              </div>
              {job.status === 'running' && (
                <button className="btn-text" onClick={() => void onStop(job)}>
                  Stop
                </button>
              )}
            </div>
          ))}
          <div className="job-pop-note">
            Jobs keep running when this window or the chat is closed.
          </div>
        </div>
      )}
    </div>
  )
}
