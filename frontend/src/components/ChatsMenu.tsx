import { useEffect, useState } from 'react'
import { archiveSession, deleteSession, fetchSessions, forkSession, renameSession } from '../api'
import type { SessionInfo } from '../types'
import { ArchiveIcon, BranchIcon, ChevronRightIcon, PencilIcon, TrashIcon } from './icons'

interface ChatsMenuProps {
  /** The session the chat is currently showing (highlighted in the list). */
  currentSessionId: string | null
  /** Open an existing session in the chat panel. */
  onSelectSession: (id: string) => void
  /** Called when the currently-open session is deleted, so a fresh chat opens. */
  onCurrentDeleted: () => void
}

/** Coarse day bucket for the list headers, from an ISO timestamp. */
function dayLabel(iso: string | null): string {
  if (!iso) return 'Earlier'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return 'Earlier'
  const midnight = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const days = Math.floor((midnight(new Date()) - midnight(d)) / 86_400_000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return 'This week'
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

function timeLabel(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? ''
    : d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
}

/**
 * "Chats" dropdown for the chat header: a popover listing this workspace's prior
 * sessions next to the New-chat button. Click a row to resume it; rename,
 * archive, or delete from per-row actions; reveal archived ones to restore or
 * delete. Sessions load each time the menu opens.
 */
export function ChatsMenu({ currentSessionId, onSelectSession, onCurrentDeleted }: ChatsMenuProps) {
  const [open, setOpen] = useState(false)
  const [list, setList] = useState<SessionInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showArchived, setShowArchived] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')

  const load = () => {
    fetchSessions()
      .then((s) => {
        setList(s)
        setError(null)
        setEditingId(null)
      })
      .catch((e: unknown) => setError(String(e)))
  }

  useEffect(() => {
    if (open) load()
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  const act = (op: Promise<void>) => {
    op.then(load).catch((e: unknown) => setError(String(e)))
  }

  const commitRename = (id: string) => {
    const title = editValue.trim()
    setEditingId(null)
    // The editor opens pre-filled with the row's current title, which may be
    // the server's fallback (vibe's first-message preview) or a generated
    // title rather than a name the user chose. Committing it unchanged — a
    // blur, an Enter without edits — must not turn that into a user-set
    // title: a user-set title pins the chat and stops title generation.
    const current = list?.find((s) => s.id === id)?.title ?? ''
    if (title === current) return
    act(renameSession(id, title))
  }

  const confirmDelete = (id: string) => {
    if (!confirm('Delete this chat for good? Its transcript and provenance are removed too.')) return
    deleteSession(id)
      .then(() => {
        // Deleting the open chat would otherwise leave its (now-gone) transcript
        // on screen — start a fresh chat instead. That remounts Chat (and this
        // menu with it), so reloading the list here is unnecessary.
        if (id === currentSessionId) onCurrentDeleted()
        else load()
      })
      .catch((e: unknown) => setError(String(e)))
  }

  const select = (id: string) => {
    onSelectSession(id)
    setOpen(false)
  }

  const branch = (id: string) => {
    // Only offered on the open chat's row — vibe forks live sessions only.
    forkSession(id)
      .then((forkId) => select(forkId))
      .catch((e: unknown) => setError(String(e)))
  }

  const active = (list ?? []).filter((s) => !s.archived)
  const archived = (list ?? []).filter((s) => s.archived)

  const renderRow = (s: SessionInfo, isArchived: boolean) => {
    const editing = editingId === s.id
    const isCurrent = s.id === currentSessionId
    return (
      <div
        key={s.id}
        className={`session-row${isCurrent ? ' current' : ''}`}
        // Marks the open chat for assistive tech as well as visually: the row is
        // otherwise distinguished only by a background wash, which hover nearly
        // matches and which conveys nothing without colour.
        aria-current={isCurrent ? 'true' : undefined}
        onClick={() => !editing && select(s.id)}
      >
        <span
          className={`prov-dot${s.hasProvenance ? ' on' : ''}`}
          title={s.hasProvenance ? 'Has a provenance record' : 'No provenance record'}
        />
        <span className="session-main">
          {editing ? (
            <input
              className="session-rename-input"
              autoFocus
              value={editValue}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitRename(s.id)
                if (e.key === 'Escape') setEditingId(null)
              }}
              onBlur={() => commitRename(s.id)}
            />
          ) : (
            <span className="session-title">{s.title || 'Untitled chat'}</span>
          )}
          <span className="session-time">
            {timeLabel(s.updatedAt)}
            {/* Says which chat you are in, rather than leaving it to a colour a
                hover can imitate. */}
            {isCurrent && <span className="session-current-tag">open</span>}
          </span>
        </span>
        <span className="session-actions" onClick={(e) => e.stopPropagation()}>
          {!isArchived && isCurrent && (
            <button
              className="btn-icon"
              title="Branch this chat (duplicate to try a different path)"
              onClick={() => branch(s.id)}
            >
              <BranchIcon size={14} />
            </button>
          )}
          {!isArchived && (
            <button
              className="btn-icon"
              title="Rename"
              onClick={() => {
                setEditValue(s.title ?? '')
                setEditingId(s.id)
              }}
            >
              <PencilIcon size={14} />
            </button>
          )}
          <button
            className="btn-icon"
            title={isArchived ? 'Restore' : 'Archive'}
            onClick={() => act(archiveSession(s.id, !isArchived))}
          >
            <ArchiveIcon size={14} />
          </button>
          <button className="btn-icon danger" title="Delete" onClick={() => confirmDelete(s.id)}>
            <TrashIcon size={14} />
          </button>
        </span>
      </div>
    )
  }

  // Active sessions, grouped by day (the list arrives newest-first).
  const groups: { label: string; rows: SessionInfo[] }[] = []
  for (const s of active) {
    const label = dayLabel(s.updatedAt)
    const last = groups[groups.length - 1]
    if (last && last.label === label) last.rows.push(s)
    else groups.push({ label, rows: [s] })
  }

  return (
    <div className="chats-menu">
      <button
        className="btn-plain chats-menu-trigger"
        title="Open a previous chat"
        onClick={() => setOpen((v) => !v)}
      >
        Chats
        <ChevronRightIcon size={11} className="chats-caret" />
      </button>
      {open && (
        <>
          <div className="chats-menu-backdrop" onClick={() => setOpen(false)} />
          <div className="chats-menu-popover">
            {error && <div className="panel-error">{error}</div>}
            {list == null ? (
              <div className="chats-menu-empty">Loading…</div>
            ) : active.length === 0 ? (
              <div className="chats-menu-empty">No saved chats yet.</div>
            ) : (
              groups.map((g) => (
                <div key={g.label}>
                  <div className="session-group-label">{g.label}</div>
                  {g.rows.map((s) => renderRow(s, false))}
                </div>
              ))
            )}
            {archived.length > 0 && (
              <>
                <button className="show-archived-btn" onClick={() => setShowArchived((v) => !v)}>
                  {showArchived ? 'Hide' : 'Show'} archived ({archived.length})
                </button>
                {showArchived && archived.map((s) => renderRow(s, true))}
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}
