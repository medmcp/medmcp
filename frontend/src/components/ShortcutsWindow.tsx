import { useEffect, useRef } from 'react'
import { XIcon } from './icons'

interface ShortcutsWindowProps {
  open: boolean
  onClose: () => void
}

/** One line of the list; `keys` are alternatives, each drawn as its own key cap. */
type Shortcut = { keys: string[]; action: string }

type Group = { title: string; hint?: string; rows: Shortcut[] }

/* The controls worth knowing, not every binding the libraries bring along
 * (type-ahead in the file tree, separator keys, …): a short list gets read.
 * Niivue's own keyboard bindings are absent on purpose: Viewer.tsx strips the
 * canvas's tabindex so they can never fire.
 *
 * Every row is a claim about the code; change the two together. */
const VIEWER: Group = {
  title: 'Viewer',
  hint: 'Keys work after a click on the image.',
  rows: [
    { keys: ['Scroll', '↑', '↓'], action: 'Slices' },
    { keys: ['←', '→'], action: '4D frames' },
    { keys: ['Space'], action: 'Play 4D' },
    { keys: ['Ctrl + scroll', '+', '-'], action: 'Zoom' },
    { keys: ['0'], action: 'Reset zoom' },
    { keys: ['Shift + drag'], action: 'Pan' },
    { keys: ['Right drag'], action: 'Brightness and contrast' },
    { keys: ['o'], action: 'Hide overlay' },
  ],
}

const OTHER: Group[] = [
  {
    title: 'Chat',
    rows: [
      { keys: ['Enter'], action: 'Send' },
      { keys: ['Shift + Enter'], action: 'New line' },
    ],
  },
  {
    title: 'Files',
    rows: [
      { keys: ['Ctrl + click'], action: 'Select several' },
      { keys: ['Enter'], action: 'Rename' },
      { keys: ['Backspace'], action: 'Delete' },
      { keys: ['Drag onto image'], action: 'Overlay' },
    ],
  },
  {
    title: 'General',
    rows: [
      { keys: ['?'], action: 'This list' },
      { keys: ['Esc'], action: 'Close' },
    ],
  },
]

/* What a person should know before relying on a result. Each line is a claim
 * about the product; keep it true. */
const LIMITS = [
  'Research software: not for clinical decisions.',
  'Read each approval before you allow it.',
  'Check every result before you use it.',
]

function GroupList({ group }: { group: Group }) {
  return (
    <div className="sc-group">
      <div className="settings-section">{group.title}</div>
      {group.rows.map((r) => (
        <div key={r.action} className="sc-row">
          <span className="sc-action">{r.action}</span>
          <span className="sc-keys">
            {r.keys.map((k) => (
              <kbd key={k}>{k}</kbd>
            ))}
          </span>
        </div>
      ))}
      {group.hint && <div className="sc-group-hint">{group.hint}</div>}
    </div>
  )
}

/** Help: what MedMCP is and where its limits are, then the controls worth knowing. */
export function ShortcutsWindow({ open, onClose }: ShortcutsWindowProps) {
  const winRef = useRef<HTMLDivElement>(null)

  // Escape closes it, as it does the Tool stacks window.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  // Focus moves into the window while it is open, so the keyboard scrolls the
  // list rather than driving what is behind the backdrop, and goes back to
  // where it came from (the ? button, or the viewer) on close.
  useEffect(() => {
    if (!open) return
    const before = document.activeElement
    winRef.current?.focus()
    return () => {
      if (before instanceof HTMLElement) before.focus()
    }
  }, [open])

  if (!open) return null
  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="extwin scwin" role="dialog" aria-label="Help">
        <div className="panel-header">
          <span>Help</span>
          <span className="panel-actions">
            <button className="btn-icon" aria-label="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>
        {/* Focus goes to the scrolling list, so the arrow and page keys scroll it. */}
        <div ref={winRef} className="extwin-body sc-body" tabIndex={-1}>
          <div className="sc-about">
            <div className="settings-section">About MedMCP</div>
            <p>MedMCP runs tested imaging tools from a chat, on this machine.</p>
            <div className="settings-section">Know the limits</div>
            <ul>
              {LIMITS.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </div>
          <div className="sc-col">
            <GroupList group={VIEWER} />
          </div>
          <div className="sc-col">
            {OTHER.map((g) => (
              <GroupList key={g.title} group={g} />
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
