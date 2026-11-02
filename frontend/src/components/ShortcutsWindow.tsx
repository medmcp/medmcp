import { useEffect } from 'react'
import { XIcon } from './icons'

interface ShortcutsWindowProps {
  open: boolean
  onClose: () => void
}

/** One line of the list. `keys` are alternatives, each drawn as its own key cap;
 *  a combination is one cap ("Shift + ↑"), so the two never read the same. */
type Shortcut = { keys: string[]; action: string; note?: string }

type Group = { title: string; hint?: string; rows: Shortcut[] }

/* Everything the app actually binds, grouped by the thing that has to have
 * focus for it to work — the detail that makes a shortcut list useless when it
 * is left out. Several of these are not ours: the file tree's keymap comes with
 * react-arborist, the separator keys with react-resizable-panels, and the
 * viewer's mouse controls with Niivue. They are listed because a reader cannot
 * tell the difference, and they work.
 *
 * Niivue's own *keyboard* bindings (V, M, H/J/K/L and the rest) are deliberately
 * absent: Viewer.tsx strips the canvas's tabindex so they can never fire. */
const GROUPS: Group[] = [
  {
    title: 'Chat',
    rows: [
      { keys: ['Enter'], action: 'Send the message' },
      { keys: ['Shift + Enter'], action: 'Start a new line instead' },
    ],
  },
  {
    title: 'Files',
    hint: 'While the file tree has focus.',
    rows: [
      { keys: ['↑', '↓'], action: 'Move between files' },
      { keys: ['Shift + ↑', 'Shift + ↓'], action: 'Extend the selection' },
      { keys: ['→', '←'], action: 'Open or close a folder' },
      { keys: ['Space'], action: 'Open the file, or fold the folder' },
      { keys: ['Enter'], action: 'Rename' },
      {
        keys: ['Backspace'],
        action: 'Delete the selection',
        note: 'Asks first. Delete does nothing — only Backspace is bound.',
      },
      { keys: ['Ctrl / ⌘ + A'], action: 'Select everything' },
      { keys: ['Home', 'End'], action: 'Jump to the first or last file' },
      { keys: ['PageUp', 'PageDown'], action: 'Move a page at a time' },
      { keys: ['*'], action: 'Open every folder beside this one' },
      {
        keys: ['a…z'],
        action: 'Jump to the next name starting with what you type',
        note: 'Only reaches files already on screen, not ones inside a folded folder.',
      },
      {
        keys: ['Ctrl / ⌘ + click', 'Shift + click'],
        action: 'Add one file to the selection, or take a range',
      },
    ],
  },
  {
    title: 'Viewer — keys',
    hint: 'The viewer has to hold the keyboard: click the image once first.',
    rows: [
      { keys: ['↑', '↓', 'PageUp', 'PageDown'], action: 'Step one slice', note: 'Hold Shift for 10.' },
      { keys: ['←', '→'], action: 'Step one frame of a 4D volume', note: 'Hold Shift for 10.' },
      { keys: ['Space'], action: 'Play or pause a 4D volume' },
      { keys: ['o'], action: 'Hide or show the overlay, for a before and after' },
      {
        keys: ['+', '-'],
        action: 'Zoom about the crosshair',
        note: '= zooms in as well, so it needs no Shift.',
      },
      { keys: ['0'], action: 'Back to 1×' },
    ],
  },
  {
    title: 'Viewer — mouse',
    hint: 'These need no focus, so they work wherever you last clicked.',
    rows: [
      {
        keys: ['Left drag'],
        action: 'Move the crosshair',
        note: 'Inside the 3D tile it turns the view instead.',
      },
      { keys: ['Middle drag', 'Shift / Ctrl + left drag'], action: 'Pan' },
      { keys: ['Right drag'], action: 'Set the window and level' },
      { keys: ['Double-click'], action: 'Put the window and level back' },
      {
        keys: ['Wheel'],
        action: 'Step slices',
        note: 'Over the 3D tile it zooms; over a 4D graph it steps frames.',
      },
      { keys: ['Ctrl / ⌘ + wheel', 'Pinch'], action: 'Zoom about the crosshair' },
    ],
  },
  {
    title: 'Panels and windows',
    rows: [
      {
        keys: ['←', '→', '↑', '↓'],
        action: 'Move a panel separator by 5%',
        note: 'Tab to a separator first.',
      },
      { keys: ['Home', 'End'], action: 'Push that separator all the way over' },
      { keys: ['F6'], action: 'Go to the next separator' },
      {
        keys: ['Escape'],
        action: 'Close the open popover or menu',
        note: 'The settings drawer and the larger windows close on their ✕ or the backdrop.',
      },
      { keys: ['?', 'F1'], action: 'Show this list' },
    ],
  },
]

/**
 * Every keyboard and mouse control the app has, in one list.
 *
 * It exists because almost none of them were discoverable: four strings in the
 * whole frontend hinted at a binding, and three of those were aria-labels, which
 * a sighted reader never sees. Most of the list is therefore new information
 * even to someone who has used the app for months.
 */
export function ShortcutsWindow({ open, onClose }: ShortcutsWindowProps) {
  // Same as the other windows that bother: Escape closes it. Worth having here
  // more than anywhere, since a reference panel is opened and dismissed often.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null
  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="extwin" role="dialog" aria-label="Shortcuts and controls">
        <div className="panel-header">
          <span>Shortcuts and controls</span>
          <span className="panel-actions">
            <button className="btn-icon" aria-label="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>
        <div className="extwin-body sc-body">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <div className="settings-section">{g.title}</div>
              {g.hint && <div className="sc-group-hint">{g.hint}</div>}
              {g.rows.map((r) => (
                <div key={r.action} className="sc-row">
                  <span className="sc-action">
                    {r.action}
                    {r.note && <span className="sc-note">{r.note}</span>}
                  </span>
                  <span className="sc-keys">
                    {r.keys.map((k) => (
                      <kbd key={k}>{k}</kbd>
                    ))}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
