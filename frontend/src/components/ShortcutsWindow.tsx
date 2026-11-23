import { useEffect, useRef } from 'react'
import { XIcon } from './icons'

interface ShortcutsWindowProps {
  open: boolean
  onClose: () => void
}

/** One line of the list. `keys` are alternatives, each drawn as its own key cap;
 *  a combination is one cap ("Shift + ↑"), so the two never read the same. */
type Shortcut = { keys: string[]; action: string }

type Group = { title: string; hint?: string; rows: Shortcut[] }

/* Everything the app actually binds, grouped by the thing that has to have
 * focus for it to work — the detail that makes a shortcut list useless when it
 * is left out. Several of these are not ours: the file tree's keymap comes with
 * react-arborist, the separator keys with react-resizable-panels, and the
 * viewer's mouse controls with Niivue. They are listed because a reader cannot
 * tell the difference, and they work.
 *
 * Niivue's own *keyboard* bindings (V, M, H/J/K/L and the rest) are deliberately
 * absent: Viewer.tsx strips the canvas's tabindex so they can never fire.
 *
 * Every row is a claim about the code; change the two together. */
const GROUPS: Group[] = [
  {
    title: 'Chat',
    rows: [
      { keys: ['Enter'], action: 'Send the message' },
      { keys: ['Shift + Enter'], action: 'Start a new line' },
    ],
  },
  {
    title: 'Files',
    hint: 'While the file tree has focus.',
    rows: [
      { keys: ['↑', '↓'], action: 'Move between rows' },
      { keys: ['Shift + ↑', 'Shift + ↓'], action: 'Extend the selection' },
      { keys: ['→', '←'], action: 'Open or close a folder' },
      { keys: ['Space'], action: 'Open the file, or fold the folder' },
      { keys: ['Enter'], action: 'Rename' },
      { keys: ['Backspace'], action: 'Delete the selection, after asking' },
      { keys: ['Ctrl / ⌘ + A'], action: 'Select everything' },
      { keys: ['Home', 'End'], action: 'Jump to the first or last row' },
      { keys: ['PageUp', 'PageDown'], action: 'Move a page at a time' },
      { keys: ['*'], action: 'Open every folder beside this one' },
      { keys: ['a…z'], action: 'Jump to a visible name starting with what you type' },
      { keys: ['Ctrl / ⌘ + click', 'Shift + click'], action: 'Add a row or a range to the selection' },
      { keys: ['Right-click'], action: 'Row menu; on a volume, Overlay on image' },
    ],
  },
  {
    title: 'Viewer — keys',
    hint: 'Click the image once first: these need the viewer to hold the keyboard.',
    rows: [
      { keys: ['↑', '↓', 'PageUp', 'PageDown'], action: 'Step one slice, 10 with Shift' },
      { keys: ['←', '→'], action: 'Step one frame of a 4D volume, 10 with Shift' },
      { keys: ['Space'], action: 'Play or pause a 4D volume' },
      { keys: ['o'], action: 'Hide or show the overlay' },
      { keys: ['+', '=', '-'], action: 'Zoom the slices in or out about the crosshair' },
      { keys: ['0'], action: 'Slices back to 100%, centred' },
    ],
  },
  {
    title: 'Viewer — mouse',
    hint: 'These need no focus.',
    rows: [
      { keys: ['Left drag'], action: 'Move the crosshair; turn the 3D view' },
      { keys: ['Shift / Ctrl + left drag', 'Middle drag'], action: 'Pan the slices' },
      { keys: ['Right drag'], action: 'Set the intensity window' },
      { keys: ['Double-click'], action: 'Reset the intensity window' },
      { keys: ['Wheel'], action: 'Step slices; zoom the 3D view' },
      { keys: ['Ctrl / ⌘ + wheel', 'Trackpad pinch'], action: 'Zoom the slices about the crosshair' },
      { keys: ['Drag a volume from Files'], action: 'Overlay it on the image' },
    ],
  },
  {
    title: 'Panels and windows',
    rows: [
      { keys: ['←', '→', '↑', '↓'], action: 'Move the focused panel separator' },
      { keys: ['Home', 'End'], action: 'Push that separator all the way over' },
      { keys: ['F6'], action: 'Go to the next separator' },
      { keys: ['Escape'], action: 'Close a menu, a popover or this window' },
      { keys: ['?'], action: 'Show this list' },
    ],
  },
]

/**
 * Every keyboard and mouse control the app has, in one list.
 *
 * It exists because almost none of them were discoverable: four strings in the
 * whole frontend hinted at a binding, and three of those were aria-labels, which
 * a sighted reader never sees.
 */
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
      <div
        ref={winRef}
        className="extwin scwin"
        role="dialog"
        aria-label="Keyboard shortcuts"
        tabIndex={-1}
      >
        <div className="panel-header">
          <span>Keyboard shortcuts</span>
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
                  <span className="sc-action">{r.action}</span>
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
