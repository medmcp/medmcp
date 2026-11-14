import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { ArchiveIcon, ChevronRightIcon, FileIcon, FolderIcon, UploadIcon } from './icons'

export type UploadKind = 'files' | 'folder' | 'zip'

const ITEMS: { kind: UploadKind; label: string; hint: string; icon: typeof FileIcon }[] = [
  { kind: 'files', label: 'Files…', hint: 'One or more files', icon: FileIcon },
  { kind: 'folder', label: 'Folder…', hint: 'A folder with its subfolders', icon: FolderIcon },
  {
    kind: 'zip',
    label: 'Zip archive…',
    hint: 'Unpacked into the folder. Use this when a folder upload hands over no files.',
    icon: ArchiveIcon,
  },
]

/**
 * The explorer's labelled "Upload" button and its menu (files, a folder, or a
 * zip the server unpacks). A menu button in the WAI-ARIA sense: Enter, Space or
 * ArrowDown opens it on the first item, the arrows move, Escape closes it and
 * returns focus to the button.
 */
export function UploadMenu({
  target,
  onPick,
  disabled,
}: {
  /** Where an upload would go, as shown to the person ("" = the workspace). */
  target: () => string
  onPick: (kind: UploadKind) => void
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [into, setInto] = useState('')
  const wrapRef = useRef<HTMLSpanElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([])

  const show = (focusIndex: number | null) => {
    setInto(target())
    setOpen(true)
    if (focusIndex !== null) requestAnimationFrame(() => itemRefs.current[focusIndex]?.focus())
  }
  const close = (refocus: boolean) => {
    setOpen(false)
    if (refocus) buttonRef.current?.focus()
  }

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    // Opened by a click, focus stays on the button: Escape must close it from there too.
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const onButtonKey = (e: ReactKeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      show(0)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      show(ITEMS.length - 1)
    }
  }

  const onMenuKey = (e: ReactKeyboardEvent) => {
    const current = itemRefs.current.findIndex((el) => el === document.activeElement)
    const focus = (i: number) => itemRefs.current[(i + ITEMS.length) % ITEMS.length]?.focus()
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      focus(current + 1)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      focus(current - 1)
    } else if (e.key === 'Home') {
      e.preventDefault()
      focus(0)
    } else if (e.key === 'End') {
      e.preventDefault()
      focus(ITEMS.length - 1)
    } else if (e.key === 'Escape') {
      e.preventDefault()
      e.stopPropagation()
      close(true)
    } else if (e.key === 'Tab') {
      close(false)
    }
  }

  return (
    <span className="upload-menu" ref={wrapRef}>
      <button
        ref={buttonRef}
        className="explorer-btn explorer-btn-primary"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls="explorer-upload-menu"
        title="Upload files, a folder or a zip archive into the selected folder"
        disabled={disabled}
        onClick={() => (open ? close(false) : show(null))}
        onKeyDown={onButtonKey}
      >
        <UploadIcon size={13} />
        Upload
        <span className="upload-menu-caret" aria-hidden="true">
          <ChevronRightIcon size={11} />
        </span>
      </button>
      {open && (
        <div
          id="explorer-upload-menu"
          className="upload-menu-list"
          role="menu"
          aria-label="Upload"
          onKeyDown={onMenuKey}
        >
          <div className="upload-menu-into" aria-hidden="true">
            Into {into ? `${into}/` : 'the workspace'}
          </div>
          {ITEMS.map((item, i) => {
            const Icon = item.icon
            return (
              <button
                key={item.kind}
                ref={(el) => {
                  itemRefs.current[i] = el
                }}
                role="menuitem"
                tabIndex={-1}
                title={item.hint}
                onClick={() => {
                  close(true)
                  onPick(item.kind)
                }}
              >
                <Icon size={14} />
                <span className="upload-menu-text">
                  <span>{item.label}</span>
                  <span className="upload-menu-hint">{item.hint}</span>
                </span>
              </button>
            )
          })}
        </div>
      )}
    </span>
  )
}
