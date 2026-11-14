import { memo, useEffect, useRef, useState } from 'react'
import type { DragEvent as ReactDragEvent } from 'react'
import { useConfirm } from '../confirm'
import { createPortal } from 'react-dom'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { Tree } from 'react-arborist'
import type { NodeApi, NodeRendererProps, TreeApi } from 'react-arborist'
import { deletePath, downloadPath, fetchTree, mkdir, renamePath, uploadFiles } from '../api'
import type { UploadItem } from '../api'
import { setDraggedFilePath } from '../dragState'
import { classify } from '../fileKinds'
import { DRAG_PATH_MIME, type TreeNode } from '../types'
import {
  FileIcon,
  FileTextIcon,
  FolderIcon,
  FolderOpenIcon,
  FolderPlusIcon,
  ImageIcon,
  LayersIcon,
  TableIcon,
  RefreshIcon,
  UploadIcon,
} from './icons'

interface FileExplorerProps {
  onOpenFile: (path: string) => void
  /** Bumped by the app when the workspace may have changed (agent/replay writes). */
  refreshSignal?: number
  /** Reports the selected file paths (multi-select via ctrl/shift-click). */
  onSelectionChange?: (paths: string[]) => void
  /** True while a separator is being dragged; defer tree resize until it ends. */
  isResizing?: boolean
}

function parentDir(path: string): string {
  const idx = path.lastIndexOf('/')
  return idx === -1 ? '' : path.slice(0, idx)
}

function joinPath(dir: string, name: string): string {
  return dir ? `${dir}/${name}` : name
}

/** Files from the picker, each going into `dir` (a picked folder keeps its subfolders). */
function pickedItems(list: FileList | null, dir: string): UploadItem[] {
  return Array.from(list ?? []).map((file) => {
    const rel = file.webkitRelativePath
    const sub = rel.includes('/') ? rel.slice(0, rel.lastIndexOf('/')) : ''
    return { file, dir: joinPath(dir, sub) }
  })
}

/** Whether a drag carries files from outside the browser (not one of our own rows). */
function carriesFiles(dt: DataTransfer): boolean {
  return Array.from(dt.types).includes('Files')
}

/**
 * The workspace folder a drop at this point lands in: the folder row under the
 * pointer, a file row's parent folder, or the root when over empty space.
 */
function dropTargetDir(target: EventTarget | null): string {
  const row = (target as HTMLElement | null)?.closest<HTMLElement>('[data-path]')
  const path = row?.dataset.path
  if (!path || path === NEW_FOLDER_ID) return ''
  return row?.dataset.dir !== undefined ? path : parentDir(path)
}

/** Walk a dropped entry (file or folder) into upload items under `dir`. */
async function walkEntry(entry: FileSystemEntry, dir: string, out: UploadItem[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) =>
      (entry as FileSystemFileEntry).file(resolve, reject),
    )
    out.push({ file, dir })
    return
  }
  if (!entry.isDirectory) return
  const reader = (entry as FileSystemDirectoryEntry).createReader()
  const sub = joinPath(dir, entry.name)
  // readEntries hands out batches (Chromium: 100 at a time) until an empty one.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
      reader.readEntries(resolve, reject),
    )
    if (batch.length === 0) break
    for (const child of batch) await walkEntry(child, sub, out)
  }
}

/**
 * Everything dropped from the desktop, as upload items under `dir`. The entries
 * must be taken from the DataTransfer synchronously, inside the drop event —
 * they are gone after it — so this is called with them already collected.
 */
async function droppedItems(entries: (FileSystemEntry | null)[], files: File[], dir: string) {
  const out: UploadItem[] = []
  if (entries.some((e) => e !== null)) {
    for (const entry of entries) if (entry) await walkEntry(entry, dir, out)
    return out
  }
  // No entry API (older browsers): plain files only, no folders.
  return files.map((file) => ({ file, dir }))
}

function FileTypeIcon({ name }: { name: string }) {
  switch (classify(name)) {
    case 'volume':
    case 'dicom':
      return <LayersIcon />
    case 'table':
    case 'sheet':
      return <TableIcon />
    case 'pdf':
    case 'markdown':
    case 'json':
    case 'text':
      return <FileTextIcon />
    case 'image':
      return <ImageIcon />
    default:
      return <FileIcon />
  }
}

function NodeRow({
  node,
  style,
  dragHandle,
  onMenu,
}: NodeRendererProps<TreeNode> & {
  onMenu: (e: ReactMouseEvent, node: NodeApi<TreeNode>) => void
}) {
  const isDir = !node.isLeaf
  // While the inline rename input is open the row must stop behaving like a row.
  // Leaving the drag handle attached makes a mouse-drag inside the input drag the
  // file instead of selecting text, and the row's own click/double-click would
  // toggle the folder or open the file when you click into the name or
  // double-click a word in it.
  const editing = node.isEditing
  return (
    <div
      ref={editing ? undefined : dragHandle}
      style={style}
      className={`tree-row${node.isSelected ? ' selected' : ''}`}
      data-path={node.data.id}
      data-dir={isDir ? '' : undefined}
      onClick={() => {
        if (!editing && isDir) node.toggle()
      }}
      onContextMenu={(e) => onMenu(e, node)}
      onDoubleClick={() => {
        if (!editing && !isDir) node.activate()
      }}
      onDragStart={(e) => {
        // Carry the workspace-relative path so drop targets outside the tree
        // (e.g. the viewer's overlay drop zone) can read it. This coexists
        // with react-arborist's own react-dnd move handling. The module-level
        // fallback covers the case where react-dnd intercepts dataTransfer.
        if (!isDir) {
          e.dataTransfer.setData(DRAG_PATH_MIME, node.data.id)
          setDraggedFilePath(node.data.id)
        }
      }}
      onDragEnd={() => setDraggedFilePath(null)}
    >
      <span className="tree-icon">
        {isDir ? node.isOpen ? <FolderOpenIcon /> : <FolderIcon /> : <FileTypeIcon name={node.data.name} />}
      </span>
      {node.isEditing ? (
        <input
          autoFocus
          defaultValue={node.data.name}
          // Belt and braces with the omitted drag handle above. Detaching the
          // handle should be enough, but the drag is wired by react-dnd rather
          // than by a plain attribute, so guard the element under the pointer
          // too: opt it out of being a drag source, keep mousedown from reaching
          // the row, and cancel any drag that still tries to begin here. Without
          // this, dragging across the text to select it drags the file instead.
          draggable={false}
          onMouseDown={(e) => e.stopPropagation()}
          onDragStart={(e) => {
            e.preventDefault()
            e.stopPropagation()
          }}
          onFocus={(e) => {
            // Preselect the name without its suffix, as a file manager does — a
            // rename almost always means changing the name, not ".nii.gz". Cut at
            // the first dot so double suffixes come off whole. A click into the
            // input still places the caret where you clicked.
            const dot = node.data.name.indexOf('.')
            e.currentTarget.setSelectionRange(0, dot > 0 ? dot : node.data.name.length)
          }}
          onBlur={() => node.reset()}
          onKeyDown={(e) => {
            if (e.key === 'Escape') node.reset()
            if (e.key === 'Enter') node.submit(e.currentTarget.value)
          }}
        />
      ) : (
        <span className="tree-name">
          {node.data.name}
        </span>
      )}
    </div>
  )
}

/** Drop the nodes with the given ids from a tree (for optimistic delete). */
function removeFromTree(nodes: TreeNode[], ids: Set<string>): TreeNode[] {
  return nodes
    .filter((n) => !ids.has(n.id))
    .map((n) => (n.children ? { ...n, children: removeFromTree(n.children, ids) } : n))
}

function insertIntoTree(nodes: TreeNode[], parentId: string, node: TreeNode): TreeNode[] {
  if (!parentId) return [node, ...nodes]
  return nodes.map((n) =>
    n.id === parentId
      ? { ...n, children: [node, ...(n.children ?? [])] }
      : n.children
        ? { ...n, children: insertIntoTree(n.children, parentId, node) }
        : n,
  )
}

/** The row a new folder is named in before it exists; never a real path. */
const NEW_FOLDER_ID = '\u0000new-folder'

/** Workspace file tree: open/rename/move/delete, upload (picker or desktop drop), download. */
// Memoized so a separator drag — which re-renders the panel content every frame
// via react-resizable-panels — doesn't re-run react-arborist's tree on each
// frame. Props from App are stable refs (refreshSignal changes only on fs
// activity, not during a drag).
export const FileExplorer = memo(function FileExplorer({
  onOpenFile,
  refreshSignal,
  onSelectionChange,
  isResizing,
}: FileExplorerProps) {
  const confirmDialog = useConfirm()
  const [data, setData] = useState<TreeNode[]>([])
  const [error, setError] = useState<string | null>(null)
  const [size, setSize] = useState({ width: 280, height: 400 })
  // The context menu: for a row, or for the empty space (node null) as file managers have it.
  const [menu, setMenu] = useState<{ x: number; y: number; node: NodeApi<TreeNode> | null } | null>(
    null,
  )
  const containerRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const treeRef = useRef<TreeApi<TreeNode> | null>(null)
  const newFolderParent = useRef('')
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(null)
  // Folder a desktop drag currently hovering the tree would upload into; null when none.
  const [dropDir, setDropDir] = useState<string | null>(null)
  // Read inside the mount-once observer effect without re-subscribing.
  const isResizingRef = useRef(isResizing)
  useEffect(() => {
    isResizingRef.current = isResizing
  }, [isResizing])

  const reload = () => {
    fetchTree()
      .then((tree) => {
        setData(tree)
        setError(null)
      })
      .catch((e: unknown) => setError(String(e)))
  }

  useEffect(reload, [])

  // Tool calls / replay steps bump refreshSignal; reload shortly after. The
  // cleanup-timer pattern collapses a burst of signals into one fetch.
  useEffect(() => {
    if (!refreshSignal) return
    const timer = window.setTimeout(reload, 300)
    return () => window.clearTimeout(timer)
  }, [refreshSignal])

  // Catch changes made outside the app (terminal, scripts) on tab focus.
  useEffect(() => {
    window.addEventListener('focus', reload)
    return () => window.removeEventListener('focus', reload)
  }, [])

  // Resize the (virtualized) tree to its container. While a separator is being
  // dragged we skip entirely — re-rendering react-arborist mid-drag is what made
  // those drags janky — and resync once when the drag ends (the effect below).
  // The trailing debounce covers non-drag resizes (window, drawer).
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let timer: number | null = null
    const obs = new ResizeObserver(() => {
      // Clear first so a resize scheduled in the frame before `isResizing`
      // commits can't fire mid-drag; then skip entirely while dragging.
      if (timer != null) window.clearTimeout(timer)
      if (isResizingRef.current) return
      timer = window.setTimeout(() => {
        timer = null
        setSize({ width: el.clientWidth, height: el.clientHeight })
      }, 120)
    })
    obs.observe(el)
    return () => {
      obs.disconnect()
      if (timer != null) window.clearTimeout(timer)
    }
  }, [])

  // A separator drag changed our size while the observer was skipping; sync once
  // on the next frame after it ends (when the layout has settled).
  useEffect(() => {
    if (isResizing) return
    const id = requestAnimationFrame(() => {
      const el = containerRef.current
      if (el) setSize({ width: el.clientWidth, height: el.clientHeight })
    })
    return () => cancelAnimationFrame(id)
  }, [isResizing])

  const run = (op: Promise<void>) => {
    op.then(reload).catch((e: unknown) => {
      setError(String(e))
      reload()
    })
  }

  // Folder the open picker uploads into; read when the picker returns.
  const pickDir = useRef('')
  const pickFiles = (dir: string) => {
    pickDir.current = dir
    fileInputRef.current?.click()
  }
  // `webkitdirectory` is not in React's input typings; set it on the element.
  useEffect(() => {
    folderInputRef.current?.setAttribute('webkitdirectory', '')
  }, [])
  const pickFolder = (dir: string) => {
    pickDir.current = dir
    folderInputRef.current?.click()
  }

  const startUpload = (items: UploadItem[]) => {
    if (items.length === 0) return
    setError(null)
    setUploading({ done: 0, total: items.length })
    void uploadFiles(items, (done, total) => setUploading({ done, total }))
      .then((failed) => {
        if (failed > 0) setError(`failed to upload ${failed} of ${items.length} file(s)`)
      })
      .finally(() => {
        setUploading(null)
        reload()
      })
  }

  // Files and folders dragged in from the desktop. A picker cannot offer files
  // and folders at once, so folders come in this way: dropped on a folder row
  // they go inside it, on a file row beside it, on empty space into the root.
  // react-arborist's own row drags carry no 'Files' type and are left alone.
  const onDragOver = (e: ReactDragEvent<HTMLDivElement>) => {
    if (!carriesFiles(e.dataTransfer)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    const dir = dropTargetDir(e.target)
    if (dir !== dropDir) setDropDir(dir)
  }
  const onDragLeave = (e: ReactDragEvent<HTMLDivElement>) => {
    if (e.currentTarget.contains(e.relatedTarget as Node | null)) return
    setDropDir(null)
  }
  const onDrop = (e: ReactDragEvent<HTMLDivElement>) => {
    if (!carriesFiles(e.dataTransfer)) return
    e.preventDefault()
    setDropDir(null)
    const dir = dropTargetDir(e.target)
    // Both must be read now, synchronously: the DataTransfer is emptied after the event.
    const entries = Array.from(e.dataTransfer.items).map((item) => item.webkitGetAsEntry())
    const files = Array.from(e.dataTransfer.files)
    void droppedItems(entries, files, dir).then(startUpload)
  }

  // Delete one or more paths: drop them from the tree immediately so they vanish
  // without waiting on a full refetch, fire the deletes together, then reconcile
  // with a single reload (which also restores anything whose delete failed).
  // A new folder is named inline, like a rename: a placeholder row goes into the
  // tree and opens in edit mode; onRename creates the folder (or drops the row).
  /** The folder the selection points at: a selected folder itself, a selected file's parent. */
  const selectedDir = (): string => {
    const tree = treeRef.current
    if (!tree) return ''
    const focused = tree.focusedNode
    const node = focused?.isSelected ? focused : tree.selectedNodes.at(-1)
    if (!node || node.data.id === NEW_FOLDER_ID) return ''
    return node.isLeaf ? parentDir(node.data.id) : node.data.id
  }

  const startNewFolder = (parentId: string) => {
    const placeholder: TreeNode = { id: NEW_FOLDER_ID, name: '', children: [] }
    newFolderParent.current = parentId
    setData((d) => insertIntoTree(removeFromTree(d, new Set([NEW_FOLDER_ID])), parentId, placeholder))
    // The row exists only after the re-render; arborist's own create() defers the same way.
    setTimeout(() => {
      const tree = treeRef.current
      if (!tree) return
      if (parentId) tree.open(parentId)
      void tree.edit(NEW_FOLDER_ID).then((res) => {
        if (res.cancelled) setData((d) => removeFromTree(d, new Set([NEW_FOLDER_ID])))
      })
    })
  }

  const deleteMany = (ids: string[]) => {
    if (ids.length === 0) return
    const idSet = new Set(ids)
    setData((d) => removeFromTree(d, idSet))
    void Promise.allSettled(ids.map((id) => deletePath(id))).then((results) => {
      const failed = results.filter((r) => r.status === 'rejected').length
      if (failed > 0) setError(`failed to delete ${failed} item(s)`)
      reload()
    })
  }

  // Any click/Escape outside the menu dismisses it. The menu is portaled to
  // <body> (so it can't be clipped by a panel), which means it's outside the
  // React root's event delegation — stopPropagation on its mousedown wouldn't
  // reliably stop this window listener. Instead, skip dismissal when the event
  // originates inside the menu (a contains-check, position-independent); its
  // buttons close it explicitly via menuAction.
  useEffect(() => {
    if (!menu) return
    const close = (e: Event) => {
      if (menuRef.current?.contains(e.target as Node)) return
      setMenu(null)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenu(null)
    }
    window.addEventListener('mousedown', close)
    window.addEventListener('blur', close)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', close)
      window.removeEventListener('blur', close)
      window.removeEventListener('keydown', onKey)
    }
  }, [menu])

  const onActivate = (node: NodeApi<TreeNode>) => {
    if (node.isLeaf) onOpenFile(node.data.id)
  }

  const openMenu = (e: ReactMouseEvent, node: NodeApi<TreeNode>) => {
    e.preventDefault()
    e.stopPropagation()
    // Keep an existing multi-selection if the right-clicked row is part of it, so
    // Delete can act on every selected row; otherwise select just this one.
    if (!node.isSelected) node.select()
    setMenu({ x: e.clientX, y: e.clientY, node })
  }

  // Right-click on empty space: the background menu (New folder, …), as in any
  // file manager. A row's own handler stops propagation, so this only sees the
  // space between and below the rows; the click also drops the selection.
  const openBackgroundMenu = (e: ReactDragEvent<HTMLDivElement> | ReactMouseEvent) => {
    if ((e.target as HTMLElement | null)?.closest('[data-path]')) return
    e.preventDefault()
    treeRef.current?.deselectAll()
    setMenu({ x: e.clientX, y: e.clientY, node: null })
  }

  const menuAction = (fn: () => void) => () => {
    setMenu(null)
    fn()
  }

  // Hoisted so the row branch below narrows once, closures included.
  const menuNode = menu?.node ?? null

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Files</span>
        <span className="panel-actions">
          <button
            className="btn-icon"
            aria-label="New folder"
            onClick={() => startNewFolder(selectedDir())}
          >
            <FolderPlusIcon />
          </button>
          <button
            className="btn-icon"
            aria-label="Upload files"
            onClick={() => pickFiles(selectedDir())}
          >
            <UploadIcon />
          </button>
          <button className="btn-icon" aria-label="Refresh" onClick={reload}>
            <RefreshIcon />
          </button>
        </span>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            startUpload(pickedItems(e.target.files, pickDir.current))
            e.target.value = ''
          }}
        />
        <input
          ref={folderInputRef}
          type="file"
          hidden
          onChange={(e) => {
            startUpload(pickedItems(e.target.files, pickDir.current))
            e.target.value = ''
          }}
        />
      </div>
      {uploading ? (
        <div className="panel-status" role="status">
          Uploading {uploading.done} of {uploading.total} file(s)…
        </div>
      ) : (
        dropDir !== null && (
          <div className="panel-status" role="status">
            Drop to upload into {dropDir ? `${dropDir}/` : 'the workspace'}
          </div>
        )
      )}
      {error && <div className="panel-error">{error}</div>}
      <div
        className={`panel-body${dropDir !== null ? ' drop-target' : ''}`}
        ref={containerRef}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onContextMenu={openBackgroundMenu}
      >
        {/* Every other panel says what to do when it is empty; this one showed
            a blank rectangle, which reads as a failure to load. */}
        {data.length === 0 && !error && (
          <div className="viewer-message">
            This workspace is empty. Upload files, or drop files or folders here.
          </div>
        )}
        <Tree<TreeNode>
          ref={treeRef}
          data={data}
          width={size.width}
          height={size.height}
          rowHeight={27}
          indent={14}
          openByDefault={false}
          onActivate={onActivate}
          onSelect={(nodes) =>
            onSelectionChange?.(nodes.filter((n) => n.isLeaf).map((n) => n.data.id))
          }
          onRename={({ node, name }) => {
            if (node.data.id === NEW_FOLDER_ID) {
              setData((d) => removeFromTree(d, new Set([NEW_FOLDER_ID])))
              if (name.trim()) run(mkdir(joinPath(newFolderParent.current, name.trim())))
              return
            }
            run(renamePath(node.data.id, joinPath(parentDir(node.data.id), name)))
          }}
          onMove={({ dragNodes, parentNode }) => {
            const destDir = parentNode ? parentNode.data.id : ''
            for (const node of dragNodes) {
              run(renamePath(node.data.id, joinPath(destDir, node.data.name)))
            }
          }}
          onDelete={({ nodes }) => {
            if (nodes.length === 0) return
            const label = nodes.length > 1 ? `${nodes.length} items` : (nodes[0]?.data.name ?? '')
            void confirmDialog(`Delete ${label}?`).then(
              (ok) => ok && deleteMany(nodes.map((n) => n.data.id)),
            )
          }}
        >
          {(props) => <NodeRow {...props} onMenu={openMenu} />}
        </Tree>
        {menu &&
          createPortal(
            <div
              ref={menuRef}
              className="ctx-menu"
              style={{
                left: Math.min(menu.x, window.innerWidth - 170),
                top: Math.min(menu.y, window.innerHeight - 190),
              }}
              onContextMenu={(e) => e.preventDefault()}
            >
            {menuNode === null ? (
              <>
                <button
                  onClick={() => {
                    setMenu(null)
                    startNewFolder('')
                  }}
                >
                  New folder
                </button>
                <button
                  onClick={() => {
                    setMenu(null)
                    pickFiles('')
                  }}
                >
                  Upload files…
                </button>
                <button
                  onClick={() => {
                    setMenu(null)
                    pickFolder('')
                  }}
                >
                  Upload folder…
                </button>
                <button onClick={menuAction(reload)}>Refresh</button>
              </>
            ) : (
              <>
                {menuNode.isLeaf ? (
                  <>
                    <button onClick={menuAction(() => onOpenFile(menuNode.data.id))}>Open</button>
                    <button onClick={menuAction(() => downloadPath(menuNode.data.id))}>
                      Download
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => {
                        setMenu(null)
                        startNewFolder(menuNode.data.id)
                      }}
                    >
                      New folder inside
                    </button>
                    <button onClick={menuAction(() => downloadPath(menuNode.data.id))}>
                      Download as .zip
                    </button>
                  </>
                )}
                <button onClick={menuAction(() => void menuNode.edit())}>Rename</button>
                {(() => {
                  const sel = menuNode.tree.selectedNodes
                  const targets =
                    sel.length > 1 && sel.some((n) => n.id === menuNode.id) ? sel : [menuNode]
                  const label = targets.length > 1 ? `${targets.length} items` : menuNode.data.name
                  return (
                    <button
                      className="danger"
                      onClick={menuAction(() => {
                        void confirmDialog(`Delete ${label}?`).then(
                          (ok) => ok && deleteMany(targets.map((n) => n.data.id)),
                        )
                      })}
                    >
                      {targets.length > 1 ? `Delete ${targets.length} items` : 'Delete'}
                    </button>
                  )
                })()}
              </>
            )}
            </div>,
            document.body,
          )}
      </div>
    </div>
  )
})
