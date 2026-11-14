import { createContext, memo, useContext, useEffect, useRef, useState } from 'react'
import type { DragEvent as ReactDragEvent } from 'react'
import { useConfirm } from '../confirm'
import { createPortal } from 'react-dom'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { Tree } from 'react-arborist'
import type { NodeApi, NodeRendererProps, TreeApi } from 'react-arborist'
import {
  deletePath,
  downloadPath,
  fetchDir,
  mkdir,
  renamePath,
  uploadFiles,
  uploadZip,
} from '../api'
import type { DirListing, UploadFailure, UploadItem } from '../api'
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
  XIcon,
} from './icons'
import { UploadMenu } from './UploadMenu'
import type { UploadKind } from './UploadMenu'

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
  if (!name) return dir
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
  dropDir,
  loading,
}: NodeRendererProps<TreeNode> & {
  onMenu: (e: ReactMouseEvent, node: NodeApi<TreeNode>) => void
  /** Folder a desktop drag would upload into (highlighted), or null. */
  dropDir: string | null
  /** Folders whose entries are being fetched. */
  loading: ReadonlySet<string>
}) {
  if (node.data.more) {
    return (
      <div style={style} className="tree-row tree-more" aria-disabled="true">
        <span className="tree-name">{node.data.name}</span>
      </div>
    )
  }
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
      className={`tree-row${node.isSelected ? ' selected' : ''}${
        isDir && dropDir === node.data.id ? ' drop-into' : ''
      }`}
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
      {isDir && node.isOpen && loading.has(node.data.id) && (
        <span className="tree-loading">loading…</span>
      )}
    </div>
  )
}

/** What every row needs from the explorer, passed by context rather than props. */
interface RowExtras {
  onMenu: (e: ReactMouseEvent, node: NodeApi<TreeNode>) => void
  dropDir: string | null
  loading: ReadonlySet<string>
}

const RowContext = createContext<RowExtras>({
  onMenu: () => undefined,
  dropDir: null,
  loading: new Set(),
})

/**
 * The row renderer handed to react-arborist. One stable component: an inline
 * `(props) => <NodeRow …/>` is a new component type on every explorer render,
 * so React remounts every row each time — and a row remounted between mousedown
 * and mouseup loses the click (the first click on a folder did not open it).
 */
function Row(props: NodeRendererProps<TreeNode>) {
  return <NodeRow {...props} {...useContext(RowContext)} />
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
/** Prefix of the row a capped listing ends with; never a real path. */
const MORE_PREFIX = '\u0000more:'

function findNode(nodes: TreeNode[], id: string): TreeNode | undefined {
  for (const n of nodes) {
    if (n.id === id) return n
    if (n.children?.length && id.startsWith(`${n.id}/`)) {
      const hit = findNode(n.children, id)
      if (hit) return hit
    }
  }
  return undefined
}

/**
 * A folder's fresh entries, keeping what is already known below them: a loaded
 * subfolder keeps its children (so open folders stay open and filled), and the
 * new-folder row being named stays on top.
 */
function mergeChildren(old: TreeNode[] | undefined, fresh: TreeNode[]): TreeNode[] {
  const prev = new Map((old ?? []).map((n) => [n.id, n]))
  const merged = fresh.map((f) => {
    const o = prev.get(f.id)
    return f.children && o?.loaded ? { ...f, children: o.children, loaded: true } : f
  })
  const placeholder = prev.get(NEW_FOLDER_ID)
  return placeholder ? [placeholder, ...merged] : merged
}

/** The tree with one folder's level replaced by a fresh listing of it. */
function withListing(nodes: TreeNode[], listing: DirListing): TreeNode[] {
  const kids = [...listing.children]
  const hidden = listing.total - listing.children.length
  if (hidden > 0) {
    kids.push({ id: MORE_PREFIX + listing.path, name: `${hidden} more items not shown`, more: true })
  }
  if (!listing.path) return mergeChildren(nodes, kids)
  return nodes.map((n) =>
    n.id === listing.path
      ? { ...n, children: mergeChildren(n.children, kids), loaded: true, hasChildren: kids.length > 0 }
      : n.children?.length && listing.path.startsWith(`${n.id}/`)
        ? { ...n, children: withListing(n.children, listing) }
        : n,
  )
}

/**
 * The loaded folders shown open (reachable through open folders), and the tree
 * with every loaded folder that is not marked unloaded, so opening it again
 * fetches it afresh rather than showing what it held minutes ago.
 */
function splitOpen(
  nodes: TreeNode[],
  isOpen: (id: string) => boolean,
  open: string[],
): TreeNode[] {
  return nodes.map((n) => {
    if (!n.children || !n.loaded) return n
    if (!isOpen(n.id)) return { ...n, loaded: false }
    open.push(n.id)
    return { ...n, children: splitOpen(n.children, isOpen, open) }
  })
}

function depth(path: string): number {
  return path ? path.split('/').length : 0
}

function humanBytes(n: number): string {
  if (n < 1024) return `${n} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

/** What the progress line shows while an upload runs. */
interface UploadState {
  label: string
  doneFiles?: number
  totalFiles?: number
  sentBytes: number
  totalBytes: number
  /** The bytes are all sent; the server is still working (unpacking a zip). */
  serverBusy?: boolean
}

/** A failure to show, with the per-file reasons when there are any. */
interface ExplorerError {
  message: string
  details?: string[]
  /** Reported by a failed listing; the next working listing clears it. */
  fromLoad?: boolean
}

const NO_FILES_HINT =
  'The browser handed over no files from that folder. Try Upload → Zip archive…, ' +
  'or drag the folder into the file list.'

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
  // Read by async loaders that would otherwise see the data of their closure.
  const dataRef = useRef<TreeNode[]>([])
  useEffect(() => {
    dataRef.current = data
  }, [data])
  const [rootLoaded, setRootLoaded] = useState(false)
  const [loading, setLoading] = useState<ReadonlySet<string>>(new Set())
  const [error, setErrorState] = useState<ExplorerError | null>(null)
  const setError = (message: string | null, details?: string[]) =>
    setErrorState(message === null ? null : { message, details })
  const [notice, setNotice] = useState<string | null>(null)
  const [size, setSize] = useState({ width: 280, height: 400 })
  // The context menu: for a row, or for the empty space (node null) as file managers have it.
  const [menu, setMenu] = useState<{ x: number; y: number; node: NodeApi<TreeNode> | null } | null>(
    null,
  )
  const containerRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const zipInputRef = useRef<HTMLInputElement>(null)
  const treeRef = useRef<TreeApi<TreeNode> | null>(null)
  const newFolderParent = useRef('')
  const [uploading, setUploading] = useState<UploadState | null>(null)
  // Bumped to name a freshly inserted new-folder row once it is rendered.
  const [pendingEdit, setPendingEdit] = useState(0)
  // Folder a desktop drag currently hovering the tree would upload into; null when none.
  const [dropDir, setDropDir] = useState<string | null>(null)
  // Read inside the mount-once observer effect without re-subscribing.
  const isResizingRef = useRef(isResizing)
  useEffect(() => {
    isResizingRef.current = isResizing
  }, [isResizing])

  /**
   * Fetch the given folders' levels and merge them in, parents first, so a
   * subfolder's listing lands in the node its parent's listing kept. A folder
   * that has vanished meanwhile is skipped quietly.
   */
  const loadDirs = async (paths: string[]): Promise<void> => {
    const unique = [...new Set(paths)].sort((a, b) => depth(a) - depth(b))
    if (unique.length === 0) return
    setLoading((prev) => new Set([...prev, ...unique]))
    const results = await Promise.allSettled(unique.map((p) => fetchDir(p)))
    setLoading((prev) => new Set([...prev].filter((p) => !unique.includes(p))))
    const listings: DirListing[] = []
    let failure: string | null = null
    results.forEach((r) => {
      if (r.status === 'fulfilled') listings.push(r.value)
      else if (!String(r.reason).includes('folder not found')) failure = String(r.reason)
    })
    setData((d) => listings.reduce(withListing, d))
    if (unique.includes('')) {
      setRootLoaded(true)
      // A listing that works again clears the failure an earlier one reported.
      if (failure === null) setErrorState((e) => (e?.fromLoad ? null : e))
    }
    if (failure !== null) setErrorState({ message: failure, fromLoad: true })
  }

  /** Refresh the root and every folder shown open; closed folders refetch when next opened. */
  const reload = () => {
    const tree = treeRef.current
    const open: string[] = []
    const next = splitOpen(dataRef.current, (id) => tree?.isOpen(id) ?? false, open)
    dataRef.current = next
    setData(next)
    void loadDirs(['', ...open])
  }

  /** A folder was opened: fetch its level the first time (or again once stale). */
  const onToggle = (id: string) => {
    const tree = treeRef.current
    const node = tree?.get(id)
    if (!tree?.isOpen(id) || !node || node.isLeaf || node.data.loaded) return
    void loadDirs([id])
  }

  // The root level, once mounted (deferred a tick like the refresh below, so the
  // loader's state updates never run inside the effect itself).
  useEffect(() => {
    const timer = window.setTimeout(() => void loadDirs(['']), 0)
    return () => window.clearTimeout(timer)
  }, [])

  // The latest reload, for the subscriptions below that are made once.
  const reloadRef = useRef(reload)
  useEffect(() => {
    reloadRef.current = reload
  })

  // Tool calls / replay steps bump refreshSignal; reload shortly after. The
  // cleanup-timer pattern collapses a burst of signals into one fetch.
  useEffect(() => {
    if (!refreshSignal) return
    const timer = window.setTimeout(() => reloadRef.current(), 300)
    return () => window.clearTimeout(timer)
  }, [refreshSignal])

  // Catch changes made outside the app (terminal, scripts) on tab focus.
  useEffect(() => {
    const onFocus = () => reloadRef.current()
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
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

  /** Run a change, then refresh only the folders it touched. */
  const run = (op: Promise<void>, dirs: string[]) => {
    op.then(() => loadDirs(dirs)).catch((e: unknown) => {
      setError(String(e))
      void loadDirs(dirs)
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
  const pickZip = (dir: string) => {
    pickDir.current = dir
    zipInputRef.current?.click()
  }
  const pick = (kind: UploadKind, dir: string) =>
    kind === 'files' ? pickFiles(dir) : kind === 'folder' ? pickFolder(dir) : pickZip(dir)

  /** After an upload into `dir`: show it open, with its fresh entries. */
  const revealDir = (dir: string) => {
    void loadDirs([dir]).then(() => {
      if (dir) treeRef.current?.open(dir)
    })
  }

  const startUpload = (items: UploadItem[], dir: string) => {
    if (items.length === 0) return
    setError(null)
    setNotice(null)
    const totalBytes = items.reduce((sum, i) => sum + i.file.size, 0)
    const label = `Uploading into ${dir ? `${dir}/` : 'the workspace'}`
    setUploading({ label, doneFiles: 0, totalFiles: items.length, sentBytes: 0, totalBytes })
    void uploadFiles(items, (p) => setUploading({ label, ...p }))
      .then((failures: UploadFailure[]) => {
        if (failures.length > 0) {
          setError(
            `${failures.length} of ${items.length} file(s) did not upload:`,
            failures.map((f) => `${f.path}: ${f.reason}`),
          )
        } else {
          setNotice(`Uploaded ${items.length} file(s) (${humanBytes(totalBytes)}).`)
        }
      })
      .finally(() => {
        setUploading(null)
        revealDir(dir)
      })
  }

  const startZipUpload = (file: File, dir: string) => {
    setError(null)
    setNotice(null)
    const totalBytes = file.size
    const label = `Uploading ${file.name}`
    setUploading({ label, sentBytes: 0, totalBytes })
    void uploadZip(file, dir, (sent) =>
      setUploading({
        label: sent >= totalBytes ? `Unpacking ${file.name} on the server…` : label,
        sentBytes: sent,
        totalBytes,
        serverBusy: sent >= totalBytes,
      }),
    )
      .then((res) => {
        const into = res.path ? `${res.path}/` : 'the workspace'
        setNotice(
          `Unpacked ${res.files} file(s) (${humanBytes(res.bytes)}) from ${file.name} into ${into}.`,
        )
      })
      .catch((e: unknown) => {
        setError(`${file.name} was not unpacked: ${e instanceof Error ? e.message : String(e)}`)
      })
      .finally(() => {
        setUploading(null)
        revealDir(dir)
      })
  }

  /** Files from a picker; a folder picker that hands over nothing says so. */
  const onPicked = (list: FileList | null, fromFolder: boolean) => {
    const items = pickedItems(list, pickDir.current)
    if (items.length === 0) {
      if (fromFolder) setError(NO_FILES_HINT)
      return
    }
    startUpload(items, pickDir.current)
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
    const hadEntries = entries.length > 0 || files.length > 0
    void droppedItems(entries, files, dir)
      .then((items) => {
        if (items.length === 0 && hadEntries) setError(NO_FILES_HINT.replace(/, or drag.*$/, '.'))
        else startUpload(items, dir)
      })
      .catch((e: unknown) => setError(`could not read the dropped items: ${String(e)}`))
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
    if (node.data.more) return node.data.id.slice(MORE_PREFIX.length)
    return node.isLeaf ? parentDir(node.data.id) : node.data.id
  }

  const startNewFolder = async (parentId: string) => {
    setMenu(null)
    newFolderParent.current = parentId
    const parent = parentId ? findNode(dataRef.current, parentId) : undefined
    if (parent && !parent.loaded) await loadDirs([parentId])
    const placeholder: TreeNode = { id: NEW_FOLDER_ID, name: '', children: [] }
    setData((d) => insertIntoTree(removeFromTree(d, new Set([NEW_FOLDER_ID])), parentId, placeholder))
    setPendingEdit((n) => n + 1)
  }

  // The new-folder row exists only once the tree has rendered it: then open its
  // parent, bring the row into view (at the top of its folder) and name it there.
  // Without the scroll it could sit far off-screen in a long workspace.
  useEffect(() => {
    if (!pendingEdit) return
    const frame = requestAnimationFrame(() => {
      const tree = treeRef.current
      if (!tree) return
      const parentId = newFolderParent.current
      if (parentId) {
        tree.openParents(parentId)
        tree.open(parentId)
      }
      void tree.scrollTo(NEW_FOLDER_ID, 'center')
      void tree.edit(NEW_FOLDER_ID).then((res) => {
        if (res.cancelled) setData((d) => removeFromTree(d, new Set([NEW_FOLDER_ID])))
      })
    })
    return () => cancelAnimationFrame(frame)
  }, [pendingEdit])

  const deleteMany = (ids: string[]) => {
    if (ids.length === 0) return
    const idSet = new Set(ids)
    setData((d) => removeFromTree(d, idSet))
    void Promise.allSettled(ids.map((id) => deletePath(id))).then((results) => {
      const failed = results.flatMap((r, i) =>
        r.status === 'rejected' ? [`${ids[i]}: ${String(r.reason).replace(/^Error: /, '')}`] : [],
      )
      if (failed.length > 0) setError(`${failed.length} item(s) were not deleted:`, failed)
      void loadDirs(ids.map(parentDir))
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
    if (node.isLeaf && !node.data.more) onOpenFile(node.data.id)
  }

  const openMenu = (e: ReactMouseEvent, node: NodeApi<TreeNode>) => {
    e.preventDefault()
    e.stopPropagation()
    if (node.data.more) return
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
    <div className="panel explorer-panel">
      <div className="panel-header">
        <span>Files</span>
        <span className="panel-actions explorer-actions">
          <button
            className="explorer-btn"
            aria-label="New folder"
            title="Create a folder inside the selected folder"
            onClick={() => void startNewFolder(selectedDir())}
          >
            <FolderPlusIcon size={13} />
            <span className="explorer-btn-label">New folder</span>
          </button>
          <UploadMenu
            target={selectedDir}
            disabled={uploading !== null}
            onPick={(kind) => pick(kind, selectedDir())}
          />
          <button
            className="btn-icon"
            aria-label="Refresh"
            title="Refresh the file list"
            onClick={reload}
          >
            <RefreshIcon />
          </button>
        </span>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            onPicked(e.target.files, false)
            e.target.value = ''
          }}
        />
        <input
          ref={folderInputRef}
          type="file"
          hidden
          onChange={(e) => {
            onPicked(e.target.files, true)
            e.target.value = ''
          }}
        />
        <input
          ref={zipInputRef}
          type="file"
          accept=".zip,application/zip,application/x-zip-compressed"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) startZipUpload(file, pickDir.current)
            e.target.value = ''
          }}
        />
      </div>
      {uploading && (
        <div className="panel-status upload-progress" role="status" aria-live="polite">
          <div>
            {uploading.label}
            {!uploading.serverBusy && (
              <>
                {uploading.totalFiles !== undefined &&
                  ` · ${uploading.doneFiles ?? 0} of ${uploading.totalFiles} file(s)`}
                {` · ${humanBytes(uploading.sentBytes)} of ${humanBytes(uploading.totalBytes)}`}
              </>
            )}
          </div>
          <progress
            max={uploading.serverBusy ? undefined : Math.max(uploading.totalBytes, 1)}
            value={uploading.serverBusy ? undefined : uploading.sentBytes}
            aria-label="Upload progress"
          />
        </div>
      )}
      {notice && !uploading && !error && (
        <div className="panel-status panel-notice" role="status">
          <span>{notice}</span>
          <button className="btn-icon" aria-label="Dismiss" onClick={() => setNotice(null)}>
            <XIcon size={12} />
          </button>
        </div>
      )}
      {error && (
        <div className="panel-error" role="alert">
          <div className="panel-error-head">
            <span>{error.message}</span>
            <button
              className="btn-icon"
              aria-label="Dismiss"
              onClick={() => setErrorState(null)}
            >
              <XIcon size={12} />
            </button>
          </div>
          {error.details && error.details.length > 0 && (
            <ul className="panel-error-list">
              {error.details.slice(0, 8).map((d) => (
                <li key={d}>{d}</li>
              ))}
              {error.details.length > 8 && <li>… and {error.details.length - 8} more</li>}
            </ul>
          )}
        </div>
      )}
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
        {rootLoaded && data.length === 0 && !error && (
          <div className="viewer-message">
            This workspace is empty. Use <strong>Upload</strong> above, or drop files or folders
            here.
          </div>
        )}
        {dropDir !== null && (
          <div className="drop-hint" aria-hidden="true">
            Drop to upload into {dropDir ? `${dropDir}/` : 'the workspace'}
          </div>
        )}
        <RowContext.Provider value={{ onMenu: openMenu, dropDir, loading }}>
        <Tree<TreeNode>
          ref={treeRef}
          data={data}
          width={size.width}
          height={size.height}
          rowHeight={27}
          indent={14}
          openByDefault={false}
          onToggle={onToggle}
          disableDrag={(d) => !!d.more || d.id === NEW_FOLDER_ID}
          disableEdit={(d) => !!d.more}
          disableSelect={(d) => !!d.more}
          onActivate={onActivate}
          onSelect={(nodes) =>
            onSelectionChange?.(nodes.filter((n) => n.isLeaf).map((n) => n.data.id))
          }
          onRename={({ node, name }) => {
            if (node.data.id === NEW_FOLDER_ID) {
              setData((d) => removeFromTree(d, new Set([NEW_FOLDER_ID])))
              const parent = newFolderParent.current
              if (name.trim()) run(mkdir(joinPath(parent, name.trim())), [parent])
              return
            }
            run(renamePath(node.data.id, joinPath(parentDir(node.data.id), name)), [
              parentDir(node.data.id),
            ])
          }}
          onMove={({ dragNodes, parentNode }) => {
            const destDir = parentNode ? parentNode.data.id : ''
            for (const node of dragNodes) {
              run(renamePath(node.data.id, joinPath(destDir, node.data.name)), [
                parentDir(node.data.id),
                destDir,
              ])
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
          {Row}
        </Tree>
        </RowContext.Provider>
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
                <button onClick={() => void startNewFolder('')}>New folder</button>
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
                <button
                  onClick={() => {
                    setMenu(null)
                    pickZip('')
                  }}
                >
                  Upload zip archive…
                </button>
                <button
                  onClick={() => {
                    setMenu(null)
                    reload()
                  }}
                >
                  Refresh
                </button>
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
                    <button onClick={() => void startNewFolder(menuNode.data.id)}>
                      New folder inside
                    </button>
                    <button
                      onClick={() => {
                        setMenu(null)
                        pickFiles(menuNode.data.id)
                      }}
                    >
                      Upload files here…
                    </button>
                    <button
                      onClick={() => {
                        setMenu(null)
                        pickFolder(menuNode.data.id)
                      }}
                    >
                      Upload folder here…
                    </button>
                    <button
                      onClick={() => {
                        setMenu(null)
                        pickZip(menuNode.data.id)
                      }}
                    >
                      Upload zip archive here…
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
