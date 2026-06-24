import { memo, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { Tree } from 'react-arborist'
import type { NodeApi, NodeRendererProps } from 'react-arborist'
import { deletePath, fetchTree, mkdir, renamePath, uploadFile } from '../api'
import { setDraggedFilePath } from '../dragState'
import { DRAG_PATH_MIME, type TreeNode } from '../types'
import {
  FileIcon,
  FileTextIcon,
  FolderIcon,
  FolderOpenIcon,
  FolderPlusIcon,
  ImageIcon,
  LayersIcon,
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

function FileTypeIcon({ name }: { name: string }) {
  const lower = name.toLowerCase()
  if (/\.(nii|nii\.gz|mgz|mgh|nrrd|nhdr|mha|mhd|dcm)$/.test(lower)) return <LayersIcon />
  if (/\.(pdf|md|txt|log)$/.test(lower)) return <FileTextIcon />
  if (/\.(png|jpe?g|gif|svg|webp|bmp)$/.test(lower)) return <ImageIcon />
  return <FileIcon />
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
  return (
    <div
      ref={dragHandle}
      style={style}
      className={`tree-row${node.isSelected ? ' selected' : ''}`}
      onClick={() => {
        if (isDir) node.toggle()
      }}
      onContextMenu={(e) => onMenu(e, node)}
      onDoubleClick={() => {
        if (!isDir) node.activate()
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
          onBlur={() => node.reset()}
          onKeyDown={(e) => {
            if (e.key === 'Escape') node.reset()
            if (e.key === 'Enter') node.submit(e.currentTarget.value)
          }}
        />
      ) : (
        <span className="tree-name" title={node.data.id}>
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

/** Workspace file tree with open/rename/move/delete/upload. */
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
  const [data, setData] = useState<TreeNode[]>([])
  const [error, setError] = useState<string | null>(null)
  const [size, setSize] = useState({ width: 280, height: 400 })
  const [menu, setMenu] = useState<{ x: number; y: number; node: NodeApi<TreeNode> } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
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

  // Delete one or more paths: drop them from the tree immediately so they vanish
  // without waiting on a full refetch, fire the deletes together, then reconcile
  // with a single reload (which also restores anything whose delete failed).
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

  const menuAction = (fn: () => void) => () => {
    setMenu(null)
    fn()
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Files</span>
        <span className="panel-actions">
          <button
            className="btn-icon"
            title="New folder"
            onClick={() => {
              const name = window.prompt('New folder name')
              if (name) run(mkdir(name))
            }}
          >
            <FolderPlusIcon />
          </button>
          <button className="btn-icon" title="Upload file" onClick={() => fileInputRef.current?.click()}>
            <UploadIcon />
          </button>
          <button className="btn-icon" title="Refresh" onClick={reload}>
            <RefreshIcon />
          </button>
        </span>
        <input
          ref={fileInputRef}
          type="file"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) run(uploadFile(file, ''))
            e.target.value = ''
          }}
        />
      </div>
      {error && <div className="panel-error">{error}</div>}
      <div className="panel-body" ref={containerRef}>
        <Tree<TreeNode>
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
            if (window.confirm(`Delete ${label}?`)) {
              deleteMany(nodes.map((n) => n.data.id))
            }
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
                top: Math.min(menu.y, window.innerHeight - 150),
              }}
              onContextMenu={(e) => e.preventDefault()}
            >
            {menu.node.isLeaf ? (
              <button onClick={menuAction(() => onOpenFile(menu.node.data.id))}>Open</button>
            ) : (
              <button
                onClick={menuAction(() => {
                  const name = window.prompt('New folder name')
                  if (name) run(mkdir(joinPath(menu.node.data.id, name)))
                })}
              >
                New folder inside
              </button>
            )}
            <button onClick={menuAction(() => void menu.node.edit())}>Rename</button>
            {(() => {
              const sel = menu.node.tree.selectedNodes
              const targets =
                sel.length > 1 && sel.some((n) => n.id === menu.node.id) ? sel : [menu.node]
              const label = targets.length > 1 ? `${targets.length} items` : menu.node.data.name
              return (
                <button
                  className="danger"
                  onClick={menuAction(() => {
                    if (window.confirm(`Delete ${label}?`)) {
                      deleteMany(targets.map((n) => n.data.id))
                    }
                  })}
                >
                  {targets.length > 1 ? `Delete ${targets.length} items` : 'Delete'}
                </button>
              )
            })()}
            </div>,
            document.body,
          )}
      </div>
    </div>
  )
})
