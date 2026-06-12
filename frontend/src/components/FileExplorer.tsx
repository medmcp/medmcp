import { useEffect, useRef, useState } from 'react'
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

/** Workspace file tree with open/rename/move/delete/upload. */
export function FileExplorer({ onOpenFile, refreshSignal }: FileExplorerProps) {
  const [data, setData] = useState<TreeNode[]>([])
  const [error, setError] = useState<string | null>(null)
  const [size, setSize] = useState({ width: 280, height: 400 })
  const [menu, setMenu] = useState<{ x: number; y: number; node: NodeApi<TreeNode> } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  // Trailing debounce: re-rendering the tree on every observer tick adds
  // React work to each frame of a separator drag; once it settles is enough.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let timer: number | null = null
    const obs = new ResizeObserver(() => {
      if (timer != null) window.clearTimeout(timer)
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

  const run = (op: Promise<void>) => {
    op.then(reload).catch((e: unknown) => {
      setError(String(e))
      reload()
    })
  }

  // Any click/Escape outside the menu dismisses it; the menu itself stops
  // mousedown propagation so its buttons still receive their click.
  useEffect(() => {
    if (!menu) return
    const close = () => setMenu(null)
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
    node.select()
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
          disableMultiSelection
          onActivate={onActivate}
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
            const names = nodes.map((n) => n.data.name).join(', ')
            if (window.confirm(`Delete ${names}?`)) {
              for (const node of nodes) run(deletePath(node.data.id))
            }
          }}
        >
          {(props) => <NodeRow {...props} onMenu={openMenu} />}
        </Tree>
        {menu && (
          <div
            className="ctx-menu"
            style={{
              left: Math.min(menu.x, window.innerWidth - 170),
              top: Math.min(menu.y, window.innerHeight - 150),
            }}
            onMouseDown={(e) => e.stopPropagation()}
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
            <button
              className="danger"
              onClick={menuAction(() => {
                if (window.confirm(`Delete ${menu.node.data.name}?`)) {
                  run(deletePath(menu.node.data.id))
                }
              })}
            >
              Delete
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
