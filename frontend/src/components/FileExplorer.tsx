import { useEffect, useRef, useState } from 'react'
import { Tree } from 'react-arborist'
import type { NodeApi, NodeRendererProps } from 'react-arborist'
import { deletePath, fetchTree, mkdir, renamePath, uploadFile } from '../api'
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

function NodeRow({ node, style, dragHandle }: NodeRendererProps<TreeNode>) {
  const isDir = !node.isLeaf
  return (
    <div
      ref={dragHandle}
      style={style}
      className={`tree-row${node.isSelected ? ' selected' : ''}`}
      onClick={() => {
        if (isDir) node.toggle()
      }}
      onDoubleClick={() => {
        if (!isDir) node.activate()
      }}
      onDragStart={(e) => {
        // Carry the workspace-relative path so drop targets outside the tree
        // (e.g. the viewer's overlay drop zone) can read it. This coexists
        // with react-arborist's own react-dnd move handling.
        if (!isDir) e.dataTransfer.setData(DRAG_PATH_MIME, node.data.id)
      }}
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
export function FileExplorer({ onOpenFile }: FileExplorerProps) {
  const [data, setData] = useState<TreeNode[]>([])
  const [error, setError] = useState<string | null>(null)
  const [size, setSize] = useState({ width: 280, height: 400 })
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

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(() => setSize({ width: el.clientWidth, height: el.clientHeight }))
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const run = (op: Promise<void>) => {
    op.then(reload).catch((e: unknown) => {
      setError(String(e))
      reload()
    })
  }

  const onActivate = (node: NodeApi<TreeNode>) => {
    if (node.isLeaf) onOpenFile(node.data.id)
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
          {NodeRow}
        </Tree>
      </div>
    </div>
  )
}
