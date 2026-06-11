import { useEffect, useRef, useState } from 'react'
import { Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import { fetchTree, rawUrl } from '../api'
import { getDraggedFilePath } from '../dragState'
import { DRAG_PATH_MIME, type TreeNode } from '../types'
import { DownloadIcon, XIcon } from './icons'

const VOLUME_EXT = /\.(nii|nii\.gz|mgz|mgh|nrrd|nhdr|mha|mhd|hdr|img|v16|dcm)$/
const IMAGE_EXT = /\.(png|jpe?g|gif|svg|webp|bmp)$/
const TEXT_EXT = /\.(md|txt|py|json|yaml|yml|toml|csv|tsv|log|sh|js|ts|html|css|xml)$/

/** Distinct, pleasant hues cycled across integer labels of a segmentation. */
const LABEL_HUES: [number, number, number][] = [
  [230, 60, 60],
  [235, 140, 50],
  [235, 210, 70],
  [70, 200, 110],
  [70, 200, 220],
  [80, 140, 240],
  [160, 110, 230],
  [210, 90, 200],
  [240, 160, 170],
  [120, 200, 140],
  [150, 160, 255],
  [200, 200, 120],
  [90, 210, 200],
  [230, 120, 90],
  [180, 180, 235],
  [140, 225, 90],
]

/** Build a discrete label colormap: index 0 transparent (background), then a
 * distinct hue per label, cycling the palette to cover up to ``n`` labels. So a
 * binary mask renders in one color and a multi-label segmentation gets a
 * distinct color per region. */
function labelColorMap(n = 64): { R: number[]; G: number[]; B: number[]; A: number[]; I: number[] } {
  const R = [0]
  const G = [0]
  const B = [0]
  const A = [0]
  const I = [0]
  for (let i = 1; i <= n; i++) {
    const [r, g, b] = LABEL_HUES[(i - 1) % LABEL_HUES.length]
    R.push(r)
    G.push(g)
    B.push(b)
    A.push(255)
    I.push(i)
  }
  return { R, G, B, A, I }
}

const LABEL_COLORMAP = labelColorMap()

type FileKind = 'volume' | 'pdf' | 'image' | 'text' | 'other'

function classify(path: string): FileKind {
  const lower = path.toLowerCase()
  if (VOLUME_EXT.test(lower)) return 'volume'
  if (lower.endsWith('.pdf')) return 'pdf'
  if (IMAGE_EXT.test(lower)) return 'image'
  if (TEXT_EXT.test(lower)) return 'text'
  return 'other'
}

/** Flatten the explorer tree into the workspace-relative paths of all volumes. */
function collectVolumePaths(nodes: TreeNode[], out: string[] = []): string[] {
  for (const node of nodes) {
    if (node.children) {
      collectVolumePaths(node.children, out)
    } else if (VOLUME_EXT.test(node.name.toLowerCase())) {
      out.push(node.id)
    }
  }
  return out
}

/**
 * Niivue-backed volume view: multiplanar slices + 3D render, wheel scrolls
 * slices. A second volume (e.g. a segmentation mask) can be overlaid — picked
 * from the dropdown or dragged from the file explorer onto the image. The
 * overlay is rendered with a label colormap (distinct color per integer label)
 * and adjustable opacity.
 */
function VolumeView({ path }: { path: string }) {
  const url = rawUrl(path)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [volumePaths, setVolumePaths] = useState<string[]>([])
  const [overlayPath, setOverlayPath] = useState<string>('')
  const [overlayOpacity, setOverlayOpacity] = useState(0.5)
  const [dragOver, setDragOver] = useState(false)

  // Base volume: one Niivue instance per mounted view (remounted via key on
  // path change), so overlay state always starts clean for a new image.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const nv = new Niivue({
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
      backColor: [0.051, 0.059, 0.078, 1],
      // Niivue's own canvas drop handler stopPropagation()s every drop (to load
      // OS files as a new base image). We handle overlay drops ourselves in the
      // capture phase, so disable Niivue's to avoid it hijacking the base image.
      dragAndDropEnabled: false,
    })
    nvRef.current = nv
    let cancelled = false
    const load = async () => {
      try {
        await nv.attachToCanvas(canvas)
        nv.setSliceType(SLICE_TYPE.MULTIPLANAR)
        await nv.loadVolumes([{ url }])
      } catch (e) {
        if (!cancelled) setLoadError(String(e))
      }
    }
    void load()
    return () => {
      cancelled = true
      nvRef.current = null
    }
  }, [url])

  // Candidate overlays: every other volume in the workspace.
  useEffect(() => {
    fetchTree()
      .then((tree) => setVolumePaths(collectVolumePaths(tree).filter((p) => p !== path)))
      .catch(() => setVolumePaths([]))
  }, [path])

  const setOverlay = async (newPath: string) => {
    const nv = nvRef.current
    if (!nv) return
    try {
      // Drop the previous overlay (index 1) before adding the new one.
      if (overlayPath && nv.volumes.length > 1) {
        nv.removeVolumeByUrl(rawUrl(overlayPath))
      }
      setOverlayPath(newPath)
      if (newPath) {
        await nv.addVolumeFromUrl({ url: rawUrl(newPath), opacity: overlayOpacity })
        // Render as discrete labels: each integer value gets a distinct color,
        // value 0 stays transparent. Nicer than a single-hue ramp for masks
        // and segmentations alike.
        const overlay = nv.volumes[nv.volumes.length - 1]
        overlay.setColormapLabel(LABEL_COLORMAP)
        nv.updateGLVolume()
      }
      setLoadError(null)
    } catch (e) {
      setOverlayPath('')
      setLoadError(`Could not load overlay: ${String(e)}`)
    }
  }

  const setOpacity = (value: number) => {
    setOverlayOpacity(value)
    const nv = nvRef.current
    if (nv && nv.volumes.length > 1) {
      nv.setOpacity(1, value)
    }
  }

  // A path is a valid overlay if it's a volume other than the base image.
  const isOverlayCandidate = (p: string | null): p is string =>
    !!p && p !== path && VOLUME_EXT.test(p.toLowerCase())

  // These run in the CAPTURE phase (see the JSX): the wrapper is an ancestor of
  // Niivue's canvas, whose own bubble-phase drop listener stopPropagation()s, so
  // a bubble-phase handler here would never fire. Capture runs first.
  //
  // Resolve the dragged path from dataTransfer, falling back to the module-level
  // channel (react-dnd can strip dataTransfer for tree drags). dataTransfer data
  // is only readable on drop, not during dragover — hence the fallback there too.
  const onDragOver = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(DRAG_PATH_MIME) || isOverlayCandidate(getDraggedFilePath())) {
      e.preventDefault()
      setDragOver(true)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    setDragOver(false)
    const fromData = e.dataTransfer.getData(DRAG_PATH_MIME)
    const p = isOverlayCandidate(fromData) ? fromData : getDraggedFilePath()
    if (isOverlayCandidate(p)) {
      e.preventDefault()
      e.stopPropagation()
      void setOverlay(p)
    }
  }

  return (
    <div className="volume-view">
      <div className="overlay-bar">
        <span className="overlay-label">Overlay</span>
        <select
          className="overlay-select"
          value={overlayPath}
          title="Pick a volume, or drag one from the file explorer onto the image"
          onChange={(e) => void setOverlay(e.target.value)}
        >
          <option value="">none — or drag a file here</option>
          {volumePaths.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        {overlayPath && (
          <>
            <span className="overlay-opacity-label">Opacity</span>
            <input
              className="overlay-opacity"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={overlayOpacity}
              title={`Opacity ${Math.round(overlayOpacity * 100)}%`}
              onChange={(e) => setOpacity(Number(e.target.value))}
            />
            <button className="btn-icon" title="Remove overlay" onClick={() => void setOverlay('')}>
              <XIcon size={13} />
            </button>
          </>
        )}
      </div>
      {loadError && <div className="panel-error">{loadError}</div>}
      <div
        className={`niivue-dropzone${dragOver ? ' drag-over' : ''}`}
        onDragOverCapture={onDragOver}
        onDragLeave={() => setDragOver(false)}
        onDropCapture={onDrop}
      >
        <canvas ref={canvasRef} className="niivue-canvas" />
        {dragOver && <div className="dropzone-hint">Drop to overlay</div>}
      </div>
    </div>
  )
}

function TextView({ url }: { url: string }) {
  const [text, setText] = useState<string>('loading…')
  useEffect(() => {
    fetch(url)
      .then((r) => r.text())
      .then((t) => setText(t.length > 200_000 ? t.slice(0, 200_000) + '\n… (truncated)' : t))
      .catch((e: unknown) => setText(String(e)))
  }, [url])
  return <pre className="text-view">{text}</pre>
}

/** Routes the selected file to the right renderer (volume / PDF / image / text). */
export function Viewer({ path }: { path: string | null }) {
  if (!path) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>Viewer</span>
        </div>
        <div className="viewer-message">Select a file in the explorer to view it here.</div>
      </div>
    )
  }
  const url = rawUrl(path)
  const kind = classify(path)
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="viewer-title" title={path}>
          {path}
        </span>
        <span className="panel-actions">
          <a href={url} download title="Download">
            <DownloadIcon />
          </a>
        </span>
      </div>
      <div className="panel-body viewer-body">
        {kind === 'volume' && <VolumeView key={path} path={path} />}
        {kind === 'pdf' && <iframe className="pdf-frame" src={url} title={path} />}
        {kind === 'image' && <img className="image-view" src={url} alt={path} />}
        {kind === 'text' && <TextView key={url} url={url} />}
        {kind === 'other' && (
          <div className="viewer-message">
            No viewer for this file type.{' '}
            <a href={url} download>
              Download {path}
            </a>
          </div>
        )}
      </div>
    </div>
  )
}
