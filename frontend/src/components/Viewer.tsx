import { useEffect, useRef, useState } from 'react'
import { Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import { fetchTree, rawUrl } from '../api'
import type { TreeNode } from '../types'
import { DownloadIcon, XIcon } from './icons'

const VOLUME_EXT = /\.(nii|nii\.gz|mgz|mgh|nrrd|nhdr|mha|mhd|hdr|img|v16|dcm)$/
const IMAGE_EXT = /\.(png|jpe?g|gif|svg|webp|bmp)$/
const TEXT_EXT = /\.(md|txt|py|json|yaml|yml|toml|csv|tsv|log|sh|js|ts|html|css|xml)$/

/** Colormaps offered for segmentation overlays (subset of Niivue's list). */
const OVERLAY_COLORMAPS = ['red', 'green', 'blue', 'warm', 'cool', 'plasma'] as const

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
 * slices. A second volume (e.g. a segmentation mask) can be overlaid with a
 * chosen colormap and adjustable opacity.
 */
function VolumeView({ path }: { path: string }) {
  const url = rawUrl(path)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [volumePaths, setVolumePaths] = useState<string[]>([])
  const [overlayPath, setOverlayPath] = useState<string>('')
  const [overlayColormap, setOverlayColormap] = useState<string>('red')
  const [overlayOpacity, setOverlayOpacity] = useState(0.5)

  // Base volume: one Niivue instance per mounted view (remounted via key on
  // path change), so overlay state always starts clean for a new image.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const nv = new Niivue({
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
      backColor: [0.051, 0.059, 0.078, 1],
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
        await nv.addVolumeFromUrl({
          url: rawUrl(newPath),
          colormap: overlayColormap,
          opacity: overlayOpacity,
        })
      }
      setLoadError(null)
    } catch (e) {
      setOverlayPath('')
      setLoadError(`Could not load overlay: ${String(e)}`)
    }
  }

  const setColormap = (cmap: string) => {
    setOverlayColormap(cmap)
    const nv = nvRef.current
    if (nv && nv.volumes.length > 1) {
      nv.volumes[1].colormap = cmap
      nv.updateGLVolume()
    }
  }

  const setOpacity = (value: number) => {
    setOverlayOpacity(value)
    const nv = nvRef.current
    if (nv && nv.volumes.length > 1) {
      nv.setOpacity(1, value)
    }
  }

  return (
    <div className="volume-view">
      <div className="overlay-bar">
        <span className="overlay-label">Overlay</span>
        <select
          className="overlay-select"
          value={overlayPath}
          onChange={(e) => void setOverlay(e.target.value)}
        >
          <option value="">none</option>
          {volumePaths.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        {overlayPath && (
          <>
            <select
              className="overlay-select overlay-select-cmap"
              value={overlayColormap}
              onChange={(e) => setColormap(e.target.value)}
            >
              {OVERLAY_COLORMAPS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
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
      <canvas ref={canvasRef} className="niivue-canvas" />
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
