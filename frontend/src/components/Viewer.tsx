import { useEffect, useRef, useState } from 'react'
import { Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import { rawUrl } from '../api'

const VOLUME_EXT = /\.(nii|nii\.gz|mgz|mgh|nrrd|nhdr|mha|mhd|hdr|img|v16|dcm)$/
const IMAGE_EXT = /\.(png|jpe?g|gif|svg|webp|bmp)$/
const TEXT_EXT = /\.(md|txt|py|json|yaml|yml|toml|csv|tsv|log|sh|js|ts|html|css|xml)$/

type FileKind = 'volume' | 'pdf' | 'image' | 'text' | 'other'

function classify(path: string): FileKind {
  const lower = path.toLowerCase()
  if (VOLUME_EXT.test(lower)) return 'volume'
  if (lower.endsWith('.pdf')) return 'pdf'
  if (IMAGE_EXT.test(lower)) return 'image'
  if (TEXT_EXT.test(lower)) return 'text'
  return 'other'
}

/** Niivue-backed volume view: multiplanar slices + 3D render, wheel scrolls slices. */
function VolumeView({ url }: { url: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const nv = new Niivue({
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
      backColor: [0.06, 0.07, 0.09, 1],
    })
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
    }
  }, [url])

  if (loadError) return <div className="viewer-message">Could not load volume: {loadError}</div>
  return <canvas ref={canvasRef} className="niivue-canvas" />
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
        <div className="panel-header"><span>Viewer</span></div>
        <div className="viewer-message">Select a file in the explorer to view it here.</div>
      </div>
    )
  }
  const url = rawUrl(path)
  const kind = classify(path)
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="viewer-title" title={path}>{path}</span>
        <span className="panel-actions">
          <a href={url} download title="Download">⬇️</a>
        </span>
      </div>
      <div className="panel-body viewer-body">
        {kind === 'volume' && <VolumeView key={url} url={url} />}
        {kind === 'pdf' && <iframe className="pdf-frame" src={url} title={path} />}
        {kind === 'image' && <img className="image-view" src={url} alt={path} />}
        {kind === 'text' && <TextView key={url} url={url} />}
        {kind === 'other' && (
          <div className="viewer-message">
            No viewer for this file type. <a href={url} download>Download {path}</a>
          </div>
        )}
      </div>
    </div>
  )
}
