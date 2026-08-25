import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import { rawUrl } from '../api'
import { getDraggedFilePath } from '../dragState'
import { DRAG_PATH_MIME } from '../types'
import { DownloadIcon, GearIcon, RecenterIcon, XIcon } from './icons'
import { ViewerSettingsPanel } from './ViewerSettings'

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

// Niivue COLORMAP_TYPE.ZERO_TO_MAX_TRANSPARENT_BELOW_MIN — voxels below cal_min
// are fully transparent in the 3D volume render. The enum isn't exported, so we
// use its numeric value.
const COLORMAP_TYPE_TRANSPARENT_BELOW_MIN = 1

/** Add `url` as a styled label overlay on top of `nv`'s base volume: a discrete
 *  label colormap (distinct color per integer label) with the background label
 *  dropped out in both the 2D slices and the 3D render. Shared by the user
 *  overlay action and the post-load restore so they style identically. */
async function addStyledOverlay(nv: Niivue, url: string, opacity: number): Promise<void> {
  await nv.addVolumeFromUrl({ url, opacity })
  const overlay = nv.volumes[nv.volumes.length - 1]
  overlay.setColormapLabel(LABEL_COLORMAP)
  // The label LUT's zero-alpha hides the background only in the 2D slices. In the
  // 3D volume render, transparency is driven by colormapType + cal_min: mark
  // sub-min voxels transparent and put the threshold just above 0 so the
  // background label (0) drops out there too.
  overlay.colormapType = COLORMAP_TYPE_TRANSPARENT_BELOW_MIN
  overlay.cal_min = 0.5
  nv.updateGLVolume()
}

// Legacy standalone convention key — still read once to migrate the preference
// into the consolidated viewer settings below.
const RADIOLOGICAL_KEY = 'medmcp.radiologicalView'

export type SlicePlane = 'multiplanar' | 'axial' | 'coronal' | 'sagittal'

/** User-tunable viewer display options (the gear popover); persisted per browser. */
export type RenderScale = 'native' | '2x' | '4x'

export interface ViewerSettings {
  /** Slice sampling: nearest (faithful voxels, sharp label edges) or linear (smoothed). */
  interpolation: 'nearest' | 'linear'
  /** true = radiological (image-left is patient-right), false = neurological. */
  radiological: boolean
  /** Multiplanar (3 orthogonal + optional 3D) or a single plane. */
  slicePlane: SlicePlane
  /** Show the 3D volume render alongside the slices (multiplanar only). */
  showRender: boolean
  /** Draw the crosshair lines. */
  crosshair: boolean
  /** WebGL MSAA edge anti-aliasing (browser picks the sample count). */
  antialias: boolean
  /** Supersampling factor via forceDevicePixelRatio: 'native' matches the
   *  display, '2x'/'3x' render larger then downsample (smoother edges, more
   *  GPU), '1x' is the cheapest. Both antialias and renderScale are set at
   *  canvas-attach time, so a change remounts the view rather than applying live. */
  renderScale: RenderScale
}

/** Map a RenderScale to Niivue's forceDevicePixelRatio (0 = window.devicePixelRatio). */
const RENDER_SCALE_DPR: Record<RenderScale, number> = { native: 0, '2x': 2, '4x': 4 }

const VIEWER_SETTINGS_KEY = 'medmcp.viewerSettings'

const DEFAULT_VIEWER_SETTINGS: ViewerSettings = {
  interpolation: 'nearest',
  radiological: false,
  slicePlane: 'multiplanar',
  showRender: true,
  crosshair: true,
  antialias: true,
  renderScale: 'native',
}

const SLICE_TYPE_BY_PLANE: Record<SlicePlane, SLICE_TYPE> = {
  multiplanar: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
}

function loadViewerSettings(): ViewerSettings {
  try {
    const raw = localStorage.getItem(VIEWER_SETTINGS_KEY)
    if (raw) {
      const merged = { ...DEFAULT_VIEWER_SETTINGS, ...(JSON.parse(raw) as Partial<ViewerSettings>) }
      // Drop a renderScale persisted under an older option set (e.g. '1x'/'3x').
      if (!(merged.renderScale in RENDER_SCALE_DPR)) merged.renderScale = 'native'
      return merged
    }
  } catch {
    // malformed storage — fall back to defaults (+ legacy migration below)
  }
  return { ...DEFAULT_VIEWER_SETTINGS, radiological: localStorage.getItem(RADIOLOGICAL_KEY) === 'true' }
}

/** Apply the full settings set to a live Niivue instance (idempotent). */
function applyViewerSettings(nv: Niivue, s: ViewerSettings): void {
  nv.setInterpolation(s.interpolation === 'nearest')
  nv.setRadiologicalConvention(s.radiological)
  nv.setSliceType(SLICE_TYPE_BY_PLANE[s.slicePlane])
  nv.setCrosshairWidth(s.crosshair ? 1 : 0)
  nv.opts.multiplanarShowRender = s.showRender ? SHOW_RENDER.ALWAYS : SHOW_RENDER.NEVER
  nv.drawScene()
}

// Niivue's INITIAL_SCENE_DATA, which it does not expose. Kept here so the reset
// lands exactly where a freshly loaded volume starts.
const INITIAL_AZIMUTH = 110
const INITIAL_ELEVATION = 10

/** Return pan, zoom, rotation, and slice position to their just-loaded state.
 *
 * Deliberately not Niivue's `setDefaults()`: that also replaces the whole opts
 * object, which would undo the settings applied by `applyViewerSettings` and
 * re-enable Niivue's own drag-and-drop handler — the one this viewer switches
 * off so it cannot swallow overlay drops. Assigning the scene fields touches
 * the view and nothing else.
 */
function resetView(nv: Niivue): void {
  nv.scene.pan2Dxyzmm = [0, 0, 0, 1]
  nv.scene.volScaleMultiplier = 1
  nv.scene.renderAzimuth = INITIAL_AZIMUTH
  nv.scene.renderElevation = INITIAL_ELEVATION
  nv.scene.crosshairPos = [0.5, 0.5, 0.5]
  nv.drawScene()
}

type FileKind = 'volume' | 'pdf' | 'html' | 'image' | 'text' | 'other'

function classify(path: string): FileKind {
  const lower = path.toLowerCase()
  if (VOLUME_EXT.test(lower)) return 'volume'
  if (lower.endsWith('.pdf')) return 'pdf'
  // Render HTML (e.g. QC reports) rather than showing source — checked before
  // TEXT_EXT, which also matches .html.
  if (lower.endsWith('.html') || lower.endsWith('.htm')) return 'html'
  if (IMAGE_EXT.test(lower)) return 'image'
  if (TEXT_EXT.test(lower)) return 'text'
  return 'other'
}

/**
 * Niivue-backed volume view: multiplanar slices + 3D render, wheel scrolls
 * slices. A second volume (e.g. a segmentation mask) is overlaid by dragging it
 * from the file explorer onto the image; it renders with a label colormap
 * (distinct color per integer label) and adjustable opacity.
 */
function VolumeView({
  path,
  settings,
  isResizing,
  overlayPath,
  overlayOpacity,
  onOverlayChange,
  onOpacityChange,
  resetToken,
}: {
  path: string
  settings: ViewerSettings
  isResizing?: boolean
  overlayPath: string
  overlayOpacity: number
  onOverlayChange: (path: string) => void
  onOpacityChange: (opacity: number) => void
  /** Bumped by the panel's reset button; each new value restores the view. */
  resetToken: number
}) {
  const url = rawUrl(path)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sizerRef = useRef<HTMLDivElement>(null)
  const dropRef = useRef<HTMLDivElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  // Read inside the url-keyed load effect (constructor seeding + post-load) and
  // the live-settings effect without making settings a dependency of the load
  // effect — a setting change must not tear down and reload the volume.
  const settingsRef = useRef(settings)
  useEffect(() => {
    settingsRef.current = settings
  }, [settings])
  // Read inside the mount-once observer effect without re-subscribing.
  const isResizingRef = useRef(isResizing)
  useEffect(() => {
    isResizingRef.current = isResizing
  }, [isResizing])
  // Overlay path + opacity are owned by the parent Viewer so they survive the
  // resize-rebuild remount. Mirror them into refs so the url-keyed load effect
  // can restore the overlay after the base volume loads, without taking them as
  // dependencies (which would tear down and reload the base volume).
  const overlayPathRef = useRef(overlayPath)
  useEffect(() => {
    overlayPathRef.current = overlayPath
  }, [overlayPath])
  const overlayOpacityRef = useRef(overlayOpacity)
  useEffect(() => {
    overlayOpacityRef.current = overlayOpacity
  }, [overlayOpacity])

  // Size the dropzone (Niivue's observed parent) to the panel. We NEVER do this
  // during a separator drag: Niivue leaks GPU resources each time its canvas
  // resizes, so resizing the live instance is what made the viewer degrade. The
  // sizer tracks the panel; the dropzone gets an explicit pixel size synced only
  // while NOT dragging. After a drag the Viewer remounts this view fresh at the
  // new size instead (see `resizeGen`). This observer therefore only matters for
  // the initial mount and non-drag resizes (window/drawer).
  const syncCanvasSize = () => {
    const sizer = sizerRef.current
    const drop = dropRef.current
    if (sizer && drop) {
      drop.style.width = `${sizer.clientWidth}px`
      drop.style.height = `${sizer.clientHeight}px`
    }
  }
  useEffect(() => {
    const sizer = sizerRef.current
    if (!sizer) return
    let timer: number | null = null
    syncCanvasSize()
    const obs = new ResizeObserver(() => {
      // Clear first so a sync scheduled in the frame before `isResizing` commits
      // can't fire mid-drag; then skip entirely while dragging.
      if (timer != null) window.clearTimeout(timer)
      if (isResizingRef.current) return // never resize the WebGL canvas mid-drag
      timer = window.setTimeout(() => {
        timer = null
        syncCanvasSize()
      }, 120)
    })
    obs.observe(sizer)
    return () => {
      obs.disconnect()
      if (timer != null) window.clearTimeout(timer)
    }
  }, [])

  // Base volume: one Niivue instance per mounted view (remounted via key on
  // path change, and on resize-settle), so overlay state always starts clean.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const s0 = settingsRef.current
    const nv = new Niivue({
      // Seed the creation-time opts from the current viewer settings (the rest
      // are applied after load / live). Nearest interpolation keeps a
      // segmentation overlay's integer labels crisp instead of blending IDs at
      // boundaries; native DPR renders crisp slices; the 3D render is part of
      // the multiplanar layout. All are user-toggleable in the viewer settings.
      isNearestInterpolation: s0.interpolation === 'nearest',
      forceDevicePixelRatio: RENDER_SCALE_DPR[s0.renderScale],
      multiplanarShowRender: s0.showRender ? SHOW_RENDER.ALWAYS : SHOW_RENDER.NEVER,
      backColor: [0, 0, 0, 1],
      // Suppress Niivue's own canvas "loading ..." text — our spinner overlay
      // is the single loading affordance.
      loadingText: '',
      // Niivue's own canvas drop handler stopPropagation()s every drop (to load
      // OS files as a new base image). We handle overlay drops ourselves in the
      // capture phase, so disable Niivue's to avoid it hijacking the base image.
      dragAndDropEnabled: false,
    })
    nvRef.current = nv
    let cancelled = false
    const load = async () => {
      setLoading(true)
      setLoadError(null)
      try {
        // MSAA (anti-aliasing) follows the antialias setting. AA smooths edges
        // (3D render, crosshair, slice boundaries) but multiplies the
        // framebuffer's GPU memory — the trigger for WebGL context loss on
        // memory-constrained GPUs — so it's toggleable. Independent of the
        // render scale (supersampling), which is set via forceDevicePixelRatio.
        await nv.attachToCanvas(canvas, s0.antialias)
        nv.setSliceType(SLICE_TYPE.MULTIPLANAR)
        // Niivue streams the download/inflate, but parsing the volume and the
        // initial WebGL upload + 3D render run synchronously on the main thread
        // — a big volume briefly freezes the tab. Yield a frame here so the
        // loading overlay actually paints before that blocking work starts,
        // instead of the viewer just appearing hung.
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
        if (cancelled) return
        await nv.loadVolumes([{ url }])
        applyViewerSettings(nv, settingsRef.current)
        // Restore a persisted overlay so it survives the resize-rebuild remount
        // (which rebuilds this view fresh — see the parent Viewer). The overlay
        // path/opacity are owned there and mirrored into refs above.
        if (overlayPathRef.current && !cancelled) {
          await addStyledOverlay(nv, rawUrl(overlayPathRef.current), overlayOpacityRef.current)
        }
      } catch (e) {
        if (!cancelled) setLoadError(String(e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    // Seed the overlay-op chain with the base load (which now also restores a
    // persisted overlay) so an overlay dropped before the base finished loading
    // can't race it.
    overlayOpRef.current = load()
    return () => {
      cancelled = true
      nvRef.current = null
      // Each opened file (and each resize-settle) remounts this view and builds
      // a fresh Niivue + WebGL context. cleanup() removes Niivue's observers and
      // listeners, then we force-release the GL context: browsers cap live
      // contexts (~16) and drop the oldest once exceeded, which janks the whole
      // tab — so a long session must not leak them.
      try {
        nv.cleanup()
        nv.gl?.getExtension('WEBGL_lose_context')?.loseContext()
      } catch {
        // best-effort teardown
      }
    }
  }, [url])

  // Apply live setting changes to the open volume. `antialias` and
  // `renderScale` are excluded here — both are creation-time, so changing them
  // remounts this view via its key (see Viewer). The guard skips the initial
  // mount before the volume has loaded; the load effect applies settings then.
  useEffect(() => {
    const nv = nvRef.current
    if (nv && nv.volumes.length > 0) applyViewerSettings(nv, settings)
  }, [settings])

  // Restore the default view when the panel's reset button fires. Token 0 is the
  // initial mount, where the view is already at its defaults and there is nothing
  // to undo.
  useEffect(() => {
    if (resetToken === 0) return
    const nv = nvRef.current
    if (nv && nv.volumes.length > 0) resetView(nv)
  }, [resetToken])

  // Overlay operations are serialized on a promise chain, seeded with the
  // base-volume load: a switch while the previous add was still in flight
  // would otherwise skip the removal (volumes.length is still 1) and stack a
  // phantom overlay that the opacity slider and remove button can no longer
  // address.
  const overlayOpRef = useRef<Promise<void>>(Promise.resolve())

  const setOverlay = (newPath: string) => {
    // Persist in the parent Viewer so the overlay survives a resize-rebuild
    // remount. Only ever called from event handlers (drop/select/remove), where
    // ref writes are fine — the lint rule just can't see the call sites.
    onOverlayChange(newPath)
    // eslint-disable-next-line react-hooks/immutability
    overlayOpRef.current = overlayOpRef.current.then(() => applyOverlay(newPath))
  }

  const applyOverlay = async (newPath: string) => {
    const nv = nvRef.current
    if (!nv) return
    try {
      // Drop every overlay, whatever got stacked (index 0 is the base image).
      while (nv.volumes.length > 1) {
        nv.removeVolume(nv.volumes[nv.volumes.length - 1])
      }
      if (newPath) {
        await addStyledOverlay(nv, rawUrl(newPath), overlayOpacityRef.current)
      }
      setLoadError(null)
    } catch (e) {
      onOverlayChange('')
      setLoadError(`Could not load overlay: ${String(e)}`)
    }
  }

  const setOpacity = (value: number) => {
    onOpacityChange(value)
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
      setOverlay(p)
    }
  }

  return (
    <div className="volume-view">
      {/* Only rendered once something is actually overlaid: with drag-and-drop
          as the only way in, an always-present bar would be a permanent strip of
          controls for a state the viewer is usually not in. */}
      {overlayPath && (
        <div className="overlay-bar">
          <span className="overlay-label" title={overlayPath}>
            {overlayPath.split('/').pop()}
          </span>
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
          <button className="btn-icon" title="Remove overlay" onClick={() => setOverlay('')}>
            <XIcon size={13} />
          </button>
        </div>
      )}
      {loadError && <div className="panel-error">{loadError}</div>}
      <div className="niivue-sizer" ref={sizerRef}>
        <div
          ref={dropRef}
          className={`niivue-dropzone${dragOver ? ' drag-over' : ''}`}
          onDragOverCapture={onDragOver}
          onDragLeave={() => setDragOver(false)}
          onDropCapture={onDrop}
        >
          <canvas ref={canvasRef} className="niivue-canvas" />
          {dragOver && <div className="dropzone-hint">Drop to overlay</div>}
        </div>
      </div>
      {/* Covers the whole volume view (overlay bar + canvas) so it centers in
          the same box as the Viewer's rebuild spinner — otherwise the wheel
          jumps when one hands off to the other across a resize-rebuild. */}
      {loading && (
        <div className="volume-loading">
          <span className="volume-spinner" />
          <span>Loading volume…</span>
        </div>
      )}
    </div>
  )
}

function TextView({ url }: { url: string }) {
  const [text, setText] = useState<string>('loading…')
  useEffect(() => {
    fetch(url)
      .then((r) => {
        // Without this, a 404's JSON body would render as the file's content.
        if (!r.ok) throw new Error(`could not load file (HTTP ${r.status})`)
        return r.text()
      })
      .then((t) => setText(t.length > 200_000 ? t.slice(0, 200_000) + '\n… (truncated)' : t))
      .catch((e: unknown) => setText(String(e)))
  }, [url])
  return <pre className="text-view">{text}</pre>
}

/** Routes the selected file to the right renderer (volume / PDF / image / text). */
// Memoized so a separator drag doesn't re-render the viewer (and its WebGL/React
// subtree) every frame; props from App are stable except when the open file
// actually changes.
export const Viewer = memo(function Viewer({
  path,
  isResizing,
}: {
  path: string | null
  /** True while a separator is being dragged; freezes the volume canvas size. */
  isResizing?: boolean
}) {
  // Niivue leaks GPU memory each time its canvas is resized, so we never resize
  // the live instance — we rebuild it fresh at the new size once a separator
  // drag settles. `resizeGen` is part of the VolumeView key; bumping it remounts
  // the volume (same mechanism that resets it when you open a file). Debounced so
  // a burst of adjustments collapses into one rebuild; gated by `didResize` so
  // mount / non-resize renders don't trigger a spurious reload.
  const [resizeGen, setResizeGen] = useState(0)
  // True from the moment a resize drag ends until the fresh view has remounted,
  // so we hide the stale (wrong-size) canvas and show the spinner instead of
  // letting the old image flash at the wrong size during the debounce window.
  const [rebuilding, setRebuilding] = useState(false)
  const didResizeRef = useRef(false)
  const [settings, setSettings] = useState<ViewerSettings>(loadViewerSettings)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [resetToken, setResetToken] = useState(0)
  // Overlay selection lives here, not in VolumeView, so it survives the
  // resize-rebuild remount (which bumps resizeGen and rebuilds VolumeView fresh).
  // It's tagged with the base file it belongs to, so it's transparently ignored
  // once a different file is opened — no state reset (which the hooks lint
  // forbids both in render and in effects) is needed.
  const [overlay, setOverlay] = useState<{ base: string; path: string; opacity: number }>({
    base: '',
    path: '',
    opacity: 0.5,
  })
  const overlayForThisFile = overlay.base === path
  const overlayPath = overlayForThisFile ? overlay.path : ''
  const overlayOpacity = overlayForThisFile ? overlay.opacity : 0.5
  const updateSettings = useCallback((patch: Partial<ViewerSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch }
      try {
        localStorage.setItem(VIEWER_SETTINGS_KEY, JSON.stringify(next))
      } catch {
        // ignore storage failure — settings still apply for this session
      }
      return next
    })
  }, [])
  useEffect(() => {
    if (isResizing) {
      didResizeRef.current = true
      return
    }
    if (!didResizeRef.current) return
    didResizeRef.current = false
    setRebuilding(true)
    const t = window.setTimeout(() => {
      setResizeGen((g) => g + 1)
      setRebuilding(false)
    }, 250)
    return () => window.clearTimeout(t)
  }, [isResizing])

  if (!path) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>Viewer</span>
        </div>
        {/* Black is the right ground for an image and the wrong one for an
            empty panel, where it reads as a hole in the app. The layout stays,
            only the colour changes. */}
        <div className="panel-body viewer-body is-empty">
          <div className="viewer-message">Select a file in the explorer to view it here.</div>
        </div>
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
          {kind === 'volume' && (
            <button
              className="btn-icon"
              title="Reset view"
              onClick={() => setResetToken((t) => t + 1)}
            >
              <RecenterIcon />
            </button>
          )}
          {kind === 'volume' && (
            <span className="viewer-settings-anchor">
              <button
                className={settingsOpen ? 'btn-icon active' : 'btn-icon'}
                title="Viewer settings"
                onClick={() => setSettingsOpen((v) => !v)}
              >
                <GearIcon />
              </button>
              {settingsOpen && (
                <ViewerSettingsPanel
                  settings={settings}
                  onChange={updateSettings}
                  onClose={() => setSettingsOpen(false)}
                />
              )}
            </span>
          )}
          <a href={url} download title="Download">
            <DownloadIcon />
          </a>
        </span>
      </div>
      <div className="panel-body viewer-body">
        {kind === 'volume' && (
          <div className={`volume-slot${rebuilding ? ' rebuilding' : ''}`}>
            <VolumeView
              key={`${path}#${resizeGen}#${settings.antialias ? 'aa' : 'noaa'}#${settings.renderScale}`}
              path={path}
              settings={settings}
              isResizing={isResizing}
              resetToken={resetToken}
              overlayPath={overlayPath}
              overlayOpacity={overlayOpacity}
              onOverlayChange={(p) =>
                setOverlay((prev) => ({
                  base: path,
                  path: p,
                  opacity: prev.base === path ? prev.opacity : 0.5,
                }))
              }
              onOpacityChange={(o) =>
                setOverlay((prev) => ({
                  base: path,
                  path: prev.base === path ? prev.path : '',
                  opacity: o,
                }))
              }
            />
            {rebuilding && (
              <div className="volume-loading">
                <span className="volume-spinner" />
                <span>Loading volume…</span>
              </div>
            )}
          </div>
        )}
        {kind === 'pdf' && <iframe className="pdf-frame" src={url} title={path} />}
        {kind === 'html' && (
          // Render reports (e.g. the QC report.html) instead of showing source.
          // sandbox="allow-scripts" runs self-contained inline JS (the QC flicker
          // toggle) while isolating the frame: no same-origin access, no navigation.
          <iframe className="html-frame" src={url} title={path} sandbox="allow-scripts" />
        )}
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
})
