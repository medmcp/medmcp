import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import type { NVImage, NiiVueLocation } from '@niivue/niivue'
import { rawUrl } from '../api'
import { getDraggedFilePath } from '../dragState'
import { classify, isVolumePath } from '../fileKinds'
import { DRAG_PATH_MIME } from '../types'
import {
  CameraIcon,
  EyeIcon,
  EyeOffIcon,
  GearIcon,
  ListIcon,
  RecenterIcon,
  XIcon,
} from './icons'
import {
  CONTINUOUS_COLORMAPS,
  CT_WINDOW_PRESETS,
  EMPTY_OVERLAY,
  buildLabelColormap,
  classifyOverlayData,
  cssColor,
  formatIntensity,
  formatMl,
  formatMm,
  labelColor,
  labelFileCandidates,
  labelStats,
  looksLikeCT,
  parseLabelNames,
  overlayFor,
  voxelsToMl,
  type LabelColorFn,
  type LabelStat,
  type OverlayKind,
  type OverlayState,
} from './viewerData'
import { ViewerSettingsPanel } from './ViewerSettings'

// Niivue COLORMAP_TYPE.ZERO_TO_MAX_TRANSPARENT_BELOW_MIN — voxels below cal_min
// are fully transparent. The enum isn't exported, so we use its numeric value.
const COLORMAP_TYPE_TRANSPARENT_BELOW_MIN = 1

// NIfTI intent code for label images. Niivue draws a volume carrying it with
// its atlas shader: an exact per-id lookup into the label colour table whose
// alpha it honours. Its generic shader instead samples the table as a
// linearly filtered gradient and clamps small ids to one texel — measured on a
// FreeSurfer map, ids 2–13 all came out the same colour and a hidden id showed
// whenever its neighbour was visible.
const NII_INTENT_LABEL = 1002
const DT_UINT8 = 2
const DT_INT16 = 4
const DT_UINT16 = 512

/** Mark *vol* as a label image so Niivue's atlas shader draws it. That shader
 *  reads integer textures only, so the voxels are converted to the narrowest
 *  integer type that holds the ids (as Niivue itself does for FreeSurfer
 *  files it recognises by name). Returns false — leaving the generic shader in
 *  charge — when the data cannot be expressed that way. */
function routeThroughAtlasShader(vol: NVImage, minId: number, maxId: number): boolean {
  const hdr = vol.hdr
  const img = vol.img
  if (!hdr || !img) return false
  // The atlas shader reads raw voxel values; a scaled file would mislabel.
  if ((hdr.scl_slope !== 0 && hdr.scl_slope !== 1) || hdr.scl_inter !== 0) return false
  let target: number
  if (minId < 0) {
    if (minId < -32768 || maxId > 32767) return false
    target = DT_INT16
  } else if (maxId <= 255) {
    target = DT_UINT8
  } else if (maxId <= 65535) {
    target = DT_UINT16
  } else {
    return false
  }
  const dt = hdr.datatypeCode
  const alreadyFits =
    dt === target || (dt === DT_INT16 && minId >= 0 && maxId <= 32767) || (dt === DT_UINT16 && minId >= 0)
  if (!alreadyFits) {
    const out =
      target === DT_UINT8
        ? new Uint8Array(img.length)
        : target === DT_INT16
          ? new Int16Array(img.length)
          : new Uint16Array(img.length)
    for (let i = 0; i < img.length; i++) out[i] = Math.round(img[i])
    vol.img = out
    hdr.datatypeCode = target
    hdr.numBitsPerVoxel = target === DT_UINT8 ? 8 : 16
  }
  hdr.intent_code = NII_INTENT_LABEL
  return true
}

// ── Volume bytes cache ───────────────────────────────────────────
//
// Every file open, panel resize and overlay restore remounts the Niivue view,
// which would re-download the volume each time. The bytes are kept per URL (a
// handful of compressed files) and revalidated with a HEAD request against the
// server's Last-Modified/Content-Length, so a file the agent rewrote in place is
// fetched again rather than shown stale.

interface CachedVolume {
  buffer: ArrayBuffer
  etag: string
}

const volumeCache = new Map<string, CachedVolume>()
const VOLUME_CACHE_MAX = 4

function headerStamp(r: Response): string {
  return `${r.headers.get('last-modified') ?? ''}|${r.headers.get('content-length') ?? ''}|${r.headers.get('etag') ?? ''}`
}

async function fetchVolumeBytes(url: string): Promise<ArrayBuffer> {
  const hit = volumeCache.get(url)
  if (hit) {
    try {
      const head = await fetch(url, { method: 'HEAD' })
      if (head.ok && headerStamp(head) === hit.etag) {
        volumeCache.delete(url)
        volumeCache.set(url, hit)
        return hit.buffer
      }
    } catch {
      // revalidation failed — fall through to a full fetch
    }
  }
  const r = await fetch(url)
  if (!r.ok) throw new Error(`could not load volume (HTTP ${r.status})`)
  const buffer = await r.arrayBuffer()
  volumeCache.set(url, { buffer, etag: headerStamp(r) })
  while (volumeCache.size > VOLUME_CACHE_MAX) {
    const oldest = volumeCache.keys().next().value
    if (oldest === undefined) break
    volumeCache.delete(oldest)
  }
  return buffer
}

/** Load *path* into `nv` as the base volume (replace = true) or as an added
 *  overlay, from the bytes cache. Niivue takes the file type from `name`. */
async function loadVolume(nv: Niivue, path: string, opts: { opacity?: number; replace: boolean }) {
  const bytes = await fetchVolumeBytes(rawUrl(path))
  const blobUrl = URL.createObjectURL(new Blob([bytes]))
  const name = path.split('/').pop() ?? path
  try {
    if (opts.replace) await nv.loadVolumes([{ url: blobUrl, name }])
    else await nv.addVolumeFromUrl({ url: blobUrl, name, opacity: opts.opacity ?? 1 })
  } finally {
    URL.revokeObjectURL(blobUrl)
  }
}

/** What the loaded overlay turned out to be — derived from its data once. */
interface OverlayInfo {
  kind: OverlayKind
  maxLabel: number
  stats: LabelStat[]
  names: Map<number, string>
  /** Colour per label id (the generated palette). */
  color: LabelColorFn
  dims: [number, number, number]
  pixDims: [number, number, number]
  min: number
  max: number
  robustMin: number
  robustMax: number
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
  nv.createOnLocationChange()
}

// ── Overlay styling ──────────────────────────────────────────────

/** Read what the loaded overlay is, once, from its data. */
async function describeOverlay(vol: NVImage, path: string, forcedKind: OverlayKind | null): Promise<OverlayInfo> {
  const hdr = vol.hdr
  const img = vol.img ?? new Uint8Array()
  const dims: [number, number, number] = [hdr?.dims[1] ?? 1, hdr?.dims[2] ?? 1, hdr?.dims[3] ?? 1]
  const pixDims: [number, number, number] = [
    hdr?.pixDims[1] ?? 1,
    hdr?.pixDims[2] ?? 1,
    hdr?.pixDims[3] ?? 1,
  ]
  const kind = forcedKind ?? classifyOverlayData(img, hdr?.scl_slope ?? 1, hdr?.scl_inter ?? 0)
  let stats: LabelStat[] = []
  let names = new Map<number, string>()
  const color: LabelColorFn = labelColor
  let maxLabel = 0
  if (kind === 'label') {
    stats = labelStats(img, dims)
    maxLabel = stats.length ? stats[stats.length - 1].id : Math.ceil(vol.global_max ?? 1)
    routeThroughAtlasShader(vol, Math.min(0, Math.floor(vol.global_min ?? 0)), maxLabel)
    names = await fetchLabelNames(path)
  }
  return {
    kind,
    maxLabel,
    stats,
    names,
    color,
    dims,
    pixDims,
    min: vol.global_min ?? 0,
    max: vol.global_max ?? 1,
    robustMin: vol.robust_min ?? vol.cal_min ?? 0,
    robustMax: vol.robust_max ?? vol.cal_max ?? 1,
  }
}

/** The first label-name sidecar that exists next to the segmentation. */
async function fetchLabelNames(volumePath: string): Promise<Map<number, string>> {
  for (const candidate of labelFileCandidates(volumePath)) {
    try {
      const r = await fetch(rawUrl(candidate))
      if (!r.ok) continue
      const names = parseLabelNames(await r.text())
      if (names.size > 0) return names
    } catch {
      // unreachable sidecar — try the next candidate
    }
  }
  return new Map()
}

/** Default threshold for a continuous overlay: hide exact zeros of a
 *  non-negative map (probabilities, lesion maps), show everything of a signed one. */
function defaultThreshold(info: OverlayInfo): number {
  if (info.min >= 0) return Math.max(info.robustMin, info.min + 1e-6)
  return info.robustMin
}

/** Apply the overlay's styling (kind, colormap, threshold, isolation, visibility)
 *  to the loaded overlay volume. Re-run on every state change; cheap. */
function styleOverlay(nv: Niivue, vol: NVImage, info: OverlayInfo, state: OverlayState): void {
  if (info.kind === 'label') {
    vol.setColormapLabel(
      buildLabelColormap(
        Math.max(info.maxLabel, 1),
        info.color,
        state.isolate ? new Set(state.isolate) : undefined,
      ),
    )
    // The atlas shader (see routeThroughAtlasShader) takes colour and alpha
    // straight from the table; these only matter on the generic fallback path.
    vol.colormapType = COLORMAP_TYPE_TRANSPARENT_BELOW_MIN
    vol.cal_min = 0.5
    vol.cal_max = Math.max(info.maxLabel, 1) + 0.5
  } else {
    vol.colormapLabel = null
    vol.colormap = state.colormap
    vol.colormapType = COLORMAP_TYPE_TRANSPARENT_BELOW_MIN
    vol.cal_min = state.threshold ?? defaultThreshold(info)
    // Top of the colour scale at the robust maximum, not the global one: a few
    // bright voxels would otherwise squeeze everything else into the dark end.
    vol.cal_max = Math.max(info.robustMax, vol.cal_min + 1e-6)
  }
  vol.opacity = state.hidden ? 0 : state.opacity
  nv.updateGLVolume()
}

// ── Readout ──────────────────────────────────────────────────────

interface Readout {
  vox: [number, number, number]
  mm: [number, number, number]
  baseValue: number
  overlayValue: number | null
  /** 0 axial, 1 coronal, 2 sagittal — the pane the pointer is over. */
  pane: number
}

type OverlayVol = { path: string; vol: NVImage; info: OverlayInfo }

/**
 * Niivue-backed volume view: multiplanar slices + 3D render, wheel scrolls
 * slices. A second volume (a segmentation or a continuous map) is overlaid by
 * dragging it from the file explorer onto the image, or from a chat result.
 * A status bar reads out position, intensity and label under the crosshair
 * and holds the intensity window.
 */
function VolumeView({
  path,
  settings,
  isResizing,
  overlay,
  onOverlayChange,
  resetToken,
  snapshotToken,
}: {
  path: string
  settings: ViewerSettings
  isResizing?: boolean
  overlay: OverlayState
  onOverlayChange: (next: OverlayState) => void
  /** Bumped by the panel's reset button; each new value restores the view. */
  resetToken: number
  /** Bumped by the panel's snapshot button; each new value saves a PNG. */
  snapshotToken: number
}) {
  const url = rawUrl(path)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sizerRef = useRef<HTMLDivElement>(null)
  const dropRef = useRef<HTMLDivElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const [readout, setReadout] = useState<Readout | null>(null)
  const [win, setWin] = useState<{ min: number; max: number } | null>(null)
  const [baseInfo, setBaseInfo] = useState<{
    dims: [number, number, number]
    pixDims: [number, number, number]
    min: number
    max: number
    robustMin: number
    robustMax: number
    frames: number
    isCT: boolean
  } | null>(null)
  const [overlayVol, setOverlayVol] = useState<OverlayVol | null>(null)
  const [legendOpen, setLegendOpen] = useState(false)
  const [windowOpen, setWindowOpen] = useState(false)
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
  // The overlay state is owned by the parent so it survives the resize-rebuild
  // remount. Mirror it into a ref so the url-keyed load effect can restore the
  // overlay after the base volume loads without taking it as a dependency
  // (which would tear down and reload the base volume).
  const overlayRef = useRef(overlay)
  useEffect(() => {
    overlayRef.current = overlay
  }, [overlay])
  const onOverlayChangeRef = useRef(onOverlayChange)
  useEffect(() => {
    onOverlayChangeRef.current = onOverlayChange
  }, [onOverlayChange])
  // The pane the pointer was last over, for keyboard slice stepping.
  const paneRef = useRef(0)

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

  // Overlay operations are serialized on a promise chain, seeded with the
  // base-volume load: a switch while the previous add was still in flight
  // would otherwise skip the removal (volumes.length is still 1) and stack a
  // phantom overlay that the opacity slider and remove button can no longer
  // address.
  const overlayOpRef = useRef<Promise<void>>(Promise.resolve())
  const overlayVolRef = useRef<OverlayVol | null>(null)

  /** Bring the overlaid volume in line with the latest overlay state: load a
   *  new file (or remove the old one), re-read it when the kind was overridden,
   *  otherwise just restyle. Idempotent, so it can run after every change. */
  const applyOverlay = useCallback(async () => {
    const nv = nvRef.current
    if (!nv) return
    const state = overlayRef.current
    const current = overlayVolRef.current
    if (current && current.path === state.path && (!state.kind || state.kind === current.info.kind)) {
      styleOverlay(nv, current.vol, current.info, state)
      return
    }
    try {
      // Drop every overlay, whatever got stacked (index 0 is the base image).
      while (nv.volumes.length > 1) {
        nv.removeVolume(nv.volumes[nv.volumes.length - 1])
      }
      overlayVolRef.current = null
      setOverlayVol(null)
      if (state.path) {
        await loadVolume(nv, state.path, { opacity: state.opacity, replace: false })
        const vol = nv.volumes[nv.volumes.length - 1]
        const info = await describeOverlay(vol, state.path, state.kind)
        // The instance may have been torn down, or the request superseded,
        // while the data was being read; a later chained call handles the rest.
        if (nvRef.current !== nv || overlayRef.current.path !== state.path) return
        styleOverlay(nv, vol, info, overlayRef.current)
        const loaded = { path: state.path, vol, info }
        overlayVolRef.current = loaded
        setOverlayVol(loaded)
      }
      setLoadError(null)
    } catch (e) {
      onOverlayChangeRef.current({ ...EMPTY_OVERLAY, base: path })
      setLoadError(`Could not load overlay: ${String(e)}`)
    }
  }, [path])

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
    // Crosshair readout. Niivue fires this on every pointer move over the
    // canvas; coalesce to one React update per frame.
    let pending: Readout | null = null
    let raf = 0
    nv.onLocationChange = (location: unknown) => {
      const loc = location as NiiVueLocation
      const base = loc.values[0]
      const ov = loc.values[1]
      pending = {
        vox: [loc.vox[0], loc.vox[1], loc.vox[2]],
        mm: [loc.mm[0], loc.mm[1], loc.mm[2]],
        baseValue: base?.value ?? NaN,
        overlayValue: ov ? ov.value : null,
        pane: loc.axCorSag,
      }
      if (!raf) {
        raf = requestAnimationFrame(() => {
          raf = 0
          if (pending) setReadout(pending)
        })
      }
    }
    // Right-drag (Niivue's contrast mode) changes the base window; mirror it.
    nv.onIntensityChange = (vol: NVImage) => {
      if (vol === nv.volumes[0]) setWin({ min: vol.cal_min ?? 0, max: vol.cal_max ?? 1 })
    }
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
        await loadVolume(nv, path, { replace: true })
        if (cancelled) return
        applyViewerSettings(nv, settingsRef.current)
        const base = nv.volumes[0]
        const hdr = base.hdr
        const gmin = base.global_min ?? 0
        const gmax = base.global_max ?? 1
        setBaseInfo({
          dims: [hdr?.dims[1] ?? 0, hdr?.dims[2] ?? 0, hdr?.dims[3] ?? 0],
          pixDims: [hdr?.pixDims[1] ?? 1, hdr?.pixDims[2] ?? 1, hdr?.pixDims[3] ?? 1],
          min: gmin,
          max: gmax,
          robustMin: base.robust_min ?? base.cal_min ?? gmin,
          robustMax: base.robust_max ?? base.cal_max ?? gmax,
          frames: base.nFrame4D ?? 1,
          isCT: looksLikeCT(gmin, gmax),
        })
        setWin({ min: base.cal_min ?? gmin, max: base.cal_max ?? gmax })
        nv.createOnLocationChange()
        // Restore a persisted overlay so it survives the resize-rebuild remount
        // (which rebuilds this view fresh — see the parent Viewer).
        if (overlayRef.current.path && !cancelled) {
          await applyOverlay()
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
      overlayVolRef.current = null
      if (raf) cancelAnimationFrame(raf)
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
  }, [url, path, applyOverlay])

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

  // Save the current canvas as a PNG named after the file.
  useEffect(() => {
    if (snapshotToken === 0) return
    const nv = nvRef.current
    if (!nv || nv.volumes.length === 0) return
    const stem = (path.split('/').pop() ?? 'volume').replace(/\.(nii(\.gz)?|mgz|mgh|nrrd|nhdr|mha|mhd)$/i, '')
    void nv.saveScene(`${stem}.png`)
  }, [snapshotToken, path])

  // Every overlay change goes through the serialized chain; applyOverlay works
  // out whether that means a load, a re-read or only a restyle.
  useEffect(() => {
    overlayOpRef.current = overlayOpRef.current.then(applyOverlay)
  }, [overlay, applyOverlay])

  const patchOverlay = (patch: Partial<OverlayState>) =>
    onOverlayChange({ ...overlayRef.current, ...patch, base: path })

  const setOverlayPath = (newPath: string) => {
    onOverlayChange(newPath ? overlayFor(path, newPath) : { ...EMPTY_OVERLAY, base: path })
  }

  // A path is a valid overlay if it's a volume other than the base image.
  const isOverlayCandidate = (p: string | null): p is string => !!p && p !== path && isVolumePath(p)

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
      setOverlayPath(p)
    }
  }

  // Keyboard: step slices in the pane under the pointer (↑/↓, PageUp/PageDown;
  // Shift = 10 at a time), `o` hides/shows the overlay for a before/after look.
  const onKeyDown = (e: React.KeyboardEvent) => {
    const nv = nvRef.current
    if (!nv || nv.volumes.length === 0) return
    if (e.key === 'o' || e.key === 'O') {
      if (overlayRef.current.path) patchOverlay({ hidden: !overlayRef.current.hidden })
      e.preventDefault()
      return
    }
    let dir = 0
    if (e.key === 'ArrowUp' || e.key === 'PageUp') dir = 1
    else if (e.key === 'ArrowDown' || e.key === 'PageDown') dir = -1
    if (dir === 0) return
    e.preventDefault()
    const step = dir * (e.shiftKey ? 10 : 1)
    const plane = settingsRef.current.slicePlane
    const pane =
      plane === 'axial' ? 0 : plane === 'coronal' ? 1 : plane === 'sagittal' ? 2 : paneRef.current
    if (pane === 0) nv.moveCrosshairInVox(0, 0, step)
    else if (pane === 1) nv.moveCrosshairInVox(0, step, 0)
    else nv.moveCrosshairInVox(step, 0, 0)
    nv.createOnLocationChange()
  }

  /** Read position, intensity and label under the pointer — what a reader
   *  expects from a status bar, where Niivue itself only reports the crosshair
   *  (set by clicking). Also remembers which pane the pointer is over, for the
   *  slice keys. Coalesced to one update per frame. */
  const hoverRaf = useRef(0)
  const onPointerMove = (e: React.PointerEvent) => {
    const nv = nvRef.current
    const canvas = canvasRef.current
    if (!nv || !canvas || nv.volumes.length === 0) return
    const rect = canvas.getBoundingClientRect()
    const dpr = nv.uiData.dpr ?? 1
    const x = (e.clientX - rect.left) * dpr
    const y = (e.clientY - rect.top) * dpr
    const tile = nv.screenSlices.find((t) => {
      const [l, tp, w, h] = t.leftTopWidthHeight
      return x >= l && x < l + w && y >= tp && y < tp + h
    })
    if (!tile || tile.axCorSag > 2) return
    paneRef.current = tile.axCorSag
    const frac = nv.canvasPos2frac([x, y])
    if (frac[0] < 0) return
    const mm = nv.frac2mm(frac)
    const vox = nv.frac2vox(frac)
    const values = nv.volumes.map((v) => {
      const vx = v.mm2vox([mm[0], mm[1], mm[2]])
      return v.getValue(vx[0], vx[1], vx[2], v.frame4D)
    })
    const next: Readout = {
      vox: [vox[0], vox[1], vox[2]],
      mm: [mm[0], mm[1], mm[2]],
      baseValue: values[0] ?? NaN,
      overlayValue: values.length > 1 ? values[1] : null,
      pane: tile.axCorSag,
    }
    if (!hoverRaf.current) {
      hoverRaf.current = requestAnimationFrame(() => {
        hoverRaf.current = 0
        setReadout(next)
      })
    }
  }
  const onPointerLeave = () => {
    // Back to the crosshair's values once the pointer is off the image.
    const nv = nvRef.current
    if (nv && nv.volumes.length > 0) nv.createOnLocationChange()
  }

  const applyWindow = (min: number, max: number) => {
    const nv = nvRef.current
    if (!nv || nv.volumes.length === 0) return
    const base = nv.volumes[0]
    base.cal_min = Math.min(min, max)
    base.cal_max = Math.max(min, max)
    nv.updateGLVolume()
    setWin({ min: base.cal_min, max: base.cal_max })
  }

  /** Move the crosshair into a label: its centroid, or — for a curved or
   *  hollow structure whose centroid lies outside it — the nearest voxel that
   *  carries the id (native voxel → world mm → frac). */
  const jumpToLabel = (stat: LabelStat) => {
    const nv = nvRef.current
    const ov = overlayVolRef.current
    if (!nv || !ov) return
    const affine = ov.vol.hdr?.affine
    const img = ov.vol.img
    if (!affine || !img) return
    const [nx, ny, nz] = ov.info.dims
    const at = (i: number, j: number, k: number) => Math.round(img[i + j * nx + k * nx * ny])
    let [i, j, k] = stat.centroid.map(Math.round)
    if (at(i, j, k) !== stat.id) {
      // Grow a cube around the centroid until a voxel of the label turns up.
      let best: [number, number, number] | null = null
      let bestD = Infinity
      for (let r = 1; r <= 64 && !best; r++) {
        for (let dk = -r; dk <= r; dk++) {
          for (let dj = -r; dj <= r; dj++) {
            for (let di = -r; di <= r; di++) {
              if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== r) continue
              const x = i + di
              const y = j + dj
              const z = k + dk
              if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) continue
              if (at(x, y, z) !== stat.id) continue
              const d = di * di + dj * dj + dk * dk
              if (d < bestD) {
                bestD = d
                best = [x, y, z]
              }
            }
          }
        }
      }
      if (best) [i, j, k] = best
    }
    const mm: [number, number, number] = [0, 0, 0]
    for (let r = 0; r < 3; r++) {
      mm[r] = affine[r][0] * i + affine[r][1] * j + affine[r][2] * k + affine[r][3]
    }
    const frac = nv.mm2frac(mm)
    nv.scene.crosshairPos = [frac[0], frac[1], frac[2]]
    nv.drawScene()
    nv.createOnLocationChange()
  }

  const toggleIsolate = (id: number) => {
    const cur = overlayRef.current.isolate
    if (!cur) {
      patchOverlay({ isolate: [id] })
      return
    }
    const next = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]
    patchOverlay({ isolate: next.length ? next : null })
  }

  const overlayName = overlay.path.split('/').pop() ?? ''
  const info = overlayVol?.info ?? null
  const isCT = baseInfo?.isCT ?? false
  const overlayLabel =
    info && info.kind === 'label' && readout?.overlayValue != null && readout.overlayValue > 0
      ? `${Math.round(readout.overlayValue)}${info.names.get(Math.round(readout.overlayValue)) ? ' · ' + info.names.get(Math.round(readout.overlayValue)) : ''}`
      : null

  return (
    <div className="volume-view">
      {/* Only rendered once something is actually overlaid: with drag-and-drop
          as the only way in, an always-present bar would be a permanent strip of
          controls for a state the viewer is usually not in. */}
      {overlay.path && (
        <div className="overlay-bar">
          <span className="overlay-label" title={overlay.path}>
            {overlayName}
          </span>
          {info && (
            <select
              className="vs-select overlay-kind"
              title="How the overlay is drawn: a label map (one colour per id) or a continuous map"
              value={info.kind}
              onChange={(e) => patchOverlay({ kind: e.target.value as OverlayKind, isolate: null })}
            >
              <option value="label">Labels</option>
              <option value="continuous">Continuous</option>
            </select>
          )}
          {info?.kind === 'continuous' && (
            <>
              <select
                className="vs-select overlay-cmap"
                title="Colormap"
                value={overlay.colormap}
                onChange={(e) => patchOverlay({ colormap: e.target.value })}
              >
                {CONTINUOUS_COLORMAPS.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              <label className="overlay-threshold" title="Voxels below this value are transparent">
                <span className="overlay-opacity-label">≥</span>
                <input
                  type="number"
                  step="any"
                  value={formatNumber(overlay.threshold ?? defaultThreshold(info))}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    if (Number.isFinite(v)) patchOverlay({ threshold: v })
                  }}
                />
              </label>
            </>
          )}
          {info?.kind === 'label' && (
            <button
              className={legendOpen ? 'btn-icon active' : 'btn-icon'}
              title="Labels: names, volumes, isolate"
              onClick={() => setLegendOpen((v) => !v)}
            >
              <ListIcon size={13} />
            </button>
          )}
          <span className="overlay-opacity-label">Opacity</span>
          <input
            className="overlay-opacity"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={overlay.opacity}
            title={`Opacity ${Math.round(overlay.opacity * 100)}%`}
            onChange={(e) => patchOverlay({ opacity: Number(e.target.value) })}
          />
          <button
            className={overlay.hidden ? 'btn-icon active' : 'btn-icon'}
            title={overlay.hidden ? 'Show overlay (o)' : 'Hide overlay (o)'}
            onClick={() => patchOverlay({ hidden: !overlay.hidden })}
          >
            {overlay.hidden ? <EyeOffIcon size={13} /> : <EyeIcon size={13} />}
          </button>
          <button className="btn-icon" title="Remove overlay" onClick={() => setOverlayPath('')}>
            <XIcon size={13} />
          </button>
        </div>
      )}
      {loadError && <div className="panel-error">{loadError}</div>}
      <div className="niivue-sizer" ref={sizerRef}>
        <div
          ref={dropRef}
          className={`niivue-dropzone${dragOver ? ' drag-over' : ''}`}
          tabIndex={0}
          onDragOverCapture={onDragOver}
          onDragLeave={() => setDragOver(false)}
          onDropCapture={onDrop}
          onKeyDown={onKeyDown}
          onPointerDownCapture={() => dropRef.current?.focus({ preventScroll: true })}
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
        >
          <canvas ref={canvasRef} className="niivue-canvas" />
          {dragOver && <div className="dropzone-hint">Drop to overlay</div>}
        </div>
        {legendOpen && info?.kind === 'label' && (
          <LabelLegend
            info={info}
            path={overlay.path}
            isolate={overlay.isolate}
            onJump={jumpToLabel}
            onToggle={toggleIsolate}
            onShowAll={() => patchOverlay({ isolate: null })}
            onClose={() => setLegendOpen(false)}
          />
        )}
      </div>
      {baseInfo && (
        <div className="viewer-status">
          <span
            className="st-item"
            title={`Voxel index under the crosshair. Grid ${baseInfo.dims.join('×')} voxels of ${formatSpacing(baseInfo.pixDims)}${baseInfo.frames > 1 ? `, ${baseInfo.frames} frames` : ''}`}
          >
            <span className="st-key">Voxel</span>
            <span className="st-val">{readout ? readout.vox.join(' ') : '–'}</span>
          </span>
          <span className="st-item" title="World coordinates under the crosshair">
            <span className="st-key">mm</span>
            <span className="st-val">{readout ? readout.mm.map(formatMm).join(' ') : '–'}</span>
          </span>
          <span className="st-item" title="Intensity under the crosshair">
            <span className="st-key">Value</span>
            <span className="st-val st-strong">
              {readout ? formatIntensity(readout.baseValue, isCT) : '–'}
            </span>
          </span>
          {info && (
            <span className="st-item st-label" title="Overlay under the crosshair">
              <span className="st-key">{info.kind === 'label' ? 'Label' : 'Overlay'}</span>
              <span className="st-val st-strong">
                {!readout
                  ? '–'
                  : info.kind === 'label'
                    ? (overlayLabel ?? 'background')
                    : readout.overlayValue == null
                      ? '–'
                      : formatIntensity(readout.overlayValue, false)}
              </span>
            </span>
          )}
          <span className="status-spacer" />
          <span className="st-item status-window-anchor" title="Intensity window (right-drag on the image also adjusts it)">
            <span className="st-key">Window</span>
            <button
              className={windowOpen ? 'status-window active' : 'status-window'}
              onClick={() => setWindowOpen((v) => !v)}
            >
              {win ? `${formatNumber(win.min)} – ${formatNumber(win.max)}` : '–'}
              {isCT ? ' HU' : ''}
            </button>
            {windowOpen && win && (
              <WindowPopover
                key={`${win.min}|${win.max}`}
                window={win}
                isCT={isCT}
                robust={[baseInfo.robustMin, baseInfo.robustMax]}
                full={[baseInfo.min, baseInfo.max]}
                onApply={applyWindow}
                onClose={() => setWindowOpen(false)}
              />
            )}
          </span>
        </div>
      )}
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

/** Voxel spacing: one number when isotropic ("0.7 mm"), else all three. */
function formatSpacing(p: [number, number, number]): string {
  const f = (v: number) => v.toFixed(2).replace(/\.?0+$/, '')
  const iso = Math.abs(p[0] - p[1]) < 1e-3 && Math.abs(p[1] - p[2]) < 1e-3
  return iso ? `${f(p[0])} mm` : `${p.map(f).join('×')} mm`
}

function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return '–'
  if (Number.isInteger(v)) return String(v)
  return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)
}

/** The labels present in the overlay: colour, id, name, volume. A row jumps the
 *  crosshair to the structure; its eye isolates it (several can be combined). */
function LabelLegend({
  info,
  path,
  isolate,
  onJump,
  onToggle,
  onShowAll,
  onClose,
}: {
  info: OverlayInfo
  path: string
  isolate: number[] | null
  onJump: (s: LabelStat) => void
  onToggle: (id: number) => void
  onShowAll: () => void
  onClose: () => void
}) {
  const [query, setQuery] = useState('')
  const q = query.trim().toLowerCase()
  const rows = info.stats.filter((s) => {
    if (!q) return true
    const name = info.names.get(s.id) ?? ''
    return name.toLowerCase().includes(q) || String(s.id) === q
  })
  return (
    <div className="label-legend">
      <div className="label-legend-head">
        <span>
          {info.stats.length} label{info.stats.length === 1 ? '' : 's'}
        </span>
        {isolate && (
          <button className="btn-text" onClick={onShowAll}>
            Show all
          </button>
        )}
        <button className="btn-icon" title="Close" onClick={onClose}>
          <XIcon size={12} />
        </button>
      </div>
      {info.stats.length > 12 && (
        <input
          className="label-legend-search"
          placeholder="Find a structure…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      )}
      <div className="label-legend-list">
        {rows.map((s) => {
          const shown = !isolate || isolate.includes(s.id)
          return (
            <div
              key={s.id}
              className={`label-row${shown ? '' : ' dim'}`}
              title="Click to jump to this structure"
              onClick={() => onJump(s)}
            >
              <span className="label-swatch" style={{ background: cssColor(info.color(s.id)) }} />
              <span className="label-id">{s.id}</span>
              <span className="label-name">{info.names.get(s.id) ?? ''}</span>
              <span className="label-ml">{formatMl(voxelsToMl(s.voxels, info.pixDims))}</span>
              <button
                className="btn-icon"
                title={shown && isolate ? 'Hide' : 'Show only this (add others with further clicks)'}
                onClick={(e) => {
                  e.stopPropagation()
                  onToggle(s.id)
                }}
              >
                {shown ? <EyeIcon size={12} /> : <EyeOffIcon size={12} />}
              </button>
            </div>
          )
        })}
        {rows.length === 0 && <div className="label-row empty">No match</div>}
      </div>
      {info.names.size === 0 && (
        <div className="label-legend-foot" title={LABEL_NAMES_HELP}>
          No names. Add <code>{labelFileCandidates(path)[0].split('/').pop()}</code> beside the
          file.
        </div>
      )}
    </div>
  )
}

const LABEL_NAMES_HELP =
  'A CSV with a label,structure header: one row per label id and its name, as the stacks write it. The viewer ships no colour table of its own.'

/** Intensity window: presets plus editable bounds. Opens above the status bar. */
function WindowPopover({
  window: win,
  isCT,
  robust,
  full,
  onApply,
  onClose,
}: {
  window: { min: number; max: number }
  isCT: boolean
  robust: [number, number]
  full: [number, number]
  onApply: (min: number, max: number) => void
  onClose: () => void
}) {
  // Remounted by the parent (keyed on the window) whenever the window changes,
  // so the fields start from the current bounds without syncing in an effect.
  const [min, setMin] = useState(formatNumber(win.min))
  const [max, setMax] = useState(formatNumber(win.max))
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const commit = () => {
    const a = Number(min)
    const b = Number(max)
    if (Number.isFinite(a) && Number.isFinite(b) && a !== b) onApply(a, b)
  }
  const presets = [
    { name: 'Auto', min: robust[0], max: robust[1] },
    { name: 'Full range', min: full[0], max: full[1] },
    ...(isCT ? CT_WINDOW_PRESETS : []),
  ]
  return (
    <>
      <div className="vs-backdrop" onClick={onClose} />
      <div className="wl-popover" role="dialog" aria-label="Intensity window">
        <div className="wl-presets">
          {presets.map((p) => (
            <button
              key={p.name}
              type="button"
              className={
                Math.abs(p.min - win.min) < 1e-6 && Math.abs(p.max - win.max) < 1e-6
                  ? 'vs-seg active'
                  : 'vs-seg'
              }
              onClick={() => onApply(p.min, p.max)}
            >
              {p.name}
            </button>
          ))}
        </div>
        <div className="wl-bounds">
          <label>
            Min
            <input
              type="number"
              step="any"
              value={min}
              onChange={(e) => setMin(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => e.key === 'Enter' && commit()}
            />
          </label>
          <label>
            Max
            <input
              type="number"
              step="any"
              value={max}
              onChange={(e) => setMax(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => e.key === 'Enter' && commit()}
            />
          </label>
        </div>
        <div className="wl-hint">Right-drag on the image adjusts the window too.</div>
      </div>
    </>
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
  overlay,
  onOverlayChange,
}: {
  path: string | null
  /** True while a separator is being dragged; freezes the volume canvas size. */
  isResizing?: boolean
  /** The overlay on the open volume (App-owned; see OverlayState). */
  overlay: OverlayState
  onOverlayChange: (next: OverlayState) => void
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
  const [snapshotToken, setSnapshotToken] = useState(0)
  // The overlay is tagged with the base file it belongs to, so it's transparently
  // ignored once a different file is opened — no state reset needed.
  const overlayForThisFile = overlay.base === path ? overlay : { ...EMPTY_OVERLAY, base: path ?? '' }
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
              title="Save a PNG of the current view"
              onClick={() => setSnapshotToken((t) => t + 1)}
            >
              <CameraIcon />
            </button>
          )}
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
              snapshotToken={snapshotToken}
              overlay={overlayForThisFile}
              onOverlayChange={onOverlayChange}
            />
            {rebuilding && (
              <div className="volume-loading">
                <span className="volume-spinner" />
                <span>Loading volume…</span>
              </div>
            )}
          </div>
        )}
        {kind === 'dicom' && (
          <div className="viewer-message">
            DICOM files are not displayed directly. Ask the agent to convert the series to NIfTI
            with the DICOM stack, then open the result here.
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
