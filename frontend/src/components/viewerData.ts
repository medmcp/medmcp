import { FREESURFER_LUT } from './freesurferLut'

/** Pure helpers behind the volume viewer: label palettes, overlay classification,
 *  label statistics, label-name files and intensity-window presets. No React, no
 *  Niivue instance — everything here is testable with plain arrays. */

export interface LabelColorMap {
  R: number[]
  G: number[]
  B: number[]
  A: number[]
  I: number[]
}

/** Color for label *i* (1-based): hue steps by the golden angle so neighbouring
 *  ids are far apart on the wheel, with lightness and saturation cycling on
 *  periods that are coprime to each other so two ids only share a colour after
 *  hundreds of labels — a whole-body segmentation has about 120. */
export function labelColor(i: number): [number, number, number] {
  const hue = (i * 137.50776405) % 360
  const sat = [0.78, 0.62, 0.9][i % 3]
  const light = [0.55, 0.42, 0.66, 0.5][i % 4]
  return hslToRgb(hue, sat, light)
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const hp = h / 60
  const x = c * (1 - Math.abs((hp % 2) - 1))
  const sector: [number, number, number] =
    hp < 1
      ? [c, x, 0]
      : hp < 2
        ? [x, c, 0]
        : hp < 3
          ? [0, c, x]
          : hp < 4
            ? [0, x, c]
            : hp < 5
              ? [x, 0, c]
              : [c, 0, x]
  const m = l - c / 2
  return [
    Math.round((sector[0] + m) * 255),
    Math.round((sector[1] + m) * 255),
    Math.round((sector[2] + m) * 255),
  ]
}

export function cssColor(rgb: [number, number, number]): string {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

/** Largest label id a colormap is built for. Niivue uploads one texel per id, so
 *  the table is sized to the data rather than fixed; this is only a sanity cap
 *  for a volume that is not really a label map. */
export const MAX_LABELS = 4096

export type LabelColorFn = (id: number) => [number, number, number]

/** A discrete colormap covering ids 0..maxLabel: 0 transparent (background),
 *  every other id a distinct colour from *color*. With *visible* set, ids
 *  outside it are transparent too — how "isolate this structure" is drawn. */
export function buildLabelColormap(
  maxLabel: number,
  color: LabelColorFn = labelColor,
  visible?: ReadonlySet<number>,
): LabelColorMap {
  const n = Math.max(1, Math.min(MAX_LABELS, Math.ceil(maxLabel)))
  const R = [0]
  const G = [0]
  const B = [0]
  const A = [0]
  const I = [0]
  for (let i = 1; i <= n; i++) {
    const [r, g, b] = color(i)
    R.push(r)
    G.push(g)
    B.push(b)
    A.push(visible && !visible.has(i) ? 0 : 255)
    I.push(i)
  }
  return { R, G, B, A, I }
}

export type OverlayKind = 'label' | 'continuous'

/** Colormaps offered for a continuous overlay (Niivue built-ins). */
export const CONTINUOUS_COLORMAPS = ['hot', 'warm', 'cool', 'winter', 'viridis', 'plasma', 'jet', 'gray']

/** What is overlaid on the base volume and how. Owned by App (so the chat can
 *  set it and so it survives the resize-rebuild remount); tagged with the base
 *  file it belongs to, so it is ignored once another file is opened. */
export interface OverlayState {
  base: string
  path: string
  opacity: number
  /** null = auto-detect from the data; the bar lets the user override. */
  kind: OverlayKind | null
  /** Continuous overlays only. */
  colormap: string
  /** Continuous overlays only: voxels below this are transparent. null = auto. */
  threshold: number | null
  /** Label overlays only: ids shown; null = all. */
  isolate: number[] | null
  /** Temporarily hidden (the eye button / the `o` key) for before/after checks. */
  hidden: boolean
}

export const EMPTY_OVERLAY: OverlayState = {
  base: '',
  path: '',
  opacity: 0.6,
  kind: null,
  colormap: 'hot',
  threshold: null,
  isolate: null,
  hidden: false,
}

export function overlayFor(base: string, path: string): OverlayState {
  return { ...EMPTY_OVERLAY, base, path }
}


/** Decide whether an overlay is a label map (integer ids) or a continuous map
 *  (probabilities, intensities, deformation magnitudes). Sampled, so a large
 *  volume costs the same as a small one. Integer-typed anatomical images (an
 *  int16 T1) fall out as continuous through the distinct-value count. */
export function classifyOverlayData(
  img: ArrayLike<number>,
  sclSlope = 1,
  sclInter = 0,
): OverlayKind {
  const n = img.length
  if (n === 0) return 'continuous'
  const slope = sclSlope === 0 ? 1 : sclSlope
  const step = Math.max(1, Math.floor(n / 300_000))
  const seen = new Set<number>()
  let max = 0
  for (let i = 0; i < n; i += step) {
    const v = img[i] * slope + sclInter
    if (v !== Math.round(v) || v < 0) return 'continuous'
    if (v > max) max = v
    if (seen.size <= 1024) seen.add(v)
  }
  if (max > MAX_LABELS || seen.size > 1024) return 'continuous'
  return 'label'
}

export interface LabelStat {
  id: number
  voxels: number
  /** Voxel-index centroid [i, j, k] in the overlay's own grid. */
  centroid: [number, number, number]
}

/** Count voxels and a centroid per label id in one pass over the raw data. */
export function labelStats(img: ArrayLike<number>, dims: [number, number, number]): LabelStat[] {
  const [nx, ny, nz] = dims
  const counts = new Map<number, { n: number; x: number; y: number; z: number }>()
  let idx = 0
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++, idx++) {
        const v = img[idx]
        if (v === 0 || v === undefined) continue
        const id = Math.round(v)
        let acc = counts.get(id)
        if (!acc) {
          acc = { n: 0, x: 0, y: 0, z: 0 }
          counts.set(id, acc)
        }
        acc.n++
        acc.x += i
        acc.y += j
        acc.z += k
      }
    }
  }
  return [...counts.entries()]
    .map(([id, a]) => ({
      id,
      voxels: a.n,
      centroid: [a.x / a.n, a.y / a.n, a.z / a.n] as [number, number, number],
    }))
    .sort((a, b) => a.id - b.id)
}

/** Whether a set of present label ids reads as a FreeSurfer/FastSurfer
 *  segmentation: at least a handful of ids, nearly all of them in the table.
 *  Then FreeSurfer's own names and colours apply — the ones its users know. */
export function looksLikeFreeSurfer(ids: readonly number[]): boolean {
  if (ids.length < 5) return false
  const known = ids.filter((i) => FREESURFER_LUT.has(i)).length
  return known / ids.length >= 0.9
}

export function freesurferNames(ids: readonly number[]): Map<number, string> {
  const names = new Map<number, string>()
  for (const i of ids) {
    const e = FREESURFER_LUT.get(i)
    if (e) names.set(i, e.name)
  }
  return names
}

export function freesurferColor(id: number): [number, number, number] {
  return FREESURFER_LUT.get(id)?.rgb ?? labelColor(id)
}

/** Candidate sidecar files naming the labels of a segmentation, best first:
 *  the TotalSegmentator stack's `<stem>_labels.csv` (stem = name without
 *  `_dseg`), and the BIDS `<name>_dseg.tsv` beside a `_dseg.nii.gz`. */
export function labelFileCandidates(volumePath: string): string[] {
  const base = volumePath.replace(/\.(nii(\.gz)?|mgz|mgh|nrrd|nhdr|mha|mhd)$/i, '')
  const out: string[] = []
  if (/_dseg$/i.test(base)) {
    const stem = base.replace(/_dseg$/i, '')
    out.push(`${stem}_labels.csv`, `${base}.tsv`, `${stem}_labels.tsv`)
  } else {
    out.push(`${base}_labels.csv`, `${base}_labels.tsv`, `${base}.tsv`, `${base}.csv`)
  }
  return out
}

/** Parse a label-name table. Accepts a header row (`label,structure`,
 *  `index\tname`, …) or none; the first numeric column is the id, the first
 *  non-numeric column after it the name. Comma, tab or semicolon separated. */
export function parseLabelNames(text: string): Map<number, string> {
  const names = new Map<number, string>()
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const cells = line.split(/\t|,|;/).map((c) => c.trim().replace(/^"|"$/g, ''))
    const idIdx = cells.findIndex((c) => /^\d+$/.test(c))
    if (idIdx < 0) continue
    const name = cells.slice(idIdx + 1).find((c) => c && !/^[\d.]+$/.test(c))
    if (!name) continue
    names.set(Number(cells[idIdx]), name)
  }
  return names
}

export interface WindowPreset {
  name: string
  min: number
  max: number
}

/** Standard CT windows, expressed as the [min, max] Hounsfield range. */
export const CT_WINDOW_PRESETS: WindowPreset[] = [
  { name: 'Soft tissue', min: -150, max: 250 },
  { name: 'Lung', min: -1350, max: 150 },
  { name: 'Bone', min: -500, max: 1300 },
  { name: 'Brain', min: 0, max: 80 },
  { name: 'Liver', min: -45, max: 105 },
]

/** A volume whose intensities reach well below zero is in Hounsfield units;
 *  MR and derived maps are non-negative. */
export function looksLikeCT(globalMin: number, globalMax: number): boolean {
  return globalMin <= -500 && globalMax >= 200
}

export function formatIntensity(v: number, isCT: boolean): string {
  if (!Number.isFinite(v)) return '–'
  if (isCT || Number.isInteger(v)) return `${Math.round(v)}${isCT ? ' HU' : ''}`
  const a = Math.abs(v)
  return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toPrecision(3)
}

export function formatMm(v: number): string {
  return Number.isFinite(v) ? v.toFixed(1) : '–'
}

/** Volume of *voxels* voxels in millilitres, given voxel edge lengths in mm. */
export function voxelsToMl(voxels: number, pixDims: [number, number, number]): number {
  return (voxels * pixDims[0] * pixDims[1] * pixDims[2]) / 1000
}

export function formatMl(ml: number): string {
  if (ml >= 100) return `${ml.toFixed(0)} mL`
  if (ml >= 1) return `${ml.toFixed(1)} mL`
  return `${(ml * 1000).toFixed(0)} mm³`
}
