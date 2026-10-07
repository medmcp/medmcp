import { VOLUME_EXT } from '../fileKinds'

/** Case-insensitive form of the viewer's volume extensions, for stripping. */
const VOLUME_EXT_I = new RegExp(VOLUME_EXT.source, 'i')

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
  const n = Number.isFinite(maxLabel) ? Math.max(1, Math.min(MAX_LABELS, Math.ceil(maxLabel))) : 1
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

export const EMPTY_OVERLAY: OverlayState = Object.freeze({
  base: '',
  path: '',
  opacity: 0.6,
  kind: null,
  colormap: 'hot',
  threshold: null,
  isolate: null,
  hidden: false,
})

export function overlayFor(base: string, path: string): OverlayState {
  return { ...EMPTY_OVERLAY, base, path }
}


/** NIfTI intent code marking a label image. */
export const NII_INTENT_LABEL = 1002

/** Decide whether an overlay is a label map (integer ids) or a continuous map
 *  (probabilities, intensities, deformation magnitudes). A header that says
 *  label is believed; otherwise the data decides, sampled so a large volume
 *  costs the same as a small one: any non-integer or negative value, more
 *  than 1024 distinct values or an id above MAX_LABELS means continuous. An
 *  integer anatomical image (a uint8 template, an int16 T1) can pass all of
 *  those, so the last test is what sets a label map apart from any image of
 *  anatomy: it is piecewise constant, so most neighbouring voxels are equal
 *  (measured ~1.0 for a segmentation, below 0.03 for MR). */
export function classifyOverlayData(
  img: ArrayLike<number>,
  sclSlope = 1,
  sclInter = 0,
  intentCode = 0,
): OverlayKind {
  if (intentCode === NII_INTENT_LABEL) return 'label'
  const n = img.length
  if (n === 0) return 'continuous'
  const slope = sclSlope === 0 ? 1 : sclSlope
  const step = Math.max(1, Math.floor(n / 300_000))
  const seen = new Set<number>()
  let max = 0
  let pairs = 0
  let equal = 0
  for (let i = 0; i < n; i += step) {
    const v = img[i] * slope + sclInter
    if (v !== Math.round(v) || v < 0) return 'continuous'
    if (v > max) max = v
    if (seen.size <= 1024) seen.add(v)
    if (i + 1 < n) {
      pairs++
      if (img[i + 1] === img[i]) equal++
    }
  }
  if (max > MAX_LABELS || seen.size > 1024) return 'continuous'
  if (pairs > 0 && equal / pairs < 0.5) return 'continuous'
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
        const id = Math.round(img[idx])
        // 0 is background; a NaN voxel (forced label kind on float data) or a
        // negative value (sign-flipped background) is no id.
        if (!(id > 0) || !Number.isFinite(id)) continue
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

/** Sidecar files that may name the labels of a segmentation, best first.
 *
 *  The viewer knows nothing about which tool made a label map and ships no
 *  colour table; names come only from a file beside the volume. The stacks'
 *  convention: `<stem>_dseg.nii.gz` is accompanied by `<stem>_labels.csv` with
 *  a `label,structure` header (what the TotalSegmentator stack writes). A
 *  volume not named `_dseg` is paired with `<name>_labels.csv`. Without the
 *  file the legend shows ids only. */
export function labelFileCandidates(volumePath: string): string[] {
  const base = volumePath.replace(VOLUME_EXT_I, '')
  const stem = base.replace(/_dseg$/i, '')
  return stem === base ? [`${base}_labels.csv`] : [`${stem}_labels.csv`, `${base}_labels.csv`]
}

/** Parse a label-name table. Expects the stacks' `label,structure` header but
 *  tolerates none or another (`index\tname`): the first integer column is the
 *  id, the first non-numeric column (after it, else before it) the name.
 *  Comma, tab or semicolon separated; a quoted cell may contain the separator. */
export function parseLabelNames(text: string): Map<number, string> {
  const names = new Map<number, string>()
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const cells = (line.match(/"[^"]*"|[^\t,;]+/g) ?? []).map((c) => c.trim().replace(/^"|"$/g, ''))
    const idIdx = cells.findIndex((c) => /^\d+$/.test(c))
    if (idIdx < 0) continue
    const isName = (c: string) => c !== '' && !/^[\d.]+$/.test(c)
    const name = cells.slice(idIdx + 1).find(isName) ?? cells.slice(0, idIdx).find(isName)
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

export type VolumeUnit = 'mL' | 'mm³'

/** One unit for a whole legend, chosen from its largest structure: mL once
 *  anything reaches a millilitre, else mm³. Mixed units in one column read as
 *  if the small structures were the large ones. */
export function volumeUnit(maxMl: number): VolumeUnit {
  return maxMl >= 1 ? 'mL' : 'mm³'
}

export function formatVolume(ml: number, unit: VolumeUnit): string {
  if (!Number.isFinite(ml)) return '–'
  if (unit === 'mm³') {
    const mm3 = ml * 1000
    return `${mm3 < 9.95 ? mm3.toFixed(1) : mm3.toFixed(0)} mm³`
  }
  const digits = ml >= 99.95 ? 0 : ml >= 9.995 ? 1 : 2
  return `${ml.toFixed(digits)} mL`
}
