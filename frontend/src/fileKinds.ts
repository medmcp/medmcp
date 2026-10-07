/** File-type routing shared by the viewer and by anything that offers to open a
 *  file in it (the chat's tool-result chips, the explorer). One place, so a
 *  chip never offers to "view" something the viewer will refuse. */

/** Volumes Niivue parses from a single file. DICOM is deliberately absent: Niivue
 *  needs its separate dcm2niix loader for `.dcm`, which this app does not ship —
 *  the DICOM stack converts a series to NIfTI, which is the path we show. */
export const VOLUME_EXT = /\.(nii|nii\.gz|mgz|mgh|nrrd|nhdr|mha|mhd|hdr|img|v16)$/
export const DICOM_EXT = /\.(dcm|dicom)$/
export const IMAGE_EXT = /\.(png|jpe?g|gif|svg|webp|bmp)$/
export const TEXT_EXT = /\.(md|txt|py|json|yaml|yml|toml|csv|tsv|log|sh|js|ts|html|css|xml)$/

export type FileKind = 'volume' | 'dicom' | 'pdf' | 'html' | 'image' | 'text' | 'other'

export function classify(path: string): FileKind {
  const lower = path.toLowerCase()
  if (VOLUME_EXT.test(lower)) return 'volume'
  if (DICOM_EXT.test(lower)) return 'dicom'
  if (lower.endsWith('.pdf')) return 'pdf'
  // Render HTML (e.g. QC reports) rather than showing source — checked before
  // TEXT_EXT, which also matches .html.
  if (lower.endsWith('.html') || lower.endsWith('.htm')) return 'html'
  if (IMAGE_EXT.test(lower)) return 'image'
  if (TEXT_EXT.test(lower)) return 'text'
  return 'other'
}

export function isVolumePath(path: string | null | undefined): path is string {
  return !!path && VOLUME_EXT.test(path.toLowerCase())
}

/** Kinds the viewer renders with something better than a download link. */
export function isViewable(kind: FileKind): boolean {
  return kind === 'volume' || kind === 'image' || kind === 'pdf' || kind === 'html'
}

const VIEWABLE_EXT = String.raw`\.(?:nii(?:\.gz)?|mgz|mgh|nrrd|nhdr|mha|mhd|png|jpe?g|gif|webp|pdf|html?)`

// A path inside a JSON/quoted string: may contain spaces, ends at the quote.
const QUOTED_PATH = new RegExp(String.raw`"([^"\n]*?${VIEWABLE_EXT})"`, 'gi')
// A path token in free text: slash-separated segments of filename characters,
// ending in an extension the viewer handles. Absolute or relative.
const PATH_TOKEN = new RegExp(
  String.raw`(?:/|(?<![\w./-]))(?:[\w.+@%-]+/)*[\w.+@%-]+${VIEWABLE_EXT}\b`,
  'gi',
)

/** Workspace-relative paths of viewable files mentioned in *text*.
 *
 *  Tool results name their outputs as absolute host paths (the agent runs with
 *  the workspace as cwd, so relative ones appear too). An absolute path outside
 *  the workspace is dropped — the viewer could not serve it. Order of first
 *  mention, no duplicates. */
export function extractWorkspacePaths(text: string, workspaceRoot: string | null): string[] {
  const root = workspaceRoot ? workspaceRoot.replace(/\/+$/, '') : null
  const out: string[] = []
  const seen = new Set<string>()
  // Quoted paths first (they may contain spaces), then the free text with those
  // spans removed, so a quoted path is not re-found as a truncated token.
  const candidates = [
    ...[...text.matchAll(QUOTED_PATH)].map((m) => m[1]),
    ...[...text.replace(QUOTED_PATH, '""').matchAll(PATH_TOKEN)].map((m) => m[0]),
  ]
  for (const raw of candidates) {
    let p = raw.trim()
    if (p.startsWith('/')) {
      if (!root || !p.startsWith(root + '/')) continue
      p = p.slice(root.length + 1)
    } else {
      p = p.replace(/^\.\//, '')
    }
    if (!p || p.includes('..') || seen.has(p)) continue
    if (!isViewable(classify(p))) continue
    seen.add(p)
    out.push(p)
  }
  return out
}
