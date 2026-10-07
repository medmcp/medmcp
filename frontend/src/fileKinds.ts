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

// Extensions offered as chips: the volume and image lists above (minus the
// paired Analyze `.hdr`/`.img` and `.v16`, which a tool result rarely names and
// which need their sibling file), plus pdf and html. Must end the token: a
// following word or `.word` (`x.nii.gz.bak`) means it is not the extension.
const VIEWABLE_EXT = String.raw`\.(?:nii(?:\.gz)?|mgz|mgh|nrrd|nhdr|mha|mhd|png|jpe?g|gif|svg|webp|bmp|pdf|html?)(?!\w|\.\w)`

// A path inside a JSON/quoted string: may contain spaces, ends at the quote.
const QUOTED_PATH = new RegExp(String.raw`"([^"\n]*?${VIEWABLE_EXT})"`, 'giu')
// A path token in free text: slash-separated segments of filename characters,
// ending in an extension the viewer handles. Absolute or relative; a token
// glued to `:` or `\` (a URL port, a Windows path) is not a workspace path.
const SEGMENT = String.raw`[\p{L}\p{N}_.+@%-]+`
const PATH_TOKEN = new RegExp(
  String.raw`(?:/|(?<![\p{L}\p{N}_./:\\-]))(?:${SEGMENT}/)*${SEGMENT}${VIEWABLE_EXT}`,
  'giu',
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
  // Quoted spans may hold a path with spaces — or a sentence that happens to
  // end in one ("Wrote x to /ws/out.nii.gz"), told apart by how the span
  // starts. The spans are blanked before the free-text pass so a quoted path
  // is not re-found as a truncated token; both passes keep their offsets so
  // the result is in order of first mention.
  const candidates: { at: number; path: string }[] = []
  for (const m of text.matchAll(QUOTED_PATH)) {
    const span = m[1]
    const at = m.index + 1
    // A JSON key that happens to look like a file name is not a path.
    if (/^\s*:/.test(text.slice(m.index + m[0].length))) continue
    if (!/\s/.test(span) || /^\.?\//.test(span)) candidates.push({ at, path: span })
    else for (const t of span.matchAll(PATH_TOKEN)) candidates.push({ at: at + t.index, path: t[0] })
  }
  const blanked = text.replace(QUOTED_PATH, (m) => '"'.padEnd(m.length - 1) + '"')
  for (const m of blanked.matchAll(PATH_TOKEN)) candidates.push({ at: m.index, path: m[0] })
  candidates.sort((a, b) => a.at - b.at)
  for (const { path: raw } of candidates) {
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
