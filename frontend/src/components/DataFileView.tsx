import { useEffect, useState } from 'react'
import type { FileKind } from '../fileKinds'
import { parseDelimited, parseXlsx, type Sheet } from '../tabular'
import { JsonView } from './JsonView'
import { MarkdownView } from './MarkdownView'
import { TableView } from './TableView'
import { TextView } from './TextView'

/** Text beyond this is cut; a data file this large is for a tool, not for reading. */
const MAX_TEXT = 2_000_000

type Loaded = { sheets: Sheet[] } | { text: string } | { binary: true } | { error: string }

/** Whether the first bytes of a file of unknown type read as text: no NUL byte,
 *  and almost everything printable or whitespace once decoded. */
function looksLikeText(bytes: Uint8Array): boolean {
  const sample = bytes.subarray(0, 8192)
  if (sample.length === 0) return true
  let control = 0
  for (const b of sample) {
    if (b === 0) return false
    if (b < 32 && b !== 9 && b !== 10 && b !== 13) control++
  }
  if (control > sample.length / 100) return false
  const decoded = new TextDecoder('utf-8', { fatal: false }).decode(sample)
  return (decoded.match(/\uFFFD/g) ?? []).length < decoded.length / 50
}

/** Read at most *limit* bytes of a response body, then stop downloading. */
async function readPrefix(r: Response, limit: number): Promise<{ bytes: Uint8Array; done: boolean }> {
  const reader = r.body?.getReader()
  if (!reader) return { bytes: new Uint8Array(await r.arrayBuffer()), done: true }
  const chunks: Uint8Array[] = []
  let size = 0
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    chunks.push(value)
    size += value.length
    if (size >= limit) {
      await reader.cancel()
      break
    }
  }
  const bytes = new Uint8Array(size)
  let at = 0
  for (const c of chunks) {
    bytes.set(c, at)
    at += c.length
  }
  return { bytes: bytes.subarray(0, limit), done: size < limit }
}

/** Loads a non-image file once and hands it to the view for its kind. */
export function DataFileView({ url, path, kind }: { url: string; path: string; kind: FileKind }) {
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  useEffect(() => {
    let cancelled = false
    const name = path.split('/').pop() ?? path
    fetch(url)
      .then(async (r) => {
        // Without this, a 404's JSON body would render as the file's content.
        if (!r.ok) throw new Error(`could not load file (HTTP ${r.status})`)
        if (kind === 'sheet') return { sheets: parseXlsx(new Uint8Array(await r.arrayBuffer())) }
        if (kind === 'other') {
          // Unknown extension: show it as text when the bytes say it is text
          // (.bval, .cfg, .jsonl, …); otherwise leave the download link.
          const { bytes, done } = await readPrefix(r, MAX_TEXT)
          if (!looksLikeText(bytes)) return { binary: true as const }
          const text = new TextDecoder().decode(bytes)
          return { text: done ? text : text + '\n… (truncated)' }
        }
        const text = await r.text()
        if (kind === 'table') return { sheets: [parseDelimited(text, name)] }
        return { text: text.length > MAX_TEXT ? text.slice(0, MAX_TEXT) + '\n… (truncated)' : text }
      })
      .then((v) => {
        if (!cancelled) setLoaded(v)
      })
      .catch((e: unknown) => {
        if (!cancelled) setLoaded({ error: e instanceof Error ? e.message : String(e) })
      })
    return () => {
      cancelled = true
    }
  }, [url, path, kind])
  if (loaded === null) return <div className="viewer-message">Loading…</div>
  if ('error' in loaded) return <div className="viewer-message">{loaded.error}</div>
  if ('binary' in loaded) {
    return (
      <div className="viewer-message">
        No viewer for this file type.{' '}
        <a href={url} download>
          Download {path}
        </a>
      </div>
    )
  }
  if ('sheets' in loaded) return <TableView sheets={loaded.sheets} />
  if (kind === 'json') return <JsonView text={loaded.text} />
  if (kind === 'markdown') return <MarkdownView text={loaded.text} />
  return <TextView text={loaded.text} />
}
