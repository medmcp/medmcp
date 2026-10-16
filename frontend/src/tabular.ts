/** Parsers for the tabular files the viewer shows: CSV/TSV and XLSX.
 *
 *  Both yield the same shape, so one table component renders either. The XLSX
 *  reader covers what viewing needs — cell values and cached formula results,
 *  shared and inline strings — and nothing else (no styles, no formulas), which
 *  keeps it a hundred lines on top of the browser's own XML parser instead of
 *  a spreadsheet library in the bundle. */
import { unzipSync, strFromU8 } from 'fflate'

export interface Sheet {
  name: string
  header: string[]
  rows: string[][]
}

/** The most frequent of comma, semicolon and tab on the first line wins. */
export function detectDelimiter(text: string): string {
  const line = text.slice(0, text.indexOf('\n') === -1 ? undefined : text.indexOf('\n'))
  const counts: [string, number][] = [',', ';', '\t'].map((d) => [d, line.split(d).length - 1])
  counts.sort((a, b) => b[1] - a[1])
  return counts[0][1] > 0 ? counts[0][0] : ','
}

/** RFC 4180 parsing: quoted fields, doubled quotes, newlines inside quotes. */
export function parseDelimited(text: string, name: string, delimiter = detectDelimiter(text)): Sheet {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else quoted = false
      } else field += c
    } else if (c === '"' && field === '') {
      quoted = true
    } else if (c === delimiter) {
      row.push(field)
      field = ''
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else field += c
  }
  if (field !== '' || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  const nonEmpty = rows.filter((r) => r.some((v) => v !== ''))
  const header = nonEmpty[0] ?? []
  const width = header.length
  const body = nonEmpty.slice(1).map((r) => {
    const out = r.slice(0, width)
    while (out.length < width) out.push('')
    return out
  })
  return { name, header, rows: body }
}

const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/

/** A column is numeric when every non-empty cell is a number and at least one exists. */
export function numericColumns(sheet: Sheet): boolean[] {
  return sheet.header.map((_, col) => {
    let seen = false
    for (const r of sheet.rows) {
      const v = r[col]
      if (v === '' || v == null) continue
      if (!NUMBER.test(v.trim())) return false
      seen = true
    }
    return seen
  })
}

// ── XLSX ──

function xml(text: string): Document {
  return new DOMParser().parseFromString(text, 'application/xml')
}

function byTag(root: ParentNode, tag: string): Element[] {
  return [...(root as Element).getElementsByTagNameNS('*', tag)]
}

/** "C" → 2, "AB" → 27: the column part of a cell reference like "AB12". */
function columnIndex(ref: string): number {
  let n = 0
  for (const ch of ref) {
    if (ch < 'A' || ch > 'Z') break
    n = n * 26 + (ch.charCodeAt(0) - 64)
  }
  return n - 1
}

function cellText(cell: Element, shared: string[]): string {
  const t = cell.getAttribute('t')
  if (t === 'inlineStr') return byTag(cell, 't').map((e) => e.textContent ?? '').join('')
  const v = byTag(cell, 'v')[0]?.textContent ?? ''
  if (t === 's') return shared[Number(v)] ?? ''
  if (t === 'b') return v === '1' ? 'TRUE' : 'FALSE'
  return v
}

/** Every worksheet of an .xlsx, in workbook order; the first row is the header. */
export function parseXlsx(bytes: Uint8Array): Sheet[] {
  const files = unzipSync(bytes)
  const read = (path: string): string | null => (files[path] ? strFromU8(files[path]) : null)
  const workbook = read('xl/workbook.xml')
  if (workbook === null) throw new Error('not a spreadsheet: xl/workbook.xml is missing')
  const rels = new Map<string, string>()
  for (const rel of byTag(xml(read('xl/_rels/workbook.xml.rels') ?? ''), 'Relationship')) {
    const target = rel.getAttribute('Target') ?? ''
    rels.set(rel.getAttribute('Id') ?? '', target.startsWith('/') ? target.slice(1) : `xl/${target}`)
  }
  const shared = byTag(xml(read('xl/sharedStrings.xml') ?? '<sst/>'), 'si').map((si) =>
    byTag(si, 't')
      .map((t) => t.textContent ?? '')
      .join(''),
  )
  const sheets: Sheet[] = []
  for (const s of byTag(xml(workbook), 'sheet')) {
    const name = s.getAttribute('name') ?? `Sheet ${sheets.length + 1}`
    const id = [...s.attributes].find((a) => a.localName === 'id')?.value ?? ''
    const text = read(rels.get(id) ?? '')
    if (text === null) continue
    const grid: string[][] = []
    for (const row of byTag(xml(text), 'row')) {
      const cells: string[] = []
      for (const c of byTag(row, 'c')) {
        const col = columnIndex(c.getAttribute('r') ?? '')
        while (cells.length < col) cells.push('')
        cells[col] = cellText(c, shared)
      }
      grid.push(cells)
    }
    const width = Math.max(0, ...grid.map((r) => r.length))
    const padded = grid.map((r) => {
      const out = r.slice()
      while (out.length < width) out.push('')
      return out
    })
    const nonEmpty = padded.filter((r) => r.some((v) => v !== ''))
    sheets.push({ name, header: nonEmpty[0] ?? [], rows: nonEmpty.slice(1) })
  }
  return sheets
}
