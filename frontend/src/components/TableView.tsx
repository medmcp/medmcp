import { useMemo, useState } from 'react'
import { numericColumns, type Sheet } from '../tabular'

/** Rows shown of a large table; the count line says when it was cut. */
const MAX_ROWS = 5000

type Sort = { col: number; dir: 1 | -1 } | null

/** One table per sheet with a sticky header, right-aligned numbers and click-to-sort. */
export function TableView({ sheets }: { sheets: Sheet[] }) {
  const [index, setIndex] = useState(0)
  const [sort, setSort] = useState<Sort>(null)
  const sheet = sheets[Math.min(index, sheets.length - 1)]
  const numeric = useMemo(() => (sheet ? numericColumns(sheet) : []), [sheet])
  const rows = useMemo(() => {
    if (!sheet) return []
    const out = sheet.rows.slice(0, MAX_ROWS)
    if (!sort) return out
    const { col, dir } = sort
    const asNumber = numeric[col]
    return out.sort((a, b) => {
      const x = a[col] ?? ''
      const y = b[col] ?? ''
      if (x === '' && y !== '') return 1
      if (y === '' && x !== '') return -1
      const c = asNumber ? Number(x) - Number(y) : x.localeCompare(y, undefined, { numeric: true })
      return c * dir
    })
  }, [sheet, sort, numeric])
  if (!sheet) return <div className="viewer-message">The spreadsheet has no sheets.</div>
  const toggleSort = (col: number) =>
    setSort((s) => (s && s.col === col ? (s.dir === 1 ? { col, dir: -1 } : null) : { col, dir: 1 }))
  return (
    <div className="data-view">
      <div className="data-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {sheet.header.map((h, i) => (
                <th
                  key={i}
                  className={numeric[i] ? 'num' : ''}
                  aria-sort={sort?.col === i ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}
                  onClick={() => toggleSort(i)}
                >
                  {h || <span className="data-empty">column {i + 1}</span>}
                  {sort?.col === i && <span className="data-sort">{sort.dir === 1 ? '▲' : '▼'}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((v, j) => (
                  <td key={j} className={numeric[j] ? 'num' : ''}>
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="data-foot">
        {sheets.length > 1 && (
          <span className="data-sheets" role="tablist">
            {sheets.map((s, i) => (
              <button
                key={s.name}
                role="tab"
                aria-selected={i === index}
                className={`market-chip${i === index ? ' on' : ''}`}
                onClick={() => {
                  setIndex(i)
                  setSort(null)
                }}
              >
                {s.name}
              </button>
            ))}
          </span>
        )}
        <span>
          {sheet.rows.length.toLocaleString()} rows × {sheet.header.length} columns
          {sheet.rows.length > MAX_ROWS && ` · showing the first ${MAX_ROWS.toLocaleString()}`}
        </span>
      </div>
    </div>
  )
}
