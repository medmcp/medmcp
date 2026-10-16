import { useState } from 'react'

/** Children shown of one large array or object; the rest is one line. */
const MAX_CHILDREN = 500

/** A JSON document as a collapsible tree; objects and arrays open to two levels. */
export function JsonView({ text }: { text: string }) {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch (e) {
    return (
      <div className="data-view">
        <div className="viewer-message">Not valid JSON: {(e as Error).message}</div>
        <pre className="text-view">{text}</pre>
      </div>
    )
  }
  return (
    <div className="data-view">
      <div className="json-view">
        <Node value={value} depth={0} />
      </div>
    </div>
  )
}

function Node({ value, depth, name }: { value: unknown; depth: number; name?: string }) {
  const [open, setOpen] = useState(depth < 2)
  const label = name !== undefined && <span className="json-key">{name}: </span>
  if (value === null || typeof value !== 'object') {
    return (
      <div className="json-row">
        {label}
        <Scalar value={value} />
      </div>
    )
  }
  const isArray = Array.isArray(value)
  const entries = isArray
    ? (value as unknown[]).map((v, i) => [String(i), v] as const)
    : Object.entries(value as Record<string, unknown>)
  const summary = isArray ? `[${entries.length}]` : `{${entries.length}}`
  if (entries.length === 0) {
    return (
      <div className="json-row">
        {label}
        <span className="json-punct">{isArray ? '[]' : '{}'}</span>
      </div>
    )
  }
  return (
    <div className="json-node">
      <button className="json-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="json-caret">{open ? '▾' : '▸'}</span>
        {label}
        <span className="json-punct">{summary}</span>
      </button>
      {open && (
        <div className="json-children">
          {entries.slice(0, MAX_CHILDREN).map(([k, v]) => (
            <Node key={k} name={k} value={v} depth={depth + 1} />
          ))}
          {entries.length > MAX_CHILDREN && (
            <div className="json-row json-more">… {entries.length - MAX_CHILDREN} more</div>
          )}
        </div>
      )}
    </div>
  )
}

function Scalar({ value }: { value: unknown }) {
  if (typeof value === 'string') return <span className="json-str">"{value}"</span>
  if (typeof value === 'number') return <span className="json-num">{String(value)}</span>
  if (typeof value === 'boolean') return <span className="json-bool">{String(value)}</span>
  return <span className="json-null">null</span>
}
