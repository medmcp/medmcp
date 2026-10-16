/** Lines beyond this render without a gutter; the file is still shown in full. */
const MAX_NUMBERED_LINES = 5000

/** Plain text with line numbers. */
export function TextView({ text }: { text: string }) {
  const lines = text.split('\n')
  if (lines.length > MAX_NUMBERED_LINES) return <pre className="text-view">{text}</pre>
  return (
    <div className="data-view">
      <pre className="text-view text-lines">
        {lines.map((l, i) => (
          <div key={i} className="tl">
            <span className="ln">{i + 1}</span>
            <span>{l}</span>
          </div>
        ))}
      </pre>
    </div>
  )
}
