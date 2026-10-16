import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const REMARK_PLUGINS = [remarkGfm]
// A document in the workspace is rendered, not trusted: nothing in it may make
// the browser fetch anything (raw HTML is off by default).
const NO_LOADING_ELEMENTS = ['img']

/** A Markdown file rendered as a document (GFM tables included). */
export function MarkdownView({ text }: { text: string }) {
  return (
    <div className="data-view">
      <div className="md-view">
        <div className="md-doc">
          <ReactMarkdown remarkPlugins={REMARK_PLUGINS} disallowedElements={NO_LOADING_ELEMENTS}>
            {text}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
