import { ExternalMcpSection } from './ExternalMcpSection'
import { XIcon } from './icons'

interface ExternalMcpWindowProps {
  open: boolean
  onClose: () => void
  /** Called when the connected set changed, so the standing warning catches up. */
  onChanged?: () => void
}

/**
 * Connect and manage MCP services hosted outside this machine.
 *
 * A window rather than a settings section, for the reason the stacks browser is
 * one: this is a list with per-item state and an add form, and a narrow column
 * of switches is the wrong shape for it. Settings keeps the switches and points
 * here, so the drawer stays a short list of one-line choices.
 *
 * The window is also where the space for the consent text is, which matters:
 * the decision it asks for is the one that ends the on-premise guarantee.
 */
export function ExternalMcpWindow({ open, onClose, onChanged }: ExternalMcpWindowProps) {
  if (!open) return null
  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="extwin" role="dialog" aria-label="External MCP servers">
        <div className="panel-header">
          <span>External MCP servers</span>
          <span className="panel-actions">
            <button className="btn-icon" title="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>
        <div className="extwin-body">
          <ExternalMcpSection onChanged={onChanged} />
        </div>
      </div>
    </>
  )
}
