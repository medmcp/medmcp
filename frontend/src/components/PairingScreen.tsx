import { useState } from 'react'
import { pair } from '../api'

/** Shown instead of the workspace until this browser presents the access token. */
export default function PairingScreen() {
  const [value, setValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    // A pasted link works as well as the bare token.
    let token = value.trim()
    try {
      token = new URL(token).searchParams.get('token') ?? token
    } catch {
      /* not a URL: the token itself */
    }
    setBusy(true)
    setError(null)
    try {
      await pair(token)
      window.location.reload()
    } catch {
      setError('That link or token is not this workspace’s.')
      setBusy(false)
    }
  }

  return (
    <div className="pairing">
      <form className="modal confirm-dialog pairing-card" onSubmit={(e) => void submit(e)}>
        <h3>Open MedMCP with its link</h3>
        <p>
          Run <code>docker exec medmcp-medmcp-1 medmcp url</code> and open the link it prints, or
          paste it here.
        </p>
        <input
          className="pairing-input"
          aria-label="Link or token"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoFocus
        />
        {error && <p className="pairing-error">{error}</p>}
        <div className="modal-actions">
          <button type="submit" className="btn-primary" disabled={busy || !value.trim()}>
            Open
          </button>
        </div>
      </form>
    </div>
  )
}
