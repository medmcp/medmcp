import { useState } from 'react'
import { login } from '../api'

/** Shown instead of the workspace until this browser presents the workspace password. */
export default function LoginScreen() {
  const [value, setValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(value)
      window.location.reload()
    } catch {
      setError('Wrong password.')
      setBusy(false)
    }
  }

  return (
    <div className="login">
      <form className="modal confirm-dialog" onSubmit={(e) => void submit(e)}>
        <h3>MedMCP</h3>
        <p>This workspace has a password.</p>
        <input
          className="login-input"
          type="password"
          aria-label="Password"
          autoComplete="current-password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoFocus
        />
        {error && <p className="login-error">{error}</p>}
        <div className="modal-actions">
          <button type="submit" className="btn-primary" disabled={busy || !value}>
            Open
          </button>
        </div>
      </form>
    </div>
  )
}
