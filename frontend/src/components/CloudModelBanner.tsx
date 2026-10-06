import { useCallback, useEffect, useState } from 'react'
import { fetchCloudModel, setCloudModelEnabled } from '../api'
import type { CloudModelState } from '../types'

interface CloudModelBannerProps {
  /** Bumped by anything that may have changed the cloud-model state. */
  refreshSignal: number
  /** Open the window that owns this feature. */
  onReview: () => void
}

/**
 * A standing warning, shown for as long as chats run on a cloud model.
 *
 * The consent dialog is a moment; this is the reminder that outlives it. Someone
 * who sits down at a workspace where the switch was flipped days ago has no
 * other cue that what they type and what the agent reads now leave the machine.
 *
 * Shown only while the cloud model is actually in force: enabled with a missing
 * key, chats stay on the local model and there is nothing to warn about. Not
 * dismissible, and it carries the off switch itself — it stops when the thing it
 * reports stops, never because someone clicked it away.
 */
export function CloudModelBanner({ refreshSignal, onReview }: CloudModelBannerProps) {
  const [state, setState] = useState<CloudModelState | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => {
    // Best-effort: a failed read must not blank the app. It also must not leave
    // a stale banner up, so the state is cleared either way.
    fetchCloudModel()
      .then(setState)
      .catch(() => setState(null))
  }, [])

  useEffect(load, [load, refreshSignal])

  const switchToLocal = useCallback(() => {
    setBusy(true)
    setCloudModelEnabled(false)
      .catch(() => undefined)
      .finally(() => {
        setBusy(false)
        load()
      })
  }, [load])

  const model = state?.active ? state.model : null
  if (!model) return null

  return (
    <div className="ext-banner" role="status">
      <span className="ext-banner-dot" aria-hidden="true" />
      <span className="ext-banner-text">
        <strong>Cloud model in use.</strong> Everything in a chat, including what the agent
        reads, is sent outside this machine.
      </span>
      <span className="ext-banner-names" title="The model chats are sent to">
        {model.model} · {model.api_base}
      </span>
      <button className="btn-plain ext-banner-action" onClick={onReview} disabled={busy}>
        Review
      </button>
      <button className="btn-plain ext-banner-action" onClick={switchToLocal} disabled={busy}>
        {busy ? 'Switching…' : 'Switch to local'}
      </button>
    </div>
  )
}
