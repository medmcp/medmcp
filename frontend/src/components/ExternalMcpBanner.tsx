import { useCallback, useEffect, useState } from 'react'
import { fetchExternalMcp, setExternalMcpEnabled } from '../api'
import type { ExternalMcpState } from '../types'

interface ExternalMcpBannerProps {
  /** Bumped by anything that may have changed the external-MCP state. */
  refreshSignal: number
  /** Open the settings drawer on the section that owns this feature. */
  onReview: () => void
}

/**
 * A standing warning, shown for as long as external MCP servers are enabled.
 *
 * "Enabled", not "connected": nothing here observes a handshake. What is known
 * is that the agent may reach these servers, which is the thing worth warning
 * about — a server that is unreachable today answers tomorrow, and the warning
 * should not have gone quiet in between.
 *
 * The consent dialog is a moment; this is the reminder that outlives it. Someone
 * who turned the feature on days ago — or who sat down at a workspace where
 * somebody else did — has no cue that tool calls can now leave the machine,
 * because the switch lives behind an Advanced disclosure inside a drawer that is
 * closed almost all of the time. So the warning goes where the work happens.
 *
 * Deliberately not dismissible, and it carries the off switch itself: it stops
 * when the thing it reports stops, and never because someone clicked it away.
 */
export function ExternalMcpBanner({ refreshSignal, onReview }: ExternalMcpBannerProps) {
  const [state, setState] = useState<ExternalMcpState | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => {
    // Best-effort: a failed read must not blank the app. It also must not leave
    // a stale "connected" banner up, so the state is cleared either way.
    fetchExternalMcp()
      .then(setState)
      .catch(() => setState(null))
  }, [])

  useEffect(load, [load, refreshSignal])

  const turnAllOff = useCallback(() => {
    setBusy(true)
    setExternalMcpEnabled(false)
      .catch(() => undefined)
      .finally(() => {
        setBusy(false)
        load()
      })
  }, [load])

  // The feature can be on with everything switched off; nothing is reachable
  // then, so there is nothing to warn about.
  const active = state?.enabled ? state.servers.filter((s) => s.active) : []
  if (active.length === 0) return null

  const plural = active.length === 1 ? '' : 's'
  return (
    <div className="ext-banner" role="status">
      <span className="ext-banner-dot" aria-hidden="true" />
      <span className="ext-banner-text">
        <strong>
          {active.length} external MCP server{plural} enabled.
        </strong>{' '}
        Anything the agent sends to an external server leaves this machine.
      </span>
      <span className="ext-banner-names" title="Enabled external servers">
        {active.map((s) => s.name).join(', ')}
      </span>
      <button className="btn-plain ext-banner-action" onClick={onReview} disabled={busy}>
        Review
      </button>
      <button className="btn-plain ext-banner-action" onClick={turnAllOff} disabled={busy}>
        {busy ? 'Turning off…' : 'Turn all off'}
      </button>
    </div>
  )
}
