import { useEffect, useState } from 'react'
import { fetchAgentIsolated } from '../api'

/**
 * A standing notice while the agent is not isolated (a host-native install):
 * every command it is allowed to run then runs with the account that started
 * the server. Not dismissible — it ends when the install changes.
 */
export function IsolationBanner() {
  const [isolated, setIsolated] = useState(true)

  useEffect(() => {
    fetchAgentIsolated()
      .then(setIsolated)
      .catch(() => setIsolated(true))
  }, [])

  if (isolated) return null
  return (
    <div className="ext-banner" role="status">
      <span className="ext-banner-dot" aria-hidden="true" />
      <span className="ext-banner-text">
        <strong>Agent not isolated.</strong> Approved commands run with your account&rsquo;s
        rights; use the container install for real data.
      </span>
    </div>
  )
}
