/** Shown when the workspace will not serve this browser and no password would help.
 *
 * The origin guard refuses an unexpected `Host` or `Origin` (DNS rebinding), and
 * the isolated agent's own network is refused outright. Both are 403s, and both
 * used to land on the login screen: the correct password then answered "Wrong
 * password.", because `/api/login` is refused by the very same guard. The
 * server's reason is the only thing that says what to change, so it is shown as
 * the server wrote it. A `status` of 0 means the workspace did not answer at all.
 */
export default function UnreachableScreen({
  status,
  detail,
}: {
  status: number
  detail: string
}) {
  const unanswered = status === 0
  return (
    <div className="login">
      <div className="modal confirm-dialog">
        <h3>MedMCP</h3>
        <p>{unanswered ? 'This workspace is not answering.' : 'This workspace refused the connection.'}</p>
        {detail && <p className="login-error">{detail}</p>}
        <p className="login-hint">
          {unanswered ? (
            <>If it has only just been started, give it a moment and reload.</>
          ) : (
            <>
              Opening the workspace by hostname needs that name in{' '}
              <code>MEDMCP_ALLOWED_HOSTS</code>, and a page served from another origin
              needs that origin in <code>MEDMCP_ALLOWED_ORIGINS</code>. Otherwise open it
              at <code>http://127.0.0.1:8100</code>.
            </>
          )}
        </p>
      </div>
    </div>
  )
}
