import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ConfirmContext, type ConfirmFn, type ConfirmOptions } from '../confirm'

interface Pending extends ConfirmOptions {
  question: string
  resolve: (ok: boolean) => void
}

/** The app's confirmation dialog, replacing the browser's `confirm()` popup.
 *  Portalled to `document.body` above every window (the `over-window`
 *  backdrop), cancelled by Escape, the backdrop, or Cancel, which holds focus
 *  by default so Enter never deletes anything by accident. */
export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null)
  const confirm = useCallback<ConfirmFn>(
    (question, opts) =>
      new Promise<boolean>((resolve) => {
        setPending({ question, ...opts, resolve })
      }),
    [],
  )
  const settle = useCallback(
    (ok: boolean) => {
      pending?.resolve(ok)
      setPending(null)
    },
    [pending],
  )
  useEffect(() => {
    if (!pending) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') settle(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pending, settle])
  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      {pending &&
        createPortal(
          <div className="modal-backdrop over-window" onClick={() => settle(false)}>
            <div
              className="modal confirm-dialog"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="confirm-question"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 id="confirm-question">{pending.question}</h3>
              {pending.body && <p>{pending.body}</p>}
              <div className="modal-actions">
                <button type="button" className="btn-text" autoFocus onClick={() => settle(false)}>
                  Cancel
                </button>
                <button type="button" className="btn-danger" onClick={() => settle(true)}>
                  {pending.action ?? 'Delete'}
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </ConfirmContext.Provider>
  )
}
