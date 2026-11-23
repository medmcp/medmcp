import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ChoiceContext,
  ConfirmContext,
  type Choice,
  type ChoiceFn,
  type ConfirmFn,
  type ConfirmOptions,
} from '../confirm'

interface Pending extends ConfirmOptions {
  question: string
  /** A second way to go ahead (a choice dialog); absent for a plain confirmation. */
  alt?: string
  resolve: (choice: Choice) => void
}

/** The app's confirmation dialog, replacing the browser's `confirm()` popup.
 *  Portalled to `document.body` above every window (the `over-window`
 *  backdrop), cancelled by Escape, the backdrop, or Cancel, which holds focus
 *  by default so Enter never deletes anything by accident. The same dialog
 *  serves a choice between two ways to go ahead (`useChoice`), where cancelling
 *  stays a third outcome of its own. */
export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null)
  const choose = useCallback<ChoiceFn>(
    (question, opts) =>
      new Promise<Choice>((resolve) => {
        setPending({ question, ...opts, resolve })
      }),
    [],
  )
  const confirm = useCallback<ConfirmFn>(
    (question, opts) =>
      new Promise<boolean>((resolve) => {
        setPending({ question, ...opts, resolve: (choice) => resolve(choice === 'action') })
      }),
    [],
  )
  const settle = useCallback(
    (choice: Choice) => {
      pending?.resolve(choice)
      setPending(null)
    },
    [pending],
  )
  useEffect(() => {
    if (!pending) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') settle(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pending, settle])
  return (
    <ConfirmContext.Provider value={confirm}>
      <ChoiceContext.Provider value={choose}>
        {children}
        {pending &&
          createPortal(
            <div className="modal-backdrop over-window" onClick={() => settle(null)}>
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
                  <button type="button" className="btn-text" autoFocus onClick={() => settle(null)}>
                    Cancel
                  </button>
                  {pending.alt && (
                    <button type="button" className="btn-text" onClick={() => settle('alt')}>
                      {pending.alt}
                    </button>
                  )}
                  <button
                    type="button"
                    className={pending.tone === 'primary' ? 'btn-primary' : 'btn-danger'}
                    onClick={() => settle('action')}
                  >
                    {pending.action ?? 'Delete'}
                  </button>
                </div>
              </div>
            </div>,
            document.body,
          )}
      </ChoiceContext.Provider>
    </ConfirmContext.Provider>
  )
}
