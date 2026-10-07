import { createContext, useContext } from 'react'

export interface ConfirmOptions {
  /** A second line under the question. */
  body?: string
  /** Label of the confirming button (default "Delete"). */
  action?: string
}

/** Ask the person to confirm a destructive step; resolves true when they do. */
export type ConfirmFn = (question: string, opts?: ConfirmOptions) => Promise<boolean>

/** Provided by `ConfirmProvider`; the default refuses, so a dialog can never be
 *  skipped by rendering outside the provider. */
export const ConfirmContext = createContext<ConfirmFn>(() => Promise.resolve(false))

export function useConfirm(): ConfirmFn {
  return useContext(ConfirmContext)
}
