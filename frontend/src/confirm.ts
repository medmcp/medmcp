import { createContext, useContext } from 'react'

export interface ConfirmOptions {
  /** A second line under the question. */
  body?: string
  /** Label of the confirming button (default "Delete"). */
  action?: string
  /** "primary" for a step that destroys nothing (default "danger"). */
  tone?: 'danger' | 'primary'
}

/** Ask the person to confirm a destructive step; resolves true when they do. */
export type ConfirmFn = (question: string, opts?: ConfirmOptions) => Promise<boolean>

export interface ChoiceOptions extends ConfirmOptions {
  /** Label of the second way to go ahead, shown between Cancel and the action. */
  alt: string
}

/** Which way the person went; null when they cancelled (Cancel, Escape, backdrop). */
export type Choice = 'action' | 'alt' | null

/** Ask the person to pick one of two ways to go ahead, or neither. */
export type ChoiceFn = (question: string, opts: ChoiceOptions) => Promise<Choice>

/** Provided by `ConfirmProvider`; the default refuses, so a dialog can never be
 *  skipped by rendering outside the provider. */
export const ConfirmContext = createContext<ConfirmFn>(() => Promise.resolve(false))

/** Provided by `ConfirmProvider`; outside it every choice is a cancel. */
export const ChoiceContext = createContext<ChoiceFn>(() => Promise.resolve(null))

export function useConfirm(): ConfirmFn {
  return useContext(ConfirmContext)
}

export function useChoice(): ChoiceFn {
  return useContext(ChoiceContext)
}
