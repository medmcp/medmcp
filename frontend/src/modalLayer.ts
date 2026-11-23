/**
 * Keyboard containment for everything that opens over a backdrop: the windows,
 * the settings drawer and the dialogs.
 *
 * A backdrop stops the pointer, not the keyboard: Tab walked out of a window
 * into the header and the panels behind it. Each such layer puts `modalLayer`
 * on its root as a ref; while layers are up, everything outside the topmost one
 * is `inert` (its backdrop excepted, which still closes it), Tab wraps inside
 * it, and closing it hands focus back to whatever held it before.
 *
 * A stack rather than a flag, because layers nest: a confirmation over a
 * window, a window over the drawer. Portalled dialogs take part the same way:
 * what is made inert is found by walking up from the layer, wherever it is
 * mounted.
 */

interface Layer {
  el: HTMLElement
  /** What had focus when the layer opened; gets it back when it closes. */
  opener: HTMLElement | null
}

const MARK = 'data-modal-inert'
const BACKDROPS = '.modal-backdrop, .drawer-backdrop'
const TABBABLE =
  'button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])'

const layers: Layer[] = []
/** The last two places focus was (null: nowhere). A dialog that autofocuses a
 *  button has already taken focus by the time its ref runs, and the opener is
 *  the one before. */
let focusTrail: (HTMLElement | null)[] = []

function onFocusIn(e: FocusEvent) {
  if (e.target instanceof HTMLElement) focusTrail = [e.target, ...focusTrail].slice(0, 2)
}

function onFocusOut(e: FocusEvent) {
  if (!e.relatedTarget) focusTrail = [null, ...focusTrail].slice(0, 2)
}

function tabbable(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(TABBABLE)].filter(
    (el) =>
      !(el as HTMLButtonElement).disabled &&
      el.tabIndex >= 0 &&
      el.getClientRects().length > 0 &&
      getComputedStyle(el).visibility !== 'hidden',
  )
}

function onKeyDown(e: KeyboardEvent) {
  const top = layers[layers.length - 1]
  if (!top || e.key !== 'Tab' || e.ctrlKey || e.metaKey || e.altKey) return
  const stops = tabbable(top.el)
  const active = document.activeElement
  if (stops.length === 0) {
    e.preventDefault()
    return
  }
  const first = stops[0]
  const last = stops[stops.length - 1]
  // Wrap when nothing in the layer lies further in the direction of travel.
  // Asked of the document order rather than of the list of stops, because
  // focus may sit on something that is not a stop: the help window focuses its
  // scrolling list.
  let wrap = !active || !top.el.contains(active)
  if (!wrap && active) {
    const side = e.shiftKey ? Node.DOCUMENT_POSITION_PRECEDING : Node.DOCUMENT_POSITION_FOLLOWING
    wrap = !stops.some((s) => s !== active && active.compareDocumentPosition(s) & side)
  }
  if (wrap) {
    e.preventDefault()
    ;(e.shiftKey ? last : first).focus()
  }
}

// From the start, not from the first layer: the trail has to know what held
// focus before the first window opened.
document.addEventListener('focusin', onFocusIn, true)
document.addEventListener('focusout', onFocusOut, true)
document.addEventListener('keydown', onKeyDown, true)

/** Make everything but the topmost layer inert, undoing the previous pass. */
function apply() {
  for (const el of document.querySelectorAll<HTMLElement>(`[${MARK}]`)) {
    el.inert = false
    el.removeAttribute(MARK)
  }
  const top = layers[layers.length - 1]
  if (!top) return
  let node: HTMLElement = top.el
  while (node.parentElement && node !== document.body) {
    for (const sib of node.parentElement.children) {
      if (sib === node || !(sib instanceof HTMLElement)) continue
      // A backdrop stays clickable; one that holds a dialog of its own (a
      // lower layer's) is covered by this layer's backdrop and goes inert.
      if (sib.matches(BACKDROPS) && sib.childElementCount === 0) continue
      if (sib.inert) continue
      sib.inert = true
      sib.setAttribute(MARK, '')
    }
    node = node.parentElement
  }
}

/**
 * Ref callback for the root element of a modal layer: `<div ref={modalLayer}>`.
 * React calls the returned function when the element unmounts.
 */
export function modalLayer(el: HTMLElement | null): (() => void) | undefined {
  if (!el) return undefined
  const active = document.activeElement
  let opener: HTMLElement | null = null
  if (active instanceof HTMLElement && active !== document.body) {
    // Focus already inside means the layer autofocused: the opener is whatever
    // held focus immediately before that, if anything did.
    const before = focusTrail[1] ?? null
    opener = el.contains(active) ? (before && !el.contains(before) ? before : null) : active
  }
  layers.push({ el, opener })
  apply()
  return () => {
    const at = layers.findIndex((l) => l.el === el)
    if (at < 0) return
    const [layer] = layers.splice(at, 1)
    apply()
    // Only when focus was left with the layer (or nowhere): a close that
    // already moved focus somewhere on purpose keeps it there.
    const now = document.activeElement
    const adrift = !now || now === document.body || el.contains(now) || !now.isConnected
    if (adrift && layer.opener?.isConnected) layer.opener.focus()
  }
}
