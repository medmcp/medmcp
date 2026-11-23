/** Whether a transcript keeps following new output, decided from scroll events. */

/** Slack below the viewport that still counts as "at the bottom", in px. */
export const BOTTOM_TOLERANCE = 8

export interface ScrollMetrics {
  scrollTop: number
  scrollHeight: number
  clientHeight: number
}

/** The follow decision, with the scroll position it was last made at. */
export interface FollowState {
  following: boolean
  scrollTop: number
  scrollHeight: number
}

export function isAtBottom(m: ScrollMetrics, tolerance = BOTTOM_TOLERANCE): boolean {
  return m.scrollHeight - m.scrollTop - m.clientHeight <= tolerance
}

/**
 * The follow state after one scroll event.
 *
 * Reaching the bottom attaches. Moving up while away from it detaches, but only
 * when the content is as tall as it was last seen: a move that comes with a
 * change of height is the browser re-laying out rows (clamping the position,
 * anchoring it to a row that changed size), not the reader. Moving down without
 * arriving changes nothing.
 */
export function afterScroll(prev: FollowState, m: ScrollMetrics): FollowState {
  let following = prev.following
  if (isAtBottom(m)) following = true
  else if (m.scrollTop < prev.scrollTop && m.scrollHeight === prev.scrollHeight) following = false
  return { following, scrollTop: m.scrollTop, scrollHeight: m.scrollHeight }
}
