import type { PlanProposal } from './types'

/** The core's plan tool (plan first, see src/medmcp/plan.py). */
export const PLAN_TOOL = 'medmcp-plan_propose_plan'

/** Read a proposed plan out of the plan tool's arguments (object or JSON string).
 *  Returns null when the arguments are not a plan the card can show — the core
 *  then let the call through, and the tool told the agent what to fix. */
export function parseProposal(id: string, raw: unknown): PlanProposal | null {
  let value = raw
  if (typeof value === 'string') {
    try {
      value = JSON.parse(value)
    } catch {
      return null
    }
  }
  if (value === null || typeof value !== 'object') return null
  const v = value as Record<string, unknown>
  if (typeof v.understanding !== 'string' || !Array.isArray(v.steps) || v.steps.length === 0) {
    return null
  }
  const steps = (v.steps as unknown[]).map((s) => {
    if (typeof s === 'string') return { action: s }
    const o = (s ?? {}) as Record<string, unknown>
    const tool = typeof o.tool === 'string' ? o.tool : undefined
    return {
      action: typeof o.action === 'string' && o.action ? o.action : (tool ?? ''),
      tool,
      inputs:
        o.inputs && typeof o.inputs === 'object' ? (o.inputs as Record<string, unknown>) : undefined,
      outputs: Array.isArray(o.outputs)
        ? (o.outputs as unknown[]).filter((x): x is string => typeof x === 'string')
        : undefined,
    }
  })
  return {
    id,
    understanding: v.understanding,
    steps,
    output_dir: typeof v.output_dir === 'string' ? v.output_dir : undefined,
    estimated_time: typeof v.estimated_time === 'string' ? v.estimated_time : undefined,
    questions: Array.isArray(v.questions)
      ? (v.questions as unknown[]).filter((x): x is string => typeof x === 'string')
      : undefined,
  }
}
