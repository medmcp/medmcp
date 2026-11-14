import { memo, useState } from 'react'
import type { PlanDecisionState, PlanProposal, PlanStepStatus } from '../types'

const STEP_MARK: Record<PlanStepStatus | 'pending', string> = {
  completed: '✓',
  running: '▸',
  failed: '✕',
  pending: '·',
}

const STEP_CLASS: Record<PlanStepStatus | 'pending', string> = {
  completed: 'plan-completed',
  running: 'plan-in_progress',
  failed: 'plan-failed',
  pending: 'plan-pending',
}

/** One step's key arguments, compactly: `name: value` pairs on one line. */
function formatInputs(inputs: Record<string, unknown>): string {
  return Object.entries(inputs)
    .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(' · ')
}

/**
 * The agent's plan, shown before it does anything. While it waits for an
 * answer it offers Go (approve exactly these steps) and Change (a correction
 * that goes back to the agent as the doctor's next message). After Go it is
 * the checklist of the run: each step ticks as its tool call completes.
 */
export const PlanProposalCard = memo(function PlanProposalCard({
  plan,
  state,
  steps,
  canDecide,
  idle,
  onGo,
  onChange,
}: {
  plan: PlanProposal
  state: PlanDecisionState
  /** Per-step progress after Go (index → status). */
  steps: Record<number, PlanStepStatus>
  /** False while the socket is down: an answer could not reach the agent. */
  canDecide: boolean
  /** The agent's turn is over (nothing streaming). */
  idle: boolean
  onGo: () => void
  onChange: (text: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState('')
  const toolSteps = plan.steps.map((s, i) => (s.tool ? i : -1)).filter((i) => i >= 0)
  // A step without a tool (e.g. "summarise") has nothing to tick on; it counts as
  // done once every tool step is and the agent has finished its turn.
  const allToolsDone = toolSteps.length > 0 && toolSteps.every((i) => steps[i] === 'completed')
  const status = (i: number): PlanStepStatus | 'pending' => {
    if (steps[i]) return steps[i]
    if (state === 'approved' && !plan.steps[i].tool && allToolsDone && idle) return 'completed'
    return 'pending'
  }
  const done = plan.steps.filter((_, i) => status(i) === 'completed').length
  const send = () => {
    const t = text.trim()
    if (!t) return
    onChange(t)
    setEditing(false)
    setText('')
  }
  return (
    <div className={`plan-card plan-proposal plan-state-${state}`}>
      <div className="plan-head">
        <span className="plan-title">Plan</span>
        {state === 'approved' ? (
          <span className="plan-count">
            {done}/{plan.steps.length}
          </span>
        ) : (
          plan.estimated_time && <span className="plan-count">{plan.estimated_time}</span>
        )}
        <span className={`plan-badge plan-badge-${state}`}>
          {state === 'pending'
            ? 'waiting for you'
            : state === 'approved'
              ? 'approved'
              : state === 'changed'
                ? 'change requested'
                : 'not carried out'}
        </span>
      </div>
      <div className="plan-understanding">{plan.understanding}</div>
      <ol className="plan-list plan-steps">
        {plan.steps.map((s, i) => {
          const st = state === 'approved' ? status(i) : 'pending'
          return (
            <li key={i} className={`plan-item ${STEP_CLASS[st]}`}>
              <span className="plan-mark" aria-hidden="true">
                {state === 'approved' ? STEP_MARK[st] : `${i + 1}.`}
              </span>
              <span className="plan-text">
                {s.action}
                {s.tool && <code className="plan-tool">{s.tool}</code>}
                {s.inputs && Object.keys(s.inputs).length > 0 && (
                  <span className="plan-io">{formatInputs(s.inputs)}</span>
                )}
                {s.outputs && s.outputs.length > 0 && (
                  <span className="plan-io">→ {s.outputs.join(', ')}</span>
                )}
              </span>
            </li>
          )
        })}
      </ol>
      {plan.output_dir && (
        <div className="plan-output">
          Results go to <code>{plan.output_dir}</code>
        </div>
      )}
      {plan.questions && plan.questions.length > 0 && (
        <ul className="plan-questions">
          {plan.questions.map((q, i) => (
            <li key={i}>{q}</li>
          ))}
        </ul>
      )}
      {state === 'pending' &&
        (editing ? (
          <div className="plan-change">
            <textarea
              autoFocus
              rows={2}
              value={text}
              placeholder="What should the agent do differently?"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
            />
            <div className="approval-btns">
              <button className="abtn-approve" disabled={!canDecide || !text.trim()} onClick={send}>
                Send change
              </button>
              <button className="btn-plain" onClick={() => setEditing(false)}>
                Back
              </button>
            </div>
          </div>
        ) : (
          <div className="approval-btns">
            <button className="abtn-approve plan-go" disabled={!canDecide} onClick={onGo}>
              ✓ Go
            </button>
            <button className="abtn-reject" disabled={!canDecide} onClick={() => setEditing(true)}>
              Change
            </button>
          </div>
        ))}
    </div>
  )
})
