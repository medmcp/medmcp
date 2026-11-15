# Plan first

Plan first is an optional setting, **off by default**. With it off, nothing
changes: the plan tool is not offered, the prompt is the usual one and every call
is approved as it comes.

Switched on (**Settings › Plan first**; switching restarts open chats), the agent
shows a short plan before a task: what it understood, the tool calls it will make
with their key arguments, where the results go and roughly how long it takes. You
answer **Go** or **Change**. The local model can take minutes over even a simple
question, so a misunderstanding is caught before that time is spent, and you can
say no instead of waiting for a run you did not want.

## The plan card

```
PLAN                                              waiting for you
Count the files per subject in /ws/study.
1. Scan the folder          scan_folder   folder: /ws/study
2. Count per subject        count_files
3. Report the counts
Results go to /ws/results/counts
[ ✓ Go ]  [ Change ]
```

(The tool names come from the stacks you have installed; the core names none.)

- **Go** approves exactly the calls the plan lists (see below). The card turns
  into a checklist; each step ticks as its call completes.
- **Change** opens a text box. Your correction goes to the agent as your next
  message ("Change to the plan: …"), the current turn ends, and the agent
  proposes a revised plan. Nothing runs until you say Go.
- Questions the agent needs answered are listed on the card; answer them with
  Change.
- A plan nobody answers within an hour lapses; nothing runs.

A reply that needs no tool (a greeting, "what can you do") comes without a plan.

## What Go approves

Go approves, for that plan only, a **stack tool call** whose tool a step names,
as long as every path argument lies in a path the plan shows — one of that step's
inputs or outputs, or the plan's output folder, or inside one of them. An output
argument (`output_dir`, `output_path`, …) must lie in the step's outputs or the
output folder, never in its inputs, so a Go never lets a tool write into the data
it was shown reading. An argument the step lists by name (an option such as
`model: fast`) must have the value the card shows. A step may run more than once
(a retry, the same tool over several inputs it names); the tool and the paths
bound it, not a count.

Never covered by a Go:

- **vibe's own tools** — the shell (even `mkdir`), file writes and edits — and
  **code the assistant wrote**: they ask on their own, as before;
- a call that would **overwrite** a file or reach **outside the workspace**: it
  is shown, marked as a step of the plan;
- the workspace root or `/` as a plan path: they cover only themselves, never
  everything beneath them;
- **anything else**. A call outside the plan opens the usual approval dialog,
  marked **Not in the plan**, and its card in the chat says "not in the plan".
  While a plan is in force such a call is shown even if its tool was allowed
  "for the rest of this chat".

The approval lasts until the agent proposes another plan or the chat is closed.

## "Approved" only on the core's Go

The core provides one tool, `propose_plan`, from its own MCP server
(`medmcp-plan`, the `medmcp-plan-tool` command; it is not a stack and runs
nothing). The agent calls it like a stack tool:

```ts
await tools.mcp_medmcp_plan.propose_plan({
  understanding: "Count the files per subject in /ws/study.",
  steps: [
    { action: "Scan the folder", tool: "scan_folder", inputs: { folder: "/ws/study" } },
    { action: "Count per subject", tool: "count_files" },
    { action: "Report the counts" },
  ],
  output_dir: "/ws/results/counts",
  estimated_time: "about 1 minute",
})
```

| Field | Required | Meaning |
| --- | --- | --- |
| `understanding` | yes | one or two sentences: what the agent understood |
| `steps` | yes | 1–12 steps; each `{action, tool?, inputs?, outputs?}` |
| `output_dir` | no | absolute folder the results go into |
| `estimated_time` | no | e.g. "about 2 minutes" |
| `questions` | no | up to 5 things for you to decide |

The core holds the call at its approval step and shows the card instead of the
approval dialog. A plan the card cannot show (a missing field, too many steps) is
let through without a card, and the tool tells the agent what to fix.

The plan tool says "The user approved this plan (Go)" only when the core recorded
a Go for exactly this plan's card. For each plan it decides the core keeps a
verdict — Go, or refused with the problems — keyed by the plan as validated, and
the tool takes it from the core's broker (`plan_verdict`, once). The plan comes
into force when the tool takes the Go, not before, so a plan the agent was not
told is approved never governs a call. Anything else — no verdict, a broker out
of reach, a plan allowed in an ordinary dialog — comes back as "not approved:
propose again". The plan tool is never covered by "allow for the rest of this
chat".

The tool reaches the broker through `MEDMCP_BROKER_SOCK`, set on its
`[[mcp_servers]]` entry: vibe starts an MCP server with only the variables its
entry lists. Without the stack pool, the core starts a broker for its own ops
only, and only while plan first is on.

## The prompt

The shipped prompt is unchanged. With plan first on, the core derives a twin of
the prompt in force (`medmcp-plan`, `medmcp-external-plan`, …) whose
"Planning multi-step work" (todo list) section is replaced by the plan-first
rules. The rules name no stack and no stack tool.

## Provenance

A chat's run log (`run.jsonl`) records:

- `{"event": "plan_proposed", "plan_id", "plan"}` — the plan as shown;
- `{"event": "plan_decision", "plan_id", "decision": "go" | "change", "correction"}`;
- on each tool call decided while plan first is on, `"plan"`:
  `{"plan_id", "step": n}` for a planned call, `{"plan_id", "off_plan": true}`
  for a call outside the plan (`plan_id` null when no plan had been approved).

The session report lists each plan with its answer and marks every call as a plan
step or **not in the plan**. Distilled workflows leave the plan out.
