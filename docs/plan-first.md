# Plan first

Before every task the agent shows a short plan: what it understood, the tool
calls it will make with their key arguments, where the results go and roughly
how long it takes. You answer **Go** or **Change**. The local model can take
minutes over even a simple question, so a misunderstanding is caught before
that time is spent, not after.

Plan first is on by default. Switch it off under **Settings › Plan first**; the
agent then asks for each call as it comes, as before. Switching restarts open
chats.

## The plan card

```
PLAN                                              waiting for you
Count the sessions per subject in /ws/study.
1. Index the scans             index_dataset   dataset_dir: /ws/study
2. Count sessions per subject  summarize_dataset
3. Report the counts
Results go to /ws/results/sessions
[ ✓ Go ]  [ Change ]
```

- **Go** approves exactly the calls the plan lists (see below). The card turns
  into a checklist; each step ticks as its call completes.
- **Change** opens a text box. Your correction goes to the agent as your next
  message ("Change to the plan: …"), the current turn ends, and the agent
  proposes a revised plan. Nothing runs until you say Go.
- Questions the agent needs answered (which scans, which option) are listed on
  the card; answer them with Change.
- A plan nobody answers within an hour lapses; nothing runs.

A reply that needs no tool (a greeting, "what can you do") comes without a plan.

## What Go approves

Go approves, for that plan only:

- a **stack tool call** whose tool a step names, as long as every path argument
  lies in a path the plan shows — one of that step's inputs or outputs, or the
  plan's output folder, or inside one of them. An output argument (`output_dir`,
  `output_path`, …) must lie in the step's outputs or the output folder, never
  in its inputs, so a Go never lets a tool write into the data it was shown
  reading. An argument the step lists by name (an option such as `model: fast`)
  must have the value the card shows. A step may run more than once (a retry,
  the same tool over the scans the step names); the tool and the paths bound
  it, not a count;
- `mkdir -p` of folders inside the plan's output folder.

Never covered by a Go:

- **code the assistant wrote** (a script, `python3 -c`, a heredoc, a `.py` it
  writes): it is always put to you on its own, as before;
- a call that would **overwrite** a file or reach **outside the workspace**: it
  is shown, marked as a step of the plan;
- the workspace root or `/` as a plan path: they cover only themselves, never
  everything beneath them;
- **anything else**. A call outside the plan opens the usual approval dialog,
  marked **Not in the plan**, and its card in the chat says "not in the plan".
  While a plan is in force such a call is shown even if its tool was allowed
  "for the rest of this chat".

The approval lasts until the agent proposes another plan or the chat is closed.
The agent is told to stop and propose a revised plan whenever something outside
the plan becomes necessary (another tool, a script, another output folder).

## How it works

The core provides one tool, `propose_plan`, from its own MCP server
(`medmcp-plan`, the `medmcp-plan-tool` command; it is not a stack and runs
nothing). The agent calls it from `run_typescript` like a stack tool:

```ts
await tools.mcp_medmcp_plan.propose_plan({
  understanding: "Count the sessions per subject in /ws/study.",
  steps: [
    { action: "Index the scans", tool: "index_dataset", inputs: { dataset_dir: "/ws/study" } },
    { action: "Count sessions per subject", tool: "summarize_dataset" },
    { action: "Report the counts" },
  ],
  output_dir: "/ws/results/sessions",
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
approval dialog. Go answers the call "allow", so the tool runs and tells the
agent to carry out exactly those steps; Change answers it "reject" and the
correction follows as a prompt. A plan the card cannot show (a missing field,
too many steps) is let through without a card, and the tool tells the agent
what to fix.

With plan first off, the tool is not offered and the agent's prompt asks for a
`todo` list instead (the prompt `medmcp-noplan`, derived from `medmcp`).

## Provenance

A chat's run log (`run.jsonl`) records:

- `{"event": "plan_proposed", "plan_id", "plan"}` — the plan as shown;
- `{"event": "plan_decision", "plan_id", "decision": "go" | "change", "correction"}`;
- on each tool call decided while plan first is on, `"plan"`:
  `{"plan_id", "step": n}` for a planned call, `{"plan_id", "setup": true}` for
  the output folder's `mkdir`, `{"plan_id", "off_plan": true}` for a call
  outside the plan (`plan_id` is null when no plan had been approved).

The session report lists each plan with its answer and marks every call as a
plan step or **not in the plan**. Distilled workflows leave the plan out.
