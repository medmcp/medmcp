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

- **`confirm_selection`** (cohort stack): confirming a selection is your
  decision about a review you have read, so it always opens its own dialog,
  even when a step of the plan names it. The dialog shows that review, read by
  the core from the `<name>_review.json` beside the file the call names: the
  number of rows per status (ok, only derived, needs a decision, multiple
  stations, missing), up to ten of the rows that need a decision with their
  candidate series, how many decisions the call passes, and the user's reply (`user_reply`) it would
  record. A review that cannot be read (missing, not a review, outside the
  workspace) is said so in the dialog. `confirm_selection` is also never
  allowed "for the rest of this chat";
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

## Calls outside the plan

The calls outside each approved plan are counted. From the **third** on, the
dialog adds "Ask the assistant to revise the plan: deny this and say so in the
chat", and if you allow the call anyway, its result reaches the agent with a
reminder:

> [MedMCP] This was call 3 outside the plan the user approved. Before any
> further step that is not in the plan, call propose_plan with a revised plan
> and wait for the user's Go.

A stack call's result carries it through the core's broker (as text, and as
`medmcp_note` in the structured result unless the tool's output schema forbids
extra keys); the result of one of vibe's own tools (`bash`, `write_file`, …)
through the `medmcp-plan-note` hook, a `post_tool` hook that can only add text
to a result. A denied call carries nothing. A new plan starts the count again.

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
too many steps, `review_selection` and `confirm_selection` in one plan) is let
through without a card, and the tool tells the agent what to fix.

### Plans are checked against the real tools

Before a card is shown, the core checks each step against the tool lists of the
chat's stacks: a step must name a tool a stack offers (or one of vibe's own,
such as `bash`), and every argument it lists must be a parameter of that tool.
A plan that fails is not shown either; the tool hands the problems back with the
parameters the tool does take, e.g.

> step 1 (select_series): select_series has no parameter 'session_label'; its
> parameters are: index_path, sequences, plane, …

The tool lists are the ones the stack pool already holds (each stack's tools as
last listed, kept across eviction and restarts in
`.vibe/state/tool_schemas.json`); nothing is started to check a plan. A tool
name no stack offers is refused only when every stack of the chat has a known
list; otherwise only the steps whose tool is known are checked. Without the
stack pool (`MEDMCP_STACK_POOL=0`) no plan is checked. The plan tool learns why
the core refused a plan from the broker (`check_plan`, keyed by the plan's
arguments); with no broker to ask it lets the plan through, as before.

### Calls are checked against the real tools too

The same tool lists catch a call whose arguments the tool's schema rejects
(rehearsal D7: `select_series` with `sequence: [["T2", "STIR"]]`). vibe 2.26
validates the arguments a `pre_tool` hook lets through once more, and an invalid
set there does not fail the call: the harness rejects it as an invalid command
and the whole turn ends with `Core rejected durable input`. (The sandbox's own
check before the hook lets an invalid call through when the schema uses the
`$defs`/`$ref` pydantic writes for a nested argument, so such a call reaches the
hook.) A denial is the one hook answer that
skips that check, so the path guard (`medmcp-pathguard`) validates every stack
call against the schema the pool holds (the broker's `tool_schema` op) and
denies an invalid one:

> select_series arguments are invalid: criteria.sequence[0]: ['T2', 'STIR'] is
> not of type 'string'. Expected: criteria.sequence takes string or array of
> string.

This denial is not capped like a path denial, since letting the call through
ends the turn. A tool with no known schema, no broker to ask, or any error while
checking allows the call as before. Under vibe 2.26.1 a call made from
`run_typescript` gets only `tool_skipped: Tool execution was skipped by Runtime
policy.` back from a denial, not its reason (the chat shows the reason).

## Truncated tool results

A tool result too large for the context reaches the model truncated, with the
full result saved to a file and the harness's advice to inspect it with `jq`.
The model then parsed saved results with Python one-liners. The core's LLM shim
(local models) replaces that advice with "[MedMCP] Result truncated — call the
tool again with narrower arguments; do not parse files with shell", and the
prompt says the same for every model.

With plan first off, the tool is not offered and the agent's prompt asks for a
`todo` list instead (the prompt `medmcp-noplan`, derived from `medmcp`).

## Provenance

A chat's run log (`run.jsonl`) records:

- `{"event": "plan_proposed", "plan_id", "plan"}` — the plan as shown;
- `{"event": "plan_decision", "plan_id", "decision": "go" | "change", "correction"}`;
- on each tool call decided while plan first is on, `"plan"`:
  `{"plan_id", "step": n}` for a planned call, `{"plan_id", "setup": true}` for
  the output folder's `mkdir`, `{"plan_id", "off_plan": true, "off_plan_n": k}`
  for the k-th call outside the plan (`plan_id` is null, and `off_plan_n`
  absent, when no plan had been approved).

The session report lists each plan with its answer and the number of calls made
outside it ("Calls outside this plan: 3", and the total over all plans), and
marks every call as a plan step or **not in the plan** (call k outside it).
Distilled workflows leave the plan out.
