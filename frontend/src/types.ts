/** Shared types for the workspace UI and its wire protocols. */

/** dataTransfer MIME carrying a workspace-relative path when dragging a file. */
export const DRAG_PATH_MIME = 'application/medmcp-path'

/** One node of the explorer tree (mirrors /api/tree). Directories have `children`. */
export interface TreeNode {
  id: string
  name: string
  children?: TreeNode[]
  size?: number
}

/** Tool-call state accumulated from tool_call / tool_call_update frames. */
export interface ToolCallState {
  toolCallId: string
  title: string
  status: string
  kind?: string | null
  /** The underlying tool's name (ACP's `title` is prose, so it cannot identify
   *  a tool). Lets a tool with a better rendering than a generic card get one. */
  toolName?: string
  rawInput?: unknown
  output?: string | null
  /** The path guard turned this call back before it ran: the agent corrects the
   *  path and retries, so it renders as a quiet note rather than a failure. */
  pathGuardRetry?: boolean
  /** Plan first: the call was not in the approved plan (it asked, flagged). */
  offPlan?: boolean
}

/** A risk tag resolved by the server from the fixed taxonomy. */
export interface RiskTag {
  key: string
  label: string
  severity: 'low' | 'medium' | 'high'
}

/** One path argument of a pending tool call, checked against the filesystem.
 *  Deterministic (no LLM), so unlike `explanation` it arrives with the request. */
export interface PathFinding {
  param: string
  value: string
  role: 'input' | 'output' | 'unknown'
  status:
    | 'ok'
    | 'missing'
    | 'parent_missing'
    | 'will_overwrite'
    | 'outside_workspace'
    | 'unreadable'
  severity: 'error' | 'warning' | 'info'
  note: string
  /** For a missing path: nearest existing folder ('.' is the workspace root),
   *  a capped sample of what it holds, and how many entries there are. Seeing
   *  the neighbours is what makes a missing path actionable. Empty otherwise. */
  nearest: string
  entries: string[]
  entry_total: number
}

/** How a pending call relates to the plan (plan first, see src/medmcp/plan.py). */
export interface PlanNote {
  /** The call is one of the approved plan's steps (shown anyway: it needs a look). */
  inPlan: boolean
  planId: string | null
  /** Zero-based step index when `inPlan`. */
  step: number | null
  /** Why it is flagged, e.g. "Not in the plan you approved." */
  reason: string
  /** For a call outside an approved plan: its number among them (from 1). */
  offPlanCount?: number
  /** Enough calls have left the plan that asking for a revised plan is suggested. */
  revise?: boolean
}

/** One row of a selection review that still needs a decision. */
export interface ReviewRow {
  subject: string
  session: string
  role: string
  status: string
  reason: string
  candidates: string[]
  moreCandidates: number
}

/** The review a confirm_selection call would confirm (src/medmcp/reviewsummary.py). */
export interface SelectionReview {
  path?: string
  /** Rows per status (ok, only_derived, needs_decision, multiple_stations, missing). */
  counts?: Record<string, number>
  /** Up to 10 rows that need a decision, of `rowsTotal`. */
  rows?: ReviewRow[]
  rowsTotal?: number
  /** Decisions (overrides) the call passes. */
  overrides?: number
  /** The note the call would record. */
  note?: string
  /** Why the review could not be read; the rest is empty then. */
  error?: string | null
}

/** One step of a proposed plan. */
export interface PlanStep {
  action: string
  tool?: string
  inputs?: Record<string, unknown>
  outputs?: string[]
}

/** A plan the agent proposed: what it understood and the calls it will make.
 *  `id` is the plan tool call that proposed it. */
export interface PlanProposal {
  id: string
  understanding: string
  steps: PlanStep[]
  output_dir?: string
  estimated_time?: string
  questions?: string[]
}

/** Where a plan stands: waiting for Go/Change, approved, corrected, or neither. */
export type PlanDecisionState = 'pending' | 'approved' | 'changed' | 'cancelled'

/** One step's progress after Go, as its tool call moves on. */
export type PlanStepStatus = 'running' | 'completed' | 'failed'

/** A permission request awaiting the user's decision. */
export interface PermissionRequest {
  requestId: number
  toolCall: {
    toolCallId?: string
    title?: string
    rawInput?: unknown
    [key: string]: unknown
  }
  options: { optionId: string; name?: string; kind?: string }[]
  explanation?: string | null
  /** True while the server is still generating the LLM explanation. */
  explaining?: boolean
  risks?: RiskTag[]
  /** Existence check on the call's path arguments; empty when it takes none. */
  paths?: PathFinding[]
  /** Plan first: whether this call is in the approved plan (null when off). */
  plan?: PlanNote | null
  /** confirm_selection: the review it would confirm. */
  review?: SelectionReview | null
}

/** One prior chat session, as served by GET /api/sessions. */
export interface SessionInfo {
  id: string
  title: string | null
  updatedAt: string | null
  archived: boolean
  hasProvenance: boolean
}

/** One installed stack, as served by /api/settings. */
export interface StackInfo {
  name: string
  version?: string | null
  active: boolean
}

/** A background job (from GET /api/jobs): one approved stack call running in
 *  its own container, which outlives the chat that started it. */
export interface JobInfo {
  id: string
  stack: string
  tool: string
  /** A short name the stack gave the job (may be empty). */
  name: string
  status: 'running' | 'finished' | 'stopped' | 'failed'
  started_at: string | null
  finished_at?: string | null
  /** Id of the chat that started it (may be empty). */
  session?: string
  /** Name of that chat (empty while it has none). */
  session_title?: string
  /** What the job last reported about itself (running jobs only). */
  progress?: { fraction?: number; text?: string; updated_at?: number } | null
}

/** A container-installed stack (from GET /api/stacks); uninstallable via the UI. */
export interface InstalledStack {
  name: string
  image: string
  gpu: boolean
  /** Whether this stack runs with network egress. The workspace is mounted into
   *  every stack, so this is the difference between a stack that can read your
   *  data and one that can also send it somewhere. */
  network: boolean
}

/** One catalog entry (from GET /api/catalog): an installable stack. */
export interface CatalogEntry {
  name: string
  image: string
  description: string
  gpu: boolean
  /** Declared egress — known before the pull, so the warning precedes the install. */
  network: boolean
  installed: boolean
}

export interface SettingsState {
  explain_tools: boolean
  record_provenance: boolean
  /** The agent proposes a plan (Go / Change) before each task. */
  plan_first: boolean
  /** Selected GPU (CDI device id) for container stacks; "all" = every GPU. */
  gpu: string
  /** GPU the LLM container was created with (deploy-time; read-only here). */
  llm_gpu: string
  stacks: StackInfo[]
}

/** One GPU from GET /api/gpus (best-effort enumeration). */
export interface GpuInfo {
  index: string
  uuid: string
  name: string
}

/** One workflow row from GET /api/workflows. */
export interface WorkflowListEntry {
  name: string
  description: string
}

/** A stack the workflow needs, pinned for reproducibility. */
export interface StackRequirement {
  stack: string
  version?: string
  image?: string
  digest?: string
  /** Availability of this stack in the current environment (server-computed). */
  status?: 'ok' | 'missing' | 'mismatch'
  /** Digest of the locally-present image, when it differs from the pinned one. */
  installed_digest?: string
}

/** Full recipe detail from GET /api/workflows/{name}. */
export interface WorkflowDetail {
  name: string
  description: string
  inputs: { name: string; example: string; description: string; default?: string }[]
  steps: { server: string; tool: string; arguments: Record<string, unknown> }[]
  requires: StackRequirement[]
  manual_steps: string[]
  replayable: boolean
  replay_error: string | null
}

/** A flagged manifest row from plan_batch that isn't ready to run. */
export interface BatchPlanSkip {
  subject?: string
  session?: string
  status?: string
  reason?: string
}

/** What confirm_selection recorded about a confirmed selection (its
 *  `<name>_confirmed.json`), as checked against the CSV being rolled out. */
export interface SelectionConfirmation {
  path: string
  sha256: string
  confirmed_csv_sha256: string
  confirmed_at: string | null
  note: string
  rule_version: string | null
  roles: string[]
  overrides: number
  excluded: number
}

/** Result of POST /api/workflows/{name}/batch-from-plan: rows to pre-fill the
 *  batch editor from a plan_batch manifest, plus the flagged rows to resolve. */
export interface BatchFromPlanResult {
  ok: boolean
  error: string | null
  /** The manifest cannot be rolled out as it is (`error` says why and what to
   *  do): a confirmed selection changed since confirmation, or a selection that
   *  was never confirmed under a workflow that reads several scans. */
  refused?: boolean
  /** For a confirmed selection: its confirmation record. */
  confirmation?: SelectionConfirmation | null
  /** Which manifest it was: a plan_batch plan or a select_series selection. */
  kind?: 'plan' | 'selection'
  runs: Record<string, string>[]
  /** subject[/session] per run, aligned with `runs`. */
  labels?: string[]
  skipped: BatchPlanSkip[]
  column_map?: Record<string, string>
  /** A plan's file columns, to choose from when an input could not be matched. */
  columns?: string[]
  /** A selection's series roles (FLAIR, T1+contrast, …) with how many there are. */
  roles?: { role: string; series: number; units: number }[]
  role_map?: Record<string, string>
  /** Inputs that still need a column (plan) or a role (selection). */
  unmapped?: string[]
  /** The folder each item's results go under, one sub-folder per subject. */
  output_root?: string
}

/** One resolved step from POST /api/workflows/{name}/replay-preview. */
export interface ReplayPreviewStep {
  index: number
  server: string
  tool: string
  arguments: Record<string, unknown>
}

/** Pre-flight verdict on one run item, before anything is executed. */
export interface ReplayPreviewItem {
  index: number
  ok: boolean
  error: string | null
  findings: PathFinding[]
}

/** Result of POST /api/workflows/{name}/replay-preview. */
export interface ReplayPreviewResult {
  ok: boolean
  error: string | null
  /** The resolved steps of the first item that can run. */
  steps: ReplayPreviewStep[]
  items: ReplayPreviewItem[]
}

export type RunStatus = 'running' | 'done' | 'failed' | 'cancelled'

/** One row from GET /api/runs. */
export interface RunSummary {
  id: string
  workflow: string
  status: RunStatus
  started_at: string
  finished_at: string
  error: string | null
  total: number
  succeeded: number
  failed: number
  steps_per_item: number
}

/** Frames the server sends over /ws/replay. */
export type ReplayFrame =
  | {
      type: 'started'
      run_id: string
      workflow: string
      total: number
      steps_per_item: number
      started_at: string
      runs: Record<string, string>[]
      /** Who each item is (subject/session), "" where the rows were picked by hand. */
      labels?: string[]
    }
  | {
      type: 'step_started'
      item: number
      index: number
      server: string
      tool: string
      /** When the tool was called, so a late attach shows the real elapsed time. */
      started_at?: string
    }
  | {
      type: 'step'
      /** Batch item index this step belongs to (0 for single runs). */
      item?: number
      index: number
      server: string
      tool: string
      ok: boolean
      error?: string | null
      produced: Record<string, string>
      started_at?: string
      finished_at?: string
    }
  | { type: 'item_result'; item: number; ok: boolean; error?: string | null; outputs: string[] }
  | {
      type: 'result'
      ok: boolean
      error?: string | null
      outputs?: string[]
      status?: RunStatus
      finished_at?: string
    }

/** One task in the agent's plan, from the `todo` tool's arguments. */
export interface TodoItem {
  id?: string
  content: string
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled'
  priority?: string
}

/** Ordered chat transcript entries. Tool calls render as inline cards. */
// `id` is the React key: rows are removed from the middle of the transcript
// (a superseded plan card, a rewind), so an index key would remount — and
// re-parse the markdown of — every row after the gap.
export type ChatItem = { id: number } & (
  // messageId (replayed turns only) anchors per-turn actions like rewind.
  | { kind: 'user'; text: string; messageId?: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'tool'; toolCallId: string }
  | { kind: 'error'; text: string }
  // A non-error note from the server (e.g. the turn stopped at the step limit).
  | { kind: 'notice'; text: string }
)

/** What a rewind would restore (preview) / did restore (perform). */
export interface RewindResult {
  paths?: string[]
  messageContent?: string
  restoredPaths?: string[]
  restoreErrors?: string[]
}

/** Frames the server sends over /ws/chat. */
export type ServerFrame =
  // `cloud`: the model answering is hosted outside this machine.
  | { type: 'ready'; sessionId: string; model?: string; cloud?: boolean; title?: string | null }
  | { type: 'chunk'; text: string }
  /** A generated chat title landed (a user-set name is never overwritten). */
  | { type: 'title'; title: string }
  /** vibe is backing off and retrying the model backend. */
  | { type: 'retrying'; category: string; detail: string }
  | { type: 'notice'; text: string }
  // A user turn: replayed from a resumed session (session/load), or vibe's
  // echo of a live prompt (merged into the locally-rendered bubble by id).
  | { type: 'user'; text: string; messageId?: string }
  | {
      type: 'tool_call'
      toolCallId: string
      title: string
      status: string
      kind?: string | null
      toolName?: string
      rawInput?: unknown
      /** A settled call replayed on resume carries its output here. */
      output?: string | null
      pathGuardRetry?: boolean
    }
  | {
      type: 'tool_call_update'
      toolCallId: string
      status?: string | null
      output?: string | null
      /** Completed arguments, re-sent once the model finished streaming them. */
      rawInput?: unknown
      /** Server-tagged: the path guard turned the call back before it ran. */
      pathGuardRetry?: boolean
    }
  | { type: 'usage'; used: number; size?: number }
  | {
      type: 'permission_request'
      requestId: number
      toolCall: PermissionRequest['toolCall']
      options: PermissionRequest['options']
      explanation?: string | null
      explaining?: boolean
      risks?: RiskTag[]
      paths?: PathFinding[]
      plan?: PlanNote | null
      review?: SelectionReview | null
    }
  /** Plan first: a proposed plan waiting for Go or Change. */
  | { type: 'plan_request'; requestId: number; toolCallId: string; plan: PlanProposal }
  | { type: 'plan_status'; planId: string; status: 'approved' | 'changed' | 'cancelled' }
  | {
      type: 'plan_progress'
      planId: string
      step: number
      toolCallId: string
      status: PlanStepStatus
    }
  | {
      type: 'permission_update'
      requestId: number
      explanation?: string | null
      risks?: RiskTag[]
    }
  | { type: 'done' }
  | { type: 'error'; message: string }

/** One external MCP server (GET /api/external-mcp). */
export interface ExternalServer {
  name: string
  transport: string
  url: string
  /** Name of the env var holding the token — never the token itself. */
  api_key_env: string
  /** Header to carry the token; empty means Authorization. */
  api_key_header: string
  /** Value format, e.g. "Bearer {token}"; empty means Bearer. */
  api_key_format: string
  active: boolean
  /** Whether `api_key_env` actually holds a token where the agent runs. Presence
   *  only — the value never leaves the server. */
  token_present?: boolean
  /** Whether the token is stored by MedMCP rather than named as a deployment
   *  variable. Managed tokens are present by construction. */
  token_managed?: boolean
}

/** State of the external-MCP feature (advanced settings). */
export interface ExternalMcpState {
  enabled: boolean
  /** Whether the operator has accepted responsibility; required before enabling. */
  acknowledged: boolean
  acknowledged_at: string | null
  /** Transports the server will accept, in preference order. */
  transports: string[]
  servers: ExternalServer[]
}

/** One model of the local catalog (GET /api/models). */
export interface LocalModelRow {
  id: string
  label: string
  vendor: string
  /** The tag in the Ollama library. */
  tag: string
  params: string
  size_gb: number
  license: string
  /** Not Apache-2.0: its terms are accepted before the download starts. */
  license_ack: boolean
  default: boolean
  /** The local model in use. */
  active: boolean
  downloaded: boolean
  deletable: boolean
  /** Context length the model was prepared with; known for a selected model. */
  num_ctx: number | null
}

/** The local model catalog and its state. */
export interface LocalModelsState {
  models: LocalModelRow[]
  active: string
  /** False when the model server did not answer; download state is then unknown. */
  reachable: boolean
}

/** A provider the cloud-model form offers (GET /api/cloud-model). */
export interface CloudProviderPreset {
  id: string
  label: string
  /** Fixed endpoint; empty means the operator supplies one. */
  api_base: string
  /** Suggested model ids — not an allowlist. */
  models: string[]
  compact_threshold: number
  key_required: boolean
}

/** The configured cloud model. Never carries the key. */
export interface CloudModelConfig {
  provider: string
  model: string
  api_base: string
  /** Name of the env var holding the key — never the key itself. */
  api_key_env: string
  /** Tokens of history before the agent compacts; also the context meter's size. */
  compact_threshold: number
  /** Whether the key is stored by MedMCP rather than named as a deployment variable. */
  key_managed: boolean
  /** Whether the key is actually available where the agent runs. Presence only. */
  key_present: boolean
}

/** State of the cloud-model feature (advanced settings). */
export interface CloudModelState {
  enabled: boolean
  /** Whether the operator has accepted responsibility; required before enabling. */
  acknowledged: boolean
  acknowledged_at: string | null
  /** Whether chats run on the cloud model right now. False while `enabled` when
   *  the key is missing: the workspace then stays on the local model. */
  active: boolean
  local_model: string
  model: CloudModelConfig | null
  providers: CloudProviderPreset[]
}

/** The newest release the server knows about (GET /api/update). */
export interface UpdateRelease {
  version: string
  tag: string
  /** Release notes as markdown (the CHANGELOG section). */
  notes: string
  url: string
  published_at: string
}

/** Outcome of the last update the helper ran, shown once after it. */
export interface UpdateResult {
  from: string
  to: string
  status: 'ok' | 'rolled_back' | 'failed'
  detail: string
  at: string
}

export interface UpdateState {
  current: { version: string; build: string }
  /** False when MEDMCP_UPDATE_CHECK=0: nothing is ever fetched, not even by hand. */
  enabled: boolean
  /** The daily unattended check (the operator's switch in Settings › Advanced). */
  auto_check: boolean
  checked_at: string | null
  error: string | null
  latest: UpdateRelease | null
  /** A newer release than the one running exists. */
  available: boolean
  /** The header notice for it was dismissed (Settings still shows it). */
  dismissed: boolean
  /** The UI can apply it here; otherwise `apply_reason` says what to do instead. */
  can_apply: boolean
  apply_reason: string | null
  host_commands: { update: string; rollback: string } | null
  last_result: UpdateResult | null
  /** A rehearsal is set up: the release is a stand-in and the update installs nothing. */
  rehearsal: boolean
}
