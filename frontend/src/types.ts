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
}

/** One prior chat session, as served by GET /api/sessions. */
export interface SessionInfo {
  id: string
  title: string | null
  updatedAt: string | null
  archived: boolean
  hasProvenance: boolean
}

/** One stack/workflow row plus the feature toggles, as served by /api/settings. */
export interface StackInfo {
  name: string
  version?: string | null
  active: boolean
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

/** Result of POST /api/workflows/{name}/batch-from-plan: rows to pre-fill the
 *  batch editor from a plan_batch manifest, plus the flagged rows to resolve. */
export interface BatchFromPlanResult {
  ok: boolean
  error: string | null
  runs: Record<string, string>[]
  skipped: BatchPlanSkip[]
  column_map?: Record<string, string>
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
export type ChatItem =
  // messageId (replayed turns only) anchors per-turn actions like rewind.
  | { kind: 'user'; text: string; messageId?: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'tool'; toolCallId: string }
  | { kind: 'error'; text: string }
  // A non-error note from the server (e.g. the turn stopped at the step limit).
  | { kind: 'notice'; text: string }

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
