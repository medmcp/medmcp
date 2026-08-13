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
}

/** One catalog entry (from GET /api/catalog): an installable stack. */
export interface CatalogEntry {
  name: string
  image: string
  description: string
  gpu: boolean
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
  kind: 'active' | 'draft'
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
  kind: 'active' | 'draft'
  description: string
  inputs: { name: string; example: string; description: string }[]
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

/** Frames the server sends over /ws/replay. */
export type ReplayFrame =
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
    }
  | { type: 'item_result'; item: number; ok: boolean; error?: string | null; outputs: string[] }
  | { type: 'result'; ok: boolean; error?: string | null; outputs?: string[] }

/** Ordered chat transcript entries. Tool calls render as inline cards. */
export type ChatItem =
  // messageId (replayed turns only) anchors per-turn actions like rewind.
  | { kind: 'user'; text: string; messageId?: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'tool'; toolCallId: string }
  | { kind: 'error'; text: string }

/** What a rewind would restore (preview) / did restore (perform). */
export interface RewindResult {
  paths?: string[]
  messageContent?: string
  restoredPaths?: string[]
  restoreErrors?: string[]
}

/** Frames the server sends over /ws/chat. */
export type ServerFrame =
  | { type: 'ready'; sessionId: string; model?: string }
  | { type: 'chunk'; text: string }
  // A user turn: replayed from a resumed session (session/load), or vibe's
  // echo of a live prompt (merged into the locally-rendered bubble by id).
  | { type: 'user'; text: string; messageId?: string }
  | {
      type: 'tool_call'
      toolCallId: string
      title: string
      status: string
      kind?: string | null
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
