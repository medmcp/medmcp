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
}

/** A risk tag resolved by the server from the fixed taxonomy. */
export interface RiskTag {
  key: string
  label: string
  severity: 'low' | 'medium' | 'high'
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

export interface WorkflowInfo {
  name: string
  description: string
  kind: 'active' | 'draft'
  active: boolean
}

export interface SettingsState {
  explain_tools: boolean
  record_provenance: boolean
  workflows_enabled: boolean
  stacks: StackInfo[]
  workflows: WorkflowInfo[]
}

/** One workflow row from GET /api/workflows. */
export interface WorkflowListEntry {
  name: string
  description: string
  kind: 'active' | 'draft'
}

/** Full recipe detail from GET /api/workflows/{name}. */
export interface WorkflowDetail {
  name: string
  kind: 'active' | 'draft'
  description: string
  inputs: { name: string; example: string; description: string }[]
  steps: { server: string; tool: string; arguments: Record<string, unknown> }[]
  replayable: boolean
  replay_error: string | null
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
  | { kind: 'user'; text: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'tool'; toolCallId: string }
  | { kind: 'error'; text: string }

/** Frames the server sends over /ws/chat. */
export type ServerFrame =
  | { type: 'ready'; sessionId: string; model?: string }
  // A resumed session that vibe-acp forked under a new id when continued; the
  // client should track this id for resume, distillation, and reconnect.
  | { type: 'session_migrated'; sessionId: string }
  | { type: 'chunk'; text: string }
  // Replayed user turn from a resumed session (session/load); live prompts are
  // rendered locally on send and are not echoed by the server.
  | { type: 'user'; text: string }
  | {
      type: 'tool_call'
      toolCallId: string
      title: string
      status: string
      kind?: string | null
      rawInput?: unknown
    }
  | { type: 'tool_call_update'; toolCallId: string; status?: string | null; output?: string | null }
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
