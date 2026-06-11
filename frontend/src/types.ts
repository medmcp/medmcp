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
  risks?: RiskTag[]
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

/** Ordered chat transcript entries. Tool calls render as inline cards. */
export type ChatItem =
  | { kind: 'user'; text: string }
  | { kind: 'assistant'; text: string }
  | { kind: 'tool'; toolCallId: string }
  | { kind: 'error'; text: string }

/** Frames the server sends over /ws/chat. */
export type ServerFrame =
  | { type: 'ready'; sessionId: string }
  | { type: 'chunk'; text: string }
  | {
      type: 'tool_call'
      toolCallId: string
      title: string
      status: string
      kind?: string | null
      rawInput?: unknown
    }
  | { type: 'tool_call_update'; toolCallId: string; status?: string | null; output?: string | null }
  | { type: 'usage'; used: number }
  | {
      type: 'permission_request'
      requestId: number
      toolCall: PermissionRequest['toolCall']
      options: PermissionRequest['options']
      explanation?: string | null
      risks?: RiskTag[]
    }
  | { type: 'done' }
  | { type: 'error'; message: string }
