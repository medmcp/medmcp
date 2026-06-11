/** Shared types for the workspace UI and its wire protocols. */

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
  | { type: 'permission_request'; requestId: number; toolCall: PermissionRequest['toolCall']; options: PermissionRequest['options'] }
  | { type: 'done' }
  | { type: 'error'; message: string }
