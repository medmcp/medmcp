import type {
  ReplayPreviewStep,
  SettingsState,
  TreeNode,
  WorkflowDetail,
  WorkflowListEntry,
} from './types'

/** Thin client for the workspace filesystem API. */

async function check(res: Response): Promise<Response> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res
}

async function sendJson(method: 'POST' | 'PUT', url: string, body: object): Promise<Response> {
  return check(
    await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
}

async function postJson(url: string, body: object): Promise<Response> {
  return sendJson('POST', url, body)
}

export async function fetchTree(): Promise<TreeNode[]> {
  const res = await check(await fetch('/api/tree'))
  const body = (await res.json()) as { tree: TreeNode[] }
  return body.tree
}

export function rawUrl(path: string): string {
  return `/api/raw/${path.split('/').map(encodeURIComponent).join('/')}`
}

export async function mkdir(path: string): Promise<void> {
  await postJson('/api/files/mkdir', { path })
}

export async function renamePath(path: string, newPath: string): Promise<void> {
  await postJson('/api/files/rename', { path, new_path: newPath })
}

export async function deletePath(path: string): Promise<void> {
  await check(await fetch(`/api/files?path=${encodeURIComponent(path)}`, { method: 'DELETE' }))
}

export async function uploadFile(file: File, dir: string): Promise<void> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`/api/files/upload?dir=${encodeURIComponent(dir)}`, {
    method: 'POST',
    body: form,
  })
  await check(res)
}

export async function fetchSettings(): Promise<SettingsState> {
  const res = await check(await fetch('/api/settings'))
  return (await res.json()) as SettingsState
}

/** Persist the full settings state; returns whether the agent was restarted. */
export async function saveSettings(state: SettingsState): Promise<boolean> {
  // Send the full known lists (not just the active names) so the server
  // can preserve entries this drawer never saw (created after it fetched).
  const res = await sendJson('PUT', '/api/settings', {
    explain_tools: state.explain_tools,
    record_provenance: state.record_provenance,
    workflows_enabled: state.workflows_enabled,
    stacks: state.stacks.map((s) => ({ name: s.name, active: s.active })),
    workflows: state.workflows.map((w) => ({ name: w.name, active: w.active })),
  })
  const body = (await res.json()) as { restarted: boolean }
  return body.restarted
}

// ── Workflows ────────────────────────────────────────────────

export async function fetchWorkflows(): Promise<{
  enabled: boolean
  workflows: WorkflowListEntry[]
}> {
  const res = await check(await fetch('/api/workflows'))
  return (await res.json()) as { enabled: boolean; workflows: WorkflowListEntry[] }
}

export async function fetchWorkflowDetail(name: string): Promise<WorkflowDetail> {
  const res = await check(await fetch(`/api/workflows/${encodeURIComponent(name)}`))
  return (await res.json()) as WorkflowDetail
}

/** Distill a chat session into a draft workflow; returns the new draft's detail. */
export async function distillSession(sessionId: string): Promise<WorkflowDetail> {
  const res = await postJson('/api/workflows/distill', { session_id: sessionId })
  return (await res.json()) as WorkflowDetail
}

export async function promoteWorkflow(name: string): Promise<void> {
  await postJson(`/api/workflows/${encodeURIComponent(name)}/promote`, {})
}

export async function unpromoteWorkflow(name: string): Promise<void> {
  await postJson(`/api/workflows/${encodeURIComponent(name)}/unpromote`, {})
}

/** Rename a draft; returns the new (slugified) name. */
export async function renameWorkflow(name: string, newName: string): Promise<string> {
  const res = await postJson(`/api/workflows/${encodeURIComponent(name)}/rename`, {
    new_name: newName,
  })
  const body = (await res.json()) as { name: string }
  return body.name
}

export async function refineWorkflow(name: string, instruction: string): Promise<void> {
  await postJson(`/api/workflows/${encodeURIComponent(name)}/refine`, { instruction })
}

export async function deleteWorkflow(name: string): Promise<void> {
  await check(await fetch(`/api/workflows/${encodeURIComponent(name)}`, { method: 'DELETE' }))
}

export async function replayPreview(
  name: string,
  inputs: Record<string, string>,
): Promise<{ ok: boolean; error: string | null; steps: ReplayPreviewStep[] }> {
  const res = await postJson(`/api/workflows/${encodeURIComponent(name)}/replay-preview`, {
    inputs,
  })
  return (await res.json()) as { ok: boolean; error: string | null; steps: ReplayPreviewStep[] }
}
