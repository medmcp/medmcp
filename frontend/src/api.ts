import type {
  BatchFromPlanResult,
  CatalogEntry,
  ExternalMcpState,
  GpuInfo,
  InstalledStack,
  ReplayPreviewStep,
  RewindResult,
  SessionInfo,
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

async function sendJson(
  method: 'POST' | 'PUT' | 'PATCH',
  url: string,
  body: object,
): Promise<Response> {
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
  // Send the full known list (not just the active names) so the server can
  // preserve stacks this drawer never saw (installed after it fetched).
  const res = await sendJson('PUT', '/api/settings', {
    explain_tools: state.explain_tools,
    record_provenance: state.record_provenance,
    gpu: state.gpu,
    stacks: state.stacks.map((s) => ({ name: s.name, active: s.active })),
  })
  const body = (await res.json()) as { restarted: boolean }
  return body.restarted
}

/** Best-effort list of GPUs for the settings picker (may be empty). */
export async function fetchGpus(): Promise<GpuInfo[]> {
  const res = await check(await fetch('/api/gpus'))
  const body = (await res.json()) as { gpus: GpuInfo[] }
  return body.gpus
}

// ── Container stacks (install / uninstall) ───────────────────

/** List container-installed stacks (those with a stacks.d manifest). */
export async function fetchInstalledStacks(): Promise<InstalledStack[]> {
  const res = await check(await fetch('/api/stacks'))
  const body = (await res.json()) as { stacks: InstalledStack[] }
  return body.stacks
}

/** Fetch the curated install catalog (each entry flagged whether it's installed). */
export async function fetchCatalog(): Promise<CatalogEntry[]> {
  const res = await check(await fetch('/api/catalog'))
  const body = (await res.json()) as { catalog: CatalogEntry[] }
  return body.catalog
}

/** Install a stack from a container image; returns its name. Restarts the agent. */
export async function installStack(image: string): Promise<string> {
  const res = await postJson('/api/stacks/install', { image })
  const body = (await res.json()) as { name: string }
  return body.name
}

/** Uninstall a container stack by name. Restarts the agent. */
export async function uninstallStack(name: string): Promise<void> {
  await postJson('/api/stacks/uninstall', { name })
}

// ── Sessions ─────────────────────────────────────────────────

export async function fetchSessions(): Promise<SessionInfo[]> {
  const res = await check(await fetch('/api/sessions'))
  const body = (await res.json()) as { sessions: SessionInfo[] }
  return body.sessions
}

/** Set a session's display title (empty string clears the override). */
export async function renameSession(id: string, title: string): Promise<void> {
  await postJson(`/api/sessions/${encodeURIComponent(id)}/rename`, { title })
}

export async function archiveSession(id: string, archived: boolean): Promise<void> {
  await postJson(`/api/sessions/${encodeURIComponent(id)}/archive`, { archived })
}

/** Branch the live chat into a new session; returns the fork's id. */
export async function forkSession(id: string): Promise<string> {
  const res = await postJson(`/api/sessions/${encodeURIComponent(id)}/fork`, {})
  return ((await res.json()) as { id: string }).id
}

/** Preview a rewind: which workspace files would be restored. */
export async function previewRewind(id: string, messageId: string): Promise<RewindResult> {
  const res = await postJson(`/api/sessions/${encodeURIComponent(id)}/rewind`, {
    messageId,
    preview: true,
  })
  return (await res.json()) as RewindResult
}

/** Rewind the live chat to before a message (truncates + restores files). */
export async function rewindTo(id: string, messageId: string): Promise<RewindResult> {
  const res = await postJson(`/api/sessions/${encodeURIComponent(id)}/rewind`, { messageId })
  return (await res.json()) as RewindResult
}

/** Delete a session for good (transcript + provenance + UI metadata). */
export async function deleteSession(id: string): Promise<void> {
  await check(await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }))
}

// ── Workflows ────────────────────────────────────────────────

export async function fetchWorkflows(): Promise<WorkflowListEntry[]> {
  const res = await check(await fetch('/api/workflows'))
  return ((await res.json()) as { workflows: WorkflowListEntry[] }).workflows
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

/** Download a workflow as a shareable <name>.workflow.yaml via the browser. */
export async function exportWorkflow(name: string): Promise<void> {
  const res = await check(await fetch(`/api/workflows/${encodeURIComponent(name)}/export`))
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${name}.workflow.yaml`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Import a shared workflow file (its YAML text); returns the new draft's detail. */
export async function importWorkflow(content: string): Promise<WorkflowDetail> {
  const res = await postJson('/api/workflows/import', { content })
  return (await res.json()) as WorkflowDetail
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

/** Turn a plan_batch manifest CSV into per-subject batch rows for this workflow. */
export async function batchFromPlan(name: string, planCsv: string): Promise<BatchFromPlanResult> {
  const res = await postJson(`/api/workflows/${encodeURIComponent(name)}/batch-from-plan`, {
    plan_csv: planCsv,
  })
  return (await res.json()) as BatchFromPlanResult
}

// ── External MCP (advanced) ──────────────────────────────────
// Every mutation restarts the agent server-side, so callers should refetch
// rather than assume their optimistic view survived.

export async function fetchExternalMcp(): Promise<ExternalMcpState> {
  const res = await check(await fetch('/api/external-mcp'))
  return (await res.json()) as ExternalMcpState
}

/** Record that the operator accepted the risks. Required before enabling. */
export async function acknowledgeExternalMcp(): Promise<void> {
  await postJson('/api/external-mcp/acknowledge', {})
}

export async function setExternalMcpEnabled(enabled: boolean): Promise<void> {
  await sendJson('PUT', '/api/external-mcp', { enabled })
}

export async function addExternalServer(server: {
  name: string
  transport: string
  url: string
  /** The token itself — stored server-side, never returned by any endpoint. */
  token: string
  api_key_env: string
  api_key_header: string
  api_key_format: string
}): Promise<void> {
  await postJson('/api/external-mcp/servers', server)
}

/** Replace one server's stored token. The value is write-only across this API. */
export async function replaceExternalToken(name: string, token: string): Promise<void> {
  await sendJson('PATCH', `/api/external-mcp/servers/${encodeURIComponent(name)}`, { token })
}

export async function setExternalServerActive(name: string, active: boolean): Promise<void> {
  await sendJson('PATCH', `/api/external-mcp/servers/${encodeURIComponent(name)}`, { active })
}

export async function removeExternalServer(name: string): Promise<void> {
  await check(
    await fetch(`/api/external-mcp/servers/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  )
}
