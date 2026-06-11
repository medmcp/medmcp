import type { SettingsState, TreeNode } from './types'

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

export async function fetchTree(): Promise<TreeNode[]> {
  const res = await check(await fetch('/api/tree'))
  const body = (await res.json()) as { tree: TreeNode[] }
  return body.tree
}

export function rawUrl(path: string): string {
  return `/api/raw/${path.split('/').map(encodeURIComponent).join('/')}`
}

export async function mkdir(path: string): Promise<void> {
  await check(
    await fetch('/api/files/mkdir', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }),
  )
}

export async function renamePath(path: string, newPath: string): Promise<void> {
  await check(
    await fetch('/api/files/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, new_path: newPath }),
    }),
  )
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
  const res = await check(
    await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        explain_tools: state.explain_tools,
        record_provenance: state.record_provenance,
        workflows_enabled: state.workflows_enabled,
        active_stacks: state.stacks.filter((s) => s.active).map((s) => s.name),
        active_workflows: state.workflows.filter((w) => w.active).map((w) => w.name),
      }),
    }),
  )
  const body = (await res.json()) as { restarted: boolean }
  return body.restarted
}
