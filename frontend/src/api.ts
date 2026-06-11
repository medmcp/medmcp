import type { TreeNode } from './types'

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
