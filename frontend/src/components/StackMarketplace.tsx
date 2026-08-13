import { useEffect, useMemo, useState } from 'react'
import {
  fetchCatalog,
  fetchInstalledStacks,
  fetchSettings,
  saveSettings,
  uninstallStack,
} from '../api'
import type { CatalogEntry, SettingsState } from '../types'
import { XIcon } from './icons'

interface StackMarketplaceProps {
  open: boolean
  onClose: () => void
}

type Filter = 'all' | 'installed' | 'available' | 'gpu'

/** One row of the merged view: what is installable, what is installed, and its state. */
interface StackItem {
  name: string
  image: string
  description: string
  gpu: boolean
  installed: boolean
  /** Enabled for the agent. Only meaningful once installed. */
  active: boolean
  /** Version string for a uv-tool stack; container stacks have none. */
  version?: string | null
  /** Installed but absent from the catalog — a hand-installed image or a local
   *  uv-tool stack. Shown so the list is the whole truth, not just the catalogue. */
  offCatalog: boolean
}

function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      className={`toggle${checked ? ' on' : ''}`}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span className="toggle-knob" />
    </button>
  )
}

/**
 * Browse, install and enable imaging tool stacks.
 *
 * A window of its own rather than a section of the settings drawer: installing is
 * a browsing task — you scan descriptions, compare, search — and a narrow column
 * of switches is the wrong shape for it. Settings keeps what is genuinely a
 * setting; everything about *which tools exist* lives here.
 */
export function StackMarketplace({ open, onClose }: StackMarketplaceProps) {
  const [state, setState] = useState<SettingsState | null>(null)
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [containerNames, setContainerNames] = useState<Set<string>>(new Set())
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [busy, setBusy] = useState(false)
  const [installingImage, setInstallingImage] = useState<string | null>(null)
  const [progress, setProgress] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [manualImage, setManualImage] = useState('')

  const reload = async () => {
    const [s, inst, cat] = await Promise.all([
      fetchSettings(),
      fetchInstalledStacks(),
      fetchCatalog(),
    ])
    setState(s)
    setContainerNames(new Set(inst.map((i) => i.name)))
    setCatalog(cat)
  }

  useEffect(() => {
    if (!open) return
    let cancelled = false
    Promise.all([fetchSettings(), fetchInstalledStacks(), fetchCatalog()])
      .then(([s, inst, cat]) => {
        if (cancelled) return
        setState(s)
        setContainerNames(new Set(inst.map((i) => i.name)))
        setCatalog(cat)
        setError(null)
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(String(e))
      })
    return () => {
      cancelled = true
    }
  }, [open])

  // Confirmations clear themselves; they report something already done.
  useEffect(() => {
    if (!notice) return
    const t = window.setTimeout(() => setNotice(null), 4000)
    return () => window.clearTimeout(t)
  }, [notice])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  // Catalogue entries and installed stacks are two partial views of the same
  // thing, so merge them by name: a stack can be in the catalogue, installed, or
  // both, and an installed one that no catalogue lists must still be manageable.
  const items = useMemo<StackItem[]>(() => {
    const byName = new Map<string, StackItem>()
    for (const c of catalog) {
      byName.set(c.name, {
        name: c.name,
        image: c.image,
        description: c.description,
        gpu: c.gpu,
        installed: c.installed,
        active: false,
        offCatalog: false,
      })
    }
    for (const s of state?.stacks ?? []) {
      const existing = byName.get(s.name)
      if (existing) {
        existing.installed = true
        existing.active = s.active
        existing.version = s.version
      } else {
        byName.set(s.name, {
          name: s.name,
          image: containerNames.has(s.name) ? 'container image' : '',
          description: '',
          gpu: false,
          installed: true,
          active: s.active,
          version: s.version,
          offCatalog: true,
        })
      }
    }
    return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name))
  }, [catalog, state, containerNames])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return items.filter((i) => {
      if (filter === 'installed' && !i.installed) return false
      if (filter === 'available' && i.installed) return false
      if (filter === 'gpu' && !i.gpu) return false
      if (!q) return true
      return (
        i.name.toLowerCase().includes(q) ||
        i.description.toLowerCase().includes(q) ||
        i.image.toLowerCase().includes(q)
      )
    })
  }, [items, query, filter])

  const install = (image: string, after?: () => void) => {
    if (!image || busy) return
    setBusy(true)
    setInstallingImage(image)
    setError(null)
    setProgress('Starting…')
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/stacks/install`)
    ws.onopen = () => ws.send(JSON.stringify({ image }))
    ws.onmessage = (ev: MessageEvent<string>) => {
      const m = JSON.parse(ev.data) as {
        type: 'progress' | 'done' | 'error'
        line?: string
        name?: string
        message?: string
      }
      if (m.type === 'progress') {
        setProgress(m.line ?? null)
        return
      }
      ws.close()
      setBusy(false)
      setInstallingImage(null)
      setProgress(null)
      if (m.type === 'done') {
        after?.()
        reload()
          .then(() => setNotice(`Installed ${m.name} — the agent restarted.`))
          .catch((e: unknown) => setError(String(e)))
      } else {
        setError(m.message ?? 'install failed')
      }
    }
    ws.onerror = () => {
      setBusy(false)
      setInstallingImage(null)
      setProgress(null)
      setError('install connection failed')
    }
  }

  const remove = (name: string) => {
    if (busy) return
    setBusy(true)
    setError(null)
    uninstallStack(name)
      .then(async () => {
        await reload()
        setNotice(`Uninstalled ${name} — the agent restarted.`)
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setBusy(false))
  }

  const setActive = (name: string, value: boolean) => {
    if (!state) return
    const next = {
      ...state,
      stacks: state.stacks.map((x) => (x.name === name ? { ...x, active: value } : x)),
    }
    setState(next)
    saveSettings(next)
      .then((restarted) => {
        setError(null)
        if (restarted) setNotice('Agent restarted — the chat starts a fresh session.')
      })
      .catch((e: unknown) => setError(String(e)))
  }

  if (!open) return null

  const counts = {
    all: items.length,
    installed: items.filter((i) => i.installed).length,
    available: items.filter((i) => !i.installed).length,
    gpu: items.filter((i) => i.gpu).length,
  }

  return (
    <>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="market" role="dialog" aria-label="Tool stacks">
        <div className="panel-header">
          <span>Tool stacks</span>
          <span className="panel-actions">
            <button className="btn-icon" title="Close" onClick={onClose}>
              <XIcon />
            </button>
          </span>
        </div>

        <div className="market-toolbar">
          <input
            className="market-search"
            placeholder="Search stacks…"
            value={query}
            autoFocus
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="market-filters">
            {(['all', 'installed', 'available', 'gpu'] as Filter[]).map((f) => (
              <button
                key={f}
                className={`market-chip${filter === f ? ' on' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f === 'all' ? 'All' : f === 'gpu' ? 'GPU' : f[0].toUpperCase() + f.slice(1)}
                <span className="market-count">{counts[f]}</span>
              </button>
            ))}
          </div>
        </div>

        {error && <div className="panel-error market-msg">{error}</div>}

        <div className="market-body">
          {!state ? (
            <div className="viewer-message">Loading…</div>
          ) : visible.length === 0 ? (
            <div className="viewer-message">
              {query ? `Nothing matches “${query}”.` : 'No stacks to show.'}
            </div>
          ) : (
            <div className="market-grid">
              {visible.map((i) => (
                <div className={`market-card${i.installed ? ' installed' : ''}`} key={i.name}>
                  <div className="market-card-head">
                    <span className="market-name">{i.name}</span>
                    {i.gpu && <span className="stack-badge">GPU</span>}
                    {i.installed && (
                      <span className={`market-state${i.active ? ' on' : ''}`}>
                        {i.active ? 'enabled' : 'disabled'}
                      </span>
                    )}
                  </div>
                  <div className="market-desc">
                    {i.description ||
                      (i.offCatalog
                        ? 'Installed outside the catalogue.'
                        : 'No description provided.')}
                  </div>
                  <div className="market-image" title={i.image}>
                    {i.version ? `v${i.version}` : i.image}
                  </div>
                  <div className="market-card-foot">
                    {installingImage === i.image ? (
                      <div className="install-inline">
                        <div className="install-bar">
                          <span className="install-bar-fill" />
                        </div>
                        <div className="install-line">{progress ?? 'Starting…'}</div>
                      </div>
                    ) : i.installed ? (
                      <>
                        <Toggle
                          checked={i.active}
                          disabled={busy}
                          onChange={(v) => setActive(i.name, v)}
                        />
                        {containerNames.has(i.name) && (
                          <button
                            className="btn-plain stack-uninstall"
                            disabled={busy}
                            onClick={() => remove(i.name)}
                          >
                            Uninstall
                          </button>
                        )}
                      </>
                    ) : (
                      <button
                        className="btn-primary btn-sm"
                        disabled={busy}
                        onClick={() => install(i.image)}
                      >
                        Install
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="market-foot">
          <input
            className="wf-input"
            placeholder="Install from an image, e.g. ghcr.io/medmcp/neuro:dev"
            value={manualImage}
            disabled={busy}
            onChange={(e) => setManualImage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') install(manualImage.trim(), () => setManualImage(''))
            }}
          />
          <button
            className="btn-primary btn-sm"
            disabled={busy || !manualImage.trim()}
            onClick={() => install(manualImage.trim(), () => setManualImage(''))}
          >
            Install
          </button>
        </div>

        {notice && (
          <div className="drawer-toast" role="status">
            {notice}
          </div>
        )}
      </div>
    </>
  )
}
